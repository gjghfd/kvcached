"""Local tests of the simulator's page model (not shipped to the cluster: the job bundle carries runtime/ and tests/)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))

import simulator  # noqa: E402
from simulator import PAGE_BLOCKS, PagePool, Request, Simulator  # noqa: E402

INPUTS = HERE.parents[3] / "evidence" / "xplace-inputs"


def make_request(name):
    return Request({"request_id": name, "model": 1, "prompt_tokens": 100, "output_tokens": 100, "arrival_s": 0.0})


def allocated(pool):
    return pool.pages() * PAGE_BLOCKS - pool.slots


def test_growth_fills_fullest_page_and_runs_take_smallest_page_that_fits():
    pool = PagePool()
    x, y, z, w = (make_request(name) for name in "xyzw")
    pool.alloc(x, 32)
    pool.alloc(y, 32)  # fits the rest of page 0
    pool.alloc(z, 50)  # new page 1, 14 free
    pool.alloc(w, 64)  # no page holds 64: takes the emptiest (page 1), then a new page 2 (14 free)
    assert pool.pages() == 3 and dict(pool.free) == {0: 0, 1: 0, 2: 14}
    pool.release(x)  # page 0 keeps y, so it stays mapped with 32 holes
    assert pool.pages() == 3 and pool.free[0] == 32
    growth = make_request("g")
    pool.alloc(growth, 1, packed=True)
    assert growth.holdings == {2: 1}  # fullest page with room, not the page with the most holes
    run = make_request("r")
    pool.alloc(run, 20)
    assert run.holdings == {0: 20}  # smallest page that holds the whole run
    assert allocated(pool) == 32 + 50 + 64 + 1 + 20


def test_page_unmaps_only_when_all_owners_leave():
    pool = PagePool()
    a, b = make_request("a"), make_request("b")
    pool.alloc(a, 40)
    pool.alloc(b, 24)
    assert pool.pages() == 1 and pool.owners[0] == {"a": 40, "b": 24}
    pool.release(a)
    assert pool.pages() == 1 and pool.slots == 40
    pool.release(b)
    assert pool.pages() == 0 and pool.slots == 0 and a.holdings == {} and b.holdings == {}


def test_blocks_model_run_keeps_pools_consistent_and_empties_them():
    truth = json.loads((HERE / "truth-pilot-nm-v2.json").read_text())
    belief = json.loads((INPUTS / "step-params-dp4-v1.json").read_text())
    planner = json.loads((INPUTS / "planner-params-v3.json").read_text())
    trace = json.loads((INPUTS / "cur-b16-c012-p120-s911-d900.json").read_text())
    checks = []
    original = Simulator.on_population

    def checked(self, payload):
        for tenant in self.tenants:
            for rank, pool in enumerate(tenant.pools):
                held = sum(request.held_blocks for request in list(tenant.active.values())
                           + list(tenant.joining.values()) if request.rank == rank)
                assert allocated(pool) == tenant.blocks[rank] == held
        checks.append(self.now)
        return original(self, payload)

    simulator.Simulator.on_population = checked
    try:
        for policy in ("mig_random_v2", "ca_full_tm_slo"):
            params = {"num_ranks": 4, "budget_gib": 16.0, **truth.get("sim", {}), "kv_mode": "prompt",
                      "reserve_extra_tokens": 256, "page_model": "blocks"}
            sim = Simulator(trace, policy, sim_params=params, step_params=truth, planner_step_params=belief,
                            planner_params=planner)
            out = sim.run()
            assert len(out["requests"]) == len(trace["requests"])
            assert out["migrations"]  # page ownership lets a partial migration free pages
            assert all(pool.pages() == 0 and pool.slots == 0 for tenant in sim.tenants for pool in tenant.pools)
    finally:
        simulator.Simulator.on_population = original
    assert len(checks) > 100
