import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from cost_model import StepModel, capture_bucket, default_dp2_params, integrated_cost  # noqa: E402
from planner import (DEFAULT_PARAMS, build_state, plan_admission, plan_return, plan_usage,  # noqa: E402
                     released_gib)

MIB = 1024**2
BLOCK_BYTES = int(1.5 * MIB)
PAGE_BLOCKS = 64


def rank_observation(rank, requests, start_page=0):
    """Fake full snapshot: each request owns whole pages (blocks rounded up to pages)."""
    pages, records, page_id = [], [], start_page
    for request in requests:
        blocks = math.ceil(request["tokens"] / 16)
        count = math.ceil(blocks / PAGE_BLOCKS)
        block_ids = list(range(1 + page_id * PAGE_BLOCKS, 1 + page_id * PAGE_BLOCKS + blocks))
        for _ in range(count):
            pages.append({"id": page_id, "owners": [request["id"]], "pinned": False,
                          "memory_gib": PAGE_BLOCKS * BLOCK_BYTES / 1024**3})
            page_id += 1
        records.append({"id": request["id"], "computed_tokens": request["context"],
                        "prompt_tokens": request.get("prompt", 85), "output_tokens": request["context"] - 85,
                        "blocks": block_ids, "movable": True})
    pages.append({"id": page_id, "owners": [], "pinned": True, "memory_gib": PAGE_BLOCKS * BLOCK_BYTES / 1024**3})
    return {"rank": rank, "requests": records, "pages": pages,
            "mapped_bytes": len(pages) * PAGE_BLOCKS * BLOCK_BYTES, "page_bytes": PAGE_BLOCKS * BLOCK_BYTES,
            "block_bytes": BLOCK_BYTES, "page_block_capacity": PAGE_BLOCKS, "uniform_page_geometry": True,
            "block_size": 16, "mapped_free_block_slots": 0, "free_blocks": 30000}


def make_observed(conv_per_rank, contexts=None, num_ranks=4):
    observed = [[rank_observation(rank, []) for rank in range(num_ranks)], []]
    for rank, count in enumerate(conv_per_rank):
        requests = [{"id": f"conv-{rank}-{index}", "tokens": 13300,
                     "context": (contexts or {}).get((rank, index), 2000 + 1000 * index)}
                    for index in range(count)]
        observed[1].append(rank_observation(rank, requests, start_page=0))
    return observed


CODE = {"model": 0, "tokens": 101626 + 64, "prompt_tokens": 101626}


def test_capture_bucket():
    assert [capture_bucket(value) for value in (0, 1, 2, 8, 9, 16, 17, 33)] == [0, 1, 8, 8, 16, 16, 32, 64]


def test_integrated_cost_monotone_in_imbalance():
    model = StepModel(default_dp2_params())
    balanced = [(rank, 4000, 1000) for rank in range(4) for _ in range(4)]
    skewed = [(0, 4000, 1000)] * 10 + [(1, 4000, 1000)] * 2 + [(2, 4000, 1000)] * 2 + [(3, 4000, 1000)] * 2
    assert integrated_cost(model, 1, skewed, 4) > integrated_cost(model, 1, balanced, 4)


def test_release_counts_only_exclusively_owned_pages():
    observed = make_observed([2, 0, 0, 0])
    state = build_state(observed, [10.0] * 4)
    one = released_gib(state, 0, frozenset({"conv-0-0"}))
    assert math.isclose(one, 13 * 64 * 1.5 / 1024)
    assert math.isclose(released_gib(state, 0, frozenset({"conv-0-0", "conv-0-1"})), 2 * one)


def test_code_blocked_without_moves_and_admitted_by_both_policies():
    observed = make_observed([6, 6, 6, 6])
    budgets = [10.0] * 4
    state = build_state(observed, budgets)
    assert all(plan_usage(state, [], CODE, target) is None for target in range(4))
    model = StepModel(default_dp2_params())
    for policy in ("mig_random", "ca"):
        plan = plan_admission(state, CODE, policy, model, DEFAULT_PARAMS, seed=3)
        assert plan["target"] is not None, plan
        assert all(move["source"] == plan["target"] != move["destination"] for move in plan["moves"])
        assert plan_usage(state, plan["moves"], CODE, plan["target"]) is not None


def test_cost_aware_avoids_bucket_crossing_when_random_does_not():
    observed = make_observed([16, 14, 14, 14])
    budgets = [26.0] * 4
    state = build_state(observed, budgets)
    model = StepModel(default_dp2_params())
    plan = plan_admission(state, CODE, "ca", model, DEFAULT_PARAMS)
    destinations = [move["destination"] for move in plan["moves"]]
    counts = [14 + destinations.count(rank) for rank in range(4)]
    counts[plan["target"]] = 16 - len(plan["moves"]) if plan["target"] == 0 else counts[plan["target"]] - len(plan["moves"])
    assert max(counts) <= 16, (plan["target"], destinations)
    crossings = 0
    for seed in range(20):
        random_plan = plan_admission(state, CODE, "mig_random", model, DEFAULT_PARAMS, seed=seed)
        destinations = [move["destination"] for move in random_plan["moves"]]
        base = [16, 14, 14, 14]
        counts = [base[rank] + destinations.count(rank) for rank in range(4)]
        counts[random_plan["target"]] -= len(random_plan["moves"])
        crossings += max(counts) > 16
    assert crossings > 0


def test_return_moves_back_to_vacated_rank_when_worth_it():
    observed = make_observed([2, 18, 18, 18], contexts={(rank, index): 6000 for rank in range(4) for index in range(20)})
    for rank in range(1, 4):
        for request in observed[1][rank]["requests"]:
            request["output_tokens"] = 4000
    state = build_state(observed, [40.0] * 4)
    model = StepModel(default_dp2_params())
    plan = plan_return(state, model, DEFAULT_PARAMS)
    assert plan["accepted"], plan["decisions"]
    assert all(move["destination"] == 0 for move in plan["moves"])


def test_return_rejects_when_balanced():
    observed = make_observed([8, 8, 8, 8])
    state = build_state(observed, [40.0] * 4)
    plan = plan_return(state, StepModel(default_dp2_params()), DEFAULT_PARAMS)
    assert not plan["accepted"] and not plan["moves"]


def test_planner_never_reads_output_length():
    """Online the final output length is unknown: plans must not change when a (replay-only) length is attached."""
    observed = make_observed([6, 6, 6, 6])
    with_length = make_observed([6, 6, 6, 6])
    for ranks in with_length:
        for rank in ranks:
            for request in rank["requests"]:
                request["max_tokens"] = 50
    model = StepModel(default_dp2_params())
    for policy in ("mig_random", "ca"):
        plain = plan_admission(build_state(observed, [10.0] * 4), CODE, policy, model, DEFAULT_PARAMS, seed=3)
        leaked = plan_admission(build_state(with_length, [10.0] * 4), CODE, policy, model, DEFAULT_PARAMS, seed=3)
        assert (plain["target"], plain["moves"]) == (leaked["target"], leaked["moves"])


def test_growth_headroom_keeps_room_on_target_and_destinations():
    observed = make_observed([6, 6, 6, 6])
    budgets = [12.0] * 4
    model = StepModel(default_dp2_params())
    loose = build_state(observed, budgets)
    plan = plan_admission(loose, CODE, "ca", model, DEFAULT_PARAMS)
    tight = build_state(observed, budgets, growth_gib=(0.0, 0.2), slack_gib=0.1)
    guarded = plan_admission(tight, CODE, "ca", model, DEFAULT_PARAMS)
    assert plan["target"] is not None and guarded["target"] is not None
    assert len(guarded["moves"]) > len(plan["moves"])
    used = plan_usage(tight, guarded["moves"], CODE, guarded["target"])
    destination = {move["id"]: move["destination"] for move in guarded["moves"]}
    counts = [0] * 4
    for request in tight["requests"]:
        if request["model"] == 1:
            counts[destination.get(request["id"], request["rank"])] += 1
    for rank in {guarded["target"], *destination.values()}:
        assert used[rank] + counts[rank] * 0.2 + 0.1 <= budgets[rank] + 1e-6


def test_headroom_rejects_a_rank_that_only_fits_without_growth():
    observed = make_observed([4, 9, 9, 9])
    state = build_state(observed, [14.5] * 4)
    assert plan_usage(state, [], CODE, 0) is not None
    grown = build_state(observed, [14.5] * 4, growth_gib=(0.0, 0.3), slack_gib=0.0)
    assert plan_usage(grown, [], CODE, 0) is None


def test_conv_growth_tokens_uses_rates_not_lengths():
    from planner import conv_growth_tokens

    params = dict(DEFAULT_PARAMS)
    model = StepModel(default_dp2_params())
    counts, contexts = [8, 8, 8, 8], [8 * 6000.0] * 4
    relief = conv_growth_tokens(model, params, counts, contexts)
    assert math.isclose(relief, params["relief_horizon_s"] / model.decode_step(1, counts, contexts))
    held = conv_growth_tokens(model, params, counts, contexts, arrival_prompt=100000, isolate=True)
    sliced = conv_growth_tokens(model, params, counts, contexts, 100000, True, (2.0, 1.0, 15.0))
    corun = conv_growth_tokens(model, params, counts, contexts, 100000, False)
    assert held == relief and sliced >= relief and corun >= relief


def test_arrival_counts_pages_released_by_its_moves_toward_free_blocks():
    """kvcached free_blocks is physical (holes + free pages): when the target is full, the arrival only fits once the
    moves release pages, so the free-block check must credit them (V8/V9c live: never-feasible plans)."""
    observed = make_observed([6, 6, 6, 6])
    for index in range(4):
        used = sum(page["memory_gib"] for model in (0, 1) for page in observed[model][index]["pages"])
        for model in (0, 1):  # both tenants map the same physical memory: 12 GiB per GPU
            observed[model][index]["free_blocks"] = int(max(0.0, 12.0 - used) * 1024**3 // BLOCK_BYTES)
    model = StepModel(default_dp2_params())
    state = build_state(observed, [12.0] * 4)
    blocks = math.ceil(CODE["tokens"] / 16)
    assert all(blocks > state["pools"][(0, rank)]["free_blocks"] for rank in range(4))
    assert all(plan_usage(state, [], CODE, rank) is None for rank in range(4))
    plan = plan_admission(state, CODE, "ca", model, DEFAULT_PARAMS)
    assert plan["target"] is not None and plan["moves"], plan
    assert plan_usage(state, plan["moves"], CODE, plan["target"]) is not None


def test_infeasible_plan_reports_per_target_diagnostics():
    observed = make_observed([6, 6, 6, 6])
    state = build_state(observed, [9.0] * 4)  # the Code (9.3 GiB) can never fit
    plan = plan_admission(state, CODE, "ca", StepModel(default_dp2_params()), DEFAULT_PARAMS)
    assert plan["target"] is None and plan["reason"] == "no_feasible_plan"
    assert [item["target"] for item in plan["diagnostics"]] == [0, 1, 2, 3]
    for item in plan["diagnostics"]:
        assert item["movable"] == 6 and item["need_blocks"] == math.ceil(CODE["tokens"] / 16)
        assert set(item["stages"]) <= {"select", "assign", "usage"} and item["released_if_all_moved_gib"] > 0
