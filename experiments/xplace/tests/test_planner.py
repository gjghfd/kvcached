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
                        "max_tokens": request["tokens"] - request.get("prompt", 85), "blocks": block_ids,
                        "movable": True})
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


CODE = {"model": 0, "tokens": 101626 + 64, "prompt_tokens": 101626, "output_tokens": 64}


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
            request["max_tokens"] = 12000
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
