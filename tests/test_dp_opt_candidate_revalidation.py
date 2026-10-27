import asyncio
import copy
import importlib
import time
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    controller = importlib.import_module("live_controller")
    monkeypatch.setattr(controller, "performance_tables", lambda snapshot, *args: [
        [abs(2 * count - (sum(values) + int(model == 0))) * .01
         for count in range(sum(values) + int(model == 0) + 1)] for model, values in enumerate(snapshot["counts"])])
    return controller, importlib.import_module("ilp_reference"), importlib.import_module("verify_candidate_revalidation")


def observations():
    empty = {"requests": [], "pages": [], "mapped_bytes": 0, "block_size": 16,
             "block_bytes": 1024**3 // 64, "page_bytes": 1024**3, "uniform_page_geometry": True,
             "mapped_free_block_slots": 0, "page_block_capacity": 64, "free_blocks": 10000}
    result = [[copy.deepcopy(empty) for rank in (0, 1)] for tenant in (0, 1)]
    for rank in (0, 1):
        requests = [{"id": f"conv-{rank}-{index}", "blocks": [index * 64 + 1], "movable": True,
                     "max_tokens": 1024, "output_tokens": 64, "estimated_remaining_decode_s": 100}
                    for index in range(3)]
        result[1][rank].update({"requests": requests, "mapped_bytes": 3 * 1024**3,
            "pages": [{"id": index, "owners": [request["id"]], "pinned": False, "memory_gib": 1}
                      for index, request in enumerate(requests)]})
    return result


def prepared(controller, observed):
    snapshot, horizon, population = controller.admission_planning_snapshot(
        observed, 0, 2048, 64, [4, 4], {}, True, "ilp", 8, 0, 1 / .07)
    candidates = [{"status": "optimal", "target": rank, "moves": [f"conv-{rank}-0"], "objective_s": 1}
                  for rank in (0, 1)]
    return {"settings": [1984, 64, [4, 4], "ilp", 0, 8, 8, 0, 1 / .07, 1],
            "observations": copy.deepcopy(observed), "snapshot": snapshot,
            "plan": {"status": "optimal", "candidates": candidates, "winner": candidates[0]},
            "admission_generation": 0, "ownership": controller.stable_ownership(observed),
            "elapsed_s": .01, "horizon_steps": horizon, "active_plus_incoming": population, "scope": "test"}


def remove_request(observed, rank, index):
    observed[1][rank]["requests"] = [request for request in observed[1][rank]["requests"] if request["id"] != f"conv-{rank}-{index}"]
    observed[1][rank]["pages"] = [page for page in observed[1][rank]["pages"] if page["id"] != index]
    observed[1][rank]["mapped_bytes"] -= 1024**3


def test_unrelated_population_changes_revalidate_current_counts_and_memory(modules):
    controller, reference, verifier = modules
    observed = observations()
    original = prepared(controller, observed)
    remove_request(observed, 0, 2)
    snapshot, horizon, population, plan = controller.revalidate_prepared_plan(original, observed, {}, True)
    assert plan["winner"] is not None and plan["proven_infeasible"] is False
    assert snapshot["counts"] == [[0, 0], [2, 3]] and population == 6
    assert any(not candidate.get("moves") and "objective_s" in candidate for candidate in plan["candidates"])
    for candidate in plan["candidates"][:2]:
        verifier.verify_projection(snapshot, original["plan"]["candidates"][candidate["target"]], candidate)


@pytest.mark.parametrize("change", ["completed", "blocks", "rank", "prefill", "pinned"])
def test_selected_changes_or_pinned_release_cannot_reuse_old_plan(modules, change):
    controller, reference, verifier = modules
    observed = observations()
    original = prepared(controller, observed)
    if change == "completed":
        remove_request(observed, 0, 0)
    elif change == "blocks":
        observed[1][0]["requests"][0]["blocks"] = [2]
    elif change == "rank":
        removed = observed[1][0]["requests"][0]
        remove_request(observed, 0, 0)
        removed["blocks"] = [193]
        observed[1][1]["requests"].append(removed)
        observed[1][1]["pages"].append({"id": 3, "owners": [removed["id"]], "pinned": False, "memory_gib": 1})
        observed[1][1]["mapped_bytes"] += 1024**3
    elif change == "prefill":
        observed[1][0]["requests"][0]["movable"] = False
    else:
        observed[1][0]["pages"][0]["pinned"] = True
    snapshot, horizon, population, plan = controller.revalidate_prepared_plan(original, observed, {}, True)
    assert "objective_s" not in plan["candidates"][0]
    assert plan["candidates"][0]["status"].startswith("revalidation_")


@pytest.mark.parametrize("change", ["unknown_owner", "missing_page", "geometry", "duplicate_page", "page_bytes"])
def test_corrupt_raw_inventory_is_fatal_not_silently_ignored(modules, change):
    controller, reference, verifier = modules
    observed = observations()
    original = prepared(controller, observed)
    if change == "unknown_owner":
        observed[1][0]["pages"][0]["owners"].append("ghost")
    elif change == "missing_page":
        observed[1][0]["requests"][0]["blocks"] = [257]
    elif change == "geometry":
        observed[0][0]["block_size"] = 32
    elif change == "duplicate_page":
        observed[1][0]["pages"].append(copy.deepcopy(observed[1][0]["pages"][0]))
        observed[1][0]["mapped_bytes"] += 1024**3
    else:
        observed[1][0]["pages"][0]["memory_gib"] = .5
    with pytest.raises(ValueError):
        controller.revalidate_prepared_plan(original, observed, {}, True)


def test_unrelated_new_request_revalidates_copy_cost_at_current_population(modules):
    controller, reference, verifier = modules
    observed = observations()
    original = prepared(controller, observed)
    observed[1][0]["requests"].append({"id": "new-conv", "blocks": [2], "movable": True,
                                       "max_tokens": 1024, "output_tokens": 64, "estimated_remaining_decode_s": 100})
    observed[1][0]["pages"][0]["owners"] = ["conv-0-0", "new-conv"]
    snapshot, horizon, population, plan = controller.revalidate_prepared_plan(original, observed, {}, True)
    assert population == 8 and "objective_s" not in plan["candidates"][0]
    assert plan["candidates"][1]["migration_cost_s"] == pytest.approx(8 * .07 / 64)


def test_current_direct_fit_is_independently_verified(modules):
    controller, reference, verifier = modules
    observed = observations()
    original = prepared(controller, observed)
    remove_request(observed, 0, 2)
    snapshot, horizon, population, plan = controller.revalidate_prepared_plan(original, observed, {}, True)
    identity = controller.prepared_candidate_identity(original)
    decision = {"original_candidate": identity, "winner": None,
                "candidate_revalidations": [{"stage": "before_pause", "elapsed_s": .001,
                    "observations": observed, "snapshot": snapshot, "horizon_steps": horizon,
                    "active_plus_incoming": population, "plan": plan, "wait_gate": None}]}
    verifier.verify_revalidation(decision, {}, None)
    decision["candidate_revalidations"][0]["plan"]["candidates"][-1]["objective_s"] += .01
    with pytest.raises(ValueError, match="cost mismatch"):
        verifier.verify_revalidation(decision, {}, None)


def test_copy_peak_and_reused_source_pool_reject_before_freeing(modules):
    controller, reference, verifier = modules
    snapshot = {"counts": [[1, 1], [0, 0]], "initial_gib": [3, 3], "capacity_gib": [4, 4], "new_gib": 2,
                "performance_cost_tables_s": [[0] * 4, [0]],
                "movable": [{"id": "a", "model": 0, "source": 0, "memory_gib": 2,
                             "move_cost_s": 1, "destination_pool": "0:1"}],
                "source_pages": [{"id": "p", "rank": 0, "owners": ["a"], "pinned": False,
                                  "memory_gib": 2, "pool": "0:0"}]}
    original = {"status": "optimal", "target": 0, "moves": ["a"]}
    projected = reference.replay_candidate(snapshot, original)
    assert projected["status"] == "revalidation_copy_peak_rejected"
    verifier.verify_projection(snapshot, original, projected)
    snapshot["movable"][0]["memory_gib"] = 1
    snapshot["source_pages"][0]["pool"] = "0:1"
    projected = reference.replay_candidate(snapshot, original)
    assert projected["status"] == "revalidation_admission_rejected"
    verifier.verify_projection(snapshot, original, projected)


@pytest.mark.parametrize("finish_at_boundary", [False, True])
def test_dispatcher_current_wait_rechecked_before_and_after_boundary(modules, tmp_path, finish_at_boundary):
    controller_module, reference, verifier = modules
    observed = observations()
    original = prepared(controller_module, observed)
    options = {"copy_seconds_per_gib": .07, "safety_fraction": .8,
               "boundary_cost": {"incoming_handshake_s": 3.06, "old_transaction_fixed_s": .09, "preplan_s": .59}}
    calls = []

    async def run():
        controller = controller_module.PlacementController([], {}, tmp_path, True)
        controller.revalidate_prepared = True
        controller.admission_generation = 10

        async def snapshots(*args, **kwargs):
            return copy.deepcopy(observed)

        async def pause(current):
            calls.append("pause")
            return [[{"reached_at": time.monotonic()}] * 2] * 2

        async def begin(transaction):
            if finish_at_boundary:
                for ranks in observed:
                    for rank in ranks:
                        for request in rank["requests"]:
                            request["estimated_remaining_decode_s"] = .001
            return copy.deepcopy(observed)

        async def end(transaction):
            calls.append("end")

        async def resume():
            calls.append("resume")

        async def migrate(transaction, request, budgets):
            calls.append("migrate")
            source, destination = request["source"], 1 - request["source"]
            index = int(request["id"].rsplit("-", 1)[1])
            moved = copy.deepcopy(next(item for item in observed[1][source]["requests"] if item["id"] == request["id"]))
            remove_request(observed, source, index)
            moved["blocks"] = [193]
            observed[1][destination]["requests"].append(moved)
            observed[1][destination]["pages"].append({"id": 3, "owners": [moved["id"]], "pinned": False, "memory_gib": 1})
            observed[1][destination]["mapped_bytes"] += 1024**3
            return {"request_id": request["id"], "model": 1, "source": source, "destination": destination,
                    "kv_bytes": observed[1][destination]["block_bytes"], "integrity": "device_echo_full_bytewise",
                    "bitwise_verified": True}

        async def cache():
            return []

        async def capacity(*args):
            return [2, 4]

        async def admit(target):
            calls.append("admit")

        controller.snapshots, controller.pause_boundary, controller.begin = snapshots, pause, begin
        controller.end, controller.resume, controller.migrate = end, resume, migrate
        controller.release_transfer_cache, controller.check_capacity = cache, capacity
        return await controller.schedule_admission(1984, 64, [4, 4], admit, max_moves=8, candidate_limit=8,
            move_fixed_s=0, move_bandwidth_gib_s=1 / .07, prepared=original, wait_cost_s=80,
            future_transaction_fixed_s=.09, wait_recheck_options=options)

    result = asyncio.run(run())
    assert result["preplan_candidate_revalidated"] and len(result["candidate_revalidations"]) == 2
    if finish_at_boundary:
        assert result["wait_cost_s"] == pytest.approx(.0008)
        assert calls == ["pause", "end", "resume"] and not result["admitted"]
    else:
        assert calls == ["pause", "migrate", "admit", "end", "resume"] and result["admitted"]
    verifier.verify_revalidation(result, {}, None)
    if result["moves"]:
        result["moves"][0]["kv_bytes"] += 1
        with pytest.raises(ValueError, match="frozen selection"):
            verifier.verify_revalidation(result, {}, None)
        result["moves"][0]["kv_bytes"] -= 1
    result["candidate_revalidations"][-1]["snapshot"]["initial_gib"][0] += .01
    with pytest.raises(ValueError):
        verifier.verify_revalidation(result, {}, None)
