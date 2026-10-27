import asyncio
import copy
import importlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return SimpleNamespace(pilot=importlib.import_module("admission_cost_pilot"),
                           verifier=importlib.import_module("verify_admission_cost"))


def marks():
    return [{"prompt_tokens": 3584, "output_tokens": 1024, "source_internal_row": 1234}]


def test_cases_preserve_azure_pair_population_and_alternating_order(modules):
    cases = modules.pilot.pilot_cases(marks())
    assert len(cases) == 6
    assert [case["population"] for case in cases] == [8, 8, 12, 12, 16, 16]
    assert [case["mode"] for case in cases] == ["native_inside", "preplan_boundary",
        "preplan_boundary", "native_inside", "native_inside", "preplan_boundary"]
    assert sum(case["population"] + 1 for case in cases) == 78
    for values in ([], [{**marks()[0], "output_tokens": 4096}], [{"prompt_tokens": 3584, "output_tokens": 1024}]):
        with pytest.raises(ValueError):
            modules.pilot.pilot_cases(values)


def fixture(root, modules):
    def save(name, value):
        (root / name).write_text(json.dumps(value))

    inputs = root / "inputs"
    inputs.mkdir()
    (inputs / "conv-marks.json").write_text(json.dumps(marks()))
    code = {"request_id": "code-mark", "model": 0, "prompt_tokens": 100000, "output_tokens": 64}
    save("trace.json", {"requests": [code]})
    save("config.json", {"admission_cost_pilot": True, "transfer_pilot": False, "max_model_len": 131072,
                         "cache_bytes": 24 * 1024**3, "kv_budget_mib": 11328, "allocator_gpu_utilization": .98})
    save("runtime-control.json", {"dp_pause_sync_interval": 32, "same_interval_for_all_policies": True,
        "idle_rank_expert_execution_not_disabled": True, "kv_transport": "dp_echo"})
    for model in (0, 1):
        for rank in (0, 1):
            save(f"worker-instance-{model + 1}-dp{rank}.json", {"ipc_name": f"pool{model}-{rank}",
                 "dp_pause_sync_interval": 32, "gpu_name": "A800", "gpu_uuid": f"GPU{rank}",
                 "checkpoint": "qwen", "max_model_len": 131072, "kv_cache_memory_bytes": 24 * 1024**3,
                 "groups": {name: {"world_size": size} for name, size in (("dp", 2), ("ep", 2), ("tp", 1))}})
    cases = modules.pilot.pilot_cases(marks())
    save("admission-cases.json", cases)
    save("admission-code-mark.json", code)
    save("summary.json", {"status": "admission_cost_pilot_passed", "cases": 6})

    def record(identifier, model, prompt_tokens, output_tokens):
        return {"request_id": identifier, "model": model, "prompt_tokens": prompt_tokens,
                "token_ids": [1] * output_tokens, "arrived_at": 1, "submitted_at": 2,
                "first_token_at": 3, "completed_at": 10,
                "cadence": [{"time": 3, "tokens": 1}, {"time": 10, "tokens": output_tokens}]}

    save("admission-references.json", [record("ref0", 0, 100000, 64), record("ref1", 1, 3584, 1024)])

    def observation(requests):
        owners = {0: []}
        for request in requests:
            for page in sorted({block // 64 for block in request["blocks"]}):
                owners.setdefault(page, []).append(request["id"])
        pages = [{"id": page, "owners": sorted(identifiers), "pinned": page == 0}
                 for page, identifiers in sorted(owners.items())]
        return {"requests": requests, "pages": pages, "mapped_bytes": len(pages) * 96 * 1024**2,
                "page_bytes": 96 * 1024**2, "block_bytes": int(1.5 * 1024**2),
                "block_size": 16, "page_block_capacity": 64, "mapped_free_block_slots": 0,
                "uniform_page_geometry": True}

    trials = []
    for index, case in enumerate(cases):
        before = [[observation([]), observation([])], []]
        for rank in (0, 1):
            requests = [{"id": f"conv-{index}-{rank}-{position}",
                         "blocks": list(range((1 + position * 5) * 64, (1 + position * 5) * 64 + 288)),
                         "movable": True, "output_tokens": 32, "reservation_pending": False}
                        for position in range(case["population"] // 2)]
            before[1].append(observation(requests))
        after = copy.deepcopy(before)
        moves = []
        while len(after[1][0]["requests"]) > 3:
            moved = after[1][0]["requests"].pop()
            moved["blocks"] = list(range((1 + len(after[1][1]["requests"]) * 5) * 64,
                (1 + len(after[1][1]["requests"]) * 5) * 64 + 288))
            after[1][1]["requests"].append(moved)
            moves.append({"request_id": moved["id"], "model": 1, "source": 0, "destination": 1,
                          "kv_bytes": 432 * 1024**2, "integrity": "device_echo_full_bytewise",
                          "bitwise_verified": True, "sha256_scope": "verified_integrity_receipt_not_KV_content",
                          "sha256": "a" * 64, "copy_peak_gib": [6, 8]})
        after[1] = [observation(rank["requests"]) for rank in after[1]]
        post = copy.deepcopy(after)
        post[0][0] = observation([{"id": "code-arrival", "blocks": list(range(1, 6255)),
                                  "reservation_pending": False, "output_tokens": 0}])
        block_bytes = before[0][0]["block_bytes"]
        decision = {"admitted": True, "submitted": True, "actual_allocator_blocked_after_plan": False,
                    "reservation_pending": False, "observations": before, "before_admission": after,
                    "after_admission": post, "winner": {"target": 0}, "moves": moves, "solve_s": .01, "preplan_s": 0,
                    "pause_call_s": .2, "pause_receipts": None, "pilot_arrival_to_control_finish_s": 1}
        if case["mode"] == "preplan_boundary":
            decision.update({"solve_s": 0, "preplan_s": .02,
                             "pause_receipts": [[{"rank": rank, "armed_at": 3, "reached_at": 4,
                                "callback_at": 5, "boundary_step_counter": 32} for rank in (0, 1)]
                                for model in (0, 1)]})
        used = [sum(before[model][rank]["mapped_bytes"] for model in (0, 1)) / 1024**3 for rank in (0, 1)]
        witness = {"initial_gib": used, "global_free_bytes": round((2 * 11328 / 1024 - sum(used)) * 1024**3),
                   "incoming_logical_bytes": 6254 * block_bytes, "direct_ranks": [],
                   "allocation_increment_gib": [98 * 96 / 1024] * 2}
        trials.append({"case": case, "before": before, "witness": witness, "decision": decision,
                       "arriving_at": 1, "requests": [record(f"conv-{position}", 1, 3584, 1024)
                           for position in range(case["population"])] + [record("code", 0, 100000, 64)]})
    save("admission-trials.json", trials)
    save("admission-captures.json", [[[{"histogram": [{"mode": "FULL", "steps": 100}]}]
                                     for rank in (0, 1)] for model in (0, 1)])
    (root / "exit-status").write_text("0")
    return inputs, trials


@pytest.mark.parametrize("fault", [None, "tokens", "witness", "integrity", "boundary", "missing_case", "budget"])
def test_independent_admission_verifier_rejects_corruption(modules, tmp_path, monkeypatch, fault):
    inputs, trials = fixture(tmp_path, modules)
    monkeypatch.setattr(modules.verifier, "verify_prompt_bank", lambda *args: None)
    if fault == "tokens":
        trials[0]["requests"][0]["token_ids"][0] = 2
    elif fault == "witness":
        trials[0]["witness"]["global_free_bytes"] += 1
    elif fault == "integrity":
        trials[0]["decision"]["moves"][0]["bitwise_verified"] = False
    elif fault == "boundary":
        trials[1]["decision"]["pause_receipts"][1][0]["boundary_step_counter"] = 31
    elif fault == "missing_case":
        trials.pop()
    elif fault == "budget":
        trials[0]["decision"]["moves"][0]["copy_peak_gib"][0] = 100
    (tmp_path / "admission-trials.json").write_text(json.dumps(trials))
    if fault:
        with pytest.raises(ValueError):
            modules.verifier.verify(tmp_path, inputs)
    else:
        result = modules.verifier.verify(tmp_path, inputs)
        assert result["cases"] == 6 and result["requests"] == 78
        assert "not_final" in result["scope"]


@pytest.mark.parametrize("wait_s,admitted", [(100, True), (.001, False)])
def test_prepared_migration_gate_uses_reused_plan_and_actual_transaction(modules, tmp_path, monkeypatch, wait_s, admitted):
    controller_module = importlib.import_module("live_controller")
    _, trials = fixture(tmp_path, modules)
    trial = trials[1]
    observations = copy.deepcopy(trial["before"])
    for ranks in observations:
        for rank in ranks:
            for request in rank["requests"]:
                request["max_tokens"] = 1024
            for page in rank["pages"]:
                page["memory_gib"] = rank["page_bytes"] / 1024**3
    winner = {"target": 0, "moves": [trial["decision"]["moves"][0]["request_id"]],
              "objective_s": .1, "migration_cost_s": .1, "performance_cost_s": 0}
    order = []

    def solve(*args, **options):
        order.append("solve")
        return {"winner": winner}

    monkeypatch.setattr(controller_module, "enumerate_targets", solve)
    monkeypatch.setattr(controller_module, "performance_tables", lambda *args: [[0, 0], [0] * 10])

    async def run():
        controller = controller_module.PlacementController([], {}, tmp_path / "transfer")

        async def snapshots(*args, **options):
            return copy.deepcopy(observations)

        async def boundary(current):
            order.append("pause")
            now = controller_module.time.monotonic()
            return [[{"reached_at": now - .03} for rank in (0, 1)] for model in (0, 1)]

        async def begin(transaction):
            order.append("begin")
            return copy.deepcopy(observations)

        async def migrate(*args):
            order.append("move")
            observations[:] = copy.deepcopy(trial["decision"]["before_admission"])
            return trial["decision"]["moves"][0]

        async def admit(target):
            order.append("admit")
            observations[:] = copy.deepcopy(trial["decision"]["after_admission"])
            return {"reservation_pending": False}

        async def finish(*args):
            order.append("end")

        async def resume():
            order.append("resume")

        async def release():
            return []

        controller.snapshots, controller.pause_boundary, controller.begin = snapshots, boundary, begin
        controller.migrate, controller.end, controller.resume = migrate, finish, resume
        controller.release_transfer_cache = release
        prepared = await controller.prepare_admission(100000, 64, [11328 / 1024] * 2,
                                                       candidate_limit=8, move_fixed_s=.01, move_bandwidth_gib_s=32)
        return await controller.schedule_admission(100000, 64, [11328 / 1024] * 2, admit,
            candidate_limit=8, move_fixed_s=.01, move_bandwidth_gib_s=32, prepared=prepared, wait_cost_s=wait_s)

    decision = asyncio.run(run())
    assert decision["admitted"] is admitted and order.count("solve") == 1
    assert decision["solve_s"] == 0 and order.index("solve") < order.index("pause")
    assert ("move" in order) is admitted
    assert decision["active_blocking_upper_request_s"] > 0
    assert "observations" not in decision["preplan"] and "ownership" not in decision["preplan"]


@pytest.mark.parametrize("batch", range(5))
def test_mocked_startup_tests_clear_live_transport_without_changing_inference(tmp_path, batch):
    launch = (Path(__file__).resolve().parents[1] / "experiments/dp_opt/trace-benefit.sbatch").read_text()
    command = launch.split('env -u ENABLE_KVCACHED')[batch + 1].split('\ncheckpoint=', 1)[0]
    command = 'env -u ENABLE_KVCACHED' + command
    executable = tmp_path / "python"
    executable.write_text('#!/usr/bin/env bash\n[[ -z "${DP_OPT_KV_TRANSPORT+x}" ]]\n')
    executable.chmod(0o755)
    environment = {**os.environ, "DP_OPT_KV_TRANSPORT": "dp_echo", "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    subprocess.run(["bash", "-c", command], env=environment, check=True)
    assert environment["DP_OPT_KV_TRANSPORT"] == "dp_echo"
