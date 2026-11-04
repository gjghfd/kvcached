import asyncio
import copy
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import live_trace
from analyze_native_divergence import analyze as analyze_failure
from live_controller import PlacementController
from static_batch_probe import LAYOUTS, StaticBatchController, analyze, instrument, validate_trace


def trace():
    return {"requests": [{"request_id": "conv0", "model": 1, "prompt_tokens": 74, "output_tokens": 3, "arrival_s": 0},
                         {"request_id": "conv1", "model": 1, "prompt_tokens": 149, "output_tokens": 3, "arrival_s": 1},
                         {"request_id": "code", "model": 0, "prompt_tokens": 101626, "output_tokens": 3, "arrival_s": 90}]}


def record(mark, rank=0):
    return {**{key: mark[key] for key in ("request_id", "model", "prompt_tokens")},
            "rank_initial": rank, "arrival_offset_s": mark["arrival_s"], "arrived_at": 100 + mark["arrival_s"],
            "token_ids": [10, 20, 30], "token_logprobs": [{str(token): -.5, "99": -.51} for token in (10, 20, 30)],
            "cadence": [{"tokens": count, "time": 100 + mark["arrival_s"] + count} for count in (1, 2, 3)]}


def write_fixture(root):
    marks = trace()["requests"]
    (root / "config.json").write_text(json.dumps({"policies": ["no_move"], "repeats": len(LAYOUTS),
                                                 "max_model_len": 131072, "cache_bytes": 25769803776,
                                                 "kv_budget_mib": 10240, "allocator_gpu_utilization": .98}))
    (root / "runtime-control.json").write_text(json.dumps({"dp_pause_sync_interval": 32,
                                                          "same_interval_for_all_policies": True,
                                                          "idle_rank_expert_execution_not_disabled": True}))
    for model in (0, 1):
        for rank in (0, 1):
            worker = {"ipc_name": f"model-{model}-rank-{rank}", "gpu_uuid": str(rank),
                      "max_model_len": 131072, "kv_cache_memory_bytes": 25769803776,
                      "dp_pause_sync_interval": 32,
                      "groups": {name: {"world_size": size} for name, size in (("tp", 1), ("dp", 2), ("ep", 2))}}
            (root / f"worker-instance-{model + 1}-dp{rank}.json").write_text(json.dumps(worker))
    (root / "trace.json").write_text(json.dumps(trace()))
    (root / "references.json").write_text(json.dumps([{**record(mark), "trace_request_id": mark["request_id"]} for mark in marks]))
    for repeat, layout in enumerate(LAYOUTS):
        directory = root / f"r{repeat}-no_move"
        directory.mkdir()
        ranks = [0, 1] if layout == "balanced" else [int(layout[-1])] * 2
        (directory / "static-layout.json").write_text(json.dumps({"layout": layout, "repeat": repeat,
                                                                 "copies_allowed": False, "pause_allowed": False}))
        (directory / "requests.json").write_text(json.dumps([record(mark, ranks[index] if index < 2 else 0)
                                                              for index, mark in enumerate(marks)]))
        (directory / "migrations.json").write_text("[]")
        observations = [[{"requests": [], "pages": [], "mapped_bytes": 0, "page_bytes": 1}
                         for rank in (0, 1)] for model in (0, 1)]
        (directory / "decisions.json").write_text(json.dumps([{"moves": [], "pause_call_s": 0,
                                                               "observations": observations}]))
        (directory / "captures.json").write_text(json.dumps([[{"histogram": [{"mode": "FULL", "steps": 1}]}
                                                              for rank in (0, 1)] for model in (0, 1)]))


def test_common_prefix_is_diagnostic_not_tolerance(tmp_path):
    write_fixture(tmp_path)
    records = json.loads((tmp_path / "r1-no_move/requests.json").read_text())
    records[0]["token_ids"][1] = 99
    records[0]["token_logprobs"][1] = {"20": -.501, "99": -.499}
    (tmp_path / "r1-no_move/requests.json").write_text(json.dumps(records))
    error = {"type": "RuntimeError", "message": "Selected trace request differs from its full serial greedy reference"}
    (tmp_path / "error.json").write_text(json.dumps(error))
    result = analyze(tmp_path)
    assert result["no_copy_divergence_reproduced"]
    comparison = next(item for item in result["trials"][1]["serial_comparisons"] if item["request_id"] == "conv0")
    assert comparison["first_difference_zero_based"] == 1
    assert comparison["common_candidate_contrast"]["contrast_change"] == pytest.approx(.012)
    assert result["underlying_strict_error"] == error
    assert result["strict_token_check_relaxed"] is False
    assert result["migration_correctness_proven"] is False
    assert result["is_algorithm_benefit"] is False


def test_full_mirror_order_controls(tmp_path):
    write_fixture(tmp_path)
    result = analyze(tmp_path)
    assert len(result["trials"]) == 6
    assert [item["repeats"] for item in result["same_layout_repeat_comparisons"]] == [[0, 5], [1, 4], [2, 3]]
    assert not result["no_copy_divergence_reproduced"]


@pytest.mark.parametrize("field,value", [("prompt_tokens", 73), ("token_ids", [10]),
                                         ("token_logprobs", []), ("arrival_offset_s", 2), ("model", 0)])
def test_reject_changed_lengths_arrivals_or_tenant(tmp_path, field, value):
    write_fixture(tmp_path)
    path = tmp_path / "r1-no_move/requests.json"
    records = json.loads(path.read_text())
    records[0][field] = value
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="changed"):
        analyze(tmp_path)


@pytest.mark.parametrize("filename,value,reason", [
    ("migrations.json", [{}], "copy"), ("decisions.json", [{"pause_call_s": .1}], "pause"),
    ("decisions.json", [{"moves": [{}]}], "migrate"), ("requests.json", [], "drain"),
    ("static-layout.json", {}, "declaration")])
def test_reject_copy_pause_or_incomplete_control(tmp_path, filename, value, reason):
    write_fixture(tmp_path)
    (tmp_path / "r1-no_move" / filename).write_text(json.dumps(value))
    with pytest.raises(ValueError, match=reason):
        analyze(tmp_path)


def test_reject_wrong_actual_rank_and_unrelated_error(tmp_path):
    write_fixture(tmp_path)
    path = tmp_path / "r2-no_move/requests.json"
    records = json.loads(path.read_text())
    records[0]["rank_initial"] = 0
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="ranks"):
        analyze(tmp_path)
    records[0]["rank_initial"] = 1
    path.write_text(json.dumps(records))
    (tmp_path / "error.json").write_text('{"type":"RuntimeError", "message":"CUDA OOM"}')
    with pytest.raises(ValueError, match="unrelated"):
        analyze(tmp_path)


@pytest.mark.parametrize("layout,model,candidates,expected", [
    ("balanced", 1, [0, 1], [0, 1]), ("corank0", 1, [0, 1], [0]),
    ("corank1", 1, [0, 1], [1]), ("corank0", 1, [1], []),
    ("corank1", 0, [0, 1], [0, 1])])
def test_static_filter_never_enlarges_feasible_set(monkeypatch, layout, model, candidates, expected):
    observations, used, budgets = object(), [1, 2], [10, 10]

    @asynccontextmanager
    async def epoch(controller, incoming_model, tokens, capacity):
        assert incoming_model == model and tokens == 64 and capacity is budgets
        yield observations, "verified", used, candidates

    monkeypatch.setattr(PlacementController, "direct_admission_epoch", epoch)
    controller = object.__new__(StaticBatchController)
    controller.probe_layout = layout

    async def check():
        async with controller.direct_admission_epoch(model, 64, budgets) as result:
            assert result == (observations, "verified", used, expected)

    asyncio.run(check())


def test_instrument_restores_all_production_interfaces(monkeypatch, tmp_path):
    called = []

    async def run_trace(*args):
        called.append(args[1].probe_layout)
        return "unchanged_trace"

    monkeypatch.setattr(live_trace, "run_trace", run_trace)
    monkeypatch.setattr(live_trace, "engine_arguments", lambda *args, **kwargs: SimpleNamespace())
    original = (live_trace.stream, live_trace.PlacementController, live_trace.run_trace, live_trace.engine_arguments)
    with pytest.raises(RuntimeError, match="test"):
        with instrument():
            assert live_trace.engine_arguments().worker_cls == "workaware_worker.WorkawareVerificationWorker"
            result = asyncio.run(live_trace.run_trace([], SimpleNamespace(), SimpleNamespace(repeats=6),
                                                       trace(), None, "no_move", 1, tmp_path / "trial"))
            assert result == "unchanged_trace" and called == ["corank0"]
            raise RuntimeError("test")
    assert original == (live_trace.stream, live_trace.PlacementController, live_trace.run_trace, live_trace.engine_arguments)


def test_failure_diagnosis_checks_every_complete_request(tmp_path):
    marks = trace()["requests"]
    (tmp_path / "trace.json").write_text(json.dumps(trace()))
    (tmp_path / "references.json").write_text(json.dumps([{**record(mark), "trace_request_id": mark["request_id"]} for mark in marks]))
    (tmp_path / "order.json").write_text('[{"repeat":0,"policy":"no_move"}]')
    (tmp_path / "error.json").write_text(json.dumps({"type": "RuntimeError", "message": "Selected trace request differs from its full serial greedy reference"}))
    directory = tmp_path / "r0-no_move"
    directory.mkdir()
    records = [record(mark) for mark in marks]
    records[0]["token_ids"][1] = 99
    (directory / "requests.json").write_text(json.dumps(records))
    (directory / "migrations.json").write_text("[]")
    result = analyze_failure(tmp_path)
    assert result["requests_verified"] == 3
    comparison = result["trials"][0]["comparisons"][0]
    assert comparison["first_difference_zero_based"] == 1
    assert comparison["first_difference_offset_s"] == 2
    assert not result["is_algorithm_benefit"]
    (directory / "requests.json").write_text("[]")
    with pytest.raises(ValueError, match="undrained"):
        analyze_failure(tmp_path)


def test_reject_changed_trace_population():
    changed = copy.deepcopy(trace())
    changed["requests"][2]["prompt_tokens"] = 99999
    with pytest.raises(ValueError, match="intact"):
        validate_trace(changed)


@pytest.mark.parametrize("artifact", ["graph", "interval", "budget", "ipc"])
def test_reject_invalid_runtime_evidence(tmp_path, artifact):
    write_fixture(tmp_path)
    if artifact == "graph":
        (tmp_path / "r1-no_move/captures.json").write_text("[]")
    elif artifact == "interval":
        (tmp_path / "runtime-control.json").unlink()
    elif artifact == "budget":
        path = tmp_path / "r1-no_move/decisions.json"
        decisions = json.loads(path.read_text())
        decisions[0]["observations"][0][0].update({"mapped_bytes": 11 * 1024**3,
                                                  "page_bytes": 11 * 1024**3, "pages": [{"id": 1, "owners": []}]})
        path.write_text(json.dumps(decisions))
    else:
        path = tmp_path / "worker-instance-2-dp1.json"
        worker = json.loads(path.read_text())
        worker["ipc_name"] = "model-0-rank-0"
        path.write_text(json.dumps(worker))
    with pytest.raises(ValueError):
        analyze(tmp_path)


def test_wrapper_preserves_graphs_and_all_original_lengths():
    source = (Path(__file__).resolve().parents[1] / "experiments/dp_opt/static-batch-probe.sbatch").read_text()
    assert "DP_OPT_TRACE_REPEATS=6" in source and "DP_OPT_TRACE_POLICIES=no_move" in source
    assert "DP_OPT_MAX_MODEL_LEN=131072" in source and "DP_OPT_TRACE_BUDGET_MIB=10240" in source
    assert "unset VLLM_BATCH_INVARIANT" in source
    assert "batch_invariant=1" not in source.lower()
    assert "DP_OPT_PAUSE_SYNC_INTERVAL=32" in source
    assert "static_batch_probe.py" in source and "--verify" in source
