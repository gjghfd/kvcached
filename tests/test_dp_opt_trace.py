import asyncio
import hashlib
import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("poisson_trace"), importlib.import_module("live_trace")


def test_trace_is_frozen_paired_poisson_and_reproducible(modules):
    generator, _ = modules
    rows = [{"prompt_tokens": 2048, "output_tokens": 256}, {"prompt_tokens": 8192, "output_tokens": 512}]
    trace = generator.generate_trace(rows, .6, 600, 42)
    assert trace == generator.generate_trace(rows, .6, 600, 42)
    assert {request["model"] for request in trace["requests"]} == {0, 1}
    assert all(0 < request["arrival_s"] < 600 for request in trace["requests"])
    assert all((request["prompt_tokens"], request["output_tokens"]) in ((2048, 256), (8192, 512))
               for request in trace["requests"])
    assert len({request["request_id"] for request in trace["requests"]}) == len(trace["requests"])


def test_trace_rejects_out_of_limit_marks(modules):
    generator, _ = modules
    with pytest.raises(ValueError, match="supported full sequence"):
        generator.generate_trace([{"prompt_tokens": 32768, "output_tokens": 16}], 1, 100, 42)


def test_trace_metrics_include_queue_drain_and_all_requests(modules):
    _, runner = modules
    requests = [{"request_id": "a", "arrived_at": 1, "submitted_at": 2, "first_token_at": 3,
                 "completed_at": 5, "token_ids": [1, 2, 3]},
                {"request_id": "b", "arrived_at": 8, "submitted_at": 12, "first_token_at": 13,
                 "completed_at": 15, "token_ids": [1]}]
    summary = runner.summarize_trace(requests, 0, 10, 5)
    assert summary["completed_requests"] == 2 and summary["drain_s"] == 5
    assert summary["mean_queue_s"] == 2.5 and summary["completed_within_arrival_window"] == 1
    assert summary["deadline_met_requests"] == 1 and summary["mean_tpot_s"] == 1


def test_direct_fastpath_never_pauses_and_includes_capacity_ack(modules, tmp_path):
    _, runner = modules
    observation = {"requests": [{"movable": False}], "mapped_bytes": 96 * 1024**2,
                   "uniform_page_geometry": True, "page_block_capacity": 64, "page_bytes": 96 * 1024**2,
                   "block_size": 16, "mapped_free_block_slots": 63}

    async def run():
        controller = runner.PlacementController([], {}, tmp_path)
        controller.snapshots = AsyncMock(return_value=[[observation, observation], [observation, observation]])
        controller.check_capacity = AsyncMock()
        controller.pause = AsyncMock()
        admit = AsyncMock()
        result = await controller.try_direct_admission(64, 256, [.1875, .1875], admit)
        assert result["admitted"] and result["pause_call_s"] == 0
        controller.pause.assert_not_awaited()
        admit.assert_awaited_once()
        controller.check_capacity.assert_awaited_once()
        controller.check_capacity.assert_awaited_once_with([.1875, .1875], [[observation, observation], [observation, observation]])
        assert controller.snapshots.await_count == 2

    asyncio.run(run())


def test_direct_fastpath_blocked_has_no_admission_or_pause(modules, tmp_path):
    _, runner = modules
    observation = {"requests": [], "mapped_bytes": 96 * 1024**2,
                   "uniform_page_geometry": True, "page_block_capacity": 64, "page_bytes": 96 * 1024**2,
                   "block_size": 16, "mapped_free_block_slots": 0}

    async def run():
        controller = runner.PlacementController([], {}, tmp_path)
        controller.snapshots = AsyncMock(return_value=[[observation, observation], [observation, observation]])
        controller.pause = AsyncMock()
        admit = AsyncMock()
        result = await controller.try_direct_admission(2048, 512, [.1875, .1875], admit)
        assert not result["admitted"] and result["target"] is None
        controller.pause.assert_not_awaited()
        admit.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("reverse_write_delay", [False, True])
@pytest.mark.parametrize("policy", ["ilp", "kv_only", "kv_only_rebalance"])
def test_centralized_dispatch_preserves_arrivals_and_completes_both_tenants(modules, monkeypatch, tmp_path,
                                                                        reverse_write_delay, policy):
    _, runner = modules
    trace = {"duration_s": .05, "requests": [
        {"request_id": "a", "model": 0, "arrival_s": .01, "prompt_tokens": 8, "output_tokens": 2,
         "source_fragment": "sample", "source_internal_row": 1},
        {"request_id": "b", "model": 1, "arrival_s": .02, "prompt_tokens": 8, "output_tokens": 2,
         "source_fragment": "sample", "source_internal_row": 2}]}
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace))
    observation = {"requests": [], "mapped_bytes": 0, "block_size": 16, "block_bytes": 1024}
    records = [[observation, observation], [observation, observation]]

    async def stream(engine, identifier, prompt, output, rank, model, role, arrival):
        import time
        submitted = time.monotonic()
        await asyncio.sleep(.01)
        completed = time.monotonic()
        return {"request_id": identifier, "model": model, "role": role, "rank_initial": rank,
                "prompt_tokens": len(prompt), "arrived_at": arrival, "submitted_at": submitted,
                "first_token_at": submitted, "completed_at": completed, "token_ids": [1, 2], "cadence": []}

    async def direct(prompt, output, budgets, admit, arriving_model):
        await admit(0)
        return {"admitted": True, "observations": records, "moves": [], "end_to_end_control_s": .01}

    monkeypatch.setattr(runner, "stream", stream)
    monkeypatch.setattr(runner, "wait_reserved", AsyncMock())
    monkeypatch.setattr(runner, "wait_empty", AsyncMock(return_value=records))
    monkeypatch.setattr(runner, "capture_control", AsyncMock(return_value=[]))
    if reverse_write_delay:
        append = runner.append_record

        async def delayed_append(path, value):
            if path.name == "requests.jsonl" and value["request_id"] == "a":
                await asyncio.sleep(.05)
            await append(path, value)

        monkeypatch.setattr(runner, "append_record", delayed_append)
    args = SimpleNamespace(kv_budget_mib=2112, trace=trace_path, output=tmp_path, request_timeout_s=5,
                           queue_poll_s=.01, deadline_s=20, solver_limit_s=.1, move_fixed_s=.01)
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2])

    async def run():
        controller = SimpleNamespace(lock=asyncio.Lock(), events=[], snapshots=AsyncMock(return_value=records),
                                     try_direct_admission=AsyncMock(side_effect=direct), schedule_admission=AsyncMock(),
                                     rebalance_after_release=AsyncMock(return_value={"path": "release_rebalance_ilp",
                                         "moves": [], "end_to_end_control_s": .01, "preplan_s": 0}))
        result = await runner.run_trace([None, None], controller, args, trace, tokenizer, policy, 0, tmp_path / "trial")
        controller.schedule_admission.assert_not_awaited()
        if policy == "kv_only_rebalance":
            assert controller.rebalance_after_release.await_count >= 1
            assert result["release_rebalance"]["transactions"] == 0
        else:
            controller.rebalance_after_release.assert_not_awaited()
        return result

    result = asyncio.run(run())
    assert result["completed_requests"] == 2
    workload = json.loads((tmp_path / "trial/workload.json").read_text())
    for request in json.loads((tmp_path / "trial/requests.json").read_text()):
        mark = next(mark for mark in trace["requests"] if mark["request_id"] == request["request_id"])
        assert request["arrived_at"] == workload["origin"] + mark["arrival_s"]
    for name in ("requests", "decisions", "population"):
        journal = [json.loads(line) for line in (tmp_path / f"trial/{name}.jsonl").read_text().splitlines()]
        assert journal == json.loads((tmp_path / f"trial/{name}.json").read_text())
    if policy == "kv_only_rebalance":
        decisions = json.loads((tmp_path / "trial/decisions.json").read_text())
        released = {event["request_id"] for decision in decisions if decision["path"] == "release_rebalance_ilp"
                    for event in decision["release_events"]}
        assert released == {"a", "b"}


def test_incremental_journal_retains_prior_events(modules, tmp_path):
    _, runner = modules
    path = tmp_path / "events.jsonl"

    async def run():
        await runner.append_record(path, {"index": 1, "token_ids": [1, 2]})
        prior = path.read_bytes()
        await runner.append_record(path, {"index": 2, "token_ids": [3, 4]})
        assert path.read_bytes().startswith(prior)

    asyncio.run(run())
    assert len(path.read_text().splitlines()) == 2


def test_release_worker_failure_aborts_trace_and_drains_tasks(modules, monkeypatch, tmp_path):
    _, runner = modules
    trace = {"duration_s": .01, "requests": [{"request_id": "code", "model": 0, "arrival_s": 0,
              "prompt_tokens": 8, "output_tokens": 2, "source_fragment": "sample", "source_internal_row": 1}]}
    records = [[{"requests": [], "mapped_bytes": 0, "block_size": 16, "block_bytes": 1024}] * 2] * 2

    async def stream(engine, identifier, prompt, output, rank, model, role, arrival):
        import time

        submitted = time.monotonic()
        await asyncio.sleep(.01)
        return {"request_id": identifier, "model": model, "arrived_at": arrival, "submitted_at": submitted,
                "first_token_at": submitted, "completed_at": time.monotonic(), "token_ids": [1, 2]}

    async def direct(prompt, output, budgets, admit, arriving_model):
        await admit(0)
        return {"admitted": True, "observations": records, "moves": [], "end_to_end_control_s": .01}

    monkeypatch.setattr(runner, "stream", stream)
    monkeypatch.setattr(runner, "wait_reserved", AsyncMock())
    monkeypatch.setattr(runner, "wait_empty", AsyncMock())
    monkeypatch.setattr(runner, "capture_control", AsyncMock(return_value=[]))
    args = SimpleNamespace(kv_budget_mib=2112, output=tmp_path, request_timeout_s=5, queue_poll_s=.01,
                           deadline_s=20, solver_limit_s=.1, move_fixed_s=.01,
                           prepared_prompts={"8": [1] * 8})

    async def run():
        controller = SimpleNamespace(lock=asyncio.Lock(), events=[], snapshots=AsyncMock(return_value=records),
            try_direct_admission=AsyncMock(side_effect=direct),
            rebalance_after_release=AsyncMock(side_effect=RuntimeError("uncertain KV transfer")))
        with pytest.raises(RuntimeError, match="uncertain KV transfer"):
            await runner.run_trace([None, None], controller, args, trace, None, "kv_only_rebalance", 0, tmp_path / "trial")
        assert all(task.done() or task is asyncio.current_task() for task in asyncio.all_tasks())

    asyncio.run(run())
    assert len(json.loads((tmp_path / "trial/requests.json").read_text())) == 1
    assert not (tmp_path / "trial/summary.json").exists()


def test_independent_verifier_rejects_missing_journal_event(raw_trace):
    verifier = importlib.import_module("verify_trace")
    directory = raw_trace / "r0-ilp"
    records = json.loads((directory / "requests.json").read_text())
    (directory / "requests.jsonl").write_text("\n".join(json.dumps(record) for record in reversed(records)) + "\n")
    assert verifier.verify(raw_trace)["status"] == "raw_trace_verified"
    (directory / "requests.jsonl").write_text(json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="Incremental journal"):
        verifier.verify(raw_trace)


@pytest.fixture
def raw_trace(modules, tmp_path):
    _, runner = modules

    def save(name, value):
        artifact = tmp_path / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(value))

    marks = [{"request_id": "a", "model": 0, "arrival_s": 1, "prompt_tokens": 16, "output_tokens": 2},
             {"request_id": "b", "model": 1, "arrival_s": 2, "prompt_tokens": 16, "output_tokens": 2}]
    save("trace.json", {"requests": marks, "duration_s": 10})
    save("config.json", {"max_model_len": 32768, "repeats": 1, "deadline_s": 20, "policies": runner.POLICIES})
    save("summary.json", {"status": "passed"})
    (tmp_path / "exit-status").write_text("0")
    save("order.json", [{"repeat": 0, "policy": policy} for policy in runner.POLICIES])
    references = []
    for model in (0, 1):
        for rank in (0, 1):
            save(f"worker-instance-{model + 1}-dp{rank}.json",
                 {"ipc_name": f"model{model}rank{rank}", "gpu_uuid": f"gpu{rank}",
                  "max_model_len": 32768, "dp_pause_sync_interval": 32,
                  "groups": {name: {"world_size": size} for name, size in (("tp", 1), ("dp", 2), ("ep", 2))},
                  "expert_ids": [list(range(rank * 64, (rank + 1) * 64))] * 48})
    for policy in runner.POLICIES:
        directory = f"r0-{policy}"
        requests = [{"request_id": mark["request_id"], "model": mark["model"], "prompt_tokens": 16,
                     "arrived_at": mark["arrival_s"], "submitted_at": mark["arrival_s"] + .1,
                     "first_token_at": mark["arrival_s"] + .2, "completed_at": mark["arrival_s"] + .3,
                     "token_ids": [4, 5]} for mark in marks]
        save(directory + "/requests.json", requests)
        save(directory + "/workload.json", {"origin": 0, "requests": marks, "capacity_gib": [2, 2]})
        save(directory + "/summary.json", {**runner.summarize_trace(requests, 0, 10, 20), "policy": policy,
                                            "repeat": 0, "moves": 0, "migration_kv_bytes": 0,
                                            "control_s": 0, "ilp_attempts": 0})
        save(directory + "/population.json", [{"mapped_gib": [1, 1]}])
        save(directory + "/decisions.json", [])
        save(directory + "/migrations.json", [])
        save(directory + "/captures.json", [[[{"histogram": [{"mode": "FULL", "steps": 8}]}]] * 2] * 2)
        references = [{"trace_request_id": request["request_id"], "token_ids": request["token_ids"]}
                      for request in requests]
    save("references.json", references)
    return tmp_path


def test_independent_verifier_accepts_nested_collective_capture(raw_trace):
    verifier = importlib.import_module("verify_trace")
    result = verifier.verify(raw_trace)
    assert result["status"] == "raw_trace_verified" and result["requests_per_policy"] == 2
    assert len(result["paired_comparisons"]) == 3
    assert result["trials"][0]["tenants"]["0"]["completed_requests"] == 1


def test_verifier_supports_capacity_first_trials_without_ilp(raw_trace):
    verifier = importlib.import_module("verify_trace")
    config = json.loads((raw_trace / "config.json").read_text())
    config["policies"] = ["no_move", "kv_only"]
    (raw_trace / "config.json").write_text(json.dumps(config))
    order = [{"repeat": 0, "policy": policy} for policy in config["policies"]]
    (raw_trace / "order.json").write_text(json.dumps(order))
    result = verifier.verify(raw_trace)
    assert result["ilp_vs_no_move"] is None
    assert result["paired_comparisons"] == []
    assert result["capacity_first_paired_comparisons"][0]["baseline"] == "no_move"
    assert result["capacity_first_paired_comparisons"][0]["treatment"] == "kv_only"


def test_complete_verifier_includes_fair_queue_baseline(raw_trace):
    verifier = importlib.import_module("verify_trace")
    config = json.loads((raw_trace / "config.json").read_text())
    config["policies"] = ["no_move", "no_move_fair", "kv_only"]
    config["fairness_queue_threshold_s"] = 30
    (raw_trace / "config.json").write_text(json.dumps(config))
    (raw_trace / "order.json").write_text(json.dumps([{"repeat": 0, "policy": policy} for policy in config["policies"]]))
    directory = raw_trace / "r0-no_move_fair"
    shutil.copytree(raw_trace / "r0-no_move", directory)
    records = json.loads((directory / "requests.json").read_text())
    decisions = []
    for record in records:
        record.update({"reserved_at": record["submitted_at"], "reservation_observed_at": record["submitted_at"],
                       "arrival_dispatch_lag_s": 0})
        decisions.append({"request_id": record["request_id"], "admitted": True, "submitted": True, "moves": [],
                          "pause_call_s": 0, "end_to_end_control_s": 0, "path": "direct_fastpath",
                          "fairness_authorization": {"observed_at": record["submitted_at"], "request_id": record["request_id"],
                                                     "model": record["model"], "owner_request_id": None,
                                                     "owner_model": None, "allowed": True}})
    summary = json.loads((directory / "summary.json").read_text())
    summary.update({"policy": "no_move_fair", "fairness": {"threshold_s": 30, "owner_at_end": None,
                                                           "activations": 0, "releases": 0, "suppressed_attempts": 0}})
    for name, value in (("requests", records), ("decisions", decisions), ("summary", summary), ("fairness-events", [])):
        (directory / f"{name}.json").write_text(json.dumps(value))
    result = verifier.verify(raw_trace)
    assert len(result["capacity_first_paired_comparisons"]) == 2
    fair = next(trial for trial in result["trials"] if trial["policy"] == "no_move_fair")
    assert fair["fairness_verification"]["status"] == "raw_queue_fairness_verified"


def test_agentic_verifier_requires_frozen_bank(raw_trace):
    verifier = importlib.import_module("verify_trace")
    trace_path = raw_trace / "trace.json"
    trace = json.loads(trace_path.read_text())
    mark = trace["requests"][0]
    mark.update({"prompt_key": "swe", "source_traj_id": "episode"})
    bank = raw_trace / "prompt-bank.json"
    bank.write_text(json.dumps({"prompts": {"swe": {"prompt_tokens": 16,
                                                   "token_ids": [3] * 16, "traj_id": "episode"}}}))
    trace["prompt_bank_sha256"] = hashlib.sha256(bank.read_bytes()).hexdigest()
    trace_path.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="frozen prompt bank"):
        verifier.verify(raw_trace)
    for policy in ("no_move", "own_model", "kv_only", "ilp"):
        directory = raw_trace / f"r0-{policy}"
        workload = json.loads((directory / "workload.json").read_text())
        workload["requests"] = trace["requests"]
        (directory / "workload.json").write_text(json.dumps(workload))
        requests = json.loads((directory / "requests.json").read_text())
        requests[0]["prompt_key"] = "swe"
        (directory / "requests.json").write_text(json.dumps(requests))
    assert verifier.verify(raw_trace, bank)["status"] == "raw_trace_verified"
    bank.write_text(bank.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        verifier.verify(raw_trace, bank)


def test_agentic_bank_rejects_different_episode(modules, tmp_path):
    verifier = importlib.import_module("verify_trace")
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"prompts": {"swe": {"prompt_tokens": 2,
                                                   "token_ids": [1, 2], "traj_id": "other"}}}))
    trace = {"prompt_bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
             "requests": [{"prompt_key": "swe", "source_traj_id": "episode", "prompt_tokens": 2}]}
    with pytest.raises(ValueError, match="trajectory"):
        verifier.verify_prompt_bank(trace, bank)


def test_asymmetric_residency_uses_per_model_rates_and_window_clipping(modules):
    analyzer = importlib.import_module("analyze_trace")
    requests = [{"model": 0, "prompt_tokens": 16, "token_ids": [1, 2],
                 "submitted_at": 5, "completed_at": 15},
                {"model": 1, "prompt_tokens": 16, "token_ids": [1, 2],
                 "submitted_at": 2, "completed_at": 6}]
    result = analyzer.model_residency(requests, [.01, 1], 0, 10, 16, 1024**2)
    assert result[0]["nominal_rate_times_measured_KV_seconds_mib"] == pytest.approx(.2)
    assert result[1]["nominal_rate_times_measured_KV_seconds_mib"] == pytest.approx(8)
    assert result[0]["arrival_window_time_average_reserved_mib"] == 1
    assert result[1]["arrival_window_time_average_reserved_mib"] == .8


@pytest.mark.parametrize("corruption", ("dropped_request", "changed_tokens", "capacity_excess"))
def test_independent_verifier_rejects_invalid_trace(raw_trace, corruption):
    verifier = importlib.import_module("verify_trace")
    directory = raw_trace / "r0-ilp"
    if corruption == "capacity_excess":
        (directory / "population.json").write_text(json.dumps([{"mapped_gib": [3, 1]}]))
    else:
        requests = json.loads((directory / "requests.json").read_text())
        if corruption == "dropped_request":
            requests.pop()
        else:
            requests[0]["token_ids"] = [7, 8]
        (directory / "requests.json").write_text(json.dumps(requests))
    with pytest.raises(ValueError):
        verifier.verify(raw_trace)


@pytest.mark.parametrize("metric", ("drain_s", "mean_tpot_s", "completed_within_arrival_window", "control_s"))
def test_independent_verifier_rejects_metric_corruption(raw_trace, metric):
    verifier = importlib.import_module("verify_trace")
    artifact = raw_trace / "r0-ilp/summary.json"
    summary = json.loads(artifact.read_text())
    summary[metric] += 1
    artifact.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Independent"):
        verifier.verify(raw_trace)


def test_independent_verifier_requires_every_rank_capture(raw_trace):
    verifier = importlib.import_module("verify_trace")
    artifact = raw_trace / "r0-ilp/captures.json"
    captures = json.loads(artifact.read_text())
    captures[1][1] = [{"histogram": [{"mode": "NONE", "steps": 8}]}]
    artifact.write_text(json.dumps(captures))
    with pytest.raises(ValueError, match="tenant/rank"):
        verifier.verify(raw_trace)


def test_control_breakdown_does_not_treat_direct_rpc_as_gpu_pause(modules):
    verifier = importlib.import_module("verify_trace")
    decisions = [{"path": "direct_fastpath", "end_to_end_control_s": 3},
                 {"path": "blocked_staged_ilp", "end_to_end_control_s": 7, "admitted": True,
                  "pause_call_s": 2, "solve_s": 1, "resume_s": .5, "moves": [{"elapsed_s": 3}]}]
    result = verifier.control_metrics(decisions)
    assert result["direct_control_s"] == 3 and result["staged_control_s"] == 7
    assert result["pause_call_s"] == 2 and result["staged_admissions"] == 1


def test_trace_backlog_uses_exact_frozen_arrivals_and_completions(modules):
    analyzer = importlib.import_module("analyze_trace")
    requests = [{"arrived_at": 1, "submitted_at": 2, "completed_at": 5},
                {"arrived_at": 3, "submitted_at": 4, "completed_at": 7},
                {"arrived_at": 5, "submitted_at": 6, "completed_at": 8}]
    assert analyzer.peak_intervals(requests, "completed_at") == 2
    assert analyzer.peak_intervals(requests, "submitted_at") == 1


def test_same_snapshot_action_signature_retains_migration_identity(modules):
    analyzer = importlib.import_module("analyze_trace")
    assert analyzer.plan_signature({"winner": None}) is None
    assert analyzer.plan_signature({"winner": {"target": 1, "moves": ["request-a"]}}) == (1, ("request-a",))


def test_trace_queue_includes_allocator_wait_and_independent_metrics_match(modules):
    records = [{"request_id": "queued", "arrived_at": 1, "submitted_at": 2, "reserved_at": 7,
                "first_token_at": 8, "completed_at": 10, "token_ids": [1, 2]}]
    summary = modules[1].summarize_trace(records, 0, 5, 20)
    independent = importlib.import_module("verify_trace").request_metrics(records, 0, 5, 20)
    assert summary["mean_queue_s"] == 6
    assert all(summary[key] == value for key, value in independent.items())


def test_direct_fastpath_rejects_real_allocator_shortage(modules, tmp_path):
    async def run():
        controller = modules[1].PlacementController([], {}, tmp_path)
        record = {"requests": [], "mapped_bytes": 96 * 1024**2, "uniform_page_geometry": True,
                  "page_block_capacity": 64, "page_bytes": 96 * 1024**2, "block_size": 16,
                  "mapped_free_block_slots": 63, "free_blocks": 1}
        controller.snapshots = AsyncMock(return_value=[[record, record], [record, record]])
        admit = AsyncMock()
        result = await controller.try_direct_admission(64, 256, [1, 1], admit)
        assert not result["submitted"] and not result["admitted"]
        admit.assert_not_awaited()

    asyncio.run(run())


def test_direct_submission_does_not_claim_pending_kv_is_admitted(modules, tmp_path):
    async def run():
        controller = modules[1].PlacementController([], {}, tmp_path)
        record = {"requests": [], "mapped_bytes": 96 * 1024**2, "uniform_page_geometry": True,
                  "page_block_capacity": 64, "page_bytes": 96 * 1024**2, "block_size": 16,
                  "mapped_free_block_slots": 63, "free_blocks": 100}
        controller.snapshots = AsyncMock(return_value=[[record, record], [record, record]])
        controller.check_capacity = AsyncMock()
        result = await controller.try_direct_admission(64, 256, [1, 1],
                                                       AsyncMock(return_value={"reservation_pending": True}))
        assert result["submitted"] and not result["admitted"] and result["reservation_pending"]

    asyncio.run(run())


def test_direct_confirmation_reuses_target_and_keeps_other_tenant_upper_bound(modules, tmp_path):
    async def run():
        controller = modules[1].PlacementController([], {}, tmp_path)
        observation = {"requests": [], "mapped_bytes": 96 * 1024**2, "uniform_page_geometry": True,
                       "page_block_capacity": 64, "page_bytes": 96 * 1024**2, "block_size": 16,
                       "mapped_free_block_slots": 63, "free_blocks": 100}
        before = [[dict(observation), dict(observation)], [dict(observation), dict(observation)]]
        updated = {**observation, "mapped_bytes": 192 * 1024**2}
        controller.snapshots = AsyncMock(return_value=before)
        controller.check_capacity = AsyncMock()
        result = await controller.try_direct_admission(64, 256, [1, 1],
            AsyncMock(return_value={"reservation_pending": False, "rank_observation": updated, "rank_observed": 0}))
        assert result["admitted"] and controller.snapshots.await_count == 1
        assert result["after_admission"][0][0] is updated
        assert result["after_admission"][1][0] is before[1][0]
        assert before[0][0]["mapped_bytes"] == 96 * 1024**2
        controller.check_capacity.assert_awaited_once_with([1, 1], result["after_admission"])

    asyncio.run(run())


def admission_record(mapped_mib=96, free_slots=63):
    return {"requests": [], "mapped_bytes": mapped_mib * 1024**2, "uniform_page_geometry": True,
            "page_block_capacity": 64, "page_bytes": 96 * 1024**2, "block_size": 16,
            "mapped_free_block_slots": free_slots, "free_blocks": 100}


def test_rank_local_admission_avoids_code_rpc_when_cached_headroom_is_safe(modules, tmp_path):
    async def run():
        controller = modules[1].PlacementController([], {}, tmp_path)
        controller.rank_local_admission = True
        before = [[admission_record(), admission_record()], [admission_record(), admission_record()]]
        controller.admission_observations = before
        controller.snapshots = AsyncMock(side_effect=RuntimeError("Code prefill RPC must not block Conv"))
        controller.call = AsyncMock(return_value=admission_record())
        controller.check_capacity = AsyncMock()
        updated = admission_record(192)
        result = await controller.try_direct_admission(64, 256, [1, 1], AsyncMock(return_value={
            "reservation_pending": False, "rank_observation": updated, "rank_observed": 0}), arriving_model=1)
        assert result["admitted"] and result["observations"][0] == before[0]
        assert result["after_admission"][1][0] is updated
        assert {call.args[0] for call in controller.call.await_args_list} == {1}
        controller.snapshots.assert_not_awaited()
        assert controller.admission_observations[1][0] is updated
        assert controller.admission_generation == 1
        assert "cached_other_upper_bounds" in result["observations_scope"]

    asyncio.run(run())


def test_rank_local_blocked_bound_refreshes_actual_releases(modules, tmp_path):
    async def run():
        controller = modules[1].PlacementController([], {}, tmp_path)
        controller.rank_local_admission = True
        controller.admission_observations = [[admission_record(1000), admission_record(1000)],
                                              [admission_record(), admission_record()]]
        controller.call = AsyncMock(return_value=admission_record())
        fresh = [[admission_record(), admission_record()], [admission_record(), admission_record()]]
        controller.snapshots = AsyncMock(return_value=fresh)
        controller.check_capacity = AsyncMock()
        result = await controller.try_direct_admission(64, 256, [1, 1], AsyncMock(return_value={
            "reservation_pending": False, "rank_observation": fresh[1][0], "rank_observed": 0}), arriving_model=1)
        assert result["admitted"]
        assert result["observations_scope"] == "fresh_all_tenants"
        controller.snapshots.assert_awaited_once()

    asyncio.run(run())


def test_rank_local_admission_failure_invalidates_budget_ledger(modules, tmp_path):
    async def run():
        controller = modules[1].PlacementController([], {}, tmp_path)
        controller.rank_local_admission = True
        controller.admission_observations = [[admission_record(), admission_record()],
                                              [admission_record(), admission_record()]]
        controller.call = AsyncMock(return_value=admission_record())
        with pytest.raises(RuntimeError, match="driver"):
            await controller.try_direct_admission(64, 256, [1, 1], AsyncMock(side_effect=RuntimeError("driver")))
        assert controller.admission_observations is None
        assert controller.admission_generation == 1

    asyncio.run(run())


def test_rank_local_cached_bound_never_authorizes_overbudget_submission(modules, tmp_path):
    async def run():
        controller = modules[1].PlacementController([], {}, tmp_path)
        controller.rank_local_admission = True
        observations = [[admission_record(1000), admission_record(1000)],
                        [admission_record(), admission_record()]]
        controller.admission_observations = observations
        controller.call = AsyncMock(return_value=admission_record())
        controller.snapshots = AsyncMock(return_value=observations)
        admit = AsyncMock()
        result = await controller.try_direct_admission(64, 256, [1, 1], admit, arriving_model=1)
        assert not result["admitted"] and not result["submitted"]
        admit.assert_not_awaited()
        controller.snapshots.assert_awaited_once()
        assert controller.admission_generation == 0

    asyncio.run(run())


def test_rank_local_target_growth_is_accounted_before_cache_commit(modules, tmp_path):
    async def run():
        controller = modules[1].PlacementController([], {}, tmp_path)
        controller.rank_local_admission = True
        controller.admission_observations = [[admission_record(), admission_record()],
                                              [admission_record(), admission_record()]]
        controller.call = AsyncMock(return_value=admission_record())
        with pytest.raises(MemoryError, match="mapped KV"):
            await controller.try_direct_admission(64, 256, [1, 1], AsyncMock(return_value={
                "reservation_pending": False, "rank_observation": admission_record(1100), "rank_observed": 0}))
        assert controller.admission_observations is None

    asyncio.run(run())


@pytest.mark.parametrize("changed", [False, True])
def test_monitor_rpc_does_not_hold_admission_lock_and_detects_epoch_change(modules, tmp_path, changed):
    async def run():
        controller = modules[1].PlacementController([], {}, tmp_path)
        observed = [[admission_record(), admission_record()], [admission_record(), admission_record()]]

        async def snapshot():
            assert not controller.lock.locked()
            async with controller.lock:
                controller.admission_generation += int(changed)
            return observed

        controller.snapshots = snapshot
        result = await controller.monitor_snapshot()
        assert result["admission_stable_across_observation"] is not changed
        assert result["admission_generation_before"] == 0
        assert result["admission_generation_after"] == int(changed)
        assert result["observations"] is observed

    asyncio.run(run())


@pytest.mark.parametrize("conv_mapped_mib,code_admitted", [(192, True), (920, False)])
def test_concurrent_code_observation_never_blocks_conv_and_merges_latest_budget(
        modules, tmp_path, conv_mapped_mib, code_admitted):
    async def run():
        controller = modules[1].PlacementController([], {}, tmp_path)
        controller.rank_local_admission = controller.concurrent_admission = True
        controller.admission_observations = [[admission_record(), admission_record()],
                                              [admission_record(), admission_record()]]
        code_query_started, code_query_release = asyncio.Event(), asyncio.Event()
        current = [[admission_record(), admission_record()], [admission_record(), admission_record()]]

        async def snapshot(model, rank, method):
            assert not controller.lock.locked()
            if model == 0:
                code_query_started.set()
                await code_query_release.wait()
            return current[model][rank]

        async def snapshots():
            assert not controller.lock.locked()
            return current

        async def conv_admit(rank):
            current[1][rank] = admission_record(conv_mapped_mib)
            return {"reservation_pending": False, "rank_observed": rank,
                    "rank_observation": current[1][rank]}

        controller.call, controller.snapshots = snapshot, snapshots
        code_admit = AsyncMock(return_value={"reservation_pending": False,
            "rank_observed": 0, "rank_observation": admission_record()})
        code_task = asyncio.create_task(controller.try_direct_admission(768, 256, [1, 1], code_admit))
        await asyncio.wait_for(code_query_started.wait(), 1)
        conv_result = await asyncio.wait_for(controller.try_direct_admission(
            64, 256, [1, 1], conv_admit, arriving_model=1), 1)
        assert conv_result["admitted"] and not code_task.done()
        current[1][1] = admission_record(conv_mapped_mib)
        controller.admission_observations[1][1] = current[1][1]
        code_query_release.set()
        code_result = await asyncio.wait_for(code_task, 1)
        assert code_result["admitted"] is code_admitted
        assert code_result["observations"][1][0]["mapped_bytes"] == conv_mapped_mib * 1024**2
        assert code_result["observations"][1][1]["mapped_bytes"] == conv_mapped_mib * 1024**2
        if not code_admitted:
            code_admit.assert_not_awaited()
        assert not controller.lock.locked()

    asyncio.run(run())


def test_concurrent_same_model_epoch_change_requeries_before_admission(modules, tmp_path):
    async def run():
        controller = modules[1].PlacementController([], {}, tmp_path)
        controller.rank_local_admission = controller.concurrent_admission = True
        before = [[admission_record(), admission_record()], [admission_record(), admission_record()]]
        controller.admission_observations = before
        calls = []

        async def call(model, rank, method):
            calls.append((model, rank))
            if len(calls) == 1:
                async with controller.lock:
                    controller.model_admission_generations[model] += 1
            return admission_record()

        controller.call = call
        result = await controller.try_direct_admission(64, 256, [1, 1], AsyncMock(return_value={
            "reservation_pending": False, "rank_observed": 0, "rank_observation": admission_record()}))
        assert result["admitted"] and len(calls) == 4
        assert not controller.lock.locked()

    asyncio.run(run())


def test_concurrent_cancelled_observation_leaves_budget_and_locks_untouched(modules, tmp_path):
    async def run():
        controller = modules[1].PlacementController([], {}, tmp_path)
        controller.rank_local_admission = controller.concurrent_admission = True
        before = [[admission_record(), admission_record()], [admission_record(), admission_record()]]
        controller.admission_observations = before
        started = asyncio.Event()

        async def call(*args):
            started.set()
            await asyncio.Event().wait()

        controller.call = call
        task = asyncio.create_task(controller.try_direct_admission(64, 256, [1, 1], AsyncMock()))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not controller.lock.locked() and not controller.model_admission_locks[0].locked()
        assert controller.admission_observations is before
        assert controller.admission_generation == 0

    asyncio.run(run())


@pytest.mark.parametrize("concurrent", [False, True])
def test_allocator_wait_does_not_block_other_tenant_or_resubmit_same_request(modules, monkeypatch, tmp_path, concurrent):
    runner = modules[1]
    trace = {"duration_s": .1, "requests": [
        {"request_id": identifier, "model": model, "arrival_s": arrival, "prompt_tokens": 8,
         "output_tokens": 2, "source_fragment": "sample", "source_internal_row": index}
        for index, (identifier, model, arrival) in enumerate((("a", 0, .01), ("b", 1, .02), ("c", 0, .03)))]}
    record = {"requests": [], "mapped_bytes": 0, "block_size": 16, "block_bytes": 1024}
    observations = [[record, record], [record, record]]
    submissions = []

    async def stream(engine, identifier, prompt, output, rank, model, role, arrival):
        import time
        submitted = time.monotonic()
        submissions.append((identifier, submitted))
        await asyncio.sleep(.25 if identifier == "a" else .02)
        now = time.monotonic()
        return {"request_id": identifier, "model": model, "role": role, "rank_initial": rank,
                "prompt_tokens": len(prompt), "arrived_at": arrival, "submitted_at": submitted,
                "first_token_at": now, "completed_at": now, "token_ids": [1, 2], "cadence": []}

    async def reserve(controller, task, identifier, model, rank, **options):
        import time
        if identifier == "a" and not options.get("wait_for_capacity", False):
            return {"reservation_pending": True, "reserved_at": None, "reservation_time_source": "pending"}
        if identifier == "a":
            await asyncio.sleep(.15)
        return {"reservation_pending": False, "reserved_at": time.monotonic(),
                "reservation_time_source": "scheduler_monotonic"}

    async def direct(prompt, output, budgets, admit, arriving_model):
        result = await admit(0)
        return {"admitted": not result["reservation_pending"], "submitted": True,
                "observations": observations, "moves": [], "end_to_end_control_s": .01}

    monkeypatch.setattr(runner, "stream", stream)
    monkeypatch.setattr(runner, "wait_reserved", reserve)
    monkeypatch.setattr(runner, "wait_empty", AsyncMock(return_value=observations))
    monkeypatch.setattr(runner, "capture_control", AsyncMock(return_value=[]))
    args = SimpleNamespace(kv_budget_mib=2112, trace=tmp_path / "trace.json", output=tmp_path,
                           request_timeout_s=5, queue_poll_s=.01, deadline_s=20)

    async def run():
        controller = SimpleNamespace(lock=asyncio.Lock(), events=[], snapshots=AsyncMock(return_value=observations),
                                     check_capacity=AsyncMock(), try_direct_admission=AsyncMock(side_effect=direct),
                                     schedule_admission=AsyncMock(), concurrent_admission=concurrent)
        return await runner.run_trace([None, None], controller, args, trace,
                                      SimpleNamespace(encode=lambda *args, **kwargs: [1, 2]),
                                      "ilp", 0, tmp_path / "trial")

    result = asyncio.run(run())
    assert result["completed_requests"] == 3
    assert [identifier for identifier, stamp in submissions] == ["a", "b", "c"]
    assert submissions[1][1] - submissions[0][1] < .15
    assert submissions[2][1] - submissions[0][1] >= .15
    records = json.loads((tmp_path / "trial/requests.json").read_text())
    assert next(record for record in records if record["request_id"] == "a")["reserved_at"] > submissions[1][1]


def test_independent_dispatcher_keeps_conv_fifo_while_code_observation_waits(modules, monkeypatch, tmp_path):
    runner = modules[1]
    trace = {"duration_s": .1, "requests": [
        {"request_id": identifier, "model": model, "arrival_s": arrival, "prompt_tokens": 8,
         "output_tokens": 2, "source_fragment": "sample", "source_internal_row": index}
        for index, (identifier, model, arrival) in enumerate((("code", 0, .01), ("conv1", 1, .02), ("conv2", 1, .03)))]}
    record = {"requests": [], "mapped_bytes": 0, "block_size": 16, "block_bytes": 1024}
    observations = [[record, record], [record, record]]
    submissions = []

    async def stream(engine, identifier, prompt, output, rank, model, role, arrival):
        import time
        now = time.monotonic()
        submissions.append(identifier)
        return {"request_id": identifier, "model": model, "role": role, "rank_initial": rank,
                "prompt_tokens": len(prompt), "arrived_at": arrival, "submitted_at": now,
                "first_token_at": now, "completed_at": now, "token_ids": [1, 2], "cadence": []}

    async def reserve(*args, **options):
        import time
        return {"reservation_pending": False, "reserved_at": time.monotonic(),
                "reservation_time_source": "snapshot_upper_bound"}

    async def direct(prompt, output, budgets, admit, arriving_model):
        if arriving_model == 0:
            await asyncio.sleep(.15)
        await admit(0)
        return {"admitted": True, "submitted": True, "observations": observations,
                "moves": [], "end_to_end_control_s": .01}

    monkeypatch.setattr(runner, "stream", stream)
    monkeypatch.setattr(runner, "wait_reserved", reserve)
    monkeypatch.setattr(runner, "wait_empty", AsyncMock(return_value=observations))
    monkeypatch.setattr(runner, "capture_control", AsyncMock(return_value=[]))
    args = SimpleNamespace(kv_budget_mib=2112, output=tmp_path, request_timeout_s=5, queue_poll_s=.01, deadline_s=20)

    async def run():
        controller = SimpleNamespace(lock=asyncio.Lock(), events=[], snapshots=AsyncMock(return_value=observations),
                                     check_capacity=AsyncMock(), try_direct_admission=AsyncMock(side_effect=direct),
                                     schedule_admission=AsyncMock(), concurrent_admission=True)
        return await runner.run_trace([None, None], controller, args, trace,
                                      SimpleNamespace(encode=lambda *args, **options: [1, 2]),
                                      "no_move", 0, tmp_path / "trial")

    assert asyncio.run(run())["completed_requests"] == 3
    assert submissions == ["conv1", "conv2", "code"]
