import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("long_conv_poisson_campaign")


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    files = {"code-marks.json": [{"prompt_tokens": 100123, "output_tokens": 64, "prompt_key": "history"}],
        "conv-marks.json": [{"prompt_tokens": 4000, "output_tokens": 800, "count": 3,
                              "source_fragment": "paired-source"}], "prompt-bank.json": {"prompts": {}}}
    for name, value in files.items():
        (root / name).write_text(json.dumps(value))
    return root


def test_evaluation_is_frozen_only_after_measured_rate_and_is_independently_rebuildable(campaign, inputs, tmp_path):
    root = tmp_path / "evaluation"
    config = campaign.prepare(root, inputs, .0064, 12345)
    assert campaign.validate_evaluation(root / "frozen", inputs, .0064) == config
    assert config["policies"] == list(campaign.POLICIES)
    assert config["fairness_threshold_s"] == 30 and config["duration_s"] == 7200
    assert config["warmup_s"] == 600 and config["parent_calibration_job"] == 12345
    assert config["allow_empty_tenants"] and not config["research_goal_complete"]
    for seed in campaign.SEEDS:
        trace = json.loads((root / "frozen" / f"evaluation-{seed}.json").read_text())
        assert trace["arrival"] == "two_independent_Poisson_streams"
        assert trace["model_rates_rps"] == [.001, .0064]
        for mark in trace["requests"]:
            if mark["model"] == 1:
                assert mark["output_tokens"] == 24576 and mark["original_output_tokens"] == 800


def test_preserve_prior_evaluation_inputs(campaign, inputs, tmp_path):
    root = tmp_path / "evaluation"
    campaign.prepare(root, inputs, .0064, 12345)
    with pytest.raises(RuntimeError, match="never reroll"):
        campaign.prepare(root, inputs, .0064, 12345)


@pytest.mark.parametrize("rate,seed", [(.0065, 2026103002), (.0064, 10), (float("nan"), 2026103002)])
def test_unmeasured_rate_and_undeclared_seed_are_rejected(campaign, inputs, rate, seed):
    with pytest.raises(ValueError, match="predeclared"):
        campaign.make_trace(inputs, rate, seed)


@pytest.mark.parametrize("fault", ["rate", "fairness", "trace", "rerolled_trace", "bank", "pointer"])
def test_evaluation_validator_rejects_control_or_draw_changes(campaign, inputs, tmp_path, fault):
    root = tmp_path / "evaluation"
    campaign.prepare(root, inputs, .0064, 12345)
    frozen = root / "frozen"
    config_path = frozen / "evaluation-config.json"
    config = json.loads(config_path.read_text())
    if fault == "rate":
        config["model_rates_rps"][1] = .0072
    elif fault == "fairness":
        config["fairness_threshold_s"] = 1
    elif fault == "bank":
        (inputs / "code-marks.json").write_text("[]")
    elif fault == "pointer":
        (frozen / "trace.json").write_text("{}")
    else:
        name = f"evaluation-{campaign.SEEDS[0]}.json"
        path = frozen / name
        trace = json.loads(path.read_text())
        trace["requests"][0]["arrival_s"] += 1
        path.write_text(json.dumps(trace))
        if fault == "rerolled_trace":
            config["trace_sha256"][name] = campaign.digest(path)
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        campaign.validate_evaluation(frozen, inputs, .0064)


def test_parent_requires_every_calibration_stage_to_close(campaign):
    state = {"GPU_calibration_complete": True, "state_last_observed": "COMPLETED",
             "phase": "open_loop_calibration_SHA_verified_not_final_benefit"}
    assert campaign.parent_ready(state)
    assert not campaign.parent_ready({**state, "state_last_observed": "RUNNING"})
    assert not campaign.parent_ready({**state, "GPU_calibration_complete": False})


def test_wait_for_calibration_without_SSH_or_allocation(campaign, tmp_path, monkeypatch):
    root, parent = tmp_path / "evaluation", tmp_path / "calibration"
    root.mkdir()
    parent.mkdir()
    (parent / "campaign-state.json").write_text(json.dumps({"phase": "awaiting_verified_native_report"}))
    monkeypatch.setattr(campaign, "checked_ssh", lambda *args: pytest.fail("No SSH during parent activity"))
    def stop_after_interval(seconds):
        assert seconds == 300
        raise RuntimeError("bounded test")
    monkeypatch.setattr(campaign.time, "sleep", stop_after_interval)
    with pytest.raises(RuntimeError, match="bounded test"):
        campaign.await_calibration(root, parent, {"repo": str(tmp_path), "frozen_runtime_sha256": {}})
    state = json.loads((root / "campaign-state.json").read_text())
    assert not state["SSH_queries_while_waiting"] and "job" not in state


def test_submission_is_one_ledger_guarded_future_job(campaign):
    command = campaign.submission_command({"DP_OPT_POISSON_UPLOAD": "/upload path"}, "/ledger/job.txt")
    assert command.count("sbatch --parsable") == 1
    assert "squeue -h -u" in command and "test ! -e /ledger/job.txt" in command
    assert "long-conv-poisson.sbatch" in command and "'/upload path'" in command


def test_118_sources_extend_both_immutable_parent_sets(campaign):
    previous = importlib.import_module("long_conv_open_loop_campaign")
    assert set(previous.SYNC_FILES) <= set(campaign.SYNC_FILES)
    assert len(campaign.SYNC_FILES) == len(set(campaign.SYNC_FILES)) == 118


def test_driver_keeps_identical_settings_and_all_five_policies(campaign):
    text = Path(campaign.__file__).with_name("long-conv-poisson.sbatch").read_text()
    assert "#SBATCH --time=36:00:00" in text and "#SBATCH --partition=gpu_a800" in text
    assert "#SBATCH --gpus=2" in text and "DP_OPT_TRACE_BUDGET_MIB=10240" in text
    assert "DP_OPT_MAX_MODEL_LEN=131072" in text and "DP_OPT_PAUSE_SYNC_INTERVAL=32" in text
    assert "DP_OPT_FAIRNESS_QUEUE_THRESHOLD_S=30" in text
    assert "DP_OPT_WORKAWARE_OPTIMIZER=robust" in text
    for policy in campaign.POLICIES:
        assert policy in text
    assert "for seed in 2026103002 2026103003" in text
    assert "ENFORCE_EAGER" not in text


@pytest.fixture
def raw(campaign, tmp_path):
    root = tmp_path / "raw"
    for seed in campaign.SEEDS:
        phase = root / f"evaluation-{seed}"
        phase.mkdir(parents=True)
        offered = [{"request_id": name, "model": model, "arrival_s": arrival}
                   for name, model, arrival in (("early-conv", 1, 10), ("conv", 1, 700), ("code", 0, 1000))]
        (phase / "trace.json").write_text(json.dumps({"requests": offered}))
        for ordinal, policy in enumerate(campaign.POLICIES):
            case = phase / f"r0-{policy}"
            case.mkdir()
            origin = seed * 10
            (case / "workload.json").write_text(json.dumps({"origin": origin}))
            requests = [{**source, "arrived_at": origin + source["arrival_s"],
                "submitted_at": origin + source["arrival_s"] + .1,
                "reserved_at": origin + source["arrival_s"] + .2,
                "first_token_at": origin + source["arrival_s"] + 2 + ordinal,
                "completed_at": origin + source["arrival_s"] + 10 + ordinal, "token_ids": [1, 2, 3, 4]}
                for source in offered]
            (case / "requests.json").write_text(json.dumps(requests))
    return root


def test_pooled_metrics_keep_warmup_all_requests_and_report_every_policy(campaign, raw):
    result = campaign.pooled_metrics(raw)
    assert set(result) == set(campaign.POLICIES)
    for policy in campaign.POLICIES:
        assert result[policy]["combined"]["all_offered"]["completed_requests"] == 6
        assert result[policy]["combined"]["measurement_arrivals"]["completed_requests"] == 4
        assert result[policy]["Conv"]["all_offered"]["completed_requests"] == 4
        assert result[policy]["Conv"]["measurement_arrivals"]["completed_requests"] == 2
    assert result["no_move"]["Conv"]["all_offered"]["mean_ttft_s"] == 2
    assert "makespan_s" not in result["no_move"]["combined"]["all_offered"]
    text = campaign.markdown(12345, result)
    assert "all_offered" in text and "measurement_arrivals" in text
    assert "not P99 significance" in text and "Negative results" in text


def test_partial_policy_drain_cannot_produce_pooled_metrics(campaign, raw):
    path = raw / f"evaluation-{campaign.SEEDS[0]}" / "r0-kv_only_workaware/requests.json"
    requests = json.loads(path.read_text())
    path.write_text(json.dumps(requests[:-1]))
    with pytest.raises(ValueError, match="drain all"):
        campaign.pooled_metrics(raw)


def test_empty_tenants_report_NA_instead_of_invented_tail(campaign, raw):
    for phase in raw.iterdir():
        (phase / "trace.json").write_text('{"requests": []}')
        for policy in campaign.POLICIES:
            (phase / f"r0-{policy}/requests.json").write_text("[]")
    result = campaign.pooled_metrics(raw)
    assert result["no_move"]["Code"]["all_offered"]["mean_ttft_s"] is None
    assert "N/A" in campaign.markdown(12345, result)
    assert campaign.comparisons(result)["no_move"]["Code"]["all_offered"]["mean_ttft_s"] is None


def test_comparisons_preserve_regressions_and_every_baseline(campaign, raw):
    result = campaign.comparisons(campaign.pooled_metrics(raw))
    assert set(result) == set(campaign.POLICIES) - {"kv_only_workaware"}
    assert result["no_move"]["Conv"]["all_offered"]["mean_ttft_s"] == -3
    assert result["random_no_return"]["Code"]["measurement_arrivals"]["mean_ttft_s"] == pytest.approx(1 - 8 / 6)


def test_no_qualifying_rate_blocks_allocation_without_retuning(campaign, tmp_path, monkeypatch):
    root, parent = tmp_path / "evaluation", tmp_path / "calibration"
    root.mkdir()
    parent.mkdir()
    state = {"GPU_calibration_complete": True, "state_last_observed": "COMPLETED", "job": 12345,
        "phase": "open_loop_calibration_SHA_verified_not_final_benefit", "frozen_runtime_sha256": {},
        "inputs": str(tmp_path), "candidate_case": None}
    (parent / "campaign-state.json").write_text(json.dumps(state))
    (parent / "raw-12345.sha256").write_text("fake_test_manifest")
    (parent / "calibration-analysis.json").write_text('{"candidate_case": null}')
    monkeypatch.setattr(campaign, "verify_raw", lambda *args: None)
    monkeypatch.setattr(campaign, "validate_upload", lambda *args: {})
    monkeypatch.setattr(campaign, "analyze", lambda *args: {"candidate_case": None})
    monkeypatch.setattr(campaign, "checked_ssh", lambda *args: pytest.fail("No qualifying rate must not allocate"))
    with pytest.raises(RuntimeError, match="No independently qualifying rate"):
        campaign.await_calibration(root, parent, {"repo": str(tmp_path), "frozen_runtime_sha256": {}})


def test_calibration_failure_is_not_silently_retried(campaign, tmp_path, monkeypatch):
    root, parent = tmp_path / "evaluation", tmp_path / "calibration"
    root.mkdir()
    parent.mkdir()
    (parent / "campaign-state.json").write_text('{"phase": "diagnosis_required"}')
    monkeypatch.setattr(campaign, "checked_ssh", lambda *args: pytest.fail("No allocation on failed calibration"))
    with pytest.raises(RuntimeError, match="Calibration failure"):
        campaign.await_calibration(root, parent, {"repo": str(tmp_path), "frozen_runtime_sha256": {}})


def test_actual_live_cli_accepts_seven_policy_robust_evaluation(campaign, tmp_path, monkeypatch):
    live = importlib.import_module("live_trace")
    observed = []
    async def collect(args):
        observed.append(args)
    monkeypatch.setattr(live, "run", collect)
    monkeypatch.setattr(sys, "argv", ["live_trace.py", "--checkpoint", "test-checkpoint", "--fit", "test-fit",
        "--trace", "test-trace", "--output", str(tmp_path), "--policies", *campaign.POLICIES,
        "--workaware-optimizer", "robust", "--workaware-fit", "test-workaware-fit",
        "--workaware-fit-sha256", "a" * 64, "--workaware-source-root", "test-calibration-source",
        "--workaware-rate-priors", ".001", ".0064", "--fairness-queue-threshold-s", "30"])
    live.main()
    assert observed[0].policies == list(campaign.POLICIES)
    assert observed[0].workaware_optimizer == "robust"
