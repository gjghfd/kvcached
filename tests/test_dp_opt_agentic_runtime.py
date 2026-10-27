import asyncio
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("live_trace")


@pytest.mark.parametrize("fault", [None, "trace_sha", "rate_sha", "capacity", "formal_failure"])
def test_frozen_formal_replay_requires_same_inputs_and_sufficient_capacity(tmp_path, fault):
    scripts = Path(__file__).resolve().parents[1] / "experiments/dp_opt"
    launch = (scripts / "agentic-trace.sbatch").read_text()
    branch = 'if [[ "${DP_OPT_FROZEN_TRACE_REPLAY:-0}" == 1 ]]; then'
    replay = branch + launch.split(branch, 1)[1].split("\nfi\n", 1)[0] + "\nfi\n"
    frozen, evidence, repo = (tmp_path / name for name in ("source-frozen", "new-job", "repo"))
    for directory in (frozen, evidence, repo / "experiments/dp_opt", tmp_path / "bin"):
        directory.mkdir(parents=True)
    trace = frozen / "trace.json"
    rate = frozen / "rate-decision.json"
    trace.write_text(json.dumps({"requests": [{"model": 0, "prompt_tokens": 101675}], "duration_s": 1200}))
    rate.write_text(json.dumps({"conv_rate_rps": .61, "code_rate_rps": .004}))
    helper = tmp_path / "bin/sha256sum"
    helper.write_text(f'#!{sys.executable}\nimport hashlib\nimport pathlib\nimport sys\n'
                      'for filename in sys.argv[1:]:\n'
                      '    print(hashlib.sha256(pathlib.Path(filename).read_bytes()).hexdigest() + "  " + filename)\n')
    helper.chmod(0o755)
    serving = repo / "experiments/dp_opt/trace-benefit.sbatch"
    serving.write_text('printf "%s %s %s %s %s\\n" "$DP_OPT_TRACE_BUDGET_MIB" '
                       '"$DP_OPT_MAX_MOVES" "$DP_OPT_CANDIDATE_LIMIT" "$DP_OPT_CAPACITY_PILOT" '
                       '"$DP_OPT_RATE_CALIBRATION" > "$evidence/formal-called"\n'
                       + ("exit 3\n" if fault == "formal_failure" else ""))
    environment = {**os.environ, "LC_ALL": "C", "LANG": "C", "DP_OPT_FROZEN_TRACE_REPLAY": "1",
                   "PATH": str(tmp_path / "bin") + os.pathsep + os.environ["PATH"],
                   "DP_OPT_FROZEN_TRACE_PATH": str(trace),
                   "DP_OPT_FROZEN_TRACE_SHA256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                   "DP_OPT_FROZEN_RATE_DECISION_SHA256": hashlib.sha256(rate.read_bytes()).hexdigest(),
                   "DP_OPT_FROZEN_BUDGET_MIB": "11328", "DP_OPT_TRACE_BUDGET_MIB": "11424",
                   "DP_OPT_MAX_MOVES": "8", "DP_OPT_CANDIDATE_LIMIT": "8",
                   "evidence": str(evidence), "repo": str(repo)}
    if fault == "trace_sha":
        trace.write_text(trace.read_text() + " ")
    elif fault == "rate_sha":
        rate.write_text(rate.read_text() + " ")
    elif fault == "capacity":
        environment["DP_OPT_TRACE_BUDGET_MIB"] = "11232"
    result = subprocess.run(["bash", "-c", "set -euo pipefail\n" + replay], env=environment,
                            text=True, capture_output=True, timeout=30)
    assert (result.returncode == 0) == (fault is None)
    if fault in ("trace_sha", "rate_sha", "capacity"):
        assert not (evidence / "formal-called").exists()
    else:
        assert (evidence / "formal-called").read_text().strip() == "11328 8 8 0 0"
        assert (evidence / "frozen/trace.json").read_bytes() == trace.read_bytes()
        assert (evidence / "frozen/rate-decision.json").read_bytes() == rate.read_bytes()
        phase = json.loads((evidence / "phase.json").read_text())["stage"]
        assert phase == ("formal_trace_complete" if fault is None else "formal_trace_running")
    serving_source = (scripts / "trace-benefit.sbatch").read_text()
    assert '"$repo/experiments/dp_opt/agentic-formal.sbatch"' in serving_source
    formal = (scripts / "agentic-formal.sbatch").read_text()
    assert "export DP_OPT_MAX_MOVES=8 DP_OPT_CANDIDATE_LIMIT=8" in formal
    assert "#SBATCH --partition=gpu_a800" in formal


def test_real_prompt_bank_matches_runtime_template_and_ids(runner):
    text = "agent history"
    tokenizer = SimpleNamespace(encode=lambda content, **options: [1, 2],
                                apply_chat_template=lambda *args, **options: text)
    entry = {"messages": [{"role": "user", "content": text}], "token_ids": [1, 2],
             "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}
    trace = {"requests": [{"prompt_tokens": 2, "prompt_key": "swe"}]}
    assert runner.prepare_prompts(trace, tokenizer, {"prompts": {"swe": entry}}) == {"swe": [1, 2]}
    entry["token_ids"] = [2, 1]
    with pytest.raises(ValueError, match="token IDs"):
        runner.prepare_prompts(trace, tokenizer, {"prompts": {"swe": entry}})


def test_real_prompt_rejects_template_drift(runner):
    tokenizer = SimpleNamespace(encode=lambda content, **options: [1, 2],
                                apply_chat_template=lambda *args, **options: "changed")
    bank = {"prompts": {"swe": {"messages": [], "token_ids": [1, 2], "prompt_sha256": "old"}}}
    with pytest.raises(ValueError, match="chat template"):
        runner.prepare_prompts({"requests": [{"prompt_tokens": 2, "prompt_key": "swe"}]}, tokenizer, bank)


def test_a800_retry_bracket_is_fixed_before_policy_outcomes():
    script = (Path(__file__).resolve().parents[1] / "experiments/dp_opt/agentic-trace.sbatch").read_text()
    assert 'export DP_OPT_CALIBRATION_CONV_RATES="0.42 0.51 0.61 0.70"' in script
    assert script.index('source "$evidence/capacity/capacity-config.env"') < script.index("calibration-bracket.json")
    assert script.index("calibration-bracket.json") < script.index('DP_OPT_RATE_CALIBRATION=1')


def test_a800_resource_preflight_preserves_existing_gpu_processes(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    binaries = tmp_path / "bin"
    binaries.mkdir()
    fake = binaries / "nvidia-smi"
    fake.write_text('#!/bin/sh\ncase "$*" in\n*query-gpu*) printf "gpu0, NVIDIA A800, 81920, 1000\\ngpu1, NVIDIA A800, 81920, 80000\\n" ;;\n*) printf "foreign-process\\n" ;;\nesac\n')
    fake.chmod(0o755)
    control = binaries / "scontrol"
    control.write_text('#!/bin/sh\nprintf "allocated-pair\\n"\n')
    control.chmod(0o755)
    environment = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                   "KVCACHED_DATA_ROOT": str(tmp_path / "state"), "SLURM_JOB_ID": "123"}
    result = subprocess.run(["bash", str(repo / "experiments/dp_opt/agentic-trace.sbatch")],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 1 and "preserve" in result.stderr
    evidence = tmp_path / "state/state/live-trace-123"
    assert json.loads((evidence / "phase.json").read_text())["stage"] == "a800_resource_preflight_rejected"
    assert (evidence / "exit-status").read_text() == "1\n"
    assert "foreign-process" in (evidence / "gpu-processes-before-load.csv").read_text()


def test_open_loop_calibration_is_predeclared_and_saves_independent_traces(runner, monkeypatch, tmp_path):
    marks = tmp_path / "conv.json"
    marks.write_text(json.dumps([{"prompt_tokens": 2048, "output_tokens": 512,
                                 "source_fragment": "conv", "source_internal_row": 0}]))
    args = SimpleNamespace(calibration_conv_marks=marks, calibration_code_marks=None,
                           calibration_conv_rates=[1.65, 2.0, 2.35],
                           calibration_code_rate=.013, calibration_duration_s=300, seed=20261021,
                           max_model_len=131072, prompt_bank_data={}, output=tmp_path,
                           pilot_target_conv_mib=23552)
    trace = {"prompt_bank_sha256": "bank", "requests": [
        {"model": 0, "prompt_key": "swe", "prompt_tokens": 100084, "output_tokens": 64}]}
    calls = []

    async def run_trial(engines, controller, options, generated, tokenizer, policy, index, directory):
        calls.append((generated, policy, index))
        samples = [{"elapsed_s": 200, "reserved_mib": [[0, 0], [11000, 11000]],
                    "unfinished": 55, "active": 50,
                    "pending_heads": [{"model": 0, "global_enough_local_blocked": True}]}]
        (directory / "population.json").write_text(json.dumps(samples))
        return {"completed_requests": len(generated["requests"])}

    monkeypatch.setattr(runner, "run_trace", run_trial)
    monkeypatch.setattr(runner, "prepare_prompts", lambda *args: {})
    asyncio.run(runner.run_rate_calibration([], None, args, trace, None))
    assert [call[0]["model_rates_rps"] for call in calls] == [[.013, 1.65], [.013, 2.0], [.013, 2.35]]
    assert all(call[1] == "no_move" for call in calls)
    summary = json.loads((tmp_path / "calibration-summary.json").read_text())
    assert summary["status"] == "passed" and len(summary["trials"]) == 3
    assert all(trial["mean_sampled_conv_reserved_mib"] == 22000 for trial in summary["trials"])
    assert all(trial["global_enough_local_blocked_code_samples"] == 1 for trial in summary["trials"])


def test_a800_launcher_extends_calibration_and_uses_measured_kv_seconds(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "experiments/dp_opt"
    monkeypatch.syspath_prepend(str(scripts))
    launcher = (scripts / "agentic-trace.sbatch").read_text()
    record = next(line for line in launcher.splitlines() if line.startswith("printf '{\"conv_rates_rps\""))
    bracket = json.loads(record.split("'", 2)[1].removesuffix("\\n"))
    assert bracket["conv_rates_rps"] == [.42, .51, .61, .70]
    assert bracket["code_rate_rps"] == .004
    assert bracket["Code_rate_unchanged"] is True
    assert bracket["source"] == "240982_fixed_30_120s_arrival_subset_all_completed_before_Code"
    central = bracket["prior_target_conv_reserved_mib"] / bracket["reference_mean_reserved_mib_seconds"]
    assert [round(central * factor, 2) for factor in bracket["multipliers"]] == bracket["conv_rates_rps"]
    assert bracket["reference_requests"] == 82 and bracket["reference_conv_rate_rps"] == .80
    assert bracket["prior_nonmonotone_calibration_job"] == 241036
    assert bracket["seed_unchanged"] == 20261021 and bracket["calibration_duration_s"] == 1200
    assert "export DP_OPT_CALIBRATION_DURATION_S=1200" in launcher
    assert "export DP_OPT_CALIBRATION_CODE_RATE=0.004" in launcher
    assert "export DP_OPT_RANK_LOCAL_ADMISSION=1" in launcher
    assert "export DP_OPT_CONCURRENT_ADMISSION=1" in launcher
    serving = (scripts / "trace-benefit.sbatch").read_text()
    assert 'extra_args+=(--calibration-code-rate "$DP_OPT_CALIBRATION_CODE_RATE")' in serving
    assert 'extra_args+=(--calibration-duration-s "$DP_OPT_CALIBRATION_DURATION_S")' in serving


@pytest.mark.parametrize("with_transition", [False, True])
def test_a800_capacity_pilot_preserves_full_input_and_records_measured_budget(runner, monkeypatch, tmp_path,
                                                                           with_transition):
    from unittest.mock import AsyncMock

    code = {"prompt_tokens": 101675, "output_tokens": 64, "prompt_key": "swe"}
    conv = {"prompt_tokens": 3949, "output_tokens": 659}
    code_path, conv_path = tmp_path / "code-marks.json", tmp_path / "conv-marks.json"
    code_path.write_text(json.dumps([code]))
    conv_path.write_text(json.dumps([conv]))
    args = SimpleNamespace(calibration_code_marks=code_path, calibration_conv_marks=conv_path,
                           prompt_bank_data={}, output=tmp_path, capacity_guard_mib=1024,
                           allocator_gpu_utilization=.98)
    snapshots = [[{"mapped_bytes": 96 * 1024**2, "page_bytes": 96 * 1024**2} for rank in (0, 1)]
                 for model in (0, 1)]
    for ranks in snapshots:
        for record in ranks:
            record.update({"pages": [], "requests": []})
    calls = []

    async def memory(model, rank, *options):
        return [{"gpu_uuid": f"gpu-{rank}", "gpu_name": "NVIDIA A800", "free_mib": 12000,
                 "total_mib": 81920}]

    async def stream(engine, identifier, prompt, output, rank, model, role, arrived):
        calls.append({"model": model, "rank": rank, "prompt_tokens": len(prompt), "output_tokens": output})
        return {"model": model, "submitted_at": arrived, "completed_at": arrived + 10}

    monkeypatch.setattr(runner, "prepare_prompts", lambda *args: {"swe": [1] * 101675, "3949": [1] * 3949})
    monkeypatch.setattr(runner, "stream", stream)
    monkeypatch.setattr(runner, "wait_empty", AsyncMock())
    monkeypatch.setattr(runner, "capture_control", AsyncMock(return_value=[]))
    controller = SimpleNamespace(call=AsyncMock(side_effect=memory), snapshots=AsyncMock(return_value=snapshots),
                                 pause=AsyncMock(), resume=AsyncMock(), begin=AsyncMock(), end=AsyncMock(),
                                 probe=AsyncMock(return_value={"fits_budget": True, "rollback_verified": True}))
    if with_transition:
        changed = [[dict(record) for record in ranks] for ranks in snapshots]
        changed[0][0]["mapped_bytes"] = 2 * 96 * 1024**2
        controller.snapshots.side_effect = [snapshots, snapshots, changed, snapshots, snapshots, snapshots]
    asyncio.run(runner.run_capacity_pilot([None, None], controller, args, {}, None))
    assert controller.probe.await_count == 4
    assert next(call for call in calls if call["model"] == 0)["prompt_tokens"] == 101675
    assert all(call["rank"] == 1 for call in calls if call["model"] == 1)
    setting = json.loads((tmp_path / "capacity-setting.json").read_text())
    assert setting["kv_budget_mib_per_gpu"] == 10464
    assert len(setting["calibration"]["conv_rates_rps"]) == 4
    budget_line = (tmp_path / "capacity-config.env").read_text().splitlines()[0]
    assert int(budget_line.split("=", 1)[1]) == 10464
    samples = json.loads((tmp_path / "memory-samples.json").read_text())
    assert all(sample["ownership_before_memory_query"] == sample["ownership_after_memory_query"] for sample in samples)
    if with_transition:
        transitions = json.loads((tmp_path / "memory-transitions.json").read_text())
        assert len(transitions) == 1 and len(samples) == 2
