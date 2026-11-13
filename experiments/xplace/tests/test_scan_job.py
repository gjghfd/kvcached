import ast
import copy
import json
import os
from pathlib import Path
import sys
import subprocess
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
import scan_job as job


def inputs(root):
    root.mkdir(exist_ok=True)
    protocol = job.protocol_core()
    protocol["configs"] = {}
    for block in (0, 1):
        for mode in ("off", "collect"):
            path = root / f"block{block}-{mode}.json"
            path.write_text(json.dumps(job.overhead_config(block, mode)))
            protocol["configs"][f"{block}-{mode}"] = dict(file=path.name, sha256=job.sha256(path))
    path = root / "protocol.json"
    path.write_text(json.dumps(protocol))
    bank = root / "prompt-bank.json"
    bank.write_text("{}")
    manifest = dict(purpose=job.PURPOSE, protocol_sha256=job.sha256(path), prompt_bank_sha256=job.sha256(bank))
    common = ["--checkpoint", "fixed-model", "--prompt-bank", str(bank), "--phases", "dataplane,imbalance",
              "--policies", "no_move", "--num-ranks", "2", "--tensor-parallel-size", "2", "--enable-tp-migration",
              "--budget-margin-gib", "3", "--seed", "20261110", "--kv-mode", "prompt", "--reserve-extra-tokens", "256",
              "--max-inflight", "0", "0", "--drain-timeout-s", "210"]
    environment = dict(os.environ, XPLACE_JOB_STARTED_AT=str(time.time()), DP_OPT_TRANSFER_ROOT=str(root))
    return path, manifest, common, environment


def test_plan_has_exact_fixed_pairs_and_rotates_rank(tmp_path):
    path, manifest, common, environment = inputs(tmp_path)
    protocol, paths = job.load_protocol(path, manifest["protocol_sha256"])
    assert protocol["total_measured_pairs"] == 8 and protocol["requests_per_condition"] == 768
    for block in (0, 1):
        plain = job.scan_cases(json.loads(paths[f"{block}-off"].read_text()), 2, 2)
        observed = job.scan_cases(json.loads(paths[f"{block}-collect"].read_text()), 2, 2)
        assert [{key: value for key, value in case.items() if key != "observe_graph_execution"} for case in observed] == plain
        assert len(plain) == 4 and sum(case["batch_size"] for case in plain) == 768
        extremes = [case for case in plain if case["delta"]]
        assert all(case["layout"][block] == case["batch_size"] for case in extremes)


@pytest.mark.parametrize("mutation", ["order", "limit", "metric", "coverage", "bool_count", "config_hash", "config_seed", "config_context",
                                      "config_output", "config_bool_repeat", "config_path", "missing_mode", "extra", "manifest_hash"])
def test_protocol_rejects_changes(tmp_path, mutation):
    path, manifest, common, environment = inputs(tmp_path)
    protocol = json.loads(path.read_text())
    if mutation == "order":
        protocol["blocks"][1].reverse()
    elif mutation == "limit":
        protocol["overhead"]["per_metric_worst_limit"] = 0.9
    elif mutation == "metric":
        protocol["overhead"]["primary_metrics"].pop()
    elif mutation == "coverage":
        protocol["overhead"]["coverage_required"] = 0.5
    elif mutation == "bool_count":
        protocol["schema"] = True
    elif mutation == "config_hash":
        protocol["configs"]["0-off"]["sha256"] = "0" * 64
    elif mutation.startswith("config_") and mutation != "config_path":
        entry = protocol["configs"]["1-off"]
        config = path.parent / entry["file"]
        value = json.loads(config.read_text())
        key, replacement = dict(config_seed=("seed", 9), config_context=("context_tokens", 4096),
                                config_output=("output_tokens", 64), config_bool_repeat=("repeat_start", True))[mutation]
        value[key] = replacement
        config.write_text(json.dumps(value))
        entry["sha256"] = job.sha256(config)
    elif mutation == "config_path":
        protocol["configs"]["0-off"]["file"] = "../escaped.json"
    elif mutation == "missing_mode":
        del protocol["configs"]["1-off"]
    elif mutation == "extra":
        protocol["extra_override"] = True
    path.write_text(json.dumps(protocol))
    digest = "0" * 64 if mutation == "manifest_hash" else job.sha256(path)
    with pytest.raises(ValueError):
        job.load_protocol(path, digest)


@pytest.mark.parametrize("change", ["reserve", "max_inflight", "budget", "policy", "seed", "output", "duplicate", "abbreviated", "bank"])
def test_cli_freezes_common_conditions(tmp_path, change):
    path, manifest, common, environment = inputs(tmp_path)
    config = path.parent / "block0-off.json"
    job.validate_common(common, config, manifest["prompt_bank_sha256"])
    values = dict(reserve=("--reserve-extra-tokens", "0"), max_inflight=("--max-inflight", "1"),
                  budget=("--budget-margin-gib", "2"), policy=("--policies", "ca_full"), seed=("--seed", "9"))
    if change in values:
        key, value = values[change]
        common[common.index(key) + 1] = value
    elif change == "output":
        common += ["--output=escape"]
    elif change == "duplicate":
        common += ["--seed", "20261110"]
    elif change == "abbreviated":
        common += ["--out", "escape"]
    elif change == "bank":
        (tmp_path / "prompt-bank.json").write_text("changed")
    with pytest.raises(ValueError):
        job.validate_common(common, config, manifest["prompt_bank_sha256"])


@pytest.mark.parametrize("block", [0, 1])
def test_fixed_orders_restarts_isolate_and_clear_inherited_instrumentation(tmp_path, block):
    path, manifest, common, environment = inputs(tmp_path / "inputs")
    environment.update({key: "1" for key in job.DISABLED_INSTRUMENTATION})
    original = dict(environment)
    calls = []

    def runner(command, env, root, deadline):
        calls.append((command, env, root, deadline))
        return dict(status="passed", returncode=0)

    result = job.run_block(block=block, root=tmp_path / "out", protocol_path=path, manifest=manifest,
                           common_args=common, environment=environment, runner=runner)
    assert [Path(call[0][-1]).stem for call in calls] == [f"block{block}-{mode}" for mode in job.BLOCKS[block]]
    assert len({call[1]["DP_OPT_TRANSFER_ROOT"] for call in calls}) == 2 and calls[0][3] == calls[1][3]
    assert all(call[1]["XPLACE_SCAN_OVERHEAD"] == "1" for call in calls)
    assert all(all(call[1][key] == "0" for key in job.DISABLED_INSTRUMENTATION) for call in calls)
    assert environment == original and not result["overhead_accepted"] and not result["prediction_gate_passed"]
    assert result["status"] == "collected_not_overhead_accepted" and result["not_executed"] == []
    with pytest.raises(FileExistsError):
        job.run_block(block=block, root=tmp_path / "out", protocol_path=path, manifest=manifest,
                      common_args=common, environment=environment, runner=runner)


@pytest.mark.parametrize("failure", ["returncode", "timeout", "signal", "tampered_between_modes", "old_manifest", "no_budget"])
def test_all_failure_paths_leave_not_run_and_never_retry(tmp_path, failure):
    path, manifest, common, environment = inputs(tmp_path / "inputs")
    calls = []
    if failure == "old_manifest":
        manifest["purpose"] = "causal_training"
    if failure == "no_budget":
        environment["XPLACE_JOB_STARTED_AT"] = str(time.time() - 1900)

    def runner(command, env, root, deadline):
        calls.append(command)
        if failure == "timeout":
            raise TimeoutError("cpu test deadline")
        if failure == "signal":
            raise InterruptedError("cpu test signal")
        if failure == "tampered_between_modes":
            (path.parent / "block0-collect.json").write_text("{}")
            return dict(status="passed", returncode=0)
        return dict(status="failed", returncode=7)

    with pytest.raises((RuntimeError, TimeoutError, InterruptedError, ValueError)):
        job.run_block(block=0, root=tmp_path / "out", protocol_path=path, manifest=manifest,
                      common_args=common, environment=environment, runner=runner)
    result = json.loads((tmp_path / "out/status.json").read_text())
    assert result["status"] == "failed" and result["not_executed"] == (["off", "collect"] if not calls else ["collect"])
    assert len(calls) == (0 if failure in ("old_manifest", "no_budget") else 1)
    assert not result["overhead_accepted"]


def test_real_cpu_children_use_existing_cleanup_and_separate_logs(tmp_path):
    path, manifest, common, environment = inputs(tmp_path / "inputs")

    def runner(command, env, root, deadline):
        mode = Path(command[-1]).stem
        return job.run_child([sys.executable, "-c", f"print({mode!r})"], env, root, deadline,
                             sample=lambda group: dict(group=group, cpu_test=True))

    result = job.run_block(block=1, root=tmp_path / "out", protocol_path=path, manifest=manifest,
                           common_args=common, environment=environment, runner=runner)
    assert result["status"] == "collected_not_overhead_accepted"
    for mode in ("off", "collect"):
        record = json.loads((tmp_path / "out" / mode / "process.json").read_text())
        assert record["returncode"] == 0 and not record["forced_cleanup"]
        assert f"block1-{mode}" in (tmp_path / "out" / mode / "stdout.log").read_text()
        with pytest.raises(ProcessLookupError):
            os.kill(record["pid"], 0)


def test_native_main_attestation_guards_include_scan_and_preserve_causal(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "runtime/main.py"
    tree = ast.parse(path.read_text())
    conditions = [node.test for node in ast.walk(tree) if isinstance(node, ast.If)
                  and "XPLACE_SCAN_OVERHEAD" in ast.unparse(node.test) and '"causal"' in ast.unparse(node.test).replace("'", '"')]
    assert len(conditions) == 3
    for flag, phases, expected in (("1", ["dataplane", "imbalance"], True), ("0", ["causal"], True),
                                    ("0", ["dataplane", "imbalance"], False)):
        monkeypatch.setenv("XPLACE_SCAN_OVERHEAD", flag)
        for condition in conditions:
            assert eval(compile(ast.Expression(condition), str(path), "eval"), dict(os=os, phases=phases, verify=True)) is expected


@pytest.mark.parametrize("mode", ["scan", "causal", "direct", "conflict_block", "conflict_mode", "conflict_config", "conflict_imbalance"])
def test_actual_sbatch_dispatch_branch_without_slurm_or_gpu(tmp_path, mode):
    path = Path(__file__).resolve().parents[1] / "xplace.sbatch"
    source = path.read_text()
    branch = source[source.index('if [[ -n "${XPLACE_SCAN_BLOCK:-}"') :]
    prefix = '''set -euo pipefail
work="$TEST_ROOT"
args=(--output "$work/out" --checkpoint fixed-model)
python() {
    printf '%s\\n' "$@" > "$work/command.txt"
    printf '%s\\n' "${XPLACE_CAUSAL_MANIFEST:-}" "${XPLACE_CAUSAL_MANIFEST_SHA256:-}" > "$work/manifest-env.txt"
}
'''
    environment = {key: value for key, value in os.environ.items() if not key.startswith("XPLACE_")}
    environment.update(TEST_ROOT=str(tmp_path), XPLACE_SCAN_MANIFEST_FILE="scan-manifest.json",
                       XPLACE_SCAN_MANIFEST_SHA256="a" * 64, XPLACE_SCAN_PROTOCOL_FILE="scan-protocol.json",
                       XPLACE_CAUSAL_MANIFEST_FILE="causal-manifest.json", XPLACE_CAUSAL_PROTOCOL_FILE="causal-protocol.json",
                       XPLACE_CAUSAL_MANIFEST_SHA256="b" * 64)
    if mode == "scan" or mode.startswith("conflict"):
        environment["XPLACE_SCAN_BLOCK"] = "0"
    if mode in ("causal", "conflict_block"):
        environment["XPLACE_CAUSAL_BLOCK"] = "1"
    if mode in ("conflict_mode", "conflict_config", "conflict_imbalance"):
        key = dict(conflict_mode="XPLACE_CAUSAL_MODE", conflict_config="XPLACE_CAUSAL_CONFIG",
                   conflict_imbalance="XPLACE_IMBALANCE_CONFIG")[mode]
        environment[key] = "forbidden"
    result = subprocess.run(["bash", "-c", prefix + branch], env=environment, capture_output=True, text=True)
    if mode.startswith("conflict"):
        assert result.returncode != 0 and not (tmp_path / "command.txt").exists()
    else:
        assert result.returncode == 0, result.stderr
        command = (tmp_path / "command.txt").read_text().splitlines()
        assert command[0] == str(tmp_path / "runtime" / dict(scan="scan_job.py", causal="causal_job.py", direct="main.py")[mode])
        if mode == "scan":
            assert command[1:5] == ["--block", "0", "--protocol", str(tmp_path / "inputs/scan-protocol.json")]
            assert (tmp_path / "manifest-env.txt").read_text().splitlines() == [str(tmp_path / "inputs/scan-manifest.json"), "a" * 64]
