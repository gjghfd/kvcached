import copy
import hashlib
import io
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import upstream_qkv_mirror_campaign as campaign


REAL_DRIVER_ALIVE = campaign.driver_alive
REAL_FROZEN_HOST = campaign.frozen_host
SOURCE = Path(campaign.__file__).parent


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def seal_parent(parent, job=101):
    raw = parent / f"live-trace-{job}"
    (parent / f"raw-{job}.sha256").write_text("".join(
        f"{checksum}  ./{name}\n" for name, checksum in campaign.hashes(raw).items()))


def source_records(raw, service, rank):
    directory = raw / f"installed-upstream-sources/{service}-dp{rank}"
    directory.mkdir(parents=True)
    content = "def apply(self, layer, x, bias=None):\n    return F.linear(x, layer.weight, bias)\n"
    path = directory / "linear.py"
    path.write_text(content)
    checksum = campaign.digest(path)
    qualified = {name: "fake." + name for name in campaign.upstream.INSTALLED_SYMBOLS}
    qualified.update({"linear_class": "fake.QKVParallelLinear", "linear_base": "fake.LinearBase",
                      "column_parallel_linear": "fake.ColumnParallelLinear",
                      "quant_method_class": "fake.UnquantizedLinearMethod", "quant_method_base": "fake.LinearMethodBase",
                      "quant_method_apply": "fake.UnquantizedLinearMethod.apply", "cudagraph_wrapper": "fake.CUDAGraphWrapper",
                      "cudagraph_call": "fake.CUDAGraphWrapper.__call__", "runner_model_class": "fake.CUDAGraphWrapper",
                      "model_class": "fake.Model"})
    return {name: {"qualified_name": qualified[name], "installed_path": str(SOURCE / "upstream_first_layer_campaign.py"),
                   "copied_path": str(path.relative_to(raw)), "file_sha256": checksum,
                   "symbol_source_sha256": checksum, "source_first_line": 1, "source_line_count": 2}
            for name in campaign.upstream.INSTALLED_SYMBOLS}


def full_matrices(raw):
    entries = []
    prefix = [7] * 202
    for padded in (1, 8):
        name = f"upstream-first-layer-snapshots/p53-b{padded}.json"
        witness = {"generation_before": 4, "generation_after": 5}
        snapshot = {"forward_mode": "FULL", "prompt_tokens": 149, "position": 53, "refresh_witness": witness,
                    "num_tokens_padded": padded, "token_row_index": padded - 1, "context_token_ids": prefix,
                    "projection": {"weight_sha256": "f" * 64}, "buffers": {}}
        matrices = {}
        for buffer_name in ("qkv_input", "qkv_output"):
            content = b"\x80\x3f\x00\x40" * padded
            path = raw / f"full-matrices/{buffer_name}-b{padded}.bf16"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(content)
            checksum = campaign.digest(path)
            snapshot["buffers"][buffer_name] = {"shape": [padded, 2], "dtype": "torch.bfloat16",
                                               "row_sha256": hashlib.sha256(content[-4:]).hexdigest()}
            matrices[buffer_name] = {"shape": [padded, 2], "dtype": "torch.bfloat16", "encoding": "bf16-little-endian",
                                    "path": str(path.relative_to(raw)), "sha256": checksum}
        write(raw / name, snapshot)
        entries.append({"snapshot_path": name, "snapshot_sha256": campaign.digest(raw / name),
                        "refresh_witness": witness, "matrices": matrices})
    write(raw / "upstream-first-layer-full-matrices.json", {"schema": campaign.MATRIX_SCHEMA, "job": 101,
          "scientific_scope": campaign.SCOPE, "captured_on_compute": True,
          "source_frame_before_parent_consume": True, "records": entries})


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    repo, parent, performance, root = [tmp_path / name for name in ("repo", "first-layer", "performance", "mirror")]
    raw = parent / "live-trace-101"
    original_names = ["experiments/dp_opt/upstream_first_layer_campaign.py", "experiments/dp_opt/upstream_first_layer_probe.py",
                      "experiments/dp_opt/performance_followup.py", "experiments/dp_opt/trace-benefit.sbatch", campaign.upstream.COPY_ONLY]
    names = original_names + [f"fake-frozen/source-{index}.py" for index in range(228 - len(original_names))]
    report = {"status": "completed_first_layer_observation_not_benefit", "all_in_scope_target_attributions_available": True,
              "four_worker_QKV_weights_verified": True, "scientific_scope": campaign.SCOPE, "numerical_root_cause_proven": False}
    for name in names + list(campaign.NEW_FILES + campaign.MIRROR_FILES):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen fake " + name)
    for name in ("upstream_first_layer_campaign.py", "upstream_qkv_mirror_campaign.py", "trace-benefit.sbatch"):
        (repo / "experiments/dp_opt" / name).write_bytes((SOURCE / name).read_bytes())
    analyzer = "import argparse,json\nparser=argparse.ArgumentParser()\nparser.add_argument('--verify')\nparser.add_argument('--output')\nargs=parser.parse_args()\nwith open(args.output,'x') as output: json.dump(" + repr(report) + ",output)\n"
    (repo / "experiments/dp_opt/upstream_first_layer_probe.py").write_text(analyzer)
    for name in names:
        path = raw / "runtime" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((repo / name).read_bytes())
    runtime = {name: campaign.digest(repo / name) for name in names}
    parent_state = {"job": 101, "state_last_observed": "COMPLETED", "GPU_diagnostic_complete": True,
                    "controls": campaign.upstream.CONTROLS, "runtime_sha256": runtime}
    write(parent / "campaign-state.json", parent_state)
    write(raw / "runtime-manifest.json", runtime)
    write(raw / "frozen/trace.json", {"original_full_trace": True, "P149": 149, "P74": 74})
    write(raw / "frozen/inputs/prompt-bank.json", {"original_not_resampled": True})
    write(raw / "placement-fit.json", {"original": "fit"})
    write(raw / "compute-upstream-first-layer-verification.json", report)
    write(parent / "independent-upstream-first-layer-verification.json", report)
    (raw / "exit-status").write_text("0\n")
    (parent / "accounting-101.txt").write_text("101|COMPLETED|0:0|start|end|elapsed|node|gres/gpu=2|\n")
    (raw / "gpu-memory-before-load.csv").write_text("GPU-fake0, A800, 81920, 80000\nGPU-fake1, A800, 81920, 80000\n")
    for suffix in ("out", "err"):
        (parent / f"live-trace-101.{suffix}").write_text("synthetic GPU fixture only " + suffix)
    write(parent / "archive-101.json", {"job": 101, "state": "COMPLETED", "missing_job_logs": [],
          "log_sha256": {suffix: campaign.digest(parent / f"live-trace-101.{suffix}") for suffix in ("out", "err")}})
    for service in ("instance-1", "instance-2"):
        for rank in (0, 1):
            sources = source_records(raw, service, rank)
            worker = {"service": service, "dp_rank": rank, "checkpoint": "fake/checkpoint", "model_class": "fake.Model",
                      "model_dtype": "torch.bfloat16", "projection": {"weight_sha256": "f" * 64, "weight_shape": [2, 2]},
                      "attention_tp": 1, "dp": 2, "ep": 2, "pp": 1, "cudagraph_mode": "FULL_DECODE_ONLY",
                      "source_sha256": {"qkv": sources["linear_class"]["symbol_source_sha256"],
                                        "runner": sources["model_runner"]["symbol_source_sha256"]}}
            path = raw / f"upstream-first-layer-worker-{service}-dp{rank}.json"
            write(path, worker)
            record = {**worker, "schema": campaign.upstream.INSTALLED_SCHEMA, "slurm_job_id": 101,
                      "compute_hostname": "synthetic-compute", "runner_model_is_cudagraph_wrapper": True,
                      "worker_metadata_sha256": campaign.digest(path), "sources": sources,
                      "linear_mro": ["fake.QKVParallelLinear", "fake.ColumnParallelLinear", "fake.LinearBase"],
                      "quant_method_mro": ["fake.UnquantizedLinearMethod", "fake.LinearMethodBase"],
                      "runner_model_class": "fake.CUDAGraphWrapper", "specific_kernel_proven": False,
                      "quality_equivalence_proven": False, "is_algorithm_benefit": False}
            write(raw / f"installed-upstream-linear-graph-{service}-dp{rank}.json", record)
    full_matrices(raw)
    seal_parent(parent)
    registry = performance / "authorized-debug/00000000.json"
    write(registry, {"campaign_root": str(performance), "sequence": 0, "previous_sha256": None,
          "identities": [{"job": 101, "name": campaign.upstream.JOB_NAME}]})
    write(performance / "campaign-state.json", {"repo": str(repo), "performance_independent_of_debug": True,
          "controls": {"monitor_interval_s": 300},
          "frozen_runtime_sha256": {name: runtime[name] for name in names[:216]},
          "debug_registry": {"path": str(registry.parent), "initial_sha256": campaign.digest(registry)},
          "stages": {"native": {"job": 202, "phase": "monitoring", "stage_id": "native"}}})
    identity = {"pid": os.getpid(), "argv": [sys.executable, "fake-driver", "--root", str(performance)],
                "actual_process_start": "Mon Nov 2 07:16:00 2026", "monitor_interval_s": 300}
    write(performance / "local-monitor-identity.json", identity)
    def alive(performance_root, expected=None):
        actual = campaign.read(Path(performance_root) / "local-monitor-identity.json")
        if expected is not None and actual != expected:
            raise ValueError("identity changed")
        return actual
    monkeypatch.setattr(campaign, "driver_alive", alive)
    monkeypatch.setattr(campaign, "frozen_host", lambda *args: None)
    return SimpleNamespace(repo=repo, parent=parent, performance=performance, root=root, raw=raw)


@pytest.fixture
def prepared(inputs):
    inputs.state = campaign.prepare(inputs.root, inputs.parent, inputs.performance, inputs.repo)
    return inputs


def fresh_poll(inputs, age=2, ssh_exit=0):
    state_path = inputs.performance / "campaign-state.json"
    performance = campaign.read(state_path)
    epoch = time.time() - age
    stamp = datetime.fromtimestamp(epoch).astimezone().isoformat()
    registry = inputs.performance / "authorized-debug/00000000.json"
    path = inputs.performance / "stages/native/poll-202.log"
    write(path, {"at": stamp, "ssh_exit": ssh_exit, "debug_authorization_SHA": {"sequence": 0,
          "sha256": campaign.digest(registry)}, "monotonic_gap_s": 300})
    with path.open("a") as output:
        output.write("\n202|RUNNING|0:0|start|Unknown|00:05:00|fake-node\n")
    os.utime(path, (epoch + 1, epoch + 1))
    performance["stages"]["native"].update({"last_poll_at": stamp, "last_poll_ssh_exit": ssh_exit})
    write(state_path, performance)
    return path


def staged(inputs):
    state = inputs.state
    state.update({"phase": "submission_intent_do_not_duplicate", "remote_stage": str(inputs.root / "fake-remote"),
                  "submission_ledger": str(inputs.root / "remote-ledger"), "job": 303,
                  "submit_register_deadline_epoch": time.time() + 60})
    (inputs.root / "submission-receipt.txt").write_text("303\n")
    (inputs.root / "remote-ledger").write_text("303;cluster\n")
    campaign.save(inputs.root / "campaign-state.json", state)
    return state


def test_parent_requires_real_gpu_complete_full_matrices(inputs):
    state, raw, gate = campaign.parent_evidence(inputs.parent)
    assert state["job"] == 101 and raw == inputs.raw
    assert gate["matrices"]["common_prefix_B1_padded_pairs"] == 1
    assert gate["installed"]["workers_verified"] == 4
    assert all(gate[name] is False for name in campaign.CLAIMS)


@pytest.mark.parametrize("mutation", ["row_only", "missing_record", "stale", "input_sha", "index", "dtype", "short", "escape", "no_b1", "prefix"])
def test_parent_full_matrix_negatives(inputs, mutation):
    path = inputs.raw / "upstream-first-layer-full-matrices.json"
    index = campaign.read(path)
    if mutation == "row_only":
        path.unlink()
    elif mutation == "missing_record":
        index["records"].pop()
        write(path, index)
    else:
        entry = index["records"][0]
        snapshot_path = inputs.raw / entry["snapshot_path"]
        snapshot = campaign.read(snapshot_path)
        matrix = entry["matrices"]["qkv_input"]
        if mutation == "stale":
            entry["refresh_witness"]["generation_after"] = 4
        elif mutation == "input_sha":
            snapshot["buffers"]["qkv_input"]["row_sha256"] = "0" * 64
        elif mutation == "index":
            snapshot["token_row_index"] = 1
        elif mutation == "dtype":
            matrix["dtype"] = "torch.float32"
        elif mutation == "short":
            (inputs.raw / matrix["path"]).write_bytes(b"\x00\x00")
        elif mutation == "escape":
            matrix["path"] = "../outside"
        elif mutation == "no_b1":
            snapshot["num_tokens_padded"] = 8
        elif mutation == "prefix":
            snapshot["context_token_ids"][0] = 8
        write(snapshot_path, snapshot)
        entry["snapshot_sha256"] = campaign.digest(snapshot_path)
        write(path, index)
    with pytest.raises((ValueError, OSError)):
        campaign.full_parent_matrices(inputs.raw, 101)


@pytest.mark.parametrize("mutation", ["failed", "startup", "accounting", "log", "runtime", "report", "gpu", "reconstruction"])
def test_parent_gpu_gate_negatives(inputs, mutation):
    state = campaign.read(inputs.parent / "campaign-state.json")
    if mutation == "failed":
        state["state_last_observed"] = "FAILED"
    elif mutation == "startup":
        state["GPU_diagnostic_complete"] = False
    elif mutation == "accounting":
        (inputs.parent / "accounting-101.txt").write_text("101|COMPLETED|0:0|gres/gpu=1\n")
    elif mutation == "log":
        (inputs.parent / "live-trace-101.err").write_text("changed")
    elif mutation == "runtime":
        state["runtime_sha256"].pop(next(iter(state["runtime_sha256"])))
    elif mutation == "report":
        report = campaign.read(inputs.parent / "independent-upstream-first-layer-verification.json")
        report["all_in_scope_target_attributions_available"] = False
        write(inputs.parent / "independent-upstream-first-layer-verification.json", report)
        write(inputs.raw / "compute-upstream-first-layer-verification.json", report)
    elif mutation == "gpu":
        (inputs.raw / "gpu-memory-before-load.csv").write_text("GPU-fake0,A800,80000,70000\nGPU-fake0,A800,80000,70000\n")
    elif mutation == "reconstruction":
        analyzer = inputs.raw / "runtime/experiments/dp_opt/upstream_first_layer_probe.py"
        analyzer.write_text(analyzer.read_text().replace("completed_first_layer_observation_not_benefit", "fake_success"))
        checksum = campaign.digest(analyzer)
        state["runtime_sha256"]["experiments/dp_opt/upstream_first_layer_probe.py"] = checksum
        write(inputs.raw / "runtime-manifest.json", state["runtime_sha256"])
    write(inputs.parent / "campaign-state.json", state)
    seal_parent(inputs.parent)
    with pytest.raises(ValueError):
        campaign.parent_evidence(inputs.parent)


def test_accounting_without_trailing_pipe_is_supported(inputs):
    path = inputs.parent / "accounting-101.txt"
    path.write_text(path.read_text().rstrip("\n").rstrip("|") + "\n")
    assert campaign.parent_evidence(inputs.parent)[2]["parent_job"] == 101


@pytest.mark.parametrize("text", [
    "def apply(self, layer, x, bias=None):\n    return self.linear(layer, x, bias)\n",
    "def apply(self, layer, x, bias=None):\n    return F.linear(x, layer.weight_scale, bias)\n",
    "def apply(self, layer, x, bias=None):\n    layer.weight = x\n    return x\n",
    "def apply(self, layer, x, bias=None, **kwargs):\n    return F.linear(x, layer.weight, bias)\n",
    "def apply(self, layer, x, bias=None, *, extra=None):\n    return F.linear(x, layer.weight, bias)\n",
    "def apply(self, layer, x, ignored=None):\n    return F.linear(x, layer.weight)\n"])
def test_installed_apply_proxy_unknown_abi_fails_closed(text):
    with pytest.raises(ValueError):
        campaign.review_apply_abi(text)


@pytest.mark.parametrize("mutation", ["symbol", "mro", "apply_class", "job", "dtype", "tp", "weight", "per_worker", "missing"])
def test_installed_parent_negatives(inputs, mutation):
    path = inputs.raw / "installed-upstream-linear-graph-instance-1-dp0.json"
    record = campaign.read(path)
    if mutation == "missing":
        path.unlink()
    else:
        if mutation == "symbol":
            record["sources"]["model_runner"]["symbol_source_sha256"] = "1" * 64
        elif mutation == "mro":
            record["linear_mro"][0] = "fake.WrongLinear"
        elif mutation == "apply_class":
            record["sources"]["quant_method_apply"]["qualified_name"] = "fake.Unknown.apply"
        elif mutation == "job":
            record["slurm_job_id"] = 102
        elif mutation == "dtype":
            record["model_dtype"] = "torch.float32"
        elif mutation == "tp":
            record["attention_tp"] = 2
        elif mutation == "weight":
            record["projection"]["weight_sha256"] = "1" * 64
        elif mutation == "per_worker":
            record["sources"]["linear_class"]["copied_path"] = record["sources"]["linear_class"]["copied_path"].replace("instance-1-dp0", "instance-1-dp1")
        write(path, record)
    with pytest.raises(ValueError):
        campaign.installed_parent(inputs.raw, 101)


def test_symbol_sha_is_not_file_sha(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("import math\ndef apply(self, layer, x, bias=None):\n    return F.linear(x, layer.weight, bias)\n")
    text = "".join(path.read_text().splitlines(keepends=True)[1:])
    source = {"copied_path": "source.py", "file_sha256": campaign.digest(path), "source_first_line": 2,
              "source_line_count": 2, "symbol_source_sha256": hashlib.sha256(text.encode()).hexdigest()}
    assert source["file_sha256"] != source["symbol_source_sha256"]
    assert campaign.source_symbol(tmp_path, source) == text


def test_prepare_only_is_readonly_originals_and_no_ssh(prepared, monkeypatch):
    before = {name: campaign.hashes(getattr(prepared, name)) for name in ("parent", "performance", "repo")}
    monkeypatch.setattr(campaign, "ssh", lambda *args, **kwargs: pytest.fail("prepare-only SSH"))
    campaign.run(prepared.root, prepared.parent, prepared.performance, prepared.repo, prepare_only=True)
    assert {name: campaign.hashes(getattr(prepared, name)) for name in before} == before
    assert len(prepared.state["runtime_sha256"]) == 234
    assert prepared.state["GPU_diagnostic_complete"] is False
    assert campaign.hashes(prepared.root / "upload/frozen") == campaign.hashes(prepared.raw / "frozen")
    assert not os.stat(prepared.root / "upload/runtime" / campaign.NEW_FILES[0]).st_mode & 0o222


def test_prepare_conflicts_record_both_sha_without_fixing_sources(inputs):
    path = inputs.performance / "campaign-state.json"
    state = campaign.read(path)
    name = next(iter(state["frozen_runtime_sha256"]))
    state["frozen_runtime_sha256"][name] = "0" * 64
    write(path, state)
    before = campaign.hashes(inputs.parent), campaign.hashes(inputs.performance), campaign.hashes(inputs.repo)
    with pytest.raises(ValueError, match="intersection conflicts"):
        campaign.prepare(inputs.root, inputs.parent, inputs.performance, inputs.repo)
    report = campaign.read(inputs.root / "source-union-review.json")
    assert report["conflicts"][name]["performance_sha256"] == "0" * 64
    assert report["conflicts"][name]["parent_sha256"] != "0" * 64
    assert before == (campaign.hashes(inputs.parent), campaign.hashes(inputs.performance), campaign.hashes(inputs.repo))


@pytest.mark.parametrize("protected", ["parent", "performance", "repo"])
def test_root_isolation_precedes_any_lock_or_write(inputs, protected):
    root = getattr(inputs, protected)
    before = campaign.hashes(root)
    with pytest.raises(ValueError, match="isolated"):
        campaign.run(root, inputs.parent, inputs.performance, inputs.repo, prepare_only=True)
    assert campaign.hashes(root) == before
    assert not (root / "mirror-campaign.lock").exists()


def test_prepare_refuses_partial_or_duplicate_root(prepared):
    with pytest.raises(ValueError, match="empty"):
        campaign.prepare(prepared.root, prepared.parent, prepared.performance, prepared.repo)


def test_loaded_working_campaign_cannot_network(prepared):
    with pytest.raises(ValueError, match="copied immutable"):
        REAL_FROZEN_HOST(prepared.root, prepared.state)


@pytest.mark.parametrize("mutation", ["upload", "payload", "driver", "allocation", "parent_anchor"])
def test_submit_preflight_refuses_mutation(prepared, mutation):
    if mutation == "upload":
        (prepared.root / "upload/runtime" / campaign.NEW_FILES[0]).chmod(0o644)
        (prepared.root / "upload/runtime" / campaign.NEW_FILES[0]).write_text("changed")
    elif mutation == "payload":
        (prepared.root / "payload.tar.gz").write_bytes(b"changed")
    elif mutation == "driver":
        path = prepared.performance / "local-monitor-identity.json"
        value = campaign.read(path)
        value["pid"] += 1
        write(path, value)
    elif mutation == "allocation":
        path = prepared.performance / "campaign-state.json"
        value = campaign.read(path)
        value["stages"]["native"]["job"] = 204
        write(path, value)
    else:
        (prepared.parent / "live-trace-101.out").write_text("changed")
    with pytest.raises(ValueError):
        campaign.validate_local(prepared.root, prepared.state, submit=True)


@pytest.mark.parametrize("mutation", ["none", "start", "argv", "uid", "dead", "bad_record"])
def test_driver_checks_actual_pid_start_argv_not_launcher(inputs, monkeypatch, mutation):
    path = inputs.performance / "local-monitor-identity.json"
    identity = campaign.read(path)
    identity["launch_argv"] = ["deliberately", "different", "framework", "launcher"]
    write(path, identity)
    values = {"command": shlex.join(identity["argv"]), "lstart": identity["actual_process_start"], "uid": str(os.getuid())}
    if mutation == "bad_record":
        identity["argv"] = ["fake", "--root"]
        write(path, identity)
    if mutation in ("start", "argv", "uid"):
        values[{"start": "lstart", "argv": "command", "uid": "uid"}[mutation]] = "invalid"
    def fake_ps(command, **kwargs):
        return SimpleNamespace(returncode=int(mutation == "dead"), stdout=values[command[-1][:-1]])
    monkeypatch.setattr(campaign.subprocess, "run", fake_ps)
    if mutation == "none":
        assert REAL_DRIVER_ALIVE(inputs.performance) == identity
    else:
        with pytest.raises(ValueError):
            REAL_DRIVER_ALIVE(inputs.performance)


def test_protected_sync_only_new_isolated_copy(prepared, monkeypatch):
    calls = []
    def fake_ssh(root, label, script, *args, **kwargs):
        calls.append((label, script))
        return "HOME_RESULT|/home/fake\n" if label == "before-sync" else ""
    monkeypatch.setattr(campaign, "ssh", fake_ssh)
    campaign.protected_sync(prepared.root, prepared.state)
    text = next(script for label, script in calls if label == "protected-copy-only")
    assert "backup/" in text and "before.sha256" in text and "after.sha256" in text
    assert "MISSING  upload/" in text and ".new; mv" in text
    assert "upload/runtime/" + campaign.upstream.COPY_ONLY in text
    assert "patch -" not in text and "git " not in text
    assert prepared.state["phase"] == "staged"


@pytest.mark.parametrize("age,exit_code", [(35, 0), (2, 1)])
def test_authorization_window_stale_or_failed_poll_refused(prepared, age, exit_code):
    fresh_poll(prepared, age, exit_code)
    with pytest.raises(ValueError):
        campaign.upstream.performance_poll_window(prepared.performance)


def test_authorization_chain_order_and_nonatomic_receipt(prepared):
    staged(prepared)
    text = campaign.authorization_chain(prepared.root, prepared.state)
    assert text.index("--submit-held >") < text.index("--register-debug-job") < text.index("--release-registered-job")
    assert "--cancel-owned-held-job" in text and "set -o noclobber" in text
    assert "performance_followup.py" in text and "--register-debug-name " + campaign.JOB_NAME in text
    assert prepared.state["controls"]["atomic_authorization"] is False
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)


@pytest.mark.parametrize("mutation", ["expired", "changed_poll"])
def test_submit_rechecks_receipt_deadline_without_ssh(prepared, monkeypatch, mutation):
    fresh_poll(prepared)
    state = staged(prepared)
    state["performance_poll_window"] = campaign.upstream.performance_poll_window(prepared.performance)
    if mutation == "expired":
        state["submit_register_deadline_epoch"] = time.time() - 1
    else:
        state["performance_poll_window"]["receipt_sha256"] = "0" * 64
    monkeypatch.setattr(campaign, "ssh", lambda *args, **kwargs: pytest.fail("invalid submit SSH"))
    with pytest.raises(ValueError):
        campaign.submit_held(prepared.root, state)


def test_submit_helper_does_not_rerun_expensive_parent_analyzer(prepared, monkeypatch):
    fresh_poll(prepared)
    state = staged(prepared)
    state["performance_poll_window"] = campaign.upstream.performance_poll_window(prepared.performance)
    monkeypatch.setattr(campaign, "parent_evidence", lambda *args: pytest.fail("600s analyzer inside 60s chain"))
    calls = []
    def fake_ssh(root, label, script, **kwargs):
        calls.append(script)
        return "303;cluster\n"
    monkeypatch.setattr(campaign, "ssh", fake_ssh)
    assert campaign.submit_held(prepared.root, state) == 303
    assert "sbatch --hold --parsable" in calls[0]


def local_slurm(script, state, queue, control=None, account=None):
    command = state["remote_stage"] + "/upload/runtime/experiments/dp_opt/upstream-qkv-mirror.sbatch"
    control = control or f"JobId=303 JobState=PENDING Reason=JobHeldUser Command={command}"
    account = account or f"303|fake|{campaign.JOB_NAME}|gpu_a800|CANCELLED|"
    prefix = "USER=fake\nid() { printf 'fake\\n'; }\n"
    prefix += f"squeue() {{ printf '%s\\n' {shlex.quote(queue)}; }}\n"
    prefix += f"scontrol() {{ printf '%s\\n' {shlex.quote(control)}; }}\n"
    prefix += f"sacct() {{ printf '%s\\n' {shlex.quote(account)}; }}\n"
    prefix += "scancel() { printf 'FAKE_CANCEL|%s\\n' \"$1\"; }\n"
    return subprocess.run(["bash", "-c", prefix + script], text=True, capture_output=True)


@pytest.mark.parametrize("mutation", ["none", "running", "wrong_owner", "wrong_command", "not_held", "already_cancelled", "wrong_cancelled"])
def test_precise_held_cleanup_never_cancels_running_or_perf(prepared, monkeypatch, mutation):
    state = staged(prepared)
    queue = f"303|fake|{campaign.JOB_NAME}|gpu_a800|PENDING"
    control = account = None
    if mutation == "running":
        queue = queue.replace("PENDING", "RUNNING")
    elif mutation == "wrong_owner":
        queue = queue.replace("|fake|", "|foreign|")
    elif mutation == "wrong_command":
        control = "JobId=303 JobState=PENDING Reason=JobHeldUser Command=/other.sbatch"
    elif mutation == "not_held":
        control = "JobId=303 JobState=PENDING Reason=Priority Command=/other.sbatch"
    elif mutation in ("already_cancelled", "wrong_cancelled"):
        queue = ""
        if mutation == "wrong_cancelled":
            account = f"303|foreign|{campaign.JOB_NAME}|gpu_a800|CANCELLED|"
    results = []
    def fake_ssh(root, label, script, **kwargs):
        result = local_slurm(script, state, queue, control, account)
        results.append(result)
        if result.returncode:
            raise RuntimeError("fake rejected exact held identity")
        return result.stdout
    monkeypatch.setattr(campaign, "ssh", fake_ssh)
    if mutation in ("none", "already_cancelled"):
        campaign.cancel_owned_held(prepared.root, state, 303)
        assert ("FAKE_CANCEL|303" in results[0].stdout) is (mutation == "none")
    else:
        with pytest.raises(RuntimeError):
            campaign.cancel_owned_held(prepared.root, state, 303)
        assert "FAKE_CANCEL" not in results[0].stdout
    for forbidden in (101, 202):
        (prepared.root / "submission-receipt.txt").write_text(str(forbidden))
        with pytest.raises(ValueError, match="performance/parent"):
            campaign.receipt(prepared.root, state, forbidden)


def test_unknown_third_job_fails_closed(prepared):
    performance, identities, head = campaign.upstream.performance_state(prepared.performance)
    text = campaign.guard(prepared.state, 0)
    prefix = "USER=fake\nid() { printf 'fake\\n'; }\n"
    prefix += "squeue() { printf '202|fake|qwen-public-perf-followup|gpu_a800|RUNNING\\n999|fake|unknown|gpu_a800|PENDING\\n'; }\n"
    result = subprocess.run(["bash", "-c", prefix + text], text=True, capture_output=True)
    assert result.returncode != 0


@pytest.mark.parametrize("output,valid", [
    ("OWNER|fake\n303|fake|qwen-upstream-qkv-mirror|gpu_a800|RUNNING|0:0|gres/gpu=2|\n", True),
    ("OWNER|fake\n303|fake|qwen-upstream-qkv-mirror|gpu_a800|CANCELLED by 12|0:0|gres/gpu=2|\n", True),
    ("OWNER|fake\n303|fake\n", False),
    ("OWNER|fake\n303|foreign|qwen-upstream-qkv-mirror|gpu_a800|RUNNING\n", False),
    ("OWNER|fake\n303|fake|qwen-public-perf-followup|gpu_a800|RUNNING\n", False),
    ("OWNER|fake\n303|fake|qwen-upstream-qkv-mirror|gpu_a800|UNKNOWN\n", False)])
def test_poll_actual_identity_and_terminal_state(prepared, monkeypatch, output, valid):
    state = staged(prepared)
    monkeypatch.setattr(campaign, "ssh", lambda *args, **kwargs: output)
    if valid:
        assert campaign.poll(prepared.root, state) in ("RUNNING", "CANCELLED")
    else:
        with pytest.raises(ValueError):
            campaign.poll(prepared.root, state)


def test_registration_chain_failure_preserved_no_resubmit_or_perf_cancel(prepared, monkeypatch):
    fresh_poll(prepared)
    state = staged(prepared)
    state["phase"] = "staged"
    calls = []
    monkeypatch.setattr(campaign, "offline_parent", lambda *args: campaign.read(prepared.parent / "independent-upstream-first-layer-verification.json"))
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(
        returncode=1, communicate=lambda **kwargs: ("", "registration rejected")))
    monkeypatch.setattr(campaign, "cancel_owned_held", lambda root, bound, job: calls.append(job))
    with pytest.raises(RuntimeError, match="never cancel performance"):
        campaign.submit_and_register(prepared.root, state)
    assert calls == [303]
    assert campaign.read(prepared.root / "submit-register-chain.json")["exit"] == 1


def test_chain_timeout_only_signals_own_local_group_then_exact_held_job(prepared, monkeypatch):
    fresh_poll(prepared)
    state = staged(prepared)
    signals, cleanups = [], []
    monkeypatch.setattr(campaign, "offline_parent", lambda *args: campaign.read(prepared.parent / "independent-upstream-first-layer-verification.json"))
    class TimedChain:
        pid = 987654321
        returncode = -15
        calls = 0

        def communicate(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("fake-chain", 60)
            return "", "fake timeout preserved"
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *args, **kwargs: TimedChain())
    monkeypatch.setattr(campaign.os, "killpg", lambda pid, signum: signals.append((pid, signum)))
    monkeypatch.setattr(campaign, "cancel_owned_held", lambda root, bound, job: cleanups.append(job))
    with pytest.raises(RuntimeError, match="Bounded DEBUG"):
        campaign.submit_and_register(prepared.root, state)
    assert signals == [(987654321, campaign.signal.SIGTERM)]
    assert cleanups == [303]
    assert campaign.read(prepared.root / "submit-register-chain.json")["timed_out"] is True


def test_window_wait_does_not_ssh_and_is_bounded(prepared, monkeypatch):
    monkeypatch.setattr(campaign, "offline_parent", lambda *args: campaign.read(prepared.parent / "independent-upstream-first-layer-verification.json"))
    monkeypatch.setattr(campaign.upstream, "performance_poll_window", lambda *args: (_ for _ in ()).throw(ValueError("no fresh poll")))
    monkeypatch.setattr(campaign, "ssh", lambda *args, **kwargs: pytest.fail("SSH during local window wait"))
    initial = time.time()
    clock = {"now": initial}
    monkeypatch.setattr(campaign.time, "time", lambda: clock["now"])
    monkeypatch.setattr(campaign.time, "sleep", lambda delay: clock.update(now=clock["now"] + 601))
    with pytest.raises(ValueError, match="600s"):
        campaign.submit_and_register(prepared.root, prepared.state)
    assert list(prepared.root.glob("waiting-perf-poll-*.json"))
    assert not (prepared.root / "submission-receipt.txt").exists()


def test_release_requires_explicit_owner_registration_before_ssh(prepared, monkeypatch):
    state = staged(prepared)
    monkeypatch.setattr(campaign, "ssh", lambda *args, **kwargs: pytest.fail("unregistered release SSH"))
    with pytest.raises(ValueError, match="registration"):
        campaign.release_registered(prepared.root, state, 303)


def test_record_compute_sources_after_keyword_compatible_parent_initialize(monkeypatch):
    calls = []
    class Parent:
        def initialize_from_config(self, kv_cache_config):
            calls.append(("initialize", kv_cache_config))
            return "initialized"
    worker = type("InstalledProvenanceMirrorWorker", (Parent,), {"initialize_from_config": campaign._initialize_installed})
    monkeypatch.setitem(campaign.__dict__, "InstalledProvenanceMirrorWorker", worker)
    monkeypatch.setattr(campaign.upstream, "collect_installed_provenance", lambda subject: calls.append(("collect", subject)))
    instance = worker()
    assert instance.initialize_from_config(kv_cache_config="config") == "initialized"
    assert calls == [("initialize", "config"), ("collect", instance)]


def test_compute_preflight_rejects_install_change_before_launch(prepared, monkeypatch):
    monkeypatch.setattr(campaign, "require_compute", lambda: None)
    monkeypatch.setattr(campaign.subprocess, "run", lambda *args, **kwargs: pytest.fail("model/startup before ABI check"))
    execution = prepared.root / "compute-execution"
    shutil.copytree(prepared.root / "upload/runtime", execution / "runtime")
    with pytest.raises(ValueError, match="Current compute installed"):
        campaign.compute_preflight(prepared.root / "upload", execution)


def test_compute_preflight_accepts_only_same_installed_file_sha(prepared, monkeypatch):
    monkeypatch.setattr(campaign, "require_compute", lambda: None)
    upload = prepared.root / "upload"
    gate_path = upload / "provenance/parent-gate.json"
    gate = campaign.read(gate_path)
    copied = campaign.read(prepared.raw / "installed-upstream-linear-graph-instance-1-dp0.json")["sources"]
    for name, source in gate["installed"]["sources"].items():
        source["installed_path"] = str(prepared.raw / copied[name]["copied_path"])
    gate_path.chmod(0o644)
    write(gate_path, gate)
    plan_path = upload / "plan.json"
    plan = campaign.read(plan_path)
    plan["parent_gate_sha256"] = campaign.digest(gate_path)
    plan_path.chmod(0o644)
    write(plan_path, plan)
    integrity_path = upload / "compute-integrity.json"
    integrity_path.chmod(0o644)
    integrity = campaign.hashes(upload)
    integrity.pop("compute-integrity.json")
    write(integrity_path, integrity)
    execution = prepared.root / "compute-execution"
    shutil.copytree(upload / "runtime", execution / "runtime")
    campaign.compute_preflight(upload, execution)
    (upload / "runtime/extra.py").write_text("unexpected")
    with pytest.raises(ValueError, match="integrity"):
        campaign.compute_preflight(upload, execution)


def test_sbatch_real_stdlib_bootstrap_refuses_corruption_before_any_gpu_import(prepared, tmp_path):
    home = tmp_path / "compute-home"
    common = home / "aegaeon/paracloud/scripts/common.sh"
    common.parent.mkdir(parents=True)
    common.write_text("require_compute() { :; }\nload_toolchains() { ENV_PREFIX=\"$HOME/env\"; DEPLOY_ROOT=\"$HOME/deploy\"; }\n")
    python = home / "run/kvcached-paracloud/env/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    python.chmod(0o755)
    path = prepared.root / "upload/runtime" / campaign.NEW_FILES[0]
    path.chmod(0o644)
    path.write_text("corrupted before runtime imports")
    environment = {**os.environ, "HOME": str(home), "SLURM_JOB_ID": "555", "DP_OPT_QKV_MIRROR_UPLOAD": str(prepared.root / "upload")}
    environment.pop("KVCACHED_DATA_ROOT", None)
    process = subprocess.run(["bash", str(SOURCE / "upstream-qkv-mirror.sbatch")], env=environment,
                             text=True, capture_output=True, timeout=30)
    evidence = home / "run/kvcached-paracloud/state/live-trace-555"
    assert process.returncode != 0 and "immutable SHA/file set differs" in process.stderr
    assert (evidence / "exit-status").read_text().strip() != "0"
    assert not (evidence / "gpu-memory-before-load.csv").exists()
    assert not (evidence / "mirror-startup-tests.log").exists()


def test_sbatch_private_runtime_and_amortized_copy_are_owner_writable_upload_stays_444(prepared, tmp_path):
    upload = prepared.root / "upload"
    before = campaign.hashes(upload)
    source = upload / "runtime" / campaign.NEW_FILES[0]
    assert source.stat().st_mode & 0o777 == 0o444
    evidence = tmp_path / "private-execution"
    evidence.mkdir()
    text = (SOURCE / "upstream-qkv-mirror.sbatch").read_text()
    private_copy = text[text.index('cp -a "$upload/runtime" "$evidence/runtime"'):text.index('cd "$evidence"')]
    subprocess.run(["bash", "-c", private_copy], env={**os.environ, "upload": str(upload), "evidence": str(evidence)},
                   text=True, capture_output=True, check=True)
    runtime = evidence / "runtime"
    assert campaign.hashes(runtime) == campaign.hashes(upload / "runtime")
    assert all(path.stat().st_mode & 0o200 for path in runtime.rglob("*") if path.is_file())
    amortized = evidence / "amortized-runtime"
    amortized.mkdir()
    target = amortized / "source-tamper-fixture.py"
    shutil.copy2(runtime / campaign.NEW_FILES[0], target)
    assert target.stat().st_mode & 0o200
    target.write_text("private source-tamper fixture can write without PermissionError")
    assert campaign.hashes(runtime) == campaign.hashes(upload / "runtime")
    assert campaign.hashes(upload) == before
    assert all(path.stat().st_mode & 0o777 == 0o444 for path in upload.rglob("*") if path.is_file())


def make_archive(tmp_path, names, pointer=None):
    bundle = tmp_path / "bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name in names:
            content = b"kept raw failure"
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        if pointer is not None:
            member = tarfile.TarInfo("progress.json")
            member.type = tarfile.SYMTYPE
            member.linkname = pointer
            archive.addfile(member)
    manifest = hashlib.sha256(b"kept raw failure").hexdigest() + "  ./result.json\n"
    staging = tmp_path / "extracted"
    staging.mkdir()
    return bundle, staging, manifest


@pytest.mark.parametrize("terminal", ["CANCELLED", "FAILED", "TIMEOUT"])
def test_actual_failed_terminal_dangling_progress_preserved_without_fake_marker(tmp_path, terminal):
    bundle, staging, manifest = make_archive(tmp_path, ["result.json"], "/home/fake/run/kvcached-paracloud/state/live-trace-303/r1/progress.json")
    pointers = campaign.extract_terminal(bundle, staging, manifest, 303, terminal)
    assert pointers[0]["actual_terminal"] == terminal and pointers[0]["target_present"] is False
    assert not (staging / "exit-status").exists()
    assert not (staging / "progress.json").exists()


@pytest.mark.parametrize("mutation", ["completed_missing", "active", "duplicate", "traversal", "foreign", "sha"])
def test_terminal_archive_negative_security(tmp_path, mutation):
    names, pointer, terminal = ["result.json"], None, "FAILED"
    if mutation in ("completed_missing", "foreign"):
        pointer = "/home/fake/run/kvcached-paracloud/state/live-trace-303/r1/progress.json"
        if mutation == "completed_missing":
            terminal = "COMPLETED"
        else:
            pointer = pointer.replace("303", "404")
    elif mutation == "active":
        terminal = "RUNNING"
    elif mutation == "duplicate":
        names.append("./result.json")
    elif mutation == "traversal":
        names = ["../outside"]
    bundle, staging, manifest = make_archive(tmp_path, names, pointer)
    if mutation == "sha":
        manifest = "0" * 64 + "  ./result.json\n"
    with pytest.raises(ValueError):
        campaign.extract_terminal(bundle, staging, manifest, 303, terminal)


def test_compute_guard_rejects_host_even_with_flag(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("DP_OPT_QKV_MIRROR_COMPUTE_CONFIRMED", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-p no:cacheprovider")
    with pytest.raises(RuntimeError):
        campaign.require_compute()


def test_mirror_worker_is_lazy_host_import_fails_before_gpu_module(monkeypatch):
    monkeypatch.setattr(campaign, "require_compute", lambda: (_ for _ in ()).throw(RuntimeError("host guard")))
    with pytest.raises(RuntimeError, match="host guard"):
        getattr(campaign, "InstalledProvenanceMirrorWorker")


def test_compute_environment_preserves_ids32_compatible_original_trace(prepared, monkeypatch):
    monkeypatch.setenv("DP_OPT_HIDDEN_TARGETS", "wrong")
    monkeypatch.setenv("DP_OPT_QKV_MIRROR_REPETITIONS", "2")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    runtime = prepared.root / "upload/runtime"
    environment = campaign.compute_environment(runtime, prepared.root, prepared.root / "upload")
    assert "DP_OPT_HIDDEN_TARGETS" not in environment and "VLLM_BATCH_INVARIANT" not in environment
    assert environment["DP_OPT_QKV_MIRROR_REPETITIONS"] == "3"
    assert environment["DP_OPT_TRACE_REPEATS"] == "6" and environment["DP_OPT_TRACE_POLICIES"] == "no_move"
    assert environment["PYTEST_ADDOPTS"] == "-p no:cacheprovider"
    assert environment["DP_OPT_TRACE_PATH"] == str(prepared.root / "frozen/trace.json")
    assert environment["DP_OPT_UPSTREAM_PLACEMENT_FIT"] == str(prepared.root / "upload/placement-fit.json")
    assert "INPUT_IDS_DTYPE" not in environment


def test_frozen_template_replaced_only_in_new_launch():
    template = (SOURCE / "trace-benefit.sbatch").read_text()
    launch = campaign.make_launch(template)
    assert '"$evidence/upstream_qkv_mirror_entry.py"' in launch
    assert 'mkdir -p "$evidence"\ncd "$evidence"' in launch
    assert '--fit "$DP_OPT_UPSTREAM_PLACEMENT_FIT"' in launch
    assert '"$DP_OPT_UPSTREAM_PLACEMENT_FIT" "${DP_OPT_TRACE_PATH:?Set DP_OPT_TRACE_PATH}"' in launch
    assert '"$evidence/placement-fit.json"' not in launch
    assert "--repeats" in launch and "-p no:cacheprovider" in launch
    subprocess.run(["bash", "-n"], input=launch, text=True, check=True)
    with pytest.raises(ValueError):
        campaign.make_launch(template.replace("live_trace.py", "unsupported.py"))


def test_legacy_parent_helper_fit_is_normalized_without_editing_parent(monkeypatch):
    template = (SOURCE / "trace-benefit.sbatch").read_text()
    current = campaign.upstream.make_launch(template)
    legacy = current.replace('"$DP_OPT_UPSTREAM_PLACEMENT_FIT"', '"$evidence/placement-fit.json"')
    monkeypatch.setattr(campaign.upstream, "make_launch", lambda supplied: legacy)
    launch = campaign.make_launch(template)
    assert launch.count('"$DP_OPT_UPSTREAM_PLACEMENT_FIT"') == 2
    assert '"$evidence/placement-fit.json"' not in launch
    monkeypatch.setattr(campaign.upstream, "make_launch", lambda supplied: legacy.replace('--fit "$evidence/placement-fit.json"', '--fit "/unsupported/fit.json"'))
    with pytest.raises(ValueError, match="fit CLI/SHA"):
        campaign.make_launch(template)


@pytest.mark.parametrize("alias", ["same_path", "hardlink", "symlink"])
def test_placement_fit_alias_fails_closed_not_ignored(prepared, tmp_path, alias):
    upload = prepared.root / "upload"
    source = upload / "placement-fit.json"
    evidence = upload if alias == "same_path" else tmp_path / "execution"
    if alias != "same_path":
        evidence.mkdir()
        target = evidence / "placement-fit.json"
        if alias == "hardlink":
            os.link(source, target)
        else:
            target.symlink_to(source)
    with pytest.raises(ValueError):
        campaign.placement_fit_input(upload, evidence)
    with pytest.raises(shutil.SameFileError):
        shutil.copy2(source, evidence / "placement-fit.json")


def test_compute_live_trace_fit_copy_is_distinct_and_output_keeps_input_sha(prepared, tmp_path, monkeypatch):
    upload = prepared.root / "upload"
    evidence = tmp_path / "execution"
    shutil.copytree(upload / "runtime", evidence / "runtime")
    source = upload / "placement-fit.json"
    source_sha = campaign.digest(source)
    calls = []
    monkeypatch.setenv("SLURM_JOB_ID", "303")
    monkeypatch.setattr(campaign, "compute_preflight", lambda *args: None)
    monkeypatch.setattr(campaign, "installed_parent", lambda *args: prepared.state["parent_gate"]["installed"])
    monkeypatch.setattr(campaign, "compute_main", lambda: pytest.fail("compute-generated entry reentered probe during spawn"))
    def actual_copy(command, env, cwd):
        runpy.run_path(str(Path(cwd) / "upstream_qkv_mirror_entry.py"), run_name="__mp_main__")
        actual_input = Path(env["DP_OPT_UPSTREAM_PLACEMENT_FIT"])
        actual_output = Path(cwd) / "placement-fit.json"
        assert actual_input == source and not actual_output.exists()
        assert actual_input.resolve() != actual_output.resolve()
        shutil.copy2(actual_input, actual_output)
        calls.append((actual_input, actual_output))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(campaign.subprocess, "run", actual_copy)
    campaign.compute(upload, evidence)
    assert len(calls) == 1
    assert campaign.digest(evidence / "placement-fit.json") == campaign.digest(source) == source_sha
    assert source.stat().st_mode & 0o777 == 0o444


def test_generated_entry_runpy_spawn_reentry_is_noop_before_and_after_main(tmp_path, monkeypatch):
    entry = campaign.write_compute_entry(tmp_path)
    artifact = tmp_path / "head-hidden-exclusive-output.bin"
    calls = []
    def exclusive_probe():
        calls.append("main")
        with artifact.open("xb") as output:
            output.write(b"one real main execution")
    monkeypatch.setattr(campaign, "compute_main", exclusive_probe)
    before = campaign.hashes(tmp_path)
    for run_name in ("__mp_main__", "mirror_import", "__mp_main__"):
        runpy.run_path(str(entry), run_name=run_name)
    assert calls == [] and campaign.hashes(tmp_path) == before
    runpy.run_path(str(entry), run_name="__main__")
    assert calls == ["main"] and artifact.read_bytes() == b"one real main execution"
    after_main = campaign.hashes(tmp_path)
    for run_name in ("__mp_main__", "__mp_main__", "mirror_import"):
        runpy.run_path(str(entry), run_name=run_name)
    assert calls == ["main"] and campaign.hashes(tmp_path) == after_main


def test_generated_entry_true_duplicate_main_keeps_exclusive_failure(tmp_path, monkeypatch):
    entry = campaign.write_compute_entry(tmp_path)
    artifact = tmp_path / "head-hidden-exclusive-output.bin"
    def exclusive_probe():
        with artifact.open("xb") as output:
            output.write(b"original evidence")
    monkeypatch.setattr(campaign, "compute_main", exclusive_probe)
    runpy.run_path(str(entry), run_name="__main__")
    before = campaign.hashes(tmp_path)
    with pytest.raises(FileExistsError):
        runpy.run_path(str(entry), run_name="__main__")
    assert campaign.hashes(tmp_path) == before


def test_compute_entry_creation_is_exclusive_and_preserves_existing_evidence(tmp_path):
    entry = campaign.write_compute_entry(tmp_path)
    before = campaign.digest(entry)
    with pytest.raises(FileExistsError):
        campaign.write_compute_entry(tmp_path)
    assert campaign.digest(entry) == before


def test_parent_manifest_count_is_actual_not_228(inputs):
    state_path = inputs.parent / "campaign-state.json"
    state = campaign.read(state_path)
    name = "new-parent-runtime/source.py"
    path = inputs.raw / "runtime" / name
    path.parent.mkdir(parents=True)
    path.write_text("new actual source epoch")
    state["runtime_sha256"][name] = campaign.digest(path)
    write(state_path, state)
    write(inputs.raw / "runtime-manifest.json", state["runtime_sha256"])
    seal_parent(inputs.parent)
    assert len(campaign.parent_evidence(inputs.parent)[0]["runtime_sha256"]) == 229


def test_prepare_binds_actual_performance_manifest_not_old_r5_count(inputs):
    path = inputs.repo / "actual-performance-extra.py"
    path.write_text("caller supplied current performance manifest")
    state_path = inputs.performance / "campaign-state.json"
    state = campaign.read(state_path)
    state["frozen_runtime_sha256"]["actual-performance-extra.py"] = campaign.digest(path)
    write(state_path, state)
    prepared = campaign.prepare(inputs.root, inputs.parent, inputs.performance, inputs.repo)
    assert prepared["performance_runtime_sha256"] == state["frozen_runtime_sha256"]
    report = campaign.read(inputs.root / "source-union-review.json")
    assert report["performance_source_count"] == 217 and report["parent_source_count"] == 228


def test_sbatch_compute_validation_order_and_no_host_operations():
    text = (SOURCE / "upstream-qkv-mirror.sbatch").read_text()
    assert "#SBATCH --time=03:00:00" in text and "#SBATCH --gpus=2" in text
    assert text.index("trap finish EXIT") < text.index("load_toolchains")
    assert text.index('cd "$evidence"') < text.index("python - ")
    assert text.index('cp -a "$upload/runtime" "$evidence/runtime"') < text.index('find "$evidence/runtime" -type f -exec chmod u+w {} +') < text.index("python - ")
    assert 'find "$upload/runtime"' not in text
    assert text.index("compute-integrity.json") < text.index("--compute-preflight") < text.index("python -m pytest")
    assert "PYTEST_ADDOPTS='-p no:cacheprovider'" in text
    assert "ssh " not in text and "sbatch " not in text and "scancel " not in text
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)


@pytest.mark.parametrize("arguments", [
    ["--compute-upload", "/fake", "--compute-evidence", "/fake", "--once"],
    ["--validate-parent", "--parent", "/fake", "--once"],
    ["--root", "/fake", "--parent", "/fake", "--concurrent-performance-root", "/other", "--prepare-only", "--once"],
    ["--compute-preflight"], ["--root", "/fake", "--submit-held", "--repo", "/fake"]])
def test_cli_conflicting_modes_rejected_without_effects(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", [campaign.__file__, *arguments])
    monkeypatch.setattr(campaign, "ssh", lambda *args, **kwargs: pytest.fail("invalid CLI SSH"))
    with pytest.raises(SystemExit) as error:
        campaign.main()
    assert error.value.code == 2


@pytest.fixture
def kernel_prepared(inputs):
    inputs.state = campaign.prepare(inputs.root, inputs.parent, inputs.performance, inputs.repo, kernel_activity=True)
    return inputs


def kernel_worker_evidence(prepared, root, enabled=True, profiler_status="unsupported"):
    upload = prepared.root / "upload"
    shutil.copytree(upload / "runtime", root / "runtime")
    shutil.copy2(upload / "runtime-manifest.json", root / "runtime-manifest.json")
    plan = campaign.read(upload / "plan.json")
    for service in ("instance-1", "instance-2"):
        for rank in (0, 1):
            write(root / f"upstream-qkv-mirror-worker-{service}-dp{rank}.json", {
                "service": service, "dp_rank": rank, "repetitions": 3,
                "kernel_activity_policy": campaign.mirror_probe.kernel_activity_policy(enabled),
                "kernel_profiler_sources": {"status": profiler_status, "sources": {}, "reason": "fake CUPTI unavailable"},
                "source_sha256": {symbol: plan["kernel_activity_source_sha256"][name]
                                  for symbol, name in zip(("mirror_worker", "mirror_probe"), campaign.MIRROR_FILES[:2])}})
    return plan


def test_kernel_controls_default_disabled_and_match_reviewed_worker_policy():
    controls = campaign.CONTROLS
    assert controls == campaign.campaign_controls(False)
    assert controls["kernel_activity"] == campaign.mirror_probe.kernel_activity_policy(False)
    assert controls["kernel_activity"]["maximum_shadow_calls_per_worker"] == 6
    assert controls["kernel_activity"]["maximum_observations_per_worker"] == 1
    assert controls["kernel_activity"]["maximum_trace_bytes_per_call"] == 4 * 1024 * 1024
    assert controls["kernel_activity"]["maximum_trace_events_per_call"] == 4096
    assert controls["kernel_trace_directory"] == "upstream-qkv-mirror-kernel-traces"
    assert controls["kernel_retention_budget"] == {"workers": 4, "maximum_trace_files": 24,
                                                    "maximum_trace_bytes": 96 * 1024 * 1024}
    assert controls["kernel_activity"]["profiler_buffer_and_export_peak_not_hard_bounded"] is True
    assert controls["kernel_shadow_timings_are_native_performance"] is False


@pytest.mark.parametrize("value", [0, 1, "0", "1", None, "yes"])
def test_kernel_prepare_requires_boolean_opt_in_before_writes(inputs, value):
    with pytest.raises(ValueError, match="explicit boolean"):
        campaign.prepare(inputs.root, inputs.parent, inputs.performance, inputs.repo, kernel_activity=value)
    assert not inputs.root.exists()


@pytest.mark.parametrize("field,value", [
    ("maximum_shadow_calls_per_worker", 7), ("maximum_observations_per_worker", 2),
    ("maximum_trace_bytes_per_call", 8 * 1024 * 1024), ("maximum_trace_events_per_call", 8192),
    ("native_cuda_graph_profiled", True), ("selection_rule", "largest_difference"),
    ("unsupported_is_nonfatal", False), ("profiler_buffer_and_export_peak_not_hard_bounded", False)])
def test_kernel_policy_cannot_silently_widen_reviewed_bounds(monkeypatch, field, value):
    original = campaign.mirror_probe.kernel_activity_policy
    monkeypatch.setattr(campaign.mirror_probe, "kernel_activity_policy", lambda enabled: {**original(enabled), field: value})
    with pytest.raises(ValueError, match="reviewed shadow"):
        campaign.campaign_controls(True)


def test_kernel_prepare_binds_explicit_policy_sources_and_preserves_originals(kernel_prepared, monkeypatch):
    monkeypatch.setattr(campaign, "ssh", lambda *args, **kwargs: pytest.fail("local opt-in prepare SSH"))
    upload = kernel_prepared.root / "upload"
    plan = campaign.read(upload / "plan.json")
    state = kernel_prepared.state
    assert plan["controls"] == state["controls"] == campaign.campaign_controls(True)
    assert plan["kernel_activity_policy_sha256"] == campaign.mirror_probe.canonical_sha(plan["controls"]["kernel_activity"])
    assert plan["kernel_activity_source_sha256"] == {name: state["runtime_sha256"][name] for name in campaign.MIRROR_FILES[:2]}
    for name, checksum in plan["kernel_activity_source_sha256"].items():
        assert campaign.digest(upload / "runtime" / name) == campaign.digest(kernel_prepared.repo / name) == checksum
        assert (upload / "runtime" / name).stat().st_mode & 0o777 == 0o444
    assert campaign.read(upload / "frozen/trace.json") == campaign.read(kernel_prepared.raw / "frozen/trace.json")
    assert campaign.read(kernel_prepared.parent / "campaign-state.json")["controls"] == campaign.upstream.CONTROLS
    campaign.validate_local(kernel_prepared.root, state)


@pytest.mark.parametrize("enabled", [False, True])
def test_kernel_environment_uses_frozen_declaration_not_external_env(inputs, monkeypatch, enabled):
    campaign.prepare(inputs.root, inputs.parent, inputs.performance, inputs.repo, kernel_activity=enabled)
    for name, value in {"DP_OPT_QKV_MIRROR_KERNEL_ACTIVITY": "0" if enabled else "1",
                        "DP_OPT_QKV_MIRROR_KERNEL_DIRECTORY": "/outside",
                        "DP_OPT_QKV_MIRROR_KERNEL_MAX_CALLS": "999",
                        "DP_OPT_QKV_MIRROR_KERNEL_MAX_TRACE_BYTES": "999999999",
                        "DP_OPT_PHASE_EVIDENCE_ROOT": "/outside"}.items():
        monkeypatch.setenv(name, value)
    upload = inputs.root / "upload"
    environment = campaign.compute_environment(upload / "runtime", inputs.root, upload)
    assert environment["DP_OPT_QKV_MIRROR_KERNEL_ACTIVITY"] == ("1" if enabled else "0")
    assert environment["DP_OPT_PHASE_EVIDENCE_ROOT"] == str(inputs.root)
    assert environment["DP_OPT_QKV_MIRROR_REPETITIONS"] == "3"
    assert not any(name in environment for name in ("DP_OPT_QKV_MIRROR_KERNEL_DIRECTORY",
                   "DP_OPT_QKV_MIRROR_KERNEL_MAX_CALLS", "DP_OPT_QKV_MIRROR_KERNEL_MAX_TRACE_BYTES"))


@pytest.mark.parametrize("mutation", ["missing_policy", "flag_string", "calls", "directory", "retention", "source", "policy_sha", "missing_source"])
def test_kernel_plan_tampering_rejected_before_launch(kernel_prepared, monkeypatch, mutation):
    upload = kernel_prepared.root / "upload"
    plan_path = upload / "plan.json"
    plan = campaign.read(plan_path)
    if mutation == "missing_policy":
        del plan["controls"]["kernel_activity"]
    elif mutation == "flag_string":
        plan["controls"]["kernel_activity"]["enabled"] = "1"
    elif mutation == "calls":
        plan["controls"]["kernel_activity"]["maximum_shadow_calls_per_worker"] = 7
    elif mutation == "directory":
        plan["controls"]["kernel_trace_directory"] = "../elsewhere"
    elif mutation == "retention":
        plan["controls"]["kernel_retention_budget"]["maximum_trace_bytes"] *= 2
    elif mutation == "source":
        plan["kernel_activity_source_sha256"][campaign.MIRROR_FILES[0]] = "a" * 64
    elif mutation == "policy_sha":
        plan["kernel_activity_policy_sha256"] = "b" * 64
    else:
        del plan["kernel_activity_source_sha256"]
    plan_path.chmod(0o644)
    write(plan_path, plan)
    monkeypatch.setattr(campaign, "require_compute", lambda: None)
    monkeypatch.setattr(campaign.subprocess, "run", lambda *args, **kwargs: pytest.fail("tampered opt-in launch"))
    with pytest.raises(ValueError):
        campaign.compute_preflight(upload, kernel_prepared.root / "execution")
    with pytest.raises(ValueError):
        campaign.compute_environment(upload / "runtime", kernel_prepared.root, upload)


def test_kernel_directory_symlink_escape_rejected(kernel_prepared, tmp_path):
    root = tmp_path / "execution"
    root.mkdir()
    (root / campaign.KERNEL_TRACE_DIRECTORY).symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        campaign.compute_environment(kernel_prepared.root / "upload/runtime", root, kernel_prepared.root / "upload")


def test_kernel_enabled_parent_failure_or_missing_full_matrix_still_refused(inputs):
    state = campaign.read(inputs.parent / "campaign-state.json")
    state["GPU_diagnostic_complete"] = False
    write(inputs.parent / "campaign-state.json", state)
    with pytest.raises(ValueError, match="real successful"):
        campaign.prepare(inputs.root, inputs.parent, inputs.performance, inputs.repo, kernel_activity=True)
    assert not inputs.root.exists()
    state["GPU_diagnostic_complete"] = True
    write(inputs.parent / "campaign-state.json", state)
    (inputs.raw / "upstream-first-layer-full-matrices.json").unlink()
    seal_parent(inputs.parent)
    with pytest.raises((ValueError, FileNotFoundError)):
        campaign.prepare(inputs.root, inputs.parent, inputs.performance, inputs.repo, kernel_activity=True)
    assert not inputs.root.exists()


@pytest.mark.parametrize("profiler_status", ["unsupported", "available"])
def test_kernel_campaign_binding_does_not_require_or_claim_cupti_attribution(kernel_prepared, tmp_path, profiler_status):
    root = tmp_path / "execution"
    plan = kernel_worker_evidence(kernel_prepared, root, profiler_status=profiler_status)
    binding = campaign.verify_kernel_campaign(root, plan)
    assert binding["controls"]["kernel_activity"]["enabled"] is True
    assert len(binding["worker_metadata_sha256"]) == 4
    assert binding["unsupported_activity_is_partial_not_kernel_attribution"] is True
    assert binding["specific_kernel_proven"] is binding["numerical_root_cause_proven"] is False
    assert binding["native_graph_kernel_attribution_proven"] is False


@pytest.mark.parametrize("mutation", ["worker_flag", "worker_budget", "worker_source", "runtime_source", "rank"])
def test_actual_worker_kernel_policy_and_source_must_match_frozen_plan(kernel_prepared, tmp_path, mutation):
    root = tmp_path / "execution"
    plan = kernel_worker_evidence(kernel_prepared, root)
    path = root / "upstream-qkv-mirror-worker-instance-1-dp0.json"
    worker = campaign.read(path)
    if mutation == "worker_flag":
        worker["kernel_activity_policy"]["enabled"] = False
    elif mutation == "worker_budget":
        worker["kernel_activity_policy"]["maximum_shadow_calls_per_worker"] = 7
    elif mutation == "worker_source":
        worker["source_sha256"]["mirror_probe"] = "f" * 64
    elif mutation == "rank":
        worker["dp_rank"] = True
    else:
        source = root / "runtime" / campaign.MIRROR_FILES[0]
        source.chmod(0o644)
        source.write_text("changed worker")
    write(path, worker)
    with pytest.raises(ValueError):
        campaign.verify_kernel_campaign(root, plan)


def test_disabled_campaign_cannot_accept_enabled_worker(prepared, tmp_path):
    root = tmp_path / "execution"
    plan = kernel_worker_evidence(prepared, root, enabled=False, profiler_status="disabled")
    assert campaign.verify_kernel_campaign(root, plan)["controls"]["kernel_activity"]["enabled"] is False
    path = root / "upstream-qkv-mirror-worker-instance-2-dp1.json"
    worker = campaign.read(path)
    worker["kernel_activity_policy"]["enabled"] = True
    write(path, worker)
    with pytest.raises(ValueError, match="Actual worker kernel"):
        campaign.verify_kernel_campaign(root, plan)


def test_prepare_cannot_retoggle_an_existing_immutable_kernel_root(prepared, monkeypatch):
    before = campaign.hashes(prepared.root)
    monkeypatch.setattr(campaign, "ssh", lambda *args, **kwargs: pytest.fail("retoggle SSH"))
    with pytest.raises(ValueError, match="NEW root"):
        campaign.run(prepared.root, prepared.parent, prepared.performance, prepared.repo, prepare_only=True, kernel_activity=True)
    assert campaign.hashes(prepared.root) == before


@pytest.mark.parametrize("arguments", [
    ["--kernel-activity", "--root", "/fake", "--parent", "/fake", "--concurrent-performance-root", "/other"],
    ["--kernel-activity", "--compute-upload", "/fake", "--compute-evidence", "/fake"],
    ["--kernel-activity", "--validate-parent", "--parent", "/fake"],
    ["--kernel-activity", "--prepare-only", "--root", "/fake", "--submit-held"]])
def test_kernel_cli_only_opt_in_at_new_prepare_without_effects(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", [campaign.__file__, *arguments])
    monkeypatch.setattr(campaign, "run", lambda *args, **kwargs: pytest.fail("invalid kernel CLI run"))
    with pytest.raises(SystemExit) as error:
        campaign.main()
    assert error.value.code == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_kernel_cli_forwards_only_explicit_prepare_control(monkeypatch, enabled):
    arguments = [campaign.__file__, "--root", "/new", "--parent", "/parent", "--concurrent-performance-root", "/perf", "--prepare-only"]
    if enabled:
        arguments.append("--kernel-activity")
    monkeypatch.setenv("DP_OPT_QKV_MIRROR_KERNEL_ACTIVITY", "1")
    monkeypatch.setattr(sys, "argv", arguments)
    calls = []
    monkeypatch.setattr(campaign, "run", lambda *args: calls.append(args))
    campaign.main()
    assert len(calls) == 1 and calls[0][-1] is enabled and calls[0][-3] is True


def test_kernel_compute_child_gets_frozen_opt_in_and_plan_not_external_env(kernel_prepared, tmp_path, monkeypatch):
    upload = kernel_prepared.root / "upload"
    root = tmp_path / "execution"
    shutil.copytree(upload / "runtime", root / "runtime")
    monkeypatch.setenv("SLURM_JOB_ID", "303")
    monkeypatch.setenv("DP_OPT_QKV_MIRROR_KERNEL_ACTIVITY", "0")
    monkeypatch.setenv("DP_OPT_QKV_MIRROR_KERNEL_MAX_CALLS", "999")
    monkeypatch.setattr(campaign, "compute_preflight", lambda *args: None)
    monkeypatch.setattr(campaign, "installed_parent", lambda *args: kernel_prepared.state["parent_gate"]["installed"])
    def child(command, env, cwd):
        assert env["DP_OPT_QKV_MIRROR_KERNEL_ACTIVITY"] == "1"
        assert "DP_OPT_QKV_MIRROR_KERNEL_MAX_CALLS" not in env
        assert env["DP_OPT_PHASE_EVIDENCE_ROOT"] == str(root)
        plan = campaign.read(root / "plan.json")
        assert plan["controls"] == kernel_prepared.state["controls"]
        assert plan["controls"]["kernel_activity"]["maximum_shadow_calls_per_worker"] == 6
        assert plan["controls"]["kernel_trace_directory"] == campaign.KERNEL_TRACE_DIRECTORY
        shutil.copy2(env["DP_OPT_UPSTREAM_PLACEMENT_FIT"], root / "placement-fit.json")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(campaign.subprocess, "run", child)
    campaign.compute(upload, root)
    saved = campaign.read(root / "mirror-runtime-environment.json")
    assert saved["DP_OPT_QKV_MIRROR_KERNEL_ACTIVITY"] == "1"
    assert saved["DP_OPT_PHASE_EVIDENCE_ROOT"] == str(root)


@pytest.mark.parametrize("worker_matches", [True, False])
def test_compute_main_binds_policy_after_probe_without_promoting_partial(kernel_prepared, tmp_path, monkeypatch, worker_matches):
    import head_hidden_probe

    root = tmp_path / "execution"
    plan = kernel_worker_evidence(kernel_prepared, root, enabled=worker_matches)
    write(root / "plan.json", plan)
    report = {"status": "bounded_qkv_mirror_observation_not_quality_fix", "kernel_activity": "partial", **campaign.CLAIMS}
    events = []
    original_instrument = head_hidden_probe.instrument
    monkeypatch.setattr(campaign, "require_compute", lambda: None)
    monkeypatch.setattr(head_hidden_probe, "main", lambda: events.append("baseline_complete"))
    def analyzed(actual):
        assert actual == root and events == ["baseline_complete"]
        events.append("analyzed")
        return report
    monkeypatch.setattr(campaign.mirror_probe, "analyze", analyzed)
    monkeypatch.setattr(sys, "argv", ["fake-entry", "--output", str(root)])
    if worker_matches:
        campaign.compute_main()
        assert campaign.read(root / "compute-upstream-qkv-mirror-verification.json") == report
        binding = campaign.read(root / "compute-qkv-kernel-campaign-binding.json")
        assert binding == campaign.verify_kernel_campaign(root, plan)
        assert binding["specific_kernel_proven"] is binding["numerical_root_cause_proven"] is False
    else:
        with pytest.raises(ValueError, match="Actual worker kernel"):
            campaign.compute_main()
        assert not (root / "compute-upstream-qkv-mirror-verification.json").exists()
        assert not (root / "compute-qkv-kernel-campaign-binding.json").exists()
    assert events == ["baseline_complete", "analyzed"]
    assert head_hidden_probe.instrument is original_instrument
