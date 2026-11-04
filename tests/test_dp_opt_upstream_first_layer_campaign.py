import copy
import json
import os
import pickle
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import upstream_first_layer_campaign as campaign


def test_private_execution_modes_do_not_change_frozen_upload_bytes(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "experiments/dp_opt/upstream-first-layer-probe.sbatch").read_text()
    copy_command = 'cp -a "$upload/runtime" "$evidence/runtime"'
    mode_command = 'find "$evidence/runtime" -type f -exec chmod u+w {} +'
    assert text.count(copy_command) == text.count(mode_command) == 1
    assert text.index(copy_command) < text.index(mode_command) < text.index('cd "$evidence/runtime"')
    assert "private_execution_copy_only" in text
    upload, evidence = tmp_path / "upload", tmp_path / "evidence"
    (upload / "runtime").mkdir(parents=True)
    evidence.mkdir()
    source = upload / "runtime/source.py"
    source.write_text("immutable bytes\n")
    source.chmod(0o444)
    subprocess.run(["bash", "-c", copy_command + "\n" + mode_command],
                   env={**os.environ, "upload": str(upload), "evidence": str(evidence)}, check=True)
    copied = evidence / "runtime/source.py"
    assert source.stat().st_mode & 0o777 == 0o444
    assert copied.stat().st_mode & 0o777 == 0o644
    assert copied.read_bytes() == source.read_bytes()
    fixture = tmp_path / "tamper-fixture.py"
    shutil.copy2(copied, fixture)
    assert fixture.stat().st_mode & 0o200
    fixture.write_text("changed fixture only")
    assert copied.read_bytes() == source.read_bytes() == b"immutable bytes\n"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    repo, parent, performance, root = [tmp_path / name for name in ("repo", "parent", "performance-r5", "upstream")]
    names = (*campaign.NEW_FILES, *campaign.PROBE_FILES, campaign.COPY_ONLY,
             "experiments/dp_opt/live_trace.py", "experiments/dp_opt/performance_followup.py")
    for name in names:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen " + name)
    baseline = {name: campaign.digest(repo / name) for name in names[-2:]}
    parent_state = {"job": 278529, "state_last_observed": "COMPLETED", "GPU_diagnostic_complete": True,
                    "frozen_runtime_sha256": baseline}
    write(parent / "campaign-state.json", parent_state)
    raw = parent / "live-trace-278529"
    trace = {"requests": [{"request_id": "conv1", "prompt_tokens": 149},
                           {"request_id": "conv0", "prompt_tokens": 74}]}
    write(raw / "trace.json", trace)
    write(raw / "frozen/trace.json", trace)
    write(raw / "frozen/inputs/prompt-bank.json", {"original": True})
    write(raw / "placement-fit.json", {"original": "fit"})
    rows = [{"request_id": "conv1", "repeat": repeat, "layout": layout,
             "first_top1_attribution_available": True, "first_divergence_position": 53,
             "full_common_prefix": True, "hidden_changed_elements": 1845,
             "hidden_contribution": .04467149} for repeat, layout in
            ((1, "corank0"), (2, "corank1"), (3, "corank1"), (4, "corank0"))]
    report = {"projection_comparisons": rows}
    write(raw / "independent-head-hidden-verification.json", report)
    write(parent / "independent-head-hidden-verification.json", report)
    (parent / "raw-278529.sha256").write_text("".join(
        f"{checksum}  ./{name}\n" for name, checksum in campaign.hashes(raw).items()))
    for suffix in ("out", "err"):
        (parent / f"live-trace-278529.{suffix}").write_text("original " + suffix)
    write(parent / "archive-278529.json", {"job": 278529, "state": "COMPLETED", "missing_job_logs": [],
        "log_sha256": {suffix: campaign.digest(parent / f"live-trace-278529.{suffix}") for suffix in ("out", "err")}})
    initial = performance / "authorized-debug/00000000.json"
    write(initial, {"campaign_root": str(performance), "sequence": 0, "previous_sha256": None,
        "identities": [{"job": 278529, "name": "qwen-head-hidden-probe"}]})
    poll_epoch = time.time() - 2
    polled_at = datetime.fromtimestamp(poll_epoch).astimezone().isoformat()
    write(performance / "stages/native/poll-101.log", {"at": polled_at, "ssh_exit": 0,
        "debug_authorization_SHA": {"sequence": 0, "sha256": campaign.digest(initial)}, "monotonic_gap_s": 300})
    with (performance / "stages/native/poll-101.log").open("a") as output:
        output.write("\n101|RUNNING|0:0|start|Unknown|00:04:00|node\n")
    os.utime(performance / "stages/native/poll-101.log", (poll_epoch + 1, poll_epoch + 1))
    write(performance / "campaign-state.json", {"repo": str(repo), "performance_independent_of_debug": True,
        "controls": {"monitor_interval_s": 300},
        "frozen_runtime_sha256": baseline, "debug_registry": {"path": str(initial.parent),
            "initial_sha256": campaign.digest(initial)}, "stages": {"native": {"job": 101, "phase": "monitoring",
                "stage_id": "native", "last_poll_at": polled_at, "last_poll_ssh_exit": 0}}})
    write(performance / "local-monitor-identity.json", {"pid": os.getpid(),
        "argv": [sys.executable, str(repo / "experiments/dp_opt/performance_followup.py"), "--root", str(performance)]})
    state = campaign.prepare(root, parent, performance, repo)
    return root, parent, performance, repo, state


def test_prepare_and_prepare_only_are_strictly_local(prepared, monkeypatch):
    root, parent, performance, repo, state = prepared
    monkeypatch.setattr(campaign, "ssh", lambda *args, **kwargs: pytest.fail("prepare-only contacted SSH"))
    monkeypatch.setattr(campaign.subprocess, "run", lambda *args, **kwargs: pytest.fail("prepare-only spawned runtime/Slurm"))
    campaign.run(root, parent, performance, repo, prepare_only=True)
    assert state["phase"] == "prepared" and "job" not in state
    assert campaign.digest(repo / campaign.COPY_ONLY) == campaign.digest(root / "upload/runtime" / campaign.COPY_ONLY)
    assert (root / "upload/runtime" / campaign.COPY_ONLY).stat().st_mode & 0o222 == 0
    assert campaign.read(root / "upload/plan.json")["controls"]["layouts"] == campaign.LAYOUTS


def test_reviewed_sources_are_snapshotted_at_prepare_not_hardcoded(prepared):
    root, parent, performance, repo, state = prepared
    expected = {name: campaign.digest(repo / name) for name in campaign.PROBE_FILES}
    assert state["first_layer_implementation_sha256"] == expected
    assert campaign.read(root / "upload/plan.json")["first_layer_implementation_sha256"] == expected
    source = repo / campaign.PROBE_FILES[0]
    source.write_text("new reviewed worker epoch")
    with pytest.raises(ValueError, match="SHA mismatch"):
        campaign.validate_local(root, state)
    replacement = campaign.prepare(root.parent / "next-upstream", parent, performance, repo)
    assert replacement["first_layer_implementation_sha256"][campaign.PROBE_FILES[0]] == campaign.digest(source)


@pytest.mark.parametrize("mutation", ["runtime", "payload", "registry", "identity"])
def test_resume_rejects_changed_immutable_runtime_or_identity(prepared, mutation):
    root, parent, performance, repo, state = prepared
    if mutation == "runtime":
        (repo / "experiments/dp_opt/live_trace.py").write_text("changed")
    elif mutation == "payload":
        (root / "payload.tar.gz").write_bytes(b"changed")
    elif mutation == "registry":
        write(performance / "authorized-debug/00000000.json", {})
    else:
        state["parent"] = "other-parent"
        write(root / "campaign-state.json", state)
    with pytest.raises((ValueError, KeyError)):
        campaign.run(root, parent, performance, repo, prepare_only=True)


def run_guard(queue, accounts, debug_limit=0):
    identities = [{"job": 278529, "name": "qwen-head-hidden-probe"}]
    guard = campaign.queue_guard(101, identities, debug_limit)
    lines = ['id() { printf researcher; }', 'export USER=researcher',
             "squeue() { printf '%s\\n' " + shlex.quote(queue) + "; }", 'sacct() { case "$*" in']
    for job, account in accounts.items():
        lines.append(f"*'-j {job} '*) printf '%s\\n' {shlex.quote(account)};;")
    lines.extend(['*) return 1;; esac; }', guard])
    return subprocess.run(["bash", "-c", "\n".join(lines)], text=True, capture_output=True)


ACCOUNTS = {101: "101|researcher|qwen-public-perf-followup|gpu_a800|RUNNING|",
            278529: "278529|researcher|qwen-head-hidden-probe|gpu_a800|COMPLETED|"}


def test_exact_authorized_performance_job_and_terminal_debug_pass_guard():
    assert run_guard("101|researcher|qwen-public-perf-followup|gpu_a800|RUNNING", ACCOUNTS).returncode == 0


@pytest.mark.parametrize("queue", [
    "101|other|qwen-public-perf-followup|gpu_a800|RUNNING",
    "101|researcher|qwen-public-perf-followup|gpu_h100|RUNNING",
    "101|researcher|qwen-public-perf-wrong|gpu_a800|RUNNING",
    "101|researcher|qwen-public-perf-followup|gpu_a800|RUNNING\n202|researcher|qwen-unknown|gpu_a800|PENDING",
    "101|researcher|qwen-public-perf-followup|gpu_a800|RUNNING\n278529|researcher|qwen-head-hidden-probe|gpu_a800|RUNNING",
])
def test_unknown_third_job_owner_name_partition_or_old_debug_is_rejected(queue):
    accounts = copy.deepcopy(ACCOUNTS)
    if "278529|researcher" in queue:
        accounts[278529] = "278529|researcher|qwen-head-hidden-probe|gpu_a800|RUNNING|"
    assert run_guard(queue, accounts).returncode != 0


def test_copy_only_protected_sync_never_updates_canonical_remote_repo(prepared, monkeypatch):
    root, parent, performance, repo, state = prepared
    calls = []

    def fake_ssh(root, label, script, *args, **kwargs):
        calls.append((label, script))
        return "HOME_RESULT|/home/researcher" if label == "before-sync" else ""

    monkeypatch.setattr(campaign, "ssh", fake_ssh)
    campaign.protected_sync(root, state)
    installer = dict(calls)["protected-copy-only-install"]
    assert "backup/" in installer and "before.sha256" in installer and "after.sha256" in installer
    assert "mv upload/" in installer and "sha256sum incoming/" in installer
    assert "/kvcached/kvcached/" not in installer
    assert not any(re.search(r"(^|[;\n])\s*(python\S*|sbatch)\s", script) for label, script in calls)


def test_submit_command_is_held_exact_two_A800_and_write_once_ledger(prepared, monkeypatch):
    root, parent, performance, repo, state = prepared
    state.update({"remote_stage": "/home/researcher/stage", "submission_ledger": "/home/researcher/stage/submitted-debug-job.txt"})
    state["performance_poll_window"] = campaign.performance_poll_window(performance)
    state["submit_register_deadline_epoch"] = time.time() + 60
    calls = []
    monkeypatch.setattr(campaign, "ssh", lambda root, label, script, **kwargs: calls.append(script) or "202;cluster")
    assert campaign.submit_held(root, state) == 202
    script = calls[0]
    assert "sbatch --hold --parsable --partition=gpu_a800 --nodes=1 --gpus=2 --time=03:00:00" in script
    assert "test ! -e" in script and "submission-intent" in script
    assert "submitted-debug-job.txt; cat" in script
    assert "scontrol release" not in script


def test_authorization_chain_registers_immediately_after_submit_then_releases(prepared):
    root, parent, performance, repo, state = prepared
    chain = campaign.authorization_chain(root, state)
    submit = chain.index("--submit-held >")
    register = chain.index('--register-debug-job "$job"')
    release = chain.index('--release-registered-job "$job"')
    assert submit < register < release
    assert "--register-debug-name " + campaign.JOB_NAME in chain
    assert str(performance) in chain and "set -o noclobber" in chain
    assert '--cancel-owned-held-job "$job"' in chain


def test_release_refuses_unregistered_or_other_receipt_job(prepared, monkeypatch):
    root, parent, performance, repo, state = prepared
    (root / "submission-receipt.txt").write_text("202\n")
    monkeypatch.setattr(campaign, "ssh", lambda *args, **kwargs: pytest.fail("unauthorized release"))
    with pytest.raises(ValueError, match="authorization"):
        campaign.release_registered(root, state, 202)
    with pytest.raises(ValueError, match="receipt"):
        campaign.release_registered(root, state, 203)


def test_registration_requires_verified_SHA_append_before_release(prepared, monkeypatch):
    root, parent, performance, repo, state = prepared
    initial = performance / "authorized-debug/00000000.json"
    write(performance / "authorized-debug/00000001.json", {"campaign_root": str(performance), "sequence": 1,
        "previous_sha256": campaign.digest(initial), "operator_explicit_registration": True, "remote_verified": True,
        "identities": [{"job": 202, "name": campaign.JOB_NAME, "owner": "researcher"}]})
    (root / "submission-receipt.txt").write_text("202\n")
    calls = []
    monkeypatch.setattr(campaign, "ssh", lambda root, label, script, **kwargs: calls.append(script))
    campaign.release_registered(root, state, 202)
    assert "scontrol release 202" in calls[0]
    assert "202:" + campaign.JOB_NAME in calls[0]


def test_existing_submission_intent_is_never_resubmitted(prepared, monkeypatch):
    root, parent, performance, repo, state = prepared
    state["phase"] = "submission_intent_do_not_duplicate"
    write(root / "campaign-state.json", state)
    monkeypatch.setattr(campaign, "ssh", lambda *args, **kwargs: pytest.fail("duplicate submission"))
    with pytest.raises(RuntimeError, match="duplicate"):
        campaign.run(root, parent, performance, repo)


def test_owned_held_cleanup_never_cancels_performance_or_running_job(prepared, monkeypatch):
    root, parent, performance, repo, state = prepared
    (root / "submission-receipt.txt").write_text("202\n")
    calls = []
    monkeypatch.setattr(campaign, "ssh", lambda root, label, script, **kwargs: calls.append(script))
    campaign.cancel_owned_held(root, state, 202)
    assert "scancel 202" in calls[0] and "scancel 101" not in calls[0]
    assert "Reason=JobHeldUser" in calls[0] and "JobState=PENDING" in calls[0]
    assert "squeue -h -j 202" in calls[0] and "sacct" not in calls[0]
    assert campaign.JOB_NAME in calls[0]


@pytest.mark.parametrize("state_name", ["RUNNING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"])
def test_actual_poll_preserves_true_state_and_identity(prepared, monkeypatch, state_name):
    root, parent, performance, repo, state = prepared
    state["job"] = 202
    monkeypatch.setattr(campaign, "ssh", lambda *args, **kwargs:
        f"OWNER|researcher\n202|researcher|{campaign.JOB_NAME}|gpu_a800|{state_name}|")
    assert campaign.poll(root, state) == state_name
    assert state["last_poll_ssh_exit"] == 0 and state["state_last_observed"] == state_name


def test_accounting_lag_does_not_archive_a_still_active_queue_job(prepared, monkeypatch):
    root, parent, performance, repo, state = prepared
    state["job"] = 202
    monkeypatch.setattr(campaign, "ssh", lambda *args, **kwargs:
        f"OWNER|researcher\n202|researcher|{campaign.JOB_NAME}|gpu_a800|COMPLETED|\n"
        f"202|researcher|{campaign.JOB_NAME}|gpu_a800|RUNNING")
    assert campaign.poll(root, state) == "RUNNING"


def test_runtime_imports_are_compute_only(prepared, monkeypatch):
    root, parent, performance, repo, state = prepared
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="compute node"):
        campaign.compute(root / "upload", root)


def test_frozen_shell_dynamic_replacement_is_copy_only_and_fail_closed():
    repo = Path(campaign.__file__).resolve().parents[2]
    source = repo / "experiments/dp_opt/trace-benefit.sbatch"
    before = campaign.digest(source)
    launch = campaign.make_launch(source.read_text())
    assert campaign.digest(source) == before
    assert '"$evidence/upstream_first_layer_entry.py"' in launch
    assert '--fit "$DP_OPT_UPSTREAM_PLACEMENT_FIT"' in launch
    assert '--fit "$evidence/placement-fit.json"' not in launch
    assert '"$DP_OPT_UPSTREAM_PLACEMENT_FIT" "${DP_OPT_TRACE_PATH:?Set DP_OPT_TRACE_PATH}"' in launch
    assert "DP_OPT_UPSTREAM_RUNTIME" in launch
    assert '"$root/state/sparse-fit-231467.json"' not in launch
    assert "python -m pytest -q -p no:cacheprovider" in launch
    with pytest.raises(ValueError, match="entry contract"):
        campaign.make_launch("unknown source")


def test_sbatch_preserves_graphs_compute_gating_and_immutable_worker_imports():
    source = Path(campaign.__file__).with_name("upstream-first-layer-probe.sbatch").read_text()
    assert "#SBATCH --partition=gpu_a800" in source and "#SBATCH --gpus=2" in source
    assert "#SBATCH --time=03:00:00" in source and "require_compute" in source
    assert "test ! -e \"$evidence/runtime\"" in source
    assert "upstream_first_layer_worker, upstream_first_layer_probe" in source
    assert "DP_OPT_UPSTREAM_COMPUTE_CONFIRMED=1" in source
    assert "InstalledProvenanceUpstreamWorker" in source
    assert "--enforce-eager" not in source
    assert "cp -a \"$upload/runtime\" \"$evidence/runtime\"" in source


@pytest.mark.parametrize("argv", [[], ["--root", "x"], ["--concurrent-performance-root", "x"],
    ["--root", "x", "--submit-held", "--release-registered-job", "202"]])
def test_cli_missing_explicit_bindings_or_mixed_helpers_fails_closed(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["campaign", *argv])
    with pytest.raises(SystemExit):
        campaign.main()


class LinearBase:
    pass


class ColumnParallelLinear(LinearBase):
    def forward(self, inputs):
        return inputs


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self):
        self.quant_method = UnquantizedLinearMethod()
        self.weight = SimpleNamespace(dtype="torch.bfloat16", shape=(6, 4))


class LinearMethodBase:
    pass


class UnquantizedLinearMethod(LinearMethodBase):
    def apply(self, layer, inputs, bias=None):
        return inputs


class FakeCUDAGraphWrapper:
    def __call__(self, inputs):
        return inputs


class FakeModel:
    def __init__(self):
        self.model = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(qkv_proj=QKVParallelLinear()))])


class FakeRunner:
    def __init__(self):
        self.model = FakeCUDAGraphWrapper()
        self.unwrapped = FakeModel()

    def get_model(self):
        return self.unwrapped

    def _model_forward(self, inputs):
        return self.model(inputs)


class FakeUpstreamWorker:
    def initialize_from_config(self, configuration):
        return configuration


@pytest.fixture
def installed_workers(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "202")
    monkeypatch.setenv("DP_OPT_UPSTREAM_COMPUTE_CONFIRMED", "1")
    monkeypatch.setenv("DP_OPT_EVIDENCE_ROOT", str(tmp_path))
    for service in ("instance-1", "instance-2"):
        for rank in (0, 1):
            monkeypatch.setenv("DP_OPT_SERVICE", service)
            worker = SimpleNamespace(model_runner=FakeRunner(), parallel_config=SimpleNamespace(data_parallel_rank=rank))
            metadata = {"service": service, "dp_rank": rank, "checkpoint": "same-checkpoint",
                "model_class": campaign.qualified(FakeModel), "model_dtype": "torch.bfloat16",
                "attention_tp": 1, "dp": 2, "ep": 2, "pp": 1, "cudagraph_mode": "FULL_DECODE_ONLY",
                "projection": {"weight_dtype": "torch.bfloat16", "weight_shape": [6, 4], "weight_sha256": "a" * 64},
                "source_sha256": {"qkv": campaign.digest(__file__), "runner": campaign.digest(__file__)}}
            write(tmp_path / f"upstream-first-layer-worker-{service}-dp{rank}.json", metadata)
            campaign.collect_installed_provenance(worker, FakeCUDAGraphWrapper)
    return tmp_path


def test_compute_installed_sources_bind_actual_linear_base_apply_and_graph(installed_workers):
    result = campaign.verify_installed_provenance(installed_workers, 202)
    assert result["workers_verified"] == 4 and result["specific_kernel_proven"] is False
    record = campaign.read(installed_workers / "installed-upstream-linear-graph-instance-1-dp0.json")
    assert set(record["sources"]) == campaign.INSTALLED_SYMBOLS
    assert record["sources"]["quant_method_apply"]["qualified_name"].endswith("UnquantizedLinearMethod.apply")
    assert record["sources"]["linear_forward"]["qualified_name"].endswith("ColumnParallelLinear.forward")
    assert record["sources"]["linear_base"]["qualified_name"].endswith("LinearBase")
    assert record["sources"]["cudagraph_call"]["qualified_name"].endswith("FakeCUDAGraphWrapper.__call__")
    assert record["quality_equivalence_proven"] is False


@pytest.mark.parametrize("mutation", ["worker", "job", "source", "symbol", "path", "kernel", "model", "weight"])
def test_independent_installed_verifier_rejects_incomplete_or_mismatched_provenance(installed_workers, mutation):
    path = installed_workers / "installed-upstream-linear-graph-instance-1-dp0.json"
    record = campaign.read(path)
    if mutation == "worker":
        path.unlink()
    elif mutation == "job":
        record["slurm_job_id"] = 203
    elif mutation == "source":
        (installed_workers / record["sources"]["quant_method_apply"]["copied_path"]).write_text("changed source")
    elif mutation == "symbol":
        record["sources"]["quant_method_apply"]["symbol_source_sha256"] = "0" * 64
    elif mutation == "path":
        record["sources"]["quant_method_apply"]["copied_path"] = "../outside"
    elif mutation == "kernel":
        record["specific_kernel_proven"] = True
    elif mutation == "model":
        record["model_dtype"] = "torch.float32"
    else:
        record["projection"]["weight_sha256"] = "b" * 64
    if path.exists():
        write(path, record)
    with pytest.raises(ValueError):
        campaign.verify_installed_provenance(installed_workers, 202)


@pytest.mark.parametrize("attribute", ["model", "quant_method", "apply", "layers"])
def test_installed_unknown_runtime_attributes_fail_closed(tmp_path, monkeypatch, attribute):
    monkeypatch.setenv("SLURM_JOB_ID", "202")
    monkeypatch.setenv("DP_OPT_UPSTREAM_COMPUTE_CONFIRMED", "1")
    monkeypatch.setenv("DP_OPT_EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.setenv("DP_OPT_SERVICE", "instance-1")
    runner = FakeRunner()
    projection = runner.get_model().model.layers[0].self_attn.qkv_proj
    if attribute == "model":
        runner.model = object()
    elif attribute == "quant_method":
        del projection.quant_method
    elif attribute == "apply":
        monkeypatch.delattr(UnquantizedLinearMethod, "apply")
    else:
        runner.get_model().model.layers = []
    with pytest.raises(RuntimeError, match="Unsupported installed"):
        campaign.collect_installed_provenance(SimpleNamespace(model_runner=runner,
            parallel_config=SimpleNamespace(data_parallel_rank=0)), FakeCUDAGraphWrapper)


def test_spawn_resolvable_worker_adapter_preserves_parent_outputs_and_compute_guard(monkeypatch):
    monkeypatch.delitem(campaign.__dict__, "InstalledProvenanceUpstreamWorker", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="compute node"):
        getattr(campaign, "InstalledProvenanceUpstreamWorker")
    monkeypatch.setenv("SLURM_JOB_ID", "202")
    monkeypatch.setenv("DP_OPT_UPSTREAM_COMPUTE_CONFIRMED", "1")
    monkeypatch.setitem(sys.modules, "upstream_first_layer_worker", SimpleNamespace(
        UpstreamFirstLayerVerificationWorker=FakeUpstreamWorker))
    collected = []
    monkeypatch.setattr(campaign, "collect_installed_provenance", lambda worker: collected.append(worker))
    try:
        worker_type = getattr(campaign, "InstalledProvenanceUpstreamWorker")
        worker = pickle.loads(pickle.dumps(worker_type()))
        sentinel = object()
        assert worker.initialize_from_config(sentinel) is sentinel
        assert collected == [worker]
        assert worker_type.__module__ == campaign.__name__
    finally:
        campaign.__dict__.pop("InstalledProvenanceUpstreamWorker", None)


def test_performance_window_binds_fresh_successful_actual_poll_at_and_mtime(prepared):
    root, parent, performance, repo, state = prepared
    window = campaign.performance_poll_window(performance)
    assert window["performance_job"] == 101
    assert window["remaining_cycle_s"] > 90
    assert window["completed_age_s"] < 30
    assert window["max_submission_registration_s"] == 60
    assert window["atomic_authorization"] is False
    assert window["receipt_sha256"] == campaign.digest(window["receipt_path"])


def test_performance_window_accepts_ordered_coarse_filesystem_mtime(prepared):
    root, parent, performance, repo, state = prepared
    path = performance / "stages/native/poll-101.log"
    timestamp = int(path.stat().st_mtime)
    os.utime(path, (timestamp, timestamp))
    window = campaign.performance_poll_window(performance)
    assert window["receipt_mtime_ns"] == path.stat().st_mtime_ns
    assert window["completed_age_s"] < 30


def test_sbatch_avoids_submission_home_namespace_before_worker_imports():
    script = Path(campaign.__file__).with_name("upstream-first-layer-probe.sbatch").read_text()
    assert script.index('cd "$evidence"') < script.index("python -c '")


def test_failed_startup_archives_without_claiming_missing_frozen_inputs_passed(prepared, monkeypatch):
    root, parent, performance, repo, state = prepared
    state["job"] = 102
    raw = root / "live-trace-102"
    campaign.shutil.copytree(root / "upload/runtime", raw / "runtime")
    (raw / "exit-status").write_text("1\n")
    (raw / "upstream-startup-tests.log").write_text("5 failed, 334 passed\n")
    for suffix in ("out", "err"):
        (root / f"live-trace-102.{suffix}").write_text("preserved startup failure\n")
    write(root / "archive-102.json", {"job": 102, "state": "FAILED", "missing_job_logs": [],
          "log_sha256": {suffix: campaign.digest(root / f"live-trace-102.{suffix}") for suffix in ("out", "err")}})
    (root / "raw-102.sha256").write_text("".join(
        f"{checksum}  ./{name}\n" for name, checksum in campaign.hashes(raw).items()))
    monkeypatch.setattr(campaign.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="archived", stderr=""))
    monkeypatch.setattr(campaign, "verify_installed_provenance", lambda *args:
                        pytest.fail("Failed startup must not claim installed worker validation"))
    with pytest.raises(RuntimeError, match="DEBUG allocation FAILED"):
        campaign.archive_and_verify(root, state)
    report = campaign.read(root / "failed-terminal-diagnostic.json")
    assert report["terminal_raw_and_logs_SHA_verified"] is True
    assert report["frozen_inputs_present"] is False
    assert report["GPU_diagnostic_complete"] is False


@pytest.mark.parametrize("mutation", ["stale", "long_poll", "state_at", "failed_ssh", "terminal", "interval"])
def test_performance_window_refuses_stale_failed_partial_or_too_little_headroom(prepared, mutation):
    root, parent, performance, repo, state = prepared
    path = performance / "campaign-state.json"
    perf = campaign.read(path)
    entry = perf["stages"]["native"]
    poll_path = performance / "stages/native/poll-101.log"
    lines = poll_path.read_text().splitlines()
    header = json.loads(lines[0])
    now = time.time()
    if mutation == "stale":
        now += 31
    elif mutation == "long_poll":
        header["at"] = datetime.fromtimestamp(now - 250).astimezone().isoformat()
        entry["last_poll_at"] = header["at"]
    elif mutation == "state_at":
        entry["last_poll_at"] = datetime.fromtimestamp(now - 1).astimezone().isoformat()
    elif mutation == "failed_ssh":
        header["ssh_exit"] = entry["last_poll_ssh_exit"] = 124
    elif mutation == "terminal":
        lines[1] = lines[1].replace("RUNNING", "COMPLETED")
    else:
        perf["controls"]["monitor_interval_s"] = 60
    lines[0] = json.dumps(header)
    poll_path.write_text("\n".join(lines) + "\n")
    write(path, perf)
    with pytest.raises(ValueError):
        campaign.performance_poll_window(performance, now=max(now, poll_path.stat().st_mtime))


def test_local_window_waits_for_next_receipt_without_ssh(prepared, monkeypatch):
    root, parent, performance, repo, state = prepared
    expected = campaign.performance_poll_window(performance)
    calls = []

    def next_receipt(path):
        calls.append(path)
        if len(calls) == 1:
            raise ValueError("wait for next completed receipt")
        return expected

    sleeps = []
    monkeypatch.setattr(campaign, "performance_poll_window", next_receipt)
    monkeypatch.setattr(campaign.time, "sleep", sleeps.append)
    monkeypatch.setattr(campaign, "ssh", lambda *args, **kwargs: pytest.fail("local receipt wait contacted SSH"))
    assert campaign.wait_for_performance_window(root, state) == expected
    assert sleeps == [1] and len(calls) == 2


def test_window_wait_has_finite_bound_and_never_submits(prepared, monkeypatch):
    root, parent, performance, repo, state = prepared
    ticks = iter([0, 361])
    monkeypatch.setattr(campaign.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(campaign, "performance_poll_window", lambda root: (_ for _ in ()).throw(ValueError("stale")))
    monkeypatch.setattr(campaign, "ssh", lambda *args, **kwargs: pytest.fail("unsafe submission"))
    with pytest.raises(RuntimeError, match="360s"):
        campaign.wait_for_performance_window(root, state)


@pytest.mark.parametrize("mutation", ["receipt_changed", "deadline_expired"])
def test_submit_rechecks_poll_receipt_and_deadline_before_any_ssh(prepared, monkeypatch, mutation):
    root, parent, performance, repo, state = prepared
    state["performance_poll_window"] = campaign.performance_poll_window(performance)
    state["submit_register_deadline_epoch"] = time.time() + 60
    if mutation == "receipt_changed":
        with Path(state["performance_poll_window"]["receipt_path"]).open("a") as output:
            output.write("\n")
    else:
        state["submit_register_deadline_epoch"] = time.time() - 1
    monkeypatch.setattr(campaign, "ssh", lambda *args, **kwargs: pytest.fail("unsafe held submission"))
    with pytest.raises(ValueError, match="poll changed|deadline"):
        campaign.submit_held(root, state)


def test_60s_chain_timeout_kills_local_group_and_only_cleans_receipt_bound_debug(prepared, monkeypatch):
    root, parent, performance, repo, state = prepared
    state["submit_register_deadline_epoch"] = time.time() + 60
    (root / "submission-receipt.txt").write_text("202\n")
    calls, kills, cleanup = [], [], []

    class FakeProcess:
        pid = 999999
        returncode = -15

        def communicate(self, timeout=None):
            calls.append(timeout)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired("chain", timeout)
            return "", "registered or held; preserve actual evidence"

    options = []
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *args, **kwargs: options.append(kwargs) or FakeProcess())
    monkeypatch.setattr(campaign.os, "killpg", lambda *args: kills.append(args))
    monkeypatch.setattr(campaign, "cancel_owned_held", lambda root, state, job: cleanup.append(job))
    with pytest.raises(RuntimeError, match="60s"):
        campaign.bounded_authorization(root, state)
    assert options[0]["start_new_session"] is True and 0 < calls[0] <= 60
    assert kills == [(999999, campaign.signal.SIGTERM)] and cleanup == [202]
    assert 101 not in cleanup
    assert campaign.read(root / "submit-register-chain.json")["atomic_authorization"] is False


def test_source_epoch_conflicts_list_every_path_and_both_original_hashes():
    parent = {"same.py": "a" * 64, "first.py": "b" * 64, "second.py": "c" * 64}
    performance = {"same.py": "a" * 64, "first.py": "d" * 64, "second.py": "e" * 64}
    original = copy.deepcopy((parent, performance))
    report = campaign.source_manifest_diff(parent, performance)
    assert set(report["conflicts"]) == {"first.py", "second.py"}
    with pytest.raises(ValueError) as error:
        campaign.source_union(parent, performance)
    for name in report["conflicts"]:
        assert name in str(error.value)
        assert parent[name] in str(error.value) and performance[name] in str(error.value)
    assert (parent, performance) == original


def test_source_validation_is_read_only_and_creates_no_real_prepare_root(prepared, monkeypatch):
    root, parent, performance, repo, state = prepared
    before = campaign.hashes(parent), campaign.hashes(performance)
    monkeypatch.setattr(campaign, "ssh", lambda *args, **kwargs: pytest.fail("SHA-only validation contacted SSH"))
    monkeypatch.setattr(campaign.subprocess, "run", lambda *args, **kwargs: pytest.fail("SHA-only validation launched runtime"))
    report = campaign.validate_source_epochs(parent, performance, repo)
    assert report["conflicts"] == {} and report["local_union_source_sha_verified"] is True
    assert report["prepare_performed"] is False and report["ssh_performed"] is False
    assert before == (campaign.hashes(parent), campaign.hashes(performance))


def test_sha_only_cli_does_not_call_prepare_or_run(prepared, monkeypatch, capsys):
    root, parent, performance, repo, state = prepared
    monkeypatch.setattr(sys, "argv", ["campaign", "--validate-only", "--parent", str(parent),
        "--concurrent-performance-root", str(performance)])
    monkeypatch.setattr(campaign, "validate_source_epochs", lambda *args: {"prepare_performed": False})
    monkeypatch.setattr(campaign, "run", lambda *args, **kwargs: pytest.fail("readonly CLI ran coordinator"))
    campaign.main()
    assert json.loads(capsys.readouterr().out)["prepare_performed"] is False


def test_compute_environment_restores_guard_and_preserves_original_trace_without_gpu(prepared, monkeypatch):
    root, parent, performance, repo, state = prepared
    upload = root / "upload"
    template_name = "experiments/dp_opt/trace-benefit.sbatch"
    template = upload / "runtime" / template_name
    template.write_text(Path(campaign.__file__).with_name("trace-benefit.sbatch").read_text())
    manifest = campaign.read(upload / "runtime-manifest.json")
    manifest[template_name] = campaign.digest(template)
    write(upload / "runtime-manifest.json", manifest)
    evidence = root / "fake-compute"
    campaign.shutil.copytree(upload / "runtime", evidence / "runtime")
    monkeypatch.setenv("SLURM_JOB_ID", "202")
    monkeypatch.setenv("DP_OPT_UPSTREAM_COMPUTE_CONFIRMED", "1")
    monkeypatch.setenv("DP_OPT_STRESS_SWEEP", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    environments = []
    monkeypatch.setattr(campaign.subprocess, "run", lambda command, env: environments.append(env) or SimpleNamespace(returncode=0))
    campaign.compute(upload, evidence)
    environment = environments[0]
    assert environment["DP_OPT_UPSTREAM_COMPUTE_CONFIRMED"] == "1"
    assert environment["PYTEST_ADDOPTS"] == "-p no:cacheprovider"
    assert "DP_OPT_STRESS_SWEEP" not in environment and "VLLM_BATCH_INVARIANT" not in environment
    assert environment["DP_OPT_TRACE_REPEATS"] == "6" and environment["DP_OPT_TRACE_POLICIES"] == "no_move"
    assert environment["DP_OPT_KV_CACHE_BYTES"] == "25769803776"
    input_fit = Path(environment["DP_OPT_UPSTREAM_PLACEMENT_FIT"])
    output_fit = evidence / "placement-fit.json"
    assert input_fit.resolve() == (upload / "placement-fit.json").resolve()
    assert input_fit.resolve() != output_fit.resolve()
    before = campaign.digest(input_fit)
    shutil.copy2(input_fit, output_fit)
    assert campaign.digest(input_fit) == campaign.digest(output_fit) == before
    assert campaign.digest(evidence / "frozen/trace.json") == campaign.digest(upload / "frozen/trace.json")
    invocations = []
    monkeypatch.setattr(campaign, "compute_main", lambda: invocations.append("parent"))
    entry = evidence / "upstream_first_layer_entry.py"
    for run_name in ("__mp_main__", "upstream_first_layer_entry"):
        runpy.run_path(str(entry), run_name=run_name)
        assert invocations == []
    runpy.run_path(str(entry), run_name="__main__")
    assert invocations == ["parent"]
