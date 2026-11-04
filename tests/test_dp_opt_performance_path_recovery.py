import importlib
import json
from contextlib import contextmanager
from pathlib import Path

import pytest


@pytest.fixture
def recovery(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("performance_path_recovery")


@pytest.fixture
def inventory(recovery, tmp_path):
    phase = tmp_path / "phase"
    runtime = tmp_path / "runtime"
    (phase / "runtime-source").mkdir(parents=True)
    (runtime / "experiments/dp_opt").mkdir(parents=True)
    payload = b"immutable runtime\n"
    (phase / "runtime-source/worker.py").write_bytes(payload)
    (runtime / "experiments/dp_opt/worker.py").write_bytes(payload)
    checksum = recovery.digest(phase / "runtime-source/worker.py")
    alias = "/home/test/run/runtime"
    canonical = "/data/test/runtime"
    relative = "experiments/dp_opt/worker.py"
    (phase / "runtime-source.json").write_text(json.dumps({"sources": {
        "worker.py": {"sha256": checksum, "path": canonical + "/" + relative}}}))
    (phase / "source.sha256").write_text((checksum + "  " + alias + "/" + relative + "\n") * 2)
    return phase, runtime, {relative: checksum}, alias, canonical


def test_duplicate_identical_aliases_preserve_inventory_and_bytes(recovery, inventory):
    phase, runtime, expected, alias, canonical = inventory
    before = recovery.digest(phase / "source.sha256")
    normalized, aliases = recovery.normalized_inventory(*inventory)
    assert normalized == (next(iter(expected.values())) + "  " + canonical + "/" + next(iter(expected)) + "\n") * 2
    assert aliases == {alias + "/" + next(iter(expected)): canonical + "/" + next(iter(expected))}
    assert recovery.digest(phase / "source.sha256") == before


@pytest.mark.parametrize("change", ["raw_bytes", "archived_bytes", "conflict", "canonical_conflict", "traversal", "manifest", "bad_sha"])
def test_alias_identity_never_uses_basename_or_ignores_corruption(recovery, inventory, change):
    phase, runtime, expected, alias, canonical = inventory
    source = phase / "source.sha256"
    if change == "raw_bytes":
        (runtime / next(iter(expected))).write_bytes(b"changed")
    elif change == "archived_bytes":
        (phase / "runtime-source/worker.py").write_bytes(b"changed")
    elif change == "conflict":
        source.write_text(source.read_text() + "0" * 64 + "  " + alias + "/" + next(iter(expected)) + "\n")
    elif change == "canonical_conflict":
        source.write_text(source.read_text() + "0" * 64 + "  " + canonical + "/" + next(iter(expected)) + "\n")
    elif change == "traversal":
        source.write_text(source.read_text().replace("experiments/dp_opt/worker.py", "../worker.py"))
    elif change == "manifest":
        manifest = json.loads((phase / "runtime-source.json").read_text())
        manifest["sources"]["worker.py"]["path"] = "/different/path/worker.py"
        (phase / "runtime-source.json").write_text(json.dumps(manifest))
    else:
        source.write_text("wrong  " + alias + "/" + next(iter(expected)) + "\n")
    with pytest.raises(ValueError):
        recovery.normalized_inventory(*inventory)


@pytest.mark.parametrize("alias,canonical", [("relative", "/absolute"), ("/same", "/same"), ("/a/../b", "/c")])
def test_absolute_distinct_nontraversing_roots_required(recovery, inventory, alias, canonical):
    with pytest.raises(ValueError):
        recovery.normalized_inventory(*inventory[:3], alias, canonical)


def test_stage_adapter_preserves_nonterminal_invariants(recovery):
    amended = recovery.alias_stage_function(recovery.followup.verify_stage, lambda root, entry: False)
    with pytest.raises(ValueError, match="not reclassified"):
        amended(Path("/unused"), {"stage_id": "native", "job": 1})


def test_stage_adapter_rejects_an_unknown_validator(recovery):
    with pytest.raises(ValueError, match="exactly two"):
        recovery.alias_stage_function(recovery.digest, lambda root, entry: True)


def test_low_level_alias_hook_covers_all_verification_callers(recovery, inventory, tmp_path, monkeypatch):
    phase = inventory[0]
    calls = []

    def strict_runtime(actual, headroom=False):
        calls.append((Path(actual), headroom))
        assert (Path(actual) / "source.sha256").read_text() == "canonical inventory\n"
        return "strictly-verified"

    monkeypatch.setattr(recovery.verify_trace, "verify_amortized_runtime", strict_runtime)
    monkeypatch.setattr(recovery, "audit_binding", lambda root, state, actual: ("canonical inventory\n", {}))
    original_stage = recovery.followup.verify_stage
    monkeypatch.setattr(recovery.followup, "verify_stage", original_stage)
    recovery.install_verifier(tmp_path, {"stages": {}})
    assert recovery.verify_trace.verify_amortized_runtime(phase, True) == "strictly-verified"
    assert calls[0][1] is True
    assert not calls[0][0].exists()


@pytest.mark.parametrize("job,stage_id", [(2, "native"), (1, "other"), (True, "native")])
def test_alias_receipt_requires_exact_caller_identity(recovery, job, stage_id):
    with pytest.raises(ValueError, match="caller job/stage"):
        recovery.verify_caller_binding({"job": job, "stage_id": stage_id}, {"job": 1, "stage_id": "native"})


@pytest.mark.parametrize("keys", [[], ["out"], ["err"], ["out", "err", "extra"]])
def test_terminal_log_sha_inventory_requires_both_logs(recovery, tmp_path, keys):
    archive = {"job": 1, "missing_job_logs": [], "log_sha256": {name: "0" * 64 for name in keys}}
    with pytest.raises(ValueError, match="both original"):
        recovery.verify_terminal_logs(tmp_path, 1, archive)


def test_terminal_log_digest_is_independently_checked(recovery, tmp_path):
    hashes = {}
    for suffix in ("out", "err"):
        path = tmp_path / f"live-trace-1.{suffix}"
        path.write_text("original " + suffix)
        hashes[suffix] = recovery.digest(path)
    archive = {"job": 1, "missing_job_logs": [], "log_sha256": hashes}
    assert recovery.verify_terminal_logs(tmp_path, 1, archive) == hashes
    (tmp_path / "live-trace-1.err").write_text("changed")
    with pytest.raises(ValueError, match="log SHA changed"):
        recovery.verify_terminal_logs(tmp_path, 1, archive)


def test_loaded_host_manifest_binds_exact_source_set(recovery, tmp_path):
    names = ("experiments/dp_opt/performance_path_recovery.py", "experiments/dp_opt/risk_certificate_portability.py")
    source = tmp_path / names[0]
    source.parent.mkdir(parents=True)
    source.write_bytes(Path(recovery.__file__).read_bytes())
    risk = tmp_path / names[1]
    risk.write_text("risk source\n")
    manifest = tmp_path / "host-manifest.json"
    manifest.write_text(json.dumps({name: recovery.digest(tmp_path / name) for name in names}))
    assert recovery.verify_host_manifest(manifest, tmp_path) == recovery.digest(manifest)
    assert recovery.verify_host_manifest(manifest, tmp_path, recovery.digest(risk)) == recovery.digest(manifest)
    with pytest.raises(ValueError, match="loaded risk source"):
        recovery.verify_host_manifest(manifest, tmp_path, "0" * 64)
    risk.write_text("changed\n")
    with pytest.raises(ValueError):
        recovery.verify_host_manifest(manifest, tmp_path)
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="exact independently"):
        recovery.verify_host_manifest(manifest, tmp_path)


def test_risk_stage_scope_binds_ledger_raw_logs_and_host_manifest(recovery, tmp_path, monkeypatch):
    entry = {"job": 1, "stage_id": "native", "plan": {"phase_name": "native", "trace_sha256": "a" * 64}}
    (tmp_path / "campaign-state.json").write_text(json.dumps({"stages": {"native": entry}}))
    directory = tmp_path / "stages/native"
    directory.mkdir(parents=True)
    logs = {}
    for suffix in ("out", "err"):
        path = directory / f"live-trace-1.{suffix}"
        path.write_text(suffix)
        logs[suffix] = recovery.digest(path)
    (directory / "archive-1.json").write_text(json.dumps({"job": 1, "missing_job_logs": [], "log_sha256": logs}))
    (directory / "raw-1.sha256").write_text("original raw manifest\n")
    scopes = []

    class Portability:
        @staticmethod
        @contextmanager
        def verification_scope(binding):
            scopes.append(binding)
            yield

    monkeypatch.setattr(recovery.followup, "verify_stage", lambda root, stage: "strict result")
    recovery.bind_risk_stage_verification(tmp_path, Portability, "b" * 64)
    assert recovery.followup.verify_stage(tmp_path, entry) == "strict result"
    assert scopes[0]["job"] == 1 and scopes[0]["root"] == str(tmp_path.resolve())
    assert scopes[0]["log_sha256"] == logs
    assert scopes[0]["host_source_manifest_sha256"] == "b" * 64
    with pytest.raises(ValueError, match="root differs"):
        recovery.followup.verify_stage(tmp_path / "other", entry)
    with pytest.raises(ValueError, match="stage ledger"):
        recovery.followup.verify_stage(tmp_path, {**entry, "job": 2})
