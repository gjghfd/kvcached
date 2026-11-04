import hashlib
import importlib
import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def observer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("closed_phase_observer")


@pytest.fixture
def remote(tmp_path):
    home = tmp_path / "home"
    root = home / "run/kvcached-paracloud/state/live-trace-123"
    stage = home / "run/kvcached-paracloud/sync/closed-phases-123-test"
    root.mkdir(parents=True)
    stage.mkdir(parents=True)
    tools = tmp_path / "bin"
    tools.mkdir()
    command = tools / "sacct"
    command.write_text("#!/bin/sh\nprintf '123|COMPLETED|0:0\\n'\n")
    command.chmod(0o755)
    environment = {**os.environ, "HOME": str(home), "PATH": str(tools) + ":" + os.environ["PATH"]}
    script = Path(__file__).resolve().parents[1] / "experiments/dp_opt/archive-closed-phases.sh"
    return root, stage, lambda: subprocess.run(["bash", str(script), "123", str(stage)], env=environment,
                                              text=True, capture_output=True, timeout=10)


def test_metadata_validates_phase_attempt_and_digest(observer):
    digest = "a" * 64
    records = observer.ready_records("calibration-n567\tphase-calibration-n567.Ab12\t" + digest)
    assert records["calibration-n567"]["bundle_sha256"] == digest


@pytest.mark.parametrize("text", ("unknown\tphase-unknown.AB\t" + "a" * 64,
    "calibration-n567\t../../outside\t" + "a" * 64,
    "calibration-n567\tphase-calibration-n567.AB\tinvalid",
    "calibration-n567\tphase-calibration-n567.AB\t" + "a" * 64 + "\n" +
    "calibration-n567\tphase-calibration-n567.AB\t" + "a" * 64))
def test_unsafe_or_duplicate_metadata_is_rejected(observer, text):
    with pytest.raises(ValueError):
        observer.ready_records(text)


def test_closed_phase_has_matching_manifests_and_bundle(remote):
    root, stage, run = remote
    source = root / "calibration-n567"
    source.mkdir()
    (source / "exit-status").write_text("0\n")
    (source / "evidence.json").write_text('{"real_raw":true}\n')
    result = run()
    assert result.returncode == 0, result.stderr
    fields = dict(line.split("=", 1) for line in (stage / "calibration-n567.ready").read_text().splitlines())
    archived = stage / fields["attempt"]
    assert (archived / "raw-before.sha256").read_bytes() == (archived / "raw-after.sha256").read_bytes()
    assert hashlib.sha256((archived / "raw.tar.gz").read_bytes()).hexdigest() == fields["bundle_sha256"]
    assert not (stage / "lower-code-rate.ready").exists()


def test_active_phase_is_not_archived(remote):
    root, stage, run = remote
    (root / "calibration-n567").mkdir()
    assert run().returncode == 0
    assert not (stage / "calibration-n567.ready").exists()


def test_composite_requires_transition_not_only_partially_written_fit(remote):
    root, stage, run = remote
    source = root / "composite-profile"
    source.mkdir()
    (source / "fit.json").write_text("{}")
    (source / "workaware-profile-sources.json").write_text("{}")
    (root / "phase.json").write_text('{"phase":"calibration-n567"}')
    assert run().returncode == 0
    assert not (stage / "composite-profile.ready").exists()


def test_failed_closed_phase_raw_is_preserved_not_claimed_valid(remote):
    root, stage, run = remote
    source = root / "calibration-n567"
    source.mkdir()
    (source / "exit-status").write_text("1\n")
    (source / "error.json").write_text('{"failed":true}')
    assert run().returncode == 0
    assert (stage / "calibration-n567.ready").exists()


def test_symbolic_raw_is_rejected_without_archiving(remote):
    root, stage, run = remote
    source = root / "calibration-n567"
    source.mkdir()
    (source / "exit-status").write_text("0\n")
    (source / "outside-link").symlink_to(root / "phase.json")
    assert run().returncode != 0
    assert not (stage / "calibration-n567.ready").exists()


def test_remote_watcher_cannot_restart_same_stage(remote):
    _, _, run = remote
    assert run().returncode == 0
    assert run().returncode == 2
