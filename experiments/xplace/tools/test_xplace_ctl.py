import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xplace_ctl


@pytest.fixture
def staging(tmp_path, monkeypatch):
    monkeypatch.setattr(xplace_ctl, "STAGING", tmp_path)
    root = tmp_path / "test-tag"
    root.mkdir()
    (root / "upload.tar.gz").write_bytes(b"frozen-tar")
    (root / "build.json").write_text(json.dumps({
        "tarball_sha256": xplace_ctl.sha256(root / "upload.tar.gz"), "manifest_sha256": "manifest-hash"}))
    calls = []
    def remote(script, **kwargs):
        calls.append(script)
        return "VERIFIED\n"
    monkeypatch.setattr(xplace_ctl, "remote", remote)
    return root, calls


def test_atomic_upload_verifies_before_publish(staging, monkeypatch):
    root, calls = staging
    transfers = []
    def transfer(command, stdin, **kwargs):
        assert stdin.read() == b"frozen-tar"
        transfers.append(command)
    monkeypatch.setattr(subprocess, "run", transfer)
    result = xplace_ctl.upload("test-tag")
    assert result["uploaded_at"]
    assert result["upload_transport"] == "ssh_stdin_verified_atomic_publish"
    assert transfers[0][:len(xplace_ctl.SSH)] == xplace_ctl.SSH
    assert "upload.tar.gz.part-" in transfers[0][-1]
    finalize = next(script for script in calls if "tar xzf" in script)
    assert finalize.index("sha256sum -c") < finalize.index("mv upload.tar.gz")
    assert "test ! -e" in calls[1] and '"job-*"' in calls[1]
    assert calls[-1].startswith("rmdir")


def test_failed_transfer_preserves_unpublished_state(staging, monkeypatch):
    root, calls = staging
    def transfer(*args, **kwargs):
        raise subprocess.CalledProcessError(255, "ssh")
    monkeypatch.setattr(subprocess, "run", transfer)
    with pytest.raises(subprocess.CalledProcessError):
        xplace_ctl.upload("test-tag", resume=True)
    assert "uploaded_at" not in json.loads((root / "build.json").read_text())
    assert not any("tar xzf" in script for script in calls)
    assert calls[-1].startswith("rmdir")
    assert not any("remote tag exists" in script for script in calls)


def test_local_tar_drift_fails_before_network(staging):
    root, calls = staging
    (root / "upload.tar.gz").write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA"):
        xplace_ctl.upload("test-tag")
    assert not calls
