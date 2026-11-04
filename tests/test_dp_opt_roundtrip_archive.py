import hashlib
import io
import tarfile

import pytest

from archive_roundtrip_job import extract_control_raw


def fixture(root, target="state-control/pilot/state-control-progress.json", failed=False, present=True):
    bundle = root / "raw.tar.gz"
    files = {"exit-status": b"1\n" if failed else b"0\n"}
    if present:
        files["state-control/pilot/state-control-progress.json"] = b'{"trials":[]}\n'
    with tarfile.open(bundle, "w:gz") as output:
        for name, payload in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            output.addfile(member, io.BytesIO(payload))
        member = tarfile.TarInfo("progress.json")
        member.type = tarfile.SYMTYPE
        member.linkname = target
        output.addfile(member)
    staging = root / "extracted"
    staging.mkdir()
    manifest = "".join(f"{hashlib.sha256(payload).hexdigest()}  ./{name}\n" for name, payload in files.items())
    return bundle, staging, manifest


@pytest.mark.parametrize("absolute", [False, True])
def test_exact_declared_nested_pointer_is_recorded_without_extracting_a_link(tmp_path, absolute):
    target = "state-control/pilot/state-control-progress.json"
    if absolute:
        target = "/data/home/user/run/kvcached-paracloud/state/live-trace-273701/" + target
    bundle, staging, manifest = fixture(tmp_path, target=target)
    receipts = extract_control_raw(bundle, staging, manifest, 273701)
    assert receipts[0]["target_present"] is True
    assert not (staging / "progress.json").exists()


@pytest.mark.parametrize("target", ["../../escape", "state-control/pilot/../../escape",
    "unrelated/progress.json", "/tmp/progress.json",
    "/data/home/user/run/kvcached-paracloud/state/live-trace-273702/plain-no-move/progress.json"])
def test_reject_traversal_unknown_phase_and_cross_job_pointer(tmp_path, target):
    bundle, staging, manifest = fixture(tmp_path, target=target)
    with pytest.raises(ValueError):
        extract_control_raw(bundle, staging, manifest, 273701)


def test_absent_failed_target_is_not_fabricated(tmp_path):
    bundle, staging, manifest = fixture(tmp_path, failed=True, present=False)
    receipts = extract_control_raw(bundle, staging, manifest, 273701, allow_missing_failed_progress=True)
    assert receipts[0]["target_present"] is False
    assert not (staging / "progress.json").exists()


def test_successful_target_must_exist_in_regular_SHA_inventory(tmp_path):
    bundle, staging, manifest = fixture(tmp_path, present=False)
    with pytest.raises(ValueError, match="lacks"):
        extract_control_raw(bundle, staging, manifest, 273701)
