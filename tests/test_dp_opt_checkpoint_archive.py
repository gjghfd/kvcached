import hashlib
import importlib
import io
import tarfile
from pathlib import Path

import pytest


@pytest.mark.parametrize("exit_marker", [None, b"0\n", b"143\n"])
@pytest.mark.parametrize("authorized", [False, True])
def test_dangling_cancelled_pointer_never_fabricates_exit(tmp_path, monkeypatch, exit_marker, authorized):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    module = importlib.import_module("checkpoint_archive")
    canonical = module.archive_trace_job
    bundle = tmp_path / "raw.tar.gz"
    files = {"existing.json": b"{}"}
    if exit_marker is not None:
        files["exit-status"] = exit_marker
    with tarfile.open(bundle, "w:gz") as archive:
        for name, payload in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        pointer = tarfile.TarInfo("progress.json")
        pointer.type = tarfile.SYMTYPE
        pointer.linkname = "/data/home/test/run/kvcached-paracloud/state/live-trace-278514/native-N3/progress.json"
        archive.addfile(pointer)
    manifest = "".join(hashlib.sha256(payload).hexdigest() + "  " + name + "\n" for name, payload in files.items())
    staging = tmp_path / "extract"
    staging.mkdir()
    extractor = module.cancelled_extractor(canonical.extract_raw, 278514)
    if not authorized:
        with pytest.raises(ValueError, match="archived regular target"):
            extractor(bundle, staging, manifest, 278514, allow_missing_failed_progress=False)
        return
    pointers = extractor(bundle, staging, manifest, 278514, allow_missing_failed_progress=True)
    assert pointers[0]["target_present"] is False
    assert not (staging / "progress.json").exists()
    assert (staging / "exit-status").exists() is (exit_marker is not None)
    canonical.verify_raw(staging, manifest)
