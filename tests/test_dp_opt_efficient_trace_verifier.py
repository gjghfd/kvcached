import hashlib
import json

import pytest

import verify_efficient_trace
import workaware_robust_adapter


def fixture(root):
    hashes = {}
    for name in verify_efficient_trace.SOURCES:
        payload = name.encode()
        (root / (name + ".executed")).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    (root / "solver-efficiency-source.json").write_text(json.dumps({"source_sha256": hashes,
        "objective_and_uncertainty_unchanged": True, "original_strict_greedy_token_checks_unchanged": True}))
    (root / "config.json").write_text(json.dumps({"workaware_optimizer": "robust"}))


@pytest.mark.parametrize("failure", [False, True])
def test_delegate_original_strict_trace_checks_and_restore_verifier_on_exit(tmp_path, monkeypatch, failure):
    fixture(tmp_path)
    original = workaware_robust_adapter.verify_raw_plan

    def delegate(root, bank):
        assert workaware_robust_adapter.verify_raw_plan is verify_efficient_trace.verify_plan
        assert root == tmp_path and bank == tmp_path / "bank"
        if failure:
            raise ValueError("strict tokens failed")
        return {"status": "raw_trace_verified"}

    monkeypatch.setattr(verify_efficient_trace, "verify_trace", delegate)
    if failure:
        with pytest.raises(ValueError, match="strict tokens"):
            verify_efficient_trace.verify(tmp_path, tmp_path / "bank")
    else:
        result = verify_efficient_trace.verify(tmp_path, tmp_path / "bank")
        assert result["original_full_serial_and_paired_token_checks_preserved"] is True
    assert workaware_robust_adapter.verify_raw_plan is original


@pytest.mark.parametrize("case", ["hash", "missing", "optimizer", "assertion"])
def test_reject_incomplete_source_or_implicit_configuration(tmp_path, case):
    fixture(tmp_path)
    if case == "hash":
        (tmp_path / (verify_efficient_trace.SOURCES[0] + ".executed")).write_bytes(b"changed")
    elif case == "optimizer":
        (tmp_path / "config.json").write_text(json.dumps({"workaware_optimizer": "point"}))
    else:
        path = tmp_path / "solver-efficiency-source.json"
        manifest = json.loads(path.read_text())
        if case == "missing":
            manifest["source_sha256"].pop(verify_efficient_trace.SOURCES[0])
        else:
            manifest["original_strict_greedy_token_checks_unchanged"] = False
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_efficient_trace.verify(tmp_path, tmp_path / "bank")
