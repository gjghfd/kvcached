import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def verifier(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("verify_workaware_profile")


def test_numeric_payload_tolerance_does_not_relax_schema(verifier):
    before = {"profiles": {"model": {"cost": 1e-4, "count": 2, "verified": True}}}
    after = copy.deepcopy(before)
    after["profiles"]["model"]["cost"] += 1e-13
    assert verifier.same_payload(before, after)
    after["profiles"]["model"]["count"] = 2.0
    assert not verifier.same_payload(before, after)
    assert not verifier.same_payload({"verified": True}, {"verified": 1})


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -.0005))
def test_invalid_or_changed_coefficients_are_not_accepted(verifier, value):
    assert not verifier.same_payload({"attention_token_s": 1e-7}, {"attention_token_s": value})


def test_partial_profiles_are_rejected_before_any_raw_source_access(verifier, tmp_path):
    path = tmp_path / "fit.json"
    path.write_text(json.dumps({"population_profiles": {"status": "partial_diagnostic_not_verified_not_for_runtime"}}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Diagnostic"):
        verifier.load_verified_profile(path, digest, tmp_path / "missing_raw")


def test_profile_sha_mismatch_cannot_be_bypassed(verifier, tmp_path):
    path = tmp_path / "fit.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="frozen SHA"):
        verifier.load_verified_profile(path, "0" * 64, tmp_path)


def test_source_changes_and_fitted_cost_changes_are_rejected(verifier, monkeypatch, tmp_path):
    source = tmp_path / "trials.json"
    source.write_text("[]")
    expected = {"source_files_sha256": {"trials.json": hashlib.sha256(source.read_bytes()).hexdigest()},
                "population_profiles": {"status": "raw_verified_workaware_population_profiles", "cost": .006}}
    monkeypatch.setattr(verifier, "SOURCE_FILES", ("trials.json",))
    monkeypatch.setattr(verifier, "verify_profile", lambda root: copy.deepcopy(expected))
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(expected))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verifier.load_verified_profile(path, digest, tmp_path) == expected
    altered = copy.deepcopy(expected)
    altered["population_profiles"]["cost"] = .007
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="reproduce"):
        verifier.load_verified_profile(path, hashlib.sha256(path.read_bytes()).hexdigest(), tmp_path)
    path.write_text(json.dumps(expected))
    source.write_text("[1]")
    with pytest.raises(ValueError, match="source changed"):
        verifier.load_verified_profile(path, hashlib.sha256(path.read_bytes()).hexdigest(), tmp_path)
