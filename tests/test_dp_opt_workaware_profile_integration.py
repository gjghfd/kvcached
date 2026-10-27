import hashlib
import importlib
import json
from pathlib import Path

import pytest

from test_dp_opt_workaware_profile_cost import contrast_rows, forecasting_fixture


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return tuple(importlib.import_module(name) for name in (
        "workaware_cost", "workaware_profile_cost", "workaware_pilot", "verify_workaware_profile", "workaware_evidence"))


def test_verified_profile_dispatch_and_completion_segments(modules):
    cost, profile, _, _, _ = modules
    observed, fit = forecasting_fixture()
    fit["population_profiles"] = profile.fit_profiles(contrast_rows())
    fit["population_profiles"]["status"] = "raw_verified_workaware_population_profiles"
    result = cost.remaining_segments(observed, fit)
    assert [segment["population"] for segment in result] == [3, 2, 1]
    assert result[1]["graph_costs_s"] == pytest.approx([.012, 0, .012])
    assert result[-1]["uncertainty_request_s"] == 0
    fit["population_profiles"]["status"] = "diagnostic"
    with pytest.raises(ValueError, match="raw-verified"):
        cost.remaining_segments(observed, fit)


def test_serving_loader_independently_rebuilds_profiles(modules, monkeypatch, tmp_path):
    _, _, pilot, verifier, _ = modules
    expected = {"source_files_sha256": {}, "population_profiles": {"status": "raw_verified_workaware_population_profiles"}}
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(expected))
    monkeypatch.setattr(verifier, "SOURCE_FILES", ())
    monkeypatch.setattr(verifier, "verify_profile", lambda root: expected)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert pilot.load_verified_fit(path, digest, tmp_path) == expected


@pytest.mark.parametrize("profile_mode", (False, True))
def test_legacy_calibration_and_new_runtime_manifests_are_both_exact(modules, monkeypatch, tmp_path, profile_mode):
    _, _, _, _, evidence = modules
    monkeypatch.setenv("DP_OPT_WORKAWARE_PROFILES", "1" if profile_mode else "0")
    evidence.archive_runtime(tmp_path)
    manifest = json.loads((tmp_path / "workaware-runtime-source.json").read_text())
    inventory = "\n".join(source["sha256"] + "  " + source["path"] for source in manifest["sources"].values())
    (tmp_path / "source.sha256").write_text(inventory + "\n")
    assert evidence.verify_runtime(tmp_path) == manifest
    assert set(manifest["sources"]) == set(evidence.PROFILE_RUNTIME_FILES if profile_mode else evidence.RUNTIME_FILES)
    manifest["algorithm_version"] = "unknown"
    (tmp_path / "workaware-runtime-source.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unknown"):
        evidence.verify_runtime(tmp_path)
