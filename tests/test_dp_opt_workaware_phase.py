import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return tuple(importlib.import_module(name) for name in (
        "workaware_calibration_plan", "workaware_pilot", "workaware_profile_bundle",
        "workaware_evidence", "workaware_profile_cost", "analyze_workaware_cost", "workaware_phase_campaign"))


def test_supplement_covers_every_partition_both_tenants_and_preserves_legacy(modules):
    plan, pilot, *_ = modules
    selected = plan.supplemental_plan()
    cases = pilot.calibration_cases(selected)
    assert cases == pilot.calibration_cases(selected) and len(cases) == 252
    assert pilot.calibration_cases() == pilot.calibration_cases(None)
    assert {case["population"] for case in pilot.calibration_cases()} == {2, 3, 4, 8, 9}
    for model in (0, 1):
        for repeat in (0, 1):
            for entry in selected["populations"]:
                for length in entry["prompt_tokens"]:
                    assert {case["rank_zero"] for case in cases if case["model"] == model
                            and case["repeat"] == repeat and case["population"] == entry["population"]
                            and case["prompt_tokens"] == length} == set(range(entry["population"] + 1))
    assert plan.screen_plan(selected)["maximum_mapped_mib"] == 9984


def test_capacity_screen_includes_both_pinned_pools_and_rejects_old_N7_candidate(modules):
    plan = modules[0]
    selected = plan.supplemental_plan()
    for population, prompt, expected in ((5, 16384, 9312), (6, 14336, 9984), (7, 11264, 9600)):
        assert plan.conservative_mapped_mib({"population": population, "rank_zero": population,
                                           "prompt_tokens": prompt}) == [expected, 192]
    selected["populations"][-1]["prompt_tokens"][-1] = 12288
    with pytest.raises(ValueError, match="pinned pools"):
        plan.screen_plan(selected)


@pytest.mark.parametrize("field,value", (("models", [1]), ("repeats", [0]), ("output_tokens", 1024),
                                       ("window_tokens", [1, 100]), ("budget_mib", 11328)))
def test_plan_cannot_weaken_fixed_controls(modules, field, value):
    plan = modules[0]
    selected = plan.supplemental_plan()
    selected[field] = value
    with pytest.raises(ValueError):
        plan.screen_plan(selected)


@pytest.mark.parametrize("name", ("../external", "/absolute", ".", None))
def test_composite_paths_cannot_escape_archive(modules, name):
    with pytest.raises(ValueError, match="inside"):
        modules[2].relative_path(name)


def test_composite_independently_preserves_old_profiles_and_rejects_overlap(modules, tmp_path):
    bundle = modules[2]
    fits = {}
    for name, population in (("legacy", 2), ("supplement", 5)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "raw.json").write_text(name)
        (directory / "workaware-runtime-source.json").write_text('{"sources":{}}')
        hashes = {filename: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
                  for filename in ("raw.json", "workaware-runtime-source.json")}
        fits[name] = {"max_population": population, "max_context_tokens": 10000,
            "max_attention_tokens": 50000, "cases": 10, "requests": 20, "source_files_sha256": hashes,
            "population_profiles": {"status": "raw_verified_workaware_population_profiles",
                                    "profiles": {f"0:{population}": {"cost": population}}}}
    manifest = {"schema_version": 1, "sources": ["legacy", "supplement"], "required_profiles": ["0:2", "0:5"]}
    (tmp_path / bundle.MANIFEST).write_text(json.dumps(manifest))
    verified = []

    def verify(directory):
        verified.append(directory.name)
        return copy.deepcopy(fits[directory.name])

    result = bundle.verify_composite(tmp_path, verify)
    assert verified == ["legacy", "supplement"]
    assert result["population_profiles"]["profiles"]["0:2"] == fits["legacy"]["population_profiles"]["profiles"]["0:2"]
    assert "legacy/raw.json" in result["source_files_sha256"]
    assert "supplement/raw.json" in result["source_files_sha256"]
    bundle.archive_sources(tmp_path, tmp_path / "archive", result)
    assert (tmp_path / "archive/supplement/raw.json").read_text() == "supplement"
    fits["supplement"]["population_profiles"]["profiles"] = {"0:2": {"cost": 100}}
    with pytest.raises(ValueError, match="overwrite"):
        bundle.verify_composite(tmp_path, verify)


@pytest.mark.parametrize("random_baseline", (False, True))
def test_phase_runtime_versions_are_exact_and_raw_verified(modules, monkeypatch, tmp_path, random_baseline):
    evidence = modules[3]
    monkeypatch.setenv("DP_OPT_WORKAWARE_PHASE_PROFILES", "1")
    monkeypatch.setenv("DP_OPT_RANDOM_BASELINE", "1" if random_baseline else "0")
    evidence.archive_runtime(tmp_path)
    manifest = json.loads((tmp_path / "workaware-runtime-source.json").read_text())
    (tmp_path / "source.sha256").write_text("\n".join(
        source["sha256"] + "  " + source["path"] for source in manifest["sources"].values()))
    assert evidence.verify_runtime(tmp_path) == manifest
    expected = evidence.PHASE_RANDOM_RUNTIME_FILES if random_baseline else evidence.PHASE_RUNTIME_FILES
    assert set(manifest["sources"]) == set(expected)


def test_supplement_validates_heldout_repeats_contexts_and_layouts(modules):
    plan, pilot, _, _, costs, analysis, _ = modules
    trials = []
    for case in plan.planned_cases(plan.supplemental_plan()):
        counts = [case["rank_zero"], case["population"] - case["rank_zero"]]
        contexts = [count * (case["prompt_tokens"] + 2047) for count in counts]
        graph = costs.graph_padding(counts, pilot.SPARSE_CAPTURE_SIZES)
        sample = {"counts": counts, "context_tokens": contexts,
                  "cadence_s": .01 + graph * .001 + max(contexts) * 1e-7}
        trials.append({"case": case, "sample": sample})
    contrasts = analysis.paired_contrasts(trials, {"capture_sizes": pilot.SPARSE_CAPTURE_SIZES, "residual_bound_s": 0,
        "coefficients": {"graph_batch_s": .001, "attention_token_s": 1e-7}})
    profiles = costs.fit_profiles(contrasts["contrasts"], extended_validation=True)
    assert len(profiles["profiles"]) == 6
    for profile in profiles["profiles"].values():
        assert len(profile["heldout_contexts"]) == 3
        assert len(profile["heldout_layouts"]) >= 4
        assert profile["attention_token_s"] == pytest.approx(1e-7)
        assert "context_layout" in profile["uncertainty_scope"]


def test_submission_preserves_one_allocation_origin_and_paused_transfer(modules):
    campaign = modules[-1]
    command = campaign.submission_command({"DP_OPT_WORKAWARE_CONV_RATE": ".03"}, "/stage/ledger")
    assert command.count("sbatch --parsable") == 1
    assert 'test -z "$(squeue' in command and "test ! -e /stage/ledger" in command
    assert "git@github.com:gjghfd/kvcached.git" in command and "workaware-phase-return.sbatch" in command
    script = (Path(campaign.__file__).parent / "workaware-phase-return.sbatch").read_text()
    assert "DP_OPT_PAUSE_SYNC_INTERVAL=32" in script and "DP_OPT_KV_TRANSPORT=dp_echo" in script
    assert "phase calibration-n567" in script and "phase cache-fixed-original" in script
    assert "phase lower-code-rate" in script and "0.00025 $DP_OPT_WORKAWARE_CONV_RATE" in script
