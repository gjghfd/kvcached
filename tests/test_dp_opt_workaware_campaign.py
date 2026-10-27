import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("workaware_campaign")


def test_generated_patch_replaces_only_explicit_runtime_paths(module, tmp_path):
    repo, staging = tmp_path / "repo", tmp_path / "staging"
    for name in module.PATCH_FILES:
        before, after = repo / "experiments/dp_opt" / name, staging / "experiments/dp_opt" / name
        before.parent.mkdir(parents=True, exist_ok=True)
        after.parent.mkdir(parents=True, exist_ok=True)
        before.write_text("first\nold\nlast\n")
        after.write_text("first\nnew\nlast\n")
    patch = module.integration_patch(repo, staging)
    assert patch.startswith("*** Begin Patch\n") and patch.endswith("*** End Patch\n")
    assert patch.count("*** Update File:") == len(module.PATCH_FILES)
    assert "-old\n+new" in patch and "@@ -" not in patch


def test_fixture_has_enough_total_KV_but_no_balanced_rank_fit_and_low_code_rate_is_not_used_as_a_slo_claim(module):
    from workaware_fixture import make_fixture

    code = [{"prompt_tokens": 101371, "output_tokens": 64}]
    conv = [{"prompt_tokens": 2168, "output_tokens": 600}]
    for population in (2, 9):
        trace = make_fixture(code, conv, population, "a" * 64)
        assert len(trace["requests"]) == population + 1
        assert "not_Poisson_final_trace" in trace["sampling"]
        conv_mib = ((2168 + 8192 + 15) // 16) * 1.5
        code_mib = ((101371 + 64 + 15) // 16) * 1.5
        assert population * conv_mib + code_mib + 4 * 96 < 2 * 10240
        assert (population // 2) * conv_mib + code_mib + 2 * 96 > 10240
        assert population * conv_mib + 2 * 96 < 10240
