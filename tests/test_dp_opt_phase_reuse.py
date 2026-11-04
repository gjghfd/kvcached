import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def phase(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("workaware_phase_campaign")


def test_reuse_source_requires_successful_phase_and_exact_frozen_hashes(phase, monkeypatch, tmp_path):
    (tmp_path / "exit-status").write_text("0\n")
    hashes = {"workaware-trials.json": "a" * 64}
    monkeypatch.setattr(phase, "reuse_source_files", lambda source: hashes)
    config = {"reuse_source_root": str(tmp_path), "reuse_source_files_sha256": hashes}
    assert phase.validate_reuse_source(tmp_path, config)["files"] == 1
    with pytest.raises(ValueError, match="frozen successful"):
        phase.validate_reuse_source(tmp_path, {**config, "reuse_source_root": "/other"})
    with pytest.raises(ValueError, match="frozen successful"):
        phase.validate_reuse_source(tmp_path, {**config, "reuse_source_files_sha256": {}})
    (tmp_path / "exit-status").write_text("1\n")
    with pytest.raises(ValueError, match="frozen successful"):
        phase.validate_reuse_source(tmp_path, config)


def test_reuse_composite_requires_full_independent_reproduction(phase, monkeypatch, tmp_path):
    expected = {"coefficients": {"attention_token_s": .000001}}
    monkeypatch.setattr(phase, "verify_profile", lambda root: expected)
    assert phase.validate_reused_profile(tmp_path, {"reuse_fit_data": expected})["status"].startswith("reused_composite")
    with pytest.raises(ValueError, match="independent local"):
        phase.validate_reused_profile(tmp_path, {"reuse_fit_data": {"coefficients": {"attention_token_s": .1}}})


def test_fresh_and_reused_launches_both_anchor_supplement_and_preserve_trace_order(phase):
    text = Path(phase.__file__).with_name("workaware-phase-return.sbatch").read_text()
    assert '--anchor-supplement' in text and '--validate-reuse-source "$reuse"' in text
    assert '--validate-reused-profile "$evidence/composite-profile"' in text
    assert text.index("phase cache-fixed-original") < text.index("phase lower-code-rate")
    assert '"$evidence/composite-profile/supplement" --output' in text
    assert 'DP_OPT_REQUEST_TIMEOUT_S=1800' in text and 'DP_OPT_DEADLINE_S=120' in text
    assert 'DP_OPT_KV_TRANSPORT=dp_echo' in text and 'DP_OPT_PAUSE_SYNC_INTERVAL=32' in text
