import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("workaware_pressure_campaign")


def test_pressure_candidate_uses_dynamic_pages_not_ratio_scaling(module):
    pressure = {"idle_pool_mib": [[96, 96], [96, 96]],
        "Conv_physical_mean_mib_per_gpu": 1739.2806284373632,
        "budget_mib_per_gpu": 10240, "Code_single_physical_pool_mib": 9600}
    old_rate = .01956673845194241
    result = module.rate_for_ratio(pressure, old_rate, 1.2, maximum_rate=2 * old_rate)
    assert result["target_Conv_physical_mib_per_gpu"] == 2688
    assert result["Conv_rate_rps_candidate"] == pytest.approx(.030863253171591748)
    assert result["Conv_rate_fraction"] > 1.5


def test_pressure_sync_includes_new_archive_barrier_and_same_profile_runtime(module):
    assert len(module.SYNC_FILES) == len(set(module.SYNC_FILES))
    assert set(module.PROFILE_SYNC_FILES) <= set(module.SYNC_FILES)
    assert "experiments/dp_opt/watch-trace-remote.sh" in module.SYNC_FILES
    assert "experiments/dp_opt/archive_trace_job.py" in module.SYNC_FILES


def test_wrapper_skip_is_explicit_and_default_recovers_both_populations():
    wrapper = (Path(__file__).resolve().parents[1] / "experiments/dp_opt/workaware-profile-return.sbatch").read_text()
    assert "recovery_populations=(2 9)" in wrapper
    assert "DP_OPT_WORKAWARE_SKIP_RECOVERY:-0" in wrapper
    assert 'DP_OPT_TRACE_POLICIES="no_move no_move_fair kv_only kv_only_headroom kv_only_workaware"' in wrapper


def test_retarget_keeps_Code_and_Conv_mark_prefix_without_selected_arrivals(module, tmp_path):
    for filename, data in (("prompt-bank.json", {}),
            ("code-marks.json", [{"prompt_tokens": 101371, "output_tokens": 64, "prompt_key": "code"}]),
            ("conv-marks.json", [{"prompt_tokens": 2056, "output_tokens": 512}])):
        (tmp_path / filename).write_text(json.dumps(data))
    old_rate, new_rate = .01956673845194241, .030863253171591748
    old = module.make_stress_trace(json.loads((tmp_path / "code-marks.json").read_text()),
        json.loads((tmp_path / "conv-marks.json").read_text()), 8192, .0005,
        old_rate, 4200, 600, 20261024, module.digest(tmp_path / "prompt-bank.json"))
    new = module.retarget_trace(old, tmp_path, new_rate)
    assert [mark for mark in old["requests"] if mark["model"] == 0] == [
        mark for mark in new["requests"] if mark["model"] == 0]
    old_conv = [mark for mark in old["requests"] if mark["model"] == 1]
    new_conv = [mark for mark in new["requests"] if mark["model"] == 1]
    assert len(new_conv) > len(old_conv)
    for before, after in zip(old_conv, new_conv):
        assert after["arrival_s"] == pytest.approx(before["arrival_s"] * old_rate / new_rate)
        assert before["request_id"] == after["request_id"]
    with pytest.raises(ValueError, match="preserve every old Conv"):
        module.retarget_trace(old, tmp_path, old_rate / 2)
    (tmp_path / "prompt-bank.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="cannot be rebuilt"):
        module.retarget_trace(old, tmp_path, new_rate)
