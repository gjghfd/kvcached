from pathlib import Path

import pytest

import roundtrip_state_probe


@pytest.mark.parametrize("transport", [None, "host_file", "dp_allgatherv"])
def test_reject_wrong_or_implicit_transport_before_model_initialization(monkeypatch, transport):
    if transport is None:
        monkeypatch.delenv("DP_OPT_KV_TRANSPORT", raising=False)
    else:
        monkeypatch.setenv("DP_OPT_KV_TRANSPORT", transport)
    monkeypatch.setattr("sys.argv", ["roundtrip_state_probe.py", "--pilot"])
    calls = []
    monkeypatch.setattr(roundtrip_state_probe.live_trace, "main", lambda: calls.append("engine"))
    with pytest.raises(ValueError, match="before engine initialization"):
        roundtrip_state_probe.main()
    assert not calls


def test_explicit_device_echo_transport_is_accepted():
    roundtrip_state_probe.validate_environment({"DP_OPT_KV_TRANSPORT": "dp_echo"})


def test_control_launch_declares_transport_and_canonical_absolute_progress_targets():
    path = Path(__file__).resolve().parents[1] / "experiments/dp_opt/roundtrip-state-probe.sbatch"
    text = path.read_text()
    assert "export DP_OPT_KV_TRANSPORT=dp_echo DP_OPT_TRANSFER_CHUNK_MIB=32" in text
    assert 'ln -s "$evidence/pilot/progress.json" "$evidence/progress.json"' in text
    assert 'ln -sfn "$evidence/plain-no-move/progress.json" "$evidence/progress.json"' in text
