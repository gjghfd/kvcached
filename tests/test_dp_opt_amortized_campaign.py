import importlib
from pathlib import Path

import pytest


@pytest.fixture
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("amortized_campaign")


def test_deployment_preserves_all_baselines_and_frozen_costs(campaign):
    config = {"max_model_len": 131072, "cache_bytes": 25769803776, "deadline_s": 120,
              "max_moves": 8, "candidate_limit": 8, "kv_budget_mib": 10240,
              "frozen_trace_sha256": "a" * 64, "move_fixed_s": 1.7743364535272121,
              "move_bandwidth_gib_s": 11.64086282917902, "seed": 20261024}
    environment = campaign.deployment_environment(config, "/frozen/trace.json")
    assert environment["DP_OPT_TRACE_POLICIES"].split() == list(campaign.POLICIES)
    assert environment["DP_OPT_REBALANCE_OVERHEAD_FRACTION"] == "0.01"
    assert environment["DP_OPT_MOVE_FIXED_S"] == str(config["move_fixed_s"])
    assert environment["DP_OPT_REBALANCE_BUDGET_MIB"] == "10240"


def test_new_sync_includes_actual_new_runtime_and_regressions(campaign):
    assert len(campaign.SYNC_FILES) == len(set(campaign.SYNC_FILES))
    for name in ("amortized_rebalance.py", "amortized_campaign.py"):
        assert "experiments/dp_opt/" + name in campaign.SYNC_FILES
    for name in ("test_dp_opt_amortized_rebalance.py", "test_dp_opt_amortized_campaign.py"):
        assert "tests/" + name in campaign.SYNC_FILES
    repo = Path(__file__).resolve().parents[1]
    assert all((repo / name).is_file() for name in campaign.SYNC_FILES)


def test_runtime_freeze_rejects_changed_source(campaign, tmp_path):
    (tmp_path / "source.py").write_text("changed")
    with pytest.raises(RuntimeError, match="Frozen runtime"):
        campaign.check_frozen_sources(tmp_path, {"frozen_runtime_sha256": {"source.py": "a" * 64}})


def test_headroom_variant_keeps_original_four_and_fair_baselines(campaign):
    config = {"max_model_len": 131072, "cache_bytes": 25769803776, "deadline_s": 120,
              "max_moves": 8, "candidate_limit": 8, "kv_budget_mib": 10240,
              "frozen_trace_sha256": "a" * 64, "move_fixed_s": 1.7743364535272121,
              "move_bandwidth_gib_s": 11.64086282917902, "seed": 20261024}
    environment = campaign.deployment_environment(config, "/frozen/trace.json", headroom=True)
    assert environment["DP_OPT_TRACE_POLICIES"].split() == list(campaign.HEADROOM_POLICIES)
    assert environment["DP_OPT_REBALANCE_HEADROOM_QUANTILE"] == "0.95"
    assert environment["DP_OPT_REBALANCE_HEADROOM_HISTORY_LIMIT"] == "64"
    assert "experiments/dp_opt/headroom_rebalance.py" in campaign.SYNC_FILES
    assert "tests/test_dp_opt_headroom_rebalance.py" in campaign.SYNC_FILES
