import hashlib
import importlib
import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def freezer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("freeze_measured_control")


def trials():
    return [{"case": {"population": population, "mode": "preplan_boundary"},
             "decision": {"pause_call_s": 3.052, "preplan_s": .587, "solve_s": 0,
                          "end_to_end_control_s": 3.19,
                          "moves": [{"elapsed_s": .027, "kv_bytes": 432 * 1024**2}]}}
            for population in (8, 12, 16)]


def test_freeze_uses_max_whole_control_cost_not_only_copy_collective(freezer):
    parameters = freezer.parameters(trials())
    assert parameters == {"wait_incoming_handshake_s": 3.06, "wait_old_transaction_fixed_s": .12,
                          "wait_copy_seconds_per_gib": .07, "wait_solve_floor_s": .59,
                          "move_fixed_s": 0, "move_bandwidth_gib_s": 1 / .07}
    with pytest.raises(ValueError):
        freezer.parameters(trials()[:-1])
    changed = trials()
    changed[0]["decision"]["solve_s"] = .01
    with pytest.raises(ValueError):
        freezer.parameters(changed)


def test_measured_config_requires_pinned_hash_and_consistent_copy_slope(freezer, tmp_path):
    path = tmp_path / "control.json"
    payload = {"status": "frozen_measured_boundary_control", "parameters": freezer.parameters(trials())}
    path.write_text(json.dumps(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert freezer.load_control(path, digest) == payload
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="frozen SHA256"):
        freezer.load_control(path, digest)
    payload["parameters"]["move_bandwidth_gib_s"] = 32
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="inconsistent"):
        freezer.load_control(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_prepared_control_verifier_counts_planning_and_default_boundary_receipts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    verifier = importlib.import_module("verify_trace")
    decision = {"path": "blocked_staged_ilp", "end_to_end_control_s": 2, "preplan_s": .5,
                "end_to_end_control_s_including_preplan": 2.5, "pause_call_s": 1, "solve_s": 0,
                "resume_s": .01, "admitted": True, "moves": [{"elapsed_s": .03}],
                "active_blocking_upper_request_s": .3,
                "pause_receipts": [[{"rank": rank, "armed_at": 1, "reached_at": 2,
                                    "callback_at": 2.01, "boundary_step_counter": 32} for rank in (0, 1)]
                                   for model in (0, 1)]}
    verifier.verify_preplan_decision(decision)
    assert verifier.control_metrics([decision])["total_control_s_including_preplan"] == 2.5
    decision["end_to_end_control_s_including_preplan"] = 2
    with pytest.raises(ValueError, match="omits or duplicates"):
        verifier.verify_preplan_decision(decision)
    decision["end_to_end_control_s_including_preplan"] = 2.5
    decision["pause_receipts"][0][1]["boundary_step_counter"] = 31
    with pytest.raises(ValueError, match="default32"):
        verifier.verify_preplan_decision(decision)


@pytest.mark.parametrize("inherited_mode", [None, "DP_OPT_CAPACITY_PILOT", "DP_OPT_RATE_CALIBRATION",
                                           "DP_OPT_TRANSFER_PILOT", "DP_OPT_LAYOUT_COST_PILOT"])
def test_measured_launcher_passes_frozen_costs_for_every_policy(tmp_path, monkeypatch, inherited_mode):
    if inherited_mode is not None:
        monkeypatch.setenv(inherited_mode, "1")
    scripts = Path(__file__).resolve().parents[1] / "experiments/dp_opt"
    launcher = (scripts / "trace-benefit.sbatch").read_text()
    block = launcher.split("extra_args=()", 1)[1].split('python "$repo/experiments/dp_opt/live_trace.py"', 1)[0]
    assert '--measured-control-config "$DP_OPT_MEASURED_CONTROL_CONFIG"' in block
    names = {"DP_OPT_PREPLAN": "1", "DP_OPT_WAIT_AWARE": "1", "DP_OPT_REVALIDATE_PREPARED": "1",
             "DP_OPT_WAIT_INCOMING_HANDSHAKE_S": "3.06",
             "DP_OPT_WAIT_OLD_TRANSACTION_FIXED_S": ".09", "DP_OPT_WAIT_COPY_SECONDS_PER_GIB": ".07",
             "DP_OPT_WAIT_SOLVE_FLOOR_S": ".59", "DP_OPT_MOVE_FIXED_S": "0",
             "DP_OPT_MOVE_BANDWIDTH_GIB_S": "14.285714285714285",
             "DP_OPT_LAYOUT_COST_LOOKUP": "/frozen/lookup.json", "DP_OPT_LAYOUT_COST_LOOKUP_SHA256": "a" * 64,
             "DP_OPT_LAYOUT_COST_SOURCE_ROOT": "/frozen/raw", "DP_OPT_MEASURED_CONTROL_CONFIG": "/frozen/control.json",
             "DP_OPT_MEASURED_CONTROL_CONFIG_SHA256": "b" * 64, "repo": str(scripts.parents[1]),
             "evidence": str(tmp_path)}
    environment = {name: value for name, value in os.environ.items() if not name.startswith("DP_OPT_")}
    result = subprocess.run(["bash", "-c", 'set -eu; sha256sum() { true; }; extra_args=();\n' + block
                             + '\nprintf "%s\\n" "${extra_args[@]}"'],
                            env={**environment, **names}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    for flag, name in (("--wait-incoming-handshake-s", "DP_OPT_WAIT_INCOMING_HANDSHAKE_S"),
                       ("--wait-old-transaction-fixed-s", "DP_OPT_WAIT_OLD_TRANSACTION_FIXED_S"),
                       ("--move-bandwidth-gib-s", "DP_OPT_MOVE_BANDWIDTH_GIB_S"),
                       ("--layout-cost-lookup-sha256", "DP_OPT_LAYOUT_COST_LOOKUP_SHA256")):
        assert arguments[arguments.index(flag) + 1] == names[name]
    assert "--preplan" in arguments and "--wait-aware" in arguments
    assert "--revalidate-prepared" in arguments
