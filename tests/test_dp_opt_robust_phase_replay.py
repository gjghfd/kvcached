import copy
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("replay_workaware_robust_phase")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def case_fixture(tmp_path):
    case = tmp_path / "r0-kv_only_workaware"
    save(case / "workload.json", {"policy": "kv_only_workaware", "capacity_gib": [10., 10.]})
    gate = {"accepted": False, "measured_blocking_so_far_request_s": 7.}
    planned = {"path": "release_rebalance_ilp", "reason": "rebalanced",
               "observations": [["live"]], "frozen_observations": [["frozen"]],
               "settings": {"solver_limit_s": 9., "nested": {"original": True}},
               "before_pause": {"gate": gate, "winner": {"moves": ["first"]}},
               "after_pause": {"gate": gate, "winner": {"moves": ["first"]}}}
    save(case / "decisions.json", [{"path": "admission"}, planned,
                                  {"path": "release_rebalance_ilp", "reason": "outside_measured_work_domain"}])
    return case


def test_all_phases_budgets_and_original_paid_cost_are_verified(replay, tmp_path, monkeypatch):
    case = case_fixture(tmp_path)
    original = json.loads((case / "decisions.json").read_text())
    calls = []

    def planner(observed, budgets, fit, override, settings, lookup, blocking, phase):
        calls.append((copy.deepcopy(observed), budgets, override, settings["solver_limit_s"], blocking, phase))
        settings["nested"]["original"] = False
        observed.append("mutated")
        return {"winner": {"moves": ["first"]}, "gate": {"accepted": True}}

    def verifier(check, observed, budgets, fit, settings, phase, blocking, **options):
        assert options["controller_planning_upper_s"] >= 0
        assert settings["nested"]["original"] is True
        assert observed == ([["live"]] if phase == "before_pause" else [["frozen"]])
        return {"accepted": True}

    monkeypatch.setattr(replay, "plan_workaware_robust_rebalance", planner)
    monkeypatch.setattr(replay, "verify_plan", verifier)
    result = replay.replay_decisions(case, {}, [1., 8.])
    assert len(result["replays"]) == 4
    assert calls == [([["live"]], [10., 10.], True, budget, 0., "before_pause") for budget in [1., 8.]] + [
        ([["frozen"]], [10., 10.], True, budget, 7., "after_pause") for budget in [1., 8.]]
    assert all(entry["new_guard_passing_opportunity"] for entry in result["replays"])
    assert all(entry["same_selected_moves_as_point_plan"] for entry in result["replays"])
    assert result["unplanned_releases"] == [{"decision_index": 2, "reason": "outside_measured_work_domain"}]
    assert json.loads((case / "decisions.json").read_text()) == original


def test_independent_verification_failure_is_not_reported_as_success(replay, tmp_path, monkeypatch):
    case = case_fixture(tmp_path)
    monkeypatch.setattr(replay, "plan_workaware_robust_rebalance", lambda *args: {})

    def fail(*args, **kwargs):
        raise ValueError("Independent objective mismatch")

    monkeypatch.setattr(replay, "verify_plan", fail)
    with pytest.raises(ValueError, match="objective mismatch"):
        replay.replay_decisions(case, {}, [1.])


def test_reject_orphan_after_pause_plan(replay, tmp_path):
    case = case_fixture(tmp_path)
    decisions = json.loads((case / "decisions.json").read_text())
    del decisions[1]["before_pause"]
    save(case / "decisions.json", decisions)
    with pytest.raises(ValueError, match="lacks its original"):
        replay.replay_decisions(case, {}, [1.])


@pytest.mark.parametrize("budgets", [[], [0.], [-1.], [True], [float("nan")], [float("inf")], [1., 1.]])
def test_bad_solver_budgets_fail_before_raw_verification(replay, tmp_path, budgets):
    with pytest.raises(ValueError, match="solver budgets"):
        replay.replay_phase(tmp_path / "raw", tmp_path / "manifest", tmp_path / "bank",
                            tmp_path / "sources", tmp_path / "state", tmp_path / "output", budgets, tmp_path)


@pytest.mark.parametrize("inside_raw", [True, False])
def test_output_protection_precedes_execution(replay, tmp_path, inside_raw):
    raw = tmp_path / "raw"
    output = raw / "new.json" if inside_raw else tmp_path / "existing.json"
    if not inside_raw:
        output.write_text("preserve")
    with pytest.raises(ValueError, match="immutable raw"):
        replay.replay_phase(raw, tmp_path / "manifest", tmp_path / "bank", tmp_path / "sources",
                            tmp_path / "state", output, [1.], tmp_path)


def test_source_hashes_and_original_runtime_must_match(replay, tmp_path):
    source = tmp_path / "planner.py"
    source.write_text("pass\n")
    manifest = tmp_path / "sources.sha256"
    manifest.write_text(replay.digest(source) + "  planner.py\n")
    state = tmp_path / "state.json"
    save(state, {"frozen_runtime_sha256": {"planner.py": replay.digest(source)}})
    sources, runtime = replay.verify_sources(tmp_path, manifest, state)
    assert sources == runtime
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="source mismatch"):
        replay.verify_sources(tmp_path, manifest, state)


@pytest.mark.parametrize("kind", ["empty", "duplicate", "unsafe", "missing_runtime"])
def test_invalid_source_contract_is_rejected(replay, tmp_path, kind):
    source = tmp_path / "planner.py"
    source.write_text("pass\n")
    line = replay.digest(source) + "  planner.py\n"
    manifest = tmp_path / "sources.sha256"
    manifest.write_text("" if kind == "empty" else line * (2 if kind == "duplicate" else 1))
    if kind == "unsafe":
        manifest.write_text(replay.digest(source) + "  ../planner.py\n")
    state = tmp_path / "state.json"
    save(state, {} if kind == "missing_runtime" else {"frozen_runtime_sha256": {"planner.py": replay.digest(source)}})
    with pytest.raises(ValueError):
        replay.verify_sources(tmp_path, manifest, state)


def phase_fixture(replay, tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    case_fixture(raw)
    save(raw / "placement-fit.json", {"raw": True})
    source = tmp_path / "planner.py"
    source.write_text("pass\n")
    manifest = tmp_path / "sources.sha256"
    manifest.write_text(replay.digest(source) + "  planner.py\n")
    state = tmp_path / "state.json"
    save(state, {"frozen_runtime_sha256": {"planner.py": replay.digest(source)}})
    raw_manifest = tmp_path / "raw.sha256"
    raw_manifest.write_text("already-verified-by-test-hook")
    bank = tmp_path / "bank.json"
    save(bank, {})
    calls = []
    monkeypatch.setattr(replay, "verify_raw", lambda *args: calls.append("raw"))
    monkeypatch.setattr(replay, "verify", lambda *args: calls.append("trace") or {"status": "raw_trace_verified"})
    monkeypatch.setattr(replay, "replay_decisions", lambda *args: calls.append("replay") or {"replays": []})
    return (raw, raw_manifest, bank, manifest, state, tmp_path / "result.json", [1.], tmp_path), calls


def test_complete_phase_is_reverified_before_exclusive_output(replay, tmp_path, monkeypatch):
    arguments, calls = phase_fixture(replay, tmp_path, monkeypatch)
    result = replay.replay_phase(*arguments)
    assert calls == ["raw", "trace", "replay", "raw"]
    assert result["runtime_changed"] is False
    assert result["GPU_submissions"] is False
    assert result["performance_extrapolation"] is False
    assert json.loads(arguments[5].read_text()) == result


@pytest.mark.parametrize("failure", ["raw", "trace", "changed_source", "missing_case"])
def test_verification_failures_leave_no_success_output(replay, tmp_path, monkeypatch, failure):
    arguments, calls = phase_fixture(replay, tmp_path, monkeypatch)

    def fail(*args):
        raise ValueError("Independent validation failure")

    if failure == "raw":
        monkeypatch.setattr(replay, "verify_raw", fail)
    elif failure == "trace":
        monkeypatch.setattr(replay, "verify", fail)
    elif failure == "changed_source":
        def mutate(*args):
            (tmp_path / "planner.py").write_text("changed\n")
            return {"replays": []}
        monkeypatch.setattr(replay, "replay_decisions", mutate)
    else:
        (arguments[0] / "r0-kv_only_workaware").rename(arguments[0] / "r0-no_move")
    with pytest.raises(ValueError):
        replay.replay_phase(*arguments)
    assert not arguments[5].exists()
