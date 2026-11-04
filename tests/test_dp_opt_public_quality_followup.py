import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def followup(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("public_quality_followup")


@pytest.fixture
def source_plan(followup, tmp_path, monkeypatch):
    root, parent = tmp_path / "coordinator", tmp_path / "stratos"
    calibration, evaluation = tmp_path / "quality", tmp_path / "evaluation"
    inputs, original = calibration / "frozen/inputs", parent / "frozen/inputs"
    cohort = tmp_path / "cohort/cohort-response-row-map.jsonl"
    records = [{"source_sha256": "raw-shard-sha", "source_row": index, "assistant_index": 1,
                "original_response_sha256": f"response-{index}", "profile_selection_sha256": f"{index:064x}",
                "prompt_tokens": 10, "output_tokens": 13000,
                "marker_flags": {"open_marker_count": 1, "close_marker_count": 1, "unique_pair_in_order": True,
                                 "nonempty_original_final_text": True, "cohort_eligible": True}}
               for index in range(257)]
    cohort.parent.mkdir()
    cohort.write_text("".join(json.dumps(record) + "\n" for record in records))
    (cohort.parent / "full-source-response-row-map.jsonl").write_text(cohort.read_text())
    identity = {"cohort_definition": {"no_token_length_selection": True, "no_GPU_result_selection": True,
                                       "no_response_trim_concat_padding_or_extension": True}}
    write(cohort.parent / "predeclared-identity.json", identity)
    write(cohort.parent / "summary.json", {"rows": len(records)})
    identity_sha = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    hashes = {name: followup.digest(cohort.parent / name) for name in
              ("cohort-response-row-map.jsonl", "full-source-response-row-map.jsonl", "predeclared-identity.json", "summary.json")}
    write(cohort.parent / "archive-manifest.json", {"identity_sha256": identity_sha,
                                                   "outputs": {name: {"sha256": digest} for name, digest in hashes.items()}})
    code = [{"prompt_key": "code", "prompt_tokens": 100001, "output_tokens": 64}]
    write(original / "input-manifest.json", {"dataset_label": "stratos"})
    write(original / "code-marks.json", code)
    write(original / "prompt-bank.json", {"prompts": {"code": {"token_ids": [1, 2]}}})
    binding = {"cohort_identity_sha256": identity_sha, "cohort_files_sha256": hashes,
               "C0_manifest_sha256": followup.digest(original / "input-manifest.json"),
               "cohort_rows": len(records), "natural_eos_verified": False, "not_full_population": True}
    write(inputs / "input-manifest.json", {"dataset_label": followup.LABEL, "source_quality": binding})
    write(inputs / "code-marks.json", code)
    write(inputs / "prompt-bank.json", {"prompts": {"code": {"token_ids": [1, 2]}}, "source_quality": binding})
    marks = [{**{field: record[field] for field in ("source_sha256", "source_row", "assistant_index",
                                                   "original_response_sha256", "prompt_tokens", "output_tokens")},
              "selection_sha256": record["profile_selection_sha256"]} for record in records[:256]]
    write(inputs / "conv-marks.json", marks)
    plan = {"schema_version": 1, "dataset_label": followup.LABEL,
            "source_selection": "original_marker_quality_only_not_output_length_or_GPU_gain", "Stratos_bank_replaced": False,
            "calibration_root": str(calibration), "evaluation_root": str(evaluation),
            "input_manifest_sha256": followup.digest(inputs / "input-manifest.json"),
            "parent_input_manifest_sha256": followup.digest(original / "input-manifest.json"),
            "cohort_row_map": str(cohort), "cohort_row_map_sha256": followup.digest(cohort),
            "cohort_identity_sha256": identity_sha}
    write(root / followup.PLAN_FILE, plan)
    monkeypatch.setattr(followup, "validate", lambda path: json.loads((path / "input-manifest.json").read_text()))
    return SimpleNamespace(root=root, parent=parent, calibration=calibration, evaluation=evaluation,
                           inputs=inputs, cohort=cohort, plan=plan, marks=marks)


def test_quality_plan_keeps_code_and_lowest_hash_original_bank(followup, source_plan):
    plan, calibration, evaluation = followup.load_plan(source_plan.root, source_plan.parent)
    assert plan == source_plan.plan
    assert calibration == source_plan.calibration and evaluation == source_plan.evaluation


@pytest.mark.parametrize("field,value", [("Stratos_bank_replaced", True), ("dataset_label", "stratos"),
                                          ("source_selection", "longest_response"), ("schema_version", 2)])
def test_quality_plan_rejects_source_replacement_or_length_selection(followup, source_plan, field, value):
    source_plan.plan[field] = value
    write(source_plan.root / followup.PLAN_FILE, source_plan.plan)
    with pytest.raises(ValueError, match="predeclare"):
        followup.load_plan(source_plan.root, source_plan.parent)


@pytest.mark.parametrize("kind", ["duplicate_root", "outside_root", "recursive_plan"])
def test_quality_roots_prevent_aliasing_or_recursive_delegation(followup, source_plan, tmp_path, kind):
    if kind == "recursive_plan":
        write(source_plan.evaluation / followup.PLAN_FILE, {})
    else:
        source_plan.plan["evaluation_root"] = str(source_plan.calibration if kind == "duplicate_root" else tmp_path / "outside/child")
        write(source_plan.root / followup.PLAN_FILE, source_plan.plan)
    with pytest.raises(ValueError, match="distinct"):
        followup.load_plan(source_plan.root, source_plan.parent)


@pytest.mark.parametrize("kind", ["Code_ids", "output_length", "duplicate_conv", "rerolled_bank", "cohort_change", "identity_change"])
def test_quality_plan_rejects_bank_length_provenance_and_hash_rerolls(followup, source_plan, kind):
    if kind == "Code_ids":
        bank = json.loads((source_plan.inputs / "prompt-bank.json").read_text())
        bank["prompts"]["code"]["token_ids"] = [3, 4]
        write(source_plan.inputs / "prompt-bank.json", bank)
    elif kind == "cohort_change":
        source_plan.cohort.write_text(source_plan.cohort.read_text() + "\n")
    elif kind == "identity_change":
        write(source_plan.cohort.parent / "predeclared-identity.json", {"changed": True, "cohort_definition": {}})
    else:
        if kind == "output_length":
            source_plan.marks[0]["output_tokens"] += 1
        elif kind == "duplicate_conv":
            source_plan.marks[-1] = source_plan.marks[0]
        else:
            source_plan.marks.reverse()
        write(source_plan.inputs / "conv-marks.json", source_plan.marks)
    with pytest.raises(ValueError):
        followup.load_plan(source_plan.root, source_plan.parent)


def closed_root(followup, tmp_path, field="GPU_calibration_complete"):
    root = tmp_path / "closed"
    raw = root / "live-trace-15"
    analysis_name = "search-analysis.json" if field == "GPU_calibration_complete" else "evaluation-analysis.json"
    write(raw / analysis_name, {"candidate_case": "rate-1"})
    (raw / "exit-status").write_text("0\n")
    manifest = "".join(f"{followup.digest(path)}  {path.name}\n" for path in sorted(raw.iterdir()))
    (root / "raw-15.sha256").write_text(manifest)
    write(root / "campaign-state.json", {"job": 15, "state_last_observed": "COMPLETED", field: True, "candidate_case": "rate-1"})
    write(root / "archive-15.json", {"state": "COMPLETED", "watcher_barrier": "shared_markers", "raw_files_verified": 2})
    write(root / "independent-verification.json", {"candidate_case": "rate-1"})
    return root


@pytest.mark.parametrize("field", ["GPU_calibration_complete", "GPU_evaluation_complete"])
def test_quality_child_requires_independently_verified_terminal_success(followup, tmp_path, field):
    root = closed_root(followup, tmp_path, field)
    assert followup.verify_closed(root, field)[field] is True


@pytest.mark.parametrize("kind", ["pending", "raw_change", "count", "watcher", "independent", "candidate"])
def test_quality_terminal_failure_cannot_pass_as_source_or_algorithm_success(followup, tmp_path, kind):
    root = closed_root(followup, tmp_path)
    if kind == "raw_change":
        (root / "live-trace-15/exit-status").write_text("1\n")
    elif kind in ("count", "watcher"):
        archive = json.loads((root / "archive-15.json").read_text())
        archive["raw_files_verified" if kind == "count" else "watcher_barrier"] = 0
        write(root / "archive-15.json", archive)
    elif kind == "independent":
        write(root / "independent-verification.json", {"candidate_case": None})
    else:
        state = json.loads((root / "campaign-state.json").read_text())
        state["state_last_observed" if kind == "pending" else "candidate_case"] = "PENDING" if kind == "pending" else None
        write(root / "campaign-state.json", state)
    with pytest.raises((ValueError, RuntimeError)):
        followup.verify_closed(root, "GPU_calibration_complete")


@pytest.mark.parametrize("failure", [False, True])
def test_child_mirror_has_single_local_producer_and_final_status(followup, source_plan, monkeypatch, failure):
    state = {"quality_followup_plan_sha256": followup.digest(source_plan.root / followup.PLAN_FILE), "last_poll_at": "stale"}
    monkeypatch.setattr(followup, "check_pause", lambda root: None)
    monkeypatch.setattr(followup, "check_frozen_sources", lambda repo, state: None)
    sleeps = []
    monkeypatch.setattr(followup.time, "sleep", sleeps.append)
    calls = []

    class Child:
        pid = 11111
        returncode = None

        def poll(self):
            if not calls:
                calls.append(True)
                write(source_plan.calibration / "campaign-state.json", {"phase": "RUNNING", "job": 16,
                      "last_poll_at": "fresh", "last_poll_ssh_exit": 0})
                return None
            self.returncode = 1 if failure else 0
            write(source_plan.calibration / "campaign-state.json", {"phase": "FAILED" if failure else "COMPLETED", "job": 16})
            return self.returncode

    spawned = []
    monkeypatch.setattr(followup.subprocess, "Popen", lambda command, **kwargs: spawned.append(command) or Child())
    arguments = [source_plan.root, state, source_plan.calibration, "public_workload_campaign.py", [], "calibration"]
    if failure:
        with pytest.raises(RuntimeError, match="failed"):
            followup.run_child(Path("repository"), *arguments)
    else:
        followup.run_child(Path("repository"), *arguments)
    assert sleeps == [300] and len(spawned) == 1
    assert state["job"] == 16 and "last_poll_at" not in state
    assert state["source_child_phase"] == ("FAILED" if failure else "COMPLETED")
    assert state["source_child_exit"] == (1 if failure else 0)
    assert not any("ssh" in part for part in spawned[0])


def test_duplicate_quality_child_identity_prevents_any_spawn(followup, source_plan, monkeypatch):
    write(source_plan.calibration / "local-monitor-identity.json", {"pid": 12345})
    monkeypatch.setattr(followup, "check_pause", lambda root: None)
    monkeypatch.setattr(followup, "check_frozen_sources", lambda repo, state: None)
    monkeypatch.setattr(followup.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Duplicate child spawned"))
    state = {"quality_followup_plan_sha256": followup.digest(source_plan.root / followup.PLAN_FILE)}
    with pytest.raises(RuntimeError, match="existing"):
        followup.run_child(Path("repository"), source_plan.root, state, source_plan.calibration, "unused.py", [], "calibration")


def test_changed_descriptor_blocks_launch_before_child_can_submit(followup, source_plan, monkeypatch):
    monkeypatch.setattr(followup, "check_pause", lambda root: None)
    monkeypatch.setattr(followup, "check_frozen_sources", lambda repo, state: None)
    monkeypatch.setattr(followup.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Changed plan launched"))
    with pytest.raises(RuntimeError, match="plan changed"):
        followup.run_child(Path("repository"), source_plan.root, {"quality_followup_plan_sha256": "old"},
                           source_plan.calibration, "unused.py", [], "calibration")


@pytest.mark.parametrize("calibration_failure,qualified", [(True, True), (False, False), (False, True)])
def test_followup_gate_does_not_evaluate_failed_or_unqualified_source(followup, source_plan, monkeypatch, calibration_failure, qualified):
    roles = []
    monkeypatch.setattr(followup, "load_plan", lambda root, parent: (source_plan.plan, source_plan.calibration, source_plan.evaluation))
    monkeypatch.setattr(followup, "run_child", lambda *args: roles.append(args[-1]))

    def verify(root, field):
        if root == source_plan.calibration and calibration_failure:
            raise RuntimeError("calibration failed")
        return {"job": 17, "candidate_case": "rate-1" if qualified else None}

    monkeypatch.setattr(followup, "verify_closed", verify)
    state = {}
    if calibration_failure or not qualified:
        with pytest.raises(RuntimeError):
            followup.follow_quality(source_plan.root, source_plan.parent, Path("repository"), state)
        assert roles == ["quality_source_calibration"]
        assert not state.get("GPU_evaluation_complete")
    else:
        assert followup.follow_quality(source_plan.root, source_plan.parent, Path("repository"), state)
        assert roles == ["quality_source_calibration", "quality_source_grid_and_fixed_heldout"]
        assert state["GPU_evaluation_complete"] is True and state["total_research_goal_completed"] is False


def test_no_descriptor_keeps_existing_stratos_followup(followup, tmp_path):
    assert followup.follow_quality(tmp_path, tmp_path / "parent", Path("repository"), {}) is False


def test_full_future_regression_precedes_any_quality_dispatch(followup, tmp_path, monkeypatch):
    module = importlib.import_module("public_poisson_followup")
    parent, root = tmp_path / "parent", tmp_path / "future"
    write(parent / "campaign-state.json", {"phase": "closed", "GPU_calibration_complete": True})
    monkeypatch.setattr(module, "SYNC_FILES", ())
    monkeypatch.setattr(module, "check_pause", lambda root: None)
    monkeypatch.setattr(module, "check_frozen_sources", lambda repo, state: None)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    monkeypatch.setattr(followup, "follow_quality", lambda *args: pytest.fail("Untested quality dispatch"))
    with pytest.raises(RuntimeError, match="regression failed"):
        module.run(root, parent)
    assert (root / "cpu-regression.exit").read_text() == "1\n"
