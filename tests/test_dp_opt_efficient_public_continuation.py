import json
import subprocess
from contextlib import nullcontext
from pathlib import Path

import pytest

import efficient_public_continuation as continuation
import verify_efficient_trace
import verify_trace


@pytest.mark.parametrize("optimizer,expected", [("point", "original"), ("robust", "certified")])
def test_phase_verifier_preserves_strict_checks_and_requires_certificates(tmp_path, monkeypatch, optimizer, expected):
    (tmp_path / "config.json").write_text(json.dumps({"workaware_optimizer": optimizer}))
    calls = []
    monkeypatch.setattr(verify_trace, "verify", lambda *args: calls.append("original") or {"strict": True})
    monkeypatch.setattr(verify_efficient_trace, "verify", lambda *args: calls.append("certified") or {"strict": True})
    assert continuation.verify_phase(tmp_path, tmp_path / "bank") == {"strict": True}
    assert calls == [expected]


@pytest.mark.parametrize("optimizer,manifest", [("unknown", False), (None, False), ("point", True)])
def test_unknown_or_uncertified_phase_rejected(tmp_path, optimizer, manifest):
    (tmp_path / "config.json").write_text(json.dumps({"workaware_optimizer": optimizer}))
    if manifest:
        (tmp_path / "solver-efficiency-source.json").write_text("{}")
    with pytest.raises(ValueError, match="declared point"):
        continuation.verify_phase(tmp_path, None)


def test_verifier_context_restores_after_error_and_rejects_nesting():
    originals = [module.verify for module in (continuation.public_workload_search, continuation.public_poisson_followup)]
    with pytest.raises(ValueError, match="stage failed"):
        with continuation.adapted_verifiers():
            assert continuation.public_workload_search.verify is continuation.verify_phase
            with pytest.raises(RuntimeError, match="nested"):
                with continuation.adapted_verifiers():
                    pytest.fail("nested context entered")
            raise ValueError("stage failed")
    assert [module.verify for module in (continuation.public_workload_search, continuation.public_poisson_followup)] == originals


@pytest.mark.parametrize("role", ["search", "poisson"])
def test_real_canonical_launch_only_routes_robust_phase_through_efficient_entry(tmp_path, role):
    repo = Path(__file__).resolve().parents[1]
    canonical, wrapper, count = continuation.LAUNCHES[role]
    hashes = {name: continuation.digest(repo / "experiments/dp_opt" / name)
              for name in (canonical, "trace-benefit.sbatch")}
    generated = continuation.make_launch(repo, tmp_path / "launch with spaces", role)
    text = generated.read_text()
    assert text.count('if test "$DP_OPT_WORKAWARE_OPTIMIZER" = robust;') == count
    assert '--verify-phase' in text
    assert f'--module {role}' in text
    assert '/verify_trace.py"' not in text
    assert '/efficient_workaware_trace.py"' in generated.with_name("efficient-trace-launch.sh").read_text()
    assert 'DP_OPT_TRACE_BUDGET_MIB=10240' in text
    assert 'DP_OPT_MAX_MODEL_LEN=131072' in text
    assert 'DP_OPT_DEADLINE_S=120' in text
    subprocess.run(["bash", "-n", str(generated)], check=True)
    for name, expected in hashes.items():
        assert continuation.digest(repo / "experiments/dp_opt" / name) == expected
    with pytest.raises(RuntimeError, match="never overwrite"):
        continuation.make_launch(repo, generated.parent, role)


def test_changed_canonical_structure_refuses_launch_before_writing(tmp_path):
    directory = tmp_path / "repo/experiments/dp_opt"
    directory.mkdir(parents=True)
    (directory / "public-workload-search.sbatch").write_text("wrong command\n")
    (directory / "trace-benefit.sbatch").write_text("wrong entry\n")
    with pytest.raises(ValueError, match="Canonical launch structure changed"):
        continuation.make_launch(tmp_path / "repo", tmp_path / "output", "search")
    assert not list((tmp_path / "output").iterdir())


def test_campaign_adapters_restore_submission_and_sync_on_exception():
    modules = (continuation.public_workload_campaign, continuation.public_poisson_followup)
    originals = [(module.SYNC_FILES, module.protected_sync, module.verify if hasattr(module, "verify") else None)
                 for module in modules]
    command = continuation.public_workload_campaign.submission_command
    with pytest.raises(ValueError, match="abort"):
        with continuation.campaign_adapters(273918):
            assert set(continuation.EXTRA_FILES).issubset(continuation.public_workload_campaign.SYNC_FILES)
            generated = continuation.public_workload_campaign.submission_command({}, "/ledger")
            assert "efficient-public-search.sbatch" in generated
            assert "--parsable" in generated
            assert continuation.public_workload_search.verify is continuation.verify_phase
            with pytest.raises(RuntimeError, match="nested"):
                with continuation.campaign_adapters(273918):
                    pytest.fail("nested campaign entered")
            raise ValueError("abort")
    assert not continuation.CAMPAIGN_ENABLED
    assert continuation.public_workload_campaign.submission_command is command
    for module, original in zip(modules, originals):
        assert module.SYNC_FILES == original[0]
        assert module.protected_sync is original[1]


@pytest.mark.parametrize("mismatch", [False, True])
def test_input_reuse_uploads_new_traces_but_never_resends_bank(tmp_path, monkeypatch, mismatch):
    root = tmp_path / "case"
    inputs = root / "frozen/inputs"
    inputs.mkdir(parents=True)
    (inputs / "prompt-bank.json").write_text('{"tokens":[1,2]}')
    home, stage = "/remote/home", "/remote/stage"
    original = home + "/run/kvcached-paracloud/state/live-trace-273918/frozen/inputs/prompt-bank.json"
    manifest = continuation.digest(inputs / "prompt-bank.json") + "  " + original
    calls, uploads = [], []

    def checked(command):
        calls.append(command)
        if len(calls) == 1:
            return home
        if len(calls) == 2:
            return "wrong" if mismatch else manifest
        return ""

    monkeypatch.setattr(continuation, "checked_ssh", checked)
    monkeypatch.setattr(continuation, "protected_sync", lambda *args, **kwargs: uploads.append(kwargs) or (stage, home))
    state = {}
    kwargs = {"files": continuation.EXTRA_FILES, "prefix": "efficient-test",
              "upload_files": ("frozen/inputs/prompt-bank.json", "frozen/explore-0.json")}
    if mismatch:
        with pytest.raises(RuntimeError, match="do not install"):
            continuation.sync_without_prompt_bank(tmp_path, root, state, 273918, **kwargs)
        assert not uploads
        assert not state
    else:
        assert continuation.sync_without_prompt_bank(tmp_path, root, state, 273918, **kwargs) == (stage, home)
        assert uploads[0]["upload_files"] == ("frozen/explore-0.json",)
        assert calls[-1].count(continuation.digest(inputs / "prompt-bank.json")) == 2
        assert "cp -p" in calls[-1]
        assert state["public_input_parent_reused_after_double_SHA"]["job"] == 273918


@pytest.mark.parametrize("phase,complete,error", [
    ("submitted_state_and_plain_controls", False, None),
    ("diagnosis_required_no_automatic_duplicate_submission", False, "Control parent failed"),
    ("submitted_state_and_plain_controls", True, "terminal closure"),
])
def test_parent_gate_does_not_treat_submission_or_diagnosis_as_success(tmp_path, phase, complete, error):
    (tmp_path / "campaign-state.json").write_text(json.dumps({"phase": phase, "GPU_controls_complete": complete}))
    if error:
        with pytest.raises(RuntimeError, match=error):
            continuation.verified_control_parent(tmp_path)
    else:
        assert continuation.verified_control_parent(tmp_path) is None


@pytest.mark.parametrize("tamper", [False, True])
def test_complete_parent_requires_independent_raw_and_full_token_rebuild(tmp_path, monkeypatch, tamper):
    previous = {"phase": "controls_SHA_and_full_tokens_verified_not_benefit", "GPU_controls_complete": True,
                "state_last_observed": "COMPLETED", "job": 273918, "original_trace_sha256": "trace"}
    (tmp_path / "campaign-state.json").write_text(json.dumps(previous))
    (tmp_path / "raw-273918.sha256").write_text("manifest")
    (tmp_path / "local-monitor-identity.json").write_text('{"coordinator_pid":5222}')
    rebuilt = {"state_controls": {"trace_sha256": "trace"},
               "plain_no_move": {"trace_sha256": "trace"}, "is_algorithm_benefit": False}
    (tmp_path / "independent-controls-verification.json").write_text(json.dumps({} if tamper else rebuilt))
    calls = []
    monkeypatch.setattr(continuation, "verify_raw", lambda *args: calls.append("raw"))
    monkeypatch.setattr(continuation, "verify_controls", lambda *args: calls.append("controls") or rebuilt["state_controls"])
    monkeypatch.setattr(continuation, "verify_trace", lambda *args: calls.append("tokens") or rebuilt["plain_no_move"])
    monkeypatch.setattr(continuation.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 1))
    if tamper:
        with pytest.raises(RuntimeError, match="reconstruction differs"):
            continuation.verified_control_parent(tmp_path)
    else:
        assert continuation.verified_control_parent(tmp_path) == previous
    assert calls == ["raw", "controls", "tokens"]


def test_unqualified_native_search_closes_without_submitting_evaluation(tmp_path, monkeypatch):
    parent, root, bootstrap = tmp_path / "parent", tmp_path / "campaign", tmp_path / "bootstrap"
    (parent / "live-trace-273918/frozen").mkdir(parents=True)
    previous = {"job": 273918, "frozen_runtime_sha256": {}}
    (parent / "campaign-state.json").write_text(json.dumps(previous))
    monkeypatch.setattr(continuation, "EXTRA_FILES", ())
    monkeypatch.setattr(continuation, "verified_control_parent", lambda path: previous)
    monkeypatch.setattr(continuation, "campaign_adapters", lambda job: nullcontext())
    monkeypatch.setattr(continuation, "verify_launch", lambda *args: {"verified": True})
    calls = []

    def search(path, old):
        calls.append("search")
        (path / "campaign-state.json").write_text('{"candidate_case":null,"job":273999}')

    monkeypatch.setattr(continuation.public_workload_campaign, "run", search)
    monkeypatch.setattr(continuation.public_poisson_followup, "run", lambda *args: calls.append("evaluation"))
    continuation.run(root, parent, bootstrap)
    assert calls == ["search"]
    state = json.loads((root / "campaign-state.json").read_text())
    assert "no_qualified_pressure" in state["phase"]
    assert state["research_goal_complete"] is False
    with pytest.raises(RuntimeError, match="reconcile"):
        continuation.run(root, parent, bootstrap)


@pytest.mark.parametrize("role", ["search", "poisson"])
def test_launch_receipt_rebuilds_exact_executed_scripts(tmp_path, role):
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "live-trace-273999"
    script = continuation.make_launch(repo, root / "launch-sources", role)
    receipt = continuation.verify_launch(root, repo, role)
    assert receipt["role"] == role
    script.write_text(script.read_text() + "changed command\n")
    with pytest.raises(ValueError, match="independent reconstruction"):
        continuation.verify_launch(root, repo, role)


def test_launch_receipt_rejects_other_job_directory(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "live-trace-273999"
    continuation.make_launch(repo, root / "launch-sources", "search")
    path = root / "launch-sources/efficient-public-launch-source.json"
    receipt = json.loads(path.read_text())
    receipt["output_directory"] = str(tmp_path / "live-trace-999/launch-sources")
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="immutable job"):
        continuation.verify_launch(root, repo, "search")
