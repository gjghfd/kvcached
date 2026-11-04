import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import performance_numerics
import verify_trace
import performance_public


async def runtime_fixture(args):
    if args.kind == "serial":
        raise RuntimeError("Selected trace request differs from its full serial greedy reference")
    if args.kind == "paired":
        raise RuntimeError("Paired policy trace token IDs differ from no-move baseline")
    if args.kind == "oom":
        raise RuntimeError("CUDA out of memory")
    return "completed"


@pytest.mark.parametrize("kind", ["serial", "paired", "normal"])
def test_runtime_changes_only_the_two_exact_greedy_raises(kind):
    events = []
    function = performance_numerics.rewrite_greedy_raises(runtime_fixture, events.append)
    result = asyncio.run(function(SimpleNamespace(kind=kind)))
    assert len(events) == (0 if kind == "normal" else 1)
    if kind == "normal":
        assert result == "completed"


def test_runtime_oom_still_fails():
    function = performance_numerics.rewrite_greedy_raises(runtime_fixture, lambda event: None)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        asyncio.run(function(SimpleNamespace(kind="oom")))


def test_missing_runtime_gate_is_rejected():
    with pytest.raises(ValueError, match="exactly the two"):
        performance_numerics.rewrite_greedy_raises(performance_numerics.observe_root, lambda event: None)


def token_fixture():
    config = {"policies": ["no_move", "kv_only_workaware"], "repeats": 1}
    maps = {(0, "no_move"): {"req": [1, 2, 3]}, (0, "kv_only_workaware"): {"req": [1, 4, 3]}}
    references = [{"trace_request_id": "req", "token_ids": [1, 2, 3]}]
    return config, maps, references


def test_numerical_difference_is_recorded_without_weakening_original_verifier():
    config, maps, references = token_fixture()
    original = verify_trace.verify_paired_tokens
    result = performance_numerics.observe_maps(config, maps, references)
    assert not result["full_greedy_equivalence_passed"]
    assert result["paired"][1]["first_difference_zero_based"] == 1
    assert verify_trace.verify_paired_tokens is original
    with pytest.raises(ValueError, match="Paired policy outputs differ"):
        original(config, maps, references)


@pytest.mark.parametrize("change", ["missing_request", "output_length"])
def test_changed_workload_is_not_accepted_as_numeric_noise(change):
    config, maps, references = token_fixture()
    if change == "missing_request":
        maps[(0, "kv_only_workaware")] = {}
    else:
        maps[(0, "kv_only_workaware")]["req"].append(9)
    with pytest.raises(ValueError, match="request IDs|output workloads"):
        performance_numerics.observe_maps(config, maps, references)


@pytest.mark.parametrize("message,allowed", [
    (performance_numerics.GREEDY_ERRORS[0], True),
    (performance_numerics.GREEDY_ERRORS[1], True), ("CUDA out of memory", False),
    ("Migration ownership differs", False),
])
def test_only_drained_greedy_failure_can_be_analyzed_with_explicit_opt_in(tmp_path, message, allowed):
    (tmp_path / "exit-status").write_text("1\n")
    (tmp_path / "summary.json").write_text('{"status":"running"}')
    (tmp_path / "error.json").write_text(json.dumps({"type": "RuntimeError", "message": message}))
    assert not performance_numerics.terminal_acceptable(tmp_path, False)
    assert performance_numerics.terminal_acceptable(tmp_path, True) is allowed


def test_real_source_gate_transformations_are_precisely_bound():
    assert performance_numerics.rewrite_greedy_raises(performance_numerics.live_trace.run, lambda event: None)
    function = performance_numerics.verification_function(lambda *args: None, False)
    assert function.__name__ == "verify"
    original = inspect.getsource(verify_trace.verify)
    assert "Sampled mapped budget exceeded" in original
    assert "Real transfer cost not included" in original or "migration" in original.lower()


def test_performance_runtime_restores_original_and_records_actual_divergence(tmp_path):
    config, maps, references = token_fixture()
    original = performance_numerics.live_trace.run
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "references.json").write_text(json.dumps(references))
    (tmp_path / "summary.json").write_text('{"status":"passed"}')
    for (repeat, policy), requests in maps.items():
        directory = tmp_path / f"r{repeat}-{policy}"
        directory.mkdir()
        (directory / "requests.json").write_text(json.dumps([
            {"request_id": identifier, "token_ids": tokens} for identifier, tokens in requests.items()]))
    with performance_numerics.runtime_observation(tmp_path):
        assert performance_numerics.live_trace.run is not original
    assert performance_numerics.live_trace.run is original
    policy = json.loads((tmp_path / "performance-numerics-policy.json").read_text())
    assert not policy["greedy_gate_is_fatal"] and policy["no_hidden_head_CPU_snapshot_hooks"]
    assert not json.loads((tmp_path / "greedy-observations.json").read_text())["full_greedy_equivalence_passed"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["performance_completed"] and not summary["full_greedy_equivalence_passed"]


def test_fatal_error_still_restores_runtime_without_marking_performance_complete(tmp_path):
    original = performance_numerics.live_trace.run
    with pytest.raises(RuntimeError, match="owner mismatch"):
        with performance_numerics.runtime_observation(tmp_path):
            raise RuntimeError("owner mismatch")
    assert performance_numerics.live_trace.run is original
    assert not (tmp_path / "greedy-observations.json").exists()


def test_public_verifier_context_restores_after_failure():
    original = lambda *args: None
    module = SimpleNamespace(verify=original)
    with pytest.raises(RuntimeError):
        with performance_public.adapted(module):
            assert module.verify is performance_numerics.verify
            raise RuntimeError("test")
    assert module.verify is original


def test_trace_launch_changes_only_the_explicit_entry_and_never_overwrites(tmp_path):
    directory = tmp_path / "repo/experiments/dp_opt"
    directory.mkdir(parents=True)
    original = '#!/usr/bin/env bash\npython "$repo/experiments/dp_opt/live_trace.py" --arg unchanged\n'
    (directory / "trace-benefit.sbatch").write_text(original)
    output = tmp_path / "launch.sh"
    result = performance_public.make_trace_launch(tmp_path / "repo", output)
    assert output.read_text() == original.replace('/live_trace.py"', '/performance_trace.py"')
    assert result["only_entry_changed"]
    with pytest.raises(FileExistsError):
        performance_public.make_trace_launch(tmp_path / "repo", output)


def test_performance_deployment_scope_and_default_setting_are_explicit():
    import performance_campaign

    repo = Path(performance_campaign.__file__).resolve().parents[2]
    for name in performance_campaign.SYNC_FILES:
        assert (repo / name).is_file()
    script = (repo / "experiments/dp_opt/performance-native.sbatch").read_text()
    assert "#SBATCH --job-name=qwen-public-perf-native" in script
    assert "DP_OPT_PERFORMANCE_NUMERICS=observe" in script
    assert "--time=04:30:00" in script
    assert "head_hidden" not in script and "fp32_head" not in script
