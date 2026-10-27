import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("stress_sweep"), importlib.import_module("analyze_stress_sweep")


def test_runtime_archive_does_not_serialize_loaded_prompt_bank(modules, tmp_path):
    runner, analyzer = modules
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    (source / "worker-instance-1-dp0.json").write_text("{}")
    args = SimpleNamespace(output=target, fit=Path("fit.json"), prepared_prompts={"huge": [1]},
                           prompt_bank_data={"huge": [2]}, capacity_pilot=False)
    runner.runtime_copy(source, target, args, {"requests": []}, capacity=True)
    config = json.loads((target / "config.json").read_text())
    assert config["capacity_pilot"] is True and "prepared_prompts" not in config
    assert "prompt_bank_data" not in config and (target / "worker-instance-1-dp0.json").exists()


def test_selection_requires_both_baselines_and_keeps_negative_cases(modules):
    runner, analyzer = modules
    cases = [{"setting": {"id": identifier}, "analysis": {"minimum_Code_p95_reduction_fraction": value}}
             for identifier, value in enumerate((-.2, .3, None, .6))]
    assert analyzer.select_best(cases)["setting"]["id"] == 3
    assert len(cases) == 4 and cases[0]["analysis"]["minimum_Code_p95_reduction_fraction"] == -.2
    assert analyzer.select_best([cases[2]]) is None


def test_long_output_profile_does_not_transform_half_million_marks_for_one_prompt(modules):
    runner, analyzer = modules
    import inspect

    source = inspect.getsource(runner.profile_output)
    assert "long_output_marks([representative], output_tokens)" in source


def test_cli_launch_retains_sparse_and_has_no_old_frozen_replay():
    root = Path(__file__).resolve().parents[1] / "experiments/dp_opt"
    text = (root / "stress-sweep.sbatch").read_text()
    assert "DP_OPT_FROZEN_TRACE_REPLAY=0" in text and "DP_OPT_STRESS_SWEEP=1" in text
    assert "DP_OPT_KV_TRANSPORT=dp_echo" in text and "131072" in text
    assert "agentic-formal.sbatch" not in text and "agentic-trace.sbatch" not in text


def test_window_selection_keeps_warmup_conv_in_old_request_cost(modules, monkeypatch, tmp_path):
    runner, analyzer = modules
    trace = {"evaluation_start_s": 600, "duration_s": 1800,
             "requests": [{"request_id": "old", "model": 1, "arrival_s": 100},
                          {"request_id": "code", "model": 0, "arrival_s": 700}]}
    (tmp_path / "trace.json").write_text(json.dumps(trace))
    (tmp_path / "config.json").write_text(json.dumps({"deadline_s": 120}))
    monkeypatch.setattr(analyzer, "analyze", lambda *args: {"all_warmup_records_retained": True})
    for policy, origin, code_latency, old_latency in (("no_move", 1000, 200, 800),
                                                     ("no_move_fair", 2000, 200, 800),
                                                     ("kv_only", 3000, 80, 900)):
        directory = tmp_path / f"r0-{policy}"
        directory.mkdir()
        requests = []
        for identifier, model, offset, latency in (("old", 1, 100, old_latency), ("code", 0, 700, code_latency)):
            arrived = origin + offset
            requests.append({"request_id": identifier, "model": model, "arrived_at": arrived,
                             "reserved_at": arrived, "submitted_at": arrived,
                             "first_token_at": arrived + 1, "completed_at": arrived + latency,
                             "token_ids": [1, 2]})
        (directory / "requests.json").write_text(json.dumps(requests))
    result = analyzer.analyze_case(tmp_path)
    assert result["Code_samples"] == 1 and result["minimum_Code_p95_reduction_fraction"] == pytest.approx(.6)
    assert result["observed_50pct_both_baselines"] is True and result["large_sample_or_heldout_confirmation"] is False
    for comparison in result["comparisons"].values():
        assert comparison["evaluation_window"]["all_requests"]["requests"] == 1
        assert comparison["evaluation_window"]["old_Conv"]["request_ids"] == ["old"]
        assert comparison["evaluation_window"]["old_Conv"]["paired_latency_sum_delta_s"] == 100


def test_heldout_pools_seed_clusters_and_retains_conv_penalty(modules, monkeypatch, tmp_path):
    runner, analyzer = modules
    monkeypatch.setattr(analyzer, "analyze", lambda *args: {})
    roots, seeds = [], list(range(8))
    for seed in seeds:
        root = tmp_path / str(seed)
        root.mkdir()
        roots.append(root)
        marks = [{"request_id": f"code-{index}", "model": 0, "arrival_s": 700 + index} for index in range(13)]
        marks.append({"request_id": "old-conv", "model": 1, "arrival_s": 100})
        trace = {"seed": seed, "duration_s": 16600, "evaluation_start_s": 600,
                 "prompt_bank_sha256": "a" * 64, "requests": marks}
        setting = {"seed": seed, "output_tokens": 8192, "budget_mib_per_gpu": 10240,
                   "conv_rate_rps": .023, "code_rate_rps": .001, "fixed_validation_conv_rate_rps": .023}
        (root / "trace.json").write_text(json.dumps(trace))
        (root / "config.json").write_text(json.dumps({"deadline_s": 120}))
        (root / "stress-setting.json").write_text(json.dumps(setting))
        for policy in ("no_move", "no_move_fair", "kv_only"):
            directory = root / f"r0-{policy}"
            directory.mkdir()
            records = []
            for mark in marks:
                latency = (80 if policy == "kv_only" else 200) if mark["model"] == 0 else (900 if policy == "kv_only" else 800)
                arrived = 10000 + mark["arrival_s"]
                records.append({**mark, "arrived_at": arrived, "submitted_at": arrived,
                    "reserved_at": arrived, "first_token_at": arrived + 1,
                    "completed_at": arrived + latency, "token_ids": [1, 2]})
            (directory / "requests.json").write_text(json.dumps(records))
    result = analyzer.analyze_confirmations(roots, seeds, bootstrap_repeats=30)
    assert result["Code_samples"] == 104 and result["confirmed_50pct_Code_p95_both_baselines"]
    assert result["pooled_comparisons"]["no_move"]["old_Conv"]["paired_latency_sum_delta_s"] == 800
    assert result["paired_seed_cluster_bootstrap"]["minimum_reduction_interval_95"] == pytest.approx([.6, .6])
    with pytest.raises(ValueError, match="every predeclared"):
        analyzer.analyze_confirmations(roots[:-1], seeds)
    setting["conv_rate_rps"] = .04
    (roots[-1] / "stress-setting.json").write_text(json.dumps(setting))
    with pytest.raises(ValueError, match="fixed rate"):
        analyzer.analyze_confirmations(roots, seeds)


def test_independent_profile_checks_owner_geometry_bytes_and_tokens(modules, tmp_path):
    import copy

    verifier = importlib.import_module("verify_stress_sweep")

    def rank(identifiers):
        return {"requests": [{"id": identifier, "blocks": [index + 1]} for index, identifier in enumerate(identifiers)],
                "pages": [{"id": 0, "owners": sorted(identifiers)}], "mapped_bytes": 96 * 1024**2,
                "page_bytes": 96 * 1024**2, "block_bytes": int(1.5 * 1024**2), "block_size": 16,
                "page_block_capacity": 64}

    before = [[rank([]), rank([])], [rank(["conv-0", "conv-2"]), rank(["conv-1", "conv-3"])]]
    after = [[rank([]), rank([])], [rank([]), rank(["conv-0", "conv-1", "conv-2", "conv-3"])]]
    references = [{"model": 1, "prompt_tokens": 3949, "token_ids": [1] * 8192},
                  {"model": 0, "prompt_tokens": 101000, "token_ids": [2] * 64}]
    trials = []
    for layout in ("balanced", "consolidated", "migrated"):
        requests = [{**references[0], "submitted_at": 0, "completed_at": 10} for index in range(4)]
        if layout != "balanced":
            requests.append({**references[1], "submitted_at": 0, "completed_at": 60})
        trial = {"layout": layout, "decode_ready": before, "budgets_gib": [10, 10], "requests": requests}
        if layout == "migrated":
            trial.update({"before": before, "after": after,
                "probe_after": {"allocated": True, "fits_budget": True, "rollback_verified": True},
                "probes_before": [{"fits_budget": False, "rollback_verified": True}] * 2,
                "events": [{"request_id": identifier, "model": 1, "source": 0, "destination": 1,
                    "bitwise_verified": True, "kv_bytes": int(1.5 * 1024**2), "copy_peak_gib": [1, 1]}
                    for identifier in ("conv-0", "conv-2")]})
        trials.append(trial)
    setting = {"Conv": {"output_tokens": 8192}, "Code": {"output_tokens": 64, "prompt_tokens": 101000}}

    def write(trials):
        for name, value in (("setting", setting), ("references", references), ("trials", trials),
                            ("summary", {"status": "profile_passed", "trials": trials,
                                         "rate": {"representative_service_s": 10}})):
            (tmp_path / f"{name}.json").write_text(json.dumps(value))

    write(trials)
    assert verifier.verify_profile(tmp_path)["both_before_probes_blocked"]
    broken = copy.deepcopy(trials)
    broken[-1]["events"][0]["kv_bytes"] *= 2
    write(broken)
    with pytest.raises(ValueError, match="migration lacks"):
        verifier.verify_profile(tmp_path)
    broken = copy.deepcopy(trials)
    broken[0]["decode_ready"][0][0]["block_bytes"] = 2
    write(broken)
    with pytest.raises(ValueError, match="geometry"):
        verifier.verify_profile(tmp_path)
