import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def pilot(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("layout_cost_pilot")


def test_layout_cases_measure_all_counts_both_orientations_without_population_extrapolation(pilot):
    cases = pilot.layout_cases()
    assert cases == pilot.layout_cases()
    assert len(cases) == 144
    assert sum(case["population"] for case in cases) == 1664
    for population in pilot.POPULATIONS:
        for repeat in range(2):
            assert {case["rank_zero"] for case in cases if case["population"] == population
                    and case["repeat"] == repeat} == set(range(population + 1))


def test_cost_lookup_uses_measured_cadence_and_rejects_missing_or_changed_cases(pilot):
    trials = []
    for case in pilot.layout_cases():
        token_s = .01 + .001 * abs(2 * case["rank_zero"] - case["population"])
        trials.append({"case": case, "requests": [{"cadence": [
            {"tokens": 256, "time": 1}, {"tokens": 768, "time": 1 + 512 * token_s}]}] * case["population"]})
    tables = pilot.cost_tables(trials)
    assert tables["tables"]["8"]["cadence_s"][4] == pytest.approx(.01)
    assert tables["tables"]["8"]["cadence_s"][0] == pytest.approx(.018)
    assert "not_Code_prefill" in tables["scope"]
    analyzer = importlib.import_module("analyze_layout_cost")
    diagnostics = analyzer.summarize(tables, trials)
    assert diagnostics["populations"][0]["balanced_cadence_ms"] == pytest.approx(10)
    assert diagnostics["populations"][0]["descriptive_affine_alpha"] == pytest.approx(.8)
    assert diagnostics["populations"][0]["descriptive_affine_max_abs_ratio_residual"] < 1e-12
    assert diagnostics["populations"][0]["extreme_throughput_loss_fraction"] == pytest.approx([4 / 9] * 2)
    with pytest.raises(ValueError):
        pilot.cost_tables(trials[:-1])
    trials[0]["requests"][0]["cadence"][-1]["tokens"] = 256
    with pytest.raises((ValueError, StopIteration)):
        pilot.cost_tables(trials)


def test_measured_layout_cost_preserves_differences_and_refuses_unmeasured_population(pilot):
    lookup = {"status": "raw_verified_small_population_lookup", "tables": {"8": {
        "cadence_s": [.02, .02, .02, .015, .01, .015, .02, .02, .02], "all_counts_measured": True}}}
    table = pilot.measured_table(lookup, 8, 100)
    assert table[4] == 0 and table[0] == pytest.approx(8)
    with pytest.raises(ValueError):
        pilot.measured_table(lookup, 9, 100)
    lookup["tables"]["8"]["cadence_s"][0] = float("nan")
    with pytest.raises(ValueError):
        pilot.measured_table(lookup, 8, 100)


@pytest.mark.parametrize("fault", [None, "owner", "tokens", "count", "graph"])
def test_layout_raw_verifier_requires_all_population_owner_token_graph_evidence(pilot, tmp_path, monkeypatch, fault):
    def save(name, value):
        (tmp_path / name).write_text(json.dumps(value))

    save("config.json", {"max_model_len": 131072, "cache_bytes": 24 * 1024**3,
                         "kv_budget_mib": 11328, "allocator_gpu_utilization": .98})
    cases = pilot.layout_cases()
    save("summary.json", {"status": "layout_cost_pilot_passed", "cases": len(cases)})
    save("layout-cases.json", cases)
    save("layout-mark.json", {"prompt_tokens": 3584, "output_tokens": 1024, "source_internal_row": 1234})
    save("layout-reference.json", {"token_ids": [1] * 1024})
    save("runtime-control.json", {"dp_pause_sync_interval": 32, "same_interval_for_all_policies": True,
                                 "idle_rank_expert_execution_not_disabled": True})
    for model in (0, 1):
        for rank in (0, 1):
            save(f"worker-instance-{model + 1}-dp{rank}.json", {"dp_pause_sync_interval": 32,
                "ipc_name": f"pool-{model}-{rank}", "gpu_name": "A800", "gpu_uuid": f"GPU{rank}",
                "checkpoint": "qwen", "max_model_len": 131072, "kv_cache_memory_bytes": 24 * 1024**3,
                "groups": {name: {"world_size": size} for name, size in (("dp", 2), ("ep", 2), ("tp", 1))}})

    def snapshot(count, minimum, offset):
        requests = [{"id": f"request-{offset}-{position}", "blocks": [position + 64],
                     "movable": True, "output_tokens": minimum} for position in range(count)]
        return {"requests": requests, "pages": [{"id": 0, "owners": [], "pinned": True}]
                + ([{"id": 1, "owners": sorted(request["id"] for request in requests), "pinned": False}]
                   if count else []), "page_bytes": 96 * 1024**2,
                "mapped_bytes": (2 if count else 1) * 96 * 1024**2, "page_block_capacity": 64}

    trials = [{"case": case, "requests": [{"request_id": f"request-{position}", "model": 1,
                    "rank_initial": int(position >= case["rank_zero"]), "prompt_tokens": 3584,
                    "first_token_at": .5, "completed_at": 8,
                    "token_ids": [1] * 1024, "cadence": [{"tokens": 256, "time": 1},
                                                       {"tokens": 768, "time": 7}]} for position in range(case["population"])],
               "window_observations": [[[snapshot(0, minimum, 0), snapshot(0, minimum, 0)],
                    [snapshot(case["rank_zero"], minimum, 0),
                     snapshot(case["population"] - case["rank_zero"], minimum, 1)]] for minimum in (256, 768)]}
              for case in cases]
    if fault == "tokens":
        trials[0]["requests"][0]["token_ids"][0] = 2
    elif fault == "owner":
        active = next(rank for rank in trials[0]["window_observations"][0][1] if rank["requests"])
        active["pages"][1]["owners"] = []
    elif fault == "count":
        trials[0]["requests"].pop()
    save("layout-trials.json", trials)
    captures = [[[{"histogram": [{"mode": "FULL", "steps": 100}]}] for rank in (0, 1)] for model in (0, 1)]
    if fault == "graph":
        captures[0][1][0]["histogram"][0]["steps"] = 0
    save("layout-captures.json", captures)
    (tmp_path / "exit-status").write_text("0")
    if fault:
        with pytest.raises(ValueError):
            pilot.verify(tmp_path, tmp_path / "lookup.json")
    else:
        report = pilot.verify(tmp_path, tmp_path / "lookup.json")
        assert report["cases"] == 144 and report["requests"] == 1664
        assert json.loads((tmp_path / "lookup.json").read_text()) == report
        assert set(report["source_files_sha256"]) == set(pilot.LOOKUP_SOURCE_FILES)
        import hashlib

        digest = hashlib.sha256((tmp_path / "lookup.json").read_bytes()).hexdigest()
        assert pilot.load_verified_lookup(tmp_path / "lookup.json", digest, tmp_path) == report
        with pytest.raises(ValueError, match="frozen SHA256"):
            pilot.load_verified_lookup(tmp_path / "lookup.json", "0" * 64, tmp_path)
        with pytest.raises(ValueError, match="explicitly frozen"):
            pilot.load_verified_lookup(tmp_path / "lookup.json", None, tmp_path)
        (tmp_path / "layout-mark.json").write_text((tmp_path / "layout-mark.json").read_text() + " ")
        with pytest.raises(ValueError, match="raw sources"):
            pilot.load_verified_lookup(tmp_path / "lookup.json", digest, tmp_path)


def test_shared_window_refuses_prefill_growth_and_completion_drain(pilot):
    requests = [{"first_token_at": .5, "completed_at": 8,
                 "cadence": [{"tokens": 256, "time": 1}, {"tokens": 768, "time": 7}]} for index in (0, 1)]
    pilot.verify_shared_decode_window(requests)
    requests[1]["first_token_at"] = 2
    with pytest.raises(ValueError, match="population growth"):
        pilot.verify_shared_decode_window(requests)
    requests[1]["first_token_at"] = .5
    requests[1]["completed_at"] = 6
    with pytest.raises(ValueError, match="drain"):
        pilot.verify_shared_decode_window(requests)
