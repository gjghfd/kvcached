import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def pilot(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("workaware_pilot")


def test_cases_are_repeated_direction_paired_and_not_population_context_extrapolation(pilot):
    cases = pilot.calibration_cases()
    assert cases == pilot.calibration_cases()
    assert {case["model"] for case in cases} == {0, 1}
    for case in cases:
        assert {**case, "rank_zero": case["population"] - case["rank_zero"]} in cases
        assert {**case, "repeat": 1 - case["repeat"]} in cases
        assert case["population"] * (case["prompt_tokens"] + pilot.OUTPUT_TOKENS) * 96 * 1024 < 10 * 1024**3
    assert {case["population"] for case in cases} == {2, 3, 4, 8, 9}
    assert max(case["prompt_tokens"] for case in cases) == 22528


def request(identifier="request", rank=0, prompt=4096, cadence_s=.02):
    return {"request_id": identifier, "model": 0, "rank_initial": rank, "prompt_tokens": prompt,
            "first_token_at": .5, "completed_at": 80., "token_ids": [1] * 3072,
            "cadence": [{"tokens": 1536, "time": 1.}, {"tokens": 2560, "time": 1. + 1024 * cadence_s}]}


def test_stable_sample_uses_actual_context_and_full_population_window(pilot):
    result = pilot.stable_sample([request(), request("second", 1, 8192)])
    assert result["counts"] == [1, 1]
    assert result["context_tokens"] == [6143., 10239.]
    assert result["cadence_s"] == pytest.approx(.02)
    requests = [request(), request("second", 1)]
    requests[1]["first_token_at"] = 2.
    with pytest.raises(ValueError, match="growth"):
        pilot.stable_sample(requests)
    requests[1]["first_token_at"] = .5
    requests[1]["completed_at"] = 2.
    with pytest.raises(ValueError, match="drain"):
        pilot.stable_sample(requests)


@pytest.fixture
def raw(pilot, tmp_path):
    def save(name, value):
        (tmp_path / name).write_text(json.dumps(value))

    cases = pilot.calibration_cases()
    save("config.json", {"max_model_len": 131072, "cache_bytes": 24 * 1024**3,
                         "kv_budget_mib": 10240, "allocator_gpu_utilization": .98})
    save("summary.json", {"status": "workaware_cadence_pilot_passed", "cases": len(cases),
                          "capture_sizes": pilot.SPARSE_CAPTURE_SIZES, "scope": "one_active_tenant_other_resident_idle"})
    save("workaware-cases.json", cases)
    lengths = {case["prompt_tokens"] for case in cases}
    save("workaware-prompts.json", {str(length): [1] * length for length in lengths})
    references = {f"{model}:{length}": {**request(prompt=length), "model": model}
                  for model in (0, 1) for length in lengths}
    save("workaware-references.json", references)
    save("runtime-control.json", {"dp_pause_sync_interval": 32, "same_interval_for_all_policies": True,
                                  "idle_rank_expert_execution_not_disabled": True})
    for model in (0, 1):
        for rank in (0, 1):
            save(f"worker-instance-{model + 1}-dp{rank}.json", {"dp_pause_sync_interval": 32,
                "ipc_name": f"pool-{model}-{rank}", "gpu_name": "A800", "gpu_uuid": f"GPU{rank}",
                "service": f"instance-{model + 1}", "checkpoint": "qwen", "max_model_len": 131072,
                "kv_cache_memory_bytes": 24 * 1024**3, "max_num_batched_tokens": 4096,
                "kv_block_size": 16,
                "expert_ids": [list(range(rank * 64, (rank + 1) * 64))],
                "groups": {name: {"world_size": size} for name, size in (("dp", 2), ("ep", 2), ("tp", 1))}})
            save(f"workaware-engine-instance-{model + 1}-dp{rank}.json", {"service": f"instance-{model + 1}",
                "dp_rank": rank, "dtype": "torch.bfloat16", "compile_mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": pilot.SPARSE_CAPTURE_SIZES,
                "max_num_batched_tokens": 4096})

    def observed(requests, threshold, prompt):
        demand = (prompt + 3072 + 15) // 16
        records = [{"id": record["request_id"], "blocks": list(range(64 + position * demand, 64 + (position + 1) * demand)), "movable": True,
                    "output_tokens": threshold, "max_tokens": 3072, "computed_tokens": prompt + threshold - 1}
                   for position, record in enumerate(requests)]
        pages = [{"id": 0, "owners": [], "pinned": True}]
        for page in sorted({block // 64 for record in records for block in record["blocks"]}):
            pages.append({"id": page, "owners": sorted(record["id"] for record in records if any(
                block // 64 == page for block in record["blocks"])), "pinned": False})
        return {"requests": records, "pages": pages, "page_bytes": 96 * 1024**2,
                "mapped_bytes": len(pages) * 96 * 1024**2, "page_block_capacity": 64,
                "block_size": 16, "block_bytes": 1536 * 1024, "uniform_page_geometry": True}

    from motivation import graph_padding

    trials = []
    for index, case in enumerate(cases):
        counts = [case["rank_zero"], case["population"] - case["rank_zero"]]
        peak = max(counts) * (case["prompt_tokens"] + 2047)
        cadence = .01 + .001 * graph_padding(counts, pilot.SPARSE_CAPTURE_SIZES) + peak * 1e-7
        requests = [{**request(f"case-{index}-request-{position}", int(position >= case["rank_zero"]),
                              case["prompt_tokens"], cadence), "model": case["model"]}
                    for position in range(case["population"])]
        windows = []
        for threshold in pilot.WINDOW:
            windows.append([[observed([record for record in requests if record["rank_initial"] == rank]
                                      if model == case["model"] else [], threshold, case["prompt_tokens"])
                             for rank in (0, 1)] for model in (0, 1)])
        trials.append({"case": case, "requests": requests, "window_observations": windows,
                       "sample": pilot.stable_sample(requests)})
    save("workaware-trials.json", trials)
    save("workaware-captures.json", [[[{'histogram': [{"mode": "FULL", "steps": 100}]}]
                                     for rank in (0, 1)] for model in (0, 1)])
    (tmp_path / "exit-status").write_text("0\n")
    evidence = importlib.import_module("workaware_evidence")
    evidence.archive_runtime(tmp_path)
    manifest = json.loads((tmp_path / "workaware-runtime-source.json").read_text())
    (tmp_path / "source.sha256").write_text("".join(
        f"{source['sha256']}  {source['path']}\n" for source in manifest["sources"].values()))
    return tmp_path


@pytest.mark.parametrize("fault", [None, "tokens", "owner", "count", "window", "graph", "dense", "precision", "budget", "fit"])
def test_raw_verification_and_loading_cannot_trust_status_strings(pilot, raw, fault):
    verifier = importlib.import_module("verify_workaware_pilot")

    def change(name, mutate):
        value = json.loads((raw / name).read_text())
        mutate(value)
        (raw / name).write_text(json.dumps(value))

    if fault == "tokens":
        change("workaware-trials.json", lambda trials: trials[0]["requests"][0]["token_ids"].__setitem__(0, 2))
    elif fault == "owner":
        def break_owner(trials):
            active = next(rank for rank in trials[0]["window_observations"][0][trials[0]["case"]["model"]] if rank["requests"])
            active["pages"][1]["owners"] = []
        change("workaware-trials.json", break_owner)
    elif fault == "count":
        change("workaware-trials.json", lambda trials: trials[0]["requests"].pop())
    elif fault == "window":
        change("workaware-trials.json", lambda trials: trials[0]["requests"][0].__setitem__("first_token_at", 2.))
    elif fault == "graph":
        change("workaware-captures.json", lambda captures: captures[0][1][0]["histogram"][0].__setitem__("steps", 0))
    elif fault in ("dense", "precision"):
        change("workaware-engine-instance-1-dp0.json", lambda engine: engine.__setitem__(
            "dtype" if fault == "precision" else "cudagraph_capture_sizes", "torch.float16" if fault == "precision" else [1, 2, 3, 4]))
    elif fault == "budget":
        change("config.json", lambda config: config.__setitem__("kv_budget_mib", 11328))
    if fault and fault != "fit":
        with pytest.raises(ValueError):
            verifier.verify(raw)
        return
    fit = verifier.verify(raw, raw / "fit.json")
    assert fit["coefficients"]["attention_token_s"] == pytest.approx(1e-7)
    assert fit["coefficients"]["graph_batch_s"] == pytest.approx(.001)
    assert fit["calibrated_models"] == [0, 1]
    digest = hashlib.sha256((raw / "fit.json").read_bytes()).hexdigest()
    assert pilot.load_verified_fit(raw / "fit.json", digest, raw) == fit
    if fault == "fit":
        fit["coefficients"]["attention_token_s"] = 1e-3
        (raw / "fit.json").write_text(json.dumps(fit))
        digest = hashlib.sha256((raw / "fit.json").read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="reproduce"):
            pilot.load_verified_fit(raw / "fit.json", digest, raw)
    else:
        with pytest.raises(ValueError, match="frozen SHA256"):
            pilot.load_verified_fit(raw / "fit.json", "0" * 64, raw)
        (raw / "source.sha256").write_text("changed source\n")
        with pytest.raises(ValueError, match="raw sources"):
            pilot.load_verified_fit(raw / "fit.json", digest, raw)
