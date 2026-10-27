import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def verifier(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "experiments/dp_opt"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("dp_opt_raw_verifier", root / "verify_motivation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def raw_case(verifier, tmp_path):
    from test_dp_opt_motivation import trace

    (tmp_path / "trials").mkdir()
    (tmp_path / "traces").mkdir()
    report = {"status": "passed", "protocol": {"repeats": 3, "requests": 16, "extremes_only": True,
              "trim_steps_each_end": 32, "minimum_stable_steps": 128, "input_tokens": 128,
              "output_tokens": 256, "expected_cudagraph_mode": "NONE"}, "trials": []}
    for planned in verifier.build_plan(extremes_only=True):
        traces = [trace(rank, count) for rank, count in enumerate(planned["layout"])]
        window = verifier.stable_window(traces, planned["layout"])
        requests = [{"rank": rank, "status": "passed", "finish_reason": "length", "token_ids": [42] * 256,
                     "usage": {"prompt_tokens": 128, "completion_tokens": 256}, "max_tokens_per_chunk": 1,
                     "token_events": [{"time_ns": step * 100000000 + 10000000, "tokens": 1} for step in range(256)]}
                    for rank, count in enumerate(planned["layout"]) for index in range(count)]
        memory = {str(rank): {"mapped_live_bytes": 96 * 1024 ** 2, "reserved_backing_bytes": 0} for rank in (0, 1)}
        metrics = {str(rank): {"vllm:num_requests_running": count, "vllm:num_requests_waiting": 0,
                              "vllm:num_preemptions_total": 0, "vllm:kv_cache_usage_perc": 0.2}
                   for rank, count in enumerate(planned["layout"])}
        trial = {**planned, "requests": requests, "stable_window": window,
                 "stable_client_itl_s": verifier.client_window(requests, window["start_ns"], window["end_ns"]),
                 "summary": {"output_tokens_per_second": 160}, "memory_before": memory, "memory_after": memory,
                 "metrics_before": metrics, "metrics_after": metrics,
                 "samples": [{"time_ns": window["start_ns"] + 10000000, "metrics": metrics}]}
        report["trials"].append(trial)
        (tmp_path / "trials" / f'{trial["id"]}.json').write_text(json.dumps(trial))
        for rank, item in enumerate(traces):
            (tmp_path / "traces" / f'{trial["id"]}-dp{rank}.json').write_text(json.dumps(item))
    report["summary"] = verifier.paired_summary(report["trials"])
    (tmp_path / "results.json").write_text(json.dumps(report))
    return tmp_path


def test_raw_case_is_independently_reproduced(verifier, raw_case):
    result = verifier.verify_case(raw_case)
    assert result["status"] == "raw_verified"
    assert result["trials"] == 12
    assert result["requests"] == 192
    assert result["output_tokens"] == 49152


@pytest.mark.parametrize("corruption", ("tokens", "reclamation", "population"))
def test_raw_corruption_is_rejected(verifier, raw_case, corruption):
    path = next((raw_case / "trials").glob("*.json"))
    trial = json.loads(path.read_text())
    if corruption == "tokens":
        trial["requests"][0]["token_ids"].pop()
    elif corruption == "reclamation":
        trial["memory_after"]["0"]["mapped_live_bytes"] += 1
    else:
        trial["samples"][0]["metrics"]["0"]["vllm:num_requests_running"] += 1
    path.write_text(json.dumps(trial))
    with pytest.raises(ValueError):
        verifier.verify_case(raw_case)
