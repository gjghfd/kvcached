import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_imbalance import analyze, scan_cases, steady_step


def write_scan(root, num_ranks=4):
    config = {"seed": 1, "context_tokens": 4096, "output_tokens": 1024, "repeats": 2,
              "case_timeout_s": 120, "max_scan_s": 2800, "series": [{"batch_size": 4, "deltas": [0, 1]}]}
    config.update(num_ranks=num_ranks, tensor_parallel_size=4 // num_ranks)
    cases = scan_cases(config, num_ranks)
    (root / "plan.json").write_text(json.dumps({"config": config, "cases": cases}))
    records = []
    for case in cases:
        requests = [{"request_id": f"{rank}-{index}", "rank_initial": rank, "model": 1,
                     "prompt_tokens": 4096, "output_tokens": 1024, "first_token_at": 1.0, "completed_at": 15.0,
                     "cadence": [[2.0, 64], [6.0, 256], [12.0, 1024]]}
                    for rank, count in enumerate(case["layout"]) for index in range(count)]
        filename = f"{case['name']}-requests.json"
        (root / filename).write_text(json.dumps(requests))
        step, window = steady_step(requests)
        records.append({**case, "status": "measured", "preemptions": 0, "requests_file": filename,
                        "step_s": step, "window": window, "common_start": 2.0, "common_end": 14.0})
    (root / "results.json").write_text(json.dumps(records))
    (root / "summary.json").write_text(json.dumps({"status": "passed", "planned": 4, "completed": 4, "measured": 4}))
    return records


@pytest.mark.parametrize("num_ranks", [2, 4])
def test_raw_cadence_verified_and_repeats_summarized(tmp_path, num_ranks):
    write_scan(tmp_path, num_ranks)
    report = analyze(tmp_path)
    assert report["num_ranks"] == num_ranks and report["expert_parallel_size"] == 4
    assert report["trials"] == 4 and report["requests"] == 16
    assert len(report["points"]) == 2
    assert report["points"][1]["normalized_imbalance"] == 2 / (4 / num_ranks)
    assert all(len(point["samples_ms"]) == 2 and point["change_pct"] == 0 for point in report["points"])


@pytest.mark.parametrize("damage", ["step", "missing", "preemption", "placement"])
def test_inconsistent_measurements_rejected(tmp_path, damage):
    records = write_scan(tmp_path)
    if damage == "step":
        records[0]["step_s"] *= 2
    elif damage == "missing":
        records.pop()
    elif damage == "preemption":
        records[0]["preemptions"] = 1
    else:
        path = tmp_path / records[0]["requests_file"]
        requests = json.loads(path.read_text())
        requests[0]["rank_initial"] = 99
        path.write_text(json.dumps(requests))
    (tmp_path / "results.json").write_text(json.dumps(records))
    with pytest.raises(ValueError):
        analyze(tmp_path)
