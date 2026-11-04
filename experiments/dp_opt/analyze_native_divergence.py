import argparse
import hashlib
import json
from pathlib import Path

from batch_numerics_probe import EXPECTED_FAILURES


def analyze(root):
    root = Path(root)
    error = json.loads((root / "error.json").read_text())
    if error["type"] != "RuntimeError" or error["message"] not in EXPECTED_FAILURES:
        raise ValueError("Diagnose unrelated native failures separately")
    trace = json.loads((root / "trace.json").read_text())
    marks = {mark["request_id"]: mark for mark in trace["requests"]}
    references = {}
    for reference in json.loads((root / "references.json").read_text()):
        identifier = reference["trace_request_id"]
        if identifier in references and references[identifier]["token_ids"] != reference["token_ids"]:
            raise ValueError("Duplicate serial references already disagree")
        references[identifier] = reference
    if set(references) != set(marks):
        raise ValueError("This diagnosis requires full serial references for every original request")
    order = json.loads((root / "order.json").read_text())
    if "no_move" not in {item["policy"] for item in order}:
        raise ValueError("Require the unchanged no-move control")
    trials = []
    for item in order:
        directory = root / f'r{item["repeat"]}-{item["policy"]}'
        records = json.loads((directory / "requests.json").read_text())
        if len(records) != len(marks) or {record["request_id"] for record in records} != set(marks):
            raise ValueError("Native diagnosis cannot omit undrained requests")
        comparisons = []
        for record in records:
            mark = marks[record["request_id"]]
            expected = references[record["request_id"]]["token_ids"]
            actual = record["token_ids"]
            if record["prompt_tokens"] != mark["prompt_tokens"] or len(actual) != mark["output_tokens"] or len(expected) != len(actual):
                raise ValueError("Native diagnosis must retain complete original input/output")
            differences = [index for index, (left, right) in enumerate(zip(actual, expected)) if left != right]
            position = differences[0] if differences else None
            cadence = next((sample for sample in record["cadence"]
                            if position is not None and sample["tokens"] > position), None)
            comparisons.append({"request_id": record["request_id"], "model": mark["model"],
                                "prompt_tokens": mark["prompt_tokens"], "output_tokens": len(actual),
                                "different_tokens": len(differences), "first_difference_zero_based": position,
                                "actual_token": actual[position] if position is not None else None,
                                "reference_token": expected[position] if position is not None else None,
                                "first_difference_offset_s": cadence["time"] - record["arrived_at"] + mark["arrival_s"]
                                if cadence is not None else None})
        hops = json.loads((directory / "migrations.json").read_text())
        trials.append({**item, "comparisons": comparisons,
                       "migration_hops": [{name: event.get(name) for name in
                                           ("request_id", "model", "source", "destination", "purpose",
                                            "transport", "integrity", "bitwise_verified", "kv_bytes")} for event in hops]})
    return {"status": "full_native_output_failure_diagnosed_not_benefit", "error": error,
            "trace_sha256": hashlib.sha256((root / "trace.json").read_bytes()).hexdigest(),
            "requests_verified": sum(len(trial["comparisons"]) for trial in trials), "trials": trials,
            "is_algorithm_benefit": False, "migration_correctness_proven": False,
            "numerical_root_cause_proven": False, "strict_token_check_relaxed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "requests_verified": result["requests_verified"]}))


if __name__ == "__main__":
    main()
