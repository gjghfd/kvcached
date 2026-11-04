import json
import hashlib
import time
import sys
from pathlib import Path

import live_trace


EXPECTED_FAILURES = {
    "Selected trace request differs from its full serial greedy reference",
    "Paired policy trace token IDs differ from no-move baseline",
}


def serialize_logprobs(values):
    if values is None or any(not value for value in values):
        raise RuntimeError("Numerics probe requires complete per-token logprobs")
    return [{str(token): float(value.logprob) for token, value in frame.items()}
            for frame in values]


async def stream_with_logprobs(engine, request_id, prompt, output_tokens, rank, model, role, arrived_at):
    from vllm import SamplingParams

    submitted_at = time.monotonic()
    cadence, latest, first_token_at = [], None, None
    parameters = SamplingParams(temperature=0, min_tokens=output_tokens,
                                max_tokens=output_tokens, ignore_eos=True, logprobs=2)
    async for output in engine.generate({"prompt_token_ids": prompt}, parameters, request_id,
                                        data_parallel_rank=rank):
        now = time.monotonic()
        latest = output
        count = len(output.outputs[0].token_ids)
        if count and first_token_at is None:
            first_token_at = now
        cadence.append({"time": now, "tokens": count})
    if latest is None or len(latest.outputs[0].token_ids) != output_tokens or first_token_at is None:
        raise RuntimeError("Request failed exact output and first-token check")
    logprobs = serialize_logprobs(latest.outputs[0].logprobs)
    if len(logprobs) != output_tokens:
        raise RuntimeError("Numerics probe logprob count differs from output length")
    return {"request_id": request_id, "model": model, "role": role, "rank_initial": rank,
            "prompt_tokens": len(prompt), "arrived_at": arrived_at, "submitted_at": submitted_at,
            "first_token_at": first_token_at, "completed_at": time.monotonic(),
            "token_ids": list(latest.outputs[0].token_ids), "cadence": cadence,
            "token_logprobs": logprobs}


def compare_tokens(actual, reference):
    tokens, expected = actual["token_ids"], reference["token_ids"]
    if len(tokens) != len(expected):
        raise ValueError("Numerics probe cannot discard or truncate output tokens")
    differences = [index for index, (token, original) in enumerate(zip(tokens, expected))
                   if token != original]
    result = {"request_id": actual["request_id"], "model": actual["model"],
              "output_tokens": len(tokens), "different_tokens": len(differences),
              "first_difference_zero_based": differences[0] if differences else None}
    if not differences:
        return result
    position = differences[0]
    before = reference["token_logprobs"][position]
    after = actual["token_logprobs"][position]
    common = sorted(before.keys() & after.keys(), key=int)
    reference_top = sorted(before.values(), reverse=True)
    actual_top = sorted(after.values(), reverse=True)
    result.update({"shared_prefix_tokens": position, "reference_token": expected[position],
                   "actual_token": tokens[position], "reference_top_logprobs": before,
                   "actual_top_logprobs": after, "common_candidate_ids": common,
                   "common_top_logprob_max_abs_delta": max(
                       (abs(before[token] - after[token]) for token in common), default=None),
                   "reference_top2_margin": reference_top[0] - reference_top[1]
                       if len(reference_top) > 1 else None,
                   "actual_top2_margin": actual_top[0] - actual_top[1]
                       if len(actual_top) > 1 else None,
                   "numerical_tolerance_acceptance_not_implemented": True})
    return result


def analyze(root):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    trace = json.loads((root / "trace.json").read_text())
    if config["policies"] != ["no_move"] or config["repeats"] != 2:
        raise ValueError("Require two identical no-move repetitions, not a benefit experiment")
    marks = {mark["request_id"]: mark for mark in trace["requests"]}
    references = json.loads((root / "references.json").read_text())
    if not references:
        raise ValueError("Require original full serial references")
    trials, records_by_repeat = [], []
    for repeat in range(2):
        directory = root / f"r{repeat}-no_move"
        records = json.loads((directory / "requests.json").read_text())
        by_id = {record["request_id"]: record for record in records}
        if len(records) != len(marks) or set(by_id) != set(marks):
            raise ValueError("No-move probe must drain every unchanged original request")
        if json.loads((directory / "migrations.json").read_text()):
            raise ValueError("Numerics control must not migrate KV")
        for identifier, record in by_id.items():
            mark = marks[identifier]
            if (record["prompt_tokens"] != mark["prompt_tokens"]
                    or len(record["token_ids"]) != mark["output_tokens"]
                    or len(record["token_logprobs"]) != mark["output_tokens"]):
                raise ValueError("No-move probe changed original input/output lengths")
        trials.append({"repeat": repeat, "migrations": 0,
                       "serial_comparisons": [compare_tokens(by_id[reference["trace_request_id"]], reference)
                                              for reference in references]})
        records_by_repeat.append(by_id)
    paired = [compare_tokens(records_by_repeat[1][identifier], records_by_repeat[0][identifier])
              for identifier in sorted(marks)]
    error = json.loads((root / "error.json").read_text()) if (root / "error.json").exists() else None
    if error is not None and (error["type"] != "RuntimeError" or error["message"] not in EXPECTED_FAILURES):
        raise ValueError("Numerics probe must not mask unrelated runtime failures")
    return {"status": "completed_no_move_numerics_diagnostic_not_benefit",
            "baseline_serial_divergence_reproduced": any(
                comparison["different_tokens"] for trial in trials for comparison in trial["serial_comparisons"]),
            "trials": trials, "paired_no_move_comparisons": paired, "underlying_trace_error": error,
            "strict_underlying_trace_summary_preserved": True,
            "original_trace_sha256": hashlib.sha256((root / "trace.json").read_bytes()).hexdigest(),
            "logprobs": 2, "is_algorithm_benefit": False, "migration_correctness_proven": False,
            "batch_invariant_kernels_enabled_by_probe": False,
            "numerical_tolerance_acceptance_not_implemented": True}


def main():
    root = Path(sys.argv[sys.argv.index("--output") + 1])
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    with (root / "batch-numerics-probe-source.py").open("xb") as output:
        output.write(source)
    live_trace.stream = stream_with_logprobs
    try:
        live_trace.main()
    except RuntimeError as error:
        if str(error) not in EXPECTED_FAILURES:
            raise
    result = analyze(root)
    with (root / "batch-numerics-diagnostic.json").open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"diagnostic_completed": True,
                      "baseline_divergence_reproduced": result["baseline_serial_divergence_reproduced"],
                      "benefit_validated": False}), flush=True)


if __name__ == "__main__":
    main()
