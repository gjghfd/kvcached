import hashlib
import json
import math
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import fp32_head_probe
import live_trace
import static_batch_probe
from batch_numerics_probe import EXPECTED_FAILURES


TARGETS = {74: {"position": 442, "candidates": [582, 279]},
           149: {"position": 53, "candidates": [12822, 20632]}}


def selected_position(state):
    target = TARGETS.get(state.num_prompt_tokens)
    position = len(state.output_token_ids)
    return position if target and abs(position - target["position"]) <= 1 else None


def context_digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def native_request_matches(native, logical):
    return native == logical or re.fullmatch(re.escape(logical) + r"-[0-9a-f]{8}", native) is not None


def validate_weights(root):
    paths = sorted(root.glob("head-candidate-weights-*.json"))
    expected_names = {f"head-candidate-weights-{service}-dp{rank}.json"
                      for service in ("instance-1", "instance-2") for rank in (0, 1)}
    if {path.name for path in paths} != expected_names:
        raise ValueError("Require identical BF16 candidate weight rows from all four workers")
    weights = []
    for path in paths:
        record = json.loads(path.read_text())
        if (path.name != f"head-candidate-weights-{record['service']}-dp{record['dp_rank']}.json"
                or record["weight_dtype"] != "torch.bfloat16"
                or set(record["weights"]) != {str(prompt) for prompt in TARGETS}):
            raise ValueError("Head weight worker identity or BF16 dtype differs")
        for prompt, target in TARGETS.items():
            entry = record["weights"][str(prompt)]
            if entry["candidates"] != target["candidates"]:
                raise ValueError("Head weight candidate ordering differs from predeclared targets")
            rows = entry["rows"]
            if len(rows) != 2 or not rows[0]:
                raise ValueError("Require two nonempty head candidate weight rows")
            dot_contrast([0.0] * len(rows[0]), rows)
        weights.append(record["weights"])
    if any(record != weights[0] for record in weights):
        raise ValueError("Require identical BF16 candidate weight rows from all four workers")
    return weights[0]


def validate_snapshot(snapshot, request, target, prompt, role, repeat, layout, position):
    if (not native_request_matches(snapshot["native_request_id"], request["request_id"])
            or snapshot["service"] != f"instance-{request['model'] + 1}"
            or snapshot["dp_rank"] != request["rank_initial"]
            or snapshot["role"] != role or snapshot["repeat"] != repeat
            or snapshot["layout"] != layout or snapshot["position"] != position
            or snapshot["prompt_tokens"] != request["prompt_tokens"]
            or snapshot["candidates"] != target["candidates"]):
        raise ValueError("Snapshot request/worker/layout identity or candidate ordering differs")
    batch = snapshot["batch_native_request_ids"]
    row = snapshot["row_index"]
    if (snapshot["batch_rows"] != len(batch) or not isinstance(row, int)
            or not 0 <= row < len(batch) or batch[row] != snapshot["native_request_id"]
            or batch.count(snapshot["native_request_id"]) != 1):
        raise ValueError("Snapshot logits row is not bound to its native request")
    if (snapshot["output_prefix_token_ids"] != request["token_ids"][:position]
            or snapshot["prompt_token_ids"] != prompt
            or context_digest(snapshot["context_token_ids"]) != snapshot["context_sha256"]
            or snapshot["context_token_ids"] != prompt + snapshot["output_prefix_token_ids"]
            or snapshot["hidden_dtype"] != "torch.bfloat16" or snapshot["logits_dtype"] != "torch.float32"
            or snapshot["sample_discarded"] is not False
            or snapshot["forward_consumed_prefix_tokens"] != len(snapshot["context_token_ids"])
            or len(snapshot["candidate_logits"]) != 2
            or any(not math.isfinite(value) for value in snapshot["candidate_logits"])):
        raise ValueError("Snapshot differs from the actual complete request prefix or precision")
    frame = request["token_logprobs"][position]
    if all(str(candidate) in frame for candidate in snapshot["candidates"]):
        contrast = frame[str(snapshot["candidates"][0])] - frame[str(snapshot["candidates"][1])]
        if not math.isclose(contrast, snapshot["candidate_logits"][0] - snapshot["candidate_logits"][1], abs_tol=2e-5):
            raise ValueError("Captured GPU logits differ from the actual same-token sampler logprobs")
        return True
    return False


def dot_contrast(hidden, rows):
    if len(rows) != 2 or any(len(row) != len(hidden) for row in rows):
        raise ValueError("Head weight/hidden dimensions differ")
    if any(not math.isfinite(value) for values in (hidden, *rows) for value in values):
        raise ValueError("Head snapshots must contain only finite values")
    return math.fsum(value * (left - right) for value, left, right in zip(hidden, rows[0], rows[1]))


def decompose(reference, actual, rows):
    if (reference["context_token_ids"] != actual["context_token_ids"]
            or reference["position"] != actual["position"]
            or reference["candidates"] != actual["candidates"]):
        raise ValueError("Projection decomposition requires identical complete common prefixes")
    expected = dot_contrast(reference["hidden"], rows)
    observed = dot_contrast(actual["hidden"], rows)
    before = reference["candidate_logits"][0] - reference["candidate_logits"][1]
    after = actual["candidate_logits"][0] - actual["candidate_logits"][1]
    hidden_change = observed - expected
    head_change = (after - observed) - (before - expected)
    if not math.isclose(after - before, hidden_change + head_change, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("Head/hidden decomposition does not reconstruct the observed contrast")
    differences = [abs(left - right) for left, right in zip(reference["hidden"], actual["hidden"])]
    return {"hidden_byte_values_identical": reference["hidden"] == actual["hidden"],
            "hidden_changed_elements": sum(value != 0 for value in differences),
            "hidden_max_abs_delta": max(differences),
            "reference_CPU_float64_contrast": expected, "actual_CPU_float64_contrast": observed,
            "reference_original_GPU_contrast": before, "actual_original_GPU_contrast": after,
            "observed_contrast_change": after - before, "hidden_contribution": hidden_change,
            "head_projection_contribution": head_change,
            "numerical_root_cause_proven": False}


@contextmanager
def instrument():
    original = live_trace.engine_arguments

    def engine_arguments(*args, **kwargs):
        arguments = original(*args, **kwargs)
        arguments.worker_cls = "head_hidden_worker.HeadHiddenVerificationWorker"
        return arguments

    live_trace.engine_arguments = engine_arguments
    try:
        yield
    finally:
        live_trace.engine_arguments = original


def analyze(root):
    root = Path(root)
    base = fp32_head_probe.analyze(root, "static")
    references = {record["trace_request_id"]: record for record in json.loads((root / "references.json").read_text())}
    snapshots = [json.loads(path.read_text()) for path in sorted((root / "head-hidden-snapshots").glob("*.json"))]
    weights = validate_weights(root)
    trace = json.loads((root / "trace.json").read_text())
    marks = {mark["request_id"]: mark for mark in trace["requests"]}
    prompt_payload = (root / "frozen/inputs/prompt-bank.json").read_bytes()
    if hashlib.sha256(prompt_payload).hexdigest() != trace["prompt_bank_sha256"]:
        raise ValueError("Head attribution prompt bank differs from the frozen trace")
    prompts = json.loads(prompt_payload)["prompts"]
    results = []
    for trial in base["diagnostic"]["trials"]:
        records = {record["request_id"]: record for record in json.loads((root / f"r{trial['repeat']}-no_move/requests.json").read_text())}
        for comparison in trial["serial_comparisons"]:
            identifier = comparison["request_id"]
            mark = records[identifier]
            target = TARGETS.get(mark["prompt_tokens"])
            if target is None:
                continue
            first_difference = comparison["first_difference_zero_based"]
            position = target["position"] if first_difference is None else first_difference
            result = {"repeat": trial["repeat"], "layout": trial["layout"], "request_id": identifier,
                      "position": position, "first_divergence_position": first_difference,
                      "is_first_divergence": first_difference == position}
            if abs(position - target["position"]) > 1:
                results.append({**result, "attribution_available": False,
                                "first_top1_attribution_available": False,
                                "reason": "actual_first_divergence_outside_captured_positions"})
                continue
            actual = [record for record in snapshots if record["role"] == "trial"
                      and record["repeat"] == trial["repeat"] and record["prompt_tokens"] == mark["prompt_tokens"]
                      and record["position"] == position]
            serial = [record for record in snapshots if record["role"] == "reference"
                      and record["prompt_tokens"] == mark["prompt_tokens"] and record["position"] == position]
            if len(actual) != 1 or len(serial) != 1:
                raise ValueError("Require one complete target snapshot for each serial and trial request")
            reference, actual = serial[0], actual[0]
            prompt = prompts[marks[identifier]["prompt_key"]]["token_ids"]
            serial_verified = validate_snapshot(reference, references[identifier], target, prompt,
                                                "reference", None, None, position)
            actual_verified = validate_snapshot(actual, mark, target, prompt, "trial",
                                                trial["repeat"], trial["layout"], position)
            common = reference["context_token_ids"] == actual["context_token_ids"]
            decision_candidates = {references[identifier]["token_ids"][position], mark["token_ids"][position]}
            decision_covered = first_difference is not None and decision_candidates == set(target["candidates"])
            result.update({"full_common_prefix": common, "attribution_available": common,
                           "sampler_candidate_contrasts_verified": serial_verified and actual_verified,
                           "sampler_contrast_coverage": {"reference": serial_verified, "actual": actual_verified},
                           "first_top1_candidates_covered": decision_covered,
                           "first_top1_attribution_available": common and decision_covered and serial_verified and actual_verified})
            if first_difference is not None and not result["first_top1_attribution_available"]:
                result["first_top1_attribution_unavailable_reason"] = (
                    "different_complete_histories" if not common else
                    "actual_first_top1_candidates_differ_from_predeclared_pair" if not decision_covered else
                    "sampler_candidate_contrast_coverage_missing")
            if common:
                result.update(decompose(reference, actual, weights[str(mark["prompt_tokens"])]["rows"]))
            results.append(result)
    complete = all(result["first_divergence_position"] is None or result["first_top1_attribution_available"]
                   for result in results)
    return {"status": ("completed_common_prefix_projection_diagnostic_not_benefit" if complete
                       else "completed_observation_missing_first_divergence_attribution_not_benefit"),
            "full_static_analysis": base, "projection_comparisons": results, "snapshots": len(snapshots),
            "same_BF16_candidate_weights_verified": True, "is_algorithm_benefit": False,
            "all_observed_first_divergences_attributed": complete,
            "strict_token_check_relaxed": False, "numerical_root_cause_proven": False}


def main():
    if "--verify" in sys.argv:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--verify", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        with args.output.open("x") as output:
            output.write(json.dumps(analyze(args.verify), indent=2) + "\n")
        return
    root = Path(sys.argv[sys.argv.index("--output") + 1])
    root.mkdir(parents=True, exist_ok=True)
    for name in ("head_hidden_probe.py", "head_hidden_worker.py", "fp32_head_probe.py", "fp32_head_worker.py"):
        with (root / name).open("xb") as output:
            output.write(Path(__file__).with_name(name).read_bytes())
    live_trace.save(root / "fp32-head-config.json", {"hf_overrides": {"head_dtype": "float32"},
        "reference_generation": "fresh_serial_same_FP32_head_configuration",
        "is_algorithm_benefit": False, "strict_token_check_relaxed": False})
    with static_batch_probe.instrument(), fp32_head_probe.instrument(), instrument():
        try:
            live_trace.main()
        except RuntimeError as error:
            if str(error) not in EXPECTED_FAILURES:
                raise
    live_trace.save(root / "head-hidden-diagnostic.json", analyze(root))


if __name__ == "__main__":
    main()
