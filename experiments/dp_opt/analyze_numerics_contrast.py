import argparse
import json
import math
from pathlib import Path

from batch_numerics_probe import analyze, compare_tokens


def candidate_contrast(actual, reference):
    comparison = compare_tokens(actual, reference)
    position = comparison["first_difference_zero_based"]
    if position is None:
        return {**comparison, "common_candidate_contrast": None}
    actual_token = str(actual["token_ids"][position])
    reference_token = str(reference["token_ids"][position])
    before = reference["token_logprobs"][position]
    after = actual["token_logprobs"][position]
    if not all(token in frame for token in (actual_token, reference_token) for frame in (before, after)):
        return {**comparison, "common_candidate_contrast": None,
                "contrast_unavailable_reason": "both_selected_candidates_not_present_in_both_frames"}
    values = [frame[token] for frame in (before, after) for token in (actual_token, reference_token)]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Common-candidate logprobs must be finite")
    before_contrast = before[actual_token] - before[reference_token]
    after_contrast = after[actual_token] - after[reference_token]
    return {**comparison, "common_candidate_contrast": {
                "actual_candidate": int(actual_token), "reference_candidate": int(reference_token),
                "reference_actual_minus_reference_logprob": before_contrast,
                "actual_actual_minus_reference_logprob": after_contrast,
                "contrast_change": after_contrast - before_contrast,
                "reference_selected_is_reported_top1": before[reference_token] == max(before.values()),
                "actual_selected_is_reported_top1": after[actual_token] == max(after.values()),
                "softmax_normalizer_cancels_from_contrast": True},
            "numerical_root_cause_proven": False, "migration_correctness_proven": False,
            "token_tolerance_acceptance": False}


def rebuild_contrasts(root):
    root = Path(root)
    diagnostic = analyze(root)
    references = json.loads((root / "references.json").read_text())
    records = [json.loads((root / f"r{repeat}-no_move/requests.json").read_text()) for repeat in (0, 1)]
    indexed = [{record["request_id"]: record for record in trial} for trial in records]
    trials = [{"repeat": repeat, "serial_comparisons": [
                  candidate_contrast(indexed[repeat][reference["trace_request_id"]], reference)
                  for reference in references]} for repeat in (0, 1)]
    paired = [candidate_contrast(indexed[1][identifier], indexed[0][identifier])
              for identifier in sorted(indexed[0])]
    return {"status": "common_prefix_candidate_contrast_diagnostic_not_benefit",
            "baseline_serial_divergence_reproduced": diagnostic["baseline_serial_divergence_reproduced"],
            "original_trace_sha256": diagnostic["original_trace_sha256"], "trials": trials,
            "paired_no_move_comparisons": paired, "is_algorithm_benefit": False,
            "numerical_root_cause_proven": False, "migration_correctness_proven": False,
            "token_tolerance_acceptance": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = rebuild_contrasts(args.raw)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
