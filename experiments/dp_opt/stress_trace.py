import argparse
import hashlib
import json
import math
from pathlib import Path

from agentic_trace import generate_mixed_trace


def long_output_marks(marks, output_tokens):
    if type(output_tokens) is not int or output_tokens <= 0 or not marks:
        raise ValueError("Require nonempty Conv marks and positive integer synthetic output")
    return [{**mark, "original_output_tokens": mark["output_tokens"],
             "output_tokens": output_tokens, "synthetic_output_tokens": output_tokens,
             "source_fragment": "Azure-Conv-input-synthetic-long-output",
             "original_source_fragment": mark.get("source_fragment", "Azure-Conv-paired-slice")}
            for mark in marks]


def make_stress_trace(code_marks, conv_marks, output_tokens, code_rate, conv_rate,
                      window_s, warmup_s, seed, prompt_bank_sha256):
    if not math.isfinite(warmup_s) or warmup_s < 0 or not math.isfinite(window_s) or window_s <= 0:
        raise ValueError("Require finite nonnegative warmup and positive measurement window")
    if len(prompt_bank_sha256) != 64:
        raise ValueError("Require frozen prompt bank SHA256")
    trace = generate_mixed_trace(code_marks, long_output_marks(conv_marks, output_tokens), code_rate,
                                 conv_rate, warmup_s + window_s, seed, max_model_len=131072)
    trace.update({"prompt_bank_sha256": prompt_bank_sha256,
                  "sampling": "verified_SWE_smith_history_Azure_input_explicit_synthetic_Conv_output",
                  "evaluation_start_s": warmup_s, "evaluation_duration_s": window_s,
                  "warmup": "natural_Poisson_both_tenants_no_bootstrap_all_requests_recorded_and_drained",
                  "synthetic_Conv_output_tokens": output_tokens})
    return trace


def rate_from_profile(conv_marks, output_tokens, representative_service_s, target_mib=4800):
    if (not conv_marks or not math.isfinite(representative_service_s) or representative_service_s <= 0
            or not math.isfinite(target_mib) or target_mib <= 0):
        raise ValueError("Require positive finite measured service and target memory")
    weights = sum(mark.get("count", 1) for mark in conv_marks)
    memory = sum(math.ceil((mark["prompt_tokens"] + output_tokens) / 16) * 1.5 * mark.get("count", 1)
                 for mark in conv_marks) / weights
    return {"conv_rate_rps": target_mib / (memory * representative_service_s),
            "mean_full_reserved_mib": memory, "representative_service_s": representative_service_s,
            "target_Conv_reserved_mib": target_mib,
            "scope": "finite_population_measured_service_full_mark_memory_not_open_loop_stability_certificate"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conv-output-tokens", type=int, default=8192)
    parser.add_argument("--conv-rate", type=float, default=.025)
    parser.add_argument("--code-rate", type=float, default=.001)
    parser.add_argument("--window-s", type=float, default=1800)
    parser.add_argument("--warmup-s", type=float, default=600)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing frozen stress trace")
    read = lambda name: json.loads((args.inputs / name).read_text())
    trace = make_stress_trace(read("code-marks.json"), read("conv-marks.json"), args.conv_output_tokens,
                              args.code_rate, args.conv_rate, args.window_s, args.warmup_s, args.seed,
                              hashlib.sha256((args.inputs / "prompt-bank.json").read_bytes()).hexdigest())
    args.output.write_text(json.dumps(trace, indent=2) + "\n")
    print(json.dumps({"counts": [sum(mark["model"] == model for mark in trace["requests"]) for model in (0, 1)],
                      "trace_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
