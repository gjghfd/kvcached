"""Freeze a two-tenant trace (Poisson Conv; Poisson or periodic Code) from the reused banks (original lengths)."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

DEFAULT_INPUTS = Path(__file__).resolve().parents[4] / "evidence/public-openthoughts3-quality-workload-20261031/frozen/inputs"


def load_marks(inputs):
    conv = json.loads((inputs / "conv-marks.json").read_text())
    code = json.loads((inputs / "code-marks.json").read_text())
    for mark in conv:
        if mark["output_tokens"] != mark.get("original_output_tokens", mark["output_tokens"]):
            raise ValueError("Conv bank must replay original response lengths")
    return code, conv


def poisson(rate, duration, rng):
    times, now = [], 0.0
    if rate <= 0:
        return times
    while True:
        now += rng.expovariate(rate)
        if now >= duration:
            return times
        times.append(now)


def make_trace(inputs, conv_rate, code_rate, duration_s, seed, max_model_len=131072, code_start_s=0.0,
               code_period_s=None, code_end_s=None):
    """Poisson Conv; Code is Poisson at code_rate, or every code_period_s from code_start_s (before code_end_s)."""
    code_marks, conv_marks = load_marks(inputs)
    bank_sha = hashlib.sha256((inputs / "prompt-bank.json").read_bytes()).hexdigest()
    requests = []
    for model, rate, marks, name in ((0, code_rate, code_marks, "code"), (1, conv_rate, conv_marks, "conv")):
        if model == 0 and code_period_s:
            end = duration_s if code_end_s is None else code_end_s
            count = math.ceil((end - code_start_s) / code_period_s)
            arrivals = [code_period_s * index for index in range(count) if code_start_s + code_period_s * index < end]
        else:
            arrivals = poisson(rate, duration_s - (code_start_s if model == 0 else 0.0),
                               random.Random(seed * 4 + model))
        chooser = random.Random(seed * 4 + model + 2)
        for index, arrival in enumerate(arrivals):
            mark = chooser.choice(marks)
            requests.append({"request_id": f"{name}_{seed}_{index:05d}", "model": model,
                             "arrival_s": round(arrival + (code_start_s if model == 0 else 0.0), 6),
                             "prompt_tokens": mark["prompt_tokens"], "output_tokens": mark["output_tokens"],
                             "prompt_key": mark["prompt_key"]})
    requests.sort(key=lambda request: (request["arrival_s"], request["model"]))
    if any(request["prompt_tokens"] + request["output_tokens"] > max_model_len for request in requests):
        raise ValueError("Trace request exceeds max_model_len")
    return {"schema_version": 1, "seed": seed, "duration_s": duration_s, "max_model_len": max_model_len,
            "prompt_bank_sha256": bank_sha, "rates_rps": {"code": code_rate, "conv": conv_rate},
            "code_start_s": code_start_s, "code_period_s": code_period_s, "code_end_s": code_end_s,
            "arrival": "poisson_conv_" + ("periodic_code" if code_period_s else "poisson_code"),
            "sampling": "uniform_with_replacement_from_frozen_banks_original_lengths",
            "requests": requests}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--conv-rate", type=float, required=True)
    parser.add_argument("--code-rate", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--code-start-s", type=float, default=0.0)
    parser.add_argument("--code-period-s", type=float, help="fixed Code interarrival instead of Poisson")
    parser.add_argument("--code-end-s", type=float, help="no periodic Code at or after this time")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = make_trace(args.inputs, args.conv_rate, args.code_rate, args.duration_s, args.seed,
                       code_start_s=args.code_start_s, code_period_s=args.code_period_s, code_end_s=args.code_end_s)
    args.output.write_text(json.dumps(trace, indent=1) + "\n")
    counts = [sum(request["model"] == model for request in trace["requests"]) for model in (0, 1)]
    print(json.dumps({"output": str(args.output), "code": counts[0], "conv": counts[1],
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
