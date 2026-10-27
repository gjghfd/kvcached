import argparse
import csv
import hashlib
import io
import json
import math
import random
from pathlib import Path


def source_rows(sample, dataset, max_model_len, minimum_context, minimum_output, maximum_output):
    rows = []
    for entry in json.loads((sample / "range-manifest.json").read_text()):
        if entry["dataset"] != dataset:
            continue
        fragment = sample / "ranges" / f"{dataset}-{entry['window']:02d}.csv.fragment"
        payload = fragment.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("Azure fragment checksum mismatch")
        for row_index, fields in enumerate(csv.reader(io.StringIO(
                payload.decode().split("\n", 1)[1].rsplit("\n", 1)[0]))):
            context, output = int(fields[1]), int(fields[2])
            if (context >= minimum_context and minimum_output <= output <= maximum_output
                    and context + output <= max_model_len):
                rows.append({"prompt_tokens": context, "output_tokens": output,
                             "source_fragment": fragment.name, "source_internal_row": row_index,
                             "source_timestamp": fields[0]})
    if not rows:
        raise ValueError("No paired Azure rows satisfy the declared slice")
    return rows


def generate_trace(rows, rate, duration, seed, max_model_len=32768):
    if (not rows or min(rate, duration, max_model_len) <= 0
            or not math.isfinite(rate) or not math.isfinite(duration)):
        raise ValueError("Trace needs rows, finite positive rate and duration")
    requests = []
    for model in (0, 1):
        arrival_rng = random.Random(seed * 4 + model)
        mark_rng = random.Random(seed * 4 + model + 2)
        arrival, index = 0, 0
        while True:
            arrival += arrival_rng.expovariate(rate / 2)
            if arrival >= duration:
                break
            mark = dict(mark_rng.choice(rows))
            if min(mark["prompt_tokens"], mark["output_tokens"]) <= 0 or mark["prompt_tokens"] + mark["output_tokens"] > max_model_len:
                raise ValueError("Sampled request exceeds supported full sequence length")
            requests.append({**mark, "request_id": f"trace-{seed}-{model}-{index}",
                             "model": model, "arrival_s": arrival})
            index += 1
    requests.sort(key=lambda request: (request["arrival_s"], request["model"]))
    if not requests:
        raise ValueError("Trace has no arrivals")
    return {"schema_version": 1, "seed": seed, "total_rate_rps": rate, "duration_s": duration,
            "max_model_len": max_model_len, "arrival": "two_independent_Poisson_streams",
            "sampling": "paired_marks_from_cached_byte_window_sample_not_uniform_full_dataset_rows",
            "requests": requests}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=0.6)
    parser.add_argument("--duration", type=float, default=600)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--dataset", choices=("code", "conv"), default="conv")
    parser.add_argument("--minimum-context", type=int, default=2048)
    parser.add_argument("--minimum-output", type=int, default=256)
    parser.add_argument("--maximum-output", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing trace")
    rows = source_rows(args.sample, args.dataset, args.max_model_len, args.minimum_context,
                       args.minimum_output, args.maximum_output)
    trace = generate_trace(rows, args.rate, args.duration, args.seed, args.max_model_len)
    trace["slice"] = {"dataset": args.dataset, "minimum_context": args.minimum_context,
                      "minimum_output": args.minimum_output, "maximum_output": args.maximum_output,
                      "eligible_source_rows": len(rows)}
    trace["source_manifest_sha256"] = hashlib.sha256((args.sample / "range-manifest.json").read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2) + "\n")
    print(json.dumps({"trace": str(args.output), "requests": len(trace["requests"]),
                      "requests_per_model": [sum(request["model"] == model for request in trace["requests"])
                                             for model in (0, 1)]}, indent=2))


if __name__ == "__main__":
    main()
