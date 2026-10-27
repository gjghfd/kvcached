import argparse
import bisect
import hashlib
import json
import math
import random
from pathlib import Path


def generate_mixed_trace(code_marks, conv_marks, code_rate, conv_rate, duration, seed,
                         max_model_len=131072):
    if (not code_marks or not conv_marks or not math.isfinite(duration) or duration <= 0
            or any(not math.isfinite(rate) or rate <= 0 for rate in (code_rate, conv_rate))):
        raise ValueError("Mixed trace requires two nonempty populations and finite positive rates")
    requests = []
    for model, marks, rate in ((0, code_marks, code_rate), (1, conv_marks, conv_rate)):
        cumulative, total = [], 0
        for mark in marks:
            if (min(mark["prompt_tokens"], mark["output_tokens"], mark.get("count", 1)) <= 0
                    or mark["prompt_tokens"] + mark["output_tokens"] > max_model_len):
                raise ValueError("Mixed trace marks exceed supported full sequence length")
            total += mark.get("count", 1)
            cumulative.append(total)
        arrival_rng, mark_rng = random.Random(seed * 4 + model), random.Random(seed * 4 + model + 2)
        arrival, index = 0, 0
        while True:
            arrival += arrival_rng.expovariate(rate)
            if arrival >= duration:
                break
            selected = marks[bisect.bisect_right(cumulative, mark_rng.randrange(total))]
            requests.append({**{key: value for key, value in selected.items() if key != "count"},
                             "request_id": f"agentic-{seed}-{model}-{index}",
                             "model": model, "arrival_s": arrival})
            index += 1
    requests.sort(key=lambda mark: (mark["arrival_s"], mark["model"]))
    if not requests or {mark["model"] for mark in requests} != {0, 1}:
        raise ValueError("Mixed trace must contain arrivals for both tenants")
    return {"schema_version": 2, "seed": seed, "total_rate_rps": code_rate + conv_rate,
            "model_rates_rps": [code_rate, conv_rate], "duration_s": duration,
            "max_model_len": max_model_len, "arrival": "two_independent_Poisson_streams",
            "sampling": "SWE_smith_real_history_prefix_and_full_Azure_paired_lengths_declared_slice",
            "model_names": ["SWE-smith-Code", "Azure-Conversation"], "requests": requests}


def build_bank(inputs, tokenizer_path, tokenizer_config_path, code_output_tokens):
    from jinja2 import Environment, StrictUndefined
    from tokenizers import Tokenizer

    config = json.loads(tokenizer_config_path.read_text())
    template = Environment(undefined=StrictUndefined).from_string(config["chat_template"])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.truncation is not None or tokenizer.padding is not None:
        raise ValueError("Prompt bank cannot truncate or pad")
    bank, marks = {}, []
    for line in inputs.read_text().splitlines():
        entry = json.loads(line)
        text = template.render(messages=entry["messages"], tools=None, add_generation_prompt=True)
        token_ids = tokenizer.encode(text, add_special_tokens=False).ids
        if (len(token_ids) != entry["prompt_tokens"]
                or hashlib.sha256(text.encode()).hexdigest() != entry["prompt_sha256"]):
            raise ValueError("SWE-smith prompt reconstruction differs from verified input")
        key = "swe-" + hashlib.sha256(entry["traj_id"].encode()).hexdigest()[:16]
        if key in bank:
            raise ValueError("Duplicate agent episode")
        bank[key] = {"token_ids": token_ids, "messages": entry["messages"],
                     "prompt_sha256": entry["prompt_sha256"], "prompt_tokens": len(token_ids),
                     "traj_id": entry["traj_id"], "instance_id": entry["instance_id"]}
        marks.append({"prompt_tokens": len(token_ids), "output_tokens": code_output_tokens,
                      "prompt_key": key, "source_fragment": "SWE-smith-tool-shard0",
                      "source_internal_row": entry["row_index"], "source_traj_id": entry["traj_id"]})
    return {"tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            "tokenizer_config_sha256": hashlib.sha256(tokenizer_config_path.read_bytes()).hexdigest(),
            "inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(), "prompts": bank}, marks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--conv-joint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-rate", type=float, default=.02)
    parser.add_argument("--conv-rate", type=float, default=1.8)
    parser.add_argument("--duration", type=float, default=1200)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--code-output-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=131072)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior mixed trace")
    bank, code = build_bank(args.inputs, args.tokenizer, args.tokenizer_config, args.code_output_tokens)
    conv = [{**entry, "source_fragment": "Azure-full-Conversation-Pge2048-O512to1024",
             "source_internal_row": index}
            for index, entry in enumerate(json.loads(args.conv_joint.read_text()))
            if entry["prompt_tokens"] >= 2048 and 512 <= entry["output_tokens"] <= 1024]
    trace = generate_mixed_trace(code, conv, args.code_rate, args.conv_rate, args.duration,
                                 args.seed, args.max_model_len)
    args.output.mkdir(parents=True)
    bank_path = args.output / "prompt-bank.json"
    bank_path.write_text(json.dumps(bank) + "\n")
    trace["prompt_bank_sha256"] = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    trace["conv_joint_sha256"] = hashlib.sha256(args.conv_joint.read_bytes()).hexdigest()
    trace["slice"] = {"code": "SWE-smith-whole-history-prefix-100K..105K-output64",
                      "conv": "full-Azure-Conversation-P>=2048-O512..1024-original-paired-O",
                      "conv_eligible_rows": sum(mark["count"] for mark in conv),
                      "conv_full_rows": 27303999, "code_eligible_episodes": len(code),
                      "code_source_scope": "one_tool_shard_unresolved_history_reconstruction"}
    (args.output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    (args.output / "conv-marks.json").write_text(json.dumps(conv) + "\n")
    print(json.dumps({"requests_per_model": [sum(mark["model"] == model for mark in trace["requests"])
                                               for model in (0, 1)],
                      "rates": trace["model_rates_rps"], "code_episodes": len(code)}, indent=2))


if __name__ == "__main__":
    main()
