import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
import time
import zipfile
from pathlib import Path


QUANTILES = (.5, .9, .95, .99, .999)
THRESHOLDS = (4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1000000)


def histogram_statistics(histogram):
    count = sum(histogram.values())
    if not count:
        return {"requests": 0}
    entries = sorted(histogram.items())
    quantiles = {}
    for quantile in QUANTILES:
        position, cumulative = math.ceil(count * quantile), 0
        for value, frequency in entries:
            cumulative += frequency
            if cumulative >= position:
                quantiles[str(quantile)] = value
                break
    return {"requests": count, "mean": sum(value * frequency for value, frequency in entries) / count,
            "min": entries[0][0], "max": entries[-1][0], "quantiles_nearest_rank": quantiles,
            "fraction_ge": {str(threshold): sum(frequency for value, frequency in entries if value >= threshold) / count
                            for threshold in THRESHOLDS}}


def paired_statistics(joint, decode_step_s=.04, prefill_s=.2):
    count = sum(joint.values())
    if not count:
        return {"requests": 0}
    contexts, outputs, totals = collections.Counter(), collections.Counter(), collections.Counter()
    sum_memory, sum_service, sum_memory_time, sum_progressive_time = 0, 0, 0, 0
    for (context, output), frequency in joint.items():
        contexts[context] += frequency
        outputs[output] += frequency
        totals[context + output] += frequency
        memory = math.ceil((context + output) / 16) * 1.5
        service = prefill_s + decode_step_s * output
        sum_memory += frequency * memory
        sum_service += frequency * service
        sum_memory_time += frequency * memory * service
        sum_progressive_time += frequency * ((context + output / 2) * .09375 * decode_step_s * output
                                            + context * .09375 * prefill_s)
    output_sum = sum(output * frequency for output, frequency in outputs.items())
    context_sum = sum(context * frequency for context, frequency in contexts.items())
    return {"requests": count, "input_tokens": histogram_statistics(contexts),
            "output_tokens": histogram_statistics(outputs), "total_tokens": histogram_statistics(totals),
            "mean_logical_reserved_mib": sum_memory / count,
            "input_fraction_of_total_token_volume": context_sum / (context_sum + output_sum),
            "fraction_output_ge": {str(threshold): sum(frequency for output, frequency in outputs.items() if output >= threshold) / count
                                   for threshold in (64, 128, 256, 512, 1024)},
            "service_proxy": {"decode_step_s": decode_step_s, "prefill_s": prefill_s,
                              "mean_resident_s": sum_service / count,
                              "mean_full_reservation_mib_seconds": sum_memory_time / count,
                              "mean_progressive_logical_mib_seconds_approx": sum_progressive_time / count,
                              "scope": "uncalibrated_proxy_not_GPU_profile_prefill_length_dependence_missing"}}


def read_azure_full(path):
    joint, invalid = collections.Counter(), 0
    started = time.monotonic()
    with path.open(newline="") as source:
        reader = csv.reader(source)
        if next(reader) != ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]:
            raise ValueError("Unexpected Azure schema")
        for fields in reader:
            if len(fields) != 3:
                raise ValueError("Unexpected Azure row width")
            context, output = int(fields[1]), int(fields[2])
            if context <= 0 or output <= 0:
                invalid += 1
            joint[(context, output)] += 1
    print(path.name, "requests", sum(joint.values()), "unique paired marks", len(joint),
          "parse_s", round(time.monotonic() - started, 2), flush=True)
    return joint, invalid


def azure_profile(root, output):
    data, results = {}, {}
    for dataset in ("code", "conv"):
        filename = f"AzureLLMInferenceTrace_{dataset}_1week.csv"
        joint, invalid = read_azure_full(root / filename)
        valid = collections.Counter({mark: count for mark, count in joint.items() if min(mark) > 0})
        source = json.loads((root / (filename + ".source.json")).read_text())
        if (root / filename).stat().st_size != source["expected_release_bytes"]:
            raise ValueError("Full file size changed")
        data[dataset] = valid
        slices = {"all_valid": valid,
                  "input_ge4096": {mark: count for mark, count in valid.items() if mark[0] >= 4096},
                  "input_ge6000": {mark: count for mark, count in valid.items() if mark[0] >= 6000},
                  "output_ge256": {mark: count for mark, count in valid.items() if mark[1] >= 256},
                  "input_lt2048_output_ge256": {mark: count for mark, count in valid.items() if mark[0] < 2048 and mark[1] >= 256}}
        results[dataset] = {"source": source["url"], "sha256": source["sha256"],
                            "all_rows": sum(joint.values()), "nonpositive_rows": invalid,
                            "slices": {name: {**paired_statistics(marks), "fraction_of_valid": sum(marks.values()) / sum(valid.values())}
                                       for name, marks in slices.items()}}
        (output / f"azure-{dataset}-joint.json").write_text(json.dumps(
            [{"prompt_tokens": mark[0], "output_tokens": mark[1], "count": count}
             for mark, count in sorted(valid.items())]) + "\n")
    code_input, conv_input = collections.Counter(), collections.Counter()
    for (context, _), frequency in data["code"].items():
        code_input[context] += frequency
    for (context, _), frequency in data["conv"].items():
        conv_input[context] += frequency
    conv_entries, cumulative, cursor, wins, ties = sorted(conv_input.items()), 0, 0, 0, 0
    for context, frequency in sorted(code_input.items()):
        while cursor < len(conv_entries) and conv_entries[cursor][0] < context:
            cumulative += conv_entries[cursor][1]
            cursor += 1
        wins += frequency * cumulative
        ties += frequency * conv_input[context]
    pairs = sum(code_input.values()) * sum(conv_input.values())
    report = {"status": "full_official_CSV_profile", "datasets": results,
              "probability_independent_code_input_gt_conv_input": wins / pairs,
              "probability_equal_input": ties / pairs, "analysis_limit_tokens": 1000000,
              "scope": "full_released_trace_not_all_Azure_traffic_current_serving_limit_unchanged"}
    (output / "azure-full-profile.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def chat_prompt(text):
    return "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"


def longbench_records(root):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(root / "qwen-tokenizer.json"))
    records = []
    with zipfile.ZipFile(root / "longbench-v1-data.zip") as source:
        for name in source.namelist():
            if not name.endswith(".jsonl") or name.endswith("_e.jsonl"):
                continue
            with source.open(name) as content:
                for line in content:
                    example = json.loads(line)
                    context = example["context"]
                    text = context + "\n\n" + example["input"]
                    records.append({"source": "LongBench-v1", "task": example["dataset"], "id": example["_id"],
                                    "context_tokens": token_count(tokenizer, context),
                                    "prompt_tokens": token_count(tokenizer, chat_prompt(text)),
                                    "reference_answer_tokens": token_count(tokenizer, example["answers"][0]),
                                    "official_length_metadata_not_tokens": example["length"]})
            print("tokenized", name, flush=True)
    examples = json.loads((root / "longbench-v2-data.json").read_text())
    for index, example in enumerate(examples):
        text = example["context"] + "\n\n" + example["question"] + "\n" + "\n".join(
            choice + ": " + example["choice_" + choice] for choice in ("A", "B", "C", "D"))
        records.append({"source": "LongBench-v2", "task": example["domain"], "sub_domain": example["sub_domain"],
                        "id": example["_id"], "context_tokens": token_count(tokenizer, example["context"]),
                        "prompt_tokens": token_count(tokenizer, chat_prompt(text)),
                        "reference_answer_tokens": token_count(tokenizer, example["answer"]),
                        "official_length_category": example["length"]})
        if (index + 1) % 25 == 0:
            print("tokenized LongBench-v2", index + 1, "/", len(examples), flush=True)
    return records


def longbench_profile(records):
    groups = {}
    for source in ("LongBench-v1", "LongBench-v2"):
        selected = [record for record in records if record["source"] == source]
        groups[source] = {}
        for task in sorted({record["task"] for record in selected}):
            examples = [record for record in selected if record["task"] == task]
            groups[source][task] = {"requests": len(examples),
                                    "context_tokens": histogram_statistics(collections.Counter(record["context_tokens"] for record in examples)),
                                    "prompt_tokens": histogram_statistics(collections.Counter(record["prompt_tokens"] for record in examples)),
                                    "reference_answer_tokens": histogram_statistics(collections.Counter(record["reference_answer_tokens"] for record in examples)),
                                    "supported_prompt_plus_512_fraction": {str(limit): sum(record["prompt_tokens"] + 512 <= limit for record in examples) / len(examples)
                                                                            for limit in (32768, 65536, 131072, 262144, 1000000)}}
    return {"datasets": groups, "tokenizer": "Qwen3-30B-A3B-Instruct-2507_tokenizers_no_truncation",
            "prompt_scope": "single_user_chat_context_plus_question_choices_not_official_eval_template",
            "output_scope": "reference_answer_tokens_are_not_measured_generation_length_or_production_residence"}


def memory_scale(config):
    bytes_per_token = 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * config["head_dim"] * 2
    return {"kv_dtype": "BF16", "attention_TP": 1, "bytes_per_token": bytes_per_token,
            "logical_kv_gib": {str(tokens): tokens * bytes_per_token / 1024**3
                               for tokens in (8192, 16384, 32768, 65536, 131072, 262144, 1000000)},
            "current2112MiB_rank_budget_max_tokens_ignoring_pinned_pages": 2112 * 1024**2 // bytes_per_token,
            "current4GiB_virtual_pool_max_tokens_ignoring_other_requests": 4 * 1024**3 // bytes_per_token,
            "scope": "single_request_unsharded_attention_KV_EP_does_not_split_KV_no_weights_or_activation_headroom"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("azure", "longbench"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expected = args.output / ("azure-full-profile.json" if args.mode == "azure" else "longbench-profile.json")
    if expected.exists():
        parser.error("Preserve existing profile; choose a new output directory")
    if args.mode == "azure":
        report = azure_profile(args.source, args.output)
    else:
        records = longbench_records(args.source)
        (args.output / "longbench-token-records.json").write_text(json.dumps(records, indent=2) + "\n")
        report = longbench_profile(records)
        report["memory_scale"] = memory_scale(json.loads((args.source / "qwen-config.json").read_text()))
        expected.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
