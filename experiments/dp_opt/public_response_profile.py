import argparse
import collections
import hashlib
import heapq
import json
import math
from pathlib import Path

from agentic_profile import render_prompt
from workload_profile import histogram_statistics


COHORTS = {"full_short_input": (1, 131072), "original_O8k_32k": (8192, 32768),
           "original_O16k_32k": (16384, 32768), "original_O24k_32k": (24576, 32768)}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def normalize(row):
    messages = next((row[key] for key in ("messages", "conversations", "conversation") if key in row), None)
    if not isinstance(messages, list) or not messages:
        raise ValueError("Require original nonempty text conversation")
    roles = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant", "system": "system"}
    result = []
    for message in messages:
        role = message.get("role", message.get("from"))
        content = message.get("content", message.get("value"))
        if role not in roles or not isinstance(content, str):
            raise ValueError("Unsupported original role or nontext content; do not flatten or synthesize it")
        result.append({"role": roles[role], "content": content})
    return result


def boundaries(messages):
    return [index for index, message in enumerate(messages)
            if index > 0 and message["role"] == "assistant"
            and any(previous["role"] == "user" for previous in messages[:index])]


def render_original(template, messages):
    prepared = [{**message, "tool_calls": []} if message["role"] == "assistant" else message
                for message in messages]
    return render_prompt(template, prepared)


def row_selection(count, limit, identity):
    if limit is None or count <= limit:
        return None
    return set(heapq.nsmallest(limit, range(count), key=lambda index:
        hashlib.sha256(f"public-response-20261031|{identity}|{index}".encode()).digest()))


def rows(path, label, identity):
    if path.suffix == ".jsonl":
        with path.open() as source:
            for index, line in enumerate(source):
                yield index, json.loads(line)
        return
    import pyarrow.parquet as parquet

    source = parquet.ParquetFile(path)
    selected = row_selection(source.metadata.num_rows, 10000 if label == "wildchat" else None, identity)
    available = [name for name in ("messages", "conversations", "conversation")
                 if name in source.schema_arrow.names]
    if len(available) != 1:
        raise ValueError("Require one original conversation column")
    column = available[0]
    offset = 0
    for batch in source.iter_batches(batch_size=128, columns=[column]):
        for ordinal, row in enumerate(batch.to_pylist()):
            index = offset + ordinal
            if selected is None or index in selected:
                yield index, row
        offset += batch.num_rows


def eligible(prompt_tokens, output_tokens, cohort):
    minimum, maximum = COHORTS[cohort]
    return (0 < prompt_tokens <= 8192 and minimum <= output_tokens <= maximum
            and prompt_tokens + output_tokens <= 108544)


def summarize(joint):
    prompts, outputs, totals = collections.Counter(), collections.Counter(), collections.Counter()
    count = sum(joint.values())
    if not count:
        return {"requests": 0}
    kv_output, pages = 0, 0
    for (prompt, output), frequency in joint.items():
        prompts[prompt] += frequency
        outputs[output] += frequency
        totals[prompt + output] += frequency
        page_mib = math.ceil((prompt + output) / 1024) * 96
        pages += page_mib * frequency
        kv_output += page_mib * output * frequency
    moment = kv_output / count
    return {"requests": count, "prompt_tokens": histogram_statistics(prompts),
        "original_output_tokens": histogram_statistics(outputs), "total_tokens": histogram_statistics(totals),
        "mean_independent_page_upper_mib": pages / count, "mean_page_upper_times_output_mib_tokens": moment,
        "Conv_RPS_priors": {str(ratio): {str(step): 2 * (ratio * 10240 - 9600) / (moment * step)
            for step in (.012, .018)} for ratio in (1.1, 1.2, 1.3)},
        "prior_scope": "A800_B10240_Code9600_page_upper_and_old_decode_step_prior_not_actual_allocator_or_new_GPU_calibration"}


def keep_export(heap, entry, limit):
    key = int(entry["selection_sha256"], 16)
    item = (-key, entry["source_row"], entry["assistant_index"], entry)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif key < -heap[0][0]:
        heapq.heapreplace(heap, item)


def profile(root, tokenizer_path, config_path, output, export_limit):
    from jinja2 import Environment, StrictUndefined
    from tokenizers import Tokenizer

    if output.exists():
        raise ValueError("Preserve existing public-response profile, rerun in a fresh evidence directory")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.padding is not None or tokenizer.truncation is not None:
        raise ValueError("Original lengths must not be truncated or padded")
    template = Environment(undefined=StrictUndefined).from_string(json.loads(config_path.read_text())["chat_template"])
    output.mkdir(parents=True)
    manifest = json.loads((root / "download-manifest.json").read_text())
    statistics, joints, cohorts, exports, records = {}, {}, {}, {}, []
    for source in manifest:
        label = source["label"]
        statistics.setdefault(label, {"sampled_rows": 0, "invalid_rows": 0, "rows_without_assistant": 0, "sources": []})
        joints.setdefault(label, collections.Counter())
        cohorts.setdefault(label, {name: collections.Counter() for name in COHORTS})
        exports.setdefault(label, {name: [] for name in COHORTS})
        path = root / source["destination"]
        if path.stat().st_size != source["size"] or digest(path) != source["sha256"]:
            raise ValueError("Public source must match the archived official LFS size and SHA")
        statistics[label]["sources"].append(source)
        identity = source["dataset"] + "@" + source["revision"] + "/" + source["path"]
        pending = []

        def consume():
            prompt_encodings = tokenizer.encode_batch([item["prompt"] for item in pending], add_special_tokens=False)
            response_encodings = tokenizer.encode_batch([item["response"] for item in pending], add_special_tokens=False)
            for item, prompt, response in zip(pending, prompt_encodings, response_encodings):
                prompt_tokens, output_tokens = len(prompt.ids), len(response.ids)
                entry = {key: value for key, value in item.items() if key not in ("prompt", "response")}
                entry.update({"prompt_tokens": prompt_tokens, "output_tokens": output_tokens})
                records.append(entry)
                joints[label][prompt_tokens, output_tokens] += 1
                for cohort in COHORTS:
                    if eligible(prompt_tokens, output_tokens, cohort):
                        cohorts[label][cohort][prompt_tokens, output_tokens] += 1
                        keep_export(exports[label][cohort], {**entry, "input_token_ids": prompt.ids}, export_limit)
            pending.clear()

        for row_index, row in rows(path, label, identity):
            statistics[label]["sampled_rows"] += 1
            try:
                messages = normalize(row)
            except ValueError:
                statistics[label]["invalid_rows"] += 1
                continue
            indices = boundaries(messages)
            if not indices:
                statistics[label]["rows_without_assistant"] += 1
            for index in indices:
                response = messages[index]["content"]
                prompt = render_original(template, messages[:index])
                response_sha = hashlib.sha256(response.encode()).hexdigest()
                selection_sha = hashlib.sha256(f"{identity}|{row_index}|{index}|{response_sha}".encode()).hexdigest()
                pending.append({"dataset": source["dataset"], "revision": source["revision"],
                    "source_file": source["destination"], "source_row": row_index, "assistant_index": index,
                    "source_sha256": source["sha256"], "original_response_sha256": response_sha,
                    "selection_sha256": selection_sha, "prompt": prompt, "response": response,
                    "original_response_has_think_tag": "<think>" in response})
                if len(pending) >= 128:
                    consume()
            if statistics[label]["sampled_rows"] % 1000 == 0:
                print(label, "rows", statistics[label]["sampled_rows"], "responses", sum(joints[label].values()), flush=True)
        if pending:
            consume()
        if digest(path) != source["sha256"]:
            raise ValueError("Dataset source changed while profiling")
    summary = {"scope": "original_assistant_response_lengths_Qwen_template_no_extension_no_output_override_no_GPU_results",
        "tokenizer_sha256": digest(tokenizer_path), "tokenizer_config_sha256": digest(config_path), "datasets": {}}
    for label in statistics:
        total = sum(joints[label].values())
        summary["datasets"][label] = {**statistics[label], "all_sampled_responses": summarize(joints[label]), "cohorts": {}}
        for cohort in COHORTS:
            values = [item[-1] for item in sorted(exports[label][cohort], key=lambda item: -item[0])]
            with (output / f"{label}-{cohort}-exports.jsonl").open("x") as target:
                for value in values:
                    target.write(json.dumps(value) + "\n")
            subset = summarize(cohorts[label][cohort])
            summary["datasets"][label]["cohorts"][cohort] = {**subset,
                "fraction_of_sampled_responses": subset["requests"] / total if total else None,
                "exported_original_pairs": len(values), "selection": "lowest_source_row_boundary_response_hash_not_GPU_benefit"}
    with (output / "response-length-records.jsonl").open("x") as target:
        for record in records:
            target.write(json.dumps(record) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-limit", type=int, default=64)
    args = parser.parse_args()
    if args.export_limit <= 0:
        raise ValueError("Require a positive bounded export count")
    result = profile(args.root, args.tokenizer, args.tokenizer_config, args.output, args.export_limit)
    print(json.dumps({label: data["all_sampled_responses"]["requests"] for label, data in result["datasets"].items()}))


if __name__ == "__main__":
    main()
