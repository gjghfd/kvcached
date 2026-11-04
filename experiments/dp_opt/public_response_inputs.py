import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def original_messages(row):
    fields = [field for field in ("messages", "conversations", "conversation") if field in row]
    if len(fields) != 1 or not isinstance(row[fields[0]], list):
        raise ValueError("Require one original textual conversation")
    roles = {"human": "user", "gpt": "assistant", "user": "user",
             "assistant": "assistant", "system": "system"}
    messages = []
    for raw in row[fields[0]]:
        role = raw.get("role", raw.get("from"))
        text = raw.get("content", raw.get("value"))
        if role not in roles or not isinstance(text, str):
            raise ValueError("Cannot synthesize or flatten source messages")
        message = {"role": roles[role], "content": text}
        if message["role"] == "assistant":
            message["tool_calls"] = []
        messages.append(message)
    return messages


def source_rows(path, selected):
    if path.suffix == ".jsonl":
        with path.open() as source:
            for index, line in enumerate(source):
                if index in selected:
                    yield index, json.loads(line)
        return
    import pyarrow.parquet as parquet

    source = parquet.ParquetFile(path)
    columns = [field for field in ("messages", "conversations", "conversation")
               if field in source.schema_arrow.names]
    if len(columns) != 1:
        raise ValueError("Ambiguous raw conversation schema")
    offset = 0
    for batch in source.iter_batches(batch_size=128, columns=columns):
        for ordinal, row in enumerate(batch.to_pylist()):
            index = offset + ordinal
            if index in selected:
                yield index, row
        offset += batch.num_rows


def selected_records(profile_root, label, maximum):
    summary = json.loads((profile_root / "summary.json").read_text())
    sources = summary["datasets"][label]["sources"]
    datasets = {(entry["dataset"], entry["revision"]) for entry in sources}
    records = []
    with (profile_root / "response-length-records.jsonl").open() as source:
        for line in source:
            entry = json.loads(line)
            if (entry["dataset"], entry["revision"]) not in datasets:
                continue
            if (0 < entry["prompt_tokens"] <= 8192 and 0 < entry["output_tokens"] <= 131072
                    and entry["prompt_tokens"] + entry["output_tokens"] <= 108544):
                records.append(entry)
    expected = sorted(records, key=lambda entry: entry["selection_sha256"])[:maximum]
    exports = [json.loads(line) for line in
               (profile_root / f"{label}-full_short_input-exports.jsonl").read_text().splitlines()]
    if (len(exports) != len(expected) or not expected
            or [entry["selection_sha256"] for entry in exports] !=
               [entry["selection_sha256"] for entry in expected]):
        raise ValueError("Require full-distribution length-independent hash exports, not a selected long tail")
    for entry, record in zip(exports, expected):
        if any(entry.get(field) != value for field, value in record.items()):
            raise ValueError("Export identity/length differs from original profile records")
    return summary, sources, exports


def rebuild_entry(source, exported, row, tokenizer, template):
    messages = original_messages(row)
    boundary = exported["assistant_index"]
    if (type(boundary) is not int or not 0 < boundary < len(messages)
            or messages[boundary]["role"] != "assistant"
            or not any(message["role"] == "user" for message in messages[:boundary])):
        raise ValueError("Export lacks an original assistant boundary")
    response = messages[boundary]["content"]
    response_sha = hashlib.sha256(response.encode()).hexdigest()
    identity = source["dataset"] + "@" + source["revision"] + "/" + source["path"]
    selection_sha = hashlib.sha256(
        f"{identity}|{exported['source_row']}|{boundary}|{response_sha}".encode()).hexdigest()
    rendered = template.render(messages=messages[:boundary], tools=None, add_generation_prompt=True)
    token_ids = tokenizer.encode(rendered, add_special_tokens=False).ids
    response_ids = tokenizer.encode(response, add_special_tokens=False).ids
    if (exported["source_sha256"] != source["sha256"]
            or exported["original_response_sha256"] != response_sha
            or exported["selection_sha256"] != selection_sha
            or exported["input_token_ids"] != token_ids
            or exported["prompt_tokens"] != len(token_ids)
            or exported["output_tokens"] != len(response_ids)):
        raise ValueError("Independent raw reconstruction failed; never override output length or prompt IDs")
    key = "public-" + selection_sha
    entry = {"messages": messages[:boundary], "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
             "token_ids": token_ids, "prompt_tokens": len(token_ids), "original_output_tokens": len(response_ids),
             "original_response_sha256": response_sha, "dataset": source["dataset"],
             "revision": source["revision"], "source_file": source["path"],
             "source_row": exported["source_row"], "assistant_index": boundary,
             "source_sha256": source["sha256"], "selection_sha256": selection_sha}
    mark = {field: value for field, value in entry.items() if field not in ("messages", "token_ids", "prompt_sha256")}
    mark.update({"prompt_key": key, "output_tokens": len(response_ids), "count": 1,
                 "source_fragment": source["dataset"] + "-original-response-length"})
    return key, entry, mark


def prepare(raw_root, profile_root, agentic_inputs, destination, tokenizer_path, config_path,
            label="openthoughts3", maximum=64):
    from jinja2 import Environment, StrictUndefined
    from tokenizers import Tokenizer

    if destination.exists() or type(maximum) is not int or maximum <= 0:
        raise ValueError("Preserve prior public inputs and freeze a positive hash-bank size")
    summary, sources, exports = selected_records(profile_root, label, maximum)
    if (summary["tokenizer_sha256"] != digest(tokenizer_path)
            or summary["tokenizer_config_sha256"] != digest(config_path)):
        raise ValueError("Qwen tokenizer/config changed since profiling")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.padding is not None or tokenizer.truncation is not None:
        raise ValueError("Source lengths cannot be padded or truncated")
    template = Environment(undefined=StrictUndefined).from_string(json.loads(config_path.read_text())["chat_template"])
    old_bank = json.loads((agentic_inputs / "prompt-bank.json").read_text())
    code = json.loads((agentic_inputs / "code-marks.json").read_text())
    if any(mark["prompt_tokens"] < 100000 or mark["output_tokens"] != 64 for mark in code):
        raise ValueError("Keep real SWE-smith 100K/O64 Code inputs")
    bank = {field: old_bank[field] for field in ("tokenizer_sha256", "tokenizer_config_sha256")}
    if any(bank[field] != summary[field] for field in bank):
        raise ValueError("Code and public inputs require the same Qwen tokenizer")
    bank["prompts"] = {mark["prompt_key"]: old_bank["prompts"][mark["prompt_key"]] for mark in code}
    marks = []
    for source in sources:
        filename = Path(source["destination"])
        if filename.name != source["destination"]:
            raise ValueError("Raw source must stay inside the evidence directory")
        path = raw_root / filename
        if path.stat().st_size != source["size"] or digest(path) != source["sha256"]:
            raise ValueError("Official raw LFS size/SHA validation failed")
        entries = [entry for entry in exports if entry["source_file"] == source["destination"]]
        grouped = {}
        for entry in entries:
            grouped.setdefault(entry["source_row"], []).append(entry)
        rebuilt = 0
        for index, row in source_rows(path, set(grouped)):
            for exported in grouped[index]:
                key, prompt, mark = rebuild_entry(source, exported, row, tokenizer, template)
                if key in bank["prompts"]:
                    raise ValueError("Duplicate public prompt identity")
                bank["prompts"][key] = prompt
                marks.append(mark)
                rebuilt += 1
        if rebuilt != len(entries) or digest(path) != source["sha256"]:
            raise ValueError("Raw source/boundary reconstruction is incomplete or source changed")
    marks.sort(key=lambda mark: mark["selection_sha256"])
    if len(marks) != len(exports):
        raise ValueError("Every public export requires independent original reconstruction")
    bank["inputs_sha256"] = {"code_prompt_bank": digest(agentic_inputs / "prompt-bank.json"),
                             "public_profile": digest(profile_root / "summary.json"),
                             **{source["destination"]: source["sha256"] for source in sources}}
    destination.mkdir(parents=True)
    for name, payload in (("prompt-bank.json", bank), ("code-marks.json", code), ("conv-marks.json", marks)):
        (destination / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    manifest = {"schema_version": 1, "dataset_label": label, "sources": sources,
                "bank_size": len(marks), "sampling": "full_short_input_outcome_independent_hash_bank",
                "files_sha256": {name: digest(destination / name) for name in
                                 ("prompt-bank.json", "code-marks.json", "conv-marks.json")},
                "raw_reconstruction_passed": True, "original_response_lengths_only": True,
                "scope": "public_reasoning_length_replay_not_human_conversation_or_Qwen_natural_EOS"}
    (destination / "input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    validate(destination)
    return manifest


def validate(inputs):
    manifest = json.loads((inputs / "input-manifest.json").read_text())
    if (manifest.get("original_response_lengths_only") is not True
            or manifest.get("raw_reconstruction_passed") is not True
            or manifest.get("sampling") != "full_short_input_outcome_independent_hash_bank"):
        raise ValueError("Require independently reconstructed original public lengths")
    for name, expected in manifest["files_sha256"].items():
        if name not in ("prompt-bank.json", "code-marks.json", "conv-marks.json") or digest(inputs / name) != expected:
            raise ValueError("Frozen public inputs changed")
    bank = json.loads((inputs / "prompt-bank.json").read_text())
    conv = json.loads((inputs / "conv-marks.json").read_text())
    code = json.loads((inputs / "code-marks.json").read_text())
    if not code or len(conv) != manifest["bank_size"] or not conv:
        raise ValueError("Both original tenant banks must be nonempty")
    for mark in code + conv:
        key = mark.get("prompt_key")
        if not key or key not in bank["prompts"]:
            raise ValueError("Forbid the synthetic prompt fallback")
        prompt = bank["prompts"][key]
        if (len(prompt["token_ids"]) != mark["prompt_tokens"]
                or prompt["prompt_tokens"] != mark["prompt_tokens"]
                or mark["prompt_tokens"] + mark["output_tokens"] > 131072):
            raise ValueError("Public replay length/IDs exceed their verified bank")
        if mark in conv and (mark.get("synthetic_output_tokens") is not None
                or mark["output_tokens"] != mark.get("original_output_tokens")
                or mark["output_tokens"] != prompt.get("original_output_tokens")
                or mark.get("original_response_sha256") != prompt.get("original_response_sha256")):
            raise ValueError("Conv response length must stay equal to the raw source response")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--profile-root", type=Path)
    parser.add_argument("--agentic-inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--tokenizer-config", type=Path)
    parser.add_argument("--label", default="openthoughts3")
    args = parser.parse_args()
    if args.validate:
        result = validate(args.validate)
    else:
        result = prepare(args.raw_root, args.profile_root, args.agentic_inputs, args.output,
                         args.tokenizer, args.tokenizer_config, args.label)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
