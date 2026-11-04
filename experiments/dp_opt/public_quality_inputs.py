import argparse
import collections
import hashlib
import json
from pathlib import Path

import public_response_inputs as source_inputs
import public_response_profile as profile


LABEL = "openthoughts3_closed"
ALIAS = LABEL + "-original-response"


def load(path):
    return json.loads(path.read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def write(path, value):
    with path.open("x") as target:
        target.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def quality(text):
    return (text.count("<think>") == text.count("</think>") == 1
            and text.index("<think>") < text.index("</think>")
            and bool(text[text.index("</think>") + 8:].strip()))


def checked_records(raw_root, original_profile, cohort_root, agentic_inputs):
    import pyarrow.parquet as parquet

    identity = load(cohort_root / "predeclared-identity.json")
    archive = load(cohort_root / "archive-manifest.json")
    names = ("predeclared-identity.json", "full-source-response-row-map.jsonl",
             "cohort-response-row-map.jsonl", "summary.json")
    hashes = {name: source_inputs.digest(cohort_root / name) for name in names}
    require(all(archive["outputs"][name]["sha256"] == hashes[name] for name in names), "Changed cohort evidence")
    require(fingerprint(identity) == (cohort_root / "predeclared-identity.sha256").read_text().strip(), "Cohort identity SHA")
    for field, name in (("profile_summary", "summary.json"), ("profile_records", "response-length-records.jsonl")):
        require(source_inputs.digest(original_profile / name) == identity[field]["sha256"], "Changed original profile")
    rows = [json.loads(line) for line in (cohort_root / names[1]).read_text().splitlines()]
    eligible = [json.loads(line) for line in (cohort_root / names[2]).read_text().splitlines()]
    key = lambda entry: (entry["source_file"], entry["source_row"], entry["assistant_index"])
    mapping = {key(entry): entry for entry in rows}
    require(len(mapping) == len(rows) == identity["full_local_source_rows"], "Incomplete full-source map")
    require(eligible == [entry for entry in rows if entry["marker_flags"]["cohort_eligible"]], "Wrong cohort map")
    records = [json.loads(line) for line in (original_profile / "response-length-records.jsonl").read_text().splitlines()]
    sources = [item["source"] for item in identity["sources"]]
    records = [entry for entry in records if entry["source_file"] in {source["destination"] for source in sources}]
    original_count = len(records)
    records = {key(entry): entry for entry in records}
    require(len(records) == original_count == len(mapping), "Incomplete or duplicate original profile")
    result = []
    for item in identity["sources"]:
        source = item["source"]
        require(Path(source["destination"]).name == source["destination"], "Raw source path escape")
        path = raw_root / source["destination"]
        require(path.stat().st_size == source["size"] and source_inputs.digest(path) == source["sha256"], "Changed original LFS source")
        require(parquet.ParquetFile(path).metadata.num_rows == item["rows"], "Wrong original source population")
        count = 0
        for index, raw in source_inputs.source_rows(path, set(range(item["rows"]))):
            messages = source_inputs.original_messages(raw)
            boundaries = profile.boundaries(messages)
            require(len(boundaries) == 1, "Original response boundary changed")
            boundary = boundaries[0]
            row_key = (source["destination"], index, boundary)
            require(row_key in records and row_key in mapping, "Wrong source row/cohort alias")
            record, mapped = records.pop(row_key), mapping.pop(row_key)
            text = messages[boundary]["content"]
            response_sha = hashlib.sha256(text.encode()).hexdigest()
            original_identity = source["dataset"] + "@" + source["revision"] + "/" + source["path"]
            require(record["original_response_sha256"] == mapped["original_response_sha256"] == response_sha
                    and record["source_sha256"] == source["sha256"] and record["dataset"] == source["dataset"]
                    and record["revision"] == mapped["revision"] == source["revision"]
                    and mapped["source_sha256"] == source["sha256"] and mapped["source_path"] == source["path"]
                    and record["selection_sha256"] == hashlib.sha256(f"{original_identity}|{index}|{boundary}|{response_sha}".encode()).hexdigest(), "Original response provenance mismatch")
            require((record["prompt_tokens"], record["output_tokens"]) == (mapped["prompt_tokens"], mapped["output_tokens"]), "Changed original length in source map")
            require(mapped["marker_flags"]["cohort_eligible"] == quality(text), "Wrong source quality predicate")
            if quality(text):
                require(profile.eligible(record["prompt_tokens"], record["output_tokens"], "full_short_input"), "Do not capacity-filter the quality cohort")
                result.append(record)
            count += 1
        require(count == item["rows"], "Incomplete raw source")
    require(not records and not mapping and len(result) == len(eligible), "Incomplete 20k/cohort join")
    code = load(agentic_inputs / "code-marks.json")
    old_bank = load(agentic_inputs / "prompt-bank.json")
    binding = {"dataset_label": LABEL, "tenant_alias": ALIAS, "cohort_identity_sha256": fingerprint(identity),
               "cohort_files_sha256": hashes, "full_source_rows": len(rows), "cohort_rows": len(result),
               "C0_manifest_sha256": source_inputs.digest(agentic_inputs / "input-manifest.json"),
               "Code_marks_sha256": source_inputs.digest(agentic_inputs / "code-marks.json"),
               "Code_prompts_sha256": {mark["prompt_key"]: fingerprint(old_bank["prompts"][mark["prompt_key"]]) for mark in code},
               "not_full_population": True, "natural_eos_verified": False, "no_GPU_benefit": True}
    return result, [{**source, "label": LABEL} for source in sources], binding


def original_exports(raw_root, sources, records, tokenizer, config, maximum):
    from jinja2 import Environment, StrictUndefined
    from tokenizers import Tokenizer

    encoder = Tokenizer.from_file(str(tokenizer))
    require(encoder.padding is None and encoder.truncation is None, "Forbid padding/truncation")
    template = Environment(undefined=StrictUndefined).from_string(load(config)["chat_template"])
    selected = sorted(records, key=lambda record: record["selection_sha256"])[:maximum]
    rebuilt = {}
    for source in sources:
        group = {entry["source_row"]: entry for entry in selected if entry["source_file"] == source["destination"]}
        for index, row in source_inputs.source_rows(raw_root / source["destination"], set(group)):
            entry = group[index]
            messages = source_inputs.original_messages(row)
            rendered = template.render(messages=messages[:entry["assistant_index"]], tools=None, add_generation_prompt=True)
            token_ids = encoder.encode(rendered, add_special_tokens=False).ids
            require(len(token_ids) == entry["prompt_tokens"] and len(encoder.encode(messages[entry["assistant_index"]]["content"], add_special_tokens=False).ids) == entry["output_tokens"], "Changed original length/tokenization")
            exported = {**entry, "input_token_ids": token_ids}
            rebuilt[entry["selection_sha256"]] = (exported, source_inputs.rebuild_entry(source, exported, row, encoder, template))
    require(len(rebuilt) == maximum == len(selected), "Incomplete original bank reconstruction")
    return [rebuilt[entry["selection_sha256"]] for entry in selected]


def validate(raw_root, original_profile, cohort_root, agentic_inputs, profile_root, destination, tokenizer, config, maximum=256):
    manifest = source_inputs.validate(destination)
    records, sources, binding = checked_records(raw_root, original_profile, cohort_root, agentic_inputs)
    require(manifest["dataset_label"] == LABEL and manifest["sources"] == sources and manifest.get("source_quality") == binding, "Quality identity/alias mismatch")
    summary, _, exports = source_inputs.selected_records(profile_root, LABEL, maximum)
    require(summary.get("source_quality") == binding and summary["tokenizer_sha256"] == source_inputs.digest(tokenizer)
            and summary["tokenizer_config_sha256"] == source_inputs.digest(config), "Profile quality/tokenizer mismatch")
    require([json.loads(line) for line in (profile_root / "response-length-records.jsonl").read_text().splitlines()] == records, "Wrong cohort profile records")
    require(manifest["quality_profile_sha256"] == source_inputs.digest(profile_root / "summary.json"), "Changed quality profile")
    require(manifest["quality_exports_sha256"] == source_inputs.digest(profile_root / f"{LABEL}-full_short_input-exports.jsonl"), "Changed quality exports")
    rebuilt = original_exports(raw_root, sources, records, tokenizer, config, maximum)
    require(exports == [entry for entry, _ in rebuilt], "Wrong whole-cohort hash selection")
    bank, marks = load(destination / "prompt-bank.json"), load(destination / "conv-marks.json")
    require(bank.get("source_quality") == binding and len(marks) == maximum, "Bank quality identity mismatch")
    for mark, (_, (key, prompt, expected)) in zip(marks, rebuilt):
        require(mark == expected and bank["prompts"].get(key) == prompt, "Original bank entry/alias mismatch")
    code = load(agentic_inputs / "code-marks.json")
    old_bank = load(agentic_inputs / "prompt-bank.json")
    require(source_inputs.digest(destination / "code-marks.json") == binding["Code_marks_sha256"], "Code mismatch")
    require(set(bank["prompts"]) == {mark["prompt_key"] for mark in code + marks}
            and all(bank["prompts"][mark["prompt_key"]] == old_bank["prompts"][mark["prompt_key"]] for mark in code), "Code token IDs/values mismatch")
    return manifest


def prepare(raw_root, original_profile, cohort_root, agentic_inputs, profile_root, destination, tokenizer, config, maximum=256):
    require(not profile_root.exists() and not destination.exists() and type(maximum) is int and 0 < maximum <= 256, "Preserve existing build; bounded bank only")
    records, sources, binding = checked_records(raw_root, original_profile, cohort_root, agentic_inputs)
    rebuilt = original_exports(raw_root, sources, records, tokenizer, config, maximum)
    original = load(original_profile / "summary.json")
    statistics = profile.summarize(collections.Counter((entry["prompt_tokens"], entry["output_tokens"]) for entry in records))
    summary = {"tokenizer_sha256": original["tokenizer_sha256"], "tokenizer_config_sha256": original["tokenizer_config_sha256"],
               "source_quality": binding, "scope": "two_shard_marker_quality_subset_not_natural_EOS",
               "datasets": {LABEL: {"sources": sources, "sampled_rows": len(records), "invalid_rows": 0,
                                    "all_sampled_responses": statistics, "cohorts": {"full_short_input": {**statistics, "exported_original_pairs": maximum}}}}}
    profile_root.mkdir(parents=True)
    write(profile_root / "summary.json", summary)
    for name, entries in (("response-length-records.jsonl", records), (f"{LABEL}-full_short_input-exports.jsonl", [entry for entry, _ in rebuilt])):
        with (profile_root / name).open("x") as target:
            for entry in entries:
                target.write(json.dumps(entry, ensure_ascii=False) + "\n")
    manifest = source_inputs.prepare(raw_root, profile_root, agentic_inputs, destination, tokenizer, config, LABEL, maximum)
    bank = load(destination / "prompt-bank.json")
    bank["source_quality"] = binding
    (destination / "prompt-bank.json").write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n")
    (destination / "code-marks.json").write_bytes((agentic_inputs / "code-marks.json").read_bytes())
    manifest.update(source_quality=binding, quality_profile_sha256=source_inputs.digest(profile_root / "summary.json"),
                    quality_exports_sha256=source_inputs.digest(profile_root / f"{LABEL}-full_short_input-exports.jsonl"))
    manifest["files_sha256"] = {name: source_inputs.digest(destination / name) for name in manifest["files_sha256"]}
    (destination / "input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return validate(raw_root, original_profile, cohort_root, agentic_inputs, profile_root, destination, tokenizer, config, maximum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    for name in ("raw-root", "original-profile", "cohort-root", "agentic-inputs", "profile-root", "destination", "tokenizer", "config"):
        parser.add_argument("--" + name, type=Path, required=True)
    arguments = vars(parser.parse_args())
    validator = arguments.pop("validate")
    print(json.dumps((validate if validator else prepare)(**arguments), indent=2))


if __name__ == "__main__":
    main()
