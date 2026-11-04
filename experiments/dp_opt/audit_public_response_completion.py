import argparse
import collections
import hashlib
import json
from pathlib import Path

from public_response_inputs import validate
from public_response_profile import boundaries, digest, normalize, rows


MARKERS = (("<think>", "</think>"),
           ("<|begin_of_thought|>", "<|end_of_thought|>"))


def marker_status(response):
    opening = sum(response.count(start) for start, _ in MARKERS)
    if not opening:
        return "no_known_thought_marker"
    unclosed = any(response.count(start) > response.count(end) for start, end in MARKERS)
    return "unclosed_thought_marker" if unclosed else "closed_thought_marker"


def response_key(source, row_index, boundary):
    return (source["dataset"], source["revision"], source["path"], row_index, boundary)


def audit(raw_root, sources, inputs=None):
    expected = {}
    if inputs is not None:
        manifest = validate(inputs)
        if manifest["sources"] != sources:
            raise ValueError("Audit sources differ from the frozen public input manifest")
        bank = json.loads((inputs / "prompt-bank.json").read_text())
        for entry in bank["prompts"].values():
            if entry.get("dataset") != sources[0]["dataset"]:
                continue
            key = (entry["dataset"], entry["revision"], entry["source_file"],
                   entry["source_row"], entry["assistant_index"])
            if key in expected:
                raise ValueError("Duplicate original response identity in the frozen bank")
            expected[key] = entry["original_response_sha256"]
        if not expected:
            raise ValueError("No original public responses in the frozen bank")
    source_counts, bank_counts, seen = [], collections.Counter(), set()
    for source in sources:
        path = raw_root / source["destination"]
        if path.stat().st_size != source["size"] or digest(path) != source["sha256"]:
            raise ValueError("Raw public source size/SHA differs from the official frozen source")
        identity = source["dataset"] + "@" + source["revision"] + "/" + source["path"]
        counts = collections.Counter()
        for row_index, row in rows(path, source["label"], identity):
            messages = normalize(row)
            for boundary in boundaries(messages):
                response = messages[boundary]["content"]
                status = marker_status(response)
                counts[status] += 1
                key = response_key(source, row_index, boundary)
                if key in expected:
                    if hashlib.sha256(response.encode()).hexdigest() != expected[key]:
                        raise ValueError("Frozen bank response SHA differs from the original source")
                    bank_counts[status] += 1
                    seen.add(key)
        source_counts.append({"source": source, "responses": sum(counts.values()),
                              "thought_marker_status": dict(counts)})
    if seen != set(expected):
        raise ValueError("Did not reconstruct every frozen public response identity")
    aggregate = collections.Counter()
    for source in source_counts:
        aggregate.update(source["thought_marker_status"])
    return {"sources": source_counts, "responses": sum(aggregate.values()),
            "thought_marker_status": dict(aggregate), "bank_responses_sha_verified": len(seen),
            "bank_thought_marker_status": dict(bank_counts),
            "natural_eos_or_semantic_completion_verified": False,
            "interpretation": "Marker balance is descriptive, not proof of truncation, correctness or natural EOS",
            "sampling_changed": False, "research_benefit_verified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.inputs is None) == (args.source_manifest is None):
        parser.error("Choose exactly one frozen input bank or raw source manifest")
    sources = (json.loads((args.inputs / "input-manifest.json").read_text())["sources"]
               if args.inputs else json.loads(args.source_manifest.read_text()))
    result = audit(args.raw_root, sources, args.inputs)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
