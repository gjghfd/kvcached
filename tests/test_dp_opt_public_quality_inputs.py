import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def builder(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("public_quality_inputs")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def archive(builder, root):
    names = ("predeclared-identity.json", "full-source-response-row-map.jsonl",
             "cohort-response-row-map.jsonl", "summary.json")
    dump(root / "archive-manifest.json", {"outputs": {name: {"sha256": builder.source_inputs.digest(root / name)} for name in names}})


@pytest.fixture
def prepared(builder, tmp_path):
    arrow = pytest.importorskip("pyarrow")
    pytest.importorskip("tokenizers")
    import pyarrow.parquet as parquet
    from tokenizers import Tokenizer, models, pre_tokenizers

    raw = tmp_path / "raw"
    raw.mkdir()
    tokenizer = tmp_path / "tokenizer.json"
    encoder = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    encoder.pre_tokenizer = pre_tokenizers.Whitespace()
    encoder.save(str(tokenizer))
    config = tmp_path / "config.json"
    dump(config, {"chat_template": "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}assistant:"})
    sources = []
    originals = {}
    responses = [["<think>r</think> final", "<think>unclosed", "<think>r</think> another final", "<think>r</think>   "],
                 ["<think>reason</think> final " + "long " * 200, "<think>unclosed", "<think>r</think> final </think>", "<think>r</think> f"]]
    for shard, answers in enumerate(responses):
        rows = [{"conversations": [{"from": "human", "value": "question"}, {"from": "gpt", "value": answer}]} for answer in answers]
        path = raw / f"shard-{shard}.parquet"
        parquet.write_table(arrow.Table.from_pylist(rows), path)
        source = {"dataset": "open-thoughts/OpenThoughts3-1.2M", "revision": "fixed", "label": "openthoughts3",
                  "path": f"data/{path.name}", "destination": path.name, "size": path.stat().st_size,
                  "sha256": builder.source_inputs.digest(path)}
        sources.append(source)
        originals[path.name] = rows
    dump(raw / "download-manifest.json", sources)
    original_profile = tmp_path / "original-profile"
    builder.profile.profile(raw, tokenizer, config, original_profile, export_limit=4)
    records = [json.loads(line) for line in (original_profile / "response-length-records.jsonl").read_text().splitlines()]
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    maps = [{**entry, "source_path": next(source["path"] for source in sources if source["destination"] == entry["source_file"]),
             "marker_flags": {"cohort_eligible": builder.quality(originals[entry["source_file"]][entry["source_row"]]["conversations"][1]["value"])}} for entry in records]
    for name, entries in (("full-source-response-row-map.jsonl", maps), ("cohort-response-row-map.jsonl", [entry for entry in maps if entry["marker_flags"]["cohort_eligible"]])):
        (cohort / name).write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    identity = {"full_local_source_rows": 8, "sources": [{"source": source, "rows": 4} for source in sources],
                "profile_summary": {"sha256": builder.source_inputs.digest(original_profile / "summary.json")},
                "profile_records": {"sha256": builder.source_inputs.digest(original_profile / "response-length-records.jsonl")}}
    dump(cohort / "predeclared-identity.json", identity)
    (cohort / "predeclared-identity.sha256").write_text(builder.fingerprint(identity) + "\n")
    dump(cohort / "summary.json", {"cohort_rows": 4})
    archive(builder, cohort)
    C0 = tmp_path / "C0"
    C0.mkdir()
    prompt = {"token_ids": [0] * 100000, "prompt_tokens": 100000, "original_messages": "code unchanged"}
    dump(C0 / "prompt-bank.json", {"tokenizer_sha256": builder.source_inputs.digest(tokenizer),
         "tokenizer_config_sha256": builder.source_inputs.digest(config), "prompts": {"swe-original": prompt, "C0-only-conv": {"token_ids": [0]}}})
    dump(C0 / "code-marks.json", [{"prompt_key": "swe-original", "prompt_tokens": 100000, "output_tokens": 64}])
    dump(C0 / "input-manifest.json", {"dataset_label": "stratos"})
    return {"raw_root": raw, "original_profile": original_profile, "cohort_root": cohort, "agentic_inputs": C0,
            "profile_root": tmp_path / "bank-profile", "destination": tmp_path / "C1/frozen/inputs",
            "tokenizer": tokenizer, "config": config, "maximum": 2}


@pytest.mark.parametrize("text,eligible", [("<think>r</think> final", True), ("prefix <think>r</think> final", True),
    ("<think>r", False), ("<think>r</think>  \n", False), ("<think>r</think> final </think>", False),
    ("</think>bad<think> final", False), ("<think><think>r</think> final", False), ("unmarked", False)])
def test_exact_original_marker_predicate(builder, text, eligible):
    assert builder.quality(text) is eligible


def test_build_reuses_original_API_and_preserves_Code_and_full_cohort(builder, prepared):
    baseline = {path: path.read_bytes() for path in prepared["agentic_inputs"].iterdir()}
    manifest = builder.prepare(**prepared)
    assert manifest["dataset_label"] == "openthoughts3_closed"
    assert manifest["source_quality"]["cohort_rows"] == 4
    assert not manifest["source_quality"]["natural_eos_verified"]
    assert builder.validate(**prepared) == manifest
    assert all(path.read_bytes() == contents for path, contents in baseline.items())
    assert (prepared["destination"] / "code-marks.json").read_bytes() == (prepared["agentic_inputs"] / "code-marks.json").read_bytes()
    records = [json.loads(line) for line in (prepared["profile_root"] / "response-length-records.jsonl").read_text().splitlines()]
    exports = builder.source_inputs.selected_records(prepared["profile_root"], builder.LABEL, 2)[2]
    assert len(records) == 4
    assert [entry["selection_sha256"] for entry in exports] == sorted(entry["selection_sha256"] for entry in records)[:2]


def test_replay_refuses_to_overwrite_existing_build(builder, prepared):
    builder.prepare(**prepared)
    with pytest.raises(ValueError, match="Preserve"):
        builder.prepare(**prepared)


def test_wrong_cohort_rejected_even_after_evidence_rehash(builder, prepared):
    root = prepared["cohort_root"]
    full = [json.loads(line) for line in (root / "full-source-response-row-map.jsonl").read_text().splitlines()]
    full[1]["marker_flags"]["cohort_eligible"] = True
    (root / "full-source-response-row-map.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in full))
    (root / "cohort-response-row-map.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in full if entry["marker_flags"]["cohort_eligible"]))
    archive(builder, root)
    with pytest.raises(ValueError, match="quality predicate"):
        builder.prepare(**prepared)


def test_changed_original_length_rejected_even_after_profile_rehash(builder, prepared):
    root = prepared["original_profile"]
    records = [json.loads(line) for line in (root / "response-length-records.jsonl").read_text().splitlines()]
    for entry in records:
        entry["output_tokens"] += 1
    (root / "response-length-records.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in records))
    cohort = prepared["cohort_root"]
    identity = json.loads((cohort / "predeclared-identity.json").read_text())
    identity["profile_records"]["sha256"] = builder.source_inputs.digest(root / "response-length-records.jsonl")
    dump(cohort / "predeclared-identity.json", identity)
    (cohort / "predeclared-identity.sha256").write_text(builder.fingerprint(identity) + "\n")
    archive(builder, cohort)
    with pytest.raises(ValueError, match="original length"):
        builder.prepare(**prepared)


def test_wrong_parquet_population_rejected_after_identity_rehash(builder, prepared):
    root = prepared["cohort_root"]
    identity = json.loads((root / "predeclared-identity.json").read_text())
    identity["sources"][0]["rows"] -= 1
    dump(root / "predeclared-identity.json", identity)
    (root / "predeclared-identity.sha256").write_text(builder.fingerprint(identity) + "\n")
    archive(builder, root)
    with pytest.raises(ValueError, match="source population"):
        builder.prepare(**prepared)


def test_unselected_record_length_change_is_not_hidden_by_hash_bank(builder, prepared):
    root = prepared["cohort_root"]
    records_path = prepared["original_profile"] / "response-length-records.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    cohort_records = [json.loads(line) for line in (root / "cohort-response-row-map.jsonl").read_text().splitlines()]
    unselected_sha = max(entry["selection_sha256"] for entry in cohort_records)
    next(entry for entry in records if entry["selection_sha256"] == unselected_sha)["output_tokens"] += 1
    records_path.write_text("".join(json.dumps(entry) + "\n" for entry in records))
    identity = json.loads((root / "predeclared-identity.json").read_text())
    identity["profile_records"]["sha256"] = builder.source_inputs.digest(records_path)
    dump(root / "predeclared-identity.json", identity)
    (root / "predeclared-identity.sha256").write_text(builder.fingerprint(identity) + "\n")
    archive(builder, root)
    with pytest.raises(ValueError, match="original length"):
        builder.prepare(**prepared)


@pytest.mark.parametrize("corruption", ["Code_marks", "Code_IDs", "alias", "prompt_alias", "original_O", "quality_identity"])
def test_frozen_validator_rejects_rehashed_corruption(builder, prepared, corruption):
    builder.prepare(**prepared)
    dest = prepared["destination"]
    manifest = json.loads((dest / "input-manifest.json").read_text())
    if corruption in ("Code_IDs", "prompt_alias"):
        path = dest / "prompt-bank.json"
        value = json.loads(path.read_text())
        if corruption == "Code_IDs":
            value["prompts"]["swe-original"]["token_ids"][0] = 1
        else:
            key = next(key for key in value["prompts"] if key.startswith("public-"))
            value["prompts"][key]["source_file"] = "alias.parquet"
        dump(path, value)
        manifest["files_sha256"][path.name] = builder.source_inputs.digest(path)
    elif corruption == "Code_marks":
        path = dest / "code-marks.json"
        value = json.loads(path.read_text())
        value[0]["source_fragment"] = "changed Code"
        dump(path, value)
        manifest["files_sha256"][path.name] = builder.source_inputs.digest(path)
    elif corruption == "original_O":
        path = dest / "conv-marks.json"
        value = json.loads(path.read_text())
        value[0]["output_tokens"] += 1
        dump(path, value)
        manifest["files_sha256"][path.name] = builder.source_inputs.digest(path)
    elif corruption == "alias":
        manifest["dataset_label"] = "stratos"
    else:
        manifest["source_quality"]["cohort_identity_sha256"] = "a" * 64
    dump(dest / "input-manifest.json", manifest)
    with pytest.raises(ValueError):
        builder.validate(**prepared)


def test_search_prepare_validate_frozen_and_fixed_native_draws_are_compatible(builder, prepared):
    builder.prepare(**prepared)
    search = importlib.import_module("public_workload_search")
    root = prepared["destination"].parents[1]
    config = search.prepare(root)
    assert search.validate_frozen(root / "frozen") == config
    assert config["dataset"] == builder.LABEL
    for seed in search.SEEDS:
        for population in search.POPULATIONS:
            trace = search.make_native(prepared["destination"], seed, population)
            assert trace["requests"][-1]["arrival_s"] == 90
            assert trace["model_names"][1] == builder.ALIAS
            assert trace == search.make_native(prepared["destination"], seed, population)
