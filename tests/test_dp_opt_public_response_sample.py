import argparse
import hashlib
import importlib
import json
import urllib.error
import urllib.parse
from pathlib import Path

import pytest


@pytest.fixture
def sampler(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("public_response_sample")


def fake_page(offset, length, total):
    return {"partial": False, "num_rows_total": total, "rows": [
        {"row_idx": offset + ordinal, "truncated_cells": [], "row": {
            "messages": [{"role": "user", "content": f"原始 question {offset + ordinal}\n"},
                         {"role": "assistant", "content": "<think>原始\n</think> answer"}],
            "generations": ["<think>原始\n</think> answer"], "is_reasoning_complete": [True],
            "finish_reasons": None}} for ordinal in range(length)]}


def install_fetch(sampler, monkeypatch, sample_identity, transform=None):
    calls, waits = [], []

    def fetch(url):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        offset, length = int(query["offset"][0]), int(query["length"][0])
        calls.append(offset)
        page = fake_page(offset, length, sample_identity["population_count"])
        if transform:
            transform(page)
        return json.dumps(page, ensure_ascii=False).encode("utf-8")

    monkeypatch.setattr(sampler.snapshot, "fetch", fetch)
    monkeypatch.setattr(sampler.time, "sleep", waits.append)
    return calls, waits


def test_fixed_selection_spans_all_population_strata_not_first1600(sampler):
    sample_identity = sampler.identity("open-r1/OpenR1-Math-220k", 93733)
    pages = sampler.select_pages(sample_identity)
    assert pages == sampler.select_pages(sample_identity)
    assert len(pages) == 16
    assert all(page["length"] == 100 for page in pages)
    assert pages[-1]["offset"] > 85000
    assert len({page["offset"] for page in pages}) == 16
    assert sampler.select_pages({**sample_identity, "seed": 20261032}) != pages
    assert sampler.select_pages({**sample_identity, "population_count": 93734}) != pages
    assert all(not any(key in page for key in ("output_tokens", "benefit", "length_score")) for page in pages)


@pytest.mark.parametrize("kwargs", [{"population": 10}, {"population": True}, {"seed": False},
                                    {"page_count": 17}, {"page_length": 101}, {"page_count": 0}])
def test_invalid_or_unbounded_sample_rejected(sampler, kwargs):
    values = {"dataset": "official/source", "population": 93733, **kwargs}
    with pytest.raises(ValueError):
        sampler.identity(**values)


def test_raw_strings_sha_and_local_global_row_mapping(sampler, monkeypatch, tmp_path):
    sample_identity = sampler.identity("official/source", 93, page_count=2, page_length=2)
    calls, waits = install_fetch(sampler, monkeypatch, sample_identity)
    manifest = sampler.download(tmp_path, sample_identity)
    check = sampler.verify(tmp_path)
    assert check["rows"] == 4 and check["pages"] == 2
    assert check["all_message_strings_reconstructed"] and check["global_row_mapping_verified"]
    assert check["sample_only"] and check["not_full_population"] and check["no_GPU_benefit"]
    assert not check["natural_eos_verified"] and not check["source_git_revision_verified"]
    assert calls == [page["offset"] for page in manifest["pages"]]
    assert waits == [2, 2]
    sources = json.loads((tmp_path / "download-manifest.json").read_text())
    for source in sources:
        path = tmp_path / source["destination"]
        assert source["revision"].startswith("viewer-sample:")
        assert source["revision"] == manifest["revision"]
        assert str(source["page_offset"]).zfill(8) in source["path"]
        records = list(sampler.profiler.rows(path, "openr1", "unused"))
        assert [index for index, row in records] == [0, 1]
        assert [row["source_row"] for index, row in records] == [source["page_offset"], source["page_offset"] + 1]
        assert records[0][1]["messages"][0]["content"].endswith("\n")
        assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert sampler.download(tmp_path, sample_identity) == manifest
    assert len(calls) == 2


@pytest.mark.parametrize("change", ["partial", "truncated", "missing", "reordered", "total", "bad_role"])
def test_invalid_source_preserved_and_not_redrawn(sampler, monkeypatch, tmp_path, change):
    sample_identity = sampler.identity("official/source", 93, page_count=2, page_length=2)

    def transform(page):
        if change == "partial":
            page["partial"] = True
        elif change == "truncated":
            page["rows"][0]["truncated_cells"] = ["messages"]
        elif change == "missing":
            page["rows"].pop()
        elif change == "reordered":
            page["rows"].reverse()
        elif change == "total":
            page["num_rows_total"] += 1
        else:
            page["rows"][0]["row"]["messages"][0]["role"] = "human"

    calls, waits = install_fetch(sampler, monkeypatch, sample_identity, transform)
    selected = sampler.select_pages(sample_identity)
    with pytest.raises(ValueError):
        sampler.download(tmp_path, sample_identity)
    assert calls == [selected[0]["offset"]]
    assert (tmp_path / f"page-{selected[0]['offset']:08d}.json").exists()
    assert not (tmp_path / "manifest.json").exists()
    with pytest.raises(ValueError, match="Unreceipted"):
        sampler.download(tmp_path, sample_identity)
    assert len(calls) == 1 and waits == [2]


def test_429_uses_existing_retry_after_with_spacing(sampler, monkeypatch, tmp_path):
    sample_identity = sampler.identity("official/source", 3, page_count=1, page_length=2)
    calls, waits = [], []

    def fetch(url):
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 429, "rate limited", {"Retry-After": "240"}, None)
        offset = sampler.select_pages(sample_identity)[0]["offset"]
        return json.dumps(fake_page(offset, 2, 3)).encode()

    monkeypatch.setattr(sampler.snapshot, "fetch", fetch)
    monkeypatch.setattr(sampler.time, "sleep", waits.append)
    sampler.download(tmp_path, sample_identity)
    assert waits == [2, 240, 2] and calls[0] == calls[1]
    status = [json.loads(line) for line in (tmp_path / "status.jsonl").read_text().splitlines()]
    assert any(item["stage"] == "network_failure" and item["retry_after_s"] == 240 for item in status)


def test_resume_keeps_completed_pages_and_frozen_selection(sampler, monkeypatch, tmp_path):
    sample_identity = sampler.identity("official/source", 93, page_count=2, page_length=2)
    selected = sampler.select_pages(sample_identity)
    calls, waits = [], []

    def fetch(url):
        offset = int(urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["offset"][0])
        calls.append(offset)
        if offset == selected[1]["offset"]:
            raise urllib.error.URLError("TLS reset")
        return json.dumps(fake_page(offset, 2, 93)).encode()

    monkeypatch.setattr(sampler.snapshot, "fetch", fetch)
    monkeypatch.setattr(sampler.time, "sleep", waits.append)
    with pytest.raises(urllib.error.URLError):
        sampler.download(tmp_path, sample_identity)
    assert calls == [selected[0]["offset"]] + [selected[1]["offset"]] * 6
    assert len(list(tmp_path.glob("*-receipt.json"))) == 1
    resumed_calls, unused_waits = install_fetch(sampler, monkeypatch, sample_identity)
    sampler.download(tmp_path, sample_identity)
    assert resumed_calls == [selected[1]["offset"]]
    assert sampler.verify(tmp_path)["rows"] == 4


@pytest.mark.parametrize("change", ["raw", "canonical", "identity", "selection", "source_offset", "revision", "EOS"])
def test_independent_verifier_rejects_tampering(sampler, monkeypatch, tmp_path, change):
    sample_identity = sampler.identity("official/source", 93, page_count=2, page_length=2)
    install_fetch(sampler, monkeypatch, sample_identity)
    manifest = sampler.download(tmp_path, sample_identity)
    if change in ("raw", "canonical"):
        path = tmp_path / manifest["pages"][0][change + "_path"]
        path.write_text(path.read_text() + " ")
    elif change in ("source_offset", "revision"):
        path = tmp_path / "download-manifest.json"
        value = json.loads(path.read_text())
        if change == "source_offset":
            value[0]["page_offset"] += 1
        else:
            value[0]["revision"] = "upstream-git-sha"
        path.write_text(json.dumps(value))
    else:
        path = tmp_path / "identity.json"
        value = json.loads(path.read_text())
        if change == "identity":
            value["identity"]["seed"] += 1
        elif change == "selection":
            value["selected_pages"][0]["offset"] += 1
        else:
            value["natural_eos_verified"] = True
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        sampler.verify(tmp_path)


def test_independent_message_reconstruction_rejects_rehashed_changed_strings(sampler, monkeypatch, tmp_path):
    sample_identity = sampler.identity("official/source", 93, page_count=2, page_length=2)
    install_fetch(sampler, monkeypatch, sample_identity)
    manifest = sampler.download(tmp_path, sample_identity)
    item = manifest["pages"][0]
    path = tmp_path / item["canonical_path"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["messages"][1]["content"] += " artificial extension"
    path.write_text("".join(sampler.canonical(row) + "\n" for row in rows))
    item["canonical_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    item["canonical_size"] = path.stat().st_size
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Canonical strings"):
        sampler.verify(tmp_path)


def test_theory_uses_joint_bO_moment_and_explicit_units(sampler):
    result = sampler.theoretical_pressure({(100, 1000): 1, (100, 10000): 1})
    expected_moment = (192 * 1000 + 960 * 10000) / 2
    assert result["E_bO_mib_tokens"] == expected_moment
    assert result["E_bO_over_E_O_mib"] == pytest.approx(expected_moment / 5500)
    assert result["Conv_population_prior"] == pytest.approx(5376 / (expected_moment / 5500))
    for step in (.015, .020, .030):
        assert result["Conv_RPS_priors"][str(step)] == pytest.approx(5376 / (expected_moment * step))
    assert result["no_GPU_benefit"] and result["sample_only"]


def test_real_profiler_and_independent_original_flags(sampler, monkeypatch, tmp_path):
    pytest.importorskip("tokenizers")
    pytest.importorskip("jinja2")
    root = tmp_path / "sample"
    output = tmp_path / "profile"
    sample_identity = sampler.identity("official/source", 93, page_count=2, page_length=2)
    install_fetch(sampler, monkeypatch, sample_identity)
    sampler.download(root, sample_identity)
    evidence = Path(__file__).resolve().parents[2] / "evidence/mixed-workload-profile-20261021"
    tokenizer = evidence / "qwen-tokenizer.json"
    config = evidence / "qwen-tokenizer-config.json"
    if not tokenizer.exists():
        pytest.skip("Original evidence tokenizer is unavailable")
    sampler.profiler.profile(root, tokenizer, config, output, export_limit=64)
    result = sampler.verify_profile(root, output, tokenizer, config)
    assert result["rows"] == result["responses"] == 4
    assert result["original_flags"]["finish_reasons_null_rows"] == 4
    assert result["original_flags"]["response_exactly_matches_generation"] == 4
    assert result["original_flags"]["matched_reasoning_complete:True"] == 4
    audits = [json.loads(line) for line in (output / "independent-response-audit.jsonl").read_text().splitlines()]
    assert all(record["source_global_row"] == record["source_page_offset"] + record["source_row"] for record in audits)
    assert not result["natural_eos_verified"]


def test_worker_retains_terminal_failure_status(sampler, monkeypatch, tmp_path):
    args = argparse.Namespace(destination=tmp_path, dataset="official/source", population=93, seed=20261031,
        config="default", split="train", page_count=2, page_length=2, label="openr1", info=None,
        output=tmp_path / "profile", tokenizer=tmp_path / "tokenizer", tokenizer_config=tmp_path / "config")
    sample_identity = sampler.identity(args.dataset, args.population, page_count=2, page_length=2)
    install_fetch(sampler, monkeypatch, sample_identity, lambda page: page.update(partial=True))
    with pytest.raises(ValueError):
        sampler.run(args)
    status = [json.loads(line) for line in (tmp_path / "status.jsonl").read_text().splitlines()]
    assert status[-1]["stage"] == "failed" and status[-1]["fixed_selection_retained"]
    assert status[-1]["pid"] > 0 and status[-1]["cwd"] and status[-1]["identity_sha256"]
