import json
from pathlib import Path
import sys
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
import dataset_viewer_snapshot as snapshot


def page_fixture(offset=0, count=2):
    return {"partial": False, "num_rows_total": count,
            "rows": [{"row_idx": offset + ordinal, "truncated_cells": [],
                      "row": {"system": "original system", "conversations": [
                          {"from": "user", "value": "original question"},
                          {"from": "assistant", "value": "original response"}]}}
                     for ordinal in range(count - offset)]}


@pytest.mark.parametrize("field,value", [("partial", True), ("num_rows_total", 0)])
def test_partial_dataset_fails(field, value):
    page = page_fixture()
    page[field] = value
    with pytest.raises(ValueError):
        snapshot.validate_page(page, 0, 100)


@pytest.mark.parametrize("field,value", [("row_idx", 50), ("truncated_cells", ["conversations"]),
                                         ("truncated_cells", None)])
def test_truncated_or_reordered_cells_fail(field, value):
    page = page_fixture()
    page["rows"][0][field] = value
    with pytest.raises(ValueError):
        snapshot.validate_page(page, 0, 100)


def test_missing_rows_fail():
    page = page_fixture()
    page["rows"].pop()
    with pytest.raises(ValueError, match="misses original rows"):
        snapshot.validate_page(page, 0, 100)


def test_source_change_fails():
    with pytest.raises(ValueError, match="row count changed"):
        snapshot.validate_page(page_fixture(), 0, 100, 3)


def test_original_system_is_not_dropped_or_synthesized():
    row = page_fixture()["rows"][0]["row"]
    messages = snapshot.original_messages(row)
    assert messages[0] == {"role": "system", "content": row["system"]}
    assert messages[1]["content"] == "original question"
    assert messages[2]["content"] == "original response"


def test_ambiguous_system_fails():
    row = {"system": "one", "messages": [{"role": "system", "content": "two"}]}
    with pytest.raises(ValueError, match="unambiguous"):
        snapshot.original_messages(row)


def test_snapshot_freezes_content_not_unverified_git_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "fetch", lambda _: json.dumps(page_fixture()).encode())
    result = snapshot.download(tmp_path, "official/source")
    assert result["rows"] == 2
    assert result["source_git_revision_verified"] is False
    assert result["endpoint_supports_git_revision"] is False
    assert result["source"]["sha256"] == snapshot.digest(tmp_path / "source.jsonl")
    assert len((tmp_path / "source.jsonl").read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="already frozen"):
        snapshot.download(tmp_path, "official/source")


def test_download_bound_fails_before_export(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "fetch", lambda _: json.dumps(page_fixture()).encode())
    with pytest.raises(ValueError, match="download bound"):
        snapshot.download(tmp_path, "official/source", maximum_rows=1)
    assert not (tmp_path / "source.jsonl").exists()


def test_cache_identity_cannot_change(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "fetch", lambda _: json.dumps(page_fixture()).encode())
    snapshot.download(tmp_path, "official/source")
    with pytest.raises(ValueError, match="different public source"):
        snapshot.download(tmp_path, "other/source")


@pytest.mark.parametrize("advertised,expected", [("300", 300), ("1", 120), ("bad", 120), (None, 120)])
def test_rate_limit_backoff_respects_retry_after(advertised, expected):
    headers = {"Retry-After": advertised} if advertised else {}
    error = urllib.error.HTTPError("https://official", 429, "rate limited", headers, None)
    assert snapshot.retry_delay(error, 0) == expected


def test_rate_limit_retry_keeps_source_identity(tmp_path, monkeypatch):
    attempts, waits = [], []

    def fetch(_):
        attempts.append(True)
        if len(attempts) == 1:
            raise urllib.error.HTTPError("https://official", 429, "rate limited", {"Retry-After": "240"}, None)
        return json.dumps(page_fixture()).encode()

    monkeypatch.setattr(snapshot, "fetch", fetch)
    monkeypatch.setattr(snapshot.time, "sleep", waits.append)
    assert snapshot.download(tmp_path, "official/source")["rows"] == 2
    assert waits == [240]
