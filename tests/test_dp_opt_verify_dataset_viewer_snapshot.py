import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
import dataset_viewer_snapshot as snapshot
import verify_dataset_viewer_snapshot as verifier


def make_snapshot(root, monkeypatch):
    page = {"partial": False, "num_rows_total": 1, "rows": [
        {"row_idx": 0, "truncated_cells": [], "row": {"system": "original system", "conversations": [
            {"from": "user", "value": "original question"},
            {"from": "assistant", "value": "original response"}]}}]}
    monkeypatch.setattr(snapshot, "fetch", lambda _: json.dumps(page).encode())
    snapshot.download(root, "official/source")


def test_independent_original_conversion_and_sha(tmp_path, monkeypatch):
    make_snapshot(tmp_path, monkeypatch)
    result = verifier.verify(tmp_path)
    assert result["rows"] == 1
    assert result["all_original_messages_independently_reconstructed"] is True
    assert result["source_git_revision_verified"] is False


def test_changed_message_even_with_updated_source_sha_fails(tmp_path, monkeypatch):
    make_snapshot(tmp_path, monkeypatch)
    source = tmp_path / "source.jsonl"
    row = json.loads(source.read_text())
    row["messages"][-1]["content"] = "extended response"
    source.write_text(json.dumps(row) + "\n")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"].update({"size": source.stat().st_size, "sha256": verifier.file_digest(source)})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="original public source cells"):
        verifier.verify(tmp_path)


def test_unsupported_git_verification_claim_fails(tmp_path, monkeypatch):
    make_snapshot(tmp_path, monkeypatch)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_git_revision_verified"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="upstream Git/LFS"):
        verifier.verify(tmp_path)


def test_source_path_cannot_escape_root(tmp_path, monkeypatch):
    make_snapshot(tmp_path / "raw", monkeypatch)
    with pytest.raises(ValueError, match="inside its raw evidence root"):
        verifier.contained(tmp_path / "raw", "../outside.json")
