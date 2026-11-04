import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def inputs_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("public_response_inputs")


def test_raw_multiturn_messages_keep_original_text(inputs_module):
    raw = {"conversations": [{"from": "human", "value": "question"},
                             {"from": "gpt", "value": "original answer"},
                             {"from": "human", "value": "follow up"}]}
    messages = inputs_module.original_messages(raw)
    assert messages[1] == {"role": "assistant", "content": "original answer", "tool_calls": []}
    assert messages[2]["content"] == "follow up"
    assert "tool_calls" not in raw["conversations"][1]


@pytest.mark.parametrize("row", [{}, {"messages": None}, {"messages": [], "conversation": []},
                                 {"messages": [{"role": "user", "content": None}]}])
def test_no_synthetic_raw_conversation_fallback(inputs_module, row):
    with pytest.raises(ValueError):
        inputs_module.original_messages(row)


def entry_fixture():
    response_sha = hashlib.sha256(b"original answer").hexdigest()
    identity = "public/data@revision/data/source.parquet"
    selection_sha = hashlib.sha256(f"{identity}|5|1|{response_sha}".encode()).hexdigest()
    source = {"dataset": "public/data", "revision": "revision", "path": "data/source.parquet", "sha256": "a" * 64}
    exported = {"source_row": 5, "assistant_index": 1, "source_sha256": "a" * 64,
                "original_response_sha256": response_sha, "selection_sha256": selection_sha,
                "input_token_ids": [1, 2], "prompt_tokens": 2, "output_tokens": 3}
    row = {"messages": [{"role": "user", "content": "original question"},
                        {"role": "assistant", "content": "original answer"}]}
    return source, exported, row


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return type("Encoding", (), {"ids": [1, 2, 3] if text == "original answer" else [1, 2]})()


class Template:
    def render(self, messages, tools, add_generation_prompt):
        assert tools is None and add_generation_prompt
        assert messages == [{"role": "user", "content": "original question"}]
        return "rendered original question"


def test_independent_rebuild_keeps_original_output_length_and_real_prompt_key(inputs_module):
    source, exported, row = entry_fixture()
    key, bank, mark = inputs_module.rebuild_entry(source, exported, row, Tokenizer(), Template())
    assert key == "public-" + exported["selection_sha256"]
    assert mark["prompt_key"] == key
    assert mark["output_tokens"] == bank["original_output_tokens"] == 3
    assert "synthetic_output_tokens" not in mark
    assert bank["messages"] == row["messages"][:1]


@pytest.mark.parametrize("field,value", [("output_tokens", 24576), ("input_token_ids", [1, 3]),
                                        ("source_sha256", "b" * 64), ("assistant_index", 0),
                                        ("original_response_sha256", "c" * 64)])
def test_raw_rebuild_rejects_length_extension_and_identity_corruption(inputs_module, field, value):
    source, exported, row = entry_fixture()
    exported[field] = value
    with pytest.raises(ValueError):
        inputs_module.rebuild_entry(source, exported, row, Tokenizer(), Template())


def test_full_distribution_hash_selection_cannot_be_replaced_with_long_tail(inputs_module, tmp_path):
    sources = [{"dataset": "public/data", "revision": "fixed"}]
    records = [{"dataset": "public/data", "revision": "fixed", "prompt_tokens": 100,
                "output_tokens": output, "selection_sha256": str(index)}
               for index, output in ((1, 1000), (2, 16000), (3, 24000))]
    (tmp_path / "summary.json").write_text(json.dumps({"datasets": {"dataset": {"sources": sources}}}))
    (tmp_path / "response-length-records.jsonl").write_text("\n".join(json.dumps(record) for record in records))
    exports = tmp_path / "dataset-full_short_input-exports.jsonl"
    exports.write_text("\n".join(json.dumps(record) for record in records[1:]))
    with pytest.raises(ValueError, match="long tail"):
        inputs_module.selected_records(tmp_path, "dataset", 2)
    exports.write_text("\n".join(json.dumps(record) for record in records[:2]))
    assert len(inputs_module.selected_records(tmp_path, "dataset", 2)[2]) == 2
