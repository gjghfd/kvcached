import importlib
from pathlib import Path

import pytest


@pytest.fixture
def agentic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("agentic_profile")


def test_text_blocks_are_not_silently_dropped_or_thoughts_duplicated(agentic):
    messages = [{"role": "user", "content": [{"type": "text", "text": "code"},
                                               {"type": "text", "text": "logs"}]},
                {"role": "assistant", "content": "public thought", "thought": "public thought",
                 "tool_calls": [{"function": {"name": "bash", "arguments": '{"command":"ls"}'}}]},
                {"role": "tool", "content": [{"type": "text", "text": "files"}],
                 "tool_call_ids": ["call-1"]}]
    normalized = agentic.normalize_messages(messages)
    assert normalized[0]["content"] == "code\nlogs"
    assert normalized[1]["content"] == "public thought" and "thought" not in normalized[1]
    assert normalized[1]["tool_calls"] == messages[1]["tool_calls"]
    assert normalized[2]["tool_call_id"] == "call-1"


def test_nontext_tool_content_is_rejected(agentic):
    with pytest.raises(ValueError, match="Only text"):
        agentic.normalize_messages([{"role": "tool", "content": [{"type": "image", "url": "x"}]}])


def test_assistant_without_tool_call_has_explicit_empty_list(agentic):
    for tool_calls in (None, []):
        message = {"role": "assistant", "content": "answer", "tool_calls": tool_calls}
        assert agentic.normalize_messages([message])[0]["tool_calls"] == []
    missing_calls = agentic.normalize_messages([{"role": "assistant", "content": "answer"}])
    assert missing_calls[0]["tool_calls"] == []


def test_input_ends_before_next_assistant_not_at_entire_trajectory(agentic):
    messages = agentic.normalize_messages('[{"role":"user","content":"task"},'
                                         '{"role":"assistant","content":"action"},'
                                         '{"role":"tool","content":"output"},'
                                         '{"role":"assistant","content":"future answer"}]')
    assert agentic.assistant_boundaries(messages) == [1, 3]
    prefix = messages[:agentic.assistant_boundaries(messages)[-1]]
    assert prefix[-1]["role"] == "tool"
    assert all(message["content"] != "future answer" for message in prefix)


def test_100k_selection_uses_whole_message_boundaries_no_padding_or_truncation(agentic):
    lengths = [(2, 72000), (4, 99730), (6, 103300), (8, 132000)]
    approximate = agentic.choose_boundary(lengths, 95000, 105000, 100000)
    assert approximate == {"message_index": 4, "prompt_tokens": 99730}
    strict = agentic.choose_boundary(lengths, 100000, 105000, 100000)
    assert strict == {"message_index": 6, "prompt_tokens": 103300}
    assert agentic.choose_boundary(lengths, 100000, 102000, 100000) is None


def test_rendering_uses_generation_prefix_without_tool_schema_invention(agentic):
    class Template:
        def render(self, **values):
            assert values["tools"] is None and values["add_generation_prompt"] is True
            return "agent input"

    assert agentic.render_prompt(Template(), [{"role": "user", "content": "task"}]) == "agent input"
