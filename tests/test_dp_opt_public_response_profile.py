import importlib
from pathlib import Path

import pytest


@pytest.fixture
def profiler(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("public_response_profile")


@pytest.mark.parametrize("field", ["messages", "conversation"])
def test_original_text_is_not_extended_or_rewritten(profiler, field):
    messages = [{"role": "user", "content": "original question"},
                {"role": "assistant", "content": "<think>original reasoning</think> answer"}]
    assert profiler.normalize({field: messages}) == messages
    assert profiler.boundaries(messages) == [1]


def test_sharegpt_roles_are_only_normalized(profiler):
    assert profiler.normalize({"conversations": [{"from": "human", "value": "question"},
        {"from": "gpt", "value": "response"}]}) == [{"role": "user", "content": "question"},
                                                       {"role": "assistant", "content": "response"}]


def test_multiturn_template_defaults_only_missing_tool_metadata(profiler):
    class Template:
        def render(self, **arguments):
            assert arguments["messages"][1]["tool_calls"] == []
            return arguments["messages"][1]["content"]
    messages = [{"role": "user", "content": "original question"},
                {"role": "assistant", "content": "original answer"},
                {"role": "user", "content": "followup"}]
    assert profiler.render_original(Template(), messages) == "original answer"
    assert "tool_calls" not in messages[1]


@pytest.mark.parametrize("row", [{}, {"messages": []}, {"messages": [{"role": "user", "content": None}]},
                                 {"messages": [{"role": "tool", "content": "not a supported text-only turn"}]}])
def test_unsupported_original_data_is_not_silently_flattened(profiler, row):
    with pytest.raises(ValueError):
        profiler.normalize(row)


def test_sampling_precedes_and_cannot_depend_on_response_length(profiler):
    selected = profiler.row_selection(100, 10, "fixed-source")
    assert selected == profiler.row_selection(100, 10, "fixed-source")
    assert len(selected) == 10 and all(0 <= index < 100 for index in selected)
    assert profiler.row_selection(10, 10, "fixed-source") is None


def test_cohorts_keep_source_lengths_and_declare_capacity_constraints(profiler):
    assert profiler.eligible(500, 24577, "original_O24k_32k")
    assert not profiler.eligible(500, 24575, "original_O24k_32k")
    assert not profiler.eligible(8193, 26000, "original_O24k_32k")
    assert not profiler.eligible(500, 110000, "full_short_input")


def test_expected_kv_times_service_not_product_of_means_drives_rate_prior(profiler):
    result = profiler.summarize({(100, 1000): 1, (100, 10000): 1})
    expected_moment = (192 * 1000 + 960 * 10000) / 2
    assert result["mean_page_upper_times_output_mib_tokens"] == expected_moment
    assert result["Conv_RPS_priors"]["1.2"]["0.012"] == pytest.approx(5376 / (expected_moment * .012))
    assert "not_actual_allocator" in result["prior_scope"]


def test_bounded_export_selection_never_uses_performance(profiler):
    heap = []
    for ordinal in (9, 1, 7, 2):
        profiler.keep_export(heap, {"selection_sha256": f"{ordinal:064x}", "source_row": ordinal,
                                   "assistant_index": 1, "output_tokens": 30000 + ordinal}, 2)
    assert sorted(item[-1]["source_row"] for item in heap) == [1, 2]
    assert sorted(item[-1]["output_tokens"] for item in heap) == [30001, 30002]
