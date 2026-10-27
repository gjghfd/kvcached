import argparse
import collections
import hashlib
import json
from pathlib import Path

from workload_profile import histogram_statistics


def normalize_messages(messages):
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected a nonempty agent message list")
    normalized = []
    for message in messages:
        role = message["role"]
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError("Unsupported agent message role")
        content = message.get("content", "")
        if content is None:
            content = ""
        if isinstance(content, list):
            if any(part.get("type") != "text" or not isinstance(part.get("text"), str)
                   for part in content):
                raise ValueError("Only text agent messages can be replayed")
            content = "\n".join(part["text"] for part in content)
        if not isinstance(content, str):
            raise ValueError("Expected textual message content")
        entry = {"role": role, "content": content}
        if role == "assistant":
            entry["tool_calls"] = message.get("tool_calls") or []
        if role == "tool" and len(message.get("tool_call_ids", [])) == 1:
            entry["tool_call_id"] = message["tool_call_ids"][0]
        normalized.append(entry)
    return normalized


def assistant_boundaries(messages):
    return [
        index for index, message in enumerate(messages)
        if index > 0 and message["role"] == "assistant"
    ]


def choose_boundary(lengths, minimum_tokens, maximum_tokens, target_tokens):
    candidates = [(abs(tokens - target_tokens), index, tokens) for index, tokens in lengths
                  if minimum_tokens <= tokens <= maximum_tokens]
    if not candidates:
        return None
    distance, index, tokens = min(candidates)
    return {"message_index": index, "prompt_tokens": tokens}


def render_prompt(template, messages):
    return template.render(messages=messages, tools=None, add_generation_prompt=True)


def profile(source, tokenizer_path, tokenizer_config_path, output, minimum_tokens, maximum_tokens,
            target_tokens, limit):
    import pyarrow.parquet as parquet
    from jinja2 import Environment, StrictUndefined
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.truncation is not None or tokenizer.padding is not None:
        raise ValueError("Tokenizer must not truncate or pad")
    template_text = json.loads(tokenizer_config_path.read_text())["chat_template"]
    template = Environment(undefined=StrictUndefined).from_string(template_text)
    rows = parquet.read_table(source).to_pylist()
    records, exports, histograms = [], [], collections.Counter()
    for row_index, row in enumerate(rows):
        messages = normalize_messages(row["messages"])
        boundaries = assistant_boundaries(messages)
        if not boundaries:
            raise ValueError("Agent trajectory has no assistant input boundary")
        last_index = boundaries[-1]
        prompt = render_prompt(template, messages[:last_index])
        maximum = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
        histograms[maximum] += 1
        record = {"row_index": row_index, "instance_id": row["instance_id"],
                  "traj_id": row["traj_id"], "resolved": row["resolved"],
                  "source_model": row["model"], "messages": len(messages),
                  "assistant_boundaries": len(boundaries), "last_assistant_input_tokens": maximum,
                  "roles": dict(collections.Counter(message["role"] for message in messages))}
        if maximum >= minimum_tokens:
            lengths = []
            for index in boundaries:
                text = render_prompt(template, messages[:index])
                count = len(tokenizer.encode(text, add_special_tokens=False).ids)
                lengths.append((index, count))
            selected = choose_boundary(lengths, minimum_tokens, maximum_tokens, target_tokens)
            record["boundary_lengths"] = [{"message_index": index, "prompt_tokens": count}
                                         for index, count in lengths]
            record["selected_boundary"] = selected
            if selected is not None:
                prefix = messages[:selected["message_index"]]
                selected_prompt = render_prompt(template, prefix)
                exports.append({"instance_id": row["instance_id"], "traj_id": row["traj_id"],
                                "row_index": row_index, "resolved": row["resolved"],
                                **selected, "messages": prefix,
                                "prompt_sha256": hashlib.sha256(
                                    selected_prompt.encode()).hexdigest(),
                                "next_assistant_reference_tokens": len(tokenizer.encode(
                                    render_prompt(
                                        template, messages[:selected["message_index"] + 1]),
                                    add_special_tokens=False).ids) - selected["prompt_tokens"]})
        records.append(record)
        if (row_index + 1) % 100 == 0:
            print(
                "profiled", row_index + 1, "/", len(rows),
                "candidate prefixes", len(exports), flush=True,
            )
    exports.sort(key=lambda item: (abs(item["prompt_tokens"] - target_tokens), item["row_index"]))
    selected_exports = exports[:limit]
    output.mkdir(parents=True, exist_ok=True)
    (output / "swesmith-input-records.json").write_text(json.dumps(records, indent=2) + "\n")
    with (output / "swesmith-100k-inputs.jsonl").open("w") as stream:
        for export in selected_exports:
            stream.write(json.dumps(export) + "\n")
    report = {"scope": (
                  "one_SWE_smith_tool_shard_reconstructed_uncompressed_history"
                  "_not_original_API_prompt_or_full_dataset"),
              "source": str(source), "rows": len(rows), "tokenizer": str(tokenizer_path),
              "tokenizer_config": str(tokenizer_config_path),
              "template_sha256": hashlib.sha256(template_text.encode()).hexdigest(),
              "rendering": (
                  "official_Qwen_chat_template_tools_none_text_parts_joined_newline"
                  "_add_generation_prompt_no_truncation"),
              "reference_scope": (
                  "next_assistant_template_delta_including_tool_calls_not_measured_Qwen_output"),
              "message_boundary_scope": (
                  "all_prior_messages_only_excludes_target_assistant_and_future_messages"),
              "length_band": [minimum_tokens, maximum_tokens], "target_tokens": target_tokens,
              "last_assistant_input_tokens": histogram_statistics(histograms),
              "rows_ge100000": sum(
                  count for tokens, count in histograms.items() if tokens >= 100000),
              "rows_ge131072": sum(
                  count for tokens, count in histograms.items() if tokens >= 131072),
              "rows_with_boundary_in_band": len(exports), "exported": len(selected_exports),
              "examples": [{key: value for key, value in item.items() if key != "messages"}
                           for item in selected_exports],
              "limitations": ["one_shard_not_uniform_sample_of_all_trajectories",
                              "no_original_request_timing_or_history_compression_verified",
                              "tool_definitions_not_in_trajectory_extra_schema_would_add_tokens",
                              "resolved_and_unresolved_kept_not_task_success_benchmark",
                              "text_only_replay_not_execution_of_dataset_tools_or_closed_loop_agent",
                              ("no_GPU_run_current_32k_4GiB_pool_and_2112MiB_budget"
                               "_cannot_serve_100k")]}
    (output / "swesmith-100k-profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-tokens", type=int, default=95000)
    parser.add_argument("--maximum-tokens", type=int, default=105000)
    parser.add_argument("--target-tokens", type=int, default=100000)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior agentic profile; choose a new output directory")
    if not 0 < args.minimum_tokens <= args.target_tokens <= args.maximum_tokens or args.limit <= 0:
        parser.error("Require positive limit and ordered token bounds")
    profile(args.source, args.tokenizer, args.tokenizer_config, args.output,
            args.minimum_tokens, args.maximum_tokens, args.target_tokens, args.limit)


if __name__ == "__main__":
    main()
