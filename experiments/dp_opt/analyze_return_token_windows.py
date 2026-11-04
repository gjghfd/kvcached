import argparse
import hashlib
import json
import math
from pathlib import Path

from analyze_return_recovery import committed_returns


def cadence_map(record, origin):
    result = {}
    previous_tokens, previous_time = 0, -math.inf
    for sample in record["cadence"]:
        tokens, elapsed = sample["tokens"], sample["time"] - origin
        if (type(tokens) is not int or not previous_tokens < tokens <= len(record["token_ids"])
                or not math.isfinite(elapsed) or elapsed < previous_time):
            raise ValueError("Require strictly increasing token ordinals and finite monotone cadence times")
        result[tokens] = elapsed
        previous_tokens, previous_time = tokens, elapsed
    if not result or previous_tokens != len(record["token_ids"]):
        raise ValueError("Require a complete final cadence token count")
    return result


def paired_token_windows(before, after, before_origin, after_origin, start_s, end_s,
                         window_tokens=128, settle_tokens=32):
    if (not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s
            or type(window_tokens) is not int or window_tokens <= 0
            or type(settle_tokens) is not int or settle_tokens < 0):
        raise ValueError("Require a finite common window and positive token blocks")
    if (before["request_id"] != after["request_id"] or before["prompt_tokens"] != after["prompt_tokens"]
            or before["token_ids"] != after["token_ids"]):
        raise ValueError("Matched decode windows require identical request, prompt length and output tokens")
    cadences = [cadence_map(before, before_origin), cadence_map(after, after_origin)]
    starts = [min((tokens for tokens, elapsed in cadence.items() if elapsed >= start_s), default=None)
              for cadence in cadences]
    if any(value is None for value in starts):
        return {"blocks": [], "missing_ordinal_blocks": 0, "outside_common_window_blocks": 0,
                "exclusion": "one_policy_has_no_tokens_after_common_start"}
    first = max(starts) + settle_tokens
    result = {"first_token_ordinal": first, "blocks": [], "missing_ordinal_blocks": 0,
              "outside_common_window_blocks": 0}
    for left in range(first, len(before["token_ids"]) - window_tokens + 1, window_tokens):
        right = left + window_tokens
        if any(left not in cadence or right not in cadence for cadence in cadences):
            result["missing_ordinal_blocks"] += 1
            continue
        intervals = [[cadence[left], cadence[right]] for cadence in cadences]
        if any(interval[0] < start_s or interval[1] >= end_s for interval in intervals):
            result["outside_common_window_blocks"] += 1
            continue
        result["blocks"].append({"token_start": left, "token_end": right,
            "baseline_elapsed_interval_s": intervals[0], "treatment_elapsed_interval_s": intervals[1],
            "baseline_TPOT_ms": 1000 * (intervals[0][1] - intervals[0][0]) / window_tokens,
            "treatment_TPOT_ms": 1000 * (intervals[1][1] - intervals[1][0]) / window_tokens})
    if not result["blocks"]:
        result["exclusion"] = "no_full_common_ordinal_block_after_settlement"
    return result


def summarize_blocks(blocks):
    if not blocks:
        return {"blocks": 0, "matched_token_intervals": 0,
                "baseline_TPOT_ms": None, "treatment_TPOT_ms": None, "reduction_percent": None}
    steps = sum(block["token_end"] - block["token_start"] for block in blocks)
    means = {}
    for policy in ("baseline", "treatment"):
        means[policy] = sum((block["token_end"] - block["token_start"]) * block[policy + "_TPOT_ms"]
                            for block in blocks) / steps
    return {"blocks": len(blocks), "matched_token_intervals": steps,
            "baseline_TPOT_ms": means["baseline"], "treatment_TPOT_ms": means["treatment"],
            "reduction_percent": (100 * (means["baseline"] - means["treatment"]) / means["baseline"]
                                  if means["baseline"] > 0 else None)}


def code_absent_end(records, code, origin, duration):
    completed = code["completed_at"]
    if any(record["model"] == 0 and record["request_id"] != code["request_id"]
           and record["arrived_at"] <= completed < record["completed_at"] for record in records):
        return None
    return min([duration, *(record["arrived_at"] - origin for record in records
                           if record["model"] == 0 and record["arrived_at"] > completed)])


def analyze(root, baseline="random_no_return", treatment="kv_only_workaware",
            window_tokens=128, settle_tokens=32):
    root = Path(root)
    if (type(window_tokens) is not int or window_tokens <= 0
            or type(settle_tokens) is not int or settle_tokens < 0):
        raise ValueError("Require positive token blocks and nonnegative settlement")
    marker = root / "exit-status"
    if not marker.is_file() or marker.read_text().strip() != "0":
        raise ValueError("Wait for a closed successful independently verified phase")
    source_hashes, payloads = {}, {}
    source_hashes["exit-status"] = hashlib.sha256(marker.read_bytes()).hexdigest()
    trace_bytes = (root / "trace.json").read_bytes()
    trace = json.loads(trace_bytes)
    source_hashes["trace.json"] = hashlib.sha256(trace_bytes).hexdigest()
    for policy in (baseline, treatment):
        payloads[policy] = {}
        for name in ("requests", "workload", "decisions"):
            relative = f"r0-{policy}/{name}.json"
            content = (root / relative).read_bytes()
            source_hashes[relative] = hashlib.sha256(content).hexdigest()
            payloads[policy][name] = json.loads(content)
        requests = payloads[policy]["requests"]
        expected = {mark["request_id"] for mark in trace["requests"]}
        if len(requests) != len(expected) or {request["request_id"] for request in requests} != expected:
            raise ValueError("Wait for a complete independently verified paired phase")
    before, after = payloads[baseline], payloads[treatment]
    origins = [payload["workload"]["origin"] for payload in (before, after)]
    records = [{record["request_id"]: record for record in payload["requests"]} for payload in (before, after)]
    returns = committed_returns(after["decisions"], origins[1])
    windows = []
    for code in sorted((record for record in after["requests"] if record["model"] == 0),
                       key=lambda record: record["completed_at"]):
        counterpart = records[0][code["request_id"]]
        ends = [code_absent_end(payload["requests"], request, origin, trace["duration_s"])
                for payload, request, origin in zip((before, after), (counterpart, code), origins)]
        if any(end is None for end in ends):
            windows.append({"Code_request_id": code["request_id"], "exclusion": "overlapping_Code_at_completion"})
            continue
        release = code["completed_at"] - origins[1]
        events = [event for event in returns if release <= event["start_s"] < ends[1]]
        if not events:
            windows.append({"Code_request_id": code["request_id"], "exclusion": "no_committed_return"})
            continue
        common_start = max(events[0]["resume_s"], counterpart["completed_at"] - origins[0],
                           trace.get("evaluation_start_s", 0))
        common_end = min(ends)
        window = {"Code_request_id": code["request_id"], "return": events[0],
                  "common_elapsed_window_s": [common_start, common_end], "requests": [], "excluded_requests": []}
        if common_end <= common_start:
            window["exclusion"] = "no_common_Code_absent_window"
            windows.append(window)
            continue
        for identifier, request in sorted(records[1].items()):
            if request["model"] != 1 or not request["first_token_at"] - origins[1] < common_start < request["completed_at"] - origins[1]:
                continue
            baseline_request = records[0][identifier]
            if not baseline_request["first_token_at"] - origins[0] < common_start < baseline_request["completed_at"] - origins[0]:
                window["excluded_requests"].append({"request_id": identifier, "reason": "baseline_not_decoding_at_common_start"})
                continue
            comparison = paired_token_windows(baseline_request, request, *origins, common_start, common_end,
                                              window_tokens, settle_tokens)
            window["requests"].append({"request_id": identifier, **comparison,
                                       "summary": summarize_blocks(comparison["blocks"])})
        window["summary"] = summarize_blocks([block for request in window["requests"] for block in request["blocks"]])
        windows.append(window)
    all_blocks = [block for window in windows for request in window.get("requests", []) for block in request["blocks"]]
    return {"status": "paired_post_return_same_ordinal_decode_diagnostic_not_causal_or_net_benefit",
            "source_sha256": source_hashes, "baseline": baseline, "treatment": treatment,
            "analysis_source_sha256": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()
                for source in (Path(__file__), Path(committed_returns.__code__.co_filename))},
            "window_tokens": window_tokens, "settle_tokens_each_request": settle_tokens,
            "Code_windows": windows, "summary": summarize_blocks(all_blocks),
            "caveats": ["Requires complete SHA-archived and independently verified raw; this is not an ownership verifier.",
                "Only matched Conv already decoding under both policies in a common Code-absent window are included.",
                "Token ordinals and output IDs match, but other requests' populations can still differ.",
                "The first return's pause and settlement are excluded from this steady-decode diagnostic, not primary latency.",
                "Whole-request Conv TPOT, per-tenant TTFT and all control costs remain the primary paired results.",
                "Excluded, already-finished and unmatched requests must not be treated as zero-cost improvements."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--window-tokens", type=int, default=128)
    parser.add_argument("--settle-tokens", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, window_tokens=args.window_tokens, settle_tokens=args.settle_tokens), indent=2))


if __name__ == "__main__":
    main()
