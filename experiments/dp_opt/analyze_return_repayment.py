import argparse
import hashlib
import json
from pathlib import Path

from analyze_return_recovery import committed_returns, finite_time
from analyze_return_token_windows import cadence_map, code_absent_end
from archive_trace_job import verify_raw
from verify_trace import verify


def paired_repayment_span(before, after, origins, event, release_bounds, end_bounds):
    for field in ("request_id", "model", "prompt_tokens", "token_ids"):
        if before[field] != after[field]:
            raise ValueError("Repayment requires identical paired requests and output tokens")
    if before["model"] != 1 or any(len(values) != 2 for values in (origins, release_bounds, end_bounds)):
        raise ValueError("Require a Conv request and both workload clocks")
    for value in (*origins, *release_bounds, *end_bounds, event["start_s"], event["resume_s"]):
        finite_time(value)
    if event["resume_s"] < event["start_s"] or any(end <= start for start, end in zip(release_bounds, end_bounds)):
        raise ValueError("Require an ordered return and positive Code-absent windows")
    cadences = [cadence_map(record, origin) for record, origin in zip((before, after), origins)]
    anchors = [ordinal for ordinal in cadences[1] if ordinal in cadences[0]
               and release_bounds[0] <= cadences[0][ordinal] < end_bounds[0]
               and release_bounds[1] <= cadences[1][ordinal] <= event["start_s"]
               and cadences[1][ordinal] < end_bounds[1]]
    if not anchors:
        return {"exclusion": "no_common_pre_return_ordinal_after_both_Code_releases"}
    anchor = max(anchors)
    prefixes = []
    for ordinal in sorted(set(cadences[0]) & set(cadences[1])):
        if (ordinal <= anchor or cadences[1][ordinal] < event["resume_s"]
                or any(cadence[ordinal] >= end for cadence, end in zip(cadences, end_bounds))):
            continue
        durations = [cadence[ordinal] - cadence[anchor] for cadence in cadences]
        prefixes.append({"token_ordinal": ordinal, "baseline_duration_s": durations[0],
                         "treatment_duration_s": durations[1], "net_saved_s": durations[0] - durations[1]})
    if not prefixes:
        return {"anchor_token_ordinal": anchor, "exclusion": "no_common_post_resume_ordinal_in_both_windows"}
    first_repaid = next((sample for sample in prefixes if sample["net_saved_s"] >= 0), None)
    last_negative = max((index for index, sample in enumerate(prefixes) if sample["net_saved_s"] < 0), default=-1)
    sustained = prefixes[last_negative + 1] if last_negative + 1 < len(prefixes) else None
    final = prefixes[-1]
    return {"anchor_token_ordinal": anchor, "anchor_relative_times_s": [cadence[anchor] for cadence in cadences],
            "last_token_ordinal": final["token_ordinal"], "matched_token_intervals": final["token_ordinal"] - anchor,
            "matched_prefix_samples": len(prefixes), "baseline_duration_s": final["baseline_duration_s"],
            "treatment_duration_s": final["treatment_duration_s"], "net_saved_s": final["net_saved_s"],
            "maximum_observed_debt_s": max(0., -min(sample["net_saved_s"] for sample in prefixes)),
            "first_repaid_prefix": first_repaid, "sustained_repaid_prefix": sustained,
            "return_control_elapsed_s": event["resume_s"] - event["start_s"],
            "scope": "same_ordinal_Code_absent_span_includes_control_and_pause_not_causal_or_whole_request_gain"}


def analyze(root, baseline="random_no_return", treatment="kv_only_workaware"):
    root = Path(root)
    source_hashes, payloads = {}, {}

    def read(relative):
        data = (root / relative).read_bytes()
        source_hashes[relative] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    marker = root / "exit-status"
    if not marker.is_file() or marker.read_text().strip() != "0":
        raise ValueError("Wait for a closed successful independently verified phase")
    source_hashes["exit-status"] = hashlib.sha256(marker.read_bytes()).hexdigest()
    trace = read("trace.json")
    expected = {mark["request_id"] for mark in trace["requests"]}
    for policy in (baseline, treatment):
        payloads[policy] = {name: read(f"r0-{policy}/{name}.json") for name in ("requests", "workload", "decisions")}
        requests = payloads[policy]["requests"]
        if len(requests) != len(expected) or {record["request_id"] for record in requests} != expected:
            raise ValueError("Repayment requires all offered paired requests, not just completed subsets")
    before, after = payloads[baseline], payloads[treatment]
    origins = [finite_time(payload["workload"]["origin"]) for payload in (before, after)]
    records = [{record["request_id"]: record for record in payload["requests"]} for payload in (before, after)]
    returns = committed_returns(after["decisions"], origins[1])
    windows, windows_by_bounds = [], {}
    for code in after["requests"]:
        if code["model"] != 0:
            continue
        paired_code = records[0][code["request_id"]]
        releases = [record["completed_at"] - origin for record, origin in zip((paired_code, code), origins)]
        ends = [code_absent_end(payload["requests"], record, origin, trace["duration_s"])
                for payload, record, origin in zip((before, after), (paired_code, code), origins)]
        bounds = (tuple(releases), tuple(ends))
        if bounds in windows_by_bounds:
            windows_by_bounds[bounds]["Code_request_ids"].append(code["request_id"])
            continue
        window = {"Code_request_id": code["request_id"], "Code_request_ids": [code["request_id"]],
                  "release_relative_times_s": releases,
                  "end_relative_times_s": ends, "requests": []}
        windows.append(window)
        windows_by_bounds[bounds] = window
        if any(end is None or end <= release for end, release in zip(ends, releases)):
            window["exclusion"] = "overlapping_Code_or_no_positive_Code_absent_window"
            continue
        events = [event for event in returns if releases[1] <= event["start_s"] < ends[1]]
        if not events:
            window["exclusion"] = "no_committed_return"
            continue
        window["return"] = events[0]
        window["additional_returns_in_window"] = len(events) - 1
        for identifier, record in sorted(records[1].items()):
            if record["model"] == 1:
                span = paired_repayment_span(records[0][identifier], record, origins, events[0], releases, ends)
                window["requests"].append({"request_id": identifier, **span})
    included = [span for window in windows for span in window["requests"] if "exclusion" not in span]
    excluded = [span for window in windows for span in window["requests"] if "exclusion" in span]
    return {"status": "pause_inclusive_paired_repayment_diagnostic_not_causal_or_final_trace_benefit",
            "baseline": baseline, "treatment": treatment, "source_files_sha256": source_hashes,
            "windows": windows, "included_spans": len(included), "excluded_spans": len(excluded),
            "observed_span_saved_request_s": sum(span["net_saved_s"] for span in included),
            "limitations": ["Requires independently SHA-verified complete raw and ownership before analysis.",
                            "Same-token work after both Code releases is aligned before return; pause is not removed.",
                            "Other populations and prior progress can differ; repayment is not a causal estimate.",
                            "Multiple returns and subsequent control in a window are included, not free work.",
                            "Coincident paired release windows are merged, with every associated Code ID retained.",
                            "Sustained repayment refers only to observed matched cadence prefixes, not interpolation.",
                            "Excluded requests are reported, not assigned zero cost or positive gains.",
                            "Full-request Conv TPOT and separate Conv/Code/combined TTFT remain primary."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior diagnostic output")
    if args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("Do not write diagnostics into immutable archived raw")
    verify_raw(args.root, args.manifest.read_text())
    verification = verify(args.root, args.prompt_bank)
    result = analyze(args.root)
    result["independent_verification"] = verification
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "included_spans": result["included_spans"],
                      "excluded_spans": result["excluded_spans"], "status": result["status"]}))


if __name__ == "__main__":
    main()
