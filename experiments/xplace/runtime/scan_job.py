import argparse
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time

from causal_deployment import read_manifest, sha256
from causal_job import BLOCKS, remaining_seconds, run_child, save
from engine import SPARSE_CAPTURE_SIZES
from imbalance_calibration import scan_cases


PURPOSE = "capture_aligned_graph_instrumentation_overhead_not_prediction_or_system_gain"
DISABLED_INSTRUMENTATION = ("XPLACE_NESTED_TIMING", "XPLACE_LIVE_OBSERVATION", "XPLACE_TIMING_ALLOWED",
                            "XPLACE_BATCH_LINK", "DP_OPT_CAPTURE_EVIDENCE", "XPLACE_GRAPH_EXECUTION_ALLOWED")
COMMON = dict(phases="dataplane,imbalance", policies="no_move", num_ranks=2, tensor_parallel_size=2,
              enable_tp_migration=True, port=31000, fairness_threshold_s=30.0, cache_bytes=48 * 1024 ** 3, max_model_len=131072,
              budget_margin_gib=3.0, budget_cap_gib=None, seed=20261110, kv_mode="prompt",
              reserve_extra_tokens=256, max_inflight=[0, 0], drain_timeout_s=210.0,
              probe_max_ms=None, trace=None, segments=None, step_params=None, planner_params=None,
              causal_config=None, causal_mode=None)


def protocol_core():
    return dict(schema=1, purpose=PURPOSE, blocks=[list(block) for block in BLOCKS],
                allocation_limit_s=3600, allocation_soft_deadline_s=3300,
                minimum_remaining_to_start_condition_s=1500, restart_engines_each_condition=True,
                common_args=copy.deepcopy(COMMON), cases_per_condition=4, requests_per_condition=768,
                warmup_requests_per_condition=256, total_measured_pairs=8,
                off="No graph hooks installed;all other optional collection flags disabled in both modes",
                collect="Hooks installed at worker initialization;case buffers active only inside scan windows;startup CPU/wall reported separately",
                pair_identity="Same block,case,request IDs,prompts,outputs,physical GPUs,CLI and observed budgets",
                interpretation="Synthetic fixed-batch mechanism pilot,not shared-queue policy or K16 prediction validation",
                raw_retention="Keep all failures,censoring,loss and not-run rows;no successful-subset gate or automatic retry",
                graph_integrity=dict(lanes=4, events_per_lane_case=20000, bytes_per_lane_case=16777216,
                                     max_dropped=0, max_invalid=0, control_timeout_s=10, actual_gpu_completion=False),
                overhead=dict(primary_metrics=["steady_step_s", "mean_request_ttft_s", "mean_request_tpot_s"],
                              ttft="mean(first_token_at-submitted_at) over every planned request",
                              tpot="mean((completed_at-first_token_at)/(output_tokens-1)) over every planned request;frontend completion",
                              pair_loss="collect/off-1", per_metric_median_limit=0.05, per_metric_worst_limit=0.15,
                              coverage_required=1.0, both_orders_required=True, all_eight_pairs_required=True,
                              incomplete_or_preempted="unavailable and gate false,not a zero or dropped denominator",
                              cpu_rss="Same 1s sampled process-group accounting in both modes;not exact CPU or peak RSS"),
                primary_prediction_gate=False, system_gain_gate=False,
                source_verification="Reuse source/config attestation fields named causal_deployment;this does not enable causal collection")


def overhead_config(block, mode):
    if type(block) is not int or block not in (0, 1) or mode not in ("off", "collect"):
        raise ValueError("Unknown fixed scan overhead condition")
    return dict(num_ranks=2, tensor_parallel_size=2, seed=20261110, context_tokens=2048, output_tokens=1024,
                repeats=1, repeat_start=block, case_timeout_s=150, max_scan_s=900,
                series=[dict(batch_size=128, deltas=[0, 64]), dict(batch_size=256, deltas=[0, 128])],
                sampling_design="fixed_balanced_and_extreme_128_256_graph_overhead",
                capture_sizes=list(SPARSE_CAPTURE_SIZES), observe_graph_execution=mode == "collect")


def load_protocol(filename, expected_sha256):
    filename = Path(filename)
    if sha256(filename) != expected_sha256:
        raise ValueError("Scan overhead protocol hash differs")
    protocol = json.loads(filename.read_text())
    if json.dumps({key: value for key, value in protocol.items() if key != "configs"}, sort_keys=True, allow_nan=False) != json.dumps(protocol_core(), sort_keys=True):
        raise ValueError("Scan overhead protocol differs from predeclared design")
    configs = protocol.get("configs")
    if not isinstance(configs, dict) or set(configs) != {f"{block}-{mode}" for block in (0, 1) for mode in ("off", "collect")}:
        raise ValueError("Missing fixed scan overhead configs")
    paths = {}
    for block in (0, 1):
        for mode in ("off", "collect"):
            key = f"{block}-{mode}"
            entry = configs[key]
            name = entry["file"]
            if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
                raise ValueError("Config must be a sibling input file")
            path = filename.parent / name
            if sha256(path) != entry["sha256"] or json.dumps(json.loads(path.read_text()), sort_keys=True, allow_nan=False) != json.dumps(overhead_config(block, mode), sort_keys=True):
                raise ValueError("Scan overhead condition differs")
            cases = scan_cases(overhead_config(block, mode), 2, 2)
            if len(cases) != protocol["cases_per_condition"] or sum(case["batch_size"] for case in cases) != 768:
                raise ValueError("Scan overhead case coverage differs")
            paths[key] = path
    return protocol, paths


def validate_common(common_args, config, prompt_bank_sha256):
    from main import parse_args

    forbidden = {"--output", "--imbalance-config", "--causal-config", "--causal-mode"}
    options = [value.split("=", 1)[0] for value in common_args if value.startswith("--")]
    allowed = {"--" + key.replace("_", "-") for key in COMMON} | {"--checkpoint", "--prompt-bank"}
    if forbidden.intersection(options) or set(options) - allowed or len(options) != len(set(options)):
        raise ValueError("Per-condition options and duplicate CLI overrides are forbidden")
    args = parse_args([*common_args, "--output", "scan-overhead-validation", "--imbalance-config", str(config)])
    for key, expected in COMMON.items():
        value = getattr(args, key)
        if key == "max_inflight":
            value = list(value)
        if value != expected or type(value) is not type(expected):
            raise ValueError(f"Scan overhead CLI differs: {key}={value!r}")
    if sha256(args.prompt_bank) != prompt_bank_sha256:
        raise ValueError("Frozen prompt bank differs")
    return dict(checkpoint=args.checkpoint, prompt_bank=str(args.prompt_bank), common=copy.deepcopy(COMMON))


def run_block(*, block, root, protocol_path, manifest, common_args, environment, runner=run_child):
    if type(block) is not int or block not in (0, 1):
        raise ValueError("Unknown scan overhead block")
    root.mkdir(parents=True, exist_ok=False)
    result = dict(block=block, planned=list(BLOCKS[block]), started=[], completed=[], status="preparing",
                  purpose=PURPOSE, overhead_accepted=False, prediction_gate_passed=False, not_executed=list(BLOCKS[block]))
    save(root / "status.json", result)
    try:
        if manifest.get("purpose") != PURPOSE:
            raise ValueError("Not a scan overhead deployment manifest")
        protocol, configs = load_protocol(protocol_path, manifest["protocol_sha256"])
        result["inputs"] = validate_common(common_args, configs[f"{block}-off"], manifest["prompt_bank_sha256"])
        remaining = remaining_seconds(environment["XPLACE_JOB_STARTED_AT"])
        deadline = time.monotonic() + remaining
        result.update(status="running", allocation_started_at=environment["XPLACE_JOB_STARTED_AT"],
                      deadline_monotonic=deadline, protocol_sha256=manifest["protocol_sha256"])
        save(root / "protocol.json", protocol)
        for mode in BLOCKS[block]:
            if deadline - time.monotonic() < protocol["minimum_remaining_to_start_condition_s"]:
                raise TimeoutError("Insufficient shared allocation time for a complete fixed scan condition")
            load_protocol(protocol_path, manifest["protocol_sha256"])
            validate_common(common_args, configs[f"{block}-{mode}"], manifest["prompt_bank_sha256"])
            transfer = Path(environment["DP_OPT_TRANSFER_ROOT"]) / f"scan-block{block}-{mode}"
            transfer.mkdir(exist_ok=False)
            output = root / mode
            command = [sys.executable, str(Path(__file__).with_name("main.py")), *common_args,
                       "--output", str(output / "out"), "--imbalance-config", str(configs[f"{block}-{mode}"])]
            child_environment = dict(environment, DP_OPT_TRANSFER_ROOT=str(transfer), XPLACE_SCAN_OVERHEAD="1")
            child_environment.update({name: "0" for name in DISABLED_INSTRUMENTATION})
            result["started"].append(mode)
            result["not_executed"] = [value for value in BLOCKS[block] if value not in result["started"]]
            save(root / "status.json", result)
            receipt = runner(command, child_environment, output, deadline)
            result["completed"].append(dict(mode=mode, **receipt))
            save(root / "status.json", result)
            if receipt["status"] != "passed" or receipt["returncode"] != 0:
                raise RuntimeError(f"Scan overhead condition {mode} failed; no automatic retry")
        result["status"] = "collected_not_overhead_accepted"
    except BaseException as error:
        result.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        result["not_executed"] = [mode for mode in BLOCKS[block] if mode not in result["started"]]
        save(root / "status.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", type=int, choices=(0, 1), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, common = parser.parse_known_args()
    manifest, identity = read_manifest()
    environment = dict(os.environ)

    def interrupted(signum, frame):
        raise InterruptedError(f"Scan job interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    print(json.dumps(dict(manifest_sha256=identity, purpose=PURPOSE, block=args.block)), flush=True)
    run_block(block=args.block, root=args.output / "scan-overhead-block", protocol_path=args.protocol,
              manifest=manifest, common_args=common, environment=environment)


if __name__ == "__main__":
    main()
