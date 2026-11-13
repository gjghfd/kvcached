import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from causal_collection import CausalCollection, positive_number
from engine import save
from live_observation import host_clock_id


MODES = ("off", "producers", "collect")


def calibration_plan(bank_path):
    payload = Path(bank_path).read_bytes()
    bank = json.loads(payload)["prompts"]
    code = sorted(key for key in bank if key.startswith("swe-"))
    conv = sorted(key for key in bank if key.startswith("public-"))
    if len(code) != 10 or len(conv) != 256:
        raise ValueError("Calibration requires the unchanged SWE10/OpenThoughts bank256")
    for key in code + conv:
        item = bank[key]
        if item["prompt_tokens"] != len(item["token_ids"]) or item["prompt_tokens"] < 1:
            raise ValueError("Prompt length differs from bank tokens")
        if key in code and item["prompt_tokens"] < 100000:
            raise ValueError("SWE calibration prompts must retain at least 100K tokens")
        output = 64 if key in code else item["original_output_tokens"]
        if type(output) is not int or output < 1 or item["prompt_tokens"] + output > 131072:
            raise ValueError("Invalid unchanged output length or model context overflow")
    cases = []
    for index, count in enumerate((4, 16, 4, 16)):
        keys = conv[index * 16:index * 16 + count]
        requests = [dict(request_id=f"conv-{offset}", model=1, arrival_s=0.0, prompt_key=key,
                         prompt_tokens=bank[key]["prompt_tokens"], output_tokens=bank[key]["original_output_tokens"])
                    for offset, key in enumerate(keys)]
        key = code[index]
        requests.append(dict(request_id="code-0", model=0, arrival_s=30.0, prompt_key=key,
                             prompt_tokens=bank[key]["prompt_tokens"], output_tokens=64))
        cases.append(dict(case=f"train-c{count}-r{index // 2}", split="training", duration_s=30.0,
                          requests=requests))
    return dict(schema=2, purpose="causal_training_collection_not_prediction_acceptance",
                prompt_bank_sha256=hashlib.sha256(payload).hexdigest(), cases=cases,
                modes=list(MODES), topology=dict(models=2, dp=2, tp=2, ep=4, gpus=4),
                policy="no_move", kv_mode="prompt", reserve_extra_tokens=256, budget_margin_gib=3.0,
                paired_budget_gib=[38.0, 39.75], capacity_record_schema=1,
                paired_capacity_basis="r6_r7_raw_free_lower_bound_minus_512MiB",
                max_inflight=[1, 16], case_window_s=240.0, drain_timeout_s=210.0,
                collection=dict(period_s=0.1, deadline_s=0.08, max_events=1000000,
                                max_bytes=512 * 1024 ** 2, max_requests=64),
                reference=dict(k=16, target="first_frontend_receipt_reaching_asof_count_plus_K",
                               cohort="all_registered_unfinished_requests_in_both_models_at_asof",
                               queued_included=True, peer_continues_after_milestone=True,
                               output_length_is_feature=False, model_production_timestamp=None),
                holdout=dict(executable=False, collect_only_after_model_and_protocol_hash_freeze=True,
                             code_keys=code[4:8], conv_keys=conv[128:192], counts=[4, 16, 4, 16],
                             old_runs_are_training_only=["r1", "r2", "r3", "r4"],
                             relative_rmse_limit=0.15, max_relative_error_limit=0.30),
                overhead=dict(full_process_restart_required=True, paired_modes=["off", "collect"],
                              diagnostic_mode="producers", order_blocks=[["off", "collect"], ["collect", "off"]],
                              same_allocation_required=True, source_and_workload_must_match=True,
                              acceptance_pending=True),
                coverage=dict(keep_all_planned_cases=True, retain_all_missing_queries=True,
                              eos_or_short_tail="unattainable_not_filtered", failures="retain_and_abort_matrix",
                              cutoff="trace_cutoff_event_before_cleanup", complete_event_history=False,
                              prediction_acceptance_pending=True))


def load_plan(plan_path, bank_path):
    plan = json.loads(Path(plan_path).read_text())
    if plan != calibration_plan(bank_path):
        raise ValueError("Causal calibration protocol differs from the frozen training plan")
    return plan


def bind_paired_budget(budget, plan):
    natural = budget.get("natural_budget_gib")
    required = plan.get("paired_budget_gib")
    if (any(not isinstance(values, list) or len(values) != 2
            or any(not positive_number(value) for value in values) for values in (natural, required))
            or budget.get("margin_gib") != plan["budget_margin_gib"]):
        raise ValueError("Invalid paired budget capacity evidence")
    available = all(actual >= target for actual, target in zip(natural, required))
    return dict(budget, natural_budget_gib=list(natural), required_budget_gib=list(required),
                budget_gib=list(required) if available else None,
                paired_budget_status="verified" if available else "unavailable")


def configure_collection(mode, root):
    if mode not in MODES:
        raise ValueError("Unknown calibration instrumentation mode")
    if os.environ.get("XPLACE_NESTED_TIMING") == "1" or os.environ.get("XPLACE_TIMING_ALLOWED") == "1":
        raise ValueError("Causal calibration must not run stop-export timing")
    enabled = mode != "off"
    os.environ["XPLACE_LIVE_OBSERVATION"] = "1" if enabled else "0"
    os.environ["XPLACE_BATCH_LINK"] = "1" if enabled else "0"
    if enabled:
        root = Path(root)
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        os.environ["XPLACE_LIVE_ROOT"] = str(root)


def process_usage():
    current = resource.getrusage(resource.RUSAGE_SELF)
    return dict(monotonic=time.monotonic(), process_cpu_s=time.process_time(),
                user_cpu_s=current.ru_utime, system_cpu_s=current.ru_stime,
                maxrss_native_units=current.ru_maxrss, maxrss_units="KiB_on_Linux_bytes_on_macOS")


async def run_causal_calibration(engines, controller, root, prompts, plan, mode, live_root, deadline):
    from cost_model import default_dp2_params
    from trace_runner import run_trace

    if mode not in MODES or controller.num_ranks != 2 or controller.tensor_parallel_size != 2:
        raise ValueError("Causal calibration requires two DP2/TP2 tenants")
    if any(case["split"] != "training" for case in plan["cases"]):
        raise ValueError("Independent validation has no frozen predictor and cannot execute")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    save(root / "plan.json", plan)
    save(root / "mode.json", dict(mode=mode, device_activity="unknown", prediction_gate_passed=False,
                                  clock_id=host_clock_id(), usage=process_usage()))
    completed = []
    try:
        for case in plan["cases"]:
            if deadline - time.monotonic() < plan["case_window_s"] + 330:
                raise TimeoutError("Insufficient allocation time for the next case and cleanup")
            case_root = root / case["case"]
            case_root.mkdir()
            save(case_root / "case.json", case)
            observer = None
            before = process_usage()
            try:
                if mode == "collect":
                    observer = CausalCollection(case_id=case["case"], clock_id=host_clock_id(), **plan["collection"])
                    await observer.start(controller, live_root)
                arguments = argparse.Namespace(
                    planner_params={}, step_params=default_dp2_params(),
                    trace_sha256=hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest(),
                    seed=20261110, fairness_threshold_s=30.0, blocked_recheck_s=2.0, plan_retry_s=2.0,
                    poll_s=0.25, return_period_s=10.0, return_cooldown_s=10.0, monitor_s=5.0,
                    drain_timeout_s=plan["drain_timeout_s"], kv_mode=plan["kv_mode"],
                    reserve_extra_tokens=plan["reserve_extra_tokens"], max_inflight=plan["max_inflight"])
                result = await run_trace(engines, controller, case, prompts, "no_move", case_root / "trace",
                                         arguments, receipt_observer=observer)
                if result["status"] not in ("passed", "censored"):
                    raise RuntimeError(f"Calibration workload failed: {result}")
                completed.append(dict(case=case["case"], status=result["status"]))
            except BaseException as error:
                save(case_root / "error.json", dict(type=type(error).__name__, message=str(error)))
                raise
            finally:
                if observer is not None:
                    report = await observer.stop()
                    await asyncio.to_thread(save, case_root / "collection.json", report)
                save(case_root / "usage.json", dict(before=before, after=process_usage(),
                                                    includes_report_serialization=True))
            save(root / "progress.json", dict(completed=completed, planned=len(plan["cases"])))
    except BaseException as error:
        save(root / "status.json", dict(status="failed", type=type(error).__name__, error=str(error),
                                        completed=completed, planned=[case["case"] for case in plan["cases"]],
                                        prediction_gate_passed=False))
        raise
    result = dict(status="collected", completed=completed, prediction_gate_passed=False,
                  instrumentation_overhead_accepted=False, coverage_accepted=False)
    save(root / "status.json", result)
    return result
