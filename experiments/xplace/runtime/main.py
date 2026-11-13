import argparse
import asyncio
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

from parallel_topology import minimum_free_bytes, verify_workers

PHASES = ("dataplane", "grid", "imbalance", "interference", "observability", "causal", "ilayout", "fit", "trace")


def verify_topology(output, num_ranks, tensor_parallel_size=1):
    return verify_workers(output, num_ranks, tensor_parallel_size)


async def measure_budget(controller, margin_gib, cap_gib=None):
    from controller import GIB

    requested_at = time.monotonic()
    memory = await controller.memory(release_unused=True)
    memory_received_at = time.monotonic()
    summaries = await controller.summaries()
    summaries_received_at = time.monotonic()
    natural = []
    calculations = []
    tensor_parallel_size = getattr(controller, "tensor_parallel_size", 1)
    for rank in range(controller.num_ranks):
        free = minimum_free_bytes(memory, rank, tensor_parallel_size)
        mapped_by_model = [summaries[model][rank]["mapped_bytes"] for model in (0, 1)]
        if any(type(value) is not int or value < 0 for value in mapped_by_model):
            raise ValueError("Invalid mapped-KV capacity input")
        mapped = sum(mapped_by_model)
        unrounded = (mapped + free) / GIB - margin_gib
        natural.append(math.floor(unrounded * 4) / 4)
        calculations.append(dict(rank=rank, minimum_free_bytes=free, mapped_bytes_by_model=mapped_by_model,
                                 mapped_bytes=mapped, unrounded_budget_gib=unrounded,
                                 rounded_budget_gib=natural[-1]))
    budgets = [min(value, cap_gib) if cap_gib else value for value in natural]
    return {"memory": memory, "natural_budget_gib": natural, "budget_gib": budgets, "margin_gib": margin_gib,
            "capacity_schema": 1, "num_ranks": controller.num_ranks, "tensor_parallel_size": tensor_parallel_size,
            "summary_snapshots": summaries, "rank_calculations": calculations,
            "measurement_times": dict(memory_requested_at=requested_at, memory_received_at=memory_received_at,
                                      summaries_received_at=summaries_received_at)}


async def run_phases(engines, args, bank, bank_sha, mark, verify=True):
    from calibrate import run_conv_grid, run_dataplane, run_interference
    from controller import PlacementController
    from cost_model import default_dp2_params
    from engine import save, stream
    from fit import fit_step_model
    from trace_runner import POLICIES, run_trace

    output = args.output
    phases = [phase for phase in args.phases.split(",") if phase]
    controller = PlacementController(engines, args.num_ranks, [1e6] * args.num_ranks)
    controller.tensor_parallel_size = getattr(args, "tensor_parallel_size", 1)
    await controller.summaries()
    topology = verify_topology(output, args.num_ranks, controller.tensor_parallel_size) if verify else None
    if ("causal" in phases or os.environ.get("XPLACE_SCAN_OVERHEAD") == "1") and verify:
        from causal_deployment import verify_worker_attestations

        save(output / "causal-worker-attestations.json",
             verify_worker_attestations(output, os.environ["XPLACE_CAUSAL_MANIFEST_SHA256"]))
    budget = await measure_budget(controller, args.budget_margin_gib, args.budget_cap_gib)
    if "causal" in phases:
        from causal_calibration import bind_paired_budget

        budget = bind_paired_budget(budget, args.causal_plan)
    save(output / "budget.json", budget)
    if "causal" in phases and budget["paired_budget_status"] != "verified":
        raise ValueError("Natural capacity cannot satisfy the frozen paired budget")
    controller.budgets = list(budget["budget_gib"] if "causal" in phases else budget["natural_budget_gib"])
    for model in (0, 1):
        for rank in range(args.num_ranks):
            await stream(engines[model], f"warm_{model}_{rank}", list(range(1000, 1064)), 16, rank, model,
                         "warmup", time.monotonic())
    summaries = await controller.summaries()
    mark("startup", "passed", topology=topology, natural_budget_gib=budget["natural_budget_gib"],
         trace_budget_gib=budget["budget_gib"],
         idle_mapped_gib=[[rank["mapped_bytes"] / 1024**3 for rank in ranks] for ranks in summaries])
    if getattr(args, "probe_max_ms", None):
        from calibrate import measure_case

        # Node health gate: busy daytime nodes ran small-batch Conv decode 2-4x slower than the calibration node,
        # which overloads the frozen trace; abort early instead of producing an off-regime run.
        probe = {"name": "probe_bal8_c1k", "model": 1, "layout": [8] * args.num_ranks,
                 "contexts": [[1024] * 8 for _ in range(args.num_ranks)], "output_tokens": 600}
        result = await measure_case(engines, controller, probe, seed=97)
        step_ms = 1e3 * result["step_s"] if result.get("step_s") else None
        healthy = step_ms is not None and step_ms <= args.probe_max_ms
        mark("probe", "passed" if healthy else "failed", step_ms=step_ms, threshold_ms=args.probe_max_ms,
             window=result.get("window"))
        if not healthy:
            raise RuntimeError(f"Slow node: probe step {step_ms} ms exceeds {args.probe_max_ms} ms")
    if controller.tensor_parallel_size > 1 and "trace" in phases:
        from measured_cost import load_measured_cost

        step_params = load_measured_cost(args.step_params, args.num_ranks, controller.tensor_parallel_size)
    else:
        step_params = json.loads(args.step_params.read_text()) if args.step_params else default_dp2_params()
    calibration = output / "calibration"
    calibration.mkdir(exist_ok=True)
    grid = interference = None
    for phase in phases:
        mark(phase, "started")
        if phase == "dataplane":
            result = await run_dataplane(engines, controller, calibration, args.num_ranks)
            mark(phase, "passed", migrations=len(result["migrations"]))
        elif phase == "grid":
            grid = await run_conv_grid(engines, controller, calibration, args.num_ranks)
            mark(phase, "passed", measured=sum(case["status"] == "measured" for case in grid))
        elif phase == "imbalance":
            from imbalance_calibration import run_imbalance_scan

            config = json.loads(args.imbalance_config.read_text())
            result = await run_imbalance_scan(engines, controller, calibration / "imbalance", config, args.num_ranks)
            mark(phase, result["status"], measured=result["measured"], planned=result["planned"])
            if result["status"] != "passed":
                raise RuntimeError(f"Incomplete imbalance calibration: {result}")
        elif phase == "interference":
            code_key = sorted(key for key in bank if key.startswith("swe-"))[0]
            interference = await run_interference(engines, controller, calibration, bank[code_key],
                                                  args.num_ranks)
            mark(phase, "passed")
        elif phase == "observability":
            from timing_calibration import run_observability

            code_key = sorted(key for key in bank if key.startswith("swe-"))[0]
            result = await run_observability(engines, controller, calibration / "observability", bank[code_key],
                                             args.timing_deadline)
            mark(phase, result["status"], cases=len(result["completed"]))
        elif phase == "causal":
            from causal_calibration import run_causal_calibration

            result = await asyncio.wait_for(
                run_causal_calibration(engines, controller, calibration / "causal", bank, args.causal_plan,
                                       args.causal_mode, args.causal_live_root, args.timing_deadline),
                timeout=max(0.0, args.timing_deadline - time.monotonic()))
            mark(phase, result["status"], cases=len(result["completed"]), prediction_gate_passed=False)
        elif phase == "ilayout":
            from calibrate import INTERFERENCE_LAYOUTS, run_interference_layouts

            code_key = sorted(key for key in bank if key.startswith("swe-"))[0]
            records = await run_interference_layouts(engines, controller, calibration, bank[code_key],
                                                     [("solo", [0] * args.num_ranks)] + INTERFERENCE_LAYOUTS,
                                                     repeats=2)
            mark(phase, "passed", cases=len(records))
        elif phase == "fit":
            try:
                step_params = fit_step_model(grid or [], interference, step_params)
                status = "passed"
            except Exception as error:  # keep the prior; the trace phase must still run
                status = f"fallback_prior: {error}"
            save(calibration / "step-params.json", step_params)
            mark(phase, status, fit=step_params.get("fit"))
        elif phase == "trace":
            segments = trace_segments(args)
            loaded = []
            for label, trace_path, cap_gib, policies in segments:  # validate every segment before running any
                trace_bytes = trace_path.read_bytes()
                trace = json.loads(trace_bytes)
                if trace.get("prompt_bank_sha256") not in (None, bank_sha):
                    raise ValueError(f"Trace {trace_path.name}: prompt bank checksum differs from the supplied bank")
                unknown = [policy for policy in policies if policy not in POLICIES]
                if unknown:
                    raise ValueError(f"Unknown policy {unknown[0]}")
                loaded.append((label, trace, hashlib.sha256(trace_bytes).hexdigest(), cap_gib, policies))
            for label, trace, trace_sha256, cap_gib, policies in loaded:
                controller.budgets = list(budget["budget_gib"])
                trace_args = argparse.Namespace(
                    planner_params=json.loads(args.planner_params.read_text()) if args.planner_params else {},
                    step_params=step_params, trace_sha256=trace_sha256, seed=args.seed,
                    fairness_threshold_s=args.fairness_threshold_s, blocked_recheck_s=2.0, plan_retry_s=2.0,
                    poll_s=.25, return_period_s=10.0, return_cooldown_s=10.0, monitor_s=5.0,
                    drain_timeout_s=args.drain_timeout_s,
                    kv_utilization=float(os.environ.get("KVCACHED_GPU_UTILIZATION", "0.95")),
                    kv_mode=getattr(args, "kv_mode", "full"),
                    max_inflight=getattr(args, "max_inflight", (0, 0)),
                    reserve_extra_tokens=getattr(args, "reserve_extra_tokens", 0))
                # a labelled segment writes trace/<label>/<policy> and marks <label>/<policy>
                root = output / "trace" if label is None else output / "trace" / label
                name = (lambda policy: policy) if label is None else (lambda policy: f"{label}/{policy}")
                stem = (lambda policy: policy) if label is None else (lambda policy: f"{label}-{policy}")
                for policy in policies:
                    # Without the legacy controller-enforced reservation, KV grows per step, so the budget must be
                    # the physical capacity kvcached sees: trim free memory (ballast) around every such trial.
                    capped = bool(POLICIES[policy].get("native")) or trace_args.kv_mode != "full"
                    if capped:
                        if not cap_gib:
                            raise ValueError("Per-step KV growth needs --budget-cap-gib to size the shared KV capacity")
                        ballast = await controller.apply_ballast(cap_gib, trace_args.kv_utilization)
                        save(output / f"ballast-{stem(policy)}.json", ballast)
                        mark(f"ballast:{name(policy)}", "applied", capacity_gib=ballast["capacity_gib"])
                        # plan against what kvcached can actually map: whole page bundles under the ballast
                        controller.budgets = [ballast["pages"] * ballast["page_bytes"] / 1024**3] * args.num_ranks
                    mark(f"trace:{name(policy)}", "started")
                    try:
                        summary = await run_trace(engines, controller, trace, bank, policy, root / policy, trace_args)
                    finally:
                        if capped:
                            controller.budgets = list(budget["budget_gib"])
                            released = await controller.release_ballast(trace_args.kv_utilization)
                            save(output / f"ballast-{stem(policy)}-released.json", released)
                            mark(f"ballast:{name(policy)}", "released", capacity_gib=released["capacity_gib"])
                    mark(f"trace:{name(policy)}", summary["status"], summary=summary)
            mark(phase, "passed")
    return controller


def trace_segments(args):
    """[(label, trace path, KV cap GiB, policies)] run in order in one allocation.

    Without --segments: one unlabelled segment from --trace, --budget-cap-gib and --policies (output layout
    trace/<policy>). A segments file holds {"segments": [{"label", "trace", "budget_cap_gib", "policies"}]}, trace paths
    relative to the file; each segment writes trace/<label>/<policy>.
    """
    path = getattr(args, "segments", None)
    if path is None:
        return [(None, args.trace, args.budget_cap_gib, args.policies.split(","))]
    segments = [(item["label"], path.parent / item["trace"], item.get("budget_cap_gib", args.budget_cap_gib),
                 list(item["policies"])) for item in json.loads(path.read_text())["segments"]]
    labels = [label for label, *_ in segments]
    if not segments or len(set(labels)) != len(labels) or any(not label or "/" in label for label in labels):
        raise ValueError(f"Segments need unique, non-empty labels without '/': {labels}")
    return segments


async def run(args):
    from vllm.v1.engine.async_llm import AsyncLLM

    from engine import engine_arguments, load_bank, save

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    phases = [phase for phase in args.phases.split(",") if phase]
    if any(phase not in PHASES for phase in phases):
        raise ValueError(f"Unknown phase in {phases}")
    log = {"phases": [], "started_at": time.time(), "args": {key: str(value) for key, value in vars(args).items()}}

    def mark(name, status, **extra):
        log["phases"].append({"phase": name, "status": status, "wall": time.time(), **extra})
        save(output / "phases.json", log)

    os.environ["DP_OPT_EVIDENCE_ROOT"] = str(output)
    os.environ["DP_OPT_CONTROLLED_RESERVATION_RETRY"] = "1"
    os.environ["DP_OPT_CAPTURE_EVIDENCE"] = "0"
    os.environ["XPLACE_GRAPH_EXECUTION_ALLOWED"] = "0"
    if "imbalance" in phases:
        graph_config = json.loads(args.imbalance_config.read_text())
        if graph_config.get("observe_graph_execution") is True:
            os.environ["XPLACE_GRAPH_EXECUTION_ALLOWED"] = "1"
    os.environ["XPLACE_TIMING_ALLOWED"] = "1" if "observability" in phases else "0"
    from timing_probe import nested_timing_enabled
    if nested_timing_enabled() and "observability" not in phases:
        raise ValueError("Nested timing is restricted to the observability pilot")
    if "observability" in phases or "causal" in phases:
        from timing_calibration import allocation_deadline

        args.timing_deadline = allocation_deadline()
    if "causal" in phases:
        from causal_calibration import configure_collection, load_plan

        args.causal_plan = load_plan(args.causal_config, args.prompt_bank)
        args.causal_live_root = Path(os.environ["DP_OPT_TRANSFER_ROOT"]) / "causal-live"
        configure_collection(args.causal_mode, args.causal_live_root)
    bank, bank_sha = load_bank(args.prompt_bank)
    engines = []
    try:
        if os.environ.get("XPLACE_SCAN_OVERHEAD") == "1" and phases != ["dataplane", "imbalance"]:
            raise ValueError("Scan overhead source attestation is restricted to dataplane,imbalance")
        if "causal" in phases or os.environ.get("XPLACE_SCAN_OVERHEAD") == "1":
            from causal_deployment import frontend_attestation

            save(output / "causal-frontend-attestation.json", frontend_attestation())
        if "observability" in phases:
            import vllm
            from timing_probe import verify_interface_sources

            try:
                source_sha256 = verify_interface_sources(Path(vllm.__file__).parent,
                                                         **({"nested": True} if nested_timing_enabled() else {}))
                save(output / "timing-interface-check.json", {"status": "passed", "source_sha256": source_sha256})
            except (OSError, ValueError) as error:
                save(output / "timing-interface-check.json", {"status": "failed", "error": str(error)})
                raise
        mark("startup", "started")
        job = os.environ.get("SLURM_JOB_ID", "local")
        for model in (0, 1):
            os.environ["DP_OPT_SERVICE"] = f"instance-{model + 1}"
            base = f"x{job}_s{model}"
            os.environ["KVCACHED_IPC_NAME"] = base
            os.environ["KVCACHED_DP_IPC_BASE"] = base
            engines.append(AsyncLLM.from_engine_args(engine_arguments(
                args.checkpoint, args.port + 100 * model, args.cache_bytes, args.max_model_len, args.num_ranks,
                tensor_parallel_size=args.tensor_parallel_size,
                enable_tp_migration=args.enable_tp_migration)))
        await run_phases(engines, args, bank, bank_sha, mark)
        mark("job", "passed")
    except BaseException as error:
        save(output / "error.json", {"type": type(error).__name__, "message": str(error),
                                     "traceback": traceback.format_exc()})
        mark("job", "failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for engine in reversed(engines):
            engine.shutdown()
    if "causal" in phases or os.environ.get("XPLACE_SCAN_OVERHEAD") == "1":
        try:
            save(output / "causal-final-attestation.json", frontend_attestation())
        except BaseException as error:
            save(output / "causal-final-attestation-error.json",
                 {"type": type(error).__name__, "message": str(error)})
            mark("job", "failed", error=f"Final source verification: {error}")
            raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="xplace: 4-GPU cross-model KV placement experiments")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-bank", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--phases", default="dataplane,grid,interference,fit,trace")
    parser.add_argument("--policies", default="no_move,ca_full")
    parser.add_argument("--num-ranks", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--enable-tp-migration", action="store_true")
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--cache-bytes", type=int, default=48 * 1024**3)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--budget-margin-gib", type=float, default=3.0)
    parser.add_argument("--budget-cap-gib", type=float)
    parser.add_argument("--step-params", type=Path)
    parser.add_argument("--planner-params", type=Path)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--fairness-threshold-s", type=float, default=30.0)
    parser.add_argument("--drain-timeout-s", type=float, default=5400.0)
    parser.add_argument("--probe-max-ms", type=float, help="abort if the Conv bal8 probe step exceeds this")
    parser.add_argument("--kv-mode", choices=("full", "prompt"), default="full",
                        help="controlled admission: full = prompt+output reservation (legacy, uses the replayed "
                             "output length); prompt = current length only, per-step growth, vLLM retract")
    parser.add_argument("--reserve-extra-tokens", type=int, default=0,
                        help="fixed tokens reserved beyond the prompt under --kv-mode prompt")
    parser.add_argument("--max-inflight", type=int, nargs=2, default=(0, 0), metavar=("MODEL0", "MODEL1"),
                        help="shared per-model submitted-request limits for every policy; 0 means unlimited")
    parser.add_argument("--segments", type=Path,
                        help="JSON list of {label, trace, budget_cap_gib, policies} run in order (replaces --trace)")
    parser.add_argument("--imbalance-config", type=Path, help="Dense fixed-context decode calibration JSON")
    parser.add_argument("--causal-config", type=Path, help="Frozen training-only causal collection plan")
    parser.add_argument("--causal-mode", choices=("off", "producers", "collect"))
    args = parser.parse_args(argv)
    if any(value < 0 for value in args.max_inflight):
        parser.error("--max-inflight limits must be nonnegative")
    if "trace" in args.phases.split(",") and args.trace is None and args.segments is None:
        parser.error("--trace or --segments is required for the trace phase")
    if "imbalance" in args.phases.split(",") and args.imbalance_config is None:
        parser.error("--imbalance-config is required for the imbalance phase")
    phases = args.phases.split(",")
    if "causal" in phases:
        if (args.phases != "dataplane,causal" or args.tensor_parallel_size != 2 or args.num_ranks != 2
                or not args.enable_tp_migration or args.causal_config is None or args.causal_mode is None
                or args.kv_mode != "prompt" or args.budget_cap_gib is not None or args.budget_margin_gib != 3.0):
            parser.error("Causal training requires dataplane,causal, DP2/TP2, prompt mode, frozen paired budget and a plan/mode")
        from causal_calibration import load_plan

        try:
            plan = load_plan(args.causal_config, args.prompt_bank)
            if (args.reserve_extra_tokens != plan["reserve_extra_tokens"]
                    or list(args.max_inflight) != plan["max_inflight"]
                    or args.drain_timeout_s != plan["drain_timeout_s"] or args.seed != 20261110
                    or args.max_model_len != 131072 or args.cache_bytes != 48 * 1024**3):
                raise ValueError("Causal CLI configuration differs from the fixed training protocol")
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
    if "observability" in phases and (args.phases != "dataplane,observability" or args.tensor_parallel_size != 2
                                      or args.num_ranks != 2 or not args.enable_tp_migration):
        parser.error("Observability requires only dataplane,observability with DP2/TP2 migration enabled")
    if args.enable_tp_migration and (args.tensor_parallel_size != 2 or args.num_ranks != 2):
        parser.error("Experimental TP migration requires DP2/TP2/EP4")
    if args.tensor_parallel_size > 1:
        if args.num_ranks != 2:
            parser.error("TP2 requires DP2/EP4 on four GPUs")
        if not args.enable_tp_migration and args.phases != "imbalance":
            parser.error("TP2 migration/trace requires explicit opt-in and a same-job dataplane gate")
        if args.enable_tp_migration:
            if any(phase not in ("dataplane", "imbalance", "grid", "interference", "observability", "causal", "trace") for phase in phases):
                parser.error("Unvalidated TP2 calibration phase")
            if "dataplane" not in phases or ("trace" in phases and phases.index("dataplane") > phases.index("trace")):
                parser.error("TP2 trace requires a preceding same-job dataplane phase")
            if "trace" in phases and (args.kv_mode != "prompt" or args.step_params is None):
                parser.error("TP2 trace requires current-length admission and measured step parameters")
            if "trace" in phases:
                from measured_cost import load_measured_cost

                try:
                    load_measured_cost(args.step_params, args.num_ranks, args.tensor_parallel_size)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    parser.error(f"Invalid measured TP2 cost: {error}")
    if "imbalance" in args.phases.split(","):
        from imbalance_calibration import scan_cases

        try:
            config = json.loads(args.imbalance_config.read_text())
            scan_cases(config, args.num_ranks, args.tensor_parallel_size)
            if config["context_tokens"] + config["output_tokens"] > args.max_model_len:
                raise ValueError("Calibration request exceeds --max-model-len")
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
    return args


def main():
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
