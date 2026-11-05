import argparse
import asyncio
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

PHASES = ("dataplane", "grid", "interference", "ilayout", "fit", "trace")


def verify_topology(output, num_ranks):
    workers = [[json.loads((output / f"worker-instance-{model + 1}-dp{rank}.json").read_text())
                for rank in range(num_ranks)] for model in (0, 1)]
    for model in workers:
        for worker in model:
            sizes = {name: worker["groups"][name]["world_size"] for name in ("tp", "dp", "ep")}
            if sizes != {"tp": 1, "dp": num_ranks, "ep": num_ranks}:
                raise RuntimeError(f"Unexpected parallel topology {sizes}")
        for layer in range(len(model[0]["expert_ids"])):
            shards = [set(worker["expert_ids"][layer]) for worker in model]
            if sum(map(len, shards)) != 128 or set().union(*shards) != set(range(128)):
                raise RuntimeError("Experts are not complementary EP shards")
    if any(workers[0][rank]["gpu_uuid"] != workers[1][rank]["gpu_uuid"] for rank in range(num_ranks)):
        raise RuntimeError("Tenant DP ranks are not colocated on the same GPUs")
    if len({worker["ipc_name"] for model in workers for worker in model}) != 2 * num_ranks:
        raise RuntimeError("Tenant/rank KV namespaces are not independent")
    return {"num_ranks": num_ranks, "gpu_uuids": [worker["gpu_uuid"] for worker in workers[0]],
            "parameter_gib": [[worker["parameter_bytes"] / 1024**3 for worker in model] for model in workers],
            "kv_num_blocks": [[worker["kv_num_blocks"] for worker in model] for model in workers]}


async def measure_budget(controller, margin_gib, cap_gib=None):
    from controller import GIB

    memory = await controller.memory(release_unused=True)
    summaries = await controller.summaries()
    natural = []
    for rank in range(controller.num_ranks):
        free = min(memory[model][rank][0]["free_mib"] for model in (0, 1)) * 1024**2
        mapped = sum(summaries[model][rank]["mapped_bytes"] for model in (0, 1))
        natural.append(math.floor(((mapped + free) / GIB - margin_gib) * 4) / 4)
    budgets = [min(value, cap_gib) if cap_gib else value for value in natural]
    return {"memory": memory, "natural_budget_gib": natural, "budget_gib": budgets, "margin_gib": margin_gib}


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
    await controller.summaries()
    topology = verify_topology(output, args.num_ranks) if verify else None
    budget = await measure_budget(controller, args.budget_margin_gib, args.budget_cap_gib)
    controller.budgets = list(budget["natural_budget_gib"])
    save(output / "budget.json", budget)
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
        elif phase == "interference":
            code_key = sorted(key for key in bank if key.startswith("swe-"))[0]
            interference = await run_interference(engines, controller, calibration, bank[code_key],
                                                  args.num_ranks)
            mark(phase, "passed")
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
    bank, bank_sha = load_bank(args.prompt_bank)
    engines = []
    try:
        mark("startup", "started")
        job = os.environ.get("SLURM_JOB_ID", "local")
        for model in (0, 1):
            os.environ["DP_OPT_SERVICE"] = f"instance-{model + 1}"
            base = f"x{job}_s{model}"
            os.environ["KVCACHED_IPC_NAME"] = base
            os.environ["KVCACHED_DP_IPC_BASE"] = base
            engines.append(AsyncLLM.from_engine_args(engine_arguments(
                args.checkpoint, args.port + 100 * model, args.cache_bytes, args.max_model_len, args.num_ranks)))
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="xplace: 4-GPU cross-model KV placement experiments")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-bank", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--phases", default="dataplane,grid,interference,fit,trace")
    parser.add_argument("--policies", default="no_move,ca_full")
    parser.add_argument("--num-ranks", type=int, default=4)
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
    parser.add_argument("--segments", type=Path,
                        help="JSON list of {label, trace, budget_cap_gib, policies} run in order (replaces --trace)")
    args = parser.parse_args(argv)
    if "trace" in args.phases.split(",") and args.trace is None and args.segments is None:
        parser.error("--trace or --segments is required for the trace phase")
    return args


def main():
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
