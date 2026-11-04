import argparse
import hashlib
import json
import sys
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import live_trace
from analyze_numerics_contrast import candidate_contrast
from batch_numerics_probe import EXPECTED_FAILURES, stream_with_logprobs
from live_controller import PlacementController
from verify_benefit import verify_ownership, verify_runtime_control
from verify_trace import capture_entries


LAYOUTS = ("balanced", "corank0", "corank1", "corank1", "corank0", "balanced")


def validate_trace(trace):
    marks = trace["requests"]
    if (len(marks) != 3 or len({mark["request_id"] for mark in marks}) != 3
            or sorted(mark["model"] for mark in marks) != [0, 1, 1]
            or any(mark["output_tokens"] <= 0 for mark in marks)
            or next(mark for mark in marks if mark["model"] == 0)["prompt_tokens"] < 100000):
        raise ValueError("Static batch diagnostic requires the intact two-background/100K-Code trace")


class StaticBatchController(PlacementController):
    probe_layout = "balanced"

    @asynccontextmanager
    async def direct_admission_epoch(self, model, tokens, budgets):
        if self.probe_layout not in LAYOUTS:
            raise ValueError("Unknown static diagnostic layout")
        async with super().direct_admission_epoch(model, tokens, budgets) as epoch:
            observations, scope, used, candidates = epoch
            if model == 1 and self.probe_layout != "balanced":
                forced = int(self.probe_layout[-1])
                candidates = [rank for rank in candidates if rank == forced]
            yield observations, scope, used, candidates


@contextmanager
def instrument():
    original_stream = live_trace.stream
    original_controller = live_trace.PlacementController
    original_trace = live_trace.run_trace
    original_arguments = live_trace.engine_arguments

    def engine_arguments(*args, **kwargs):
        arguments = original_arguments(*args, **kwargs)
        arguments.worker_cls = "workaware_worker.WorkawareVerificationWorker"
        return arguments

    async def run_trace(engines, controller, args, trace, tokenizer, policy, repeat, root):
        validate_trace(trace)
        if policy != "no_move" or args.repeats != len(LAYOUTS):
            raise ValueError("Static diagnostic is six no-move trials, not a benefit policy")
        controller.probe_layout = LAYOUTS[repeat]
        root.mkdir(parents=True, exist_ok=True)
        live_trace.save(root.parent / "phase.json", {"phase": "static-batch-no-copy-diagnostic",
                                                    "repeat": repeat, "layout": LAYOUTS[repeat],
                                                    "is_benefit": False})
        live_trace.save(root / "static-layout.json", {"layout": LAYOUTS[repeat], "repeat": repeat,
                                                     "copies_allowed": False, "pause_allowed": False})
        return await original_trace(engines, controller, args, trace, tokenizer, policy, repeat, root)

    live_trace.stream = stream_with_logprobs
    live_trace.PlacementController = StaticBatchController
    live_trace.run_trace = run_trace
    live_trace.engine_arguments = engine_arguments
    try:
        yield
    finally:
        live_trace.stream = original_stream
        live_trace.PlacementController = original_controller
        live_trace.run_trace = original_trace
        live_trace.engine_arguments = original_arguments


def analyze(root):
    root = Path(root)
    trace = json.loads((root / "trace.json").read_text())
    validate_trace(trace)
    config = json.loads((root / "config.json").read_text())
    if config["policies"] != ["no_move"] or config["repeats"] != len(LAYOUTS):
        raise ValueError("Require the full six-trial no-copy diagnostic")
    if (config["max_model_len"] != 131072 or config["cache_bytes"] != 25769803776
            or config["kv_budget_mib"] != 10240 or config["allocator_gpu_utilization"] != .98):
        raise ValueError("Static probe changed the frozen length or memory configuration")
    workers = [[json.loads((root / f"worker-instance-{model + 1}-dp{rank}.json").read_text())
                for rank in (0, 1)] for model in (0, 1)]
    if (len({worker["ipc_name"] for ranks in workers for worker in ranks}) != 4
            or any(workers[0][rank]["gpu_uuid"] != workers[1][rank]["gpu_uuid"] for rank in (0, 1))
            or any(worker["max_model_len"] != 131072 or worker["kv_cache_memory_bytes"] != 25769803776
                   or any(worker["groups"][name]["world_size"] != size for name, size in (("tp", 1), ("dp", 2), ("ep", 2)))
                   for ranks in workers for worker in ranks)):
        raise ValueError("Static worker residency, cache or parallel layout changed")
    control = verify_runtime_control(root, workers)
    if control is None or control["dp_pause_sync_interval"] != 32:
        raise ValueError("Static probe must retain the default32-step control")
    marks = {mark["request_id"]: mark for mark in trace["requests"]}
    references = {}
    for record in json.loads((root / "references.json").read_text()):
        identifier = record["trace_request_id"]
        if identifier in references and record["token_ids"] != references[identifier]["token_ids"]:
            raise ValueError("Serial references are not repeatable")
        references[identifier] = record
    if set(references) != set(marks):
        raise ValueError("Require full serial references for all original requests")
    for identifier, reference in references.items():
        if (reference["prompt_tokens"] != marks[identifier]["prompt_tokens"]
                or len(reference["token_ids"]) != marks[identifier]["output_tokens"]
                or len(reference["token_logprobs"]) != len(reference["token_ids"])):
            raise ValueError("Serial instrumentation changed original token lengths")
    trials, indexed = [], []
    for repeat, layout in enumerate(LAYOUTS):
        directory = root / f"r{repeat}-no_move"
        declaration = json.loads((directory / "static-layout.json").read_text())
        if declaration != {"layout": layout, "repeat": repeat, "copies_allowed": False, "pause_allowed": False}:
            raise ValueError("Static placement declaration differs from the predeclared sequence")
        if json.loads((directory / "migrations.json").read_text()):
            raise ValueError("Static batch control must never copy KV")
        records = json.loads((directory / "requests.json").read_text())
        by_id = {record["request_id"]: record for record in records}
        if len(records) != len(marks) or set(by_id) != set(marks):
            raise ValueError("Every static trial must drain the unchanged trace")
        for identifier, record in by_id.items():
            mark = marks[identifier]
            if (record["model"] != mark["model"] or record["prompt_tokens"] != mark["prompt_tokens"]
                    or len(record["token_ids"]) != mark["output_tokens"]
                    or len(record["token_logprobs"]) != mark["output_tokens"]
                    or record["arrival_offset_s"] != mark["arrival_s"]):
                raise ValueError("Static probe changed model, arrivals or complete original lengths")
        ranks = [by_id[identifier]["rank_initial"] for identifier in marks if marks[identifier]["model"] == 1]
        expected = [0, 1] if layout == "balanced" else [int(layout[-1])] * 2
        if sorted(ranks) != expected:
            raise ValueError("Actual background ranks do not match the declared layout")
        decisions = json.loads((directory / "decisions.json").read_text())
        if any(decision.get("moves") or decision.get("pause_call_s", 0) != 0 for decision in decisions):
            raise ValueError("Static controls may neither pause nor migrate")
        for decision in decisions:
            verify_ownership(decision["observations"], [10, 10])
            if decision.get("after_admission") is not None:
                verify_ownership(decision["after_admission"], [10, 10])
        captures = json.loads((directory / "captures.json").read_text())
        if (len(captures) != 2 or any(len(ranks) != 2 for ranks in captures)
                or any(not any("FULL" in entry["mode"] and entry["steps"] > 0
                               for entry in capture_entries(capture)) for ranks in captures for capture in ranks)):
            raise ValueError("Missing real CUDA graph decode on any model/rank")
        trials.append({"repeat": repeat, "layout": layout, "copies": 0, "pause_calls": 0,
                       "serial_comparisons": [candidate_contrast(by_id[identifier], references[identifier])
                                              for identifier in sorted(marks)]})
        indexed.append(by_id)
    error = json.loads((root / "error.json").read_text()) if (root / "error.json").exists() else None
    if error is not None and (error["type"] != "RuntimeError" or error["message"] not in EXPECTED_FAILURES):
        raise ValueError("Do not mask unrelated static diagnostic failures")
    paired = [{"layout": LAYOUTS[left], "repeats": [left, right],
               "comparisons": [candidate_contrast(indexed[right][identifier], indexed[left][identifier])
                               for identifier in sorted(marks)]} for left, right in ((0, 5), (1, 4), (2, 3))]
    return {"status": "completed_static_batch_diagnostic_not_benefit", "trials": trials,
            "same_layout_repeat_comparisons": paired, "underlying_strict_error": error,
            "original_trace_sha256": hashlib.sha256((root / "trace.json").read_bytes()).hexdigest(),
            "no_copy_divergence_reproduced": any(comparison["different_tokens"] for trial in trials
                                                for comparison in trial["serial_comparisons"]),
            "logprobs": 2, "is_algorithm_benefit": False, "numerical_root_cause_proven": False,
            "physical_budget_and_ownership_verified": True, "four_ranks_real_graph_decode_verified": True,
            "migration_correctness_proven": False, "batch_invariant_enabled": False,
            "strict_token_check_relaxed": False}


def main():
    if "--verify" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--verify", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        result = analyze(args.verify)
        with args.output.open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
        return
    root = Path(sys.argv[sys.argv.index("--output") + 1])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "static-batch-probe-source.py").open("xb") as output:
        output.write(Path(__file__).read_bytes())
    with instrument():
        try:
            live_trace.main()
        except RuntimeError as error:
            if str(error) not in EXPECTED_FAILURES:
                raise
    result = analyze(root)
    with (root / "static-batch-diagnostic.json").open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"diagnostic_completed": True, "benefit_validated": False,
                      "no_copy_divergence_reproduced": result["no_copy_divergence_reproduced"]}), flush=True)


if __name__ == "__main__":
    main()
