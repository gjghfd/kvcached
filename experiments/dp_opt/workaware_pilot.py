import asyncio
import hashlib
import json
import math
import random
import time

from calibrated_cost import SPARSE_CAPTURE_SIZES
from live_benchmark import save
from live_benefit import capture_control, stream, wait_empty, wait_reserved
from workaware_evidence import RUNTIME_FILES


OUTPUT_TOKENS = 3072
WINDOW = (1536, 2560)
POPULATIONS = (2, 3, 4, 8, 9)
PROMPT_LENGTHS = (4096, 8192)
SOURCE_FILES = (
    "config.json", "summary.json", "workaware-cases.json", "workaware-prompts.json",
    "workaware-references.json", "workaware-trials.json", "workaware-captures.json",
    "runtime-control.json", "source.sha256", "exit-status",
    *(f"worker-instance-{model + 1}-dp{rank}.json" for model in (0, 1) for rank in (0, 1)),
    *(f"workaware-engine-instance-{model + 1}-dp{rank}.json" for model in (0, 1) for rank in (0, 1)),
    "workaware-runtime-source.json", *(f"workaware-runtime-source/{name}" for name in RUNTIME_FILES),
)


def calibration_cases():
    cases = []
    for model in (0, 1):
        for repeat in (0, 1):
            paired = []
            for population in POPULATIONS:
                counts = sorted({0, 1, population // 2, population - population // 2,
                                 population - 1, population})
                lengths = PROMPT_LENGTHS + ((16384, 22528) if population <= 4 else ())
                paired.extend({"model": model, "population": population, "rank_zero": count,
                               "prompt_tokens": length, "repeat": repeat}
                              for length in lengths for count in counts)
            random.Random(20261026 + model * 100 + repeat).shuffle(paired)
            cases.extend(paired)
    return cases


def prompt_ids(tokenizer, length):
    prefix = tokenizer.encode("Read the complete context and continue the explanation. ", add_special_tokens=False)
    if not prefix:
        raise ValueError("Calibration tokenizer produced an empty prefix")
    return (prefix * math.ceil(length / len(prefix)))[:length]


def stable_sample(requests):
    if not requests or any(not math.isfinite(request[name]) for request in requests for name in ("first_token_at", "completed_at")):
        raise ValueError("Calibration requires finite full-population first-token/completion evidence")
    all_prefill_finished = max(request["first_token_at"] for request in requests)
    first_completion = min(request["completed_at"] for request in requests)
    counts, contexts, cadences = [0, 0], [0., 0.], []
    for request in requests:
        entries = request["cadence"]
        if (any(not math.isfinite(entry["time"]) or type(entry["tokens"]) is not int or entry["tokens"] < 0 for entry in entries)
                or any(later["time"] < earlier["time"] or later["tokens"] < earlier["tokens"]
                       for earlier, later in zip(entries, entries[1:]))):
            raise ValueError("Calibration cadence timestamps/counts are invalid")
        left, right = (next(entry for entry in request["cadence"] if entry["tokens"] >= threshold)
                       for threshold in WINDOW)
        if left["time"] < all_prefill_finished or right["time"] > first_completion:
            raise ValueError("Predeclared calibration window contains population growth or drain")
        if (right["tokens"] <= left["tokens"] or right["time"] <= left["time"]
                or left["tokens"] >= WINDOW[1] or right["tokens"] >= OUTPUT_TOKENS):
            raise ValueError("Calibration client did not expose the predeclared decode window")
        rank = request["rank_initial"]
        if rank not in (0, 1):
            raise ValueError("Calibration has an invalid attention rank")
        counts[rank] += 1
        contexts[rank] += request["prompt_tokens"] + (left["tokens"] + right["tokens"]) / 2 - 1
        cadences.append((right["time"] - left["time"]) / (right["tokens"] - left["tokens"]))
    return {"counts": counts, "context_tokens": contexts, "cadence_s": sum(cadences) / len(cadences),
            "request_cadences_s": cadences, "window_tokens": list(WINDOW)}


async def run_workaware_pilot(engines, controller, args, tokenizer):
    cases = calibration_cases()
    prompts = {str(length): prompt_ids(tokenizer, length) for length in sorted({case["prompt_tokens"] for case in cases})}
    save(args.output / "workaware-cases.json", cases)
    save(args.output / "workaware-prompts.json", prompts)
    references = {}
    for model in (0, 1):
        for length, prompt in prompts.items():
            await wait_empty(controller)
            await controller.release_transfer_cache()
            references[f"{model}:{length}"] = await stream(
                engines[model], f"workaware-reference-{model}-{length}", prompt,
                OUTPUT_TOKENS, 0, model, "serial_calibration_reference", time.monotonic())
    save(args.output / "workaware-references.json", references)
    await capture_control(controller, "start")
    warm_prompt = tokenizer.encode("What is two plus two?", add_special_tokens=False)
    for model in (0, 1):
        for rank in (0, 1):
            await stream(engines[model], f"workaware-warm-{model}-{rank}", warm_prompt,
                         64, rank, model, "graph_warmup", time.monotonic())
    trials = []
    for index, case in enumerate(cases):
        await wait_empty(controller)
        await controller.release_transfer_cache()
        tasks = []
        model = case["model"]
        prompt = prompts[str(case["prompt_tokens"])]
        for position in range(case["population"]):
            rank = int(position >= case["rank_zero"])
            identifier = f"workaware-case-{index}-request-{position}"
            task = asyncio.create_task(stream(engines[model], identifier, prompt, OUTPUT_TOKENS,
                                              rank, model, "workaware_calibration", time.monotonic()))
            tasks.append(task)
            await wait_reserved(controller, task, identifier, model, rank,
                                timeout=args.request_timeout_s, target_only=True)
        observations = []
        for threshold in WINDOW:
            deadline = time.monotonic() + args.request_timeout_s
            while True:
                snapshot = await controller.snapshots()
                active = [request for rank in snapshot[model] for request in rank["requests"]]
                if (len(active) == case["population"]
                        and all(request["movable"] and request["output_tokens"] >= threshold for request in active)):
                    observations.append(snapshot)
                    break
                if any(task.done() for task in tasks) or time.monotonic() >= deadline:
                    raise RuntimeError("Workaware calibration population changed before its stable window")
                await asyncio.sleep(.1)
        requests = list(await asyncio.wait_for(asyncio.gather(*tasks), args.request_timeout_s))
        reference = references[f"{model}:{case['prompt_tokens']}"]["token_ids"]
        if any(request["token_ids"] != reference for request in requests):
            raise RuntimeError("Workaware calibration differs from complete serial greedy tokens")
        trial = {"case": case, "requests": requests, "window_observations": observations,
                 "sample": stable_sample(requests)}
        trials.append(trial)
        await asyncio.to_thread(save, args.output / "workaware-trials.json", trials)
        save(args.output / "progress.json", {"stage": "workaware_cadence_calibration",
                                             "completed_cases": len(trials), "total_cases": len(cases)})
    save(args.output / "workaware-captures.json", await capture_control(controller, "stop"))
    save(args.output / "summary.json", {"status": "workaware_cadence_pilot_passed", "cases": len(cases),
                                        "capture_sizes": SPARSE_CAPTURE_SIZES,
                                        "scope": "one_active_tenant_other_resident_idle"})


def load_verified_fit(path, expected_sha256, source_root):
    from verify_workaware_pilot import verify

    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)):
        raise ValueError("Workaware fit requires an explicitly frozen SHA256")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Workaware fit differs from frozen SHA256")
    fit = json.loads(payload)
    if "population_profiles" in fit:
        from verify_workaware_profile import load_verified_profile

        return load_verified_profile(path, expected_sha256, source_root)
    for name in SOURCE_FILES:
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != fit["source_files_sha256"].get(name):
            raise ValueError("Workaware fit raw sources changed")
    independently_rebuilt = verify(source_root)
    numeric_fields = ("residual_bound_s", "rmse_s")
    numeric_match = (all(math.isclose(fit[name], independently_rebuilt[name], rel_tol=1e-8, abs_tol=1e-12)
                         for name in numeric_fields)
                     and all(math.isclose(value, independently_rebuilt["coefficients"][name], rel_tol=1e-8, abs_tol=1e-12)
                             for name, value in fit["coefficients"].items()))
    exclude = {*numeric_fields, "coefficients"}
    if not numeric_match or {name: value for name, value in fit.items() if name not in exclude} != {
            name: value for name, value in independently_rebuilt.items() if name not in exclude}:
        raise ValueError("Workaware fit does not reproduce from its raw calibration sources")
    return fit
