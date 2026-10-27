import argparse
import hashlib
import json
import math
from pathlib import Path

from calibrated_cost import SPARSE_CAPTURE_SIZES
from verify_benefit import verify_ownership, verify_runtime_control
from verify_trace import capture_entries
from workaware_cost import fit_cadence, validate_fit
from workaware_pilot import OUTPUT_TOKENS, SOURCE_FILES, WINDOW, calibration_cases, stable_sample
from workaware_evidence import verify_runtime


def verify(root, output=None):
    def read(name):
        return json.loads((root / name).read_text())

    cases = calibration_cases()
    verify_runtime(root)
    config, summary = read("config.json"), read("summary.json")
    if (summary != {"status": "workaware_cadence_pilot_passed", "cases": len(cases),
                    "capture_sizes": SPARSE_CAPTURE_SIZES, "scope": "one_active_tenant_other_resident_idle"}
            or config["max_model_len"] != 131072 or config["cache_bytes"] != 24 * 1024**3
            or config["kv_budget_mib"] != 10240 or config["allocator_gpu_utilization"] != .98
            or read("workaware-cases.json") != cases or (root / "exit-status").read_text().strip() != "0"):
        raise ValueError("Workaware calibration changed or did not complete the frozen serving settings/cases")
    workers = [[read(f"worker-instance-{model + 1}-dp{rank}.json") for rank in (0, 1)] for model in (0, 1)]
    control = verify_runtime_control(root, workers)
    if (control is None or control["dp_pause_sync_interval"] != 32
            or len({worker["ipc_name"] for ranks in workers for worker in ranks}) != 4):
        raise ValueError("Workaware calibration requires independent pools and default32 control")
    for model, ranks in enumerate(workers):
        for rank, worker in enumerate(ranks):
            engine = read(f"workaware-engine-instance-{model + 1}-dp{rank}.json")
            if ("A800" not in worker["gpu_name"] or worker["gpu_uuid"] != workers[0][rank]["gpu_uuid"]
                    or ranks[0]["gpu_uuid"] == ranks[1]["gpu_uuid"] or engine["dtype"] != "torch.bfloat16"
                    or worker["checkpoint"] != workers[0][0]["checkpoint"]
                    or worker["max_model_len"] != 131072 or worker["kv_cache_memory_bytes"] != 24 * 1024**3
                    or engine["cudagraph_capture_sizes"] != SPARSE_CAPTURE_SIZES
                    or engine["cudagraph_mode"] != "FULL_DECODE_ONLY" or engine["compile_mode"] != 0
                    or engine["service"] != worker["service"] or engine["dp_rank"] != rank
                    or engine["max_num_batched_tokens"] != 4096 or worker["max_num_batched_tokens"] != 4096
                    or any(worker["groups"][name]["world_size"] != size
                           for name, size in (("dp", 2), ("ep", 2), ("tp", 1)))):
                raise ValueError("Workaware calibration changed BF16/default sparse/DP2/EP2/TP1 topology")
        for left, right in zip(ranks[0]["expert_ids"], ranks[1]["expert_ids"]):
            if set(left) & set(right) or set(left) | set(right) != set(range(128)):
                raise ValueError("Workaware calibration does not have complementary EP weight shards")
    prompts, references, trials = read("workaware-prompts.json"), read("workaware-references.json"), read("workaware-trials.json")
    lengths = sorted({case["prompt_tokens"] for case in cases})
    if (set(prompts) != {str(length) for length in lengths}
            or any(len(prompts[str(length)]) != length or any(type(token) is not int or token < 0
                   for token in prompts[str(length)]) for length in lengths)
            or set(references) != {f"{model}:{length}" for model in (0, 1) for length in lengths}
            or [trial["case"] for trial in trials] != cases):
        raise ValueError("Workaware calibration omitted or changed prompts/references/cases")
    samples, identifiers = [], set()
    for trial, case in zip(trials, cases):
        requests, reference = trial["requests"], references[f"{case['model']}:{case['prompt_tokens']}"]
        if (reference["model"] != case["model"] or reference["prompt_tokens"] != case["prompt_tokens"]
                or len(reference["token_ids"]) != OUTPUT_TOKENS or len(requests) != case["population"]
                or sum(request["rank_initial"] == 0 for request in requests) != case["rank_zero"]
                or len(trial["window_observations"]) != 2):
            raise ValueError("Workaware calibration lost request lengths/placement or stable observations")
        for request in requests:
            if (request["request_id"] in identifiers or request["model"] != case["model"]
                    or request["prompt_tokens"] != case["prompt_tokens"] or request["token_ids"] != reference["token_ids"]):
                raise ValueError("Workaware calibration changed requests or serial output tokens")
            identifiers.add(request["request_id"])
        for threshold, observed in zip(WINDOW, trial["window_observations"]):
            verify_ownership(observed, [10., 10.])
            expected = [[0, 0], [0, 0]]
            expected[case["model"]] = [case["rank_zero"], case["population"] - case["rank_zero"]]
            if [[len(rank["requests"]) for rank in ranks] for ranks in observed] != expected:
                raise ValueError("Workaware calibration population or resident-idle scope changed")
            for rank in observed[case["model"]]:
                if (rank["block_size"] != workers[case["model"]][0]["kv_block_size"]
                        or not rank["uniform_page_geometry"] or rank["page_block_capacity"] <= 0
                        or rank["page_bytes"] != rank["page_block_capacity"] * rank["block_bytes"]):
                    raise ValueError("Workaware calibration has inconsistent real KV page/block geometry")
                for request in rank["requests"]:
                    if (not request["movable"] or request["output_tokens"] < threshold
                            or request["max_tokens"] != OUTPUT_TOKENS
                            or len(request["blocks"]) != math.ceil((case["prompt_tokens"] + OUTPUT_TOKENS) / rank["block_size"])
                            or abs(request["computed_tokens"] - case["prompt_tokens"] - request["output_tokens"]) > 1):
                        raise ValueError("Workaware stable window lacks real decode/context evidence")
        sample = stable_sample(requests)
        if trial["sample"] != sample:
            raise ValueError("Workaware stored cadence/features differ from independently reconstructed samples")
        samples.append(sample)
    captures = read("workaware-captures.json")
    if (len(captures) != 2 or any(len(ranks) != 2 for ranks in captures)
            or any(not any("FULL" in entry["mode"] and entry["steps"] > 0 for entry in capture_entries(capture))
                   for ranks in captures for capture in ranks)):
        raise ValueError("Workaware calibration lacks real sparse graph execution on all four ranks")
    fit = fit_cadence(samples)
    fit.update({"status": "raw_verified_workaware_cadence_fit", "source_verification": "raw_verified",
                "max_population": max(case["population"] for case in cases),
                "max_context_tokens": max(lengths) + OUTPUT_TOKENS - 1,
                "max_attention_tokens": max(max(sample["context_tokens"]) for sample in samples),
                "calibrated_models": [0, 1], "cases": len(cases), "requests": len(identifiers),
                "source_files_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
                "domain_scope": "measured_context_and_population_envelope_not_all_joint_combinations",
                "model_samples": {str(model): sum(case["model"] == model for case in cases) for model in (0, 1)}})
    validate_fit(fit)
    if output is not None:
        output.write_text(json.dumps(fit, indent=2) + "\n")
    return fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(verify(options.root, options.output), indent=2))


if __name__ == "__main__":
    main()
