"""Fit a fail-closed TP2 cost bundle from two complete, archived GPU jobs."""
import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from calibrate import steady_step
from cost_model import StepModel, capture_bucket
from measured_cost import (fit_residuals, interference_key, interference_residuals, interference_samples,
                           load_measured_cost, validate_calibration_sources)


def decode_features(row):
    middle = row["output_tokens"] / 2
    contexts = ([sum(values) + len(values) * middle for values in row["contexts"]] if "contexts" in row else
                [count * (row["context_tokens"] + middle) for count in row["layout"]])
    return capture_bucket(max(row["layout"])), contexts


def fit_decode(rows):
    import numpy as np
    import scipy
    from scipy.optimize import nnls

    if len(rows) < 6:
        raise ValueError("At least six decode observations are required")
    prompt_lengths = {value for row in rows for values in row.get("contexts", [[row.get("context_tokens")]])
                      for value in values}
    if len(prompt_lengths) < 2:
        raise ValueError("A fixed-context imbalance scan cannot replace context calibration")
    features = [decode_features(row) for row in rows]
    buckets = sorted({bucket for bucket, _ in features})
    if buckets[0] != 1:
        raise ValueError("The baseline graph bucket must be measured, not extrapolated")
    design = np.array([[1.0] + [float(bucket >= size) for size in buckets[1:]] + [max(contexts)]
                       for bucket, contexts in features])
    target = np.array([row["step_s"] for row in rows])
    scale = np.maximum(np.abs(design).max(axis=0), 1e-12)
    scaled = design / scale
    rank = int(np.linalg.matrix_rank(scaled))
    if rank != design.shape[1]:
        raise ValueError("Decode model is not identifiable; independent context/bucket measurements are required")
    coefficients, _ = nnls(scaled / target[:, None], np.ones(len(target)))
    coefficients = coefficients / scale
    graph, cumulative = {1: 0.0}, 0.0
    for bucket, increment in zip(buckets[1:], coefficients[1:-1]):
        cumulative += float(increment)
        graph[bucket] = cumulative
    tenant = {"base_s": float(coefficients[0]), "graph_s": graph,
              "attention_s_per_token": float(coefficients[-1]), "graph_slope_s": 0.0}
    if len(buckets) > 1:
        tenant["graph_slope_s"] = (graph[buckets[-1]] - graph[buckets[-2]]) / (buckets[-1] - buckets[-2])
    if tenant["base_s"] <= 0:
        raise ValueError("Nonpositive fitted base cost")
    params = {"source": "measured_dp2_tp2_ep4_joint_fit", "tenants": {"0": copy.deepcopy(tenant), "1": tenant}}
    params["fit"] = fit_residuals(copy.deepcopy(params), rows)
    contexts = [value for row in rows for values in row.get("contexts", [[row.get("context_tokens")]])
                for value in values]
    details = {"method": "relative_least_squares_nonnegative_graph_increments",
               "numpy_version": np.__version__, "scipy_version": scipy.__version__,
               "design_rank": rank, "design_columns": design.shape[1],
               "condition_number": float(np.linalg.cond(scaled)),
               "measured_range": {"graph_buckets": buckets,
                                  "per_request_prompt_tokens": [min(contexts), max(contexts)],
                                  "max_rank_mid_context_tokens": [min(max(contexts) for _, contexts in features),
                                                                  max(max(contexts) for _, contexts in features)]},
               "tenant0_transfer": "same weights/topology; long-context solo checked separately; not a Code grid"}
    model = StepModel(copy.deepcopy(params))
    details["rows"] = [{"name": row["name"], "observed_s": row["step_s"],
                        "predicted_s": model.decode_step(1, row["layout"], contexts),
                        "relative_error": model.decode_step(1, row["layout"], contexts) / row["step_s"] - 1}
                       for row, (_, contexts) in zip(rows, features)]
    return params, details


def fit_interference(params, samples):
    solo = [row for row in samples if not any(row["layout"])]
    active = [row for row in samples if any(row["layout"])]
    solo_step = statistics.median(row["code_tpot_s"] for row in solo)

    def table(values):
        grouped = defaultdict(list)
        for row, value in values:
            grouped[capture_bucket(max(row["layout"]))].append(value)
        return {str(bucket): max(0.0, statistics.median(grouped[bucket])) for bucket in sorted(grouped)}

    code = [(row, row["code_tpot_s"] - solo_step) for row in active]
    conv = {phase: [(row, row[f"conv_during_{phase}_s"] - row["conv_before_s"]) for row in active]
            for phase in ("prefill", "decode")}
    params["interference"] = {
        "0": {"multiplier": 1.0, "additive_s": max(0.0, statistics.median(value for _, value in code)),
              "decode": table(code)},
        "1": {"multiplier": 1.0, "additive_s": max(0.0, statistics.median(
            value for values in conv.values() for _, value in values)),
              **{phase: table(values) for phase, values in conv.items()}}}
    params["code_prefill"] = {
        "solo_s_per_100k": statistics.median(row["code_prefill_s"] * 100000 / row["code_prompt_tokens"] for row in solo),
        "with_other_s_per_100k": table([(row, row["code_prefill_s"] * 100000 / row["code_prompt_tokens"])
                                        for row in active]), "chunk_tokens": 4096}
    params["interference_fit"] = interference_residuals(copy.deepcopy(params), samples)
    return {"method": "median_baseline_anchored_additive_by_phase_and_bucket",
            "scope": "conv_before anchored increments include context growth; not an independent joint prediction",
            "fallback": "phase-blind additive fallback is also checked against every active measurement",
            "code_prompt_tokens": [row["code_prompt_tokens"] for row in samples],
            "residuals": params["interference_fit"]}


def validate_decode_requests(rows, directory):
    checked = []
    for row in rows:
        filename = row["requests_file"]
        if Path(filename).name != filename:
            raise ValueError("Invalid raw decode filename")
        data = (directory / filename).read_bytes()
        requests = json.loads(data)
        if len(requests) != sum(row["layout"]) or len({request["request_id"] for request in requests}) != len(requests):
            raise ValueError("Incomplete raw decode requests")
        for rank, count in enumerate(row["layout"]):
            residents = [request for request in requests if request["rank_initial"] == rank]
            expected = row["contexts"][rank] if "contexts" in row else [row["context_tokens"]] * count
            if sorted(request["prompt_tokens"] for request in residents) != sorted(expected):
                raise ValueError("Raw decode placement/context differs")
        for request in requests:
            cadence = request["cadence"]
            if (request["model"] != 1 or request["output_tokens"] != row["output_tokens"] or not cadence
                    or cadence[-1][1] != row["output_tokens"]
                    or any(later[0] < earlier[0] or later[1] <= earlier[1]
                           for earlier, later in zip(cadence, cadence[1:]))):
                raise ValueError("Invalid decode output/cadence")
        step, window = steady_step(requests)
        if (step is None or not math.isclose(step, row["step_s"], rel_tol=1e-9)
                or window != row["window"]):
            raise ValueError("Reported decode timing differs from raw requests")
        checked.append({"file": filename, "sha256": hashlib.sha256(data).hexdigest(), "requests": len(requests)})
    return checked


def build_bundle(imbalance_job_dir, cost_job_dir, output_dir, interference_job_dir=None):
    imbalance_job_dir, cost_job_dir, output_dir = map(Path, (imbalance_job_dir, cost_job_dir, output_dir))
    interference_job_dir = Path(interference_job_dir) if interference_job_dir is not None else cost_job_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "independent_validation": False}
    try:
        dense_root = imbalance_job_dir / "out" / "calibration" / "imbalance"
        cost_root = cost_job_dir / "out" / "calibration"
        interference_root = interference_job_dir / "out" / "calibration"
        paths = {"imbalance": dense_root / "results.json", "grid": cost_root / "conv-grid.json",
                 "interference": interference_root / "interference.json",
                 "imbalance_phases": imbalance_job_dir / "out" / "phases.json",
                 "cost_phases": cost_job_dir / "out" / "phases.json"}
        if (interference_root / "interference-plan.json").exists():
            paths["interference_plan"] = interference_root / "interference-plan.json"
        if interference_job_dir.resolve() != cost_job_dir.resolve():
            paths["interference_phases"] = interference_job_dir / "out" / "phases.json"
        sources, descriptors = {}, []
        for role, path in sorted(paths.items()):
            data = path.read_bytes()
            filename = role + ".json"
            (output_dir / filename).write_bytes(data)
            sources[role] = json.loads(data)
            source_job = (imbalance_job_dir if role.startswith("imbalance") else
                          interference_job_dir if role.startswith("interference") else cost_job_dir)
            job = source_job.name
            descriptors.append({"role": role, "file": filename, "job": job.removeprefix("job-"),
                                "sha256": hashlib.sha256(data).hexdigest()})
        validate_calibration_sources(sources, 2, 2)
        report["raw_decode_checks"] = {"imbalance": validate_decode_requests(sources["imbalance"], dense_root),
                                       "grid": validate_decode_requests(sources["grid"], cost_root)}
        raw, raw_sources = {}, []
        for row in sources["interference"]:
            key = interference_key(row)
            filename = "interference-" + key + "-requests.json"
            if Path(filename).name != filename:
                raise ValueError("Invalid interference case filename")
            data = (interference_root / filename).read_bytes()
            (output_dir / filename).write_bytes(data)
            raw[key] = json.loads(data)
            raw_sources.append({"case": key, "file": filename, "job": interference_job_dir.name.removeprefix("job-"),
                                "sha256": hashlib.sha256(data).hexdigest()})
        samples = interference_samples(sources["interference"], raw, 2, sources.get("interference_plan"))
        params, report["decode"] = fit_decode(sources["grid"] + sources["imbalance"])
        report["interference"] = fit_interference(params, samples)
        report["decode_by_source"] = {role: fit_residuals(copy.deepcopy(params), sources[role])
                                      for role in ("grid", "imbalance")}
        params["provenance"] = {"schema_version": 2 if "interference_plan" in sources else 1,
                                "topology": {"num_ranks": 2, "tensor_parallel_size": 2,
                                                                 "expert_parallel_size": 4},
                                "sources": descriptors, "interference_requests": raw_sources,
                                "fit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        params["calibration_scope"] = report["decode"]["measured_range"]
        candidate = output_dir / "step-params.candidate.json"
        candidate.write_text(json.dumps(params, indent=2, allow_nan=False) + "\n")
        verified = load_measured_cost(candidate, 2, 2)
        report.update(status="passed_in_sample_checks", fit=verified["verified_fit"],
                      next_gate="Independent capacity, long-context, and online-interference pilot required")
        (output_dir / "step-params.json").write_bytes(candidate.read_bytes())
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        (output_dir / "fit-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imbalance-job-dir", type=Path, required=True)
    parser.add_argument("--cost-job-dir", type=Path, required=True)
    parser.add_argument("--interference-job-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_bundle(args.imbalance_job_dir, args.cost_job_dir, args.output_dir, args.interference_job_dir)
    except Exception as error:
        parser.exit(1, f"Cost bundle rejected: {type(error).__name__}: {error}\n")
    print(json.dumps({"status": report["status"], "fit": report["fit"], "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
