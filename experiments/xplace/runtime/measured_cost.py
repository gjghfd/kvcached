import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from cost_model import StepModel


SOURCE_ROLES = {"imbalance", "grid", "interference", "imbalance_phases", "cost_phases"}
OPTIONAL_SOURCE_ROLES = {"interference_plan", "interference_phases"}
MAX_RELATIVE_RMSE = 0.15
MAX_RELATIVE_ERROR = 0.30


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def verified_phases(record, num_ranks, tp_size, required):
    phases = {entry["phase"]: entry for entry in record["phases"]}
    for phase in {"startup", "job", *required}:
        if phases.get(phase, {}).get("status") != "passed":
            raise ValueError("Calibration phase did not pass: " + phase)
    topology = phases["startup"]["topology"]
    expected = {"num_ranks": num_ranks, "tensor_parallel_size": tp_size,
                "expert_parallel_size": num_ranks * tp_size, "physical_gpus": num_ranks * tp_size}
    if any(topology.get(key) != value for key, value in expected.items()):
        raise ValueError("Calibration source topology mismatch")
    return phases


def validate_decode_rows(rows, num_ranks):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Missing measured decode rows")
    names = set()
    for row in rows:
        layout = row.get("layout", [])
        window = row.get("window", {})
        if (len(layout) != num_ranks or any(not isinstance(count, int) or count < 0 for count in layout)
                or not sum(layout) or row.get("status") != "measured" or row.get("model") != 1
                or row.get("preemptions") != 0 or not positive(row.get("step_s"))
                or not positive(row.get("output_tokens")) or window.get("window_s", 0) < 3
                or window.get("samples") != sum(layout)):
            raise ValueError("Invalid decode calibration row")
        if not row.get("name") or row["name"] in names:
            raise ValueError("Missing or duplicate calibration case")
        names.add(row["name"])
        if "contexts" in row:
            contexts = row["contexts"]
            if len(contexts) != num_ranks or any(len(values) != count or any(not positive(value) for value in values)
                                               for values, count in zip(contexts, layout)):
                raise ValueError("Invalid context layout")
        elif not positive(row.get("context_tokens")):
            raise ValueError("Missing calibration context length")


def fit_residuals(params, rows):
    model = StepModel(params)
    errors = []
    for row in rows:
        counts = row["layout"]
        middle = row["output_tokens"] / 2
        contexts = ([sum(values) + len(values) * middle for values in row["contexts"]] if "contexts" in row else
                    [count * (row["context_tokens"] + middle) for count in counts])
        prediction = model.decode_step(row["model"], counts, contexts)
        if not positive(prediction):
            raise ValueError("Nonpositive calibrated prediction")
        errors.append((prediction - row["step_s"]) / row["step_s"])
    return {"cases": len(errors), "relative_rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
            "max_relative_error": max(abs(value) for value in errors)}


def read_source(directory, source):
    filename = source["file"]
    if not filename or Path(filename).name != filename or filename in (".", "..") or not source.get("job"):
        raise ValueError("Invalid measured cost source filename/job")
    data = (Path(directory) / filename).read_bytes()
    if hashlib.sha256(data).hexdigest() != source["sha256"]:
        raise ValueError("Measured cost source SHA mismatch: " + filename)
    return json.loads(data)


def validate_calibration_sources(sources, num_ranks, tp_size):
    if not SOURCE_ROLES <= set(sources) or set(sources) - SOURCE_ROLES - OPTIONAL_SOURCE_ROLES:
        raise ValueError("Missing measured cost source roles")
    dense_phases = verified_phases(sources["imbalance_phases"], num_ranks, tp_size, {"imbalance"})
    cost_required = {"dataplane", "grid"}
    if "interference_phases" not in sources:
        cost_required.add("interference")
    cost_phases = verified_phases(sources["cost_phases"], num_ranks, tp_size, cost_required)
    if "interference_phases" in sources:
        separate = verified_phases(sources["interference_phases"], num_ranks, tp_size, {"dataplane", "interference"})
        if separate["dataplane"].get("migrations") != 8 or "interference_plan" not in sources:
            raise ValueError("Separate interference job needs a repeated plan and eight-move dataplane gate")
    for role, phase in (("imbalance", dense_phases["imbalance"]), ("grid", cost_phases["grid"])):
        rows = sources[role]
        validate_decode_rows(rows, num_ranks)
        if phase.get("measured") != len(rows) or (role == "imbalance" and phase.get("planned") != len(rows)):
            raise ValueError("Calibration source is incomplete")
    if cost_phases["dataplane"].get("migrations") != 8:
        raise ValueError("Missing eight-move dataplane gate")


def interference_key(row):
    return f"{row['name']}-r{row['repeat']}"


def repeated_interference_plan(num_ranks, seed=23):
    return {"schema_version": 2, "repeats": 3, "cadence_stride": 1, "seed": seed,
            "code_output_tokens": 64, "conv_output_tokens": 3000,
            "min_total_decode_window_s": 3.0,
            "layouts": [{"name": f"bal{count}", "layout": [count] * num_ranks} for count in (0, 2, 8, 16)]}


def interference_samples(rows, raw_requests, num_ranks, plan=None):
    from calibrate import window_rate

    if not rows or not any(not any(row["layout"]) for row in rows) or not any(any(row["layout"]) for row in rows):
        raise ValueError("Missing solo/active interference calibration")
    keys = [interference_key(row) for row in rows]
    if len(set(keys)) != len(keys) or set(raw_requests) != set(keys):
        raise ValueError("Missing or duplicate raw interference cases")
    if plan is not None:
        if (type(plan.get("seed")) is not int or plan != repeated_interference_plan(num_ranks, plan["seed"])):
            raise ValueError("Invalid fixed repeated-interference plan")
        expected = [(item["name"], repeat, item["layout"]) for repeat in range(plan["repeats"])
                    for item in plan["layouts"]]
        if [(row["name"], row["repeat"], row["layout"]) for row in rows] != expected:
            raise ValueError("Missing, reordered, or extra repeated interference cases")
        if any(row.get("measurement_schema") != 2 or row.get("preemptions") != 0 for row in rows):
            raise ValueError("Repeated interference requires its schema and zero preemptions")
    elif any(row.get("measurement_schema") is not None for row in rows):
        raise ValueError("Repeated interference plan is missing")
    decode_durations = {}
    samples = []
    for row in rows:
        layout = row["layout"]
        if len(layout) != num_ranks or any(type(count) is not int or count < 0 for count in layout):
            raise ValueError("Invalid interference layout")
        requests = raw_requests[interference_key(row)]
        codes = [request for request in requests if request["model"] == 0]
        conv = [request for request in requests if request["model"] == 1]
        if (len(codes) != 1 or len(requests) != 1 + sum(layout)
                or len({request["request_id"] for request in requests}) != len(requests)
                or Counter(request["rank_initial"] for request in conv) != Counter(
                    {rank: count for rank, count in enumerate(layout) if count})):
            raise ValueError("Raw interference requests do not match layout")
        for request in requests:
            cadence = request.get("cadence", [])
            if (not positive(request.get("prompt_tokens")) or not positive(request.get("output_tokens"))
                    or not cadence or cadence[-1][1] != request["output_tokens"]
                    or any(not math.isfinite(stamp) or not positive(count) for stamp, count in cadence)
                    or any(later[0] < earlier[0] or later[1] <= earlier[1]
                           for earlier, later in zip(cadence, cadence[1:]))):
                raise ValueError("Invalid raw interference cadence/length")
            if plan is not None and request.get("cadence_stride") != plan["cadence_stride"]:
                raise ValueError("Repeated interference requires dense cadence")
        code = codes[0]
        if code["output_tokens"] != 64 or code["rank_initial"] != 0:
            raise ValueError("Interference requires the original 64-token Code output on rank 0")
        if plan is not None and (code["prompt_tokens"] < 100000 or any(
                request["output_tokens"] != plan["conv_output_tokens"] for request in conv)):
            raise ValueError("Repeated interference changed original workload lengths")
        measured = {"code_prefill_s": code["first_token_at"] - code["submitted_at"],
                    "code_tpot_s": (code["completed_at"] - code["first_token_at"]) / 63}
        if conv:
            windows = {"conv_before_s": (code["arrived_at"] - 3, code["arrived_at"]),
                       "conv_during_prefill_s": (code["submitted_at"] + 1, code["first_token_at"] - 1),
                       "conv_during_decode_s": (code["first_token_at"], code["completed_at"])}
            for name, (start, end) in windows.items():
                rates = [window_rate(request, start, end) for request in conv]
                duration_ok = end - start >= 3
                if plan is not None and name == "conv_during_decode_s":
                    duration_ok = positive(end - start)
                    decode_durations.setdefault(row["name"], []).append(end - start)
                if plan is not None and any(request.get("first_token_at", math.inf) > start or
                                            request.get("completed_at", -math.inf) < end for request in conv):
                    raise ValueError("A repeated interference request does not cover the full window: " + name)
                if not duration_ok or any(not positive(rate) for rate in rates):
                    raise ValueError("Incomplete interference observation window: " + name)
                measured[name] = statistics.median(rates)
        if any(not positive(row.get(key)) or not math.isclose(row[key], value, rel_tol=1e-9, abs_tol=1e-9)
               for key, value in measured.items()):
            raise ValueError("Reported interference differs from raw requests")
        samples.append({**row, "code_prompt_tokens": code["prompt_tokens"]})
    if plan is not None and any(len(durations) != plan["repeats"] or
                                sum(durations) < plan["min_total_decode_window_s"]
                                for durations in decode_durations.values()):
        raise ValueError("Repeated interference decode duration is below the cumulative gate")
    return samples


def interference_residuals(params, samples):
    from cost_model import capture_bucket

    model = StepModel(params)
    solo = statistics.median(row["code_tpot_s"] for row in samples if not any(row["layout"]))
    groups = {"prefill": [], "interference": [], "code_solo_transfer": []}
    for row in samples:
        active = any(row["layout"])
        bucket = capture_bucket(max(row["layout"])) if active else None
        prediction = model.prefill_s_per_token(0, bucket) * row["code_prompt_tokens"]
        groups["prefill"].append((prediction - row["code_prefill_s"]) / row["code_prefill_s"])
        if not active:
            counts = [1] + [0] * (len(row["layout"]) - 1)
            contexts = [row["code_prompt_tokens"] + 32] + [0] * (len(counts) - 1)
            prediction = model.decode_step(0, counts, contexts)
            groups["code_solo_transfer"].append((prediction - row["code_tpot_s"]) / row["code_tpot_s"])
            continue
        for phase in (True, "decode"):
            prediction = model.interfere(0, solo, phase, bucket)
            groups["interference"].append((prediction - row["code_tpot_s"]) / row["code_tpot_s"])
        for phase in ("prefill", "decode"):
            observed = row[f"conv_during_{phase}_s"]
            for mode in (True, phase):
                prediction = model.interfere(1, row["conv_before_s"], mode, bucket)
                groups["interference"].append((prediction - observed) / observed)
    return {name: {"cases": len(errors),
                   "relative_rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
                   "max_relative_error": max(abs(value) for value in errors)} for name, errors in groups.items()}


def verify_residuals(reported, actual):
    if actual["relative_rmse"] > MAX_RELATIVE_RMSE or actual["max_relative_error"] > MAX_RELATIVE_ERROR:
        raise ValueError("Measured cost residual exceeds predeclared tolerance")
    if any(not math.isclose(reported.get(key, math.inf), actual[key], rel_tol=1e-6, abs_tol=1e-9) for key in actual):
        raise ValueError("Reported fit residual differs from actual model")


def load_measured_cost(path, num_ranks, tp_size):
    path = Path(path)
    params = json.loads(path.read_text())
    provenance = params.get("provenance", {})
    topology = {"num_ranks": num_ranks, "tensor_parallel_size": tp_size,
                "expert_parallel_size": num_ranks * tp_size}
    version = provenance.get("schema_version")
    if version not in (1, 2) or provenance.get("topology") != topology:
        raise ValueError("Measured cost provenance/topology missing or mismatched")
    sources, source_jobs = {}, {}
    for source in provenance.get("sources", []):
        role = source["role"]
        if role in sources or role not in SOURCE_ROLES | OPTIONAL_SOURCE_ROLES:
            raise ValueError("Invalid or duplicate measured cost source")
        sources[role] = read_source(path.parent, source)
        source_jobs[role] = source["job"]
    if version == 1 and set(sources) & OPTIONAL_SOURCE_ROLES:
        raise ValueError("Repeated or separate-job sources require provenance version 2")
    if version == 2:
        if "interference_plan" not in sources:
            raise ValueError("Version 2 requires an interference plan")
        pairs = [("grid", "cost_phases"), ("imbalance", "imbalance_phases"),
                 ("interference", "interference_plan"),
                 ("interference", "interference_phases" if "interference_phases" in sources else "cost_phases")]
        if any(source_jobs.get(first) != source_jobs.get(second) for first, second in pairs):
            raise ValueError("Measured cost sources belong to mismatched jobs")
    validate_calibration_sources(sources, num_ranks, tp_size)
    raw = {}
    for source in provenance.get("interference_requests", []):
        key = source["case"]
        if key in raw:
            raise ValueError("Duplicate raw interference source")
        if version == 2 and source["job"] != source_jobs["interference"]:
            raise ValueError("Raw interference source belongs to a different job")
        raw[key] = read_source(path.parent, source)
    samples = interference_samples(sources["interference"], raw, num_ranks, sources.get("interference_plan"))
    if set(params.get("tenants", {})) != {"0", "1"}:
        raise ValueError("Measured cost must cover both tenants")
    for tenant in params["tenants"].values():
        if not positive(tenant.get("base_s")) or not tenant.get("graph_s"):
            raise ValueError("Invalid measured tenant model")
        coefficients = [tenant.get("attention_s_per_token"), tenant.get("graph_slope_s", 0),
                        *tenant["graph_s"].values()]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in coefficients):
            raise ValueError("Invalid measured cost coefficients")
    if set(params.get("interference", {})) != {"0", "1"} or not params.get("code_prefill"):
        raise ValueError("Measured interference/prefill parameters missing")
    residuals = fit_residuals(params, sources["grid"] + sources["imbalance"])
    verify_residuals(params.get("fit", {}), residuals)
    extra = interference_residuals(params, samples)
    for name, actual in extra.items():
        verify_residuals(params.get("interference_fit", {}).get(name, {}), actual)
    params["verified_fit"] = residuals
    params["verified_interference_fit"] = extra
    return params
