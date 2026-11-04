import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path

import head_hidden_probe
from fp32_head_probe import CAPTURES


BUFFER_NAMES = ("layer_input", "qkv_input", "qkv_output")
INPUT_NAMES = ("input_ids", "positions")
MAX_DEVICE_BYTES = 256 * 1024 * 1024
SCHEMA = "upstream-first-layer-v2"
FIRST_LAYER_TARGETS = {149: head_hidden_probe.TARGETS[149]}
SCIENTIFIC_SCOPE = {"prompt_tokens": 149, "positions": [52, 53, 54], "candidates": [12822, 20632],
                    "P74_first_layer_coverage": [], "P74_known_uncovered_positions": [139, 442]}


def selected_position(state):
    return head_hidden_probe.selected_position(state) if state.num_prompt_tokens in FIRST_LAYER_TARGETS else None


def bf16_digest(values):
    encoded = bytearray()
    if not values:
        raise ValueError("Require a nonempty BF16 row")
    for value in values:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Require finite BF16 values")
        packed = struct.pack("<f", value)
        if packed[:2] != b"\x00\x00":
            raise ValueError("Row values are not exact BF16 values")
        encoded.extend(packed[2:])
    return hashlib.sha256(encoded).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_projection(record):
    shape = record["weight_shape"]
    if (record["weight_dtype"] != "torch.bfloat16" or record["weight_device"] != "cuda"
            or record["quant_method"] != "UnquantizedLinearMethod"
            or record["skip_bias_add"] is not False or not valid_sha(record["weight_sha256"])
            or len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape)):
        raise ValueError("Require CUDA BF16 unquantized QKV affine weights")
    bias = record["bias"]
    if bias is not None and (bias["dtype"] != "torch.bfloat16" or bias["device"] != "cuda"
                             or bias["shape"] != [shape[0]] or not valid_sha(bias["sha256"])):
        raise ValueError("QKV bias precision, shape or SHA differs")


def validate_worker(record):
    expected = {"schema": SCHEMA, "attention_tp": 1, "dp": 2, "ep": 2, "pp": 1,
                "ep_enabled": True, "model_dtype": "torch.bfloat16", "compile_mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": CAPTURES,
                "speculative": False, "async_scheduling": False, "use_ubatching": False,
                "use_sequence_parallel_moe": False, "batch_invariant_enabled": False,
                "layer_index": 0, "buffer_names": list(BUFFER_NAMES),
                "scientific_scope": SCIENTIFIC_SCOPE,
                "max_device_bytes": MAX_DEVICE_BYTES, "is_algorithm_benefit": False}
    if (any(record.get(name) != value or type(record.get(name)) is not type(value)
            for name, value in expected.items())
            or any(type(size) is not int for size in record["cudagraph_capture_sizes"])):
        raise ValueError("First-layer worker configuration differs")
    if (record["service"] not in ("instance-1", "instance-2")
            or type(record["dp_rank"]) is not int or record["dp_rank"] not in (0, 1)
            or not isinstance(record["checkpoint"], str) or not record["checkpoint"]
            or not isinstance(record["model_class"], str) or not record["model_class"]
            or set(record["source_sha256"]) != {"runner", "context", "layer", "qkv"}
            or not all(valid_sha(value) for value in record["source_sha256"].values())):
        raise ValueError("First-layer worker identity or installed source SHA differs")
    validate_projection(record["projection"])


def validate_weights(root):
    paths = sorted(Path(root).glob("upstream-first-layer-worker-*.json"))
    expected = {f"upstream-first-layer-worker-{service}-dp{rank}.json"
                for service in ("instance-1", "instance-2") for rank in (0, 1)}
    if {path.name for path in paths} != expected:
        raise ValueError("Require first-layer weight records from all four workers")
    records = {}
    for path in paths:
        record = json.loads(path.read_text())
        validate_worker(record)
        identity = (record["service"], record["dp_rank"])
        if path.name != f"upstream-first-layer-worker-{identity[0]}-dp{identity[1]}.json":
            raise ValueError("First-layer weight filename/worker identity differs")
        native = json.loads((Path(root) / f"worker-{identity[0]}-dp{identity[1]}.json").read_text())
        if (native["service"] != identity[0] or native["dp_rank"] != identity[1]
                or native["checkpoint"] != record["checkpoint"]
                or any(native["groups"][group]["world_size"] != size
                       for group, size in (("tp", 1), ("dp", 2), ("ep", 2)))
                or native["groups"]["dp"]["rank"] != identity[1]):
            raise ValueError("First-layer record differs from actual worker groups/checkpoint")
        records[identity] = record
    stable = [{name: record[name] for name in ("checkpoint", "model_class", "projection", "source_sha256")}
              for record in records.values()]
    if any(record != stable[0] for record in stable):
        raise ValueError("Four workers have inconsistent QKV weights, model or installed sources")
    return records


def validate_snapshot(snapshot, head, request, prompt, role, repeat, layout, worker):
    position = snapshot["position"]
    target = FIRST_LAYER_TARGETS.get(request["prompt_tokens"])
    if target is None or type(position) is not int or abs(position - target["position"]) > 1:
        raise ValueError("First-layer snapshot outside declared target positions")
    verified = head_hidden_probe.validate_snapshot(
        head, request, target, prompt, role, repeat, layout, position)
    emitted = request["token_ids"][position]
    if emitted in target["candidates"]:
        chosen = head["candidate_logits"][target["candidates"].index(emitted)]
        if any(value > chosen for value in head["candidate_logits"]):
            raise ValueError("GPU candidate ordering contradicts the actual emitted greedy token")
    bound = ("service", "dp_rank", "native_request_id", "role", "repeat", "layout", "position",
             "prompt_tokens", "prompt_token_ids", "output_prefix_token_ids", "context_token_ids",
             "context_sha256", "forward_consumed_prefix_tokens", "sample_discarded", "batch_rows",
             "batch_native_request_ids", "row_index")
    if any(snapshot[name] != head[name] for name in bound):
        raise ValueError("First-layer snapshot identity/prefix/logits row differs from head snapshot")
    if (type(snapshot["row_index"]) is not int or type(snapshot["dp_rank"]) is not int
            or type(snapshot["batch_rows"]) is not int
            or (role == "trial" and type(snapshot["repeat"]) is not int)):
        raise ValueError("First-layer worker/request row indices must be genuine integers")
    if (snapshot["schema"] != SCHEMA or snapshot["checkpoint"] != worker["checkpoint"]
            or snapshot["model_class"] != worker["model_class"]
            or snapshot["projection"] != worker["projection"]
            or snapshot["source_sha256"] != worker["source_sha256"]
            or snapshot["scientific_scope"] != SCIENTIFIC_SCOPE
            or snapshot["layer_index"] != 0 or snapshot["first_layer_residual_is_none"] is not True
            or snapshot["observation_includes_CPU_synchronization"] is not True
            or snapshot["is_algorithm_benefit"] is not False):
        raise ValueError("First-layer snapshot model/weights/provenance differs")
    flag_names = ("allow_bf16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction_split_k")
    if (set(snapshot["matmul_flags"]) != set(flag_names)
            or any(type(snapshot["matmul_flags"][name]) is not bool for name in flag_names)
            or any(snapshot["matmul_flags"][name] != head[name] for name in flag_names)):
        raise ValueError("First-layer matmul controls differ from actual same-forward head record")
    padded = snapshot["num_tokens_padded"]
    starts = snapshot["query_start_loc"]
    row = snapshot["row_index"]
    token_row = snapshot["token_row_index"]
    if (type(padded) is not int or not 0 < padded <= 4096
            or len(starts) != snapshot["batch_rows"] + 1 or starts[0] != 0
            or any(type(value) is not int for value in starts)
            or any(left >= right for left, right in zip(starts, starts[1:]))
            or starts[-1] > padded or type(token_row) is not int
            or token_row != starts[row + 1] - 1 or not 0 <= token_row < padded
            or snapshot["forward_position"] != len(snapshot["context_token_ids"]) - 1
            or snapshot["forward_token_id"] != snapshot["context_token_ids"][-1]):
        raise ValueError("First-layer token row/position is not bound to complete forward prefix")
    mode = snapshot["forward_mode"]
    origin = snapshot["buffer_origin"]
    if (snapshot["buffer_key"] != [mode, padded, snapshot["forward_descriptor"]] or mode not in ("NONE", "FULL")
            or type(snapshot["buffer_key"][1]) is not int
            or snapshot["buffer_hook_calls"] != {name: 1 for name in BUFFER_NAMES}
            or any(type(value) is not int for value in snapshot["buffer_hook_calls"].values())
            or type(snapshot["retained_device_bytes"]) is not int
            or not 0 < snapshot["retained_device_bytes"] <= MAX_DEVICE_BYTES):
        raise ValueError("First-layer buffer mode/shape or bounded provenance differs")
    descriptor = snapshot["forward_descriptor"]
    if descriptor is not None and (descriptor.get("num_tokens") != padded
                                   or type(descriptor.get("num_tokens")) is not int):
        raise ValueError("Forward descriptor and padded input shape differ")
    if mode == "FULL":
        if (padded not in CAPTURES or descriptor is None
                or json.dumps(snapshot["capture_descriptor"], sort_keys=True) != json.dumps(descriptor, sort_keys=True)
                or origin != "cuda_graph_capture"
                or snapshot["read_execution"] != "graph_replay"
                or snapshot["current_forward_hook_calls"] != 0
                or any(right - left != 1 for left, right in zip(starts, starts[1:]))):
            raise ValueError("Graph replay buffer/capture/row mapping differs")
    elif (origin != "eager_forward" or snapshot["capture_descriptor"] is not None
          or snapshot["read_execution"] != "eager_forward"
          or snapshot["current_forward_hook_calls"] != len(BUFFER_NAMES)):
        raise ValueError("Eager buffer cannot cover a graph replay")
    witness = snapshot["refresh_witness"]
    if (type(witness["forward_sequence"]) is not int or witness["forward_sequence"] <= 0
            or type(witness["capture_increment_calls"]) is not int):
        raise ValueError("Invalid buffer refresh witness sequence/increment coverage")
    if mode == "FULL":
        if (witness["kind"] != "cuda_graph_generation" or witness["capture_increment_calls"] != 1
                or type(witness["generation_before"]) is not int
                or type(witness["generation_after"]) is not int
                or not 0 <= witness["generation_before"] < 2**63 - 1
                or witness["generation_after"] != witness["generation_before"] + 1
                or witness["counter"] != {"shape": [], "dtype": "torch.int64", "device": "cuda"}):
            raise ValueError("Graph refresh generation must advance exactly once on its captured CUDA counter")
    elif (witness["kind"] != "eager_forward_sequence" or witness["capture_increment_calls"] != 0
          or witness["counter"] is not None or witness["generation_before"] is not None
          or witness["generation_after"] is not None):
        raise ValueError("Eager input binding cannot claim a graph refresh witness")
    binding = snapshot["input_binding"]
    if (set(binding) != {"origin", "hook_calls", *INPUT_NAMES} or binding["origin"] != origin
            or type(binding["hook_calls"]) is not int or binding["hook_calls"] != 1):
        raise ValueError("Require actual model input clones from the same captured/eager forward")
    for name in INPUT_NAMES:
        captured = binding[name]
        expected = snapshot["forward_token_id"] if name == "input_ids" else snapshot["forward_position"]
        supported_dtypes = ("torch.int32", "torch.int64") if name == "input_ids" else ("torch.int64",)
        if (captured["shape"] != [padded] or any(type(size) is not int for size in captured["shape"])
                or captured["dtype"] not in supported_dtypes or captured["device"] != "cuda"
                or type(captured["token_row_index"]) is not int or captured["token_row_index"] != token_row
                or type(captured["value"]) is not int or captured["value"] != expected):
            raise ValueError("Captured input token/position row is stale or unbound to the actual complete prefix")
    if set(snapshot["buffers"]) != set(BUFFER_NAMES):
        raise ValueError("Require exactly three first-layer buffers")
    output_width, input_width = worker["projection"]["weight_shape"]
    for name in BUFFER_NAMES:
        buffer = snapshot["buffers"][name]
        width = output_width if name == "qkv_output" else input_width
        if (buffer["shape"] != [padded, width] or buffer["dtype"] != "torch.bfloat16"
                or any(type(size) is not int for size in buffer["shape"])
                or buffer["device"] != "cuda" or buffer["token_row_index"] != token_row
                or len(buffer["values"]) != width
                or bf16_digest(buffer["values"]) != buffer["row_sha256"]):
            raise ValueError("First-layer buffer shape/index/dtype or BF16 row SHA differs")
    token_id_bytes = 4 if binding["input_ids"]["dtype"] == "torch.int32" else 8
    if snapshot["retained_device_bytes"] < padded * ((2 * input_width + output_width) * 2 + token_id_bytes + 8) + (8 if mode == "FULL" else 0):
        raise ValueError("Retained buffer bytes do not include the actual buffers and input/refresh witnesses")
    return verified


def compare(reference, actual):
    identical = {name: reference["buffers"][name]["row_sha256"] == actual["buffers"][name]["row_sha256"]
                 for name in BUFFER_NAMES}
    changed_shape = (reference["forward_mode"], reference["num_tokens_padded"]) != (
        actual["forward_mode"], actual["num_tokens_padded"])
    controls_equal = reference["matmul_flags"] == actual["matmul_flags"]
    affine = (identical["qkv_input"] and not identical["qkv_output"] and changed_shape
              and controls_equal and reference["projection"] == actual["projection"])
    conclusion = ("qkv_affine_difference_observed_with_batch_or_graph_change_not_causal_proof" if affine else
                  "upstream_input_changed_not_localized" if not identical["qkv_input"] else
                  "qkv_output_changed_without_isolated_shape_controls" if not identical["qkv_output"] else
                  "no_first_projection_difference_observed")
    return {"bitwise_identical": identical, "batch_or_graph_shape_changed": changed_shape,
            "matmul_flags_identical": controls_equal, "affine_batch_sensitivity_observed": affine,
            "affine_batch_sensitivity_proven": False,
            "conclusion": conclusion, "specific_cublas_algorithm_proven": False,
            "downstream_first_top1_causality_proven": False, "numerical_root_cause_proven": False,
            "historical_KV_equivalence_proven": False, "current_step_only": True}


def reference_bindings(records):
    references = {}
    native_ids = set()
    bindings = []
    for request in records:
        logical = request["trace_request_id"]
        native_id = request["request_id"]
        if not isinstance(native_id, str) or not native_id or native_id in native_ids:
            raise ValueError("Duplicate or invalid serial execution request identity")
        native_ids.add(native_id)
        previous = references.get(logical)
        if previous is not None and any(json.dumps(previous[field], allow_nan=False) !=
                json.dumps(request[field], allow_nan=False)
                for field in ("model", "rank_initial", "prompt_tokens", "token_ids")):
            raise ValueError("Serial references are not repeatable or their request identity differs")
        references[logical] = request
        bindings.append(("reference", None, None, logical, request))
    return references, bindings


def analyze(root):
    root = Path(root)
    head_result = head_hidden_probe.analyze(root)
    workers = validate_weights(root)
    trace = json.loads((root / "trace.json").read_text())
    bank_bytes = (root / "frozen/inputs/prompt-bank.json").read_bytes()
    if hashlib.sha256(bank_bytes).hexdigest() != trace["prompt_bank_sha256"]:
        raise ValueError("First-layer prompt bank SHA differs from frozen trace")
    prompts = json.loads(bank_bytes)["prompts"]
    marks = {request["request_id"]: request for request in trace["requests"]}
    reference_records = json.loads((root / "references.json").read_text())
    references, bindings = reference_bindings(reference_records)
    trials = head_result["full_static_analysis"]["diagnostic"]["trials"]
    trial_records = {}
    for trial in trials:
        repeat = trial["repeat"]
        if repeat in trial_records:
            raise ValueError("Duplicate trial repeat identity")
        records = json.loads((root / f"r{repeat}-no_move/requests.json").read_text())
        trial_records[repeat] = {request["request_id"]: request for request in records}
        if len(trial_records[repeat]) != len(records):
            raise ValueError("Duplicate trial request identities")
        bindings.extend(("trial", repeat, trial["layout"], request["request_id"], request) for request in records)
    heads = [json.loads(path.read_text()) for path in (root / "head-hidden-snapshots").glob("*.json")]
    observations = {}
    for path in sorted((root / "upstream-first-layer-snapshots").glob("*.json")):
        snapshot = json.loads(path.read_text())
        matching = [binding for binding in bindings if binding[:3] == (
            snapshot["role"], snapshot["repeat"], snapshot["layout"])
            and head_hidden_probe.native_request_matches(snapshot["native_request_id"], binding[4]["request_id"])]
        if len(matching) != 1:
            raise ValueError("Unbound or ambiguous first-layer native request/role/repeat/layout")
        role, repeat, layout, logical, request = matching[0]
        matching_heads = [head for head in heads if all(head[field] == snapshot[field] for field in
            ("native_request_id", "role", "repeat", "position", "service", "dp_rank"))]
        if len(matching_heads) != 1:
            raise ValueError("Require exactly one matching head-hidden snapshot")
        verified = validate_snapshot(snapshot, matching_heads[0], request,
            prompts[marks[logical]["prompt_key"]]["token_ids"], role, repeat, layout,
            workers[(snapshot["service"], snapshot["dp_rank"])])
        key = (role, repeat, logical, snapshot["position"])
        if key in observations:
            raise ValueError("Duplicate first-layer request/position snapshot")
        observations[key] = (snapshot, verified)
    comparisons = []
    uncovered = []
    for trial in trials:
        for logical, actual_request in trial_records[trial["repeat"]].items():
            serial = references[logical]["token_ids"]
            actual = actual_request["token_ids"]
            if len(serial) != len(actual):
                raise ValueError("Require equal complete response lengths, not truncated prefixes")
            first = next((position for position, tokens in enumerate(zip(serial, actual))
                          if tokens[0] != tokens[1]), None)
            reported = [entry for entry in trial["serial_comparisons"] if entry["request_id"] == logical]
            if len(reported) != 1 or reported[0]["first_difference_zero_based"] != first:
                raise ValueError("Reported first mismatch differs from actual complete output tokens")
            target = FIRST_LAYER_TARGETS.get(actual_request["prompt_tokens"])
            if target is None:
                uncovered.append({"request_id": logical, "repeat": trial["repeat"], "layout": trial["layout"],
                    "prompt_tokens": actual_request["prompt_tokens"], "actual_first_mismatch": first,
                    "attribution_available": False, "reason": "outside_P149_first_layer_scientific_scope"})
                continue
            position = first if first is not None else target["position"]
            result = {"request_id": logical, "repeat": trial["repeat"], "layout": trial["layout"],
                      "position": position, "actual_first_mismatch": first,
                      "is_first_mismatch": first == position, "attribution_available": False,
                      "affine_batch_sensitivity_proven": False, "affine_batch_sensitivity_observed": False}
            pairs = [observations.get((role, repeat, logical, position)) for role, repeat in
                     (("reference", None), ("trial", trial["repeat"]))]
            if abs(position - target["position"]) > 1 or any(pair is None for pair in pairs):
                result["reason"] = "actual_position_not_captured_or_missing_paired_buffers"
            else:
                reference, actual = pairs[0][0], pairs[1][0]
                common = reference["context_token_ids"] == actual["context_token_ids"]
                covered = first is None or {serial[position], actual_request["token_ids"][position]} == set(target["candidates"])
                result.update({"full_common_prefix": common, "sampler_contrasts_verified": all(pair[1] for pair in pairs),
                               "actual_first_top1_candidates_covered": covered})
                if common and covered and all(pair[1] for pair in pairs):
                    result.update(compare(reference, actual))
                    result["attribution_available"] = True
                else:
                    result["reason"] = "complete_prefix_or_actual_candidate_sampler_coverage_missing"
            comparisons.append(result)
    complete = bool(comparisons) and all(result["attribution_available"] for result in comparisons)
    return {"status": "completed_first_layer_observation_not_benefit" if complete else
            "incomplete_first_layer_attribution_not_benefit", "head_hidden_analysis": head_result,
            "comparisons": comparisons, "validated_snapshots": len(observations),
            "scientific_scope": SCIENTIFIC_SCOPE, "uncovered_requests": uncovered,
            "four_worker_QKV_weights_verified": True, "all_in_scope_target_attributions_available": complete,
            "all_trace_first_divergences_attributed": False,
            "all_trace_first_mismatch_positions_observed": complete and not any(
                record["actual_first_mismatch"] is not None for record in uncovered),
            "quality_equivalence_proven": False, "numerical_root_cause_proven": False,
            "strict_token_check_relaxed": False, "is_algorithm_benefit": False}


def main():
    parser = argparse.ArgumentParser(description="Offline first-layer analysis only; never launches a worker")
    parser.add_argument("--verify", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.verify)
    with args.output.open("x") as destination:
        destination.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
