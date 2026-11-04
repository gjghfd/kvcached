import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

import upstream_first_layer_probe as first_layer


SCHEMA = "upstream-qkv-mirror-v1"
SCIENTIFIC_SCOPE = first_layer.SCIENTIFIC_SCOPE
MAX_DEVICE_BYTES = 128 * 1024 * 1024
OPERATORS = ("quant_method.apply", "torch.nn.functional.linear")
PRIMARY_OPERATOR = OPERATORS[0]
BINDING_FIELDS = ("service", "dp_rank", "native_request_id", "role", "repeat", "layout", "position",
                  "prompt_tokens", "prompt_token_ids", "output_prefix_token_ids", "context_token_ids",
                  "context_sha256", "forward_consumed_prefix_tokens", "sample_discarded", "batch_rows",
                  "batch_native_request_ids", "row_index")
SOURCE_READ_FIELDS = ("forward_mode", "num_tokens_padded", "forward_descriptor", "query_start_loc",
                      "token_row_index", "forward_token_id", "forward_position", "buffer_key", "buffer_origin",
                      "buffer_hook_calls", "capture_descriptor", "read_execution", "current_forward_hook_calls",
                      "retained_device_bytes", "input_binding", "refresh_witness", "buffers")
CLAIMS = {"specific_cublas_algorithm_proven": False, "specific_kernel_proven": False,
          "downstream_first_top1_causality_proven": False, "quality_equivalence_proven": False,
          "quality_fixed": False, "historical_KV_equivalence_proven": False, "is_algorithm_benefit": False,
          "numerical_root_cause_proven": False, "affine_effect_due_only_to_batch_dimension_proven": False}
KERNEL_SCHEMA = "upstream-qkv-shadow-kernel-activity-v1"
MAX_KERNEL_OBSERVATIONS = 1
MAX_KERNEL_CALLS = 6
MAX_KERNEL_TRACE_BYTES = 4 * 1024 * 1024
MAX_KERNEL_TRACE_EVENTS = 4096
KERNEL_MARKER_PREFIX = "dp_opt_qkv_shadow_apply:"
KERNEL_REQUEST_FIELDS = ("service", "dp_rank", "native_request_id", "role", "repeat", "layout", "position",
                         "prompt_token_ids", "output_prefix_token_ids", "row_index", "batch_native_request_ids")


def canonical_sha(value):
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as source:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def local_path(root, relative):
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or str(path) != relative:
        raise ValueError("Unsafe mirror evidence path")
    result = Path(root) / relative
    if result.is_symlink() or not result.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Mirror evidence path escapes its archive")
    return result


def kernel_activity_policy(enabled=False):
    if type(enabled) is not bool:
        raise ValueError("Kernel activity opt-in must be an explicit boolean")
    return {"schema": KERNEL_SCHEMA, "enabled": enabled,
            "maximum_observations_per_worker": MAX_KERNEL_OBSERVATIONS,
            "maximum_shadow_calls_per_worker": MAX_KERNEL_CALLS,
            "maximum_trace_bytes_per_call": MAX_KERNEL_TRACE_BYTES,
            "maximum_trace_events_per_call": MAX_KERNEL_TRACE_EVENTS,
            "selection_rule": "first_in_scope_trial_graph_replay_per_worker",
            "operator": PRIMARY_OPERATOR, "extra_shadow_calls_only": True,
            "native_cuda_graph_profiled": False, "record_shapes": False,
            "profile_memory": False, "with_stack": False, "with_flops": False,
            "unsupported_is_nonfatal": True, "profiler_buffer_and_export_peak_not_hard_bounded": True}


def kernel_base_binding(worker, source_read, request, source_path):
    if (type(worker.get("slurm_job_id")) is not int or worker["slurm_job_id"] <= 0
            or type(worker.get("process_id")) is not int or worker["process_id"] <= 0):
        raise ValueError("Kernel activity requires the actual allocation and worker process identity")
    return {"slurm_job_id": worker["slurm_job_id"], "process_id": worker["process_id"],
            "service": worker["service"], "dp_rank": worker["dp_rank"], "gpu": worker["gpu"],
            "mirror_sources_sha256": canonical_sha(worker["source_sha256"]),
            "kernel_activity_policy_sha256": canonical_sha(worker.get("kernel_activity_policy", kernel_activity_policy())),
            "profiler_sources_sha256": canonical_sha(worker.get("kernel_profiler_sources")),
            "source_snapshot_path": source_path,
            "source_buffer_read_sha256": canonical_sha(source_read),
            "request_binding_sha256": canonical_sha({name: request[name] for name in KERNEL_REQUEST_FIELDS}),
            "forward_descriptor": source_read["forward_descriptor"],
            "capture_descriptor": source_read["capture_descriptor"],
            "refresh_witness": source_read["refresh_witness"],
            "role": request["role"], "real_call_repeat": request["repeat"], "position": request["position"]}


def kernel_call_binding(base, record, repeat_index, shape):
    private = record["private_tensors"]
    operand = private["padded_input" if shape == "padded" else "target_input"]
    baseline = record["rounds"][repeat_index][PRIMARY_OPERATOR][shape]
    return {**base, "operator": PRIMARY_OPERATOR, "shape": shape, "mirror_repeat_index": repeat_index,
            "shadow_call_index": repeat_index * 2 + (shape == "target"),
            "input_shape": operand["shape"], "input_matrix_sha256": operand["matrix_sha256"],
            "weight_sha256": private["weight"]["matrix_sha256"],
            "bias_sha256": None if private["bias"] is None else private["bias"]["matrix_sha256"],
            "baseline_output_matrix_sha256": baseline["matrix_sha256"],
            "baseline_output_row_sha256": baseline["row_sha256"],
            "cuda_stream": record["cuda_stream_before"], "matmul_flags_sha256": canonical_sha(record["flags_before"])}


def kernel_marker(binding):
    return KERNEL_MARKER_PREFIX + canonical_sha(binding)


def _event_time(event):
    values = (event.get("ts"), event.get("dur"))
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Kernel activity timestamps/durations must be finite nonnegative values")
    end = values[0] + values[1]
    if not math.isfinite(end):
        raise ValueError("Kernel activity end timestamp must be finite")
    return values[0], end


def parse_kernel_trace(trace, binding):
    result = {"status": "unsupported", "reason": "cuda_activity_absent", "events": [],
              "kernel_names": [], "kernel_activity_observed": False, "unattributed_cuda_events": 0}
    if type(trace) is not dict or type(trace.get("traceEvents")) is not list:
        return {**result, "status": "partial", "reason": "chrome_trace_events_unavailable"}
    events = trace["traceEvents"]
    if len(events) > MAX_KERNEL_TRACE_EVENTS:
        return {**result, "status": "partial", "reason": "trace_event_budget_exceeded"}
    if any(type(event) is not dict for event in events):
        return {**result, "status": "partial", "reason": "unsupported_chrome_event_schema"}
    properties_list = trace.get("deviceProperties", [])
    if type(properties_list) is not list or any(type(properties) is not dict for properties in properties_list):
        return {**result, "status": "partial", "reason": "unsupported_device_properties_schema"}
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in ("user_annotation", "cuda_runtime", "cuda_driver", "kernel"):
            if type(event.get("args", {})) is not dict:
                return {**result, "status": "partial", "reason": "unsupported_chrome_event_arguments"}
            try:
                _event_time(event)
            except ValueError:
                return {**result, "status": "partial", "reason": "invalid_activity_timestamps"}
    markers = [event for event in events if event.get("ph") == "X" and event.get("cat") == "user_annotation"
               and isinstance(event.get("name"), str) and event["name"].startswith(KERNEL_MARKER_PREFIX)]
    if any(event["name"] != kernel_marker(binding) for event in markers) or len(markers) > 1:
        raise ValueError("Kernel trace call identity/annotation differs from its binding")
    if not markers:
        return {**result, "status": "partial", "reason": "call_annotation_unavailable"}
    marker = markers[0]
    if type(marker.get("pid")) is not int or marker["pid"] != binding["process_id"]:
        raise ValueError("Kernel trace CPU process differs from the actual worker")
    if type(marker.get("tid")) is not int:
        return {**result, "status": "partial", "reason": "call_annotation_thread_unavailable"}
    start, end = _event_time(marker)
    launches = {}
    for event in events:
        if (event.get("ph") != "X" or event.get("cat") not in ("cuda_runtime", "cuda_driver")
                or not isinstance(event.get("name"), str)
                or not re.fullmatch(r"(?:cudaLaunchKernel(?:ExC)?|cudaLaunchCooperativeKernel|cuLaunchKernel(?:Ex)?)", event["name"])):
            continue
        launch_start, launch_end = _event_time(event)
        if not start <= launch_start <= launch_end <= end:
            continue
        if event.get("pid") != marker["pid"] or event.get("tid") != marker.get("tid"):
            continue
        arguments = event.get("args", {})
        correlation = arguments.get("correlation") if type(arguments) is dict else None
        if type(correlation) is not int or correlation < 0:
            continue
        if correlation in launches:
            return {**result, "status": "partial", "reason": "ambiguous_launch_correlation"}
        launches[correlation] = event
    captured = []
    incomplete = False
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        arguments = event.get("args", {})
        correlation = arguments.get("correlation") if type(arguments) is dict else None
        if type(correlation) is not int or correlation not in launches:
            result["unattributed_cuda_events"] += 1
            continue
        if type(arguments.get("device")) is not int or arguments["device"] != binding["gpu"]["device_index"]:
            raise ValueError("Correlated kernel activity differs from the bound CUDA device")
        _event_time(event)
        if not isinstance(event.get("name"), str) or not event["name"]:
            raise ValueError("Correlated CUDA kernel has no actual activity name")
        complete = (type(arguments.get("stream")) is int and arguments["stream"] >= 0
                    and all(type(arguments.get(name)) is list and len(arguments[name]) == 3
                            and all(type(value) is int and value > 0 for value in arguments[name])
                            for name in ("grid", "block")))
        incomplete = incomplete or not complete
        launch = launches[correlation]
        captured.append({"kernel_name": event["name"], "kernel_pid": event.get("pid"), "kernel_tid": event.get("tid"),
                         "kernel_ts_us": event["ts"], "kernel_duration_us": event["dur"], "kernel_args": arguments,
                         "launch_name": launch["name"], "launch_ts_us": launch["ts"],
                         "launch_duration_us": launch["dur"], "launch_args": launch["args"],
                         "correlation": correlation, "launch_geometry_complete": complete})
    if not captured:
        return {**result, "status": "partial" if result["unattributed_cuda_events"] else "unsupported",
                "reason": "kernel_launch_link_unavailable" if result["unattributed_cuda_events"] else "cuda_activity_absent"}
    for properties in properties_list:
        if type(properties) is dict and properties.get("id") == binding["gpu"]["device_index"]:
            if properties.get("name") is not None and properties["name"] != binding["gpu"]["name"]:
                raise ValueError("Kernel trace device properties differ from the bound GPU")
            if properties.get("uuid") is not None and properties["uuid"] != binding["gpu"]["uuid"]:
                raise ValueError("Kernel trace GPU UUID differs from the bound runtime GPU")
    return {**result, "status": "partial" if incomplete or result["unattributed_cuda_events"] else "supported",
            "reason": "launch_geometry_unavailable" if incomplete else (
                "unattributed_cuda_activity_present" if result["unattributed_cuda_events"] else None),
            "events": captured, "kernel_names": sorted({event["kernel_name"] for event in captured}),
            "kernel_activity_observed": True}


def validate_kernel_activity(record, source, worker, root=None):
    policy = worker.get("kernel_activity_policy", kernel_activity_policy())
    if type(policy) is not dict or canonical_sha(policy) != canonical_sha(kernel_activity_policy(policy.get("enabled"))):
        raise ValueError("Kernel activity policy exceeds or changes the predeclared shadow budget")
    activity = record.get("kernel_activity")
    if activity is None:
        if policy["enabled"]:
            raise ValueError("Opt-in mirror omitted its kernel activity diagnostic")
        return {"status": "disabled", "calls": [], "kernel_activity_observed": False}
    if type(activity) is not dict or activity.get("schema") != KERNEL_SCHEMA or activity.get("policy") != policy:
        raise ValueError("Kernel activity schema/policy differs from its worker")
    calls = activity.get("calls")
    if type(calls) is not list or len(calls) > MAX_KERNEL_CALLS:
        raise ValueError("Kernel activity shadow-call budget exceeded")
    if not calls:
        expected_status = ("disabled" if not policy["enabled"] else
                           "reference_not_profiled" if source["role"] == "reference" else "budget_exhausted")
        if activity.get("status") != expected_status:
            raise ValueError("Empty kernel activity diagnostic has no explicit disabled/budget state")
        return {"status": activity["status"], "calls": [], "kernel_activity_observed": False}
    if not policy["enabled"] or source["role"] != "trial" or len(calls) != record["repetitions"] * 2:
        raise ValueError("Kernel activity lacks its bounded B1/padded repeat coverage")
    base = kernel_base_binding(worker, {name: source[name] for name in SOURCE_READ_FIELDS}, source,
                               record["source_snapshot_path"])
    summaries = []
    stopped = False
    for ordinal, call in enumerate(calls):
        if type(call) is not dict:
            raise ValueError("Kernel activity call schema differs")
        repeat_index, shape = ordinal // 2, "padded" if ordinal % 2 == 0 else "target"
        binding = kernel_call_binding(base, record, repeat_index, shape)
        if (call.get("binding") != binding or call.get("binding_sha256") != canonical_sha(binding)
                or call.get("annotation") != kernel_marker(binding) or call.get("observed_outside_capture") is not True
                or call.get("native_graph_profiled") is not False):
            raise ValueError("Kernel activity descriptor/request/GPU/real-call repeat identity differs")
        diagnostic = call.get("activity")
        if (type(diagnostic) is not dict or diagnostic.get("status") not in ("supported", "unsupported", "partial")
                or type(diagnostic.get("kernel_activity_observed")) is not bool
                or type(diagnostic.get("events")) is not list or type(diagnostic.get("kernel_names")) is not list
                or type(diagnostic.get("unattributed_cuda_events")) is not int or diagnostic["unattributed_cuda_events"] < 0
                or type(call.get("profile_entered")) is not bool or type(call.get("shadow_call_executed")) is not bool):
            raise ValueError("Kernel activity flags/status/schema differ")
        if stopped and (call["profile_entered"] or call["shadow_call_executed"] or call.get("trace") is not None):
            raise ValueError("Kernel profiling continued after an unsupported/partial activity result")
        stopped = stopped or diagnostic["status"] != "supported"
        trace_record = call.get("trace")
        parsed = None
        if trace_record is not None:
            if worker.get("kernel_profiler_sources", {}).get("status") != "available":
                raise ValueError("Kernel trace attribution lacks its installed profiler source binding")
            if (not call["profile_entered"] or not call["shadow_call_executed"] or type(trace_record) is not dict
                    or not first_layer.valid_sha(trace_record.get("sha256")) or type(trace_record.get("bytes")) is not int
                    or not 0 < trace_record["bytes"] <= MAX_KERNEL_TRACE_BYTES):
                raise ValueError("Kernel raw trace lacks bounded bytes/SHA provenance")
            expected_path = f"upstream-qkv-mirror-kernel-traces/{worker['service']}-dp{worker['dp_rank']}/{canonical_sha(binding)}.json"
            if trace_record.get("path") != expected_path:
                raise ValueError("Kernel raw trace path differs from its call identity")
            if root is not None:
                path = local_path(root, trace_record["path"])
                if (path.stat().st_size != trace_record["bytes"]
                        or file_sha(path) != trace_record["sha256"]):
                    raise ValueError("Kernel raw trace path/size/SHA differs from its call identity")
                try:
                    parsed = parse_kernel_trace(read(path), binding)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    parsed = {"status": "partial", "reason": "invalid_chrome_trace", "events": [],
                              "kernel_names": [], "kernel_activity_observed": False, "unattributed_cuda_events": 0}
                if parsed != call.get("activity"):
                    raise ValueError("Kernel event metadata differs from independently parsed raw activity")
        elif (diagnostic["status"] == "supported" or diagnostic["kernel_activity_observed"]
              or diagnostic["events"] or diagnostic["kernel_names"] or diagnostic["unattributed_cuda_events"]):
            raise ValueError("Kernel attribution has no retained raw profiler trace")
        if call["shadow_call_executed"]:
            if not call["profile_entered"]:
                raise ValueError("Shadow apply executed outside the declared profiling context")
            output = call.get("shadow_output")
            shape_record = record["rounds"][repeat_index][PRIMARY_OPERATOR][shape]
            rows = record["private_tensors"]["padded_input"]["shape"][0] if shape == "padded" else 1
            row = source["token_row_index"] if shape == "padded" else 0
            validate_layout(output, [rows, record["projection"]["weight_shape"][0]], worker["gpu"], row)
            live_layouts = [layout for inventory in (record["originals"], record["private_tensors"])
                            for layout in inventory.values() if layout is not None]
            if any(output["storage_ptr"] == layout["storage_ptr"] for layout in live_layouts):
                raise ValueError("Profiled shadow output aliases original/private live tensors")
            matched = output["matrix_sha256"] == shape_record["matrix_sha256"] and output["row_sha256"] == shape_record["row_sha256"]
            if call.get("shadow_output_matches_unprofiled_apply") is not matched:
                raise ValueError("Profiled shadow output was substituted or its baseline comparison differs")
        elif call.get("shadow_output") is not None or call.get("shadow_output_matches_unprofiled_apply") is not None:
            raise ValueError("Kernel diagnostic fabricated an unexecuted shadow output")
        if parsed is None:
            parsed = {"status": "partial" if trace_record is not None else call["activity"]["status"],
                      "reason": "archive_root_required_for_raw_trace_verification" if trace_record is not None else call["activity"]["reason"],
                      "events": [], "kernel_names": [], "kernel_activity_observed": False}
        summaries.append({"binding": binding, "activity": parsed,
                          "shadow_output_matches_unprofiled_apply": call.get("shadow_output_matches_unprofiled_apply")})
    expected_status = ("supported" if all(call["activity"]["status"] == "supported" for call in calls) else
                       "unsupported" if all(call["activity"]["status"] == "unsupported" for call in calls) else "partial")
    if activity.get("status") != expected_status:
        raise ValueError("Kernel aggregate status differs from its per-call diagnostics")
    status = ("supported" if all(call["activity"]["status"] == "supported" for call in summaries) else
              "unsupported" if all(call["activity"]["status"] == "unsupported" for call in summaries) else "partial")
    return {"status": status,
            "calls": summaries, "kernel_activity_observed": any(call["activity"]["kernel_activity_observed"] for call in summaries),
            "native_graph_kernel_attribution_proven": False, "gemv_gemm_dispatch_inferred_from_shape": False}


def validate_layout(layout, shape, gpu, row=None):
    if (layout["shape"] != shape or layout["dtype"] != "torch.bfloat16" or layout["gpu"] != gpu
            or any(type(size) is not int or size <= 0 for size in shape)
            or any(type(size) is not int for size in layout["shape"])
            or len(layout["stride"]) != len(shape)
            or any(type(stride) is not int or stride <= 0 for stride in layout["stride"])
            or type(layout["storage_offset"]) is not int or layout["storage_offset"] < 0
            or any(type(layout[name]) is not int or layout[name] <= 0 for name in ("data_ptr", "storage_ptr"))
            or not first_layer.valid_sha(layout["matrix_sha256"])):
        raise ValueError("Mirror shape/stride/dtype/device/storage/SHA binding differs")
    if len(shape) == 2 and layout["stride"] != [shape[1], 1]:
        raise ValueError("Mirror requires dense row-major BF16 matrices, not changed strides")
    if len(shape) == 1 and layout["stride"] != [1]:
        raise ValueError("Mirror requires an unchanged dense BF16 bias")
    if row is not None and (layout["token_row_index"] != row or type(layout["token_row_index"]) is not int
                            or not 0 <= row < shape[0] or not first_layer.valid_sha(layout["row_sha256"])):
        raise ValueError("Mirror target row binding invalid")
    if row is not None and shape[0] == 1 and layout["matrix_sha256"] != layout["row_sha256"]:
        raise ValueError("B1 matrix and its sole row SHA differ")


def validate_record(record, source, worker, root=None):
    first_layer.validate_worker(worker["first_layer_worker"])
    if (record["schema"] != SCHEMA or record["scientific_scope"] != SCIENTIFIC_SCOPE
            or worker["schema"] != SCHEMA or worker["scientific_scope"] != SCIENTIFIC_SCOPE
            or record["binding"] != {name: source[name] for name in BINDING_FIELDS}
            or any(record.get(name) is not value for name, value in CLAIMS.items())
            or any(worker.get(name) is not value for name, value in CLAIMS.items())
            or source["schema"] != first_layer.SCHEMA or source["scientific_scope"] != SCIENTIFIC_SCOPE
            or source["checkpoint"] != worker["first_layer_worker"]["checkpoint"]
            or source["model_class"] != worker["first_layer_worker"]["model_class"]
            or record["checkpoint"] != source["checkpoint"] or record["projection"] != source["projection"]
            or record["projection"] != worker["first_layer_worker"]["projection"]
            or record["source_sha256"] != worker["source_sha256"]
            or source["source_sha256"] != worker["first_layer_worker"]["source_sha256"]
            or (source["service"], source["dp_rank"]) != (worker["service"], worker["dp_rank"])
            or source["prompt_tokens"] != 149 or source["position"] not in (52, 53, 54)
            or source["sample_discarded"] is not False or source["forward_mode"] != "FULL"
            or source["read_execution"] != "graph_replay" or source["buffer_origin"] != "cuda_graph_capture"
            or type(record["repetitions"]) is not int or record["repetitions"] not in (2, 3)
            or record["repetitions"] != worker["repetitions"] or len(record["rounds"]) != record["repetitions"]
            or record["source_buffer_read_sha256"] != canonical_sha({name: source[name] for name in SOURCE_READ_FIELDS})
            or record["source_buffer_read_after_sha256"] != record["source_buffer_read_sha256"]
            or record["source_refresh_witness"] != source["refresh_witness"]
            or record["source_consumed_before_finalize"] is not True
            or record["observed_outside_capture"] is not True or record["model_forward_invoked_by_mirror"] is not False):
        raise ValueError("Mirror identity/prefix/source/refresh/consume/scope binding differs")
    witness = source["refresh_witness"]
    if (witness["kind"] != "cuda_graph_generation" or witness["capture_increment_calls"] != 1
            or type(witness["capture_increment_calls"]) is not int
            or type(witness["forward_sequence"]) is not int or witness["forward_sequence"] <= 0
            or type(witness["generation_before"]) is not int or type(witness["generation_after"]) is not int
            or not 0 <= witness["generation_before"] < 2**63 - 1
            or witness["generation_after"] != witness["generation_before"] + 1
            or witness["counter"] != {"shape": [], "dtype": "torch.int64", "device": "cuda"}):
        raise ValueError("Mirror source graph did not refresh on the actual replay")
    prefix = source["prompt_token_ids"] + source["output_prefix_token_ids"]
    if (len(source["prompt_token_ids"]) != 149 or len(source["output_prefix_token_ids"]) != source["position"]
            or source["context_token_ids"] != prefix or source["forward_consumed_prefix_tokens"] != len(prefix)
            or source["context_sha256"] != first_layer.head_hidden_probe.context_digest(prefix)
            or source["forward_token_id"] != prefix[-1] or source["forward_position"] != len(prefix) - 1):
        raise ValueError("Mirror actual complete prefix/token/position binding differs")
    if any(record[name] is not True for name in ("same_gpu_verified", "alias_checks_passed",
            "original_tensors_unchanged", "private_inputs_weights_unchanged")):
        raise ValueError("Mirror tensors/weights changed, aliased or crossed GPUs")
    flags = record["flags_before"]
    if (flags != record["flags_after"] or flags != worker["flags"]
            or any(flags[name] != value for name, value in source["matmul_flags"].items())):
        raise ValueError("Mirror matmul flags differ from actual source/current controls")
    gpu = worker["gpu"]
    if (not isinstance(gpu["uuid"], str) or not re.fullmatch(
            r"(?:GPU-)?(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})", gpu["uuid"])
            or type(gpu["device_index"]) is not int
            or gpu["device_index"] < 0 or gpu["device"] != f"cuda:{gpu['device_index']}"
            or "A800" not in gpu["name"] or record["gpu"] != gpu):
        raise ValueError("Mirror GPU identity unavailable or inconsistent")
    if (type(record["cuda_stream_before"]) is not int or record["cuda_stream_before"] < 0
            or record["cuda_stream_after"] != record["cuda_stream_before"]
            or record["full_matrices_archived"] is not False
            or record["backend_workspace_bytes_not_in_explicit_tensor_budget"] is not True):
        raise ValueError("Mirror CUDA stream/bounded allocation provenance differs")
    padded, token_row = source["num_tokens_padded"], source["token_row_index"]
    starts, row = source["query_start_loc"], source["row_index"]
    identifiers = source["batch_native_request_ids"]
    descriptor = source["forward_descriptor"]
    if (type(padded) is not int or padded not in first_layer.CAPTURES
            or type(row) is not int or not 0 <= row < len(identifiers)
            or source["batch_rows"] != len(identifiers) or type(source["batch_rows"]) is not int
            or identifiers[row] != source["native_request_id"] or len(set(identifiers)) != len(identifiers)
            or len(starts) != len(identifiers) + 1 or starts[0] != 0
            or any(type(value) is not int for value in starts)
            or any(right - left != 1 for left, right in zip(starts, starts[1:])) or starts[-1] > padded
            or type(token_row) is not int or token_row != starts[row + 1] - 1
            or not isinstance(descriptor, dict) or descriptor.get("num_tokens") != padded
            or type(descriptor.get("num_tokens")) is not int or source["capture_descriptor"] != descriptor
            or source["buffer_key"] != ["FULL", padded, descriptor]
            or source["buffer_hook_calls"] != {name: 1 for name in first_layer.BUFFER_NAMES}
            or source["current_forward_hook_calls"] != 0 or type(source["current_forward_hook_calls"]) is not int):
        raise ValueError("Mirror actual graph buffer/token row mapping differs")
    binding = source["input_binding"]
    if (set(binding) != {"origin", "hook_calls", *first_layer.INPUT_NAMES}
            or binding["origin"] != "cuda_graph_capture" or binding["hook_calls"] != 1
            or type(binding["hook_calls"]) is not int):
        raise ValueError("Mirror source captured model input binding missing")
    for name in first_layer.INPUT_NAMES:
        captured = binding[name]
        expected = source["forward_token_id"] if name == "input_ids" else source["forward_position"]
        supported_dtypes = ("torch.int32", "torch.int64") if name == "input_ids" else ("torch.int64",)
        if (captured["shape"] != [padded] or captured["dtype"] not in supported_dtypes or captured["device"] != "cuda"
                or type(captured["token_row_index"]) is not int or captured["token_row_index"] != token_row
                or type(captured["value"]) is not int or captured["value"] != expected):
            raise ValueError("Mirror source token/position clone is stale or row is unbound")
    output_width, input_width = record["projection"]["weight_shape"]
    token_id_bytes = 4 if binding["input_ids"]["dtype"] == "torch.int32" else 8
    source_minimum = padded * (2 * (2 * input_width + output_width) + token_id_bytes + 8) + 8
    if (type(source["retained_device_bytes"]) is not int
            or not source_minimum <= source["retained_device_bytes"] <= first_layer.MAX_DEVICE_BYTES):
        raise ValueError("Mirror source memory budget excludes actual token/position dtype or refresh counter")
    originals = record["originals"]
    private = record["private_tensors"]
    if set(originals) != {"qkv_input", "qkv_output", "weight", "bias"} or set(private) != {"padded_input", "target_input", "weight", "bias"}:
        raise ValueError("Mirror private/original tensor inventory differs")
    for name, width in (("qkv_input", input_width), ("qkv_output", output_width)):
        validate_layout(originals[name], [padded, width], gpu, token_row)
        if originals[name]["row_sha256"] != source["buffers"][name]["row_sha256"]:
            raise ValueError("Mirror original row does not match the actual QKV source buffer")
    validate_layout(private["padded_input"], [padded, input_width], gpu, token_row)
    validate_layout(private["target_input"], [1, input_width], gpu, 0)
    if (private["padded_input"]["matrix_sha256"] != originals["qkv_input"]["matrix_sha256"]
            or private["padded_input"]["row_sha256"] != private["target_input"]["row_sha256"]
            or private["target_input"]["row_sha256"] != originals["qkv_input"]["row_sha256"]):
        raise ValueError("Mirror B1/original padded inputs are not the same real BF16 target row")
    for inventory in (originals, private):
        validate_layout(inventory["weight"], [output_width, input_width], gpu)
        if inventory["weight"]["matrix_sha256"] != record["projection"]["weight_sha256"]:
            raise ValueError("Mirror QKV weight SHA differs")
        bias = inventory["bias"]
        if record["projection"]["bias"] is None:
            if bias is not None:
                raise ValueError("Mirror introduced a projection bias")
        else:
            validate_layout(bias, [output_width], gpu)
            if bias["matrix_sha256"] != record["projection"]["bias"]["sha256"]:
                raise ValueError("Mirror QKV bias SHA differs")
    storages = [tensor["storage_ptr"] for tensor in (*originals.values(), *private.values()) if tensor is not None]
    if len(set(storages)) != len(storages):
        raise ValueError("Mirror private tensors alias original buffers/weights or each other")
    summaries = []
    for index, round_record in enumerate(record["rounds"]):
        if round_record["repeat_index"] != index or type(round_record["repeat_index"]) is not int or set(round_record) != {"repeat_index", *OPERATORS}:
            raise ValueError("Mirror repetition order/operator coverage differs")
        output_storages = []
        summary = {}
        for operator in OPERATORS:
            outputs = round_record[operator]
            if set(outputs) != {"padded", "target"}:
                raise ValueError("Mirror operator lacks B1 or padded real execution")
            for name, rows, row in (("padded", padded, token_row), ("target", 1, 0)):
                output = outputs[name]
                validate_layout(output, [rows, output_width], gpu, row)
                if (len(output["values"]) != output_width or first_layer.bf16_digest(output["values"]) != output["row_sha256"]):
                    raise ValueError("Mirror output BF16 row/SHA differs")
                output_storages.append(output["storage_ptr"])
            summary[operator] = {name: {field: outputs[name][field] for field in ("matrix_sha256", "row_sha256", "values")}
                                 for name in ("padded", "target")}
        if len(set(output_storages)) != 4 or set(output_storages) & set(storages):
            raise ValueError("Mirror operator outputs alias private/original tensors")
        summaries.append(summary)
    if any(summary != summaries[0] for summary in summaries):
        raise ValueError("Mirror repetitions are inconsistent; shape mechanism unproven")
    budget = record["explicit_temporary_tensor_bytes"]
    minimum = 2 * (padded * input_width + input_width + output_width * input_width
                   + (0 if record["projection"]["bias"] is None else output_width)
                   + 2 * padded * output_width + 2 * output_width)
    if type(budget) is not int or not minimum <= budget <= MAX_DEVICE_BYTES:
        raise ValueError("Mirror explicit temporary tensor budget invalid")
    operator_results = {}
    for operator, outputs in summaries[0].items():
        changed = outputs["target"]["row_sha256"] != outputs["padded"]["row_sha256"]
        if padded == 1 and changed:
            raise ValueError("Equal B1 shapes differ; not an isolated batch shape mechanism")
        matched = (outputs["padded"]["matrix_sha256"] == originals["qkv_output"]["matrix_sha256"]
                   and outputs["padded"]["row_sha256"] == originals["qkv_output"]["row_sha256"])
        differences = [abs(left - right) for left, right in zip(outputs["target"]["values"], outputs["padded"]["values"])]
        operator_results[operator] = {
            "isolated_eager_shape_difference_observed": changed, "padded_mirror_matches_actual_QKV_output": matched,
            "B1_mirror_matches_actual_QKV_target_row": outputs["target"]["row_sha256"] == originals["qkv_output"]["row_sha256"],
            "current_affine_operator_shape_mechanism_verified": padded != 1 and changed and matched,
            "changed_elements": sum(value != 0 for value in differences), "max_abs_difference": max(differences)}
    return {"padded_tokens": padded, "target_row": token_row, "shape_changed": padded != 1,
            "kernel_activity": validate_kernel_activity(record, source, worker, root),
            "predeclared_primary_operator": PRIMARY_OPERATOR, "operator_results": operator_results,
            "apply_and_functional_linear_outputs_identical": summaries[0][OPERATORS[0]] == summaries[0][OPERATORS[1]],
            **operator_results[PRIMARY_OPERATOR],
            "full_matrix_comparisons_are_runtime_hash_witnesses": True,
            "archived_target_rows_independently_verified": True, **CLAIMS}


def analyze(root):
    root = Path(root)
    parent_result = first_layer.analyze(root)
    workers = {}
    expected = {f"upstream-qkv-mirror-worker-{service}-dp{rank}.json": (service, rank)
                for service in ("instance-1", "instance-2") for rank in (0, 1)}
    paths = sorted(root.glob("upstream-qkv-mirror-worker-*.json"))
    if {path.name for path in paths} != set(expected):
        raise ValueError("Mirror requires all four model/worker/GPU/source records")
    for path in paths:
        worker = read(path)
        identity = expected[path.name]
        base_path = root / f"upstream-first-layer-worker-{identity[0]}-dp{identity[1]}.json"
        if ((worker["service"], worker["dp_rank"]) != identity or worker["first_layer_worker"] != read(base_path)
                or worker["first_layer_worker_sha256"] != file_sha(base_path)
                or worker["operators"] != list(OPERATORS) or worker["layer_proxy_fields"] != ["weight", "bias"]):
            raise ValueError("Mirror worker/source metadata differs from the actual first-layer worker")
        for name, checksum in worker["source_sha256"].items():
            source = worker["sources"][name]
            if source["file_sha256"] != checksum or file_sha(local_path(root, source["copied_path"])) != checksum:
                raise ValueError("Mirror installed/observer operator source SHA differs")
        required = {"quant_method_apply", "quant_method_class", "linear_base", "linear_class", "functional_linear",
                    "cudagraph_wrapper", "first_layer_worker", "first_layer_probe", "mirror_worker", "mirror_probe"}
        if set(worker["source_sha256"]) != required or set(worker["sources"]) != required:
            raise ValueError("Mirror installed linear/base/apply/F.linear provenance incomplete")
        profiler_sources = worker.get("kernel_profiler_sources", {"status": "disabled", "sources": {}})
        if (type(profiler_sources) is not dict or profiler_sources.get("status") not in ("disabled", "unsupported", "available")
                or type(profiler_sources.get("sources")) is not dict
                or (profiler_sources["status"] == "available" and set(profiler_sources["sources"]) != {"profile", "record_function"})):
            raise ValueError("Kernel profiler installed-source provenance incomplete")
        for source in profiler_sources["sources"].values():
            if (not first_layer.valid_sha(source.get("file_sha256"))
                    or file_sha(local_path(root, source["copied_path"])) != source["file_sha256"]):
                raise ValueError("Kernel profiler installed source SHA differs")
        workers[identity] = worker
    baseline = next(iter(workers.values()))
    if any(worker["source_sha256"] != baseline["source_sha256"]
           or worker.get("kernel_activity_policy") != baseline.get("kernel_activity_policy")
           or worker["first_layer_worker"]["projection"] != baseline["first_layer_worker"]["projection"]
           for worker in workers.values()):
        raise ValueError("Four mirror workers have inconsistent source epochs or QKV weights")
    if (any(workers[("instance-1", rank)]["gpu"] != workers[("instance-2", rank)]["gpu"] for rank in (0, 1))
            or workers[("instance-1", 0)]["gpu"]["uuid"] == workers[("instance-1", 1)]["gpu"]["uuid"]):
        raise ValueError("Four mirror workers are not consistently bound to two actual GPUs")
    graph_sources = {path.name: read(path) for path in (root / "upstream-first-layer-snapshots").glob("*.json")
                     if read(path)["forward_mode"] == "FULL"}
    seen = set()
    observations = []
    kernel_counts = {identity: {"observations": 0, "call_slots": 0} for identity in workers}
    trace_paths = set()
    for path in sorted((root / "upstream-qkv-mirror-snapshots").glob("*.json")):
        record = read(path)
        source_path = local_path(root, record["source_snapshot_path"])
        if source_path.parent != root / "upstream-first-layer-snapshots" or source_path.name not in graph_sources:
            raise ValueError("Unbound mirror source snapshot path")
        if source_path.name in seen or record["source_snapshot_sha256"] != file_sha(source_path):
            raise ValueError("Duplicate mirror observation or source snapshot SHA differs")
        seen.add(source_path.name)
        source = graph_sources[source_path.name]
        identity = (source["service"], source["dp_rank"])
        result = validate_record(record, source, workers[identity], root)
        calls = record.get("kernel_activity", {}).get("calls", [])
        if calls:
            counts = kernel_counts[identity]
            counts["observations"] += 1
            counts["call_slots"] += len(calls)
            if counts["observations"] > MAX_KERNEL_OBSERVATIONS or counts["call_slots"] > MAX_KERNEL_CALLS:
                raise ValueError("Archive kernel profiling exceeds its per-worker observation/call budget")
        for call in calls:
            if call.get("trace") is not None:
                relative = call["trace"]["path"]
                if relative in trace_paths:
                    raise ValueError("Duplicate/reused kernel raw trace identity")
                trace_paths.add(relative)
        observations.append({"binding": record["binding"], **result})
    if not observations or seen != set(graph_sources):
        raise ValueError("Mirror coverage incomplete; require each actual in-scope graph snapshot")
    if {str(path.relative_to(root)) for path in (root / "upstream-qkv-mirror-kernel-traces").rglob("*.json")} != trace_paths:
        raise ValueError("Kernel raw trace inventory contains unbound or missing activity evidence")
    return {"schema": SCHEMA, "status": "bounded_qkv_mirror_observation_not_quality_fix",
            "scientific_scope": SCIENTIFIC_SCOPE, "parent_first_layer_analysis": parent_result,
            "observations": observations, "validated_graph_observations": len(observations),
            "kernel_activity_worker_budgets": {f"{service}-dp{rank}": counts for (service, rank), counts in kernel_counts.items()},
            "independently_verified_kernel_traces": len(trace_paths),
            "native_graph_kernel_attribution_proven": False, "gemv_gemm_dispatch_inferred_from_shape": False,
            "all_in_scope_graph_sources_mirrored": True,
            "any_current_affine_operator_shape_mechanism_verified": any(
                result["current_affine_operator_shape_mechanism_verified"] for result in observations),
            "all_trace_first_divergences_attributed": False, "current_step_only": True, **CLAIMS}


def main():
    parser = argparse.ArgumentParser(description="Offline P149/p52-54 QKV mirror verifier; no GPU launch")
    parser.add_argument("--verify", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.verify)
    with args.output.open("x") as destination:
        destination.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
