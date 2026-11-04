import hashlib
import importlib
import inspect
import json
import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

from upstream_qkv_mirror_probe import (
    BINDING_FIELDS, CLAIMS, MAX_DEVICE_BYTES, OPERATORS, SCHEMA, SCIENTIFIC_SCOPE,
    KERNEL_SCHEMA, MAX_KERNEL_OBSERVATIONS, MAX_KERNEL_TRACE_BYTES,
    canonical_sha, file_sha, kernel_activity_policy, kernel_base_binding,
    kernel_call_binding, kernel_marker, parse_kernel_trace, read, validate_record,
)


def require_compute():
    if (not os.environ.get("SLURM_JOB_ID", "").isdigit() or int(os.environ["SLURM_JOB_ID"]) <= 0
            or os.environ.get("DP_OPT_QKV_MIRROR_COMPUTE_CONFIRMED") != "1"):
        raise RuntimeError("QKV mirror is opt-in compute-only; never initialize on a login node")


def matmul_flags(backend):
    return {"allow_bf16_reduced_precision_reduction": backend.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "allow_bf16_reduced_precision_reduction_split_k": backend.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,
            "allow_tf32": backend.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": backend.get_float32_matmul_precision(),
            "deterministic_algorithms_enabled": backend.are_deterministic_algorithms_enabled()}


def gpu_identity(tensor, backend):
    try:
        device = tensor.device
        if device.type != "cuda" or type(device.index) is not int or backend.cuda.current_device() != device.index:
            raise RuntimeError("Mirror source/worker current CUDA device differs")
        properties = backend.cuda.get_device_properties(device)
        uuid = str(properties.uuid)
        if not re.fullmatch(r"(?:GPU-)?(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})", uuid) or "A800" not in properties.name:
            raise RuntimeError("Mirror requires the actual A800 GPU UUID")
        return {"device": str(device), "device_index": device.index, "uuid": uuid, "name": properties.name,
                "capability": [properties.major, properties.minor]}
    except AttributeError as error:
        raise RuntimeError("Installed CUDA UUID/device properties unsupported; mechanism unproven") from error


def tensor_sha(tensor, backend):
    content = tensor.detach().cpu().contiguous().view(backend.uint8).numpy().tobytes()
    return hashlib.sha256(content).hexdigest()


def tensor_layout(tensor, backend, gpu, row=None, values=False):
    if str(tensor.device) != gpu["device"] or str(tensor.dtype) != "torch.bfloat16":
        raise RuntimeError("Mirror tensor crossed GPUs or changed BF16 dtype")
    result = {"shape": list(tensor.shape), "stride": list(tensor.stride()), "dtype": str(tensor.dtype),
              "gpu": gpu, "storage_offset": tensor.storage_offset(), "data_ptr": tensor.data_ptr(),
              "storage_ptr": tensor.untyped_storage().data_ptr(), "matrix_sha256": tensor_sha(tensor, backend)}
    if row is not None:
        result.update({"token_row_index": row, "row_sha256": tensor_sha(tensor[row], backend)})
        if values:
            result["values"] = tensor[row].detach().cpu().float().tolist()
    return result


def reject_aliases(tensors):
    identities = [(str(tensor.device), tensor.untyped_storage().data_ptr()) for tensor in tensors if tensor is not None]
    if len(set(identities)) != len(identities):
        raise RuntimeError("Mirror tensors alias original buffers/weights or each other")


def kernel_diagnostic(status, reason):
    return {"status": status, "reason": reason, "events": [], "kernel_names": [],
            "kernel_activity_observed": False, "unattributed_cuda_events": 0}


def close_profiler(context):
    try:
        context.__exit__(None, None, None)
        return None
    except Exception as error:
        return type(error).__name__ + ": " + str(error)[:512]


def profile_shadow_apply(call, backend, root, binding, baseline, token_row, live_tensors=()):
    if backend.cuda.is_current_stream_capturing():
        raise RuntimeError("Shadow kernel profiling is prohibited inside CUDA Graph capture")
    record = {"binding": binding, "binding_sha256": canonical_sha(binding), "annotation": kernel_marker(binding),
              "observed_outside_capture": True, "native_graph_profiled": False,
              "profile_entered": False, "shadow_call_executed": False, "shadow_output": None,
              "shadow_output_matches_unprofiled_apply": None, "trace": None, "profiler_errors": [],
              "activity": kernel_diagnostic("unsupported", "profiler_api_unavailable")}
    namespace = getattr(backend, "profiler", None)
    profiler = None
    try:
        activities = namespace.ProfilerActivity
        if activities.CUDA not in namespace.supported_activities():
            record["activity"] = kernel_diagnostic("unsupported", "cuda_profiler_activity_unavailable")
            return record
        profiler = namespace.profile(activities=[activities.CPU, activities.CUDA], record_shapes=False,
                                     profile_memory=False, with_stack=False, with_flops=False)
        profiler.__enter__()
    except Exception as error:
        record["profiler_errors"].append(type(error).__name__ + ": " + str(error)[:512])
        if profiler is not None:
            cleanup = close_profiler(profiler)
            if cleanup:
                record["profiler_errors"].append(cleanup)
        return record
    record["profile_entered"] = True
    annotation = None
    try:
        annotation = namespace.record_function(record["annotation"])
        annotation.__enter__()
    except Exception as error:
        record["activity"] = kernel_diagnostic("partial", "profiler_annotation_unavailable")
        record["profiler_errors"].append(type(error).__name__ + ": " + str(error)[:512])
        if annotation is not None:
            cleanup = close_profiler(annotation)
            if cleanup:
                record["profiler_errors"].append(cleanup)
        cleanup = close_profiler(profiler)
        if cleanup:
            record["profiler_errors"].append(cleanup)
        return record
    try:
        if backend.cuda.is_current_stream_capturing():
            raise RuntimeError("Shadow kernel profiling entered CUDA Graph capture")
        shadow_output = call()
    except BaseException:
        close_profiler(annotation)
        close_profiler(profiler)
        raise
    record["shadow_call_executed"] = True
    annotation_error = close_profiler(annotation)
    profile_error = close_profiler(profiler)
    for error in (annotation_error, profile_error):
        if error:
            record["profiler_errors"].append(error)
    reject_aliases([*live_tensors, shadow_output])
    record["shadow_output"] = tensor_layout(shadow_output, backend, binding["gpu"], token_row)
    del shadow_output
    record["shadow_output_matches_unprofiled_apply"] = (
        record["shadow_output"]["matrix_sha256"] == baseline["matrix_sha256"]
        and record["shadow_output"]["row_sha256"] == baseline["row_sha256"])
    if annotation_error or profile_error:
        record["activity"] = kernel_diagnostic("partial", "profiler_stop_or_annotation_failed")
        return record
    directory = Path(root) / "upstream-qkv-mirror-kernel-traces" / f"{binding['service']}-dp{binding['dp_rank']}"
    temporary = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".shadow-trace-", suffix=".json", dir=directory, delete=False) as output:
            temporary = Path(output.name)
        profiler.export_chrome_trace(str(temporary))
        size = temporary.stat().st_size
        if not 0 < size <= MAX_KERNEL_TRACE_BYTES:
            record["activity"] = kernel_diagnostic("partial", "trace_byte_budget_exceeded")
            record["discarded_export_bytes"] = size
            return record
        target = directory / (record["binding_sha256"] + ".json")
        checksum = file_sha(temporary)
        os.link(temporary, target)
        record["trace"] = {"path": str(target.relative_to(Path(root))), "sha256": checksum, "bytes": size}
        try:
            trace = read(target)
        except (json.JSONDecodeError, UnicodeDecodeError):
            record["activity"] = kernel_diagnostic("partial", "invalid_chrome_trace")
        else:
            record["activity"] = parse_kernel_trace(trace, binding)
    except (OSError, RuntimeError, ValueError, AttributeError, TypeError) as error:
        record["profiler_errors"].append(type(error).__name__ + ": " + str(error)[:512])
        record["activity"] = kernel_diagnostic("partial", "profiler_export_or_activity_parse_failed")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return record


def run_mirror(source_input, source_output, projection, token_row, expected_projection, repetitions=3,
               backend=None, kernel_request=None):
    if backend is None:
        require_compute()
        import torch

        backend = torch
    if backend.cuda.is_current_stream_capturing():
        raise RuntimeError("Mirror CPU reads and affine calls must be outside CUDA capture")
    if type(repetitions) is not int or repetitions not in (2, 3):
        raise RuntimeError("Mirror requires exactly two or three repetitions")
    gpu = gpu_identity(source_input, backend)
    original_tensors = [source_input, source_output, projection.weight, projection.bias]
    if any(str(tensor.device) != gpu["device"] for tensor in original_tensors if tensor is not None):
        raise RuntimeError("Mirror source/input/QKV weights crossed GPUs")
    output_width, input_width = expected_projection["weight_shape"]
    padded = source_input.shape[0]
    if (type(token_row) is not int or not 0 <= token_row < padded or not 0 < padded <= 4096
            or list(source_input.shape) != [padded, input_width] or list(source_output.shape) != [padded, output_width]
            or list(projection.weight.shape) != [output_width, input_width]
            or any(list(tensor.stride()) != [tensor.shape[1], 1] for tensor in original_tensors[:3])
            or projection.bias is not None and (list(projection.bias.shape) != [output_width] or list(projection.bias.stride()) != [1])
            or type(projection.quant_method).__name__ != "UnquantizedLinearMethod"
            or projection.skip_bias_add is not False):
        raise RuntimeError("Unsupported installed BF16 dense QKV apply/shape/row/stride binding")
    explicit_bytes = 2 * (padded * input_width + input_width + output_width * input_width
                         + (0 if projection.bias is None else output_width) + 2 * padded * output_width + 2 * output_width)
    if explicit_bytes > MAX_DEVICE_BYTES:
        raise RuntimeError("Mirror explicit temporary GPU tensor budget exceeded")
    reject_aliases(original_tensors)
    flags = matmul_flags(backend)
    stream_before = backend.cuda.current_stream(source_input.device).cuda_stream
    originals = {"qkv_input": tensor_layout(source_input, backend, gpu, token_row),
                 "qkv_output": tensor_layout(source_output, backend, gpu, token_row),
                 "weight": tensor_layout(projection.weight, backend, gpu),
                 "bias": None if projection.bias is None else tensor_layout(projection.bias, backend, gpu)}
    if (originals["weight"]["matrix_sha256"] != expected_projection["weight_sha256"]
            or (originals["bias"] is None) != (expected_projection["bias"] is None)
            or originals["bias"] is not None and originals["bias"]["matrix_sha256"] != expected_projection["bias"]["sha256"]):
        raise RuntimeError("QKV weights/bias changed before mirror")
    padded_input = source_input.detach().clone(memory_format=backend.preserve_format)
    target_input = source_input.narrow(0, token_row, 1).detach().clone(memory_format=backend.preserve_format)
    weight = projection.weight.detach().clone(memory_format=backend.preserve_format)
    bias = None if projection.bias is None else projection.bias.detach().clone(memory_format=backend.preserve_format)
    private_tensors = [padded_input, target_input, weight, bias]
    reject_aliases([*original_tensors, *private_tensors])
    private = {"padded_input": tensor_layout(padded_input, backend, gpu, token_row),
               "target_input": tensor_layout(target_input, backend, gpu, 0),
               "weight": tensor_layout(weight, backend, gpu),
               "bias": None if bias is None else tensor_layout(bias, backend, gpu)}
    if (private["padded_input"]["matrix_sha256"] != originals["qkv_input"]["matrix_sha256"]
            or private["target_input"]["row_sha256"] != originals["qkv_input"]["row_sha256"]
            or private["weight"]["matrix_sha256"] != originals["weight"]["matrix_sha256"]
            or bias is not None and private["bias"]["matrix_sha256"] != originals["bias"]["matrix_sha256"]):
        raise RuntimeError("Mirror GPU clones did not preserve actual BF16 operands")
    layer_proxy = SimpleNamespace(weight=weight, bias=bias)
    rounds = []
    policy = kernel_activity_policy() if kernel_request is None else kernel_request["policy"]
    if (type(policy) is not dict or canonical_sha(policy) != canonical_sha(kernel_activity_policy(policy.get("enabled")))
            or kernel_request is not None and type(kernel_request["eligible"]) is not bool):
        raise RuntimeError("Kernel profile policy/eligibility differs from the predeclared bounded opt-in")
    eligible = kernel_request is not None and kernel_request["eligible"] and policy["enabled"]
    reference = kernel_request is not None and kernel_request["binding"]["role"] == "reference"
    if eligible and reference:
        raise RuntimeError("Kernel profile budget is reserved for the first in-scope trial, not reference outputs")
    activity = {"schema": KERNEL_SCHEMA, "policy": policy,
                "status": "pending" if eligible else ("disabled" if not policy["enabled"] else
                    "reference_not_profiled" if reference else "budget_exhausted"), "calls": []}
    profile_failure = None
    for index in range(repetitions):
        try:
            outputs = {
                OPERATORS[0]: {"padded": projection.quant_method.apply(layer_proxy, padded_input, bias),
                               "target": projection.quant_method.apply(layer_proxy, target_input, bias)},
                OPERATORS[1]: {"padded": backend.nn.functional.linear(padded_input, weight, bias),
                               "target": backend.nn.functional.linear(target_input, weight, bias)},
            }
        except (AttributeError, TypeError) as error:
            raise RuntimeError("Installed quant_method.apply requires unsupported layer proxy fields; fail closed") from error
        reject_aliases([*original_tensors, *private_tensors, *(tensor for group in outputs.values() for tensor in group.values())])
        round_record = {"repeat_index": index}
        for operator, tensors in outputs.items():
            round_record[operator] = {name: tensor_layout(tensor, backend, gpu, token_row if name == "padded" else 0, True)
                                      for name, tensor in tensors.items()}
        rounds.append(round_record)
        del outputs, tensors
        if (tensor_sha(padded_input, backend) != private["padded_input"]["matrix_sha256"]
                or tensor_sha(target_input, backend) != private["target_input"]["matrix_sha256"]
                or tensor_sha(weight, backend) != private["weight"]["matrix_sha256"]
                or bias is not None and tensor_sha(bias, backend) != private["bias"]["matrix_sha256"]):
            raise RuntimeError("Mirror operator mutated private input/weights/bias")
    if eligible:
        if not kernel_request.get("profiler_sources_available", True):
            profile_failure = kernel_diagnostic("unsupported", "profiler_source_binding_unavailable")
        mirror_record = {"private_tensors": private, "rounds": rounds,
                         "cuda_stream_before": stream_before, "flags_before": flags}
        for index, round_record in enumerate(rounds):
            for shape, operand, row in (("padded", padded_input, token_row), ("target", target_input, 0)):
                binding = kernel_call_binding(kernel_request["binding"], mirror_record, index, shape)
                if profile_failure is None:
                    captured = profile_shadow_apply(
                        lambda: projection.quant_method.apply(layer_proxy, operand, bias), backend,
                        kernel_request["root"], binding, round_record[OPERATORS[0]][shape], row,
                        [*original_tensors, *private_tensors])
                    if captured["activity"]["status"] != "supported":
                        profile_failure = captured["activity"]
                else:
                    captured = {"binding": binding, "binding_sha256": canonical_sha(binding), "annotation": kernel_marker(binding),
                                "observed_outside_capture": True, "native_graph_profiled": False,
                                "profile_entered": False, "shadow_call_executed": False, "shadow_output": None,
                                "shadow_output_matches_unprofiled_apply": None, "trace": None, "profiler_errors": [],
                                "activity": kernel_diagnostic("unsupported", profile_failure["reason"] if not activity["calls"]
                                    else "profiling_disabled_after_partial_or_unsupported_activity")}
                activity["calls"].append(captured)
        if (tensor_sha(padded_input, backend) != private["padded_input"]["matrix_sha256"]
                or tensor_sha(target_input, backend) != private["target_input"]["matrix_sha256"]
                or tensor_sha(weight, backend) != private["weight"]["matrix_sha256"]
                or bias is not None and tensor_sha(bias, backend) != private["bias"]["matrix_sha256"]):
            raise RuntimeError("Profiled shadow apply mutated private input/weights/bias")
    for name, tensor in zip(("qkv_input", "qkv_output", "weight", "bias"), original_tensors):
        if tensor is not None and tensor_sha(tensor, backend) != originals[name]["matrix_sha256"]:
            raise RuntimeError("Original source buffer/model weights changed during mirror")
    after_flags = matmul_flags(backend)
    stream_after = backend.cuda.current_stream(source_input.device).cuda_stream
    if flags != after_flags or stream_before != stream_after:
        raise RuntimeError("Mirror flags/CUDA stream changed during repetitions")
    if eligible:
        activity["status"] = ("supported" if profile_failure is None else
                              "unsupported" if all(call["activity"]["status"] == "unsupported" for call in activity["calls"])
                              else "partial")
    return {"gpu": gpu, "originals": originals, "private_tensors": private, "rounds": rounds,
            "repetitions": repetitions, "explicit_temporary_tensor_bytes": explicit_bytes,
            "flags_before": flags, "flags_after": after_flags, "cuda_stream_before": stream_before,
            "cuda_stream_after": stream_after, "same_gpu_verified": True, "alias_checks_passed": True,
            "kernel_activity": activity,
            "original_tensors_unchanged": True, "private_inputs_weights_unchanged": True,
            "model_forward_invoked_by_mirror": False, "observed_outside_capture": True,
            "backend_workspace_bytes_not_in_explicit_tensor_budget": True, "full_matrices_archived": False, **CLAIMS}


def copy_source(symbol, directory, root, source_path=None):
    qualified_name = None if isinstance(symbol, Path) else symbol.__module__ + "." + getattr(symbol, "__qualname__", symbol.__name__)
    if source_path is None:
        if isinstance(symbol, Path):
            source_path = symbol
        else:
            try:
                source_path = inspect.getsourcefile(symbol)
            except TypeError:
                source_path = None
            if source_path is None:
                components = symbol.__module__.split(".")
                for count in range(len(components), 0, -1):
                    try:
                        module = importlib.import_module(".".join(components[:count]))
                    except ImportError:
                        continue
                    candidate = getattr(module, "__file__", None)
                    if candidate and Path(candidate).suffix not in (".py", ".pyc"):
                        source_path = candidate
                        break
    if source_path is None:
        raise RuntimeError("Installed operator source/binding file unavailable; mechanism unproven")
    source_path = Path(source_path).resolve(strict=True)
    checksum = file_sha(source_path)
    target = directory / (checksum + "-" + source_path.name)
    if not target.exists():
        with source_path.open("rb") as source, target.open("xb") as output:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                output.write(chunk)
    if file_sha(source_path) != checksum or file_sha(target) != checksum:
        raise RuntimeError("Installed operator source changed while copying")
    try:
        symbol_sha = hashlib.sha256(inspect.getsource(symbol).encode()).hexdigest() if qualified_name else None
    except (OSError, TypeError):
        symbol_sha = None
    return {"qualified_name": qualified_name, "installed_path": str(source_path), "file_sha256": checksum,
            "symbol_sha256": symbol_sha, "copied_path": str(target.relative_to(root)),
            "file_is_compiled_binding_not_specific_kernel": source_path.suffix != ".py"}


class MirrorObserver:
    def __init__(self, worker, projection, root, backend, parent):
        self.worker = worker
        self.projection = projection
        self.root = Path(root)
        self.backend = backend
        self.parent = parent
        self.metadata = None
        self.pending = None
        self.handles = []
        self.observations = 0
        self.kernel_observations = 0

    def initialize(self):
        require_compute()
        service = os.environ["DP_OPT_SERVICE"]
        rank = self.worker.parallel_config.data_parallel_rank
        if service not in ("instance-1", "instance-2") or type(rank) is not int or rank not in (0, 1):
            raise RuntimeError("Unsupported mirror worker identity")
        identity = f"{service}-dp{rank}"
        base_path = self.root / f"upstream-first-layer-worker-{identity}.json"
        base = read(base_path)
        self.parent.validate_worker(base)
        repetitions = int(os.environ.get("DP_OPT_QKV_MIRROR_REPETITIONS", "3"))
        if repetitions not in (2, 3):
            raise RuntimeError("Mirror repetitions must be 2 or 3")
        kernel_enabled = os.environ.get("DP_OPT_QKV_MIRROR_KERNEL_ACTIVITY", "0")
        if kernel_enabled not in ("0", "1"):
            raise RuntimeError("Optional kernel activity requires explicit 0 or 1")
        policy = kernel_activity_policy(kernel_enabled == "1")
        directory = self.root / "upstream-qkv-mirror-sources" / identity
        directory.mkdir(parents=True)
        linear_bases = {base.__name__: base for base in type(self.projection).__mro__}
        if "LinearBase" not in linear_bases or type(self.projection.quant_method).__name__ != "UnquantizedLinearMethod":
            raise RuntimeError("Unsupported installed LinearBase/unquantized QKV projection")
        try:
            graph_type = importlib.import_module("vllm.compilation.cuda_graph").CUDAGraphWrapper
        except (ImportError, AttributeError) as error:
            raise RuntimeError("Installed CUDAGraphWrapper source unavailable; mechanism unproven") from error
        if not isinstance(self.worker.model_runner.model, graph_type):
            raise RuntimeError("Installed runner.model is not the evidenced CUDAGraphWrapper")
        symbols = {"quant_method_apply": self.projection.quant_method.apply, "quant_method_class": type(self.projection.quant_method),
                   "linear_base": linear_bases["LinearBase"], "linear_class": type(self.projection),
                   "functional_linear": self.backend.nn.functional.linear,
                   "cudagraph_wrapper": graph_type,
                   "first_layer_worker": Path(self.parent.__file__),
                   "first_layer_probe": Path(self.parent.__file__).with_name("upstream_first_layer_probe.py"),
                   "mirror_worker": Path(__file__), "mirror_probe": Path(__file__).with_name("upstream_qkv_mirror_probe.py")}
        sources = {name: copy_source(symbol, directory, self.root) for name, symbol in symbols.items()}
        profiler_sources = {"status": "disabled", "sources": {}}
        if policy["enabled"]:
            profiler_sources["status"] = "unsupported"
            try:
                profiler_sources["sources"] = {
                    name: copy_source(getattr(self.backend.profiler, name), directory, self.root)
                    for name in ("profile", "record_function")}
                profiler_sources["status"] = "available"
            except (AttributeError, OSError, RuntimeError, TypeError) as error:
                profiler_sources["reason"] = type(error).__name__ + ": " + str(error)[:512]
        self.metadata = {"schema": SCHEMA, "service": service, "dp_rank": rank, "first_layer_worker": base,
            "first_layer_worker_sha256": file_sha(base_path), "scientific_scope": SCIENTIFIC_SCOPE,
            "gpu": gpu_identity(self.projection.weight, self.backend), "flags": matmul_flags(self.backend),
            "repetitions": repetitions, "operators": list(OPERATORS), "layer_proxy_fields": ["weight", "bias"],
            "kernel_activity_policy": policy, "process_id": os.getpid(),
            "kernel_profiler_sources": profiler_sources,
            "sources": sources, "source_sha256": {name: source["file_sha256"] for name, source in sources.items()},
            "slurm_job_id": int(os.environ["SLURM_JOB_ID"]), "compute_hostname": os.uname().nodename,
            "runner_model_is_cudagraph_wrapper": True, **CLAIMS}
        with (self.root / f"upstream-qkv-mirror-worker-{identity}.json").open("x") as output:
            output.write(json.dumps(self.metadata, indent=2, allow_nan=False) + "\n")
        (self.root / "upstream-qkv-mirror-snapshots").mkdir(exist_ok=True)

    def before(self, module, inputs, output):
        probe = getattr(self.worker, "_first_layer_probe", None)
        frame = None if probe is None else probe.frame
        if frame is None or not frame["pending"]:
            return
        if self.backend.cuda.is_current_stream_capturing():
            raise RuntimeError("Mirror post-logits observer unexpectedly entered capture")
        if self.pending is not None:
            raise RuntimeError("Previous mirror frame not finalized; no GPU history may be retained")
        identifiers = list(self.worker.model_runner.input_batch.req_ids)
        if inputs[1].shape[0] != len(identifiers) or output.shape[0] != len(identifiers):
            raise RuntimeError("Mirror logits/request rows differ")
        if frame["mode"] != "FULL":
            return
        selected = [(row, native, self.worker.model_runner.requests[native]) for row, native in enumerate(identifiers)
                    if self.parent.selected_position(self.worker.model_runner.requests[native]) is not None]
        if not selected:
            return
        if self.metadata is None:
            raise RuntimeError("Mirror target forward observed during initialization before metadata binding")
        pending = []
        try:
            for row, native, state in selected:
                before = probe.read(row, state)
                if before is None:
                    continue
                if identifiers.count(native) != 1:
                    raise RuntimeError("Mirror native request identity is duplicated")
                group = probe.graphs[self.parent.buffer_key(frame)]
                reference = native.startswith("reference-")
                phase = {} if reference else read(self.root / "phase.json")
                role, repeat = ("reference", None) if reference else ("trial", phase["repeat"])
                position = len(state.output_token_ids)
                tag = hashlib.sha256(native.encode()).hexdigest()[:12]
                filename = f"{role}-r{repeat}-dp{self.metadata['dp_rank']}-p{position}-{tag}.json"
                request = {"service": self.metadata["service"], "dp_rank": self.metadata["dp_rank"],
                           "native_request_id": native, "role": role, "repeat": repeat, "layout": phase.get("layout"),
                           "position": position, "prompt_token_ids": list(state.prompt_token_ids),
                           "output_prefix_token_ids": list(state.output_token_ids), "row_index": row,
                           "batch_native_request_ids": identifiers}
                policy = self.metadata["kernel_activity_policy"]
                eligible = policy["enabled"] and role == "trial" and self.kernel_observations < MAX_KERNEL_OBSERVATIONS
                if eligible:
                    self.kernel_observations += 1
                kernel_request = {"root": self.root, "policy": policy, "eligible": eligible,
                    "profiler_sources_available": self.metadata["kernel_profiler_sources"]["status"] == "available",
                    "binding": kernel_base_binding(self.metadata, before, request, "upstream-first-layer-snapshots/" + filename)}
                mirrored = run_mirror(group["buffers"]["qkv_input"], group["buffers"]["qkv_output"], self.projection,
                    before["token_row_index"], self.metadata["first_layer_worker"]["projection"], self.metadata["repetitions"],
                    self.backend, kernel_request=kernel_request)
                after = probe.read(row, state)
                if canonical_sha(before) != canonical_sha(after):
                    raise RuntimeError("Mirror source prefix/buffers/graph generation changed during affine calls")
                pending.append({"source_snapshot_path": "upstream-first-layer-snapshots/" + filename,
                    "early_binding": {"service": self.metadata["service"], "dp_rank": self.metadata["dp_rank"],
                        "native_request_id": native, "role": role, "repeat": repeat, "layout": phase.get("layout"),
                        "position": position, "prompt_token_ids": list(state.prompt_token_ids),
                        "output_prefix_token_ids": list(state.output_token_ids), "row_index": row,
                        "batch_native_request_ids": identifiers},
                    "source_buffer_read_sha256": canonical_sha(before), "source_buffer_read_after_sha256": canonical_sha(after),
                    "source_refresh_witness": before["refresh_witness"], **mirrored})
            self.pending = pending or None
        except BaseException:
            self.pending = None
            raise

    def after(self, module, inputs, output):
        if self.pending is None:
            return
        try:
            probe = self.worker._first_layer_probe
            if probe.frame is not None or probe.eager is not None:
                raise RuntimeError("Mirror finalizer ran before parent post-logits consume/clear")
            for item in self.pending:
                path = self.root / item["source_snapshot_path"]
                source = read(path)
                if any(source[name] != value for name, value in item.pop("early_binding").items()):
                    raise RuntimeError("Mirror pre-consume request/prefix differs from parent same-forward snapshot")
                record = {**item, "schema": SCHEMA, "scientific_scope": SCIENTIFIC_SCOPE,
                    "binding": {name: source[name] for name in BINDING_FIELDS}, "checkpoint": source["checkpoint"],
                    "projection": source["projection"], "source_sha256": self.metadata["source_sha256"],
                    "source_snapshot_sha256": file_sha(path), "source_consumed_before_finalize": True}
                self.observations += 1
                if self.observations > 64:
                    raise RuntimeError("Bounded mirror observation count exceeded")
                target = self.root / "upstream-qkv-mirror-snapshots" / path.name
                with target.open("x") as destination:
                    destination.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
                validate_record(record, source, self.metadata, self.root)
        finally:
            self.pending = None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending = None


def _initialize_worker(self, kv_cache_config):
    require_compute()
    import torch
    import upstream_first_layer_worker as parent

    model = self.model_runner.get_model()
    projection = model.model.layers[0].self_attn.qkv_proj
    observer = MirrorObserver(self, projection, Path(os.environ["DP_OPT_EVIDENCE_ROOT"]), torch, parent)
    observer.handles.append(model.logits_processor.register_forward_hook(observer.before))
    try:
        result = super(globals()["UpstreamQKVMirrorVerificationWorker"], self).initialize_from_config(kv_cache_config)
        if self.model_runner.get_model().logits_processor is not model.logits_processor:
            raise RuntimeError("Installed logits processor changed during parent initialization")
        observer.initialize()
        observer.handles.append(model.logits_processor.register_forward_hook(observer.after))
        self._qkv_mirror_observer = observer
        return result
    except BaseException:
        observer.close()
        raise


def __getattr__(name):
    if name != "UpstreamQKVMirrorVerificationWorker":
        raise AttributeError(name)
    require_compute()
    from upstream_first_layer_worker import UpstreamFirstLayerVerificationWorker

    worker_type = type(name, (UpstreamFirstLayerVerificationWorker,), {"__module__": __name__,
        "__qualname__": name, "initialize_from_config": _initialize_worker})
    globals()[name] = worker_type
    return worker_type
