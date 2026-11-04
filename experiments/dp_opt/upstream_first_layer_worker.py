import functools
import hashlib
import inspect
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.distributed.parallel_state import get_ep_group
from vllm.forward_context import get_forward_context

from head_hidden_worker import HeadHiddenVerificationWorker
from upstream_first_layer_probe import (
    BUFFER_NAMES, CAPTURES, INPUT_NAMES, MAX_DEVICE_BYTES, SCHEMA, SCIENTIFIC_SCOPE, bf16_digest,
    selected_position, validate_projection, validate_worker,
)


def source_sha(subject):
    source = inspect.getsource(subject)
    return hashlib.sha256(source.encode()).hexdigest()


def tensor_record(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return {"dtype": str(tensor.dtype), "device": tensor.device.type,
            "shape": list(tensor.shape), "sha256": hashlib.sha256(raw).hexdigest()}


def projection_record(projection):
    weight = tensor_record(projection.weight)
    record = {"weight_dtype": weight["dtype"], "weight_device": weight["device"],
              "weight_shape": weight["shape"], "weight_sha256": weight["sha256"],
              "quant_method": type(projection.quant_method).__name__,
              "skip_bias_add": projection.skip_bias_add,
              "bias": None if projection.bias is None else tensor_record(projection.bias)}
    validate_projection(record)
    return record


def execution_metadata(context, padded):
    mode = context.cudagraph_runtime_mode.name
    descriptor = context.batch_descriptor
    if mode not in ("NONE", "FULL") or type(padded) is not int or not 0 < padded <= 4096:
        raise RuntimeError("Unsupported first-layer forward mode or padded shape")
    if descriptor is not None:
        if not is_dataclass(descriptor):
            raise RuntimeError("Unsupported installed forward batch descriptor")
        descriptor = asdict(descriptor)
        if type(descriptor.get("num_tokens")) is not int or descriptor["num_tokens"] != padded:
            raise RuntimeError("Installed batch descriptor and actual input shape differ")
        json.dumps(descriptor, allow_nan=False)
    if mode == "FULL" and (descriptor is None or padded not in CAPTURES):
        raise RuntimeError("Full graph replay requires a known padded capture descriptor")
    return {"mode": mode, "padded": padded, "descriptor": descriptor}


def buffer_key(metadata):
    return (metadata["mode"], metadata["padded"],
            json.dumps(metadata["descriptor"], sort_keys=True, separators=(",", ":"), allow_nan=False))


def validate_inputs(input_ids, positions):
    if (input_ids is None or positions is None or len(input_ids.shape) != 1
            or positions.shape != input_ids.shape or input_ids.device.type != "cuda"
            or positions.device != input_ids.device or str(input_ids.dtype) not in ("torch.int32", "torch.int64")
            or str(positions.dtype) != "torch.int64"):
        raise RuntimeError("Unsupported first-layer actual token/position input mapping")


class FirstLayerBuffers:
    def __init__(self, runner, layer, projection, weights, context_reader=get_forward_context):
        self.runner = runner
        self.layer = layer
        self.projection = projection
        self.weights = weights
        self.context_reader = context_reader
        self.graphs = {}
        self.eager = None
        self.frame = None
        self.sequence = 0
        self.handles = []
        self.model = runner.get_model()
        self.model_signature = inspect.signature(self.model.forward)
        if not {"input_ids", "positions", "inputs_embeds", "intermediate_tensors"} <= set(self.model_signature.parameters):
            raise RuntimeError("Unsupported installed raw model input signature")
        self.original_forward = runner._model_forward
        self.forward_signature = inspect.signature(self.original_forward)
        if not {"input_ids", "positions", "inputs_embeds", "intermediate_tensors"} <= set(self.forward_signature.parameters):
            raise RuntimeError("Unsupported installed _model_forward signature")
        self.layer_signature = inspect.signature(layer.forward)
        if not {"hidden_states", "residual", "positions"} <= set(self.layer_signature.parameters):
            raise RuntimeError("Unsupported installed first decoder layer signature")
        self.original_warmup = runner._warmup_and_capture
        self.warmup_signature = inspect.signature(self.original_warmup)
        if not {"desc", "cudagraph_runtime_mode"} <= set(self.warmup_signature.parameters):
            raise RuntimeError("Unsupported installed _warmup_and_capture signature")

    @property
    def device_bytes(self):
        groups = list(self.graphs.values()) + ([self.eager] if self.eager is not None else [])
        return sum(tensor.numel() * tensor.element_size() for group in groups
                   for tensor in (list(group["buffers"].values()) + list(group["inputs"].values())
                                  + ([group["counter"]] if group["counter"] is not None else [])))

    def reserve(self, size):
        if self.device_bytes + size > MAX_DEVICE_BYTES:
            raise RuntimeError("First-layer device buffer budget exceeded")

    def install(self):
        self.handles.append(self.model.register_forward_pre_hook(self.model_input, with_kwargs=True))
        self.handles.append(self.layer.register_forward_pre_hook(self.layer_input, with_kwargs=True))
        self.handles.append(self.projection.register_forward_pre_hook(self.qkv_input))
        self.handles.append(self.projection.register_forward_hook(self.qkv_output))

        @functools.wraps(self.original_forward)
        def bound_forward(*args, **kwargs):
            return self.forward(*args, **kwargs)

        self.runner._model_forward = bound_forward

        @functools.wraps(self.original_warmup)
        def bound_warmup(*args, **kwargs):
            arguments = self.warmup_signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            mode = arguments.arguments["cudagraph_runtime_mode"]
            if mode.name == "FULL":
                descriptor = arguments.arguments["desc"]
                metadata = execution_metadata(SimpleNamespace(cudagraph_runtime_mode=mode,
                                                               batch_descriptor=descriptor), descriptor.num_tokens)
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Graph generation counter must be initialized outside capture")
                key = buffer_key(metadata)
                if key in self.graphs:
                    raise RuntimeError("Duplicate/ambiguous graph capture for descriptor buffer key")
                self.reserve(8)
                self.graphs[key] = {"buffers": {}, "inputs": {}, "calls": {},
                                    "descriptor": metadata["descriptor"], "origin": "cuda_graph_capture",
                                    "input_hook_calls": 0, "increment_calls": 0,
                                    "counter": torch.zeros((), dtype=torch.int64, device=self.projection.weight.device)}
            return self.original_warmup(*args, **kwargs)

        self.runner._warmup_and_capture = bound_warmup

    def close(self):
        self.runner._model_forward = self.original_forward
        self.runner._warmup_and_capture = self.original_warmup
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.frame = None
        self.eager = None
        self.graphs.clear()

    def forward(self, *args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Actual first-layer forward binding must be outside capture")
        if self.frame is not None and self.frame["pending"]:
            raise RuntimeError("Previous first-layer forward was not consumed by logits")
        arguments = self.forward_signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        input_ids, positions = arguments.arguments["input_ids"], arguments.arguments["positions"]
        validate_inputs(input_ids, positions)
        if (arguments.arguments["inputs_embeds"] is not None
                or arguments.arguments["intermediate_tensors"] is not None):
            raise RuntimeError("Unsupported first-layer actual token/position input mapping")
        metadata = execution_metadata(self.context_reader(), input_ids.shape[0])
        identifiers = list(self.runner.input_batch.req_ids)
        starts = [int(value) for value in self.runner.query_start_loc.np[:len(identifiers) + 1]]
        consumed = [int(value) for value in self.runner.optimistic_seq_lens_cpu[:len(identifiers)]]
        discarded = [bool(value) for value in self.runner.discard_request_mask.np[:len(identifiers)]]
        selected = any(selected_position(self.runner.requests[native]) is not None
                       and not discarded[index]
                       for index, native in enumerate(identifiers))
        prefixes = {native: (list(self.runner.requests[native].prompt_token_ids),
                            list(self.runner.requests[native].output_token_ids))
                    for native in identifiers if selected_position(self.runner.requests[native]) is not None}
        self.sequence += 1
        self.eager = None
        self.frame = {**metadata, "sequence": self.sequence, "input_ids": input_ids,
                      "positions": positions, "identifiers": identifiers, "selected": selected,
                      "query_start_loc": starts, "consumed": consumed, "discarded": discarded,
                      "prefixes": prefixes,
                      "hook_calls": 0, "generation_before": None, "pending": True}
        try:
            if metadata["mode"] == "FULL" and selected:
                group = self.graphs.get(buffer_key(metadata))
                if (group is None or set(group["buffers"]) != set(BUFFER_NAMES)
                        or set(group["inputs"]) != set(INPUT_NAMES) or group["input_hook_calls"] != 1
                        or group["increment_calls"] != 1):
                    raise RuntimeError("Missing captured graph buffers or refresh witness")
                self.frame["generation_before"] = int(group["counter"].detach().cpu().item())
            result = self.original_forward(*args, **kwargs)
            if metadata["mode"] == "FULL" and selected:
                group = self.graphs.get(buffer_key(metadata))
                if (group is None or group["descriptor"] != metadata["descriptor"]
                        or set(group["buffers"]) != set(BUFFER_NAMES) or self.frame["hook_calls"] != 0):
                    raise RuntimeError("Missing captured graph buffers or unexpected Python graph hooks")
            if not selected:
                self.frame["pending"] = False
                self.frame["input_ids"] = self.frame["positions"] = None
            return result
        except BaseException:
            self.frame = None
            self.eager = None
            raise

    def model_input(self, module, inputs, kwargs):
        arguments = self.model_signature.bind(*inputs, **kwargs)
        arguments.apply_defaults()
        input_ids, positions = arguments.arguments["input_ids"], arguments.arguments["positions"]
        validate_inputs(input_ids, positions)
        if (arguments.arguments["inputs_embeds"] is not None
                or arguments.arguments["intermediate_tensors"] is not None):
            raise RuntimeError("Unsupported raw model embedding/intermediate input mapping")
        metadata = execution_metadata(self.context_reader(), input_ids.shape[0])
        capturing = torch.cuda.is_current_stream_capturing()
        if metadata["mode"] == "FULL":
            if not capturing:
                if self.frame is not None and self.frame["pending"] and self.frame["selected"]:
                    raise RuntimeError("FULL mode Python hooks outside capture cannot stand for graph replay")
                return
            group = self.graphs.get(buffer_key(metadata))
            if group is None or group["counter"] is None:
                raise RuntimeError("Missing outside-capture graph generation initialization")
        else:
            if capturing:
                raise RuntimeError("Eager metadata observed in CUDA capture")
            if self.frame is None or not self.frame["pending"] or not self.frame["selected"]:
                return
            if any(self.frame[field] != metadata[field] for field in ("mode", "padded", "descriptor")):
                raise RuntimeError("Eager input hook differs from outer actual forward mode/shape")
            if self.eager is not None:
                raise RuntimeError("Duplicate raw model eager input hook")
            self.eager = {"buffers": {}, "inputs": {}, "calls": {}, "descriptor": metadata["descriptor"],
                          "origin": "eager_forward", "counter": None, "increment_calls": 0,
                          "input_hook_calls": 0, "sequence": self.frame["sequence"]}
            group = self.eager
        if group["input_hook_calls"] != 0:
            raise RuntimeError("Duplicate/ambiguous raw model input capture")
        self.reserve(input_ids.numel() * input_ids.element_size() + positions.numel() * positions.element_size())
        group["inputs"] = {"input_ids": input_ids.detach().clone(), "positions": positions.detach().clone()}
        group["input_hook_calls"] = 1

    def save(self, name, tensor):
        if (len(tensor.shape) != 2 or str(tensor.dtype) != "torch.bfloat16"
                or tensor.device.type != "cuda"):
            raise RuntimeError("First-layer hooks require two-dimensional CUDA BF16 buffers")
        metadata = execution_metadata(self.context_reader(), tensor.shape[0])
        capturing = torch.cuda.is_current_stream_capturing()
        if metadata["mode"] == "FULL":
            if not capturing:
                if self.frame is not None and self.frame["pending"] and self.frame["selected"]:
                    raise RuntimeError("FULL mode Python hooks outside capture cannot stand for graph replay")
                return
            group = self.graphs.get(buffer_key(metadata))
        else:
            if capturing:
                raise RuntimeError("Eager metadata observed in CUDA capture")
            if self.frame is None or not self.frame["pending"] or not self.frame["selected"]:
                return
            if any(self.frame[field] != metadata[field] for field in ("mode", "padded", "descriptor")):
                raise RuntimeError("Eager hook differs from outer actual forward mode/shape")
            group = self.eager
        if (group is None or name in group["buffers"] or group["input_hook_calls"] != 1
                or set(group["inputs"]) != set(INPUT_NAMES)):
            raise RuntimeError("Missing or repeated first-layer hook buffer")
        width = self.weights["weight_shape"][0 if name == "qkv_output" else 1]
        if tensor.shape[1] != width or tensor.element_size() != 2:
            raise RuntimeError("First-layer buffer shape differs from QKV weights")
        self.reserve(tensor.numel() * tensor.element_size())
        group["buffers"][name] = tensor.detach().clone()
        group["calls"][name] = 1
        if metadata["mode"] == "FULL" and name == "qkv_output":
            group["counter"].add_(1)
            group["increment_calls"] = 1
        if self.frame is not None and self.frame["pending"]:
            self.frame["hook_calls"] += 1

    def layer_input(self, module, inputs, kwargs):
        arguments = self.layer_signature.bind(*inputs, **kwargs)
        arguments.apply_defaults()
        if arguments.arguments["residual"] is not None:
            raise RuntimeError("First decoder input is not a pure embedding with empty residual")
        self.save("layer_input", arguments.arguments["hidden_states"])

    def qkv_input(self, module, inputs):
        if len(inputs) != 1:
            raise RuntimeError("Unsupported QKV projection input signature")
        self.save("qkv_input", inputs[0])

    def qkv_output(self, module, inputs, output):
        if not isinstance(output, tuple) or len(output) != 2 or output[1] is not None:
            raise RuntimeError("Unsupported QKV output/deferred bias mapping")
        self.save("qkv_output", output[0])

    def read(self, row, state):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("First-layer snapshot read must be outside capture")
        frame = self.frame
        if frame is None or not frame["pending"] or frame["identifiers"] != list(self.runner.input_batch.req_ids):
            raise RuntimeError("Missing forward binding or input batch changed before logits")
        if type(row) is not int or not 0 <= row < len(frame["identifiers"]):
            raise RuntimeError("Invalid first-layer request row index")
        consumed = int(self.runner.optimistic_seq_lens_cpu[row])
        if bool(self.runner.discard_request_mask.np[row]):
            return None
        native = frame["identifiers"][row]
        if (frame["discarded"][row] or frame["consumed"][row] != consumed
                or frame["prefixes"].get(native) != (list(state.prompt_token_ids), list(state.output_token_ids))):
            raise RuntimeError("Accepted first-layer prefix changed after actual forward binding")
        prefix = list(state.prompt_token_ids) + list(state.output_token_ids)
        if state.num_prompt_tokens != len(state.prompt_token_ids) or consumed != len(prefix):
            raise RuntimeError("First-layer forward did not consume complete accepted prefix")
        starts = [int(value) for value in self.runner.query_start_loc.np[:len(frame["identifiers"]) + 1]]
        if (len(starts) != len(frame["identifiers"]) + 1 or starts[0] != 0
                or starts != frame["query_start_loc"]
                or any(left >= right for left, right in zip(starts, starts[1:]))
                or starts[-1] > frame["padded"] or not 0 <= row < len(frame["identifiers"])):
            raise RuntimeError("Unsupported first-layer query_start_loc token row mapping")
        token_row = starts[row + 1] - 1
        forward_token = int(frame["input_ids"][token_row].detach().cpu().item())
        forward_position = int(frame["positions"][token_row].detach().cpu().item())
        if not prefix or forward_token != prefix[-1] or forward_position != len(prefix) - 1:
            raise RuntimeError("Actual forward token/position differs from complete prefix")
        group = self.graphs.get(buffer_key(frame)) if frame["mode"] == "FULL" else self.eager
        if (group is None or set(group["buffers"]) != set(BUFFER_NAMES)
                or set(group["inputs"]) != set(INPUT_NAMES) or group["input_hook_calls"] != 1):
            raise RuntimeError("Require three buffers from actual forward, not stale eager history")
        if frame["mode"] == "FULL" and (frame["hook_calls"] != 0
                or group["descriptor"] != frame["descriptor"]
                or any(right - left != 1 for left, right in zip(starts, starts[1:]))):
            raise RuntimeError("Graph capture/replay descriptor or decode rows differ")
        if frame["mode"] == "NONE" and (frame["hook_calls"] != len(BUFFER_NAMES)
                                        or group["sequence"] != frame["sequence"]):
            raise RuntimeError("Eager hook coverage differs")
        generation_after = None
        if frame["mode"] == "FULL":
            generation_after = int(group["counter"].detach().cpu().item())
            if (group["increment_calls"] != 1 or type(frame["generation_before"]) is not int
                    or not 0 <= frame["generation_before"] < 2**63 - 1
                    or generation_after != frame["generation_before"] + 1):
                raise RuntimeError("Graph refresh generation did not advance exactly once")
        input_binding = {"origin": group["origin"], "hook_calls": group["input_hook_calls"]}
        validate_inputs(group["inputs"]["input_ids"], group["inputs"]["positions"])
        for name, tensor in group["inputs"].items():
            if tuple(tensor.shape) != (frame["padded"],):
                raise RuntimeError("Captured token/position shape differs from actual forward")
            value = int(tensor[token_row].detach().cpu().item())
            if value != (forward_token if name == "input_ids" else forward_position):
                raise RuntimeError("Captured token/position input is stale or differs from actual forward")
            input_binding[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                                   "device": tensor.device.type, "token_row_index": token_row, "value": value}
        buffers = {}
        for name, tensor in group["buffers"].items():
            width = self.weights["weight_shape"][0 if name == "qkv_output" else 1]
            if tuple(tensor.shape) != (frame["padded"], width):
                raise RuntimeError("Captured buffer shape/index differs from actual forward")
            values = tensor[token_row].detach().cpu().float().tolist()
            buffers[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                             "device": tensor.device.type, "token_row_index": token_row,
                             "values": values, "row_sha256": bf16_digest(values)}
        return {"forward_mode": frame["mode"], "num_tokens_padded": frame["padded"],
                "forward_descriptor": frame["descriptor"], "query_start_loc": starts,
                "token_row_index": token_row, "forward_token_id": forward_token,
                "forward_position": forward_position,
                "buffer_key": [frame["mode"], frame["padded"], frame["descriptor"]],
                "buffer_origin": group["origin"], "buffer_hook_calls": group["calls"],
                "capture_descriptor": group["descriptor"] if frame["mode"] == "FULL" else None,
                "read_execution": "graph_replay" if frame["mode"] == "FULL" else "eager_forward",
                "current_forward_hook_calls": frame["hook_calls"], "retained_device_bytes": self.device_bytes,
                "input_binding": input_binding,
                "refresh_witness": {"kind": "cuda_graph_generation" if frame["mode"] == "FULL" else "eager_forward_sequence",
                    "forward_sequence": frame["sequence"], "generation_before": frame["generation_before"],
                    "generation_after": generation_after, "capture_increment_calls": group["increment_calls"],
                    "counter": None if group["counter"] is None else {
                        "shape": list(group["counter"].shape), "dtype": str(group["counter"].dtype),
                        "device": group["counter"].device.type}},
                "buffers": buffers}


class UpstreamFirstLayerVerificationWorker(HeadHiddenVerificationWorker):
    def initialize_from_config(self, kv_cache_config):
        model = self.model_runner.get_model()
        layer = model.model.layers[0]
        if model.model.start_layer != 0 or not hasattr(layer, "input_layernorm"):
            raise RuntimeError("Unsupported installed first decoder layer/embedding path")
        projection = layer.self_attn.qkv_proj
        compilation = self.vllm_config.compilation_config
        record = {"schema": SCHEMA, "service": os.environ["DP_OPT_SERVICE"],
                  "dp_rank": self.parallel_config.data_parallel_rank, "checkpoint": self.model_config.model,
                  "model_class": type(model).__module__ + "." + type(model).__qualname__,
                  "model_dtype": str(self.model_config.dtype), "attention_tp": self.parallel_config.tensor_parallel_size,
                  "dp": self.parallel_config.data_parallel_size, "ep": get_ep_group().world_size,
                  "pp": self.parallel_config.pipeline_parallel_size,
                  "ep_enabled": self.parallel_config.enable_expert_parallel, "compile_mode": int(compilation.mode),
                  "cudagraph_mode": compilation.cudagraph_mode.name,
                  "cudagraph_capture_sizes": list(compilation.cudagraph_capture_sizes),
                  "speculative": self.vllm_config.speculative_config is not None,
                  "async_scheduling": self.scheduler_config.async_scheduling,
                  "use_ubatching": self.parallel_config.use_ubatching,
                  "use_sequence_parallel_moe": self.parallel_config.use_sequence_parallel_moe,
                  "batch_invariant_enabled": os.environ.get("VLLM_BATCH_INVARIANT", "0") not in ("", "0"),
                  "layer_index": 0, "buffer_names": list(BUFFER_NAMES), "max_device_bytes": MAX_DEVICE_BYTES,
                  "scientific_scope": SCIENTIFIC_SCOPE,
                  "projection": projection_record(projection), "source_sha256": {
                      "runner": source_sha(type(self.model_runner)), "context": source_sha(get_forward_context),
                      "layer": source_sha(type(layer)), "qkv": source_sha(type(projection))},
                  "is_algorithm_benefit": False}
        validate_worker(record)
        probe = FirstLayerBuffers(self.model_runner, layer, projection, record["projection"])
        self._first_layer_probe = probe
        root = Path(os.environ["DP_OPT_EVIDENCE_ROOT"])
        try:
            probe.install()
            super().initialize_from_config(kv_cache_config)
            if projection_record(projection) != record["projection"]:
                raise RuntimeError("QKV weights changed during initialization/capture")
            with (root / f"upstream-first-layer-worker-{record['service']}-dp{record['dp_rank']}.json").open("x") as output:
                output.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
            directory = root / "upstream-first-layer-snapshots"
            directory.mkdir(exist_ok=True)

            def snapshot(module, inputs, output):
                frame = probe.frame
                if frame is None or not frame["pending"]:
                    return
                identifiers = list(self.model_runner.input_batch.req_ids)
                if inputs[1].shape[0] != len(identifiers) or output.shape[0] != len(identifiers):
                    raise RuntimeError("First-layer logits rows differ from actual request batch")
                try:
                    for row, native in enumerate(identifiers):
                        state = self.model_runner.requests[native]
                        if selected_position(state) is None:
                            continue
                        buffers = probe.read(row, state)
                        if buffers is None:
                            continue
                        if identifiers.count(native) != 1:
                            raise RuntimeError("Duplicate native request/logits row")
                        if projection_record(projection) != record["projection"]:
                            raise RuntimeError("QKV weights changed before target observation")
                        tag = hashlib.sha256(native.encode()).hexdigest()[:12]
                        reference = native.startswith("reference-")
                        phase = {} if reference else json.loads((root / "phase.json").read_text())
                        role, repeat = ("reference", None) if reference else ("trial", phase["repeat"])
                        position = len(state.output_token_ids)
                        head = json.loads((root / "head-hidden-snapshots" /
                            f"{role}-r{repeat}-dp{record['dp_rank']}-p{position}-{tag}.json").read_text())
                        binding = {name: head[name] for name in (
                            "service", "dp_rank", "native_request_id", "role", "repeat", "layout", "position",
                            "prompt_tokens", "prompt_token_ids", "output_prefix_token_ids", "context_token_ids",
                            "context_sha256", "forward_consumed_prefix_tokens", "sample_discarded", "batch_rows",
                            "batch_native_request_ids", "row_index")}
                        if (binding["native_request_id"] != native or binding["row_index"] != row
                                or binding["layout"] != phase.get("layout") or binding["position"] != position
                                or binding["service"] != record["service"] or binding["dp_rank"] != record["dp_rank"]
                                or binding["batch_native_request_ids"] != identifiers
                                or binding["prompt_token_ids"] != list(state.prompt_token_ids)
                                or binding["output_prefix_token_ids"] != list(state.output_token_ids)):
                            raise RuntimeError("First-layer observation differs from same-forward head identity")
                        observation = {**binding, **buffers, "schema": SCHEMA,
                            "checkpoint": record["checkpoint"], "model_class": record["model_class"],
                            "scientific_scope": SCIENTIFIC_SCOPE,
                            "projection": record["projection"], "source_sha256": record["source_sha256"],
                            "layer_index": 0, "first_layer_residual_is_none": True,
                            "matmul_flags": {"allow_bf16_reduced_precision_reduction":
                                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                                "allow_bf16_reduced_precision_reduction_split_k":
                                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k},
                            "observation_includes_CPU_synchronization": True, "is_algorithm_benefit": False}
                        with (directory / f"{role}-r{repeat}-dp{record['dp_rank']}-p{position}-{tag}.json").open("x") as destination:
                            destination.write(json.dumps(observation, indent=2, allow_nan=False) + "\n")
                finally:
                    probe.frame = None
                    probe.eager = None

            probe.handles.append(model.logits_processor.register_forward_hook(snapshot))
        except BaseException:
            probe.close()
            raise
