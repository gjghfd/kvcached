import copy
import hashlib
import json
import os
import struct
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

import upstream_first_layer_probe as first_layer
import upstream_qkv_mirror_probe as probe
import upstream_qkv_mirror_worker as worker_module


class FakeDevice:
    type = "cuda"

    def __init__(self, index=0):
        self.index = index

    def __str__(self):
        return f"cuda:{self.index}"


class FakeTensor:
    next_storage = 100
    live = weakref.WeakSet()
    cpu_reads = 0

    def __init__(self, values, backend, device=0, storage=None, offset=0):
        self.values = copy.deepcopy(values)
        self.backend = backend
        self.device = FakeDevice(device)
        self.dtype = "torch.bfloat16"
        self.shape = (len(values), len(values[0])) if values and isinstance(values[0], list) else (len(values),)
        self._stride = (self.shape[1], 1) if len(self.shape) == 2 else (1,)
        if storage is None:
            FakeTensor.next_storage += 100
            storage = FakeTensor.next_storage
        self.storage = storage
        self.offset = offset
        FakeTensor.live.add(self)

    def __getitem__(self, index):
        return FakeTensor(self.values[index], self.backend, self.device.index, self.storage, self.offset + index * self._stride[0])

    def narrow(self, dimension, start, length):
        assert dimension == 0
        return FakeTensor(self.values[start:start + length], self.backend, self.device.index,
                          self.storage, self.offset + start * self._stride[0])

    def detach(self):
        return self

    def clone(self, memory_format=None):
        if self.backend.alias_clone:
            return self
        result = FakeTensor(self.values, self.backend, self.device.index)
        if self.backend.bad_input_clone and result.shape == (1, 2):
            result.values[0][1] += .5
        return result

    def stride(self):
        return self._stride

    def storage_offset(self):
        return self.offset

    def data_ptr(self):
        return self.storage + 2 * self.offset

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.storage)

    def cpu(self):
        assert not self.backend.capturing
        FakeTensor.cpu_reads += 1
        return self

    def contiguous(self):
        return self

    def view(self, dtype):
        return self

    def numpy(self):
        values = [value for row in self.values for value in row] if len(self.shape) == 2 else self.values
        content = b"".join(struct.pack("<f", float(value))[2:] for value in values)
        return SimpleNamespace(tobytes=lambda: content)

    def float(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.values)


class FakeBackend:
    uint8 = "uint8"
    preserve_format = "preserve"

    def __init__(self, index=0, shape_effect=True):
        self.index = index
        self.shape_effect = shape_effect
        self.capturing = False
        self.alias_clone = False
        self.bad_input_clone = False
        self.alias_output = False
        self.inconsistent = False
        self.calls = 0
        self.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(
            allow_bf16_reduced_precision_reduction=False, allow_bf16_reduced_precision_reduction_split_k=True,
            allow_tf32=False)))
        self.nn = SimpleNamespace(functional=SimpleNamespace(linear=self.linear))
        self.cuda = SimpleNamespace(is_current_stream_capturing=lambda: self.capturing,
            current_device=lambda: self.index, get_device_properties=self.properties,
            current_stream=lambda device: SimpleNamespace(cuda_stream=0))

    def properties(self, device):
        return SimpleNamespace(uuid=f"GPU-aaaaaaaa-0000-0000-0000-{device.index + 1:012d}",
                               name="NVIDIA A800-SXM4-80GB", major=8, minor=0)

    def get_float32_matmul_precision(self):
        return "highest"

    def are_deterministic_algorithms_enabled(self):
        return False

    def linear(self, inputs, weight, bias):
        index = self.calls
        self.calls += 1
        if self.alias_output:
            return inputs
        values = [[sum(value * coefficient for value, coefficient in zip(row, coefficients))
                   + (0 if bias is None else bias.values[column])
                   for column, coefficients in enumerate(weight.values)] for row in inputs.values]
        if self.shape_effect and inputs.shape[0] == 1:
            values[0][-1] += .03125 * (1 + (index // 4 if self.inconsistent else 0))
        return FakeTensor(values, self, inputs.device.index)


class LinearBase:
    pass


class UnquantizedLinearMethod:
    def __init__(self, backend):
        self.backend = backend
        self.mutate_weight = False
        self.require_extra_field = False
        self.apply_shape_effect = None

    def apply(self, layer, inputs, bias):
        if self.require_extra_field:
            return layer.unsupported_private_field
        if self.mutate_weight:
            layer.weight.values[0][0] = 2.0
        result = self.backend.linear(inputs, layer.weight, bias)
        if self.apply_shape_effect is not None and inputs.shape[0] == 1:
            result.values[0][-1] += .03125 * (int(self.apply_shape_effect) - int(self.backend.shape_effect))
        return result


class QKVParallelLinear(LinearBase):
    def __init__(self, backend, bias=False):
        self.weight = FakeTensor([[1., 0.], [0., 1.]], backend, backend.index)
        self.bias = FakeTensor([.5, .5], backend, backend.index) if bias else None
        self.quant_method = UnquantizedLinearMethod(backend)
        self.skip_bias_add = False


def projection_record(projection, backend):
    return {"weight_dtype": "torch.bfloat16", "weight_device": "cuda", "weight_shape": [2, 2],
            "weight_sha256": worker_module.tensor_sha(projection.weight, backend),
            "quant_method": "UnquantizedLinearMethod", "skip_bias_add": False,
            "bias": None if projection.bias is None else {"shape": [2], "dtype": "torch.bfloat16", "device": "cuda",
                "sha256": worker_module.tensor_sha(projection.bias, backend)}}


def base_worker(projection, backend, service="instance-1", rank=0):
    return {"schema": first_layer.SCHEMA, "service": service, "dp_rank": rank, "checkpoint": "same-checkpoint",
            "model_class": "FakeModel", "model_dtype": "torch.bfloat16", "attention_tp": 1, "dp": 2, "ep": 2,
            "pp": 1, "ep_enabled": True, "compile_mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": list(first_layer.CAPTURES), "speculative": False, "async_scheduling": False,
            "use_ubatching": False, "use_sequence_parallel_moe": False, "batch_invariant_enabled": False,
            "layer_index": 0, "buffer_names": list(first_layer.BUFFER_NAMES), "max_device_bytes": first_layer.MAX_DEVICE_BYTES,
            "scientific_scope": first_layer.SCIENTIFIC_SCOPE, "projection": projection_record(projection, backend),
            "source_sha256": {name: "a" * 64 for name in ("runner", "context", "layer", "qkv")}, "is_algorithm_benefit": False}


def source_snapshot(base, padded=8, position=53, token_row=1, role="trial", input_ids_dtype="torch.int32"):
    native = "native-conv1_implementation-test"
    prompt, outputs = [8] * 149, [9] * position
    context = prompt + outputs
    identifiers = ["other0", native, "other2", "other3"] if padded == 8 else [native]
    snapshot = {"schema": first_layer.SCHEMA, "scientific_scope": first_layer.SCIENTIFIC_SCOPE,
        "service": base["service"], "dp_rank": base["dp_rank"], "native_request_id": native, "role": role,
        "repeat": 1 if role == "trial" else None, "layout": "corank0" if role == "trial" else None,
        "position": position, "prompt_tokens": 149, "prompt_token_ids": prompt, "output_prefix_token_ids": outputs,
        "context_token_ids": context, "context_sha256": first_layer.head_hidden_probe.context_digest(context),
        "forward_consumed_prefix_tokens": len(context), "sample_discarded": False, "batch_rows": len(identifiers),
        "batch_native_request_ids": identifiers, "row_index": token_row, "token_row_index": token_row,
        "checkpoint": base["checkpoint"], "model_class": base["model_class"], "projection": base["projection"],
        "source_sha256": base["source_sha256"], "num_tokens_padded": padded, "query_start_loc": list(range(len(identifiers) + 1)),
        "forward_token_id": context[-1], "forward_position": len(context) - 1, "forward_mode": "FULL",
        "forward_descriptor": {"num_tokens": padded}, "capture_descriptor": {"num_tokens": padded},
        "buffer_key": ["FULL", padded, {"num_tokens": padded}], "buffer_origin": "cuda_graph_capture",
        "buffer_hook_calls": {name: 1 for name in first_layer.BUFFER_NAMES}, "read_execution": "graph_replay",
        "current_forward_hook_calls": 0, "retained_device_bytes": padded * (24 if input_ids_dtype == "torch.int32" else 28) + 8,
        "layer_index": 0, "first_layer_residual_is_none": True,
        "observation_includes_CPU_synchronization": True, "is_algorithm_benefit": False,
        "matmul_flags": {"allow_bf16_reduced_precision_reduction": False,
                         "allow_bf16_reduced_precision_reduction_split_k": True},
        "refresh_witness": {"kind": "cuda_graph_generation", "forward_sequence": 10,
            "generation_before": 10, "generation_after": 11, "capture_increment_calls": 1,
            "counter": {"shape": [], "dtype": "torch.int64", "device": "cuda"}}, "buffers": {}}
    for name in first_layer.BUFFER_NAMES:
        snapshot["buffers"][name] = {"shape": [padded, 2], "dtype": "torch.bfloat16", "device": "cuda",
                                    "token_row_index": token_row, "values": [2., 3.], "row_sha256": first_layer.bf16_digest([2., 3.])}
    snapshot["input_binding"] = {"origin": "cuda_graph_capture", "hook_calls": 1,
        "input_ids": {"shape": [padded], "dtype": input_ids_dtype, "device": "cuda", "token_row_index": token_row, "value": context[-1]},
        "positions": {"shape": [padded], "dtype": "torch.int64", "device": "cuda", "token_row_index": token_row, "value": len(context) - 1}}
    return snapshot


def operands(backend, padded=8, bias=False):
    projection = QKVParallelLinear(backend, bias)
    values = [[1., 0.], [2., 3.], [4., 0.], [0., 0.]] + [[0., 0.]] * 4 if padded == 8 else [[2., 3.]]
    inputs = FakeTensor(values, backend, backend.index)
    raw_values = [[value + (.5 if bias else 0) for value in row] for row in values]
    actual = FakeTensor(raw_values, backend, backend.index)
    return inputs, actual, projection


@pytest.fixture
def observation():
    backend = FakeBackend()
    inputs, actual, projection = operands(backend)
    base = base_worker(projection, backend)
    source = source_snapshot(base)
    metadata = {"schema": probe.SCHEMA, "scientific_scope": probe.SCIENTIFIC_SCOPE, "service": "instance-1", "dp_rank": 0,
                "first_layer_worker": base, "gpu": worker_module.gpu_identity(inputs, backend), "repetitions": 3,
                "flags": worker_module.matmul_flags(backend), "source_sha256": {"implementation": "a" * 64}, **probe.CLAIMS}
    mirrored = worker_module.run_mirror(inputs, actual, projection, 1, base["projection"], backend=backend)
    source_read = {name: source[name] for name in probe.SOURCE_READ_FIELDS}
    record = {**mirrored, "schema": probe.SCHEMA, "scientific_scope": probe.SCIENTIFIC_SCOPE,
        "binding": {name: source[name] for name in probe.BINDING_FIELDS}, "checkpoint": source["checkpoint"],
        "projection": source["projection"], "source_sha256": metadata["source_sha256"],
        "source_buffer_read_sha256": probe.canonical_sha(source_read),
        "source_buffer_read_after_sha256": probe.canonical_sha(source_read),
        "source_refresh_witness": source["refresh_witness"], "source_consumed_before_finalize": True}
    return record, source, metadata


def test_same_gpu_private_b1_padded_mirror_can_anchor_affine_shape_effect(observation):
    record, source, metadata = observation
    result = probe.validate_record(record, source, metadata)
    assert result["current_affine_operator_shape_mechanism_verified"] is True
    assert result["padded_mirror_matches_actual_QKV_output"] is True
    assert result["changed_elements"] == 1 and result["max_abs_difference"] == .03125
    assert all(result[name] is False for name in probe.CLAIMS)
    assert record["full_matrices_archived"] is False


@pytest.mark.parametrize("mutation", ["cross_gpu", "input", "weight", "alias", "repeat", "refresh", "prefix",
    "row", "mode", "scope", "flags", "stream", "bias", "budget", "kernel", "consume"])
def test_verifier_rejects_fake_negative_controls(observation, mutation):
    record, source, metadata = copy.deepcopy(observation)
    if mutation == "cross_gpu":
        record["private_tensors"]["target_input"]["gpu"] = {**record["gpu"], "uuid": "GPU-bbbbbbbb-0000-0000-0000-000000000001"}
    elif mutation == "input":
        record["private_tensors"]["target_input"]["row_sha256"] = "b" * 64
    elif mutation == "weight":
        record["private_tensors"]["weight"]["matrix_sha256"] = "b" * 64
    elif mutation == "alias":
        record["private_tensors"]["target_input"]["storage_ptr"] = record["originals"]["qkv_input"]["storage_ptr"]
    elif mutation == "repeat":
        record["rounds"][1][probe.OPERATORS[0]]["target"]["values"] = [2., 3.0625]
    elif mutation == "refresh":
        record["source_buffer_read_after_sha256"] = "b" * 64
    elif mutation == "prefix":
        source["forward_consumed_prefix_tokens"] -= 1
    elif mutation == "row":
        record["private_tensors"]["padded_input"]["token_row_index"] = 2
    elif mutation == "mode":
        source["forward_mode"] = "NONE"
    elif mutation == "scope":
        source["prompt_tokens"] = 74
    elif mutation == "flags":
        record["flags_after"]["allow_tf32"] = True
    elif mutation == "stream":
        record["cuda_stream_after"] = 99
    elif mutation == "bias":
        record["private_tensors"]["bias"] = record["private_tensors"]["target_input"]
    elif mutation == "budget":
        record["explicit_temporary_tensor_bytes"] = probe.MAX_DEVICE_BYTES + 1
    elif mutation == "kernel":
        record["specific_kernel_proven"] = True
    else:
        record["source_consumed_before_finalize"] = False
    with pytest.raises(ValueError):
        probe.validate_record(record, source, metadata)


def test_unmatched_actual_raw_output_retains_observation_but_not_anchored_mechanism(observation):
    record, source, metadata = copy.deepcopy(observation)
    source["buffers"]["qkv_output"]["values"] = [2., 3.125]
    source["buffers"]["qkv_output"]["row_sha256"] = first_layer.bf16_digest([2., 3.125])
    record["originals"]["qkv_output"]["row_sha256"] = source["buffers"]["qkv_output"]["row_sha256"]
    record["originals"]["qkv_output"]["matrix_sha256"] = "f" * 64
    record["source_buffer_read_sha256"] = probe.canonical_sha({name: source[name] for name in probe.SOURCE_READ_FIELDS})
    record["source_buffer_read_after_sha256"] = record["source_buffer_read_sha256"]
    result = probe.validate_record(record, source, metadata)
    assert result["isolated_eager_shape_difference_observed"] is True
    assert result["padded_mirror_matches_actual_QKV_output"] is False
    assert result["current_affine_operator_shape_mechanism_verified"] is False


@pytest.mark.parametrize("mutation", ["capture", "cross_gpu", "alias_clone", "bad_input_clone", "alias_output",
    "mutated_weight", "old_weight", "private_field", "bad_row", "stride", "bad_repetitions", "uuid"])
def test_runtime_fake_negative_controls_fail_closed_without_modifying_normal_weights(mutation):
    backend = FakeBackend()
    inputs, actual, projection = operands(backend)
    expected = projection_record(projection, backend)
    before = copy.deepcopy(projection.weight.values)
    kwargs = {"token_row": 1, "repetitions": 3}
    if mutation == "capture":
        backend.capturing = True
    elif mutation == "cross_gpu":
        projection.weight.device = FakeDevice(1)
    elif mutation in ("alias_clone", "bad_input_clone", "alias_output"):
        setattr(backend, mutation, True)
    elif mutation == "mutated_weight":
        projection.quant_method.mutate_weight = True
    elif mutation == "old_weight":
        expected["weight_sha256"] = "f" * 64
    elif mutation == "private_field":
        projection.quant_method.require_extra_field = True
    elif mutation == "bad_row":
        kwargs["token_row"] = 8
    elif mutation == "stride":
        inputs._stride = (4, 1)
    elif mutation == "uuid":
        backend.cuda.get_device_properties = lambda device: SimpleNamespace(uuid=None, name="A800", major=8, minor=0)
    else:
        kwargs["repetitions"] = 4
    reads = FakeTensor.cpu_reads
    with pytest.raises(RuntimeError):
        worker_module.run_mirror(inputs, actual, projection, expected_projection=expected, backend=backend, **kwargs)
    assert projection.weight.values == before
    if mutation == "capture":
        assert FakeTensor.cpu_reads == reads


def test_runtime_does_not_retain_gpu_tensors_in_records_or_history():
    backend = FakeBackend()
    inputs, actual, projection = operands(backend)
    before = set(FakeTensor.live)
    result = worker_module.run_mirror(inputs, actual, projection, 1, projection_record(projection, backend), backend=backend)
    assert set(FakeTensor.live) == before
    assert json.loads(json.dumps(result))["original_tensors_unchanged"] is True
    assert inputs.values == actual.values


def test_nondeterministic_repetitions_are_rejected_instead_of_selected(observation):
    backend = FakeBackend()
    backend.inconsistent = True
    inputs, actual, projection = operands(backend)
    record, source, metadata = copy.deepcopy(observation)
    record.update(worker_module.run_mirror(inputs, actual, projection, 1, metadata["first_layer_worker"]["projection"], backend=backend))
    with pytest.raises(ValueError, match="repetitions are inconsistent"):
        probe.validate_record(record, source, metadata)


@pytest.mark.parametrize("bias,padded,repetitions,shape_effect", [(True, 8, 2, True), (False, 1, 3, False), (False, 8, 3, False)])
@pytest.mark.parametrize("input_ids_dtype", ["torch.int32", "torch.int64"])
def test_bias_preservation_and_no_shape_effect_controls(observation, bias, padded, repetitions, shape_effect, input_ids_dtype):
    record, source, metadata = copy.deepcopy(observation)
    backend = FakeBackend(shape_effect=shape_effect)
    inputs, actual, projection = operands(backend, padded, bias)
    base = base_worker(projection, backend)
    source = source_snapshot(base, padded=padded, token_row=1 if padded == 8 else 0, input_ids_dtype=input_ids_dtype)
    for name in ("qkv_input", "qkv_output"):
        values = (inputs if name == "qkv_input" else actual).values[source["token_row_index"]]
        source["buffers"][name]["values"] = values
        source["buffers"][name]["row_sha256"] = first_layer.bf16_digest(values)
    metadata.update({"first_layer_worker": base, "repetitions": repetitions})
    record.update(worker_module.run_mirror(inputs, actual, projection, source["token_row_index"],
        base["projection"], repetitions, backend))
    record.update({"projection": source["projection"], "binding": {name: source[name] for name in probe.BINDING_FIELDS},
        "source_buffer_read_sha256": probe.canonical_sha({name: source[name] for name in probe.SOURCE_READ_FIELDS})})
    record["source_buffer_read_after_sha256"] = record["source_buffer_read_sha256"]
    result = probe.validate_record(record, source, metadata)
    assert result["shape_changed"] is (padded != 1)
    assert result["current_affine_operator_shape_mechanism_verified"] is shape_effect
    assert result["padded_mirror_matches_actual_QKV_output"] is True
    assert (record["private_tensors"]["bias"] is not None) is bias
    assert all(result[name] is False for name in probe.CLAIMS)


@pytest.mark.parametrize("name,dtype", [("input_ids", "torch.int16"), ("input_ids", "torch.float32"),
    ("positions", "torch.int32"), ("positions", "torch.float32")])
def test_only_ids32_or_ids64_and_positions64_are_accepted(observation, name, dtype):
    record, source, metadata = copy.deepcopy(observation)
    source["input_binding"][name]["dtype"] = dtype
    record["source_buffer_read_sha256"] = probe.canonical_sha({name: source[name] for name in probe.SOURCE_READ_FIELDS})
    record["source_buffer_read_after_sha256"] = record["source_buffer_read_sha256"]
    with pytest.raises(ValueError, match="token/position clone"):
        probe.validate_record(record, source, metadata)


@pytest.mark.parametrize("input_ids_dtype", ["torch.int32", "torch.int64"])
def test_source_budget_accounts_for_actual_id_dtype_and_refresh_counter(observation, input_ids_dtype):
    record, source, metadata = copy.deepcopy(observation)
    source["input_binding"]["input_ids"]["dtype"] = input_ids_dtype
    source["retained_device_bytes"] = source["num_tokens_padded"] * (24 if input_ids_dtype == "torch.int32" else 28) + 7
    record["source_buffer_read_sha256"] = probe.canonical_sha({name: source[name] for name in probe.SOURCE_READ_FIELDS})
    record["source_buffer_read_after_sha256"] = record["source_buffer_read_sha256"]
    with pytest.raises(ValueError, match="memory budget"):
        probe.validate_record(record, source, metadata)


@pytest.mark.parametrize("apply_effect,linear_effect", [(False, True), (True, False)])
def test_apply_is_predeclared_primary_and_linear_disagreement_is_not_selected(observation, apply_effect, linear_effect):
    record, source, metadata = copy.deepcopy(observation)
    backend = FakeBackend(shape_effect=linear_effect)
    inputs, actual, projection = operands(backend)
    projection.quant_method.apply_shape_effect = apply_effect
    record.update(worker_module.run_mirror(inputs, actual, projection, 1, metadata["first_layer_worker"]["projection"], backend=backend))
    result = probe.validate_record(record, source, metadata)
    assert result["predeclared_primary_operator"] == "quant_method.apply"
    assert result["apply_and_functional_linear_outputs_identical"] is False
    assert result["current_affine_operator_shape_mechanism_verified"] is apply_effect
    assert result["operator_results"][probe.OPERATORS[1]]["current_affine_operator_shape_mechanism_verified"] is linear_effect


@pytest.mark.parametrize("mutation", ["counter", "sequence", "token_row", "query_start", "native_row", "descriptor",
    "input_token", "input_position", "capture_input_row", "python_hooks", "model", "checkpoint", "worker_claim"])
def test_self_consistent_hashes_cannot_hide_stale_or_wrong_source_binding(observation, mutation):
    record, source, metadata = copy.deepcopy(observation)
    if mutation == "counter":
        source["refresh_witness"]["counter"]["dtype"] = "torch.float32"
    elif mutation == "sequence":
        source["refresh_witness"]["forward_sequence"] = 0
    elif mutation == "token_row":
        source["token_row_index"] = 2
    elif mutation == "query_start":
        source["query_start_loc"] = [0, 1, 3, 4, 5]
    elif mutation == "native_row":
        source["batch_native_request_ids"][1] = "not-the-target"
    elif mutation == "descriptor":
        source["capture_descriptor"] = {"num_tokens": 16}
    elif mutation == "input_token":
        source["input_binding"]["input_ids"]["value"] = 10
    elif mutation == "input_position":
        source["input_binding"]["positions"]["value"] -= 1
    elif mutation == "capture_input_row":
        source["input_binding"]["positions"]["token_row_index"] = 2
    elif mutation == "python_hooks":
        source["current_forward_hook_calls"] = 3
    elif mutation == "model":
        source["model_class"] = "not-the-recorded-model"
    elif mutation == "checkpoint":
        source["checkpoint"] = record["checkpoint"] = "not-the-recorded-checkpoint"
    else:
        metadata["numerical_root_cause_proven"] = True
    record["binding"] = {name: source[name] for name in probe.BINDING_FIELDS}
    record["source_refresh_witness"] = source["refresh_witness"]
    record["source_buffer_read_sha256"] = probe.canonical_sha({name: source[name] for name in probe.SOURCE_READ_FIELDS})
    record["source_buffer_read_after_sha256"] = record["source_buffer_read_sha256"]
    with pytest.raises(ValueError):
        probe.validate_record(record, source, metadata)


def test_builtin_operator_source_falls_back_to_compiled_parent_not_package_python(tmp_path, monkeypatch):
    binary = tmp_path / "fake_C.so"
    binary.write_bytes(b"fake compiled binding, not a specific CUDA kernel")
    symbol = SimpleNamespace(__module__="fake_C._nn", __qualname__="linear", __name__="linear")
    imports = []

    def import_module(name):
        imports.append(name)
        if name == "fake_C._nn":
            raise ImportError("not an importable package")
        return SimpleNamespace(__file__=str(binary))

    monkeypatch.setattr(worker_module.importlib, "import_module", import_module)
    monkeypatch.setattr(worker_module.inspect, "getsourcefile", lambda symbol: None)

    def unavailable(symbol):
        raise TypeError("built-in")

    monkeypatch.setattr(worker_module.inspect, "getsource", unavailable)
    directory = tmp_path / "copies"
    directory.mkdir()
    record = worker_module.copy_source(symbol, directory, tmp_path)
    assert imports == ["fake_C._nn", "fake_C"]
    assert record["symbol_sha256"] is None
    assert record["file_is_compiled_binding_not_specific_kernel"] is True
    assert probe.file_sha(tmp_path / record["copied_path"]) == probe.file_sha(binary)


def test_builtin_without_owning_binary_cannot_use_unrelated_package_file(tmp_path, monkeypatch):
    package = tmp_path / "__init__.py"
    package.write_text("unrelated package source")
    symbol = SimpleNamespace(__module__="fake_package._nn", __qualname__="linear", __name__="linear")
    monkeypatch.setattr(worker_module.inspect, "getsourcefile", lambda symbol: None)
    monkeypatch.setattr(worker_module.importlib, "import_module", lambda name: SimpleNamespace(__file__=str(package)))
    with pytest.raises(RuntimeError, match="binding file unavailable"):
        worker_module.copy_source(symbol, tmp_path, tmp_path)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FakeLogitsProcessor:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def invoke(self, inputs, output):
        for hook in self.hooks:
            assert hook(self, inputs, output) is None
        return output


class CUDAGraphWrapper:
    pass


def fake_parent_module():
    return SimpleNamespace(__file__=str(Path(worker_module.__file__).with_name("upstream_first_layer_worker.py")),
        validate_worker=first_layer.validate_worker, selected_position=first_layer.selected_position,
        buffer_key=lambda frame: (frame["mode"], frame["padded"], json.dumps(frame["descriptor"], sort_keys=True, separators=(",", ":"))))


def observer_setup(tmp_path, monkeypatch):
    backend = FakeBackend()
    inputs, actual, projection = operands(backend)
    base = base_worker(projection, backend)
    source = source_snapshot(base)
    state = SimpleNamespace(num_prompt_tokens=149, prompt_token_ids=source["prompt_token_ids"],
                            output_token_ids=source["output_prefix_token_ids"])
    frame = {"mode": "FULL", "padded": 8, "descriptor": {"num_tokens": 8}, "pending": True}
    parent = fake_parent_module()
    graph = {"buffers": {"qkv_input": inputs, "qkv_output": actual}}
    recorder = SimpleNamespace(frame=frame, eager=None, graphs={parent.buffer_key(frame): graph},
        read=lambda row, state: copy.deepcopy({name: source[name] for name in probe.SOURCE_READ_FIELDS}))
    runner = SimpleNamespace(input_batch=SimpleNamespace(req_ids=source["batch_native_request_ids"]),
        model=CUDAGraphWrapper(),
        requests={native: state if native == source["native_request_id"] else SimpleNamespace(num_prompt_tokens=74,
            output_token_ids=[]) for native in source["batch_native_request_ids"]})
    worker = SimpleNamespace(_first_layer_probe=recorder, model_runner=runner,
                             parallel_config=SimpleNamespace(data_parallel_rank=0))
    monkeypatch.setenv("SLURM_JOB_ID", "303")
    monkeypatch.setenv("DP_OPT_QKV_MIRROR_COMPUTE_CONFIRMED", "1")
    monkeypatch.setenv("DP_OPT_SERVICE", "instance-1")
    monkeypatch.setitem(sys.modules, "vllm.compilation.cuda_graph", SimpleNamespace(CUDAGraphWrapper=CUDAGraphWrapper))
    write_json(tmp_path / "upstream-first-layer-worker-instance-1-dp0.json", base)
    write_json(tmp_path / "phase.json", {"repeat": 1, "layout": "corank0"})
    observer = worker_module.MirrorObserver(worker, projection, tmp_path, backend, parent)
    observer.initialize()
    return observer, source, recorder, backend


def save_parent_snapshot(observer, source):
    native = source["native_request_id"]
    tag = hashlib.sha256(native.encode()).hexdigest()[:12]
    name = f"trial-r1-dp0-p53-{tag}.json"
    write_json(observer.root / "upstream-first-layer-snapshots" / name, source)
    return name


def test_observer_reads_before_parent_consume_and_finalizes_after_without_output_change(tmp_path, monkeypatch):
    observer, source, recorder, backend = observer_setup(tmp_path, monkeypatch)
    logits = FakeTensor([[0., 0.]] * 4, backend)
    processor = FakeLogitsProcessor()
    processor.register_forward_hook(observer.before)
    events = []

    def parent_consume(module, inputs, output):
        assert observer.pending is not None
        assert backend.calls == 12
        events.append("parent_consumed")
        save_parent_snapshot(observer, source)
        recorder.frame = recorder.eager = None

    processor.register_forward_hook(parent_consume)
    processor.register_forward_hook(observer.after)
    assert processor.invoke((None, logits), logits) is logits
    assert events == ["parent_consumed"] and observer.pending is None
    assert len(list((tmp_path / "upstream-qkv-mirror-snapshots").glob("*.json"))) == 1


def test_finalizer_before_parent_consume_refuses_and_clears_pending(tmp_path, monkeypatch):
    observer, source, recorder, backend = observer_setup(tmp_path, monkeypatch)
    logits = FakeTensor([[0., 0.]] * 4, backend)
    observer.before(None, (None, logits), logits)
    with pytest.raises(RuntimeError, match="before parent"):
        observer.after(None, (None, logits), logits)
    assert observer.pending is None and recorder.frame is not None


def test_partial_discard_and_eager_do_not_mirror_graph_history(tmp_path, monkeypatch):
    observer, source, recorder, backend = observer_setup(tmp_path, monkeypatch)
    logits = FakeTensor([[0., 0.]] * 4, backend)
    recorder.read = lambda *args: None
    observer.before(None, (None, logits), logits)
    assert observer.pending is None and backend.calls == 0
    recorder.frame["mode"] = "NONE"
    observer.before(None, (None, logits), logits)
    assert backend.calls == 0


def test_source_refresh_during_mirror_is_rejected_before_parent_consume(tmp_path, monkeypatch):
    observer, source, recorder, backend = observer_setup(tmp_path, monkeypatch)
    calls = []

    def refreshed(row, state):
        calls.append(row)
        result = {name: copy.deepcopy(source[name]) for name in probe.SOURCE_READ_FIELDS}
        if len(calls) > 1:
            result["refresh_witness"]["generation_after"] += 1
        return result

    recorder.read = refreshed
    logits = FakeTensor([[0., 0.]] * 4, backend)
    with pytest.raises(RuntimeError, match="generation changed"):
        observer.before(None, (None, logits), logits)
    assert observer.pending is None


def test_worker_resolution_is_compute_only(monkeypatch):
    monkeypatch.delitem(worker_module.__dict__, "UpstreamQKVMirrorVerificationWorker", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="compute-only"):
        getattr(worker_module, "UpstreamQKVMirrorVerificationWorker")


def test_actual_lazy_subclass_installs_precapture_read_and_postconsume_finalizer(tmp_path, monkeypatch):
    seed, source, recorder, backend = observer_setup(tmp_path / "seed", monkeypatch)
    root = tmp_path / "initialized"
    base = seed.metadata["first_layer_worker"]
    write_json(root / "upstream-first-layer-worker-instance-1-dp0.json", base)
    write_json(root / "phase.json", {"repeat": 1, "layout": "corank0"})
    processor = FakeLogitsProcessor()
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(qkv_proj=seed.projection))]),
                            logits_processor=processor)
    runner = seed.worker.model_runner
    runner.get_model = lambda: model
    events = []
    logits = FakeTensor([[0., 0.]] * 4, backend)

    class FakeParent:
        def initialize_from_config(self, config):
            assert config == "unchanged-kv-config"
            assert len(processor.hooks) == 1 and processor.hooks[0].__name__ == "before"
            self._first_layer_probe = recorder
            recorder.frame["pending"] = False
            backend.capturing = True
            reads_before = FakeTensor.cpu_reads
            assert processor.invoke((None, logits), logits) is logits
            assert FakeTensor.cpu_reads == reads_before and backend.calls == 0
            backend.capturing = False
            recorder.frame["pending"] = True

            def head_snapshot(module, inputs, output):
                assert backend.calls == 12
                events.append("head")

            def parent_snapshot(module, inputs, output):
                events.append("parent")
                observer = self._qkv_mirror_observer
                assert observer.pending is not None
                save_parent_snapshot(observer, source)
                recorder.frame = recorder.eager = None

            processor.register_forward_hook(head_snapshot)
            processor.register_forward_hook(parent_snapshot)
            return "parent-return"

    parent = fake_parent_module()
    parent.UpstreamFirstLayerVerificationWorker = FakeParent
    monkeypatch.setitem(sys.modules, "torch", backend)
    monkeypatch.setitem(sys.modules, "upstream_first_layer_worker", parent)
    monkeypatch.setenv("DP_OPT_EVIDENCE_ROOT", str(root))
    monkeypatch.delitem(worker_module.__dict__, "UpstreamQKVMirrorVerificationWorker", raising=False)
    worker_type = worker_module.__getattr__("UpstreamQKVMirrorVerificationWorker")
    try:
        worker = worker_type()
        worker.model_runner = runner
        worker.parallel_config = seed.worker.parallel_config
        assert worker.initialize_from_config("unchanged-kv-config") == "parent-return"
        assert [hook.__name__ for hook in processor.hooks] == ["before", "head_snapshot", "parent_snapshot", "after"]
        assert processor.invoke((None, logits), logits) is logits
        assert events == ["head", "parent"] and worker._qkv_mirror_observer.pending is None
        assert len(list((root / "upstream-qkv-mirror-snapshots").glob("*.json"))) == 1
        worker._qkv_mirror_observer.close()
        assert [hook.__name__ for hook in processor.hooks] == ["head_snapshot", "parent_snapshot"]
    finally:
        worker_module.__dict__.pop("UpstreamQKVMirrorVerificationWorker", None)


@pytest.fixture
def archived_observation(tmp_path, monkeypatch):
    observer, source, recorder, backend = observer_setup(tmp_path, monkeypatch)
    logits = FakeTensor([[0., 0.]] * 4, backend)
    observer.before(None, (None, logits), logits)
    save_parent_snapshot(observer, source)
    recorder.frame = recorder.eager = None
    observer.after(None, (None, logits), logits)
    for service, rank in (("instance-1", 1), ("instance-2", 0), ("instance-2", 1)):
        other_backend = FakeBackend(rank)
        other_projection = QKVParallelLinear(other_backend)
        base = base_worker(other_projection, other_backend, service, rank)
        write_json(tmp_path / f"upstream-first-layer-worker-{service}-dp{rank}.json", base)
        metadata = copy.deepcopy(observer.metadata)
        metadata.update({"service": service, "dp_rank": rank, "first_layer_worker": base,
            "first_layer_worker_sha256": probe.file_sha(tmp_path / f"upstream-first-layer-worker-{service}-dp{rank}.json"),
            "gpu": worker_module.gpu_identity(other_projection.weight, other_backend)})
        write_json(tmp_path / f"upstream-qkv-mirror-worker-{service}-dp{rank}.json", metadata)
    monkeypatch.setattr(first_layer, "analyze", lambda root: {"quality_equivalence_proven": False})
    return tmp_path


def test_offline_archive_verifier_requires_sources_four_workers_and_matching_actual_rows(archived_observation):
    result = probe.analyze(archived_observation)
    assert result["validated_graph_observations"] == 1
    assert result["any_current_affine_operator_shape_mechanism_verified"] is True
    assert result["all_trace_first_divergences_attributed"] is False
    assert all(result[name] is False for name in probe.CLAIMS)


@pytest.mark.parametrize("mutation", ["missing_worker", "source_epoch", "source_file", "source_snapshot", "missing_mirror", "wrong_gpu"])
def test_offline_archive_fake_negative_provenance_fails_closed(archived_observation, mutation):
    root = archived_observation
    path = root / "upstream-qkv-mirror-worker-instance-2-dp1.json"
    metadata = probe.read(path)
    if mutation == "missing_worker":
        path.unlink()
    elif mutation == "source_epoch":
        metadata["source_sha256"]["mirror_worker"] = "b" * 64
        write_json(path, metadata)
    elif mutation == "source_file":
        (root / metadata["sources"]["mirror_worker"]["copied_path"]).write_text("changed source")
    elif mutation == "source_snapshot":
        source_path = next((root / "upstream-first-layer-snapshots").glob("*.json"))
        source_path.write_text(source_path.read_text() + "\n")
    elif mutation == "missing_mirror":
        next((root / "upstream-qkv-mirror-snapshots").glob("*.json")).unlink()
    else:
        metadata["gpu"]["uuid"] = "GPU-bbbbbbbb-0000-0000-0000-000000000001"
        write_json(path, metadata)
    with pytest.raises(ValueError):
        probe.analyze(root)


def fake_kernel_trace(marker, backend):
    process, thread, correlation = os.getpid(), 71, 42
    return {"deviceProperties": [{"id": backend.index, "name": backend.properties(FakeDevice(backend.index)).name,
                                  "uuid": backend.properties(FakeDevice(backend.index)).uuid}],
            "traceEvents": [
                {"name": marker, "cat": "user_annotation", "ph": "X", "ts": 100, "dur": 10,
                 "pid": process, "tid": thread, "args": {}},
                {"name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "ts": 101, "dur": 1,
                 "pid": process, "tid": thread, "args": {"correlation": correlation, "External id": 19}},
                {"name": "actual_fake_activity_name_not_shape_inference", "cat": "kernel", "ph": "X", "ts": 103, "dur": 3,
                 "pid": 0, "tid": 7, "args": {"correlation": correlation, "device": backend.index, "stream": 7,
                 "grid": [2, 1, 1], "block": [128, 1, 1], "registers per thread": 32}}]}


class FakeKernelProfile:
    def __init__(self, owner):
        self.owner = owner

    def __enter__(self):
        self.owner.starts.append(self.owner.backend.calls)
        if self.owner.mode == "start_failure":
            raise RuntimeError("CUPTI start unavailable")
        if self.owner.mode == "capture_after_start":
            self.owner.backend.capturing = True
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.owner.stops.append(self.owner.backend.calls)
        if self.owner.mode == "stop_failure":
            raise RuntimeError("CUPTI stop unavailable")

    def export_chrome_trace(self, path):
        if self.owner.mode == "export_failure":
            raise RuntimeError("CUPTI export unavailable")
        if self.owner.mode == "oversize":
            Path(path).write_text(" " * (probe.MAX_KERNEL_TRACE_BYTES + 1))
            return
        if self.owner.mode == "invalid_json":
            Path(path).write_text("{")
            return
        trace = fake_kernel_trace(self.owner.marker, self.owner.backend)
        if self.owner.mode == "no_cuda_events":
            trace["traceEvents"] = trace["traceEvents"][:1]
        elif self.owner.mode == "missing_geometry":
            trace["traceEvents"][-1]["args"].pop("grid")
        elif self.owner.mode == "missing_correlation":
            trace["traceEvents"][-1]["args"].pop("correlation")
        elif self.owner.mode == "event_budget":
            trace["traceEvents"].extend({"ph": "i"} for index in range(probe.MAX_KERNEL_TRACE_EVENTS))
        elif self.owner.mode == "bad_device_schema":
            trace["deviceProperties"] = None
        write_json(Path(path), trace)


class FakeKernelAnnotation:
    def __init__(self, owner, marker):
        self.owner, self.marker = owner, marker

    def __enter__(self):
        self.owner.marker = self.marker
        if self.owner.mode == "annotation_failure":
            raise RuntimeError("annotation unavailable")
        return self

    def __exit__(self, exception_type, exception, traceback):
        if self.owner.mode == "annotation_stop_failure":
            raise RuntimeError("annotation stop unavailable")


class FakeKernelProfiler:
    ProfilerActivity = SimpleNamespace(CPU="CPU", CUDA="CUDA")

    def __init__(self, backend, mode="supported"):
        self.backend, self.mode = backend, mode
        self.starts, self.stops, self.options = [], [], []
        self.marker = None

    def supported_activities(self):
        return {"CPU"} if self.mode == "cuda_unavailable" else {"CPU", "CUDA"}

    def profile(self, **options):
        self.options.append(options)
        return FakeKernelProfile(self)

    def record_function(self, marker):
        return FakeKernelAnnotation(self, marker)


def make_kernel_observation(observation, root, mode="supported", repetitions=3, source_path=None):
    record, source, metadata = copy.deepcopy(observation)
    backend = FakeBackend(metadata["dp_rank"])
    inputs, actual, projection = operands(backend)
    backend.profiler = FakeKernelProfiler(backend, mode)
    if mode == "api_unavailable":
        backend.profiler = None
    directory = root / "kernel-profiler-sources"
    directory.mkdir(exist_ok=True)
    sources = {} if backend.profiler is None else {
        name: worker_module.copy_source(getattr(backend.profiler, name), directory, root)
        for name in ("profile", "record_function")}
    metadata.update({"slurm_job_id": 303, "process_id": os.getpid(), "repetitions": repetitions,
                     "kernel_activity_policy": probe.kernel_activity_policy(True),
                     "kernel_profiler_sources": {"status": "available" if sources else "unsupported", "sources": sources}})
    relative = source_path or "upstream-first-layer-snapshots/kernel-source.json"
    source_read = {name: source[name] for name in probe.SOURCE_READ_FIELDS}
    request = {"root": root, "policy": metadata["kernel_activity_policy"], "eligible": True,
               "binding": probe.kernel_base_binding(metadata, source_read, source, relative)}
    mirrored = worker_module.run_mirror(inputs, actual, projection, source["token_row_index"], source["projection"],
                                        repetitions, backend, kernel_request=request)
    record.update(mirrored)
    record.update({"source_snapshot_path": relative, "binding": {name: source[name] for name in probe.BINDING_FIELDS},
                   "source_buffer_read_sha256": probe.canonical_sha(source_read),
                   "source_buffer_read_after_sha256": probe.canonical_sha(source_read),
                   "source_refresh_witness": source["refresh_witness"]})
    return record, source, metadata, backend


@pytest.mark.parametrize("repetitions", [2, 3])
def test_optional_kernel_capture_profiles_only_bounded_extra_true_apply_after_all_baselines(observation, tmp_path, repetitions):
    record, source, metadata, backend = make_kernel_observation(observation, tmp_path, repetitions=repetitions)
    result = probe.validate_record(record, source, metadata, tmp_path)
    activity = result["kernel_activity"]
    assert activity["status"] == "supported" and activity["kernel_activity_observed"] is True
    assert backend.profiler.starts == list(range(repetitions * 4, repetitions * 6))
    assert backend.profiler.stops == list(range(repetitions * 4 + 1, repetitions * 6 + 1))
    assert all(options == {"activities": ["CPU", "CUDA"], "record_shapes": False, "profile_memory": False,
                           "with_stack": False, "with_flops": False} for options in backend.profiler.options)
    assert len(activity["calls"]) == repetitions * 2 <= probe.MAX_KERNEL_CALLS
    for ordinal, call in enumerate(record["kernel_activity"]["calls"]):
        assert call["binding"]["operator"] == "quant_method.apply"
        assert call["binding"]["real_call_repeat"] == source["repeat"]
        assert call["binding"]["mirror_repeat_index"] == ordinal // 2
        assert call["shadow_output_matches_unprofiled_apply"] is True
        assert probe.file_sha(tmp_path / call["trace"]["path"]) == call["trace"]["sha256"]
        event = call["activity"]["events"][0]
        assert event["kernel_name"] == "actual_fake_activity_name_not_shape_inference"
        assert event["launch_name"] == "cudaLaunchKernel" and event["correlation"] == 42
        assert event["kernel_args"]["grid"] == [2, 1, 1] and event["launch_geometry_complete"] is True
    for original_round, retained_round in zip(observation[0]["rounds"], record["rounds"]):
        for operator in probe.OPERATORS:
            for shape in ("padded", "target"):
                for key in ("values", "matrix_sha256", "row_sha256"):
                    assert retained_round[operator][shape][key] == original_round[operator][shape][key]
    assert result["specific_kernel_proven"] is False
    assert activity["native_graph_kernel_attribution_proven"] is False
    assert activity["gemv_gemm_dispatch_inferred_from_shape"] is False
    assert not list(tmp_path.rglob(".shadow-trace-*"))
    assert probe.validate_record(record, source, metadata)["kernel_activity"]["kernel_activity_observed"] is False


@pytest.mark.parametrize("mode,expected", [
    ("api_unavailable", "unsupported"), ("cuda_unavailable", "unsupported"), ("start_failure", "unsupported"),
    ("no_cuda_events", "unsupported"), ("annotation_failure", "partial"), ("annotation_stop_failure", "partial"),
    ("stop_failure", "partial"), ("export_failure", "partial"), ("oversize", "partial"),
    ("missing_geometry", "partial"), ("missing_correlation", "partial"), ("event_budget", "partial"),
    ("bad_device_schema", "partial"), ("invalid_json", "partial")])
def test_unavailable_or_partial_cupti_is_explicit_and_stops_further_capture(observation, tmp_path, mode, expected):
    record, source, metadata, backend = make_kernel_observation(observation, tmp_path, mode)
    result = probe.validate_record(record, source, metadata, tmp_path)
    assert result["kernel_activity"]["status"] == expected
    assert result["current_affine_operator_shape_mechanism_verified"] is True
    calls = record["kernel_activity"]["calls"]
    assert len(calls) == 6 and calls[0]["activity"]["reason"] is not None
    assert all(call["shadow_call_executed"] is False and call["profile_entered"] is False for call in calls[1:])
    assert backend.calls in (12, 13)
    assert not list(tmp_path.rglob(".shadow-trace-*"))
    if mode == "no_cuda_events":
        assert calls[0]["trace"] is not None and calls[0]["activity"]["reason"] == "cuda_activity_absent"
        assert result["kernel_activity"]["kernel_activity_observed"] is False
    if mode == "oversize":
        assert calls[0]["trace"] is None and calls[0]["discarded_export_bytes"] > probe.MAX_KERNEL_TRACE_BYTES


@pytest.mark.parametrize("late", [False, True])
def test_kernel_profile_prohibits_capture_context_before_start_and_before_apply(observation, tmp_path, late):
    record, source, metadata = observation
    backend = FakeBackend()
    backend.profiler = FakeKernelProfiler(backend, "capture_after_start" if late else "supported")
    metadata = {**metadata, "slurm_job_id": 303, "process_id": os.getpid()}
    binding = probe.kernel_call_binding(probe.kernel_base_binding(metadata,
        {name: source[name] for name in probe.SOURCE_READ_FIELDS}, source, "upstream-first-layer-snapshots/test.json"),
        record, 0, "target")
    backend.capturing = not late
    with pytest.raises(RuntimeError, match="CUDA Graph capture"):
        worker_module.profile_shadow_apply(lambda: pytest.fail("no apply during capture"), backend, tmp_path, binding,
                                          record["rounds"][0][probe.PRIMARY_OPERATOR]["target"], 0)
    assert backend.profiler.starts == ([0] if late else [])
    assert backend.profiler.stops == ([0] if late else [])
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("mutation", ["trace_sha", "trace_bytes", "trace_path", "marker", "process", "device", "uuid", "events",
                                     "descriptor", "repeat", "source_sha", "no_raw", "execution_flag", "alias", "policy"])
def test_kernel_provenance_rejects_identity_raw_sha_and_fabricated_metadata(observation, tmp_path, mutation):
    record, source, metadata, backend = make_kernel_observation(observation, tmp_path)
    call = record["kernel_activity"]["calls"][0]
    path = tmp_path / call["trace"]["path"]
    if mutation == "trace_sha":
        path.write_text(path.read_text() + "\n")
    elif mutation == "trace_bytes":
        call["trace"]["bytes"] += 1
    elif mutation in ("marker", "process", "device", "uuid"):
        trace = probe.read(path)
        if mutation == "marker":
            trace["traceEvents"][0]["name"] = probe.KERNEL_MARKER_PREFIX + "b" * 64
        elif mutation == "process":
            trace["traceEvents"][0]["pid"] += 1
        elif mutation == "device":
            trace["traceEvents"][-1]["args"]["device"] = 1
        else:
            trace["deviceProperties"][0]["uuid"] = "GPU-bbbbbbbb-0000-0000-0000-000000000001"
        write_json(path, trace)
        call["trace"].update({"sha256": probe.file_sha(path), "bytes": path.stat().st_size})
    elif mutation == "trace_path":
        call["trace"]["path"] = record["kernel_activity"]["calls"][1]["trace"]["path"]
    elif mutation == "events":
        call["activity"]["events"][0]["kernel_name"] = "guessed_gemv"
    elif mutation == "descriptor":
        call["binding"]["forward_descriptor"] = {"num_tokens": 1}
    elif mutation == "repeat":
        call["binding"]["real_call_repeat"] = 99
    elif mutation == "source_sha":
        metadata["kernel_profiler_sources"]["status"] = "unsupported"
    elif mutation == "no_raw":
        call["trace"] = None
    elif mutation == "execution_flag":
        call["shadow_call_executed"] = 1
    elif mutation == "alias":
        call["shadow_output"]["storage_ptr"] = record["originals"]["qkv_input"]["storage_ptr"]
    else:
        metadata["kernel_activity_policy"]["maximum_shadow_calls_per_worker"] = 6.0
    with pytest.raises(ValueError):
        probe.validate_record(record, source, metadata, tmp_path)


@pytest.mark.parametrize("field", ["status", "observed", "output"])
def test_unsupported_diagnostic_cannot_fabricate_attribution_without_execution_or_raw(observation, tmp_path, field):
    record, source, metadata, backend = make_kernel_observation(observation, tmp_path, "api_unavailable")
    call = record["kernel_activity"]["calls"][0]
    if field == "status":
        call["activity"]["status"] = "supported"
    elif field == "observed":
        call["activity"]["kernel_activity_observed"] = True
    else:
        call["shadow_output_matches_unprofiled_apply"] = True
    with pytest.raises(ValueError):
        probe.validate_record(record, source, metadata, tmp_path)


def attach_kernel_to_archive(root, mode="supported", source_path=None):
    worker_path = root / "upstream-qkv-mirror-worker-instance-1-dp0.json"
    path = next((root / "upstream-qkv-mirror-snapshots").glob("*.json"))
    record = probe.read(path)
    relative = source_path or record["source_snapshot_path"]
    source = probe.read(root / relative)
    record, source, metadata, backend = make_kernel_observation((record, source, probe.read(worker_path)), root,
                                                              mode, source_path=relative)
    record["source_snapshot_sha256"] = probe.file_sha(root / relative)
    write_json(root / "upstream-qkv-mirror-snapshots" / Path(relative).name, record)
    for other_path in root.glob("upstream-qkv-mirror-worker-*.json"):
        other = probe.read(other_path)
        other.update({name: metadata[name] for name in ("kernel_activity_policy", "kernel_profiler_sources")})
        write_json(other_path, other)
    return record


def test_archive_independently_parses_raw_kernel_activity_and_reports_worker_budget(archived_observation):
    root = archived_observation
    record = attach_kernel_to_archive(root)
    result = probe.analyze(root)
    assert result["independently_verified_kernel_traces"] == 6
    assert result["kernel_activity_worker_budgets"]["instance-1-dp0"] == {"observations": 1, "call_slots": 6}
    assert result["observations"][0]["kernel_activity"]["status"] == "supported"
    path = root / record["kernel_activity"]["calls"][0]["trace"]["path"]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="SHA"):
        probe.analyze(root)


@pytest.mark.parametrize("mutation", ["orphan_trace", "profiler_source", "second_observation"])
def test_archive_rejects_unbound_trace_sources_or_multiple_profiled_observations(archived_observation, mutation):
    root = archived_observation
    record = attach_kernel_to_archive(root)
    if mutation == "orphan_trace":
        write_json(root / "upstream-qkv-mirror-kernel-traces/orphan.json", {})
    elif mutation == "profiler_source":
        metadata = probe.read(root / "upstream-qkv-mirror-worker-instance-1-dp0.json")
        (root / metadata["kernel_profiler_sources"]["sources"]["profile"]["copied_path"]).write_text("changed source")
    else:
        metadata = probe.read(root / "upstream-qkv-mirror-worker-instance-1-dp0.json")
        relative = "upstream-first-layer-snapshots/second-observation.json"
        write_json(root / relative, source_snapshot(metadata["first_layer_worker"], position=54))
        attach_kernel_to_archive(root, source_path=relative)
    with pytest.raises(ValueError):
        probe.analyze(root)


def test_observer_opt_in_consumes_single_budget_even_when_cuda_activity_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("DP_OPT_QKV_MIRROR_KERNEL_ACTIVITY", "1")
    observer, source, recorder, backend = observer_setup(tmp_path, monkeypatch)
    logits = FakeTensor([[0., 0.]] * 4, backend)
    observer.before(None, (None, logits), logits)
    assert observer.kernel_observations == 1
    assert observer.pending[0]["kernel_activity"]["status"] == "unsupported"
    observer.pending = None
    source["position"] = 54
    observer.before(None, (None, logits), logits)
    assert observer.kernel_observations == 1 and backend.calls == 24
    assert observer.pending[0]["kernel_activity"]["status"] == "budget_exhausted"


def test_profiler_without_installed_source_binding_is_not_invoked(tmp_path, monkeypatch):
    monkeypatch.setenv("DP_OPT_QKV_MIRROR_KERNEL_ACTIVITY", "1")
    observer, source, recorder, backend = observer_setup(tmp_path, monkeypatch)
    backend.profiler = FakeKernelProfiler(backend)
    logits = FakeTensor([[0., 0.]] * 4, backend)
    observer.before(None, (None, logits), logits)
    assert backend.profiler.starts == [] and backend.calls == 12
    assert observer.pending[0]["kernel_activity"]["calls"][0]["activity"]["reason"] == "profiler_source_binding_unavailable"


def test_kernel_opt_in_is_explicit_and_default_does_not_start_a_profiler(tmp_path, monkeypatch):
    monkeypatch.delenv("DP_OPT_QKV_MIRROR_KERNEL_ACTIVITY", raising=False)
    observer, source, recorder, backend = observer_setup(tmp_path / "default", monkeypatch)
    backend.profiler = SimpleNamespace(profile=lambda **options: pytest.fail("default must not profile"))
    logits = FakeTensor([[0., 0.]] * 4, backend)
    observer.before(None, (None, logits), logits)
    assert observer.pending[0]["kernel_activity"]["status"] == "disabled" and backend.calls == 12
    monkeypatch.setenv("DP_OPT_QKV_MIRROR_KERNEL_ACTIVITY", "yes")
    with pytest.raises(RuntimeError, match="explicit 0 or 1"):
        observer_setup(tmp_path / "invalid", monkeypatch)


def test_reference_graph_replay_does_not_consume_trial_kernel_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("DP_OPT_QKV_MIRROR_KERNEL_ACTIVITY", "1")
    observer, source, recorder, backend = observer_setup(tmp_path, monkeypatch)
    runner = observer.worker.model_runner
    original_native = source["native_request_id"]
    runner.requests["reference-conv1"] = runner.requests.pop(original_native)
    runner.input_batch.req_ids[1] = "reference-conv1"
    logits = FakeTensor([[0., 0.]] * 4, backend)
    observer.before(None, (None, logits), logits)
    assert observer.kernel_observations == 0
    assert observer.pending[0]["kernel_activity"]["status"] == "reference_not_profiled"
    observer.pending = None
    runner.requests[original_native] = runner.requests.pop("reference-conv1")
    runner.input_batch.req_ids[1] = original_native
    observer.before(None, (None, logits), logits)
    assert observer.kernel_observations == 1 and observer.pending[0]["kernel_activity"]["status"] == "unsupported"


@pytest.mark.parametrize("mode", ["supported", "stop_failure", "no_cuda_events"])
def test_optional_kernel_records_retain_no_gpu_tensors_or_private_outputs(observation, tmp_path, mode):
    before = set(FakeTensor.live)
    record, source, metadata, backend = make_kernel_observation(observation, tmp_path, mode)
    assert set(FakeTensor.live) == before
    assert json.loads(json.dumps(record))["original_tensors_unchanged"] is True


@pytest.mark.parametrize("fault", ["alias", "operator_failure"])
def test_shadow_operator_faults_do_not_get_disguised_as_cupti_unavailability(observation, tmp_path, fault):
    record, source, metadata = observation
    backend = FakeBackend()
    backend.profiler = FakeKernelProfiler(backend)
    inputs, actual, projection = operands(backend)
    originals = [copy.deepcopy(tensor.values) for tensor in (inputs, actual, projection.weight)]
    native_apply = projection.quant_method.apply

    def apply(layer, operand, bias):
        if backend.profiler.marker is not None:
            if fault == "alias":
                return operand
            raise RuntimeError("real shadow operator failure")
        return native_apply(layer, operand, bias)

    projection.quant_method.apply = apply
    metadata = {**metadata, "process_id": os.getpid(), "slurm_job_id": 303}
    request = {"root": tmp_path, "policy": probe.kernel_activity_policy(True), "eligible": True,
               "binding": probe.kernel_base_binding(metadata, {name: source[name] for name in probe.SOURCE_READ_FIELDS},
                                                    source, "upstream-first-layer-snapshots/test.json")}
    with pytest.raises(RuntimeError, match="alias|real shadow operator failure"):
        worker_module.run_mirror(inputs, actual, projection, 1, source["projection"], backend=backend, kernel_request=request)
    assert backend.profiler.starts == [12] and backend.profiler.stops == [12]
    assert [tensor.values for tensor in (inputs, actual, projection.weight)] == originals


def test_parser_never_assigns_unlinked_concurrent_activity_to_shadow_apply(observation, tmp_path):
    record, source, metadata, backend = make_kernel_observation(observation, tmp_path)
    binding = record["kernel_activity"]["calls"][0]["binding"]
    trace = fake_kernel_trace(probe.kernel_marker(binding), backend)
    unrelated = copy.deepcopy(trace["traceEvents"][-1])
    unrelated["args"]["correlation"] = 777
    unrelated["name"] = "concurrent_kernel_cannot_be_attributed"
    trace["traceEvents"].append(unrelated)
    result = probe.parse_kernel_trace(trace, binding)
    assert result["status"] == "partial" and result["unattributed_cuda_events"] == 1
    assert result["kernel_names"] == ["actual_fake_activity_name_not_shape_inference"]
    trace["traceEvents"][1]["tid"] += 1
    result = probe.parse_kernel_trace(trace, binding)
    assert result["status"] == "partial" and result["kernel_activity_observed"] is False
    trace["traceEvents"][0]["dur"] = float("inf")
    assert probe.parse_kernel_trace(trace, binding)["reason"] == "invalid_activity_timestamps"


def test_parser_declares_nested_or_duplicate_launch_correlation_partial(observation, tmp_path):
    record, source, metadata, backend = make_kernel_observation(observation, tmp_path)
    binding = record["kernel_activity"]["calls"][0]["binding"]
    trace = fake_kernel_trace(probe.kernel_marker(binding), backend)
    driver_launch = copy.deepcopy(trace["traceEvents"][1])
    driver_launch.update({"cat": "cuda_driver", "name": "cuLaunchKernel"})
    trace["traceEvents"].append(driver_launch)
    result = probe.parse_kernel_trace(trace, binding)
    assert result["status"] == "partial" and result["reason"] == "ambiguous_launch_correlation"
    assert result["kernel_activity_observed"] is False
