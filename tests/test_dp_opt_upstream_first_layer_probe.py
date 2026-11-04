import copy
import importlib.util
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import head_hidden_probe
import upstream_first_layer_probe as probe


@dataclass
class Descriptor:
    num_tokens: int
    uniform_decode: bool = True


class FakeTensor:
    synchronizations = 0
    capture_operations = None

    def __init__(self, values, dtype="torch.bfloat16"):
        self.values = copy.deepcopy(values)
        self.dtype = dtype
        self.device = SimpleNamespace(type="cuda")
        self.shape = self.dimensions(values)

    @classmethod
    def dimensions(cls, values):
        return (len(values),) + cls.dimensions(values[0]) if isinstance(values, list) and values else ()

    def __getitem__(self, index):
        return FakeTensor(self.values[index], self.dtype)

    def detach(self):
        return self

    def clone(self):
        result = FakeTensor(self.values, self.dtype)
        if self.capture_operations is not None:
            self.capture_operations.append(("clone", lambda: result.copy_(self)))
        return result

    def copy_(self, source):
        self.values = copy.deepcopy(source.values)
        self.shape = source.shape
        return self

    def transform(self, function):
        result = FakeTensor(function(self.values), self.dtype)

        def replay():
            result.values = function(self.values)
            result.shape = self.dimensions(result.values)

        if self.capture_operations is not None:
            self.capture_operations.append(("compute", replay))
        return result

    def add_(self, value):
        self.values += value

        def replay():
            self.values += value

        if self.capture_operations is not None:
            self.capture_operations.append(("generation", replay))
        return self

    def cpu(self):
        assert self.capture_operations is None
        FakeTensor.synchronizations += 1
        return self

    def float(self):
        return self

    def contiguous(self):
        return self

    def view(self, dtype):
        return self

    def numpy(self):
        values = self.values
        flat = [value for row in values for value in row] if values and isinstance(values[0], list) else values
        encoded = b"".join(struct.pack("<f", float(value))[2:] for value in flat)
        return SimpleNamespace(tobytes=lambda: encoded)

    def tolist(self):
        return copy.deepcopy(self.values)

    def item(self):
        return self.values

    def numel(self):
        size = 1
        for dimension in self.shape:
            size *= dimension
        return size

    def element_size(self):
        return {"torch.bfloat16": 2, "torch.int32": 4, "torch.int64": 8}[self.dtype]


class FakeModule:
    def __init__(self):
        self.pre_hooks = []
        self.post_hooks = []

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        entry = (hook, with_kwargs)
        self.pre_hooks.append(entry)
        return SimpleNamespace(remove=lambda: self.pre_hooks.remove(entry))

    def register_forward_hook(self, hook):
        self.post_hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.post_hooks.remove(hook))

    def __call__(self, *args, **kwargs):
        for hook, with_kwargs in self.pre_hooks:
            returned = hook(self, args, kwargs) if with_kwargs else hook(self, args)
            assert returned is None
        output = self.forward(*args, **kwargs)
        for hook in self.post_hooks:
            assert hook(self, args, output) is None
        return output


class UnquantizedLinearMethod:
    pass


class FakeQKV(FakeModule):
    def __init__(self):
        super().__init__()
        self.weight = FakeTensor([[2.0, 0.0], [0.0, 2.0]])
        self.bias = None
        self.skip_bias_add = False
        self.quant_method = UnquantizedLinearMethod()

    def forward(self, hidden_states):
        return hidden_states.transform(lambda values: [[2 * value for value in row] for row in values]), None


class FakeLayer(FakeModule):
    def __init__(self, projection):
        super().__init__()
        self.input_layernorm = SimpleNamespace()
        self.self_attn = SimpleNamespace(qkv_proj=projection)

    def forward(self, positions, hidden_states, residual=None):
        normalized = hidden_states.transform(lambda values: [[value + 1 for value in row] for row in values])
        return self.self_attn.qkv_proj(normalized)[0]


class FakeModel(FakeModule):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.model = SimpleNamespace(layers=[layer], start_layer=0)
        self.logits_processor = FakeModule()

    def forward(self, input_ids=None, positions=None, intermediate_tensors=None, inputs_embeds=None):
        embedded = input_ids.transform(lambda values: [[float(token), 2.0] for token in values])
        embedded.dtype = "torch.bfloat16"
        return self.layer(positions, embedded, None)


class FakeRunner:
    def __init__(self, layer, context):
        self.layer = layer
        self.context = context
        self.model = FakeModel(layer)
        self.input_dtype = "torch.int64"
        self.captured = {}
        self.graph_buffers = None
        self.native = "synthetic-conv-1-0123abcd"
        self.state = SimpleNamespace(num_prompt_tokens=149, prompt_token_ids=[12] * 149,
                                     output_token_ids=[13] * 53)
        unrelated = SimpleNamespace(num_prompt_tokens=37, prompt_token_ids=[7] * 37,
                                    output_token_ids=[8] * 3)
        self.input_batch = SimpleNamespace(req_ids=["unrelated", self.native])
        self.requests = {self.native: self.state, "unrelated": unrelated}
        self.optimistic_seq_lens_cpu = [40, 202]
        self.discard_request_mask = SimpleNamespace(np=[False, False])
        self.query_start_loc = SimpleNamespace(np=[0, 1, 2])

    def _model_forward(self, input_ids=None, positions=None, intermediate_tensors=None, inputs_embeds=None):
        if self.context.cudagraph_runtime_mode.name == "FULL":
            descriptor = self.context.batch_descriptor
            graph = self.captured[(descriptor.num_tokens, descriptor.uniform_decode)]
            graph["input_ids"].copy_(input_ids)
            graph["positions"].copy_(positions)
            for kind, operation in graph["operations"]:
                operation()
            return graph["output"]
        return self.model(input_ids=input_ids, positions=positions,
                          intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds)

    def get_model(self):
        return self.model

    def _dummy_run(self, input_ids, positions):
        return self.model(input_ids=input_ids, positions=positions,
                          intermediate_tensors=None, inputs_embeds=None)

    def _warmup_and_capture(self, desc, cudagraph_runtime_mode):
        self.context.batch_descriptor = desc
        self.context.cudagraph_runtime_mode.name = "NONE"
        supplied = {"input_ids": FakeTensor([0] * desc.num_tokens, self.input_dtype),
                    "positions": FakeTensor([0] * desc.num_tokens, "torch.int64")}
        self._dummy_run(**supplied)
        self.context.cudagraph_runtime_mode.name = cudagraph_runtime_mode.name
        graph = {**supplied, "operations": []}
        self.torch.capturing = True
        FakeTensor.capture_operations = graph["operations"]
        try:
            graph["output"] = self._dummy_run(**supplied)
        finally:
            self.torch.capturing = False
            FakeTensor.capture_operations = None
        self.captured[(desc.num_tokens, desc.uniform_decode)] = graph


@pytest.fixture
def worker_module(monkeypatch):
    torch = ModuleType("torch")
    torch.uint8 = "torch.uint8"
    torch.int64 = "torch.int64"
    torch.capturing = False
    torch.cuda = SimpleNamespace(is_current_stream_capturing=lambda: torch.capturing)

    def zeros(shape, dtype, device):
        assert not torch.capturing and shape == () and dtype == "torch.int64" and device.type == "cuda"
        return FakeTensor(0, dtype)

    torch.zeros = zeros
    torch.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(
        allow_bf16_reduced_precision_reduction=False,
        allow_bf16_reduced_precision_reduction_split_k=True)))
    context = SimpleNamespace(cudagraph_runtime_mode=SimpleNamespace(name="NONE"), batch_descriptor=None)
    forward = ModuleType("vllm.forward_context")
    forward.get_forward_context = lambda: context
    parallel = ModuleType("vllm.distributed.parallel_state")
    parallel.get_ep_group = lambda: SimpleNamespace(world_size=2)
    base = ModuleType("head_hidden_worker")

    class FakeBase:
        def initialize_from_config(self, config):
            assert self.model_runner.model.pre_hooks
            assert self.model_runner.layer.pre_hooks
            assert self.model_runner.layer.self_attn.qkv_proj.pre_hooks
            assert self.model_runner.layer.self_attn.qkv_proj.post_hooks
            self.parent_initialized = True

    base.HeadHiddenVerificationWorker = FakeBase
    for name, module in (("torch", torch), ("vllm.forward_context", forward),
                         ("vllm.distributed.parallel_state", parallel), ("head_hidden_worker", base)):
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(probe.__file__).with_name("upstream_first_layer_worker.py")
    spec = importlib.util.spec_from_file_location("first_layer_worker_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.source_sha = lambda subject: "a" * 64
    return module, torch, context


@pytest.fixture
def buffers(worker_module):
    module, torch, context = worker_module
    projection = FakeQKV()
    layer = FakeLayer(projection)
    runner = FakeRunner(layer, context)
    runner.torch = torch
    bank = module.FirstLayerBuffers(runner, layer, projection, module.projection_record(projection),
                                    context_reader=lambda: context)
    bank.install()
    return bank, runner, torch, context


def inputs(padded=2):
    return {"input_ids": FakeTensor([7, 13] + [0] * (padded - 2), "torch.int64"),
            "positions": FakeTensor([39, 201] + [0] * (padded - 2), "torch.int64")}


def graph_key(bank, context):
    return bank.save.__func__.__globals__["buffer_key"](
        {"mode": "FULL", "padded": context.batch_descriptor.num_tokens,
         "descriptor": {"num_tokens": context.batch_descriptor.num_tokens,
                        "uniform_decode": context.batch_descriptor.uniform_decode}})


def capture(bank, runner, torch, context, padded=8, uniform=True):
    runner.torch = torch
    runner._warmup_and_capture(Descriptor(padded, uniform), SimpleNamespace(name="FULL"))
    runner.graph_buffers = bank.graphs[graph_key(bank, context)]["buffers"]


def test_capture_clones_are_device_only_and_replay_does_not_repeat_hooks(buffers):
    bank, runner, torch, context = buffers
    FakeTensor.synchronizations = 0
    capture(bank, runner, torch, context)
    assert FakeTensor.synchronizations == 0
    assert set(bank.graphs[graph_key(bank, context)]["buffers"]) == set(probe.BUFFER_NAMES)
    result = runner._model_forward(**inputs(8))
    assert result is runner.captured[(8, True)]["output"]
    first = bank.read(1, runner.state)
    assert first["current_forward_hook_calls"] == 0
    assert first["buffers"]["qkv_input"]["values"] == [14.0, 3.0]
    assert first["read_execution"] == "graph_replay"
    assert first["refresh_witness"]["generation_after"] == first["refresh_witness"]["generation_before"] + 1
    assert first["input_binding"]["input_ids"]["value"] == 13
    assert first["input_binding"]["positions"]["value"] == 201
    bank.frame = None
    runner.state.output_token_ids.append(14)
    runner.optimistic_seq_lens_cpu[1] += 1
    changed = inputs(8)
    changed["input_ids"].values[1] = 14
    changed["positions"].values[1] = 202
    runner._model_forward(**changed)
    second = bank.read(1, runner.state)
    assert second["buffers"]["qkv_input"]["values"] == [15.0, 3.0]
    assert second["buffer_hook_calls"] == {name: 1 for name in probe.BUFFER_NAMES}
    assert second["refresh_witness"]["generation_before"] == first["refresh_witness"]["generation_after"]


def test_returning_old_graph_output_without_replay_is_rejected(buffers):
    bank, runner, torch, context = buffers
    capture(bank, runner, torch, context)
    bank.original_forward = lambda **kwargs: runner.graph_buffers["qkv_output"]
    runner._model_forward(**inputs(8))
    with pytest.raises(RuntimeError, match="generation did not advance exactly once"):
        bank.read(1, runner.state)


@pytest.mark.parametrize("removed", ["clone", "generation"])
def test_replay_requires_captured_clone_operations_and_generation_increment(buffers, removed):
    bank, runner, torch, context = buffers
    capture(bank, runner, torch, context)
    graph = runner.captured[(8, True)]
    graph["operations"] = [(kind, operation) for kind, operation in graph["operations"] if kind != removed]
    runner._model_forward(**inputs(8))
    expected = "input is stale" if removed == "clone" else "generation did not advance exactly once"
    with pytest.raises(RuntimeError, match=expected):
        bank.read(1, runner.state)


def test_second_replay_cannot_reuse_first_replay_input_clones(buffers):
    bank, runner, torch, context = buffers
    capture(bank, runner, torch, context)
    runner._model_forward(**inputs(8))
    bank.read(1, runner.state)
    bank.frame = None
    runner.state.output_token_ids.append(14)
    runner.optimistic_seq_lens_cpu[1] += 1
    supplied = inputs(8)
    supplied["input_ids"].values[1] = 14
    supplied["positions"].values[1] = 202
    graph = runner.captured[(8, True)]
    graph["operations"] = [(kind, operation) for kind, operation in graph["operations"] if kind != "clone"]
    runner._model_forward(**supplied)
    with pytest.raises(RuntimeError, match="input is stale"):
        bank.read(1, runner.state)


def test_multiple_graph_replays_before_logits_are_not_one_forward(buffers):
    bank, runner, torch, context = buffers
    capture(bank, runner, torch, context)
    original = bank.original_forward

    def replay_twice(**kwargs):
        original(**kwargs)
        return original(**kwargs)

    bank.original_forward = replay_twice
    runner._model_forward(**inputs(8))
    with pytest.raises(RuntimeError, match="generation did not advance exactly once"):
        bank.read(1, runner.state)


def test_same_padding_distinct_descriptors_have_independent_graph_witnesses(buffers):
    bank, runner, torch, context = buffers
    capture(bank, runner, torch, context)
    first_key = graph_key(bank, context)
    capture(bank, runner, torch, context, uniform=False)
    second_key = graph_key(bank, context)
    assert first_key != second_key and len(bank.graphs) == 2
    assert bank.graphs[first_key]["counter"] is not bank.graphs[second_key]["counter"]
    runner._model_forward(**inputs(8))
    assert bank.read(1, runner.state)["refresh_witness"]["generation_after"] == 2
    assert bank.graphs[first_key]["counter"].item() == 1
    bank.frame = None
    context.batch_descriptor = Descriptor(8)
    runner._model_forward(**inputs(8))
    assert bank.read(1, runner.state)["refresh_witness"]["generation_after"] == 2
    assert bank.graphs[second_key]["counter"].item() == 2


def test_different_padded_graphs_do_not_share_input_clones_or_counters(buffers):
    bank, runner, torch, context = buffers
    capture(bank, runner, torch, context, padded=1)
    first_key = graph_key(bank, context)
    capture(bank, runner, torch, context, padded=8)
    second_key = graph_key(bank, context)
    assert bank.graphs[first_key]["inputs"]["input_ids"].shape == (1,)
    assert bank.graphs[second_key]["inputs"]["input_ids"].shape == (8,)
    assert bank.graphs[first_key]["counter"] is not bank.graphs[second_key]["counter"]


def test_unregistered_descriptor_cannot_use_another_descriptor_graph(buffers):
    bank, runner, torch, context = buffers
    capture(bank, runner, torch, context)
    context.batch_descriptor = Descriptor(8, False)
    with pytest.raises(RuntimeError, match="Missing captured graph buffers"):
        runner._model_forward(**inputs(8))


def test_capture_requires_counter_initialized_outside_capture(buffers):
    bank, runner, torch, context = buffers
    context.cudagraph_runtime_mode.name = "FULL"
    context.batch_descriptor = Descriptor(8)
    torch.capturing = True
    try:
        with pytest.raises(RuntimeError, match="outside-capture graph generation initialization"):
            runner._dummy_run(**inputs(8))
        with pytest.raises(RuntimeError, match="initialized outside capture"):
            runner._warmup_and_capture(Descriptor(8), SimpleNamespace(name="FULL"))
    finally:
        torch.capturing = False


def test_eager_actual_inputs_cannot_mask_stale_model_input_clones(buffers):
    bank, runner, torch, context = buffers
    runner._model_forward(**inputs())
    bank.eager["inputs"]["input_ids"].values[1] = 99
    with pytest.raises(RuntimeError, match="input is stale"):
        bank.read(1, runner.state)


@pytest.mark.parametrize("mode", ["NONE", "FULL"])
def test_installed_int32_token_ids_preserve_actual_input_binding(buffers, mode):
    bank, runner, torch, context = buffers
    runner.input_dtype = "torch.int32"
    padded = 8 if mode == "FULL" else 2
    if mode == "FULL":
        capture(bank, runner, torch, context)
    supplied = inputs(padded)
    supplied["input_ids"].dtype = "torch.int32"
    runner._model_forward(**supplied)
    record = bank.read(1, runner.state)
    assert record["input_binding"]["input_ids"]["dtype"] == "torch.int32"
    assert record["input_binding"]["input_ids"]["value"] == 13
    assert record["input_binding"]["positions"]["dtype"] == "torch.int64"


def test_int32_token_support_does_not_relax_position_dtype(buffers):
    bank, runner, torch, context = buffers
    supplied = inputs()
    supplied["positions"].dtype = "torch.int32"
    with pytest.raises(RuntimeError):
        runner._model_forward(**supplied)


def test_eager_sequence_cannot_reuse_previous_forward_buffers(buffers):
    bank, runner, torch, context = buffers
    runner._model_forward(**inputs())
    bank.eager["sequence"] -= 1
    with pytest.raises(RuntimeError, match="Eager hook coverage differs"):
        bank.read(1, runner.state)


def test_selected_forward_and_snapshot_reads_never_sync_inside_capture(buffers):
    bank, runner, torch, context = buffers
    capture(bank, runner, torch, context)
    runner._model_forward(**inputs(8))
    FakeTensor.synchronizations = 0
    torch.capturing = True
    try:
        with pytest.raises(RuntimeError, match="binding must be outside capture"):
            runner._model_forward(**inputs(8))
        with pytest.raises(RuntimeError, match="read must be outside capture"):
            bank.read(1, runner.state)
        assert FakeTensor.synchronizations == 0
    finally:
        torch.capturing = False


def test_eager_buffers_never_replace_graph_buffers_or_grow_history(buffers):
    bank, runner, torch, context = buffers
    capture(bank, runner, torch, context)
    key = graph_key(bank, context)
    graph = bank.graphs[key]
    context.cudagraph_runtime_mode.name = "NONE"
    context.batch_descriptor = None
    for repeat in range(4):
        runner._model_forward(**inputs())
        snapshot = bank.read(1, runner.state)
        assert snapshot["buffer_origin"] == "eager_forward"
        assert snapshot["buffers"]["layer_input"]["shape"] == [2, 2]
        assert bank.graphs[key] is graph
        assert bank.device_bytes == (8 + 2) * (2 * 2 * 3 + 16) + 8
        assert snapshot["refresh_witness"]["kind"] == "eager_forward_sequence"
        assert snapshot["input_binding"]["input_ids"]["value"] == 13
        bank.frame = None
    assert len(bank.graphs) == 1


def test_partial_discard_never_reads_cpu_or_emits_snapshot(buffers):
    bank, runner, torch, context = buffers
    runner.discard_request_mask.np[1] = True
    runner.optimistic_seq_lens_cpu[1] = 150
    runner._model_forward(**inputs())
    assert bank.eager is None
    bank.frame["pending"] = True
    FakeTensor.synchronizations = 0
    assert bank.read(1, runner.state) is None
    assert FakeTensor.synchronizations == 0


@pytest.mark.parametrize("mutation", ["prefix", "query", "position", "token", "batch", "shape"])
def test_row_complete_prefix_and_buffer_shape_fail_closed(buffers, mutation):
    bank, runner, torch, context = buffers
    supplied = inputs()
    if mutation == "position":
        supplied["positions"].values[1] = 200
    if mutation == "token":
        supplied["input_ids"].values[1] = 99
    runner._model_forward(**supplied)
    if mutation == "prefix":
        runner.optimistic_seq_lens_cpu[1] = 201
    if mutation == "query":
        runner.query_start_loc.np = [0, 1, 3]
    if mutation == "batch":
        runner.input_batch.req_ids.reverse()
    if mutation == "shape":
        bank.eager["buffers"]["qkv_input"].shape = (1, 2)
    with pytest.raises(RuntimeError):
        bank.read(1, runner.state)


@pytest.mark.parametrize("mode,descriptor,padded", [
    ("PIECEWISE", None, 2), ("FULL", None, 8), ("FULL", Descriptor(8), 2),
    ("FULL", Descriptor(2), 2), ("NONE", SimpleNamespace(num_tokens=2), 2),
    ("NONE", Descriptor(3), 2), ("NONE", None, 4097),
])
def test_unsupported_mode_and_descriptor_fail_closed(worker_module, mode, descriptor, padded):
    module, torch, context = worker_module
    context.cudagraph_runtime_mode.name = mode
    context.batch_descriptor = descriptor
    with pytest.raises(RuntimeError):
        module.execution_metadata(context, padded)


def test_unknown_installed_context_attribute_is_not_guessed(worker_module):
    module, torch, context = worker_module
    with pytest.raises(AttributeError):
        module.execution_metadata(SimpleNamespace(runtime_mode="FULL"), 8)


def test_graph_cannot_use_eager_only_buffers(buffers):
    bank, runner, torch, context = buffers
    runner._model_forward(**inputs())
    bank.frame = None
    context.cudagraph_runtime_mode.name = "FULL"
    context.batch_descriptor = Descriptor(8)
    runner.graph_buffers = bank.eager["buffers"]
    with pytest.raises(RuntimeError, match="captured graph buffers"):
        runner._model_forward(**inputs(8))


def test_duplicate_capture_key_is_rejected_instead_of_retaining_gpu_history(buffers):
    bank, runner, torch, context = buffers
    capture(bank, runner, torch, context)
    with pytest.raises(RuntimeError, match="Duplicate/ambiguous"):
        capture(bank, runner, torch, context)


def test_capture_budget_is_enforced_before_clone(buffers, monkeypatch):
    bank, runner, torch, context = buffers
    monkeypatch.setitem(bank.save.__func__.__globals__, "MAX_DEVICE_BYTES", 1)
    FakeTensor.synchronizations = 0
    with pytest.raises(RuntimeError, match="budget exceeded"):
        capture(bank, runner, torch, context)
    assert bank.device_bytes == 0 and FakeTensor.synchronizations == 0


@pytest.mark.parametrize("budget,retained", [(96, 8), (136, 136), (231, 200)])
def test_budget_counts_input_clones_and_generation_counter_before_allocation(buffers, monkeypatch, budget, retained):
    bank, runner, torch, context = buffers
    monkeypatch.setitem(bank.save.__func__.__globals__, "MAX_DEVICE_BYTES", budget)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        capture(bank, runner, torch, context)
    assert bank.device_bytes == retained <= budget


@pytest.mark.parametrize("interface", ["raw_model", "warmup", "actual_forward"])
def test_unknown_installed_private_interfaces_are_rejected(worker_module, interface):
    module, torch, context = worker_module
    projection = FakeQKV()
    layer = FakeLayer(projection)
    runner = FakeRunner(layer, context)
    if interface == "raw_model":
        runner.model.forward = lambda hidden_states: hidden_states
    elif interface == "warmup":
        runner._warmup_and_capture = lambda padded: None
    else:
        runner._model_forward = lambda hidden_states: hidden_states
    with pytest.raises(RuntimeError, match="Unsupported installed"):
        module.FirstLayerBuffers(runner, layer, projection, module.projection_record(projection))
    assert not runner.model.pre_hooks and not layer.pre_hooks and not projection.pre_hooks


def test_close_removes_only_new_hooks_and_restores_runner(buffers):
    bank, runner, torch, context = buffers
    original = bank.original_forward
    original_warmup = bank.original_warmup
    bank.close()
    assert runner._model_forward == original
    assert runner._warmup_and_capture == original_warmup and not runner.model.pre_hooks
    assert not runner.layer.pre_hooks and not runner.layer.self_attn.qkv_proj.pre_hooks
    assert not runner.layer.self_attn.qkv_proj.post_hooks and not bank.graphs


def test_worker_hooks_are_installed_before_parent_initialize_and_direct_model_capture(worker_module, tmp_path, monkeypatch):
    module, torch, context = worker_module
    projection = FakeQKV()
    layer = FakeLayer(projection)
    runner = FakeRunner(layer, context)
    processor = FakeModule()
    model = runner.model
    model.logits_processor = processor
    runner.torch = torch
    worker = module.UpstreamFirstLayerVerificationWorker()
    worker.model_runner = runner
    worker.model_config = SimpleNamespace(model="same-checkpoint", dtype="torch.bfloat16")
    worker.parallel_config = SimpleNamespace(data_parallel_rank=0, tensor_parallel_size=1,
        data_parallel_size=2, pipeline_parallel_size=1, enable_expert_parallel=True,
        use_ubatching=False, use_sequence_parallel_moe=False)
    worker.vllm_config = SimpleNamespace(speculative_config=None, compilation_config=SimpleNamespace(
        mode=0, cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"), cudagraph_capture_sizes=probe.CAPTURES))
    worker.scheduler_config = SimpleNamespace(async_scheduling=False)
    monkeypatch.setenv("DP_OPT_EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.setenv("DP_OPT_SERVICE", "instance-2")
    initialize = module.HeadHiddenVerificationWorker.initialize_from_config

    def initialize_and_capture(self, config):
        initialize(self, config)
        self.model_runner._warmup_and_capture(Descriptor(8), SimpleNamespace(name="FULL"))

    monkeypatch.setattr(module.HeadHiddenVerificationWorker, "initialize_from_config", initialize_and_capture)
    worker.initialize_from_config(None)
    assert worker.parent_initialized
    assert len(layer.pre_hooks) == 1 and len(projection.post_hooks) == 1
    assert (tmp_path / "upstream-first-layer-worker-instance-2-dp0.json").exists()
    assert len(processor.post_hooks) == 1
    group = worker._first_layer_probe.graphs[graph_key(worker._first_layer_probe, context)]
    assert group["increment_calls"] == 1 and group["counter"].item() == 1
    assert worker._first_layer_probe.sequence == 0


def worker_record(service="instance-2", rank=0):
    return {"schema": probe.SCHEMA, "service": service, "dp_rank": rank,
        "checkpoint": "same-checkpoint", "model_class": "vllm.Qwen3MoeForCausalLM",
        "attention_tp": 1, "dp": 2, "ep": 2, "pp": 1, "ep_enabled": True,
        "model_dtype": "torch.bfloat16", "compile_mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": probe.CAPTURES, "speculative": False, "async_scheduling": False,
        "use_ubatching": False, "use_sequence_parallel_moe": False, "batch_invariant_enabled": False,
        "layer_index": 0, "buffer_names": list(probe.BUFFER_NAMES), "max_device_bytes": probe.MAX_DEVICE_BYTES,
        "scientific_scope": probe.SCIENTIFIC_SCOPE,
        "source_sha256": {name: "a" * 64 for name in ("runner", "context", "layer", "qkv")},
        "projection": {"weight_shape": [2, 2], "weight_dtype": "torch.bfloat16", "weight_device": "cuda",
                       "weight_sha256": "b" * 64, "quant_method": "UnquantizedLinearMethod",
                       "skip_bias_add": False, "bias": None}, "is_algorithm_benefit": False}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def analysis_root(tmp_path, monkeypatch):
    logical = "synthetic-conv-1"
    prompt = [12] * 149
    serial_tokens = [13] * 56
    trial_tokens = list(serial_tokens)
    serial_tokens[53], trial_tokens[53] = 20632, 12822
    frames = [{"12822": -1.0, "20632": -1.5}] * 56
    serial_frames = [{"12822": -1.5, "20632": -1.0}] * 56
    reference = {"trace_request_id": logical, "request_id": "reference-1", "model": 1,
                 "rank_initial": 0, "prompt_tokens": 149, "token_ids": serial_tokens,
                 "token_logprobs": serial_frames}
    actual = {"request_id": logical, "model": 1, "rank_initial": 0, "prompt_tokens": 149,
              "token_ids": trial_tokens, "token_logprobs": frames}
    write_json(tmp_path / "references.json", [reference])
    write_json(tmp_path / "r0-no_move/requests.json", [actual])
    bank = tmp_path / "frozen/inputs/prompt-bank.json"
    write_json(bank, {"prompts": {"public-conv": {"token_ids": prompt}}})
    import hashlib
    write_json(tmp_path / "trace.json", {"requests": [{"request_id": logical, "prompt_key": "public-conv"}],
                                       "prompt_bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest()})
    for service in ("instance-1", "instance-2"):
        for rank in (0, 1):
            write_json(tmp_path / f"upstream-first-layer-worker-{service}-dp{rank}.json", worker_record(service, rank))
            write_json(tmp_path / f"worker-{service}-dp{rank}.json", {"service": service, "dp_rank": rank,
                "checkpoint": "same-checkpoint", "groups": {group: {"world_size": size,
                    "rank": rank if group == "dp" else 0} for group, size in (("tp", 1), ("dp", 2), ("ep", 2))}})
    for role, request, native in (("reference", reference, "reference-1-0123abcd"),
                                  ("trial", actual, logical + "-0123abcd")):
        for position in (52, 53, 54):
            prefix = request["token_ids"][:position]
            batch = [native] if role == "reference" else ["other", native]
            row, padded = (0, 1) if role == "reference" else (1, 8)
            head = {"service": "instance-2", "dp_rank": 0, "native_request_id": native,
                "role": role, "repeat": None if role == "reference" else 0,
                "layout": None if role == "reference" else "corank0", "position": position,
                "prompt_tokens": 149, "prompt_token_ids": prompt, "output_prefix_token_ids": prefix,
                "context_token_ids": prompt + prefix, "context_sha256": head_hidden_probe.context_digest(prompt + prefix),
                "forward_consumed_prefix_tokens": 149 + position, "sample_discarded": False,
                "batch_rows": len(batch), "batch_native_request_ids": batch, "row_index": row,
                "hidden_dtype": "torch.bfloat16", "logits_dtype": "torch.float32", "hidden": [1.0, 2.0],
                "allow_bf16_reduced_precision_reduction": False,
                "allow_bf16_reduced_precision_reduction_split_k": True,
                "candidates": [12822, 20632],
                "candidate_logits": [1.0, 1.5] if role == "reference" else [1.5, 1.0]}
            write_json(tmp_path / f"head-hidden-snapshots/{role}-p{position}.json", head)
            identity = worker_record()
            record = {name: head[name] for name in head if name not in
                      ("hidden_dtype", "logits_dtype", "hidden", "candidates", "candidate_logits")}
            record.update({"schema": probe.SCHEMA, "checkpoint": identity["checkpoint"],
                "scientific_scope": probe.SCIENTIFIC_SCOPE,
                "model_class": identity["model_class"], "projection": identity["projection"],
                "source_sha256": identity["source_sha256"], "layer_index": 0,
                "first_layer_residual_is_none": True, "observation_includes_CPU_synchronization": True,
                "is_algorithm_benefit": False, "num_tokens_padded": padded,
                "query_start_loc": list(range(len(batch) + 1)), "token_row_index": row,
                "forward_position": 149 + position - 1, "forward_token_id": prefix[-1],
                "forward_mode": "FULL", "buffer_origin": "cuda_graph_capture",
                "buffer_key": ["FULL", padded, {"num_tokens": padded}],
                "buffer_hook_calls": {name: 1 for name in probe.BUFFER_NAMES},
                "current_forward_hook_calls": 0, "retained_device_bytes": padded * 28 + 8,
                "forward_descriptor": {"num_tokens": padded}, "capture_descriptor": {"num_tokens": padded},
                "read_execution": "graph_replay", "matmul_flags": {
                    "allow_bf16_reduced_precision_reduction": False,
                    "allow_bf16_reduced_precision_reduction_split_k": True}, "buffers": {}})
            record["refresh_witness"] = {"kind": "cuda_graph_generation", "forward_sequence": position + 1,
                "generation_before": position + 1, "generation_after": position + 2, "capture_increment_calls": 1,
                "counter": {"shape": [], "dtype": "torch.int64", "device": "cuda"}}
            record["input_binding"] = {"origin": "cuda_graph_capture", "hook_calls": 1,
                "input_ids": {"shape": [padded], "dtype": "torch.int64", "device": "cuda",
                              "token_row_index": row, "value": prefix[-1]},
                "positions": {"shape": [padded], "dtype": "torch.int64", "device": "cuda",
                              "token_row_index": row, "value": 149 + position - 1}}
            for name in probe.BUFFER_NAMES:
                values = [2.0, 3.0] if name != "qkv_output" or role == "reference" else [2.0, 3.015625]
                record["buffers"][name] = {"shape": [padded, 2], "dtype": "torch.bfloat16", "device": "cuda",
                    "token_row_index": row, "values": values, "row_sha256": probe.bf16_digest(values)}
            write_json(tmp_path / f"upstream-first-layer-snapshots/{role}-p{position}.json", record)
    base = {"full_static_analysis": {"strict_correctness_passed": False, "diagnostic": {"trials": [
        {"repeat": 0, "layout": "corank0", "serial_comparisons": [
            {"request_id": logical, "first_difference_zero_based": 53}]}]}}}
    monkeypatch.setattr(head_hidden_probe, "analyze", lambda root: copy.deepcopy(base))
    return tmp_path


def mutate_snapshot(root, field, value, name="trial-p53.json"):
    path = root / "upstream-first-layer-snapshots" / name
    record = json.loads(path.read_text())
    record[field] = value
    write_json(path, record)


def test_analyzer_labels_affine_observation_without_batch_or_kernel_proof(analysis_root):
    result = probe.analyze(analysis_root)
    comparison = result["comparisons"][0]
    assert comparison["actual_first_mismatch"] == 53 and comparison["is_first_mismatch"]
    assert comparison["affine_batch_sensitivity_observed"]
    assert not comparison["affine_batch_sensitivity_proven"]
    assert comparison["current_step_only"]
    assert not comparison["numerical_root_cause_proven"]
    assert not comparison["historical_KV_equivalence_proven"]
    assert not comparison["specific_cublas_algorithm_proven"]
    assert not comparison["downstream_first_top1_causality_proven"]
    assert not result["quality_equivalence_proven"] and not result["is_algorithm_benefit"]
    assert not result["strict_token_check_relaxed"]
    assert result["validated_snapshots"] == 6


def test_analyzer_accepts_actual_int32_id_storage_and_budget(analysis_root):
    for path in (analysis_root / "upstream-first-layer-snapshots").glob("*.json"):
        record = json.loads(path.read_text())
        record["input_binding"]["input_ids"]["dtype"] = "torch.int32"
        record["retained_device_bytes"] -= record["num_tokens_padded"] * 4
        write_json(path, record)
    assert probe.analyze(analysis_root)["validated_snapshots"] == 6


def test_repeated_serial_samples_preserve_each_native_execution_binding(analysis_root):
    path = analysis_root / "references.json"
    records = json.loads(path.read_text())
    repeated = copy.deepcopy(records[0])
    repeated["request_id"] = "reference-2"
    write_json(path, [repeated, *records])
    references, bindings = probe.reference_bindings([repeated, *records])
    assert len(references) == 1 and len(bindings) == 2
    assert [binding[4]["request_id"] for binding in bindings] == ["reference-2", "reference-1"]
    assert probe.analyze(analysis_root)["validated_snapshots"] == 6


@pytest.mark.parametrize("field,value", [
    ("model", 0), ("model", True), ("rank_initial", 1), ("rank_initial", False),
    ("prompt_tokens", 150), ("token_ids", [1]),
    ("request_id", "reference-1"), ("request_id", ""), ("request_id", None),
])
def test_repeated_serial_samples_reject_changed_content_or_native_identity(analysis_root, field, value):
    records = json.loads((analysis_root / "references.json").read_text())
    repeated = copy.deepcopy(records[0])
    repeated["request_id"] = "reference-2"
    repeated[field] = value
    with pytest.raises(ValueError):
        probe.reference_bindings([*records, repeated])


def test_repeated_serial_samples_do_not_relax_snapshot_uniqueness(analysis_root):
    path = analysis_root / "references.json"
    records = json.loads(path.read_text())
    repeated = copy.deepcopy(records[0])
    repeated["request_id"] = "reference-2"
    write_json(path, [repeated, *records])
    for directory in ("head-hidden-snapshots", "upstream-first-layer-snapshots"):
        source = analysis_root / directory / "reference-p53.json"
        record = json.loads(source.read_text())
        record["native_request_id"] = "reference-2-0123abcd"
        if "batch_native_request_ids" in record:
            record["batch_native_request_ids"] = ["reference-2-0123abcd"]
        write_json(source.with_name("duplicate-reference-p53.json"), record)
    with pytest.raises(ValueError, match="Duplicate first-layer request/position snapshot"):
        probe.analyze(analysis_root)


def test_cross_gpu_current_step_observation_is_not_batch_kernel_or_root_cause_proof(analysis_root):
    directory = analysis_root / "upstream-first-layer-snapshots"
    reference = json.loads((directory / "reference-p53.json").read_text())
    actual = json.loads((directory / "trial-p53.json").read_text())
    actual["dp_rank"] = 1
    result = probe.compare(reference, actual)
    assert result["affine_batch_sensitivity_observed"]
    assert result["conclusion"] == "qkv_affine_difference_observed_with_batch_or_graph_change_not_causal_proof"
    for name in ("affine_batch_sensitivity_proven", "specific_cublas_algorithm_proven",
                 "downstream_first_top1_causality_proven", "numerical_root_cause_proven",
                 "historical_KV_equivalence_proven"):
        assert result[name] is False


def test_identical_current_qkv_does_not_exclude_historical_KV_or_later_layers(analysis_root):
    directory = analysis_root / "upstream-first-layer-snapshots"
    reference = json.loads((directory / "reference-p53.json").read_text())
    actual = json.loads((directory / "trial-p53.json").read_text())
    actual["buffers"] = copy.deepcopy(reference["buffers"])
    result = probe.compare(reference, actual)
    assert result["conclusion"] == "no_first_projection_difference_observed"
    assert not result["affine_batch_sensitivity_observed"]
    assert not result["historical_KV_equivalence_proven"] and not result["numerical_root_cause_proven"]


@pytest.mark.parametrize("field,value", [
    ("generation_after", 1), ("generation_after", 1000), ("generation_before", True),
    ("generation_after", True), ("generation_before", -1), ("forward_sequence", 0),
    ("forward_sequence", True), ("capture_increment_calls", 0), ("capture_increment_calls", True),
    ("kind", "eager_forward_sequence"), ("counter", None),
    ("counter", {"shape": [], "dtype": "torch.int64", "device": "cpu"}),
])
def test_analyzer_rejects_missing_or_invalid_replay_generation_witness(analysis_root, field, value):
    path = analysis_root / "upstream-first-layer-snapshots/trial-p53.json"
    record = json.loads(path.read_text())
    record["refresh_witness"][field] = value
    write_json(path, record)
    with pytest.raises(ValueError):
        probe.analyze(analysis_root)


@pytest.mark.parametrize("name,field,value", [
    ("input_ids", "value", 99), ("positions", "value", 200), ("input_ids", "token_row_index", 0),
    ("positions", "token_row_index", True), ("input_ids", "dtype", "torch.bfloat16"),
    ("positions", "device", "cpu"), ("input_ids", "shape", [1]),
])
def test_analyzer_rejects_stale_or_unbound_captured_token_and_position(analysis_root, name, field, value):
    path = analysis_root / "upstream-first-layer-snapshots/trial-p53.json"
    record = json.loads(path.read_text())
    record["input_binding"][name][field] = value
    write_json(path, record)
    with pytest.raises(ValueError):
        probe.analyze(analysis_root)


@pytest.mark.parametrize("field", ["input_binding", "refresh_witness"])
def test_unwitnessed_old_snapshot_cannot_claim_verified_replay(analysis_root, field):
    path = analysis_root / "upstream-first-layer-snapshots/trial-p53.json"
    record = json.loads(path.read_text())
    del record[field]
    write_json(path, record)
    with pytest.raises(KeyError):
        probe.analyze(analysis_root)


def test_analyzer_accepts_fresh_eager_input_binding_without_claiming_graph_replay(analysis_root):
    path = analysis_root / "upstream-first-layer-snapshots/trial-p53.json"
    record = json.loads(path.read_text())
    record.update({"forward_mode": "NONE", "buffer_origin": "eager_forward", "read_execution": "eager_forward",
                   "capture_descriptor": None, "current_forward_hook_calls": 3})
    record["buffer_key"][0] = "NONE"
    record["input_binding"]["origin"] = "eager_forward"
    record["refresh_witness"].update({"kind": "eager_forward_sequence", "generation_before": None,
                                      "generation_after": None, "capture_increment_calls": 0, "counter": None})
    write_json(path, record)
    result = probe.analyze(analysis_root)
    assert result["validated_snapshots"] == 6
    assert not result["comparisons"][0]["affine_batch_sensitivity_proven"]
    record["refresh_witness"]["kind"] = "cuda_graph_generation"
    write_json(path, record)
    with pytest.raises(ValueError, match="Eager input binding cannot claim"):
        probe.analyze(analysis_root)


def test_analyzer_budget_cannot_omit_captured_inputs_or_counter(analysis_root):
    mutate_snapshot(analysis_root, "retained_device_bytes", 8 * 12)
    with pytest.raises(ValueError, match="input/refresh witnesses"):
        probe.analyze(analysis_root)


def test_input_difference_is_only_unlocalized_upstream_observation(analysis_root):
    path = analysis_root / "upstream-first-layer-snapshots/trial-p53.json"
    record = json.loads(path.read_text())
    record["buffers"]["qkv_input"]["values"] = [4.0, 3.0]
    record["buffers"]["qkv_input"]["row_sha256"] = probe.bf16_digest([4.0, 3.0])
    write_json(path, record)
    comparison = probe.analyze(analysis_root)["comparisons"][0]
    assert comparison["conclusion"] == "upstream_input_changed_not_localized"
    assert not comparison["affine_batch_sensitivity_proven"]


@pytest.mark.parametrize("field,value", [
    ("native_request_id", "synthetic-conv-10"), ("service", "instance-1"), ("dp_rank", 1),
    ("repeat", 1), ("layout", "corank1"), ("role", "reference"), ("position", 51),
    ("row_index", 0), ("sample_discarded", True), ("forward_consumed_prefix_tokens", 201),
    ("row_index", True), ("repeat", False), ("dp_rank", False),
    ("token_row_index", 0), ("query_start_loc", [0, 1, 9]), ("forward_position", 200),
    ("forward_token_id", 8), ("buffer_key", ["NONE", 8]), ("buffer_origin", "eager_forward"),
    ("current_forward_hook_calls", 3), ("capture_descriptor", {"num_tokens": 16}),
    ("retained_device_bytes", probe.MAX_DEVICE_BYTES + 1), ("first_layer_residual_is_none", False),
    ("retained_device_bytes", 1),
])
def test_analyzer_rejects_identity_prefix_row_mode_or_provenance_errors(analysis_root, field, value):
    mutate_snapshot(analysis_root, field, value)
    with pytest.raises((ValueError, KeyError)):
        probe.analyze(analysis_root)


@pytest.mark.parametrize("field,value", [
    ("shape", [8, 3]), ("dtype", "torch.float32"), ("device", "cpu"),
    ("token_row_index", 0), ("row_sha256", "c" * 64), ("values", [float("nan"), 3.0]),
])
def test_analyzer_rejects_buffer_shape_index_precision_and_value_errors(analysis_root, field, value):
    path = analysis_root / "upstream-first-layer-snapshots/trial-p53.json"
    record = json.loads(path.read_text())
    record["buffers"]["qkv_input"][field] = value
    write_json(path, record)
    with pytest.raises(ValueError):
        probe.analyze(analysis_root)


@pytest.mark.parametrize("field,value", [
    ("service", "instance-1"), ("attention_tp", 2), ("ep", 1), ("model_dtype", "torch.float32"),
    ("cudagraph_mode", "NONE"), ("speculative", True), ("use_ubatching", True),
])
def test_four_worker_identity_and_configuration_must_agree(analysis_root, field, value):
    path = analysis_root / "upstream-first-layer-worker-instance-2-dp1.json"
    record = json.loads(path.read_text())
    record[field] = value
    write_json(path, record)
    with pytest.raises(ValueError):
        probe.analyze(analysis_root)


def test_qkv_weight_or_bias_sha_mismatch_is_rejected(analysis_root):
    path = analysis_root / "upstream-first-layer-worker-instance-2-dp1.json"
    record = json.loads(path.read_text())
    record["projection"]["weight_sha256"] = "c" * 64
    write_json(path, record)
    with pytest.raises(ValueError, match="inconsistent QKV weights"):
        probe.analyze(analysis_root)


def test_actual_worker_ep_group_is_verified(analysis_root):
    path = analysis_root / "worker-instance-2-dp1.json"
    record = json.loads(path.read_text())
    record["groups"]["ep"]["world_size"] = 1
    write_json(path, record)
    with pytest.raises(ValueError, match="actual worker groups"):
        probe.analyze(analysis_root)


def test_missing_fourth_worker_is_not_accepted(analysis_root):
    (analysis_root / "upstream-first-layer-worker-instance-2-dp1.json").unlink()
    with pytest.raises(ValueError, match="all four workers"):
        probe.analyze(analysis_root)


def test_missing_actual_mismatch_buffers_cannot_use_neighboring_positions(analysis_root):
    (analysis_root / "upstream-first-layer-snapshots/trial-p53.json").unlink()
    result = probe.analyze(analysis_root)
    assert not result["all_in_scope_target_attributions_available"]
    assert not result["comparisons"][0]["affine_batch_sensitivity_proven"]


def test_unbound_extra_snapshots_are_not_silently_ignored(analysis_root):
    path = analysis_root / "upstream-first-layer-snapshots/trial-p53.json"
    record = json.loads(path.read_text())
    record["native_request_id"] = "other-request"
    write_json(path.with_name("unknown.json"), record)
    with pytest.raises(ValueError, match="Unbound"):
        probe.analyze(analysis_root)


def test_first_mismatch_is_rebuilt_from_complete_token_sequences(analysis_root):
    path = analysis_root / "r0-no_move/requests.json"
    records = json.loads(path.read_text())
    records[0]["token_ids"][55] = 7
    write_json(path, records)
    assert probe.analyze(analysis_root)["comparisons"][0]["actual_first_mismatch"] == 53
    records[0]["token_ids"][0] = 7
    write_json(path, records)
    with pytest.raises(ValueError, match="complete request prefix|first mismatch"):
        probe.analyze(analysis_root)


def test_matmul_control_difference_does_not_prove_affine_batch_sensitivity(analysis_root):
    mutate_snapshot(analysis_root, "matmul_flags", {
        "allow_bf16_reduced_precision_reduction": True,
        "allow_bf16_reduced_precision_reduction_split_k": True})
    with pytest.raises(ValueError, match="matmul controls"):
        probe.analyze(analysis_root)


def test_bf16_digest_preserves_signed_zero_and_rejects_non_bf16_values():
    assert probe.bf16_digest([0.0]) != probe.bf16_digest([-0.0])
    for values in ([float("inf")], [1.0001], []):
        with pytest.raises(ValueError):
            probe.bf16_digest(values)


def test_offline_cli_never_overwrites_existing_analysis(analysis_root, monkeypatch):
    output = analysis_root / "analysis.json"
    output.write_text("original")
    monkeypatch.setattr(sys, "argv", ["probe", "--verify", str(analysis_root), "--output", str(output)])
    with pytest.raises(FileExistsError):
        probe.main()
    assert output.read_text() == "original"


@pytest.mark.parametrize("prompt,position,expected", [
    (149, 52, 52), (149, 53, 53), (149, 54, 54), (149, 51, None),
    (74, 138, None), (74, 139, None), (74, 140, None),
    (74, 441, None), (74, 442, None), (74, 443, None),
])
def test_scientific_scope_is_only_P149_not_uncovered_P74(prompt, position, expected):
    state = SimpleNamespace(num_prompt_tokens=prompt, output_token_ids=[1] * position)
    assert probe.selected_position(state) == expected
    assert head_hidden_probe.TARGETS[74]["position"] == 442


def test_P74_only_batch_does_not_retain_first_layer_eager_buffers(buffers):
    bank, runner, torch, context = buffers
    runner.state.num_prompt_tokens = 74
    runner.state.prompt_token_ids = [12] * 74
    runner.state.output_token_ids = [13] * 442
    runner.optimistic_seq_lens_cpu[1] = 516
    supplied = inputs()
    supplied["positions"].values[1] = 515
    runner._model_forward(**supplied)
    assert not bank.frame["pending"] and bank.eager is None


def test_forward_prefix_changes_cannot_rebind_identical_last_token_and_position(buffers):
    bank, runner, torch, context = buffers
    runner._model_forward(**inputs())
    runner.state.output_token_ids[0] = 99
    with pytest.raises(RuntimeError, match="prefix changed"):
        bank.read(1, runner.state)


def test_full_mode_eager_python_execution_is_not_claimed_as_graph_replay(buffers):
    bank, runner, torch, context = buffers
    context.cudagraph_runtime_mode.name = "FULL"
    context.batch_descriptor = Descriptor(8)
    bank.frame = {"pending": True, "selected": True}
    with pytest.raises(RuntimeError, match="Python hooks outside capture"):
        bank.save("layer_input", FakeTensor([[1.0, 2.0]] * 8))


@pytest.mark.parametrize("output", [FakeTensor([[1.0, 2.0]]), (FakeTensor([[1.0, 2.0]]), 1)])
def test_unsupported_projection_output_or_deferred_bias_fails_closed(buffers, output):
    bank, runner, torch, context = buffers
    with pytest.raises(RuntimeError, match="output/deferred bias"):
        bank.qkv_output(runner.layer.self_attn.qkv_proj, (), output)


def test_worker_failure_restores_only_its_hooks(worker_module, tmp_path, monkeypatch):
    module, torch, context = worker_module
    projection = FakeQKV()
    layer = FakeLayer(projection)
    runner = FakeRunner(layer, context)
    model = runner.model
    worker = module.UpstreamFirstLayerVerificationWorker()
    worker.model_runner = runner
    worker.model_config = SimpleNamespace(model="same-checkpoint", dtype="torch.bfloat16")
    worker.parallel_config = SimpleNamespace(data_parallel_rank=0, tensor_parallel_size=1,
        data_parallel_size=2, pipeline_parallel_size=1, enable_expert_parallel=True,
        use_ubatching=False, use_sequence_parallel_moe=False)
    worker.vllm_config = SimpleNamespace(speculative_config=None, compilation_config=SimpleNamespace(
        mode=0, cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"), cudagraph_capture_sizes=probe.CAPTURES))
    worker.scheduler_config = SimpleNamespace(async_scheduling=False)
    monkeypatch.setenv("DP_OPT_EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.setenv("DP_OPT_SERVICE", "instance-2")

    def fail_initialize(self, config):
        raise RuntimeError("fake initialization failed")

    monkeypatch.setattr(module.HeadHiddenVerificationWorker, "initialize_from_config", fail_initialize)
    original = runner._model_forward
    with pytest.raises(RuntimeError, match="fake initialization"):
        worker.initialize_from_config(None)
    assert runner._model_forward == original
    assert not layer.pre_hooks and not projection.pre_hooks and not projection.post_hooks


def test_P74_first139_is_explicitly_uncovered_not_later442_attribution(analysis_root, monkeypatch):
    logical = "synthetic-conv-0"
    reference_tokens = [9] * 450
    actual_tokens = list(reference_tokens)
    actual_tokens[139], actual_tokens[442] = 100, 101
    reference = {"trace_request_id": logical, "request_id": "reference-0", "model": 1,
        "rank_initial": 0, "prompt_tokens": 74, "token_ids": reference_tokens, "token_logprobs": [{}] * 450}
    actual = {"request_id": logical, "model": 1, "rank_initial": 0,
        "prompt_tokens": 74, "token_ids": actual_tokens, "token_logprobs": [{}] * 450}
    for filename, request in (("references.json", reference), ("r0-no_move/requests.json", actual)):
        path = analysis_root / filename
        records = json.loads(path.read_text())
        write_json(path, records + [request])
    bank = analysis_root / "frozen/inputs/prompt-bank.json"
    bank_data = json.loads(bank.read_text())
    bank_data["prompts"]["P74"] = {"token_ids": [7] * 74}
    write_json(bank, bank_data)
    import hashlib
    trace = json.loads((analysis_root / "trace.json").read_text())
    trace["requests"].append({"request_id": logical, "prompt_key": "P74"})
    trace["prompt_bank_sha256"] = hashlib.sha256(bank.read_bytes()).hexdigest()
    write_json(analysis_root / "trace.json", trace)
    original = head_hidden_probe.analyze

    def analyzed(root):
        result = original(root)
        result["full_static_analysis"]["diagnostic"]["trials"][0]["serial_comparisons"].append(
            {"request_id": logical, "first_difference_zero_based": 139})
        return result

    monkeypatch.setattr(head_hidden_probe, "analyze", analyzed)
    result = probe.analyze(analysis_root)
    assert result["all_in_scope_target_attributions_available"]
    assert not result["all_trace_first_divergences_attributed"]
    assert result["uncovered_requests"][0]["actual_first_mismatch"] == 139
    assert not result["uncovered_requests"][0]["attribution_available"]
    assert all(record["position"] != 442 for record in result["comparisons"])
    assert result["scientific_scope"]["P74_known_uncovered_positions"] == [139, 442]
