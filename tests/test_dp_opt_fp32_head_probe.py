import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import fp32_head_campaign
import fp32_head_probe
import live_trace
import static_batch_probe
from test_dp_opt_static_batch_probe import write_fixture


def worker_record(service="instance-1", rank=0):
    return {"service": service, "dp_rank": rank, "model_dtype": "torch.bfloat16",
            "head_weight_dtype": "torch.bfloat16", "floating_parameter_dtypes": ["torch.bfloat16"],
            "effective_head_dtype": "torch.float32", "processor_head_dtype": "torch.float32",
            "head_quant_method": "UnquantizedEmbeddingMethod", "compile_mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": fp32_head_probe.CAPTURES,
            "max_num_batched_tokens": 4096, "max_model_len": 131072,
            "kv_cache_memory_bytes": 25769803776, "attention_tp": 1, "dp": 2,
            "ep_enabled": True, "batch_invariant_enabled": False, "logits_processor_sha256": "abc"}


def precision_fixture(root):
    (root / "fp32-head-config.json").write_text(json.dumps({
        "hf_overrides": {"head_dtype": "float32"},
        "reference_generation": "fresh_serial_same_FP32_head_configuration",
        "is_algorithm_benefit": False, "strict_token_check_relaxed": False}))
    for service in ("instance-1", "instance-2"):
        for rank in (0, 1):
            (root / f"fp32-head-engine-{service}-dp{rank}.json").write_text(json.dumps(worker_record(service, rank)))
            (root / f"fp32-head-output-{service}-dp{rank}.json").write_text(json.dumps({
                "hidden_dtype": "torch.bfloat16", "hidden_device": "cuda", "logits_dtype": "torch.float32"}))


def test_instrument_changes_only_head_setting_and_restores(monkeypatch):
    original = SimpleNamespace(hf_overrides={"unrelated": 7}, worker_cls="original", dtype="bfloat16",
                               compilation_config={"unchanged": True})
    factory = lambda *args, **kwargs: copy.deepcopy(original)
    monkeypatch.setattr(live_trace, "engine_arguments", factory)
    stream = live_trace.stream
    with fp32_head_probe.instrument():
        arguments = live_trace.engine_arguments()
        assert arguments.hf_overrides == {"unrelated": 7, "head_dtype": "float32"}
        assert arguments.worker_cls == "fp32_head_worker.FP32HeadVerificationWorker"
        assert arguments.dtype == "bfloat16"
        assert arguments.compilation_config == original.compilation_config
    assert live_trace.engine_arguments is factory and live_trace.stream is stream
    assert original.hf_overrides == {"unrelated": 7}


def test_static_nested_wrapper_preserves_forced_placement(monkeypatch):
    factory = lambda *args, **kwargs: SimpleNamespace(hf_overrides={}, worker_cls="original")
    monkeypatch.setattr(live_trace, "engine_arguments", factory)
    controller = live_trace.PlacementController
    with static_batch_probe.instrument(), fp32_head_probe.instrument():
        assert live_trace.PlacementController is static_batch_probe.StaticBatchController
        assert live_trace.engine_arguments().worker_cls == "fp32_head_worker.FP32HeadVerificationWorker"
    assert live_trace.engine_arguments is factory and live_trace.PlacementController is controller


@pytest.mark.parametrize("name,value", [("head_weight_dtype", "torch.float32"),
    ("model_dtype", "torch.float32"), ("effective_head_dtype", "torch.bfloat16"),
    ("processor_head_dtype", "torch.bfloat16"), ("head_quant_method", "QuantizedMethod"),
    ("attention_tp", 2), ("dp", 1), ("ep_enabled", False), ("compile_mode", 3),
    ("cudagraph_mode", "NONE"), ("cudagraph_capture_sizes", [1, 2, 3]),
    ("max_num_batched_tokens", 8192), ("batch_invariant_enabled", True)])
def test_reject_changed_worker_configuration(name, value):
    record = worker_record()
    record[name] = value
    with pytest.raises(ValueError, match=name):
        fp32_head_probe.validate_worker(record)


def test_reject_actual_bf16_logits_even_with_fp32_declaration(tmp_path):
    precision_fixture(tmp_path)
    path = tmp_path / "fp32-head-output-instance-2-dp1.json"
    path.write_text(json.dumps({"hidden_dtype": "torch.bfloat16", "hidden_device": "cuda",
                                "logits_dtype": "torch.bfloat16"}))
    with pytest.raises(ValueError, match="actual CUDA"):
        fp32_head_probe.verify_precision(tmp_path)


def test_static_full_token_pass_is_not_benefit(tmp_path):
    write_fixture(tmp_path)
    precision_fixture(tmp_path)
    (tmp_path / "summary.json").write_text('{"status":"passed"}')
    result = fp32_head_probe.analyze(tmp_path, "static")
    assert result["strict_correctness_passed"]
    assert not result["is_algorithm_benefit"]
    assert not result["numerical_root_cause_proven"]


def test_static_divergence_still_fails_without_relaxation(tmp_path):
    write_fixture(tmp_path)
    precision_fixture(tmp_path)
    path = tmp_path / "r1-no_move/requests.json"
    records = json.loads(path.read_text())
    records[0]["token_ids"][1] = 99
    path.write_text(json.dumps(records))
    (tmp_path / "error.json").write_text(json.dumps({"type": "RuntimeError",
        "message": "Selected trace request differs from its full serial greedy reference"}))
    result = fp32_head_probe.analyze(tmp_path, "static")
    assert not result["strict_correctness_passed"]
    assert not result["strict_token_check_relaxed"]
    assert result["diagnostic"]["underlying_strict_error"]


def test_reject_batch_invariance(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with pytest.raises(ValueError, match="batch invariance"):
        with fp32_head_probe.instrument():
            pytest.fail("must reject before instrumentation")


def test_migration_verification_waits_for_shell_exit_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("DP_OPT_HEAD_PROBE_STAGE", "migration")
    monkeypatch.setattr(sys, "argv", ["fp32_head_probe.py", "--output", str(tmp_path)])
    monkeypatch.setattr(live_trace, "main", lambda: None)
    monkeypatch.setattr(fp32_head_probe, "analyze", lambda *args: pytest.fail("shell has not written exit-status"))
    fp32_head_probe.main()
    assert json.loads((tmp_path / "fp32-head-runtime-completion.json").read_text())["independent_verification_pending"]


def test_migration_requires_real_copies(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"policies": fp32_head_probe.MIGRATION_POLICIES, "repeats": 1}))
    monkeypatch.setattr(fp32_head_probe, "verify_precision", lambda root: {})
    monkeypatch.setattr(fp32_head_probe, "verify", lambda *args: {"status": "passed"})
    for policy in fp32_head_probe.MIGRATION_POLICIES:
        directory = tmp_path / f"r0-{policy}"
        directory.mkdir()
        (directory / "migrations.json").write_text("[]")
    with pytest.raises(ValueError, match="actual migrated"):
        fp32_head_probe.analyze(tmp_path, "migration")
    for policy in ("random_no_return", "kv_only_headroom"):
        (tmp_path / f"r0-{policy}/migrations.json").write_text('[{"kv_bytes":1024}]')
    assert fp32_head_probe.analyze(tmp_path, "migration")["strict_correctness_passed"]


def test_deployment_inputs_and_migration_gate():
    repo = Path(fp32_head_campaign.__file__).resolve().parents[2]
    for name in fp32_head_campaign.SYNC_FILES:
        assert (repo / name).is_file()
    script = (repo / "experiments/dp_opt/fp32-head-probe.sbatch").read_text()
    assert "#SBATCH --partition=gpu_a800" in script
    assert "for stage in static migration" in script
    assert script.index('bash "$base/fp32-head-launch.sh"') < script.index("--output \"$evidence/independent-fp32-head-verification.json\"")
    assert "set -euo pipefail" in script
    assert "unset VLLM_BATCH_INVARIANT" in script


def test_worker_records_actual_output_without_cuda_import(tmp_path, monkeypatch):
    base = ModuleType("workaware_worker")
    base.WorkawareVerificationWorker = type("Base", (), {"initialize_from_config": lambda self, config: None})
    monkeypatch.setitem(sys.modules, "workaware_worker", base)
    source = Path(fp32_head_probe.__file__).with_name("fp32_head_worker.py")
    spec = importlib.util.spec_from_file_location("head_worker_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processor = SimpleNamespace(head_dtype="torch.float32")
    handle = SimpleNamespace(remove=lambda: setattr(handle, "removed", True))
    processor.register_forward_hook = lambda hook: setattr(processor, "hook", hook) or handle
    monkeypatch.setattr(module.inspect, "getsourcefile", lambda model_type: str(source))
    parameter = SimpleNamespace(dtype="torch.bfloat16", is_floating_point=lambda: True)
    model = SimpleNamespace(lm_head=SimpleNamespace(weight=parameter,
        quant_method=type("UnquantizedEmbeddingMethod", (), {})()), logits_processor=processor,
        parameters=lambda: [parameter])
    worker = module.FP32HeadVerificationWorker()
    worker.model_runner = SimpleNamespace(get_model=lambda: model)
    worker.vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(mode=0,
        cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"), cudagraph_capture_sizes=fp32_head_probe.CAPTURES))
    worker.parallel_config = SimpleNamespace(data_parallel_rank=0, tensor_parallel_size=1,
        data_parallel_size=2, enable_expert_parallel=True)
    worker.model_config = SimpleNamespace(dtype="torch.bfloat16", head_dtype="torch.float32", max_model_len=131072)
    worker.scheduler_config = SimpleNamespace(max_num_batched_tokens=4096)
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=25769803776)
    monkeypatch.setenv("DP_OPT_SERVICE", "instance-1")
    monkeypatch.setenv("DP_OPT_EVIDENCE_ROOT", str(tmp_path))
    worker.initialize_from_config(None)
    processor.hook(processor, [model.lm_head, SimpleNamespace(dtype="torch.bfloat16",
        device=SimpleNamespace(type="cuda"))], SimpleNamespace(dtype="torch.float32"))
    record = json.loads((tmp_path / "fp32-head-engine-instance-1-dp0.json").read_text())
    fp32_head_probe.validate_worker(record)
    assert json.loads((tmp_path / "fp32-head-output-instance-1-dp0.json").read_text())["logits_dtype"] == "torch.float32"
    assert handle.removed
