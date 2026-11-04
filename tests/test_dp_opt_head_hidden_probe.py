import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import fp32_head_campaign
import head_hidden_campaign
import head_hidden_probe
import live_trace
import static_batch_probe
import fp32_head_probe


@pytest.mark.parametrize("prompt,position,expected", [
    (149, 52, 52), (149, 53, 53), (149, 54, 54), (149, 55, None),
    (74, 441, 441), (74, 442, 442), (74, 443, 443), (74, 440, None),
    (100371, 53, None),
])
def test_predeclared_observation_positions(prompt, position, expected):
    state = SimpleNamespace(num_prompt_tokens=prompt, output_token_ids=[1] * position)
    assert head_hidden_probe.selected_position(state) == expected


def projection(hidden, logits, context=None):
    return {"hidden": hidden, "candidate_logits": logits, "position": 53,
            "candidates": [12822, 20632], "context_token_ids": context or [1, 2, 3]}


def test_decomposition_distinguishes_hidden_and_projection_changes():
    rows = [[2.0, 4.0], [1.0, 2.0]]
    reference = projection([1.0, 1.0], [3.0, 0.0])
    actual = projection([2.0, 1.0], [4.5, 0.0])
    result = head_hidden_probe.decompose(reference, actual, rows)
    assert result["hidden_contribution"] == 1.0
    assert result["head_projection_contribution"] == .5
    assert result["observed_contrast_change"] == 1.5
    assert result["hidden_changed_elements"] == 1
    assert not result["hidden_byte_values_identical"]
    assert not result["numerical_root_cause_proven"]


def test_same_hidden_changed_projection_is_not_upstream_evidence():
    result = head_hidden_probe.decompose(projection([1.0], [2.0, 0.0]),
                                        projection([1.0], [2.25, 0.0]), [[2.0], [0.0]])
    assert result["hidden_byte_values_identical"]
    assert result["hidden_contribution"] == 0
    assert result["head_projection_contribution"] == .25


@pytest.mark.parametrize("field,value", [
    ("context_token_ids", [1, 2, 4]), ("position", 54), ("candidates", [20632, 12822]),
])
def test_never_decomposes_different_context_position_or_candidates(field, value):
    actual = projection([1.0], [2.0, 0.0])
    actual[field] = value
    with pytest.raises(ValueError, match="identical complete common prefixes"):
        head_hidden_probe.decompose(projection([1.0], [2.0, 0.0]), actual, [[2.0], [0.0]])


@pytest.mark.parametrize("hidden,rows", [
    ([1.0], [[1.0]]), ([1.0], [[1.0, 2.0], [2.0]]),
    ([float("nan")], [[1.0], [2.0]]), ([1.0], [[float("inf")], [2.0]]),
])
def test_rejects_invalid_dot_geometry_and_nonfinite_inputs(hidden, rows):
    with pytest.raises(ValueError):
        head_hidden_probe.dot_contrast(hidden, rows)


def test_nested_instrument_retains_original_configuration_and_restores(monkeypatch):
    original = SimpleNamespace(hf_overrides={"other": 1}, worker_cls="original", dtype="bfloat16")
    factory = lambda *args, **kwargs: copy.deepcopy(original)
    monkeypatch.setattr(live_trace, "engine_arguments", factory)
    controller = live_trace.PlacementController
    with static_batch_probe.instrument(), fp32_head_probe.instrument(), head_hidden_probe.instrument():
        arguments = live_trace.engine_arguments()
        assert arguments.worker_cls == "head_hidden_worker.HeadHiddenVerificationWorker"
        assert arguments.hf_overrides == {"other": 1, "head_dtype": "float32"}
        assert arguments.dtype == "bfloat16"
        assert live_trace.PlacementController is static_batch_probe.StaticBatchController
    assert live_trace.engine_arguments is factory
    assert live_trace.PlacementController is controller


def test_new_deployment_scope_is_complete_and_does_not_mutate_parent():
    repo = Path(head_hidden_campaign.__file__).resolve().parents[2]
    original = fp32_head_campaign.SYNC_FILES
    with head_hidden_campaign.sync_file_scope():
        assert fp32_head_campaign.SYNC_FILES == head_hidden_campaign.SYNC_FILES
        for name in head_hidden_campaign.SYNC_FILES:
            assert (repo / name).is_file()
    assert fp32_head_campaign.SYNC_FILES is original
    script = (repo / "experiments/dp_opt/head-hidden-probe.sbatch").read_text()
    assert "#SBATCH --partition=gpu_a800" in script
    assert "DP_OPT_TRACE_POLICIES=no_move" in script
    assert "unset VLLM_BATCH_INVARIANT" in script
    assert "test_dp_opt_head_hidden_probe.py" in script


def test_concurrent_sync_rewrites_the_actual_stdin_protected_script_and_restores(monkeypatch):
    import rebalance_campaign

    calls = []
    original = lambda script, **kwargs: calls.append(script) or "verified"
    monkeypatch.setattr(rebalance_campaign, "checked_ssh_script", original)
    with head_hidden_campaign.sync_file_scope(278514):
        result = rebalance_campaign.checked_ssh_script("set -eu\n" + head_hidden_campaign.EMPTY_QUEUE_GUARD)
    assert result == "verified"
    assert head_hidden_campaign.EMPTY_QUEUE_GUARD not in calls[0]
    assert "job=278514" in calls[0] and "qwen-public-perf" in calls[0]
    assert rebalance_campaign.checked_ssh_script is original


def attribution_fixture(root, monkeypatch, position=442, first_difference=442):
    identifier = "synthetic-conv-0"
    prompt = [7] * 74
    prefix = [9] * position
    candidates = [582, 279]
    reference = {"trace_request_id": identifier, "request_id": "reference-0",
                 "model": 1, "rank_initial": 0, "prompt_tokens": 74, "token_ids": prefix + [279],
                 "token_logprobs": [{}] * position + [{"582": -1.5, "279": -1.0}]}
    actual = {"request_id": identifier, "model": 1, "rank_initial": 0, "prompt_tokens": 74,
              "token_ids": prefix + [582],
              "token_logprobs": [{}] * position + [{"582": -1.0, "279": -1.5}]}
    (root / "references.json").write_text(json.dumps([reference]))
    bank = root / "frozen/inputs/prompt-bank.json"
    bank.parent.mkdir(parents=True, exist_ok=True)
    bank.write_text(json.dumps({"prompts": {"synthetic": {"token_ids": prompt}}}))
    (root / "trace.json").write_text(json.dumps({"requests": [{"request_id": identifier,
        "prompt_key": "synthetic"}], "prompt_bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest()}))
    trial = root / "r0-no_move"
    trial.mkdir()
    (trial / "requests.json").write_text(json.dumps([actual]))
    directory = root / "head-hidden-snapshots"
    directory.mkdir()
    weights = {str(prompt_length): {"candidates": target["candidates"], "rows": [[1.0], [0.0]]}
               for prompt_length, target in head_hidden_probe.TARGETS.items()}
    for service in ("instance-1", "instance-2"):
        for rank in (0, 1):
            (root / f"head-candidate-weights-{service}-dp{rank}.json").write_text(json.dumps({
                "service": service, "dp_rank": rank, "weight_dtype": "torch.bfloat16", "weights": weights}))
    for role, native, logits in (("reference", "reference-0", [1.0, 1.5]),
                                ("trial", identifier, [1.5, 1.0])):
        snapshot = {"service": "instance-2", "dp_rank": 0, "native_request_id": native,
                    "role": role, "repeat": None if role == "reference" else 0,
                    "layout": None if role == "reference" else "corank0",
                    "batch_rows": 1, "batch_native_request_ids": [native], "row_index": 0,
                    "position": position, "prompt_tokens": 74, "prompt_token_ids": prompt,
                    "output_prefix_token_ids": prefix, "context_token_ids": prompt + prefix,
                    "context_sha256": head_hidden_probe.context_digest(prompt + prefix),
                    "forward_consumed_prefix_tokens": len(prompt + prefix), "sample_discarded": False,
                    "hidden_dtype": "torch.bfloat16", "logits_dtype": "torch.float32",
                    "hidden": [1.0], "candidates": candidates, "candidate_logits": logits}
        (directory / f"{role}.json").write_text(json.dumps(snapshot))
    base = {"strict_correctness_passed": False, "diagnostic": {"trials": [{"repeat": 0,
            "layout": "corank0", "serial_comparisons": [{"request_id": identifier,
            "first_difference_zero_based": first_difference}]}]}}
    monkeypatch.setattr(fp32_head_probe, "analyze", lambda *args: copy.deepcopy(base))
    return directory


def test_attribution_is_diagnostic_and_keeps_strict_failure(tmp_path, monkeypatch):
    attribution_fixture(tmp_path, monkeypatch)
    result = head_hidden_probe.analyze(tmp_path)
    assert not result["full_static_analysis"]["strict_correctness_passed"]
    assert not result["is_algorithm_benefit"]
    assert not result["strict_token_check_relaxed"]
    comparison = result["projection_comparisons"][0]
    assert comparison["full_common_prefix"] and comparison["is_first_divergence"]
    assert comparison["head_projection_contribution"] == 1.0


def test_changed_sampler_contrast_is_rejected(tmp_path, monkeypatch):
    directory = attribution_fixture(tmp_path, monkeypatch)
    path = directory / "trial.json"
    record = json.loads(path.read_text())
    record["candidate_logits"][0] += .1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="sampler logprobs"):
        head_hidden_probe.analyze(tmp_path)


def test_changed_complete_prefix_is_rejected(tmp_path, monkeypatch):
    directory = attribution_fixture(tmp_path, monkeypatch)
    path = directory / "trial.json"
    record = json.loads(path.read_text())
    record["output_prefix_token_ids"][0] = 11
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="actual complete request prefix"):
        head_hidden_probe.analyze(tmp_path)


def test_four_workers_must_have_identical_candidate_weights(tmp_path, monkeypatch):
    attribution_fixture(tmp_path, monkeypatch)
    path = tmp_path / "head-candidate-weights-instance-2-dp1.json"
    record = json.loads(path.read_text())
    record["weights"]["74"]["rows"][0][0] = 2.0
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identical BF16 candidate weight rows"):
        head_hidden_probe.analyze(tmp_path)


@pytest.mark.parametrize("position", [441, 443])
def test_uses_actual_first_divergence_in_captured_neighbor_position(tmp_path, monkeypatch, position):
    attribution_fixture(tmp_path, monkeypatch, position=position, first_difference=position)
    result = head_hidden_probe.analyze(tmp_path)
    comparison = result["projection_comparisons"][0]
    assert comparison["position"] == position
    assert comparison["is_first_divergence"] and comparison["attribution_available"]
    assert result["all_observed_first_divergences_attributed"]


def test_first_divergence_outside_window_is_explicitly_unavailable(tmp_path, monkeypatch):
    attribution_fixture(tmp_path, monkeypatch, first_difference=440)
    result = head_hidden_probe.analyze(tmp_path)
    comparison = result["projection_comparisons"][0]
    assert not comparison["attribution_available"]
    assert comparison["reason"] == "actual_first_divergence_outside_captured_positions"
    assert "head_projection_contribution" not in comparison
    assert not result["all_observed_first_divergences_attributed"]


def test_reversed_candidates_and_logits_in_both_snapshots_do_not_misattribute(tmp_path, monkeypatch):
    directory = attribution_fixture(tmp_path, monkeypatch)
    for path in directory.glob("*.json"):
        snapshot = json.loads(path.read_text())
        snapshot["candidates"].reverse()
        snapshot["candidate_logits"].reverse()
        path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="candidate ordering"):
        head_hidden_probe.analyze(tmp_path)


@pytest.mark.parametrize("field,value", [
    ("native_request_id", "synthetic-conv-00"), ("service", "instance-1"),
    ("dp_rank", 1), ("layout", "corank1"), ("row_index", 1),
    ("batch_native_request_ids", ["wrong"]), ("batch_rows", 2),
])
def test_snapshot_identity_and_row_are_bound_to_actual_request(tmp_path, monkeypatch, field, value):
    directory = attribution_fixture(tmp_path, monkeypatch)
    path = directory / "trial.json"
    snapshot = json.loads(path.read_text())
    snapshot[field] = value
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="identity|logits row"):
        head_hidden_probe.analyze(tmp_path)


@pytest.mark.parametrize("field,value", [
    ("service", "instance-1"), ("dp_rank", 0), ("weight_dtype", "torch.float32"),
])
def test_head_weights_are_bound_to_expected_worker_and_dtype(tmp_path, monkeypatch, field, value):
    attribution_fixture(tmp_path, monkeypatch)
    path = tmp_path / "head-candidate-weights-instance-2-dp1.json"
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="worker identity or BF16 dtype"):
        head_hidden_probe.analyze(tmp_path)


def test_head_weight_candidate_order_is_predeclared(tmp_path, monkeypatch):
    attribution_fixture(tmp_path, monkeypatch)
    for path in tmp_path.glob("head-candidate-weights-*.json"):
        record = json.loads(path.read_text())
        record["weights"]["74"]["candidates"].reverse()
        record["weights"]["74"]["rows"].reverse()
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="candidate ordering"):
        head_hidden_probe.analyze(tmp_path)


@pytest.mark.parametrize("native,expected", [
    ("request-1", True), ("request-1-0abc1234", True), ("request-10", False),
    ("request-1-0abc12345", False), ("request-1-arbitrary", False),
])
def test_native_id_binding_uses_installed_exact_eight_hex_suffix(native, expected):
    assert head_hidden_probe.native_request_matches(native, "request-1") is expected


def test_snapshot_prompt_is_bound_to_frozen_input_not_only_its_length(tmp_path, monkeypatch):
    directory = attribution_fixture(tmp_path, monkeypatch)
    path = directory / "trial.json"
    snapshot = json.loads(path.read_text())
    snapshot["prompt_token_ids"][0] = 17
    snapshot["context_token_ids"] = snapshot["prompt_token_ids"] + snapshot["output_prefix_token_ids"]
    snapshot["context_sha256"] = head_hidden_probe.context_digest(snapshot["context_token_ids"])
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="actual complete request prefix"):
        head_hidden_probe.analyze(tmp_path)


class SnapshotTensor:
    def __init__(self, values, dtype="torch.bfloat16"):
        self.values = values
        self.dtype = dtype
        self.shape = (len(values),)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, columns = index
            values = [self.values[row][column] for column in columns]
        elif isinstance(index, list):
            values = [self.values[column] for column in index]
        else:
            values = self.values[index]
        return SnapshotTensor(values, self.dtype)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def tolist(self):
        return self.values


def load_mock_worker(root, monkeypatch, reference=False):
    base = ModuleType("fp32_head_worker")
    base.FP32HeadVerificationWorker = type("Base", (), {"initialize_from_config": lambda self, config: None})
    monkeypatch.setitem(sys.modules, "fp32_head_worker", base)
    torch = ModuleType("torch")
    torch.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(
        allow_bf16_reduced_precision_reduction=False,
        allow_bf16_reduced_precision_reduction_split_k=True)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    source = Path(head_hidden_probe.__file__).with_name("head_hidden_worker.py")
    spec = importlib.util.spec_from_file_location("hidden_worker_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processor = SimpleNamespace(scale=1, soft_cap=None, logits_as_input=False)
    processor.register_forward_hook = lambda hook: setattr(processor, "hook", hook)
    weight = SnapshotTensor({582: [1.0, 0.0], 279: [0.0, 1.0],
                             12822: [2.0, 0.0], 20632: [0.0, 2.0]})
    model = SimpleNamespace(logits_processor=processor, lm_head=SimpleNamespace(weight=weight, bias=None))
    native = "reference-0-0123abcd" if reference else "synthetic-conv-0-0123abcd"
    state = SimpleNamespace(num_prompt_tokens=149, prompt_token_ids=[12] * 149,
                            output_token_ids=[13] * 53, num_computed_tokens=201)
    unrelated = SimpleNamespace(num_prompt_tokens=37, prompt_token_ids=[14] * 37, output_token_ids=[15] * 17)
    worker = module.HeadHiddenVerificationWorker()
    worker.model_runner = SimpleNamespace(get_model=lambda: model,
        input_batch=SimpleNamespace(req_ids=["unrelated", native]), requests={native: state, "unrelated": unrelated},
        optimistic_seq_lens_cpu=[54, 202], discard_request_mask=SimpleNamespace(np=[False, False]))
    worker.parallel_config = SimpleNamespace(data_parallel_rank=0)
    worker.vllm_config = SimpleNamespace(speculative_config=None)
    worker.scheduler_config = SimpleNamespace(async_scheduling=False)
    monkeypatch.setenv("DP_OPT_EVIDENCE_ROOT", str(root))
    monkeypatch.setenv("DP_OPT_SERVICE", "instance-2")
    if not reference:
        (root / "phase.json").write_text(json.dumps({"repeat": 0, "layout": "corank0"}))
    return worker, processor, native


@pytest.mark.parametrize("reference", [False, True])
def test_worker_uses_current_output_prefix_and_actual_request_row(tmp_path, monkeypatch, reference):
    worker, processor, native = load_mock_worker(tmp_path, monkeypatch, reference)
    worker.initialize_from_config(None)
    hidden = SnapshotTensor([[7.0, 8.0], [3.0, 4.0]])
    logits = SnapshotTensor([{12822: 9.0, 20632: 8.0}, {12822: 6.0, 20632: 5.0}], "torch.float32")
    processor.hook(processor, [None, hidden], logits)
    files = list((tmp_path / "head-hidden-snapshots").glob("*.json"))
    assert len(files) == 1
    snapshot = json.loads(files[0].read_text())
    assert snapshot["native_request_id"] == native and snapshot["row_index"] == 1
    assert snapshot["position"] == 53
    assert snapshot["hidden"] == [3.0, 4.0] and snapshot["candidate_logits"] == [6.0, 5.0]
    assert snapshot["context_token_ids"] == [12] * 149 + [13] * 53
    assert snapshot["role"] == ("reference" if reference else "trial")
    assert snapshot["repeat"] == (None if reference else 0)
    assert not snapshot["allow_bf16_reduced_precision_reduction"]
    assert snapshot["allow_bf16_reduced_precision_reduction_split_k"]
    assert snapshot["observation_includes_CPU_synchronization"]


def test_worker_rejects_non_one_to_one_logits_rows(tmp_path, monkeypatch):
    worker, processor, _ = load_mock_worker(tmp_path, monkeypatch)
    worker.initialize_from_config(None)
    with pytest.raises(RuntimeError, match="not one-to-one"):
        processor.hook(processor, [None, SnapshotTensor([[1.0, 2.0], [3.0, 4.0]])],
                       SnapshotTensor([{12822: 1.0, 20632: 2.0}], "torch.float32"))


def test_worker_rejects_async_state_before_observation(tmp_path, monkeypatch):
    worker, _, _ = load_mock_worker(tmp_path, monkeypatch)
    worker.scheduler_config.async_scheduling = True
    with pytest.raises(RuntimeError, match="synchronous projection"):
        worker.initialize_from_config(None)


def test_missing_sampler_coverage_keeps_projection_but_not_top1_attribution(tmp_path, monkeypatch):
    attribution_fixture(tmp_path, monkeypatch)
    path = tmp_path / "r0-no_move/requests.json"
    records = json.loads(path.read_text())
    del records[0]["token_logprobs"][442]["279"]
    path.write_text(json.dumps(records))
    result = head_hidden_probe.analyze(tmp_path)
    comparison = result["projection_comparisons"][0]
    assert comparison["attribution_available"]
    assert not comparison["first_top1_attribution_available"]
    assert comparison["first_top1_attribution_unavailable_reason"] == "sampler_candidate_contrast_coverage_missing"
    assert comparison["sampler_contrast_coverage"] == {"reference": True, "actual": False}
    assert not result["all_observed_first_divergences_attributed"]


def test_changed_first_decision_candidate_is_not_explained_by_old_pair(tmp_path, monkeypatch):
    attribution_fixture(tmp_path, monkeypatch)
    path = tmp_path / "r0-no_move/requests.json"
    records = json.loads(path.read_text())
    records[0]["token_ids"][442] = 999
    path.write_text(json.dumps(records))
    result = head_hidden_probe.analyze(tmp_path)
    comparison = result["projection_comparisons"][0]
    assert comparison["attribution_available"]
    assert not comparison["first_top1_attribution_available"]
    assert comparison["first_top1_attribution_unavailable_reason"] == "actual_first_top1_candidates_differ_from_predeclared_pair"
    assert not result["all_observed_first_divergences_attributed"]


def test_partial_reprefill_is_skipped_without_snapshot_filename_collision(tmp_path, monkeypatch):
    worker, processor, _ = load_mock_worker(tmp_path, monkeypatch)
    worker.initialize_from_config(None)
    hidden = SnapshotTensor([[7.0, 8.0], [3.0, 4.0]])
    logits = SnapshotTensor([{12822: 9.0, 20632: 8.0}, {12822: 6.0, 20632: 5.0}], "torch.float32")
    worker.model_runner.discard_request_mask.np[1] = True
    for consumed in (132, 164):
        worker.model_runner.optimistic_seq_lens_cpu[1] = consumed
        processor.hook(processor, [None, hidden], logits)
    assert not list((tmp_path / "head-hidden-snapshots").glob("*.json"))
    skipped = (tmp_path / "head-hidden-skipped-instance-2-dp0.jsonl").read_text().splitlines()
    assert len(skipped) == 2
    worker.model_runner.optimistic_seq_lens_cpu[1] = 202
    worker.model_runner.discard_request_mask.np[1] = False
    processor.hook(processor, [None, hidden], logits)
    assert len(list((tmp_path / "head-hidden-snapshots").glob("*.json"))) == 1


def test_unmasked_incomplete_forward_fails_instead_of_false_common_prefix(tmp_path, monkeypatch):
    worker, processor, _ = load_mock_worker(tmp_path, monkeypatch)
    worker.initialize_from_config(None)
    worker.model_runner.optimistic_seq_lens_cpu[1] = 132
    with pytest.raises(RuntimeError, match="complete predicted prefix"):
        processor.hook(processor, [None, SnapshotTensor([[7.0, 8.0], [3.0, 4.0]])],
                       SnapshotTensor([{12822: 9.0, 20632: 8.0}, {12822: 6.0, 20632: 5.0}], "torch.float32"))


@pytest.fixture
def run_queue_guard(tmp_path):
    commands = tmp_path / "commands"
    commands.mkdir()
    for name, body in {
        "id": 'printf "%s\\n" "${GUARD_USER:-researcher}"',
        "squeue": 'printf "%s\\n" "$QUEUE_ROWS"; exit "${QUEUE_EXIT:-0}"',
        "sacct": 'printf "%s\\n" "$ACCOUNTING_ROWS"; exit "${ACCOUNTING_EXIT:-0}"',
    }.items():
        executable = commands / name
        executable.write_text("#!/bin/sh\n" + body + "\n")
        executable.chmod(0o755)

    def execute(queue="123|researcher|qwen-public-perf|RUNNING", accounting=None, job=123, **extra):
        environment = {**os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"],
                       "USER": "researcher", "QUEUE_ROWS": queue,
                       "ACCOUNTING_ROWS": queue if accounting is None else accounting, **extra}
        return subprocess.run(["bash", "-c", "set -eu\n" + head_hidden_campaign.submission_guard(job)],
                              env=environment, text=True, capture_output=True)

    return execute


@pytest.mark.parametrize("name", ["qwen-public-perf", "qwen-public-perf-main-r2"])
@pytest.mark.parametrize("status", ["PENDING", "RUNNING"])
def test_explicit_performance_job_is_verified_by_queue_and_accounting(run_queue_guard, name, status):
    assert run_queue_guard(f"123|researcher|{name}|{status}").returncode == 0


@pytest.mark.parametrize("queue,accounting,extra", [
    ("", "123|researcher|qwen-public-perf|RUNNING", {}),
    ("123|researcher|qwen-public-perf|RUNNING", "", {}),
    ("123|other|qwen-public-perf|RUNNING", None, {}),
    ("123|researcher|qwen-public-perf|RUNNING", "123|other|qwen-public-perf|RUNNING", {}),
    ("124|researcher|qwen-public-perf|RUNNING", None, {}),
    ("123|researcher|qwen-public-perf|RUNNING", "124|researcher|qwen-public-perf|RUNNING", {}),
    ("123|researcher|qwen-head-hidden-probe|RUNNING", None, {}),
    ("123|researcher|qwen-public-performance|RUNNING", None, {}),
    ("123|researcher|qwen-public-perf|RUNNING", "123|researcher|qwen-public-perf-other|RUNNING", {}),
    ("123|researcher|qwen-public-perf|RUNNING\n124|researcher|unrelated|PENDING", None, {}),
    ("123|researcher|qwen-public-perf|RUNNING\n123|researcher|qwen-public-perf|RUNNING", None, {}),
    ("123|researcher|qwen-public-perf|RUNNING", "123|researcher|qwen-public-perf|COMPLETED", {}),
    ("123|researcher|qwen-public-perf|COMPLETED", None, {}),
    ("123_1|researcher|qwen-public-perf|RUNNING", None, {}),
    ("123|researcher|qwen-public-perf|RUNNING|unexpected", None, {}),
    ("123|researcher|qwen-public-perf|RUNNING", None, {"QUEUE_EXIT": "1"}),
    ("123|researcher|qwen-public-perf|RUNNING", None, {"ACCOUNTING_EXIT": "1"}),
    ("123|researcher|qwen-public-perf|RUNNING", None, {"GUARD_USER": "other"}),
])
def test_explicit_job_guard_fails_closed(run_queue_guard, queue, accounting, extra):
    assert run_queue_guard(queue, accounting, **extra).returncode != 0


def test_default_guard_keeps_empty_queue_requirement(run_queue_guard):
    assert head_hidden_campaign.submission_guard() == head_hidden_campaign.EMPTY_QUEUE_GUARD
    assert run_queue_guard("", job=None).returncode == 0
    assert run_queue_guard(job=None).returncode != 0


@pytest.mark.parametrize("value", [0, -1, True, 123.0, "1;sbatch", "", "01", " 123"])
def test_explicit_job_argument_rejects_noncanonical_or_unsafe_ids(value):
    with pytest.raises(head_hidden_campaign.argparse.ArgumentTypeError):
        head_hidden_campaign.submission_guard(value)


def test_scoped_sync_adapts_all_inherited_guards_and_restores_on_error(monkeypatch):
    commands = []
    parent_ssh = lambda command, *args, **kwargs: commands.append(command)
    sync_ssh = lambda command, *args, **kwargs: commands.append(command)
    monkeypatch.setattr(fp32_head_campaign, "checked_ssh", parent_ssh)
    monkeypatch.setattr(head_hidden_campaign.rebalance_campaign, "checked_ssh", sync_ssh)
    original_files = fp32_head_campaign.SYNC_FILES
    with pytest.raises(RuntimeError, match="interrupted"):
        with head_hidden_campaign.sync_file_scope(123):
            fp32_head_campaign.checked_ssh("set -eu; " + head_hidden_campaign.EMPTY_QUEUE_GUARD + "; sha256sum input")
            head_hidden_campaign.rebalance_campaign.checked_ssh(
                "set -eu; " + head_hidden_campaign.EMPTY_QUEUE_GUARD + "; test origin = expected")
            fp32_head_campaign.checked_ssh("sha256sum frozen-source")
            raise RuntimeError("interrupted")
    assert all("sacct -n -X -P -j 123" in command for command in commands[:2])
    assert commands[0].endswith("; sha256sum input")
    assert commands[1].endswith("; test origin = expected")
    assert commands[2] == "sha256sum frozen-source"
    assert fp32_head_campaign.checked_ssh is parent_ssh
    assert head_hidden_campaign.rebalance_campaign.checked_ssh is sync_ssh
    assert fp32_head_campaign.SYNC_FILES is original_files


def campaign_fixture(root, parent, concurrent_job=None, **fields):
    repo = Path(head_hidden_campaign.__file__).resolve().parents[2]
    state = {"repo": str(repo), "parent": str(parent), "phase": "head_hidden_prepared",
             "monitor_interval_s": 300, "concurrent_performance_job": concurrent_job,
             "original_trace_sha256": "trace", **fields}
    (root / "campaign-state.json").write_text(json.dumps(state))
    return state


@pytest.mark.parametrize("flag", ["--prepare-only", "--preflight-only"])
def test_cli_safe_preparation_flags_forward_concurrent_job_without_sync(tmp_path, monkeypatch, flag):
    calls = []
    monkeypatch.setattr(head_hidden_campaign, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(sys, "argv", ["head_hidden_campaign", "--root", str(tmp_path),
                                     "--parent", str(tmp_path / "parent"),
                                     "--concurrent-performance-job", "123", flag])
    head_hidden_campaign.main()
    assert calls[0][1] == {"concurrent_performance_job": 123, "prepare_only": flag == "--prepare-only",
                           "preflight_only": flag == "--preflight-only"}


def test_cli_rejects_bad_job_before_running_campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(head_hidden_campaign, "run", lambda *args, **kwargs: pytest.fail("unsafe CLI called run"))
    monkeypatch.setattr(sys, "argv", ["head_hidden_campaign", "--root", str(tmp_path),
                                     "--parent", str(tmp_path), "--concurrent-performance-job", "0"])
    with pytest.raises(SystemExit) as result:
        head_hidden_campaign.main()
    assert result.value.code == 2


@pytest.mark.parametrize("preflight_only", [False, True])
def test_safe_preparation_modes_never_ssh_or_submit(tmp_path, monkeypatch, preflight_only):
    parent = tmp_path / "parent"
    state = campaign_fixture(tmp_path, parent, 123)
    calls = []
    monkeypatch.setattr(head_hidden_campaign, "check_frozen_sources", lambda *args: None)
    monkeypatch.setattr(head_hidden_campaign, "checked_ssh", lambda *args: pytest.fail("preparation attempted SSH"))
    monkeypatch.setattr(head_hidden_campaign, "preflight", lambda *args: calls.append("preflight"))
    head_hidden_campaign.run(tmp_path, parent, 123, prepare_only=not preflight_only, preflight_only=preflight_only)
    assert calls == (["preflight"] if preflight_only else [])
    assert json.loads((tmp_path / "campaign-state.json").read_text())["phase"] == state["phase"]


@pytest.mark.parametrize("requested", [None, 124])
def test_recovery_requires_same_explicit_performance_job(tmp_path, monkeypatch, requested):
    parent = tmp_path / "parent"
    campaign_fixture(tmp_path, parent, 123, job=999)
    monkeypatch.setattr(head_hidden_campaign, "checked_ssh", lambda *args: pytest.fail("identity mismatch attempted SSH"))
    with pytest.raises(ValueError, match="Preserve concurrent performance job identity"):
        head_hidden_campaign.run(tmp_path, parent, requested)
    assert json.loads((tmp_path / "campaign-state.json").read_text())["job"] == 999


def test_ambiguous_submission_intent_is_not_resubmitted(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    campaign_fixture(tmp_path, parent, 123, phase="submission_intent_do_not_duplicate", submission_ledger="remote-ledger")
    monkeypatch.setattr(head_hidden_campaign, "check_frozen_sources", lambda *args: None)
    monkeypatch.setattr(head_hidden_campaign, "checked_ssh", lambda *args: pytest.fail("uncertain intent attempted submission"))
    with pytest.raises(RuntimeError, match="Reconcile submission intent"):
        head_hidden_campaign.run(tmp_path, parent, 123)
    assert json.loads((tmp_path / "campaign-state.json").read_text())["submission_ledger"] == "remote-ledger"


def test_existing_debug_job_recovers_polling_without_sync_or_new_submission(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    campaign_fixture(tmp_path, parent, 123, job=999, phase="monitoring_common_prefix_projection_control")
    calls = []
    monkeypatch.setattr(head_hidden_campaign, "check_frozen_sources", lambda *args: None)
    monkeypatch.setattr(head_hidden_campaign, "checked_ssh", lambda *args: pytest.fail("recovery attempted new SSH submission"))

    def recovered_poll(root, job, state):
        calls.append((job, state["monitor_interval_s"]))
        raise RuntimeError("stop offline recovered poll")

    monkeypatch.setattr(head_hidden_campaign, "poll_job", recovered_poll)
    with pytest.raises(RuntimeError, match="stop offline recovered poll"):
        head_hidden_campaign.run(tmp_path, parent, 123)
    assert calls == [(999, 300)]


def test_one_coordinator_lock_prevents_duplicate_campaign_process(tmp_path):
    with head_hidden_campaign.campaign_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Existing debug coordinator"):
            with head_hidden_campaign.campaign_lock(tmp_path):
                pytest.fail("duplicate coordinator acquired lock")
    with head_hidden_campaign.campaign_lock(tmp_path):
        pass


def test_successful_mock_submission_records_ledger_job_and_300s_remote_watcher(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    campaign_fixture(tmp_path, parent, 123)
    commands = []
    monkeypatch.setattr(head_hidden_campaign, "check_frozen_sources", lambda *args: None)
    monkeypatch.setattr(head_hidden_campaign, "preflight", lambda *args: None)
    monkeypatch.setattr(fp32_head_campaign, "stage_inputs", lambda *args: ("/remote/stage", "/remote/home"))

    def checked_ssh(command):
        commands.append(command)
        if "sbatch --parsable" in command:
            assert json.loads((tmp_path / "campaign-state.json").read_text())["phase"] == "submission_intent_do_not_duplicate"
            assert "test ! -e /remote/stage/submitted-job.txt" in command
            assert "sacct -n -X -P -j 123" in command
            return "999"
        if "watch-trace-remote.sh" in command:
            return "456"
        return "verified mock queue/accounting"

    monkeypatch.setattr(head_hidden_campaign, "checked_ssh", checked_ssh)
    monkeypatch.setattr(head_hidden_campaign, "poll_job", lambda *args: (_ for _ in ()).throw(RuntimeError("stop mock poll")))
    with pytest.raises(RuntimeError, match="stop mock poll"):
        head_hidden_campaign.run(tmp_path, parent, 123)
    state = json.loads((tmp_path / "campaign-state.json").read_text())
    assert state["job"] == 999 and state["remote_watcher_pid"] == 456
    assert state["submission_ledger"] == "/remote/stage/submitted-job.txt"
    assert sum("sbatch --parsable" in command for command in commands) == 1
    assert "watch-trace-remote.sh\" 999" in commands[-1]
    assert state["monitor_interval_s"] == 300


def test_resumed_poll_uses_same_300s_interval_and_never_performance_job(tmp_path, monkeypatch):
    import stress_campaign

    calls = []
    sleeps = []
    responses = iter(["999|PENDING|0:0|start|end|elapsed|node", "999|COMPLETED|0:0|start|end|elapsed|node"])

    def query(command):
        calls.append(command)
        return SimpleNamespace(stdout=next(responses), stderr="", returncode=0)

    monkeypatch.setattr(stress_campaign, "ssh", query)
    monkeypatch.setattr(stress_campaign.time, "sleep", sleeps.append)
    state = {"concurrent_performance_job": 123, "monitor_interval_s": 300}
    assert head_hidden_campaign.poll_job(tmp_path, 999, state) == "COMPLETED"
    assert len(sleeps) == 1 and 299 <= sleeps[0] <= 300
    assert all("squeue -h -j 999" in command and "squeue -h -j 123" not in command for command in calls)
