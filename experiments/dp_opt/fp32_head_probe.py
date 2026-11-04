import argparse
import json
import os
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import live_trace
import static_batch_probe
from batch_numerics_probe import EXPECTED_FAILURES, stream_with_logprobs
from verify_trace import verify


CAPTURES = [1, 8, 16, 32, 64, 128, 256]
MIGRATION_POLICIES = ["no_move", "random_no_return", "kv_only_headroom"]


def validate_worker(record):
    expected = {"model_dtype": "torch.bfloat16", "head_weight_dtype": "torch.bfloat16",
                "floating_parameter_dtypes": ["torch.bfloat16"],
                "effective_head_dtype": "torch.float32", "processor_head_dtype": "torch.float32",
                "head_quant_method": "UnquantizedEmbeddingMethod", "compile_mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": CAPTURES,
                "max_num_batched_tokens": 4096, "max_model_len": 131072,
                "kv_cache_memory_bytes": 25769803776, "attention_tp": 1, "dp": 2,
                "ep_enabled": True, "batch_invariant_enabled": False}
    for name, value in expected.items():
        if record.get(name) != value:
            raise ValueError("FP32 head worker configuration differs: " + name)


def validate_output(record):
    if (record.get("hidden_dtype") != "torch.bfloat16"
            or record.get("logits_dtype") != "torch.float32"
            or record.get("hidden_device") != "cuda"):
        raise ValueError("Require actual CUDA BF16 hidden states and FP32 logits")


@contextmanager
def instrument():
    original_arguments, original_stream = live_trace.engine_arguments, live_trace.stream

    def engine_arguments(*args, **kwargs):
        arguments = original_arguments(*args, **kwargs)
        overrides = getattr(arguments, "hf_overrides", {})
        if not isinstance(overrides, dict) or overrides.get("head_dtype") not in (None, "model", "float32"):
            raise ValueError("Do not replace unknown HF overrides")
        arguments.hf_overrides = {**overrides, "head_dtype": "float32"}
        arguments.worker_cls = "fp32_head_worker.FP32HeadVerificationWorker"
        return arguments

    if os.environ.get("VLLM_BATCH_INVARIANT", "0") not in ("", "0"):
        raise ValueError("FP32 head diagnostic must not enable batch invariance")
    live_trace.engine_arguments, live_trace.stream = engine_arguments, stream_with_logprobs
    try:
        yield
    finally:
        live_trace.engine_arguments, live_trace.stream = original_arguments, original_stream


def verify_precision(root):
    root = Path(root)
    declaration = json.loads((root / "fp32-head-config.json").read_text())
    if declaration != {"hf_overrides": {"head_dtype": "float32"},
                       "reference_generation": "fresh_serial_same_FP32_head_configuration",
                       "is_algorithm_benefit": False, "strict_token_check_relaxed": False}:
        raise ValueError("FP32 head declaration differs")
    records = []
    for service in ("instance-1", "instance-2"):
        for rank in (0, 1):
            record = json.loads((root / f"fp32-head-engine-{service}-dp{rank}.json").read_text())
            output = json.loads((root / f"fp32-head-output-{service}-dp{rank}.json").read_text())
            if record["service"] != service or record["dp_rank"] != rank:
                raise ValueError("FP32 head worker identity differs")
            validate_worker(record)
            validate_output(output)
            records.append(record)
    if len({record["logits_processor_sha256"] for record in records}) != 1:
        raise ValueError("Workers use different logits implementations")
    return {"four_workers_verified": True, "actual_FP32_logits_verified": True,
            "BF16_model_and_head_weights_verified": True,
            "logits_processor_sha256": records[0]["logits_processor_sha256"]}


def analyze(root, stage):
    root = Path(root)
    precision = verify_precision(root)
    if stage == "static":
        result = static_batch_probe.analyze(root)
        passed = not result["no_copy_divergence_reproduced"] and result["underlying_strict_error"] is None
        if passed and json.loads((root / "summary.json").read_text())["status"] != "passed":
            raise ValueError("A static pass requires the original strict summary")
    elif stage == "migration":
        config = json.loads((root / "config.json").read_text())
        if config["policies"] != MIGRATION_POLICIES or config["repeats"] != 1:
            raise ValueError("Require the predeclared same-head three-policy migration control")
        result = verify(root, root / "frozen/inputs/prompt-bank.json")
        counts = {policy: len(json.loads((root / f"r0-{policy}/migrations.json").read_text()))
                  for policy in MIGRATION_POLICIES}
        if counts["no_move"] or not counts["random_no_return"] or not counts["kv_only_headroom"]:
            raise ValueError("Migration control requires actual migrated and no-move executions")
        result = {"full_trace_verification": result, "actual_migration_hops": counts}
        passed = True
    else:
        raise ValueError("Unknown precision diagnostic stage")
    return {"stage": stage, "strict_correctness_passed": passed, "precision": precision,
            "diagnostic": result, "is_algorithm_benefit": False,
            "strict_token_check_relaxed": False, "numerical_root_cause_proven": False}


def main():
    if "--verify" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--verify", type=Path, required=True)
        parser.add_argument("--stage", choices=("static", "migration"), required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        result = analyze(args.verify, args.stage)
        with args.output.open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
        if not result["strict_correctness_passed"]:
            raise RuntimeError("FP32 head static outputs still diverge; do not continue")
        return
    stage = os.environ["DP_OPT_HEAD_PROBE_STAGE"]
    if stage not in ("static", "migration"):
        raise ValueError("Unknown precision diagnostic stage")
    root = Path(sys.argv[sys.argv.index("--output") + 1])
    root.mkdir(parents=True, exist_ok=True)
    for name in ("fp32_head_probe.py", "fp32_head_worker.py"):
        with (root / name).open("xb") as output:
            output.write(Path(__file__).with_name(name).read_bytes())
    live_trace.save(root / "fp32-head-config.json", {
        "hf_overrides": {"head_dtype": "float32"},
        "reference_generation": "fresh_serial_same_FP32_head_configuration",
        "is_algorithm_benefit": False, "strict_token_check_relaxed": False})
    failure = None
    context = static_batch_probe.instrument() if stage == "static" else nullcontext()
    with context, instrument():
        try:
            live_trace.main()
        except RuntimeError as error:
            if str(error) not in EXPECTED_FAILURES or stage != "static":
                raise
            failure = error
    if stage == "migration":
        live_trace.save(root / "fp32-head-runtime-completion.json", {
            "strict_runtime_finished": True, "independent_verification_pending": True,
            "is_algorithm_benefit": False})
        return
    result = analyze(root, stage)
    live_trace.save(root / "fp32-head-diagnostic.json", result)
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
