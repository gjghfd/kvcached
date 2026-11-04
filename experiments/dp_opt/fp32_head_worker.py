import hashlib
import inspect
import json
import os
from pathlib import Path

from fp32_head_probe import validate_output, validate_worker
from workaware_worker import WorkawareVerificationWorker


class FP32HeadVerificationWorker(WorkawareVerificationWorker):
    def initialize_from_config(self, kv_cache_config):
        super().initialize_from_config(kv_cache_config)
        model = self.model_runner.get_model()
        compilation = self.vllm_config.compilation_config
        source = Path(inspect.getsourcefile(type(model.logits_processor)))
        record = {"service": os.environ["DP_OPT_SERVICE"],
                  "dp_rank": self.parallel_config.data_parallel_rank,
                  "model_dtype": str(self.model_config.dtype),
                  "head_weight_dtype": str(model.lm_head.weight.dtype),
                  "floating_parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()
                                                       if parameter.is_floating_point()}),
                  "effective_head_dtype": str(self.model_config.head_dtype),
                  "processor_head_dtype": str(model.logits_processor.head_dtype),
                  "head_quant_method": type(model.lm_head.quant_method).__name__,
                  "compile_mode": int(compilation.mode),
                  "cudagraph_mode": compilation.cudagraph_mode.name,
                  "cudagraph_capture_sizes": list(compilation.cudagraph_capture_sizes),
                  "max_num_batched_tokens": self.scheduler_config.max_num_batched_tokens,
                  "max_model_len": self.model_config.max_model_len,
                  "kv_cache_memory_bytes": self.cache_config.kv_cache_memory_bytes,
                  "attention_tp": self.parallel_config.tensor_parallel_size,
                  "dp": self.parallel_config.data_parallel_size,
                  "ep_enabled": self.parallel_config.enable_expert_parallel,
                  "batch_invariant_enabled": os.environ.get("VLLM_BATCH_INVARIANT", "0") not in ("", "0"),
                  "logits_processor_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        validate_worker(record)
        root = Path(os.environ["DP_OPT_EVIDENCE_ROOT"])
        identity = f"{record['service']}-dp{record['dp_rank']}"
        (root / f"fp32-head-engine-{identity}.json").write_text(json.dumps(record, indent=2) + "\n")

        def record_output(module, inputs, output):
            observation = {"hidden_dtype": str(inputs[1].dtype), "hidden_device": inputs[1].device.type,
                           "logits_dtype": str(output.dtype)}
            validate_output(observation)
            (root / f"fp32-head-output-{identity}.json").write_text(json.dumps(observation, indent=2) + "\n")
            handle.remove()

        handle = model.logits_processor.register_forward_hook(record_output)
