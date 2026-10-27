import json
import os
from pathlib import Path

from worker import VerificationWorker


class WorkawareVerificationWorker(VerificationWorker):
    def initialize_from_config(self, kv_cache_config):
        super().initialize_from_config(kv_cache_config)
        compilation = self.vllm_config.compilation_config
        record = {
            "service": os.environ["DP_OPT_SERVICE"],
            "dp_rank": self.parallel_config.data_parallel_rank,
            "dtype": str(self.model_config.dtype),
            "compile_mode": int(compilation.mode),
            "cudagraph_mode": compilation.cudagraph_mode.name,
            "cudagraph_capture_sizes": list(compilation.cudagraph_capture_sizes),
            "max_num_batched_tokens": self.scheduler_config.max_num_batched_tokens,
        }
        root = Path(os.environ["DP_OPT_EVIDENCE_ROOT"])
        target = root / f"workaware-engine-{record['service']}-dp{record['dp_rank']}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(target)
