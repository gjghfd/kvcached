import json
import os
import time
from pathlib import Path

import torch
from vllm.distributed.parallel_state import get_dp_group, get_ep_group, get_tp_group
from vllm.v1.worker.gpu_worker import Worker


class VerificationWorker(Worker):
    def initialize_from_config(self, kv_cache_config):
        super().initialize_from_config(kv_cache_config)
        from kvcached import utils
        from kvcached.integration.vllm.interfaces import observability_snapshot_dict

        runtime = observability_snapshot_dict()
        if not runtime["initialized"] or runtime["world_size"] != 1:
            raise RuntimeError(f"KV runtime is not rank-local: {runtime}")
        if runtime["device"] != str(self.device):
            raise RuntimeError(f"KV runtime device does not match worker: {runtime}")
        model = self.model_runner.get_model()
        expert_ids = []
        for layer in model.model.layers:
            experts = layer.mlp.experts
            routed = getattr(experts, "routed_experts", experts)
            expert_ids.append([
                expert_index for expert_index, local_index
                in enumerate(routed.expert_map.cpu().tolist()) if local_index >= 0
            ])
        groups = {
            group_name: {"rank": group.rank_in_group, "world_size": group.world_size}
            for group_name, group in (("tp", get_tp_group()),
                                     ("dp", get_dp_group()), ("ep", get_ep_group()))
        }
        record = {
            "service": os.environ["DP_OPT_SERVICE"],
            "pid": os.getpid(),
            "dp_rank": self.parallel_config.data_parallel_rank,
            "gpu_uuid": str(torch.cuda.get_device_properties(self.device).uuid),
            "gpu_name": torch.cuda.get_device_name(self.device),
            "checkpoint": self.model_config.model,
            "parameter_bytes": sum(parameter.numel() * parameter.element_size()
                                   for parameter in model.parameters()),
            "groups": groups,
            "expert_ids": expert_ids,
            "kv_num_blocks": kv_cache_config.num_blocks,
            "kv_block_size": self.cache_config.block_size,
            "max_model_len": self.model_config.max_model_len,
            "kv_cache_memory_bytes": self.cache_config.kv_cache_memory_bytes,
            "max_num_seqs": self.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": self.scheduler_config.max_num_batched_tokens,
            "kv_runtime": runtime,
            "ipc_name": utils.DEFAULT_IPC_NAME,
            "initialized_ns": time.time_ns(),
        }
        path = Path(os.environ["DP_OPT_EVIDENCE_ROOT"]) / (
            f'worker-{record["service"]}-dp{record["dp_rank"]}.json'
        )
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(path)
