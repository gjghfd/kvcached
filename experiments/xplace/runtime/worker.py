import json
import os
import time
from pathlib import Path

import torch
from vllm.distributed.parallel_state import get_dp_group, get_ep_group, get_tp_group
from vllm.v1.worker.gpu_worker import Worker

from parallel_topology import worker_filename


class VerificationWorker(Worker):
    def initialize_from_config(self, kv_cache_config):
        super().initialize_from_config(kv_cache_config)
        from kvcached import utils
        from kvcached.integration.vllm.interfaces import observability_snapshot_dict

        runtime = observability_snapshot_dict()
        if not runtime["initialized"] or runtime["world_size"] != get_tp_group().world_size:
            raise RuntimeError(f"KV runtime does not match the TP group: {runtime}")
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
            "dp_pause_sync_interval": int(os.environ.get("DP_OPT_PAUSE_SYNC_INTERVAL", "32")),
        }
        if os.environ.get("XPLACE_CAUSAL_MANIFEST"):
            from causal_deployment import worker_attestation

            record["causal_deployment"] = worker_attestation(self)
        self._placement_timing_identity = {"service": record["service"], "dp_rank": record["dp_rank"],
                                           "tp_rank": groups["tp"]["rank"], "gpu_uuid": record["gpu_uuid"]}
        from live_observation import start_worker_observer

        start_worker_observer(self)
        path = Path(os.environ["DP_OPT_EVIDENCE_ROOT"]) / worker_filename(
            record["service"], record["dp_rank"], groups["tp"]["rank"], groups["tp"]["world_size"])
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(path)
        self._benefit_capture_enabled = False
        self._benefit_capture_counts = {}
        if os.environ.get("DP_OPT_CAPTURE_EVIDENCE") == "1":
            original = self.model_runner._determine_batch_execution_and_padding

            def capture(*args, **kwargs):
                result = original(*args, **kwargs)
                if self._benefit_capture_enabled:
                    mode, batch, ubatch, across_dp, stats = result
                    key = (str(mode), int(batch.num_tokens), int(kwargs.get("num_tokens", 0)), bool(ubatch))
                    self._benefit_capture_counts[key] = self._benefit_capture_counts.get(key, 0) + 1
                return result

            self.model_runner._determine_batch_execution_and_padding = capture

        from graph_execution import start_worker_graph_execution

        start_worker_graph_execution(self)

    def placement_graph_execution(self, command, case):
        from graph_execution import worker_graph_control

        return worker_graph_control(self, command, case)

    def placement_memory_snapshot(self, release_unused=False):
        torch.cuda.synchronize(self.device)
        if release_unused:
            torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.device)
        return {"gpu_uuid": str(torch.cuda.get_device_properties(self.device).uuid),
                "rank": get_dp_group().rank_in_group, "tp_rank": get_tp_group().rank_in_group,
                "tp_size": get_tp_group().world_size,
                "gpu_name": torch.cuda.get_device_name(self.device),
                "free_mib": free / 1024**2, "total_mib": total / 1024**2,
                "torch_allocated_mib": torch.cuda.memory_allocated(self.device) / 1024**2,
                "torch_reserved_mib": torch.cuda.memory_reserved(self.device) / 1024**2,
                "released_unused_allocator_cache": release_unused}

    def placement_ballast(self, num_bytes):
        """Hold `num_bytes` more device memory; kvcached sizes every co-located pool from cudaMemGetInfo."""
        held = self.__dict__.setdefault("_placement_ballast", [])
        if isinstance(num_bytes, list):
            if len(num_bytes) != get_tp_group().world_size:
                raise ValueError("Ballast must specify every TP worker")
            num_bytes = num_bytes[get_tp_group().rank_in_group]
        if type(num_bytes) is not int or num_bytes < 0:
            raise ValueError("Ballast bytes must be a nonnegative integer")
        if num_bytes > 0:
            held.append(torch.empty(int(num_bytes), dtype=torch.uint8, device=self.device))
        torch.cuda.synchronize(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        return {"held_bytes": sum(tensor.numel() for tensor in held), "free_mib": free / 1024**2,
                "total_mib": total / 1024**2}

    def placement_timing(self, command, case=None):
        from timing_probe import worker_timing_control

        return worker_timing_control(self, command, case)

    def placement_ballast_release(self):
        held = self.__dict__.pop("_placement_ballast", [])
        released = sum(tensor.numel() for tensor in held)
        held.clear()
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.device)
        return {"released_bytes": released, "free_mib": free / 1024**2, "total_mib": total / 1024**2}

    def benefit_capture(self, command):
        if os.environ.get("DP_OPT_CAPTURE_EVIDENCE") != "1":
            raise RuntimeError("Capture evidence was not enabled before worker initialization")
        if command == "start":
            self._benefit_capture_counts = {}
            self._benefit_capture_enabled = True
        elif command == "stop":
            self._benefit_capture_enabled = False
        else:
            raise ValueError("Unknown capture evidence command")
        return {"rank": self.parallel_config.data_parallel_rank,
                "histogram": [{"mode": key[0], "padded_tokens": key[1], "unpadded_tokens": key[2],
                               "ubatch": key[3], "steps": count}
                              for key, count in sorted(self._benefit_capture_counts.items())]}
