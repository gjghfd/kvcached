import hashlib
import json
import os
from pathlib import Path

import torch

from fp32_head_worker import FP32HeadVerificationWorker
from head_hidden_probe import TARGETS, context_digest, selected_position


class HeadHiddenVerificationWorker(FP32HeadVerificationWorker):
    def initialize_from_config(self, kv_cache_config):
        super().initialize_from_config(kv_cache_config)
        model = self.model_runner.get_model()
        root = Path(os.environ["DP_OPT_EVIDENCE_ROOT"])
        rank = self.parallel_config.data_parallel_rank
        service = os.environ["DP_OPT_SERVICE"]
        if (model.logits_processor.scale != 1 or model.logits_processor.soft_cap is not None
                or model.logits_processor.logits_as_input or getattr(model.lm_head, "bias", None) is not None
                or self.vllm_config.speculative_config is not None or self.scheduler_config.async_scheduling):
            raise RuntimeError("Head attribution requires an unscaled unbiased non-speculative synchronous projection")
        weights = {str(prompt): {"candidates": target["candidates"],
                                "rows": model.lm_head.weight[target["candidates"]].detach().cpu().float().tolist()}
                   for prompt, target in TARGETS.items()}
        (root / f"head-candidate-weights-{service}-dp{rank}.json").write_text(json.dumps({
            "service": service, "dp_rank": rank, "weight_dtype": str(model.lm_head.weight.dtype),
            "weights": weights}, indent=2) + "\n")
        directory = root / "head-hidden-snapshots"
        directory.mkdir(exist_ok=True)

        def snapshot(module, inputs, output):
            identifiers = self.model_runner.input_batch.req_ids
            selected = [(index, identifier, self.model_runner.requests[identifier])
                        for index, identifier in enumerate(identifiers)
                        if selected_position(self.model_runner.requests[identifier]) is not None]
            if not selected:
                return
            if inputs[1].shape[0] != len(identifiers) or output.shape[0] != len(identifiers):
                raise RuntimeError("Sample hidden rows are not one-to-one with live request IDs")
            for index, identifier, state in selected:
                position = len(state.output_token_ids)
                consumed = int(self.model_runner.optimistic_seq_lens_cpu[index])
                discarded = bool(self.model_runner.discard_request_mask.np[index])
                if discarded:
                    with (root / f"head-hidden-skipped-{service}-dp{rank}.jsonl").open("a") as destination:
                        destination.write(json.dumps({"native_request_id": identifier, "position": position,
                            "forward_consumed_prefix_tokens": consumed, "sample_discarded": True,
                            "reason": "partial_reprefill_logits_not_accepted_by_sampler"}) + "\n")
                    continue
                if consumed != state.num_prompt_tokens + position:
                    raise RuntimeError("Head snapshot forward did not consume the complete predicted prefix")
                reference = identifier.startswith("reference-")
                phase = json.loads((root / "phase.json").read_text()) if not reference else {}
                repeat = None if reference else phase["repeat"]
                candidates = TARGETS[state.num_prompt_tokens]["candidates"]
                prefix = list(state.prompt_token_ids) + list(state.output_token_ids)
                observation = {"service": service, "dp_rank": rank, "native_request_id": identifier,
                    "role": "reference" if reference else "trial", "repeat": repeat,
                    "layout": phase.get("layout"), "position": position,
                    "prompt_tokens": state.num_prompt_tokens, "prompt_token_ids": list(state.prompt_token_ids),
                    "output_prefix_token_ids": list(state.output_token_ids), "context_token_ids": prefix,
                    "forward_consumed_prefix_tokens": consumed, "sample_discarded": False,
                    "context_sha256": context_digest(prefix), "batch_rows": len(identifiers),
                    "batch_native_request_ids": list(identifiers), "row_index": index, "hidden_dtype": str(inputs[1].dtype),
                    "logits_dtype": str(output.dtype), "hidden": inputs[1][index].detach().cpu().float().tolist(),
                    "candidates": candidates, "candidate_logits": output[index, candidates].detach().cpu().float().tolist(),
                    "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    "allow_bf16_reduced_precision_reduction_split_k": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,
                    "observation_includes_CPU_synchronization": True, "is_algorithm_benefit": False}
                tag = hashlib.sha256(identifier.encode()).hexdigest()[:12]
                target = directory / f"{observation['role']}-r{repeat}-dp{rank}-p{position}-{tag}.json"
                with target.open("x") as destination:
                    destination.write(json.dumps(observation, indent=2) + "\n")

        model.logits_processor.register_forward_hook(snapshot)
