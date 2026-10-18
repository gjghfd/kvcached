import json
import os
import time
from pathlib import Path

import torch
from vllm.distributed.parallel_state import get_dp_group

from worker import VerificationWorker


class MotivationWorker(VerificationWorker):
    def initialize_from_config(self, kv_cache_config):
        super().initialize_from_config(kv_cache_config)
        self._motivation_trial = None
        self._motivation_events = []
        self._motivation_prompt_lengths = {}
        self._motivation_sequence = 0
        self._motivation_profile = False
        self._motivation_phase_events = None
        self._motivation_phase_pending = {}
        self._motivation_phase_hooks = []
        self._motivation_collective_originals = []
        self._motivation_collective_calls = None
        original = self.model_runner._determine_batch_execution_and_padding

        def determine(*args, **kwargs):
            result = original(*args, **kwargs)
            if self._motivation_trial is not None:
                mode, batch, ubatch, across_dp, stats = result
                self._motivation_batch = {
                    "cudagraph_mode": str(mode), "padded_tokens": int(batch.num_tokens),
                    "unpadded_tokens": int(kwargs.get("num_tokens", 0)),
                    "should_ubatch": bool(ubatch),
                }
            return result

        self.model_runner._determine_batch_execution_and_padding = determine

    def install_phase_hooks(self):
        if self._motivation_phase_hooks:
            raise RuntimeError("Phase hooks already installed")
        model = self.model_runner.get_model()
        for layer_index, layer in enumerate(model.model.layers):
            for phase, module in (("attention", layer.self_attn),
                                  ("attention_core", layer.self_attn.attn), ("moe", layer.mlp)):
                key = f"{phase}-{layer_index}"

                def begin(module, inputs, key=key):
                    if self._motivation_phase_events is not None:
                        started = torch.cuda.Event(enable_timing=True)
                        started.record()
                        self._motivation_phase_pending[key] = started

                def end(module, inputs, output, phase=phase, key=key):
                    if self._motivation_phase_events is not None:
                        ended = torch.cuda.Event(enable_timing=True)
                        ended.record()
                        self._motivation_phase_events.append(
                            (phase, self._motivation_phase_pending.pop(key), ended)
                        )

                self._motivation_phase_hooks.extend((module.register_forward_pre_hook(begin),
                                                      module.register_forward_hook(end)))
        group = get_dp_group()
        for phase, method_name in (("dispatch", "all_gatherv"), ("combine", "reduce_scatterv")):
            original = getattr(group, method_name)
            self._motivation_collective_originals.append((group, method_name, original))

            def collective(*args, phase=phase, original=original, **kwargs):
                if self._motivation_phase_events is None:
                    return original(*args, **kwargs)
                sizes = kwargs.get("sizes")
                self._motivation_collective_calls.append({
                    "phase": phase, "sizes": None if sizes is None else list(sizes),
                })
                started = torch.cuda.Event(enable_timing=True)
                started.record()
                result = original(*args, **kwargs)
                ended = torch.cuda.Event(enable_timing=True)
                ended.record()
                self._motivation_phase_events.append((phase, started, ended))
                return result

            setattr(group, method_name, collective)

    def motivation_control(self, command, trial="", case_name=""):
        if command in ("start", "start_profile"):
            self._motivation_profile = command == "start_profile"
            if self._motivation_profile:
                self.install_phase_hooks()
            self._motivation_trial = trial
            self._motivation_events = []
            self._motivation_prompt_lengths = {}
            self._motivation_sequence = 0
        elif command == "stop":
            self._motivation_trial = None
            self._motivation_profile = False
            for handle in self._motivation_phase_hooks:
                handle.remove()
            self._motivation_phase_hooks = []
            for group, method_name, original in self._motivation_collective_originals:
                setattr(group, method_name, original)
            self._motivation_collective_originals = []
            if any(event.get("phase_cuda_events") for event in self._motivation_events):
                torch.cuda.synchronize(self.device)
            for event in self._motivation_events:
                cuda_events = event.pop("phase_cuda_events", [])
                if cuda_events:
                    for phase in ("attention", "attention_core", "moe", "dispatch", "combine"):
                        if sum(name == phase for name, started, ended in cuda_events) != 48:
                            raise RuntimeError("CUDA diagnostic must cover all 48 model layers")
                    durations = {}
                    for phase, started, ended in cuda_events:
                        durations[phase] = durations.get(phase, 0) + started.elapsed_time(ended)
                    event["phase_cuda_ms"] = durations
            report = {
                "trial": trial, "rank": self.parallel_config.data_parallel_rank,
                "events": self._motivation_events,
            }
            root = Path(os.environ["DP_OPT_EVIDENCE_ROOT"]) / "motivation"
            if case_name:
                root = root / "cases" / case_name
            path = root / "traces" / f"{trial}-dp{report['rank']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(report) + "\n")
            temporary.replace(path)
        else:
            raise ValueError("Unknown motivation control command")
        return {"rank": self.parallel_config.data_parallel_rank, "command": command,
                "trial": trial, "event_count": len(self._motivation_events)}

    def begin_phase_sample(self):
        self._motivation_phase_events = None
        if self._motivation_profile and self._motivation_sequence % 32 == 0:
            self._motivation_phase_events = []
            self._motivation_phase_pending = {}
            self._motivation_collective_calls = []
            started = torch.cuda.Event(enable_timing=True)
            started.record()
            return started
        return None

    def end_phase_sample(self, started):
        if started is None:
            return []
        ended = torch.cuda.Event(enable_timing=True)
        ended.record()
        events = self._motivation_phase_events
        events.append(("worker_execute", started, ended))
        self._motivation_phase_events = None
        return events

    def execute_model(self, scheduler_output):
        if self._motivation_trial is None:
            return super().execute_model(scheduler_output)
        for request in scheduler_output.scheduled_new_reqs:
            if request.prompt_token_ids is None:
                raise RuntimeError("Motivation fixture requires token-ID prompts")
            self._motivation_prompt_lengths[request.req_id] = len(request.prompt_token_ids)
        cached = scheduler_output.scheduled_cached_reqs
        computed = dict(zip(cached.req_ids, cached.num_computed_tokens))
        outputs = dict(zip(cached.req_ids, cached.num_output_tokens))
        requests = [
            {"id": request_id, "tokens": count,
             "computed_tokens": computed.get(request_id),
             "output_tokens": outputs.get(request_id),
             "prompt_tokens": self._motivation_prompt_lengths.get(request_id)}
            for request_id, count in scheduler_output.num_scheduled_tokens.items()
        ]
        self._motivation_batch = None
        started_ns = time.time_ns()
        phase_start = self.begin_phase_sample()
        result = super().execute_model(scheduler_output)
        phase_events = self.end_phase_sample(phase_start)
        self._motivation_events.append({
            "sequence": self._motivation_sequence, "kind": "execute",
            "start_ns": started_ns, "end_ns": time.time_ns(), "requests": requests,
            "new_requests": len(scheduler_output.scheduled_new_reqs),
            "resumed_requests": len(cached.resumed_req_ids),
            "batch": self._motivation_batch,
            "phase_cuda_events": phase_events,
            "collective_calls": self._motivation_collective_calls if phase_start is not None else [],
        })
        self._motivation_sequence += 1
        return result

    def execute_dummy_batch(self):
        if self._motivation_trial is None:
            return super().execute_dummy_batch()
        self._motivation_batch = None
        started_ns = time.time_ns()
        phase_start = self.begin_phase_sample()
        result = super().execute_dummy_batch()
        phase_events = self.end_phase_sample(phase_start)
        self._motivation_events.append({
            "sequence": self._motivation_sequence, "kind": "dummy",
            "start_ns": started_ns, "end_ns": time.time_ns(), "requests": [],
            "new_requests": 0, "resumed_requests": 0, "batch": self._motivation_batch,
            "phase_cuda_events": phase_events,
            "collective_calls": self._motivation_collective_calls if phase_start is not None else [],
        })
        self._motivation_sequence += 1
        return result
