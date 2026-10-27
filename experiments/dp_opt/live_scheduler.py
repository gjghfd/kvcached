import hashlib
import os
import pickle
import time
from concurrent.futures import Future
import uuid
from pathlib import Path

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.request import RequestStatus


def unique_views(runner):
    views, seen = [], set()
    for entry in runner.kv_caches:
        for tensor in entry if isinstance(entry, (list, tuple)) else (entry,):
            view = getattr(tensor, "_kvcached_block_copy_view", None)
            if view is None:
                raise RuntimeError("Migration requires kvcached logical block copy metadata")
            key = (view.data_ptr(), tuple(view.shape), tuple(view.stride()))
            if key not in seen:
                seen.add(key)
                views.append(view)
    if not views:
        raise RuntimeError("No KV backing views")
    return views


def kv_file(worker, operation, blocks, filename):
    import torch

    runner = worker.model_runner
    views = unique_views(runner)
    path = Path(filename)
    root = Path(os.environ["DP_OPT_TRANSFER_ROOT"]).resolve()
    if path.resolve().parent != root:
        raise ValueError("Transfer file must stay in the private job directory")
    if operation not in ("export", "import"):
        raise ValueError("Unknown KV transfer operation")
    if not blocks or len(set(blocks)) != len(blocks) or any(not isinstance(block, int) for block in blocks):
        raise ValueError("Migration requires nonempty unique integer block identifiers")
    if any(block <= 0 or block >= view.shape[0] for block in blocks for view in views):
        raise ValueError("Migration block outside cache")
    stage_s = {}
    started = time.monotonic()

    def mark(name):
        nonlocal started
        now = time.monotonic()
        stage_s[name] = now - started
        started = now

    with torch.inference_mode():
        torch.cuda.synchronize(worker.device)
        mark("initial_synchronize")
        indices = torch.tensor(blocks, dtype=torch.long, device=worker.device)
        mark("indices")
        if operation == "export":
            payload = [view.index_select(0, indices).cpu() for view in views]
            mark("gather_device_to_host")
            torch.save(payload, path)
            mark("serialize_file")
        elif operation == "import":
            payload = torch.load(path, map_location="cpu", weights_only=True)
            mark("deserialize_file")
            if len(payload) != len(views):
                raise ValueError("Mismatched KV backing pools")
            for view, data in zip(views, payload):
                if list(data.shape) != [len(blocks)] + list(view.shape[1:]) or data.dtype != view.dtype:
                    raise ValueError("Mismatched KV transfer layout")
                view.index_copy_(0, indices, data.to(worker.device))
            torch.cuda.synchronize(worker.device)
            mark("host_to_device_scatter")
            payload = [view.index_select(0, indices).cpu() for view in views]
            mark("readback_device_to_host")
    digest = hashlib.sha256()
    for data in payload:
        digest.update(data.contiguous().numpy().tobytes())
    mark("sha256")
    return {"sha256": digest.hexdigest(), "bytes": sum(data.numel() * data.element_size() for data in payload),
            "stage_s": stage_s, "views": len(views)}


def install_runner_state(worker, state):
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    runner = worker.model_runner
    request_id = state["request_id"]
    if state.get("remove"):
        runner.input_batch.remove_request(request_id)
        runner.requests.pop(request_id, None)
        return True
    request = pickle.loads(state["request"])
    if request.request_id != request_id:
        raise ValueError("Runner request identity differs from payload")
    if request.sampling_params.temperature != 0:
        raise ValueError("Experimental migration supports greedy sampling only")
    runner.input_batch.remove_request(request_id)
    runner.requests[request_id] = CachedRequestState(
        req_id=request_id, prompt_token_ids=request.prompt_token_ids, mm_features=[],
        sampling_params=request.sampling_params, pooling_params=None, generator=None,
        block_ids=tuple(state["blocks"]), num_computed_tokens=request.num_computed_tokens,
        output_token_ids=list(request.output_token_ids), lora_request=None)
    return True


def kv_peer(worker, blocks, source_rank, integrity="sha256"):
    from peer_transfer import transfer_blocks

    return transfer_blocks(worker, unique_views(worker.model_runner), blocks, source_rank,
                           int(os.environ.get("DP_OPT_TRANSFER_CHUNK_MIB", "32")), integrity)


class PlacementScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        parallel = self.parallel_config
        if (parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
                or parallel.data_parallel_size != 2 or self.scheduler_config.async_scheduling
                or self.cache_config.enable_prefix_caching or self.connector is not None
                or len(self.kv_cache_config.kv_cache_groups) != 1 or self.num_lookahead_tokens):
            raise ValueError("Placement experiment requires DP2/TP1/PP1, synchronous decode, one KV group, no APC/connector/speculation")
        self.placement_pending = {}
        self.placement_committed = set()
        self.placement_transaction = None
        self.placement_unreserved = {}
        self.placement_decode_progress = {}

    def placement_snapshot(self, refresh_physical_capacity=False):
        manager = self.kv_cache_manager.block_pool.kv_cache_manager
        if refresh_physical_capacity:
            manager._avail_physical_pages_cache = None
        if manager.in_shrink:
            raise RuntimeError("Live placement does not support an overlapping allocator shrink")
        records, owners = [], {}
        observed_at = time.monotonic()
        progress = getattr(self, "placement_decode_progress", {})
        progress = {identifier: samples for identifier, samples in progress.items() if identifier in self.requests}
        self.placement_decode_progress = progress
        for request in self.requests.values():
            groups = self.kv_cache_manager.get_block_ids(request.request_id)
            blocks = groups[0]
            grouped = manager.page_allocator.group_indices_by_page(blocks, manager.block_mem_size)
            for page_id in grouped:
                owners.setdefault(page_id, set()).add(request.request_id)
            movable = (request in self.running and request.num_computed_tokens >= request.num_prompt_tokens
                       and request.num_output_tokens > 0 and not request.num_in_flight_tokens
                       and request.sampling_params.temperature == 0 and not request.mm_features
                       and request.lora_request is None and not request.use_structured_output
                       and request.sampling_params.prompt_logprobs is None
                       and request.prompt_embeds is None and not request.resumable
                       and not request.spec_token_ids and request._block_hasher is None
                       and request.num_output_tokens < request.max_tokens)
            samples = progress.setdefault(request.request_id, [])
            if movable and self.pause_state != PauseState.PAUSED_ALL:
                if not samples or observed_at - samples[-1][0] >= .5:
                    samples.append((observed_at, request.num_output_tokens))
                    samples[:] = samples[-8:]
            duration = samples[-1][0] - samples[0][0] if len(samples) > 1 else 0
            rate = ((samples[-1][1] - samples[0][1]) / duration if duration > 0 else 0)
            remaining = (request.max_tokens - request.num_output_tokens) / rate if rate > 0 else None
            records.append({"id": request.request_id, "computed_tokens": request.num_computed_tokens,
                            "prompt_tokens": request.num_prompt_tokens, "output_tokens": request.num_output_tokens,
                            "max_tokens": request.max_tokens, "blocks": blocks, "movable": bool(movable),
                            "decode_rate_tokens_s": rate, "estimated_remaining_decode_s": remaining,
                            "reserved_at": getattr(request, "placement_reserved_at", None),
                            "reservation_pending": request.request_id in self.placement_unreserved})
        page_ids = list(manager.full_pages) + list(manager.avail_pages)
        null_pages = manager.page_allocator.group_indices_by_page([0], manager.block_mem_size)
        pages = [{"id": int(page_id), "owners": sorted(owners.get(page_id, set())),
                  "pinned": page_id not in owners or page_id in null_pages,
                  "memory_gib": manager._memory_limit_page_bundle_bytes() / 1024**3}
                 for page_id in page_ids]
        return {"rank": self.parallel_config.data_parallel_rank, "requests": records, "pages": pages,
                "observed_at": observed_at,
                "mapped_bytes": int(manager.get_mapped_memory_size()),
                "page_bytes": manager._memory_limit_page_bundle_bytes(),
                "block_bytes": manager.block_mem_size * manager.num_layers * manager.num_kv_buffers,
                "page_block_capacity": manager.page_size // manager.block_mem_size,
                "uniform_page_geometry": manager.page_size % manager.block_mem_size == 0,
                "block_size": self.cache_config.block_size,
                "mapped_free_block_slots": manager.num_avail_blocks + len(manager.reserved_blocks),
                "free_blocks": self.kv_cache_manager.block_pool.get_num_free_blocks()}

    def _require_transaction(self, transaction):
        if self.placement_transaction != transaction or self.pause_state != PauseState.PAUSED_ALL:
            raise RuntimeError("Placement transaction requires a paused scheduler and matching epoch")

    def placement_begin(self, transaction):
        if self.placement_transaction is not None and self.placement_transaction != transaction:
            raise RuntimeError("Another placement transaction is active")
        if self.pause_state != PauseState.PAUSED_ALL:
            raise RuntimeError("Pause DP group before beginning placement")
        observation = self.placement_snapshot()
        if any(not request["movable"] for request in observation["requests"]):
            raise ValueError("Controlled experiment requires an all-decode movable population")
        self.placement_transaction = transaction
        return observation

    def placement_export(self, transaction, request_id):
        self._require_transaction(transaction)
        request = self.requests[request_id]
        record = next(item for item in self.placement_snapshot()["requests"] if item["id"] == request_id)
        if not record["movable"]:
            raise ValueError("Request is not migration eligible")
        return {"request_id": request_id, "request": pickle.dumps(request), "blocks": record["blocks"]}

    def placement_probe(self, transaction, prompt_token_ids, max_tokens):
        from vllm import SamplingParams
        from vllm.v1.request import Request

        self._require_transaction(transaction)
        if (not prompt_token_ids or not isinstance(max_tokens, int) or max_tokens <= 0
                or any(not isinstance(token, int) or token < 0 for token in prompt_token_ids)
                or len(prompt_token_ids) + max_tokens > self.max_model_len):
            raise ValueError("Probe requires valid token IDs and a supported full reservation")
        if self.placement_pending:
            raise RuntimeError("Cannot probe during an uncommitted migration")
        request_id = "allocation-probe-" + uuid.uuid4().hex
        request = Request(request_id=request_id, prompt_token_ids=list(prompt_token_ids),
                          sampling_params=SamplingParams(temperature=0, max_tokens=max_tokens,
                                                         ignore_eos=True), pooling_params=None)
        before = self.placement_snapshot()
        try:
            allocation = self.kv_cache_manager.allocate_slots(
                request, len(prompt_token_ids) + max_tokens, has_scheduled_reqs=False)
            if allocation is not None:
                self.requests[request_id] = request
            return {"allocated": allocation is not None, "probe_request_id": request_id,
                    "before": before, "during": self.placement_snapshot(),
                    "blocks": list(self.kv_cache_manager.get_block_ids(request_id)[0])
                    if allocation is not None else []}
        finally:
            self.requests.pop(request_id, None)
            self.kv_cache_manager.free(request)

    def placement_prepare(self, transaction, payload):
        self._require_transaction(transaction)
        request_id = payload["request_id"]
        if request_id in self.requests or request_id in self.placement_pending:
            raise ValueError("Destination already owns request")
        request = pickle.loads(payload["request"])
        if request.request_id != request_id:
            raise ValueError("Request identity differs from payload")
        if not payload["blocks"] or len(set(payload["blocks"])) != len(payload["blocks"]):
            raise ValueError("Source reservation must contain unique blocks")
        computed = request.num_computed_tokens
        request.num_computed_tokens = 0
        try:
            allocation = self.kv_cache_manager.allocate_slots(
                request, len(payload["blocks"]) * self.cache_config.block_size, has_scheduled_reqs=False)
        except BaseException:
            self.kv_cache_manager.free(request)
            raise
        finally:
            request.num_computed_tokens = computed
        if allocation is None:
            self.kv_cache_manager.free(request)
            raise MemoryError("Destination could not allocate migration KV")
        blocks = self.kv_cache_manager.get_block_ids(request_id)[0]
        if len(blocks) != len(payload["blocks"]):
            self.kv_cache_manager.free(request)
            raise ValueError("Destination reservation block count differs from source")
        self.placement_pending[request_id] = request
        return {"request_id": request_id, "request": pickle.dumps(request), "blocks": [blocks]}

    def add_request(self, request):
        if request.request_id in self.requests or request.request_id in self.placement_pending:
            raise ValueError("Request identifier already exists")
        total = request.num_prompt_tokens + request.max_tokens
        if total > self.max_model_len:
            raise ValueError("Full sequence reservation exceeds max_model_len")
        try:
            if self.kv_cache_manager.allocate_slots(request, total, has_scheduled_reqs=False) is None:
                self.kv_cache_manager.free(request)
                self.placement_unreserved[request.request_id] = request
            else:
                request.placement_reserved_at = time.monotonic()
            super().add_request(request)
        except BaseException:
            self.placement_unreserved.pop(request.request_id, None)
            self.kv_cache_manager.free(request)
            raise

    def schedule(self, *args, **kwargs):
        for request_id, request in list(self.placement_unreserved.items()):
            if request_id not in self.requests:
                self.placement_unreserved.pop(request_id)
                continue
            if self.pause_state == PauseState.PAUSED_ALL or os.environ.get("DP_OPT_CONTROLLED_RESERVATION_RETRY") == "1":
                break
            total = request.num_prompt_tokens + request.max_tokens
            try:
                allocation = self.kv_cache_manager.allocate_slots(request, total, has_scheduled_reqs=False)
            except BaseException:
                self.kv_cache_manager.free(request)
                raise
            if allocation is None:
                self.kv_cache_manager.free(request)
                break
            request.placement_reserved_at = time.monotonic()
            self.placement_unreserved.pop(request_id)
        held = []
        for queue in (self.waiting, self.skipped_waiting):
            original = list(queue)
            requests = [request for request in original if request.request_id in self.placement_unreserved]
            queue.remove_requests(requests)
            held.append((queue, requests, original))
        try:
            return super().schedule(*args, **kwargs)
        finally:
            for queue, requests, original in held:
                if not requests:
                    continue
                remaining = list(queue)
                retained = {request.request_id for request in remaining + requests}
                original_ids = {request.request_id for request in original}
                ordered = [request for request in remaining if request.request_id not in original_ids]
                ordered.extend(request for request in original if request.request_id in retained
                               and request.request_id in self.requests)
                queue.remove_requests(remaining)
                for request in reversed(ordered):
                    queue.prepend_request(request)

    def placement_retry_reservation(self, request_id):
        request = self.requests.get(request_id)
        if request is None:
            return {"missing": True}
        if request_id not in self.placement_unreserved:
            return {"reservation_pending": False, "reserved_at": request.placement_reserved_at}
        if self.pause_state == PauseState.PAUSED_ALL:
            return {"reservation_pending": True, "reserved_at": None}
        try:
            allocation = self.kv_cache_manager.allocate_slots(
                request, request.num_prompt_tokens + request.max_tokens, has_scheduled_reqs=False)
        except BaseException:
            self.kv_cache_manager.free(request)
            raise
        if allocation is None:
            self.kv_cache_manager.free(request)
            return {"reservation_pending": True, "reserved_at": None}
        request.placement_reserved_at = time.monotonic()
        self.placement_unreserved.pop(request_id)
        return {"reservation_pending": False, "reserved_at": request.placement_reserved_at}

    def placement_commit(self, transaction, request_id):
        self._require_transaction(transaction)
        request = self.placement_pending.pop(request_id)
        request.status = RequestStatus.RUNNING
        request.events.clear()
        self.requests[request_id] = request
        self.running.append(request)
        self.placement_committed.add(request_id)
        self.prev_step_scheduled_req_ids.discard(request_id)
        return True

    def placement_remove(self, transaction, request_id):
        self._require_transaction(transaction)
        request = self.requests.pop(request_id)
        self.running.remove(request)
        self.kv_cache_manager.free(request)
        self.prev_step_scheduled_req_ids.discard(request_id)
        self.placement_committed.discard(request_id)
        return self.placement_snapshot()

    def placement_rollback(self, transaction, request_id):
        self._require_transaction(transaction)
        request = self.placement_pending.pop(request_id, None)
        if request is None and request_id in self.placement_committed:
            request = self.requests.pop(request_id)
            self.running.remove(request)
            self.placement_committed.remove(request_id)
        if request is not None:
            self.kv_cache_manager.free(request)
        return True

    def placement_end(self, transaction):
        self._require_transaction(transaction)
        if self.placement_pending:
            raise RuntimeError("Uncommitted destination KV remains")
        self.placement_transaction = None
        self.placement_committed.clear()
        return True


def bind_engine(engine):
    if not isinstance(engine.scheduler, PlacementScheduler):
        return
    interval = int(os.environ.get("DP_OPT_PAUSE_SYNC_INTERVAL", "32"))
    if interval not in (1, 32):
        raise ValueError("Experimental DP pause-sync interval must be1or32")
    if interval == 1:
        if not hasattr(engine, "_has_global_unfinished_reqs"):
            raise RuntimeError("Fast pause requires the validated DP engine")

        def sync(local_unfinished):
            from vllm.config import ParallelConfig

            engine.step_counter += 1
            unfinished, consensus = ParallelConfig.sync_dp_state(
                engine.dp_group, has_unfinished=local_unfinished, pending_pause=engine.pending_pause)
            if consensus:
                engine.ignore_start_dp_wave = True
                engine.pending_pause = False
            return unfinished

        engine._has_global_unfinished_reqs = sync
    elif hasattr(engine, "_has_global_unfinished_reqs"):
        install_boundary_pause(engine)
    for name in ("snapshot", "begin", "export", "prepare", "commit", "remove", "rollback", "end", "retry_reservation"):
        setattr(engine, "placement_" + name, getattr(engine.scheduler, "placement_" + name))

    def release(transaction, request_id, rollback=False):
        method = engine.scheduler.placement_rollback if rollback else engine.scheduler.placement_remove
        method(transaction, request_id)
        manager = engine.scheduler.kv_cache_manager.block_pool.kv_cache_manager
        if getattr(manager, "defer_physical_release", False):
            if engine.batch_queue:
                raise RuntimeError("Cannot release KV while worker batches remain in flight")
            manager.release_retired_pages_through(manager.capture_physical_release_marker())
        return True if rollback else engine.scheduler.placement_snapshot()

    def probe(transaction, prompt_token_ids, max_tokens):
        if engine.batch_queue:
            raise RuntimeError("Cannot probe KV while worker batches remain in flight")
        result = engine.scheduler.placement_probe(transaction, prompt_token_ids, max_tokens)
        manager = engine.scheduler.kv_cache_manager.block_pool.kv_cache_manager
        if getattr(manager, "defer_physical_release", False):
            manager.release_retired_pages_through(manager.capture_physical_release_marker())
        result["after"] = engine.scheduler.placement_snapshot()
        return result

    def transfer(transaction, operation, blocks, filename):
        engine.scheduler._require_transaction(transaction)
        return engine.collective_rpc(kv_file, args=(operation, blocks, filename))[0]

    def runner_state(transaction, state):
        engine.scheduler._require_transaction(transaction)
        return engine.collective_rpc(install_runner_state, args=(state,))[0]

    def peer_transfer(transaction, blocks, source_rank, integrity="sha256"):
        engine.scheduler._require_transaction(transaction)
        return engine.collective_rpc(kv_peer, args=(blocks, source_rank, integrity))[0]

    engine.placement_kv_file = transfer
    engine.placement_kv_peer = peer_transfer
    engine.placement_runner_state = runner_state
    engine.placement_probe = probe
    engine.placement_remove = lambda transaction, request_id: release(transaction, request_id)
    engine.placement_rollback = lambda transaction, request_id: release(transaction, request_id, rollback=True)


def install_boundary_pause(engine):
    from vllm.v1.core.sched.interface import PauseState

    original_sync = engine._has_global_unfinished_reqs
    engine.placement_boundary_pause_pending = False
    engine.placement_boundary_pause_receipt = None

    def pause_boundary():
        if engine.pending_pause or engine.placement_boundary_pause_pending:
            raise RuntimeError("Another DP pause is already pending")
        if engine.scheduler.pause_state == PauseState.PAUSED_ALL:
            raise RuntimeError("Boundary pause requires an unpaused scheduler")
        receipt = {"rank": engine.dp_rank, "armed_at": time.monotonic()}
        future = Future()

        def complete(current):
            if current.placement_boundary_pause_pending or "reached_at" not in receipt:
                future.set_exception(RuntimeError("Boundary pause callback preceded DP consensus"))
                return
            receipt["callback_at"] = time.monotonic()
            future.set_result(dict(receipt))

        engine.placement_boundary_pause_receipt = receipt
        engine.placement_boundary_pause_pending = True
        engine.pending_pause = True
        engine.engines_running = True
        engine._idle_state_callbacks.append(complete)
        return future

    def sync(local_unfinished):
        unfinished = original_sync(local_unfinished)
        if (engine.placement_boundary_pause_pending and not engine.pending_pause
                and engine.ignore_start_dp_wave):
            if engine.step_counter % 32 or engine.batch_queue:
                raise RuntimeError("Boundary pause requires default32 consensus with no worker batch in flight")
            engine.scheduler.set_pause_state(PauseState.PAUSED_ALL)
            engine.placement_boundary_pause_pending = False
            engine.placement_boundary_pause_receipt.update(
                reached_at=time.monotonic(), boundary_step_counter=engine.step_counter)
            return False
        return unfinished

    engine.placement_pause_boundary = pause_boundary
    engine._has_global_unfinished_reqs = sync
