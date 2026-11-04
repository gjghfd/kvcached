"""In-process fake of the vLLM AsyncLLM + placement RPC surface used by the xplace runtime (tests only)."""
import asyncio
import math
import sys
import time
import types

PAGE_BLOCKS, BLOCK_TOKENS = 64, 16
BLOCK_BYTES, PAGE_BYTES = int(1.5 * 1024**2), 96 * 1024**2


def install_fake_vllm():
    class SamplingParams:
        def __init__(self, temperature=0, min_tokens=0, max_tokens=16, ignore_eos=False):
            self.temperature, self.min_tokens, self.max_tokens, self.ignore_eos = (temperature, min_tokens,
                                                                                    max_tokens, ignore_eos)

    module = sys.modules.get("vllm")
    if module is None:
        module = sys.modules["vllm"] = types.ModuleType("vllm")
    if not hasattr(module, "SamplingParams"):
        module.SamplingParams = SamplingParams


class FakeRequest:
    def __init__(self, request_id, prompt, max_tokens, rank, chunk_tokens=None):
        self.id, self.prompt, self.max_tokens, self.rank = request_id, len(prompt), max_tokens, rank
        self.generated, self.prefilled, self.reserved_at = 0, False, time.monotonic()
        # chunked prefill: one step per chunk before the first token (only when chunk_tokens is set)
        self.prefill_left = math.ceil(self.prompt / chunk_tokens) - 1 if chunk_tokens else 0
        self.blocks = list(range(1, math.ceil((self.prompt + max_tokens) / BLOCK_TOKENS) + 1))
        self.done = asyncio.Event()
        self.queue = asyncio.Queue()


class FakeTenant:
    """All DP ranks of one tenant; advances every request by one token per step unless paused."""

    def __init__(self, index, num_ranks, step_s=0.002, chunk_tokens=None):
        self.index, self.num_ranks, self.step_s = index, num_ranks, step_s
        self.chunk_tokens = chunk_tokens
        self.requests = {}
        self.paused = False
        self.transaction = None
        self.pending = {}
        self.calls = []
        self.engine_core = FakeEngineCore(self)
        self.loop_task = None
        self.peer = None
        self.overlap_steps = 0
        self.paused_ranks = set()

    def ensure_loop(self):
        if self.loop_task is None or self.loop_task.done():
            self.loop_task = asyncio.get_running_loop().create_task(self.run())

    async def run(self):
        while self.requests:
            await asyncio.sleep(self.step_s)
            if self.paused:
                continue
            if self.peer is not None and self.peer.requests and not self.peer.paused:
                self.overlap_steps += 1
            for request in list(self.requests.values()):
                if request.prefill_left > 0:
                    request.prefill_left -= 1
                    continue
                if not request.prefilled:
                    request.prefilled = True
                request.generated += 1
                request.queue.put_nowait(request.generated)
                if request.generated >= request.max_tokens:
                    del self.requests[request.id]
                    request.done.set()

    async def generate(self, prompt, sampling_params, request_id, data_parallel_rank=None):
        request = FakeRequest(request_id, prompt["prompt_token_ids"], sampling_params.max_tokens, data_parallel_rank,
                              self.chunk_tokens)
        self.requests[request_id] = request
        self.ensure_loop()
        while True:
            count = await request.queue.get()
            yield types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=list(range(count)))])
            if count >= request.max_tokens:
                return

    def shutdown(self):
        if self.loop_task is not None:
            self.loop_task.cancel()

    # ----- per-rank placement RPCs -----
    def rank_requests(self, rank):
        return [request for request in self.requests.values() if request.rank == rank]

    def pages_of(self, request):
        return math.ceil(len(request.blocks) / PAGE_BLOCKS)

    def geometry(self, rank, requests):
        return {"rank": rank, "observed_at": time.monotonic(),
                "mapped_bytes": (1 + sum(self.pages_of(request) for request in requests)) * PAGE_BYTES,
                "page_bytes": PAGE_BYTES, "block_bytes": BLOCK_BYTES, "page_block_capacity": PAGE_BLOCKS,
                "uniform_page_geometry": True, "block_size": BLOCK_TOKENS, "mapped_free_block_slots": 0,
                "free_blocks": 10**6}

    def placement_summary(self, rank):
        requests = self.rank_requests(rank)
        return {**self.geometry(rank, requests), "requests": [
            {"id": request.id, "computed_tokens": request.prompt + request.generated,
             "prompt_tokens": request.prompt, "output_tokens": request.generated, "max_tokens": request.max_tokens,
             "decoding": request.generated > 0, "reserved_at": request.reserved_at, "reservation_pending": False}
            for request in requests]}

    def placement_snapshot(self, rank, refresh=False):
        requests = self.rank_requests(rank)
        pages, page_id = [{"id": 0, "owners": [], "pinned": True, "memory_gib": PAGE_BYTES / 1024**3}], 1
        for request in requests:
            for _ in range(self.pages_of(request)):
                pages.append({"id": page_id, "owners": [request.id], "pinned": False,
                              "memory_gib": PAGE_BYTES / 1024**3})
                page_id += 1
        return {**self.geometry(rank, requests), "pages": pages, "requests": [
            {"id": request.id, "computed_tokens": request.prompt + request.generated,
             "prompt_tokens": request.prompt, "output_tokens": request.generated, "max_tokens": request.max_tokens,
             "blocks": list(request.blocks), "movable": request.generated > 0, "reserved_at": request.reserved_at,
             "reservation_pending": False} for request in requests]}

    def placement_pause_boundary(self, rank):
        if rank in self.paused_ranks:
            raise RuntimeError("Boundary pause requires an unpaused scheduler")
        self.paused_ranks.add(rank)
        armed = time.monotonic()
        self.paused = True
        return {"rank": rank, "armed_at": armed, "reached_at": time.monotonic(), "callback_at": time.monotonic()}

    def placement_begin(self, rank, transaction):
        if not self.paused:
            raise RuntimeError("begin requires pause")
        if any(request.generated == 0 for request in self.rank_requests(rank)):
            raise ValueError("Controlled experiment requires an all-decode movable population")
        self.transaction = transaction
        return self.placement_snapshot(rank)

    def require(self, transaction):
        if self.transaction != transaction or not self.paused:
            raise RuntimeError("transaction mismatch")

    def placement_end(self, rank, transaction):
        self.require(transaction)
        if self.pending:
            raise RuntimeError("Uncommitted destination KV remains")
        return True

    def placement_export(self, rank, transaction, request_id):
        self.require(transaction)
        request = self.requests[request_id]
        assert request.rank == rank and request.generated > 0
        return {"request_id": request_id, "request": request_id, "blocks": list(request.blocks)}

    def placement_prepare(self, rank, transaction, payload):
        self.require(transaction)
        self.pending[payload["request_id"]] = rank
        blocks = [1000 + index for index in range(len(payload["blocks"]))]
        return {"request_id": payload["request_id"], "request": payload["request_id"], "blocks": [blocks]}

    def placement_kv_peer(self, rank, transaction, blocks, source, destination, num_blocks):
        self.require(transaction)
        self.calls.append(("kv_peer", rank, source, destination, num_blocks))
        role = "source" if rank == source else "destination" if rank == destination else "bystander"
        if role != "bystander" and len(blocks) != num_blocks:
            raise ValueError("endpoint block count mismatch")
        if role == "bystander" and blocks:
            raise ValueError("bystander blocks")
        return {"role": role, "rank": rank, "bytes": num_blocks * BLOCK_BYTES, "num_blocks": num_blocks,
                "stage_s": {}, "elapsed_s": 0.0, "bitwise_verified": True}

    def placement_runner_state(self, rank, transaction, state):
        self.require(transaction)
        return True

    def placement_commit(self, rank, transaction, request_id):
        self.require(transaction)
        assert self.pending.pop(request_id) == rank
        self.requests[request_id].rank = rank
        return True

    def placement_remove(self, rank, transaction, request_id):
        self.require(transaction)
        return self.geometry(rank, self.rank_requests(rank))

    def placement_rollback(self, rank, transaction, request_id):
        self.pending.pop(request_id, None)
        return True

    def placement_retry_reservation(self, rank, request_id):
        return {"reservation_pending": False, "reserved_at": time.monotonic()}

    def collective_rpc(self, rank, method, timeout, args, kwargs):
        assert method == "placement_memory_snapshot"
        return [{"free_mib": 40 * 1024, "total_mib": 81920}]


class FakeEngineCore:
    def __init__(self, tenant):
        self.tenant = tenant
        self.core_engines = list(range(tenant.num_ranks))
        self.reqs_in_flight = {}
        self.rpc_counts = {}

    async def _call_utility_async(self, method, *args, engine):
        self.rpc_counts[method] = self.rpc_counts.get(method, 0) + 1
        await asyncio.sleep(0)
        return getattr(self.tenant, method)(engine, *args)

    async def resume_scheduler_async(self):
        self.tenant.paused = False
        self.tenant.paused_ranks.clear()
        self.tenant.transaction = None

    async def pause_scheduler_async(self, mode="keep", clear_cache=False):
        self.tenant.paused = True
