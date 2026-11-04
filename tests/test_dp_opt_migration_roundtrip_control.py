import asyncio
import pytest

from migration_roundtrip_control import exact_state_receipt, locate_request, paused_state_control, unique_owner


def payload(blocks=None, state=b"unchanged"):
    return {"request_id": "request", "request": state, "blocks": [1, 2] if blocks is None else blocks}


def test_state_is_exact_but_physical_blocks_can_change():
    assert exact_state_receipt(payload(), payload([3, 4]))["serialized_request_bytewise_equal"]
    with pytest.raises(RuntimeError, match="serialized"):
        exact_state_receipt(payload(), payload(state=b"changed"))
    with pytest.raises(RuntimeError, match="reservation length"):
        exact_state_receipt(payload(), payload([3]))
    with pytest.raises(RuntimeError, match="unique"):
        exact_state_receipt(payload(), payload([3, 3]))


class Controller:
    kv_transport = "dp_echo"

    def __init__(self, corrupt=False):
        self.rank = 0
        self.hops = []
        self.resumed = False
        self.corrupt = corrupt

    def inventory(self):
        request = {"id": "request", "movable": True, "blocks": [1, 2]}
        ranks = [{"requests": [request] if rank == self.rank else [], "block_bytes": 1024,
                  "page_bytes": 1024} for rank in (0, 1)]
        return [ranks, [{"requests": []}, {"requests": []}]]

    async def pause_boundary(self, observed):
        return [[{"reached_at": 0}]]

    async def begin(self, transaction):
        return self.inventory()

    async def snapshots(self):
        return self.inventory()

    async def call(self, model, rank, method, transaction, identifier):
        assert rank == self.rank
        return payload(state=b"corrupt" if self.corrupt and self.hops else b"unchanged")

    async def migrate(self, transaction, request, budgets):
        self.hops.append(request["source"])
        self.rank = 1 - self.rank
        return {"bitwise_verified": True, "integrity": "device_echo_full_bytewise"}

    async def release_transfer_cache(self):
        pass

    async def check_capacity(self, budgets):
        pass

    async def end(self, transaction):
        pass

    async def resume(self):
        self.resumed = True


@pytest.mark.parametrize("mode,hops", [("boundary_noop", []), ("paused_roundtrip", [0, 1])])
def test_no_inference_between_hops_and_same_owner_on_resume(tmp_path, mode, hops):
    controller = Controller()
    result = asyncio.run(paused_state_control(controller, controller.inventory(), 0, 0, "request",
                                             [1, 1], mode, tmp_path))
    assert controller.hops == hops
    assert controller.rank == 0 and controller.resumed
    assert result["inference_between_roundtrip_hops"] is False


def test_corruption_keeps_engines_paused(tmp_path):
    controller = Controller(corrupt=True)
    with pytest.raises(RuntimeError, match="serialized"):
        asyncio.run(paused_state_control(controller, controller.inventory(), 0, 0, "request",
                                        [1, 1], "paused_roundtrip", tmp_path))
    assert controller.hops == [0] and not controller.resumed


def test_owner_rejects_duplicate_and_unrelated_requests():
    inventory = Controller().inventory()
    assert locate_request(inventory, 0, 0, "request")["id"] == "request"
    with pytest.raises(RuntimeError, match="uniquely"):
        locate_request(inventory, 0, 0, "other")
    inventory[0][1]["requests"] = inventory[0][0]["requests"]
    with pytest.raises(RuntimeError, match="unique KV owner"):
        unique_owner(inventory, 0, "request", 0)


def test_complete_original_lengths_in_all_fourteen_controls(tmp_path, monkeypatch):
    import live_trace
    import migration_roundtrip_control as control
    from types import SimpleNamespace

    calls = []

    async def fake_stream(engine, identifier, prompt, output_tokens, source, model, role, arrived):
        calls.append((identifier, list(prompt), output_tokens, source, model))
        return {"token_ids": [model + source] * output_tokens}

    async def empty(*args):
        pass

    async def mature(*args):
        return []

    monkeypatch.setattr(control, "stream", fake_stream)
    monkeypatch.setattr(control, "wait_empty", empty)
    monkeypatch.setattr(control, "capture_control", empty)
    monkeypatch.setattr(control, "wait_mature", mature)
    monkeypatch.setattr(control, "paused_state_control", empty)
    monkeypatch.setattr(live_trace, "request_prompt", lambda prepared, mark: prepared[mark["request_id"]])
    marks = [{"request_id": "code", "model": 0, "output_tokens": 64},
             {"request_id": "conv-0", "model": 1, "output_tokens": 2048},
             {"request_id": "conv-1", "model": 1, "output_tokens": 3072}]
    prompts = {"code": [99] * 100000, "conv-0": [10] * 74, "conv-1": [20] * 149}
    args = SimpleNamespace(output=tmp_path, kv_budget_mib=10240, request_timeout_s=120,
                           prepared_prompts=prompts)
    result = asyncio.run(control.run_state_controls([object(), object()], Controller(), args, {"requests": marks}))
    assert len(calls) == len(result["trials"]) == 14
    assert result["is_algorithm_benefit"] is False
    assert result["mixed_batch_migration_correctness_proven"] is False
    assert all(item["same_source_full_serial_tokens_equal"] for item in result["trials"])
    assert all(prompt == prompts["code"] and output == 64 for _, prompt, output, _, model in calls if model == 0)
    assert result["cross_rank_serial_comparisons"][0]["rank_serial_tokens_equal"] is False
