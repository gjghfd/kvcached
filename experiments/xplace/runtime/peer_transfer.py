import hashlib
import json
import math
import time


def transfer_geometry(views, blocks, num_blocks, source_rank, destination_rank, rank, world_size, chunk_mib):
    if (not isinstance(world_size, int) or world_size < 2 or source_rank == destination_rank
            or any(not isinstance(value, int) or not 0 <= value < world_size
                   for value in (source_rank, destination_rank, rank))):
        raise ValueError("Peer KV transfer requires distinct in-group source and destination ranks")
    if not isinstance(chunk_mib, int) or isinstance(chunk_mib, bool) or not 1 <= chunk_mib <= 64:
        raise ValueError("Peer KV transfer chunk must be 1..64 MiB")
    if not isinstance(num_blocks, int) or isinstance(num_blocks, bool) or num_blocks <= 0 or not views:
        raise ValueError("Peer KV transfer requires a positive block count and backing views")
    if rank in (source_rank, destination_rank):
        if (len(blocks) != num_blocks or len(set(blocks)) != len(blocks)
                or any(not isinstance(block, int) or isinstance(block, bool) for block in blocks)):
            raise ValueError("Transfer endpoints require unique integer blocks matching the declared count")
        if any(block <= 0 or block >= view.shape[0] for block in blocks for view in views):
            raise ValueError("Peer KV transfer block outside cache")
    elif blocks:
        raise ValueError("Bystander ranks must not supply blocks")
    layout = [{"shape": list(view.shape[1:]), "dtype": str(view.dtype),
               "element_bytes": view.element_size()} for view in views]
    block_bytes = [math.prod(item["shape"]) * item["element_bytes"] for item in layout]
    limit = chunk_mib * 1024**2
    if min(block_bytes) <= 0 or max(block_bytes) > limit:
        raise ValueError("Peer KV backing block exceeds bounded staging chunk")
    fingerprint = int(hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()[:15], 16)
    chunk_blocks = max(1, limit // max(block_bytes))
    return {"metadata": [len(views), num_blocks, chunk_blocks, source_rank, destination_rank, fingerprint],
            "chunk_blocks": chunk_blocks, "bytes": num_blocks * sum(block_bytes)}


def transfer_blocks(worker, views, blocks, source_rank, destination_rank, num_blocks, chunk_mib):
    """Copy KV blocks source->destination over the DP group; every DP rank must call this collectively."""
    import torch
    from vllm.distributed.parallel_state import get_dp_group, get_tp_group

    group = get_dp_group()
    rank, world = group.rank_in_group, group.world_size
    if not callable(getattr(group, "all_gatherv", None)):
        raise RuntimeError("Peer transfer needs the DP group all_gatherv")
    blocks = list(blocks or [])
    geometry = transfer_geometry(views, blocks, num_blocks, source_rank, destination_rank, rank, world, chunk_mib)
    if any(view.device != worker.device for view in views):
        raise ValueError("Peer KV views must belong to the local worker device")
    if any(view.dtype != torch.uint8 for view in views):
        raise ValueError("Exact bytewise echo requires uint8 KV backing views")
    role = "source" if rank == source_rank else "destination" if rank == destination_rank else "bystander"
    started = time.monotonic()
    stage_started, stage_s = started, {}

    def mark(name):
        nonlocal stage_started
        now = time.monotonic()
        stage_s[name] = stage_s.get(name, 0) + now - stage_started
        stage_started = now

    with torch.inference_mode():
        torch.cuda.synchronize(worker.device)
        metadata = torch.tensor(geometry["metadata"], dtype=torch.int64, device=worker.device)
        gathered = group.all_gatherv([metadata], dim=0, sizes=[len(metadata)] * world)
        if len(gathered) != 1 or not all(torch.equal(gathered[0][index * len(metadata):(index + 1) * len(metadata)],
                                                     metadata) for index in range(world)):
            raise RuntimeError("Peer KV metadata differs across DP ranks")
        mark("metadata_collective")
        bitwise_valid = torch.ones(1, dtype=torch.uint8, device=worker.device)
        for view in views:
            for begin in range(0, num_blocks, geometry["chunk_blocks"]):
                count = min(geometry["chunk_blocks"], num_blocks - begin)
                indices = (torch.tensor(blocks[begin:begin + count], dtype=torch.long, device=worker.device)
                           if role != "bystander" else None)
                local = (view.index_select(0, indices) if role == "source" else
                         torch.empty((0, *view.shape[1:]), dtype=view.dtype, device=worker.device))
                result = group.all_gatherv([local], dim=0, sizes=[count if peer == source_rank else 0
                                                                  for peer in range(world)])
                if len(result) != 1 or list(result[0].shape) != [count, *view.shape[1:]]:
                    raise RuntimeError("Peer collective returned unexpected KV layout")
                if role == "destination":
                    view.index_copy_(0, indices, result[0])
                torch.cuda.synchronize(worker.device)
                mark("copy_collective")
                echoed_local = (view.index_select(0, indices) if role == "destination" else
                                torch.empty((0, *view.shape[1:]), dtype=view.dtype, device=worker.device))
                echoed = group.all_gatherv([echoed_local], dim=0, sizes=[count if peer == destination_rank else 0
                                                                         for peer in range(world)])
                if len(echoed) != 1 or list(echoed[0].shape) != [count, *view.shape[1:]]:
                    raise RuntimeError("Peer echo returned unexpected KV layout")
                if role == "source":
                    bitwise_valid &= torch.eq(local, echoed[0]).all().to(torch.uint8)
                torch.cuda.synchronize(worker.device)
                mark("echo_collective")
                del indices, local, result, echoed_local, echoed
        acknowledged = group.all_gatherv([bitwise_valid], dim=0, sizes=[1] * world)
        if not bool(acknowledged[0].all().item()):
            raise RuntimeError("Peer destination KV differs bytewise from source")
        mark("acknowledgment")
    return {"role": role, "rank": rank, "tp_rank": get_tp_group().rank_in_group,
            "tp_size": get_tp_group().world_size, "bytes": geometry["bytes"], "num_blocks": num_blocks,
            "stage_s": stage_s, "elapsed_s": time.monotonic() - started, "chunk_mib": chunk_mib,
            "bitwise_verified": True, "integrity": "device_echo_full_bytewise",
            "transport": "dp_allgatherv_source_broadcast_destination_echo"}
