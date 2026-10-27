import hashlib
import json
import math
import time


def transfer_geometry(views, blocks, source_rank, rank, chunk_mib):
    if source_rank not in (0, 1) or rank not in (0, 1):
        raise ValueError("Peer KV transfer requires two DP ranks")
    if not isinstance(chunk_mib, int) or isinstance(chunk_mib, bool) or not 1 <= chunk_mib <= 64:
        raise ValueError("Peer KV transfer chunk must be 1..64 MiB")
    if (not views or not blocks or len(set(blocks)) != len(blocks)
            or any(not isinstance(block, int) or isinstance(block, bool) for block in blocks)):
        raise ValueError("Peer KV transfer requires unique integer blocks and backing views")
    if any(block <= 0 or block >= view.shape[0] for block in blocks for view in views):
        raise ValueError("Peer KV transfer block outside cache")
    layout = [{"shape": list(view.shape[1:]), "dtype": str(view.dtype),
               "element_bytes": view.element_size()} for view in views]
    block_bytes = [math.prod(item["shape"]) * item["element_bytes"] for item in layout]
    limit = chunk_mib * 1024**2
    if min(block_bytes) <= 0 or max(block_bytes) > limit:
        raise ValueError("Peer KV backing block exceeds bounded staging chunk")
    fingerprint = int(hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()[:15], 16)
    chunk_blocks = max(1, limit // max(block_bytes))
    return {"metadata": [len(views), len(blocks), chunk_blocks, source_rank, fingerprint],
            "chunk_blocks": chunk_blocks, "bytes": len(blocks) * sum(block_bytes)}


def transfer_blocks(worker, views, blocks, source_rank, chunk_mib, integrity="sha256"):
    import torch
    from vllm.distributed.parallel_state import get_dp_group

    group = get_dp_group()
    rank = group.rank_in_group
    if group.world_size != 2 or not callable(getattr(group, "all_gatherv", None)):
        raise RuntimeError("Peer transfer needs the existing two-rank DP all_gatherv")
    geometry = transfer_geometry(views, blocks, source_rank, rank, chunk_mib)
    if integrity not in ("sha256", "device_echo"):
        raise ValueError("Unknown peer transfer integrity mode")
    if any(view.device != worker.device for view in views):
        raise ValueError("Peer KV views must belong to the local worker device")
    if integrity == "device_echo" and any(view.dtype != torch.uint8 for view in views):
        raise ValueError("Exact bytewise echo requires uint8 KV backing views")
    started = time.monotonic()
    stage_started, stage_s = started, {}

    def mark(name):
        nonlocal stage_started
        now = time.monotonic()
        stage_s[name] = stage_s.get(name, 0) + now - stage_started
        stage_started = now

    digest = hashlib.sha256()
    with torch.inference_mode():
        torch.cuda.synchronize(worker.device)
        metadata = torch.tensor(geometry["metadata"] + [int(integrity == "device_echo")],
                                dtype=torch.int64, device=worker.device)
        gathered = group.all_gatherv([metadata], dim=0, sizes=[len(metadata), len(metadata)])
        if len(gathered) != 1 or not torch.equal(gathered[0][:len(metadata)], gathered[0][len(metadata):]):
            raise RuntimeError("Peer KV metadata differs across workers")
        mark("metadata_collective")
        bitwise_valid = torch.ones(1, dtype=torch.uint8, device=worker.device)
        for view in views:
            for begin in range(0, len(blocks), geometry["chunk_blocks"]):
                selected = blocks[begin:begin + geometry["chunk_blocks"]]
                indices = torch.tensor(selected, dtype=torch.long, device=worker.device)
                local = (view.index_select(0, indices) if rank == source_rank else
                         torch.empty((0, *view.shape[1:]), dtype=view.dtype, device=worker.device))
                mark("gather_local_blocks")
                sizes = [len(selected) if peer == source_rank else 0 for peer in (0, 1)]
                result = group.all_gatherv([local], dim=0, sizes=sizes)
                if len(result) != 1 or list(result[0].shape) != [len(selected), *view.shape[1:]]:
                    raise RuntimeError("Peer collective returned unexpected KV layout")
                torch.cuda.synchronize(worker.device)
                mark("dp_device_allgatherv")
                if rank != source_rank:
                    view.index_copy_(0, indices, result[0])
                    torch.cuda.synchronize(worker.device)
                mark("scatter_destination")
                if integrity == "device_echo":
                    echoed_local = (view.index_select(0, indices) if rank != source_rank else
                                    torch.empty((0, *view.shape[1:]), dtype=view.dtype, device=worker.device))
                    echo_sizes = [len(selected) if peer != source_rank else 0 for peer in (0, 1)]
                    echoed = group.all_gatherv([echoed_local], dim=0, sizes=echo_sizes)
                    if len(echoed) != 1 or list(echoed[0].shape) != [len(selected), *view.shape[1:]]:
                        raise RuntimeError("Peer echo returned unexpected KV layout")
                    if rank == source_rank:
                        bitwise_valid &= torch.eq(local, echoed[0]).all().to(torch.uint8)
                    torch.cuda.synchronize(worker.device)
                    mark("integrity_device_echo_bytewise")
                    del echoed_local, echoed
                else:
                    checked = local if rank == source_rank else view.index_select(0, indices)
                    host = checked.cpu().contiguous().numpy()
                    digest.update(memoryview(host).cast("B"))
                    mark("integrity_readback_sha256")
                    del checked, host
                del indices, local, result
        if integrity == "device_echo":
            acknowledged = group.all_gatherv([bitwise_valid], dim=0, sizes=[1, 1])
            if not bool(acknowledged[0].all().item()):
                raise RuntimeError("Peer destination KV differs bytewise from source")
            mark("integrity_device_acknowledgment")
            digest.update(json.dumps({"layout": geometry["metadata"], "bytes": geometry["bytes"],
                                      "integrity": "device_echo_full_bytewise"}, sort_keys=True).encode())
    return {"sha256": digest.hexdigest(), "bytes": geometry["bytes"], "stage_s": stage_s,
            "elapsed_s": time.monotonic() - started, "chunk_mib": chunk_mib,
            "integrity": "device_echo_full_bytewise" if integrity == "device_echo" else "full_content_sha256",
            "sha256_scope": "verified_integrity_receipt_not_KV_content" if integrity == "device_echo" else "KV_content",
            "bitwise_verified": integrity == "device_echo",
            "transport": "existing_dp_allgatherv_full_device_bytewise_echo" if integrity == "device_echo"
                         else "existing_dp_allgatherv_device_payload_host_integrity_readback"}
