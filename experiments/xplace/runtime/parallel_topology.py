import json


def worker_filename(service, dp_rank, tp_rank, tp_size):
    suffix = f"-tp{tp_rank}" if tp_size > 1 else ""
    return f"worker-{service}-dp{dp_rank}{suffix}.json"


def verify_workers(output, num_ranks, tp_size=1):
    workers = [[json.loads((output / worker_filename(f"instance-{model + 1}", rank, lane, tp_size)).read_text())
                for rank in range(num_ranks) for lane in range(tp_size)] for model in (0, 1)]
    physical_gpus = num_ranks * tp_size
    namespaces = []
    for model_index, model in enumerate(workers):
        if len({worker["gpu_uuid"] for worker in model}) != physical_gpus:
            raise RuntimeError("Parallel workers do not occupy distinct physical GPUs")
        if {worker["groups"]["ep"]["rank"] for worker in model} != set(range(physical_gpus)):
            raise RuntimeError("Missing or duplicated EP rank")
        for index, worker in enumerate(model):
            rank, lane = divmod(index, tp_size)
            sizes = {name: worker["groups"][name]["world_size"] for name in ("tp", "dp", "ep")}
            if sizes != {"tp": tp_size, "dp": num_ranks, "ep": physical_gpus}:
                raise RuntimeError(f"Unexpected parallel topology {sizes}")
            if (worker["dp_rank"] != rank or worker["groups"]["dp"]["rank"] != rank
                    or worker["groups"]["tp"]["rank"] != lane
                    or worker["service"] != f"instance-{model_index + 1}"):
                raise RuntimeError("Worker identity differs from its registered DP/TP slot")
            if not worker["kv_runtime"]["initialized"] or worker["kv_runtime"]["world_size"] != tp_size:
                raise RuntimeError("KV runtime does not match the TP group")
        layers = len(model[0]["expert_ids"])
        if not layers or any(len(worker["expert_ids"]) != layers for worker in model):
            raise RuntimeError("Inconsistent expert layer count")
        for layer in range(layers):
            shards = [worker["expert_ids"][layer] for worker in model]
            if sum(map(len, shards)) != 128 or set().union(*map(set, shards)) != set(range(128)):
                raise RuntimeError("Experts are not complementary EP shards")
        for rank in range(num_ranks):
            group = model[rank * tp_size:(rank + 1) * tp_size]
            if len({worker["ipc_name"] for worker in group}) != 1:
                raise RuntimeError("TP workers must share their DP-local KV namespace")
            if len({(worker["kv_num_blocks"], worker["kv_block_size"]) for worker in group}) != 1:
                raise RuntimeError("TP workers disagree on KV capacity")
            namespaces.append(group[0]["ipc_name"])
    if any(first["gpu_uuid"] != second["gpu_uuid"] for first, second in zip(*workers)):
        raise RuntimeError("Tenant DP/TP workers are not colocated on the same GPUs")
    if len(set(namespaces)) != 2 * num_ranks:
        raise RuntimeError("Tenant/DP KV namespaces are not independent")
    return {"num_ranks": num_ranks, "tensor_parallel_size": tp_size, "expert_parallel_size": physical_gpus,
            "physical_gpus": physical_gpus, "gpu_uuids": [worker["gpu_uuid"] for worker in workers[0]],
            "parameter_gib": [[worker["parameter_bytes"] / 1024**3 for worker in model] for model in workers],
            "kv_num_blocks": [[worker["kv_num_blocks"] for worker in model] for model in workers]}


def minimum_free_bytes(memory, rank, tp_size):
    if any(len(memory[model][rank]) != tp_size for model in (0, 1)):
        raise RuntimeError("Memory snapshot is missing a TP worker")
    for lane in range(tp_size):
        if memory[0][rank][lane]["gpu_uuid"] != memory[1][rank][lane]["gpu_uuid"]:
            raise RuntimeError("Tenant memory snapshots refer to different GPUs")
    return min(memory[model][rank][lane]["free_mib"]
               for model in (0, 1) for lane in range(tp_size)) * 1024**2
