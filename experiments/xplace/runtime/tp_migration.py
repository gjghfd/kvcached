def ordered_shards(results, tp_size, dp_rank):
    if len(results) != tp_size or any(not isinstance(result, dict) for result in results):
        raise RuntimeError("Missing TP worker acknowledgments")
    if {result.get("tp_rank") for result in results} != set(range(tp_size)):
        raise RuntimeError("Missing or duplicated TP shard identity")
    if any(result.get("tp_size") != tp_size or result.get("rank") != dp_rank for result in results):
        raise RuntimeError("TP acknowledgment belongs to a different parallel group")
    return sorted(results, key=lambda result: result["tp_rank"])


def aggregate_peer(results, tp_size, dp_rank):
    shards = ordered_shards(results, tp_size, dp_rank)
    for key in ("role", "num_blocks", "integrity", "transport"):
        if len({shard[key] for shard in shards}) != 1:
            raise RuntimeError(f"TP shards disagree on {key}")
    if any(shard.get("bitwise_verified") is not True or shard["bytes"] <= 0 for shard in shards):
        raise RuntimeError("Every TP shard must have a positive, bytewise-verified copy")
    stages = {stage for shard in shards for stage in shard["stage_s"]}
    return {**shards[0], "bytes": sum(shard["bytes"] for shard in shards), "shards": shards,
            "stage_s": {stage: max(shard["stage_s"].get(stage, 0.0) for shard in shards) for stage in stages}}


def verify_runner_results(results, tp_size, dp_rank):
    shards = ordered_shards(results, tp_size, dp_rank)
    if any(shard.get("installed") is not True for shard in shards):
        raise RuntimeError("A TP worker did not install or remove the request state")
    return True


def verify_transfer_endpoints(source, destination, tp_size):
    if not (source.get("bitwise_verified") and destination.get("bitwise_verified")
            and source.get("role") == "source" and destination.get("role") == "destination"
            and source["bytes"] == destination["bytes"]):
        raise RuntimeError("Peer transfer lacks matching endpoint proof")
    if tp_size == 1 and "shards" not in source and "shards" not in destination:
        return
    first = ordered_shards(source.get("shards", []), tp_size, source["rank"])
    second = ordered_shards(destination.get("shards", []), tp_size, destination["rank"])
    for left, right in zip(first, second):
        if (left.get("bitwise_verified") is not True or right.get("bitwise_verified") is not True
                or left["bytes"] != right["bytes"] or left["num_blocks"] != right["num_blocks"]
                or left["role"] != "source" or right["role"] != "destination"):
            raise RuntimeError("Source and destination TP shards do not match")
