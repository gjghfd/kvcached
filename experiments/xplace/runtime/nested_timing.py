import hashlib
import inspect
import sys
from collections import Counter
from pathlib import Path


SCOPES = ("runner_sample", "runner_bookkeeping", "runner_to_list", "dp_coordinate")
PARENTS = {"worker_execute": (), "worker_sample": (), "worker_dummy": (),
           "runner_sample": ("worker_sample",), "runner_bookkeeping": ("worker_sample",),
           "runner_to_list": ("runner_bookkeeping",), "dp_coordinate": ("worker_execute", "worker_dummy")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def parallel_groups():
    from vllm.distributed.parallel_state import get_dp_group, get_ep_group, get_tp_group

    return {name: {"name": group.unique_name, "ranks": list(group.ranks),
                   "rank": group.rank_in_group, "world_size": group.world_size}
            for name, group in (("dp", get_dp_group()), ("tp", get_tp_group()), ("ep", get_ep_group()))}


def validate_configuration(config, identity):
    expected = {"async_scheduling": False, "enable_dbo": False, "enable_eplb": False,
                "speculative_decoding": False, "pipeline_parallel_size": 1,
                "tensor_parallel_size": 2, "data_parallel_size": 2,
                "disable_nccl_for_dp_synchronization": False,
                "use_sequence_parallel_moe": True, "all2all_backend": "allgather_reducescatter"}
    for name, value in expected.items():
        require(type(config.get(name)) is type(value) and config[name] == value,
                "Unexpected nested timing configuration: " + name)
    rank, lane = identity["dp_rank"], identity["tp_rank"]
    require(type(rank) is int and rank in (0, 1) and type(lane) is int and lane in (0, 1),
            "Invalid nested timing lane")
    require(type(config["data_parallel_rank"]) is int and config["data_parallel_rank"] == rank,
            "Nested timing DP rank differs")
    expected_groups = {"dp": ([lane, lane + 2], rank), "tp": ([rank * 2, rank * 2 + 1], lane),
                       "ep": ([0, 1, 2, 3], rank * 2 + lane)}
    require(set(config["groups"]) == set(expected_groups), "Missing nested timing group")
    for name, (ranks, position) in expected_groups.items():
        group = config["groups"][name]
        require(group["ranks"] == ranks and all(type(item) is int for item in group["ranks"])
                and type(group["rank"]) is int and group["rank"] == position
                and type(group["world_size"]) is int and group["world_size"] == len(ranks)
                and isinstance(group["name"], str) and bool(group["name"]),
                "Unexpected nested timing group: " + name)


def worker_configuration(worker):
    runner = worker.model_runner
    parallel = runner.parallel_config
    names = ("enable_dbo", "enable_eplb", "pipeline_parallel_size", "tensor_parallel_size",
             "data_parallel_size", "data_parallel_rank", "disable_nccl_for_dp_synchronization",
             "use_sequence_parallel_moe", "all2all_backend")
    config = {name: getattr(parallel, name) for name in names}
    config.update(async_scheduling=runner.scheduler_config.async_scheduling,
                  speculative_decoding=runner.speculative_config is not None, groups=parallel_groups())
    return config


def install_worker_scopes(worker, hooks):
    runner = worker.model_runner
    config = worker_configuration(worker)
    validate_configuration(config, hooks.buffer.identity)
    dispatch = inspect.unwrap(type(runner)._determine_batch_execution_and_padding)
    owner = sys.modules.get(dispatch.__module__)
    require(owner is not None and vars(owner) is dispatch.__globals__
            and "coordinate_batch_across_dp" in dispatch.__code__.co_names,
            "Cannot locate actual runner DP lookup binding")
    require(callable(getattr(owner, "coordinate_batch_across_dp", None)), "Missing runner DP binding")
    hooks.wrap(runner, "_sample", "runner_sample")
    hooks.wrap(runner, "_bookkeeping_sync", "runner_bookkeeping")
    hooks.wrap(runner, "_to_list", "runner_to_list")
    hooks.wrap(owner, "coordinate_batch_across_dp", "dp_coordinate")
    return {"schema": 1, "scopes": list(SCOPES), "source_sha256": source_sha256(),
            "lookup_module": owner.__name__, "configuration": config, "restored": False,
            "semantics": "nested host scopes; output materialization includes existing event wait, not isolated copy or kernel time"}


def scope_intervals(record):
    stacks, spans = {}, {}
    for event in record["events"]:
        if event["kind"] in ("start", "stop"):
            continue
        label, separator, boundary = event["kind"].rpartition(":")
        require(separator and label in PARENTS and boundary in ("enter", "return"),
                "Unknown or failed nested scope")
        require(type(event["at_ns"]) is int, "Invalid nested scope clock")
        scope_id, parent, thread = event["scope_id"], event["parent_scope_id"], event["thread"]
        require(type(scope_id) is int and scope_id > 0 and type(thread) is int,
                "Invalid nested scope identity")
        require(parent is None or type(parent) is int, "Invalid nested parent")
        stack = stacks.setdefault(thread, [])
        if boundary == "enter":
            require(scope_id == event["sequence"] and scope_id not in spans, "Duplicate nested scope")
            require(parent == (stack[-1] if stack else None), "Nested scope parent or thread differs")
            if PARENTS[label]:
                require(parent is not None and spans[parent]["label"] in PARENTS[label], "Wrong nested scope parent")
            else:
                require(parent is None, "Outer worker scope is nested")
            spans[scope_id] = {"scope_id": scope_id, "label": label, "parent": parent, "thread": thread,
                               "start_ns": event["at_ns"], "start_event": event, "children": []}
            if parent is not None:
                spans[parent]["children"].append(scope_id)
            stack.append(scope_id)
        else:
            require(stack and stack[-1] == scope_id, "Missing, duplicate or overlapping nested scope return")
            span = spans[scope_id]
            require(span["label"] == label and span["parent"] == parent
                    and type(event["at_ns"]) is int and event["at_ns"] >= span["start_ns"],
                    "Nested scope return differs")
            span["end_ns"] = event["at_ns"]
            stack.pop()
    require(all(not stack for stack in stacks.values()), "Unfinished nested scope")
    for span in spans.values():
        require(type(span["start_ns"]) is int, "Invalid nested scope clock")
        for child_id in span["children"]:
            child = spans[child_id]
            require(span["start_ns"] <= child["start_ns"] <= child["end_ns"] <= span["end_ns"],
                    "Nested scope lies outside parent")
        observed = Counter(spans[child]["label"] for child in span["children"])
        label = span["label"]
        expected = Counter()
        if label == "worker_execute":
            tokens = span["start_event"]["scheduled_tokens"]
            require(type(tokens) is int and tokens >= 0, "Invalid scheduled-token metadata")
            if tokens:
                expected["dp_coordinate"] = 1
        elif label == "worker_dummy":
            expected["dp_coordinate"] = 1
        elif label == "worker_sample":
            present = span["start_event"]["runner_state_present"]
            require(type(present) is bool, "Missing runner state presence")
            if present:
                expected.update(runner_sample=1, runner_bookkeeping=1)
        elif label == "runner_bookkeeping":
            expected["runner_to_list"] = 1
        require(observed == expected, "Missing or repeated nested child scope: " + label)
        require([spans[child]["label"] for child in span["children"]] == list(expected.elements()),
                "Nested child scope order differs")
    return list(spans.values())


def validate_worker_scopes(record):
    metadata = record.get("nested_timing")
    require(isinstance(metadata, dict) and metadata.get("schema") == 1
            and metadata.get("scopes") == list(SCOPES) and metadata.get("restored") is True,
            "Missing or unrestored nested timing metadata")
    require(metadata["source_sha256"] == source_sha256(), "Nested timing source SHA differs")
    require(metadata["lookup_module"] == "vllm.v1.worker.gpu_model_runner", "Unexpected nested lookup module")
    validate_configuration(metadata["configuration"], record["identity"])
    spans = scope_intervals(record)
    counts = Counter(span["label"] for span in spans)
    return {"status": "passed", "service": record["identity"]["service"],
            "dp_rank": record["identity"]["dp_rank"], "tp_rank": record["identity"]["tp_rank"],
            "counts": {label: counts[label] for label in PARENTS}, "source_sha256": source_sha256()}


INTERFACE_SHA256 = {
    "v1/worker/gpu_model_runner.py": "81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0",
    "v1/worker/gpu_worker.py": "28ac3f85aca3f68f18ac51cb5915e4d7ebe316c4ff3c021f1c2c5d3b90daca56",
    "v1/engine/core.py": "3ae1381a6af841e21058c825702382dc66faae45c950ac5acb8495d2d3d05aad",
    "v1/executor/multiproc_executor.py": "fb279ff0e0a08681aa0078037d559351006b5453ecd1b1ea12277995925f3312",
    "v1/executor/abstract.py": "419002195d13cf050c3ea9705d18a12456c890c27c470a1efadce7de27ded0b1",
    "v1/core/sched/scheduler.py": "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941",
    "v1/sample/sampler.py": "315af950ef4c35fced53dc3a5df49a80af20b47e417e8f12bf315f535769bab2",
    "v1/sample/ops/topk_topp_sampler.py": "ad9406a08a9bfcc84f182dab4522920f73605d4999191ee8f0dbb1479d946506",
    "distributed/parallel_state.py": "ecfa5eeda697e9982591c93eca835fcc53463267c1c22110700819efcecee74f",
    "distributed/device_communicators/cuda_communicator.py": "e91777d5037b79cbfbc7930febd63b57baa5f27b15a7cabe33b3a2efc857ee40",
    "forward_context.py": "985adacc9e7ed2a7695819ac2e2ad9cb264a7d886806d3091ad75bfda7392242",
    "model_executor/models/qwen3_moe.py": "ea439daddb5e5c3c46f48f204e2f7a90612b349e33e63f2d5fc7e57916e5a415",
    "v1/worker/dp_utils.py": "2ba84bbf92a25e756576918bfb215c1fb387b006899885d811bdb2f774e843a9",
    "config/parallel.py": "a6581c267ab265e24905d2f5caa514482c28359f71380c6f894ceab25aa22541",
    "distributed/device_communicators/all2all.py": "d2482b105cced72508dcee18a6ea83c099b5b3d4cabe8527418fa7013dd4efb6",
    "distributed/device_communicators/pynccl.py": "28596817ee9fb842026624a8a6c32b0637630b6b908ddb2a79ecccbeac531e35",
    "distributed/device_communicators/base_device_communicator.py": "c4fafc71bbb3a7652ecdf425bf116e3a9ad203ba44baf005cbab5db88a8b8221",
    "model_executor/layers/fused_moe/__init__.py": "d43a21fd5636d10a5fffefeb0be882bac7710cd6159d2259d72c279ffd419cb3",
    "model_executor/layers/fused_moe/layer.py": "c888133b47507d6dc0a8e74671ef6e6b2c60c53e615e27098e354ad5b961d725",
    "config/vllm.py": "d4c31282c664c91bee81267a2134a9609d16c33e88cc40e0f020256346510a87",
    "utils/gpu_sync_debug.py": "3eddc8f7306a3a05ba4b2e60027fc86ddd9593fcf3624d56e58271d60a13c220",
    "model_executor/layers/fused_moe/runner/moe_runner.py": "3c000e17f726c9995a9a950db952415a4b2198d0778e3cedda96355421edfdae",
    "model_executor/layers/fused_moe/config.py": "f229a423db043aed9f0f7e09d586f82c5a05cbba5871784f0f46500059c745e6",
    "v1/worker/gpu_input_batch.py": "ca9b8a627cb8e5109c804970a09597c6aa019a78957df399e8b3022bd493ecfc"
}
