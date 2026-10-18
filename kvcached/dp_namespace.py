import os
import sys


def bind_data_parallel_namespace(parallel_config):
    if int(getattr(parallel_config, "data_parallel_size", 1)) <= 1:
        return

    from kvcached import utils

    rank = int(parallel_config.data_parallel_rank)
    base = os.environ.get("KVCACHED_DP_IPC_BASE", utils.DEFAULT_IPC_NAME)
    suffix = f"_dp{rank}"
    name = utils._sanitize_segment(base)[:64 - len(suffix)] + suffix
    if name == utils.DEFAULT_IPC_NAME:
        return

    interfaces = sys.modules.get("kvcached.integration.vllm.interfaces")
    if interfaces is not None and getattr(interfaces, "_kvcached_initialized", False):
        raise utils.KVCachedConfigError("Cannot change DP namespace after KV initialization")
    ipc_module = sys.modules.get("kvcached.tp_ipc_util")
    if ipc_module is not None and getattr(ipc_module, "_listeners", {}):
        raise utils.KVCachedConfigError("Cannot change DP namespace with live IPC listeners")

    os.environ["KVCACHED_DP_IPC_BASE"] = base
    os.environ["KVCACHED_IPC_NAME"] = name
    utils.DEFAULT_IPC_NAME = name
    for module_name in ("kvcached.kv_cache_manager", "kvcached.mem_info_tracker"):
        module = sys.modules.get(module_name)
        if module is not None:
            module.DEFAULT_IPC_NAME = name
    if ipc_module is not None:
        ipc_module.SOCKET_DIR = utils.get_tp_socket_dir(name)
