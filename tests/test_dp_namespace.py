import sys
import types

import pytest

from kvcached import utils
from kvcached.dp_namespace import bind_data_parallel_namespace


@pytest.fixture
def namespace_modules(monkeypatch):
    monkeypatch.delenv("KVCACHED_DP_IPC_BASE", raising=False)
    monkeypatch.setenv("KVCACHED_IPC_NAME", "tenant")
    monkeypatch.setattr(utils, "DEFAULT_IPC_NAME", "tenant")
    manager = types.SimpleNamespace(DEFAULT_IPC_NAME="tenant")
    tracker = types.SimpleNamespace(DEFAULT_IPC_NAME="tenant")
    ipc = types.SimpleNamespace(SOCKET_DIR="original", _listeners={})
    interface = types.SimpleNamespace(_kvcached_initialized=False)
    for name, module in (
        ("kvcached.kv_cache_manager", manager),
        ("kvcached.mem_info_tracker", tracker),
        ("kvcached.tp_ipc_util", ipc),
        ("kvcached.integration.vllm.interfaces", interface),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return manager, tracker, ipc, interface


def test_rank_namespaces_are_distinct_and_idempotent(namespace_modules):
    manager, tracker, ipc, _ = namespace_modules
    config = types.SimpleNamespace(data_parallel_size=2, data_parallel_rank=0)
    bind_data_parallel_namespace(config)
    assert utils.DEFAULT_IPC_NAME == "tenant_dp0"
    assert manager.DEFAULT_IPC_NAME == tracker.DEFAULT_IPC_NAME == "tenant_dp0"
    assert ipc.SOCKET_DIR == utils.get_tp_socket_dir("tenant_dp0")
    bind_data_parallel_namespace(config)
    assert utils.DEFAULT_IPC_NAME == "tenant_dp0"
    config.data_parallel_rank = 1
    bind_data_parallel_namespace(config)
    assert utils.DEFAULT_IPC_NAME == "tenant_dp1"


def test_single_rank_keeps_existing_namespace(namespace_modules):
    bind_data_parallel_namespace(types.SimpleNamespace(data_parallel_size=1))
    assert utils.DEFAULT_IPC_NAME == "tenant"


@pytest.mark.parametrize("active_resource", ["allocator", "listener"])
def test_late_rebinding_fails(namespace_modules, active_resource):
    _, _, ipc, interface = namespace_modules
    if active_resource == "allocator":
        interface._kvcached_initialized = True
    else:
        ipc._listeners = {"socket": object()}
    config = types.SimpleNamespace(data_parallel_size=2, data_parallel_rank=1)
    with pytest.raises(utils.KVCachedConfigError, match="Cannot change DP namespace"):
        bind_data_parallel_namespace(config)
