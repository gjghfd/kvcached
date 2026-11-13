import enum
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from live_observation import host_clock_id


CONFIGURATION = {
    "parallel_config.data_parallel_size": 2,
    "parallel_config.tensor_parallel_size": 2,
    "parallel_config.pipeline_parallel_size": 1,
    "parallel_config.enable_expert_parallel": True,
    "parallel_config.enable_dbo": False,
    "parallel_config.enable_eplb": False,
    "parallel_config.data_parallel_backend": "mp",
    "parallel_config.all2all_backend": "allgather_reducescatter",
    "scheduler_config.async_scheduling": False,
    "scheduler_config.enable_chunked_prefill": True,
    "scheduler_config.max_num_seqs": 256,
    "scheduler_config.max_num_batched_tokens": 4096,
    "cache_config.enable_prefix_caching": False,
    "cache_config.block_size": 16,
    "cache_config.kv_cache_memory_bytes": 48 * 1024 ** 3,
    "model_config.max_model_len": 131072,
    "model_config.dtype": "torch.bfloat16",
    "compilation_config.mode": 0,
    "compilation_config.cudagraph_mode": "FULL_DECODE_ONLY",
    "compilation_config.cudagraph_capture_sizes": [1, 8, 16, 32, 64, 128, 256],
    "speculative_config": None,
    "lora_config": None,
    "kv_transfer_config": None,
}


def sha256(filename):
    digest = hashlib.sha256()
    with Path(filename).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest():
    filename = Path(os.environ["XPLACE_CAUSAL_MANIFEST"])
    expected = os.environ["XPLACE_CAUSAL_MANIFEST_SHA256"]
    if len(expected) != 64 or sha256(filename) != expected:
        raise ValueError("Frozen causal deployment manifest checksum differs")
    manifest = json.loads(filename.read_text())
    if (manifest.get("schema") != 1 or manifest.get("configuration") != CONFIGURATION
            or set(manifest.get("packages", {})) != {"vllm", "kvcached", "cloudpickle"}):
        raise ValueError("Incomplete causal deployment source/configuration manifest")
    return manifest, expected


def check_tree(root, expected):
    root = Path(root).resolve()
    actual = {str(filename.relative_to(root)): sha256(filename) for filename in sorted(root.rglob("*.py"))}
    if not expected or actual != expected:
        changed = sorted(set(actual) ^ set(expected) | {key for key in actual.keys() & expected.keys()
                                                      if actual[key] != expected[key]})
        raise ValueError(f"Source tree differs: {root}: {changed[:12]}")
    return len(actual)


def verify_sources(manifest, runtime_root, *, imported_roots=None, binaries=True):
    started = time.monotonic()
    checked = check_tree(runtime_root, manifest["runtime_sha256"])
    for package, declaration in manifest["packages"].items():
        root = Path(declaration["root"]).resolve()
        actual_root = (Path(importlib.import_module(package).__file__).resolve().parent if imported_roots is None
                       else Path(imported_roots[package]).resolve())
        if actual_root != root:
            raise ValueError(f"Imported {package} source root differs")
        checked += check_tree(root, declaration["python_sha256"])
    if binaries:
        if not manifest["native_sha256"] or not manifest["launcher_sha256"]:
            raise ValueError("Native and launcher hashes are required")
        for filename, expected in {**manifest["native_sha256"], **manifest["launcher_sha256"]}.items():
            if sha256(filename) != expected:
                raise ValueError(f"Native/launcher source differs: {filename}")
    return dict(status="verified", python_files=checked, native_checked=binaries,
                elapsed_s=time.monotonic() - started, clock_id=host_clock_id())


def capture_configuration(worker):
    values = {}
    for key, expected in CONFIGURATION.items():
        value = worker
        for component in key.split("."):
            value = getattr(value, component)
        if key.endswith(".dtype"):
            value = str(value)
        elif isinstance(value, enum.IntEnum):
            value = int(value)
        elif isinstance(value, enum.Enum):
            value = value.name
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"Unsupported deployed configuration {key}: {value!r}")
        values[key] = value
    return values


def frontend_attestation():
    manifest, identity = read_manifest()
    versions = {name: importlib.metadata.version(name) for name in manifest["versions"]}
    if versions != manifest["versions"]:
        raise ValueError("Runtime package versions differ from deployment manifest")
    result = verify_sources(manifest, Path(__file__).parent)
    return dict(result, manifest_sha256=identity, versions=versions, pid=os.getpid())


def worker_attestation(worker):
    manifest, identity = read_manifest()
    result = verify_sources(manifest, Path(__file__).parent, binaries=False)
    return dict(result, manifest_sha256=identity, configuration=capture_configuration(worker),
                full_config_repr=repr(worker.vllm_config), pid=os.getpid(),
                native_note="Frontend checks native hashes before launch; successful runs also check after shutdown")


def verify_worker_attestations(output, expected_manifest):
    frontend_clock = host_clock_id()
    records = []
    for model in (1, 2):
        for rank in (0, 1):
            for lane in (0, 1):
                filename = Path(output) / f"worker-instance-{model}-dp{rank}-tp{lane}.json"
                item = json.loads(filename.read_text())["causal_deployment"]
                if (item["manifest_sha256"] != expected_manifest or item["clock_id"] != frontend_clock
                        or item["configuration"] != CONFIGURATION or item["status"] != "verified"):
                    raise ValueError("Worker configuration/source/clock attestation differs")
                records.append(dict(model=model - 1, dp_rank=rank, tp_rank=lane, pid=item["pid"]))
    if len({item["pid"] for item in records}) != 8:
        raise ValueError("Causal pilot needs eight distinct worker processes")
    return dict(status="verified", workers=records, clock_id=frontend_clock, device_progress="unknown")
