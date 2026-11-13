import copy
import enum
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

import causal_deployment as deployment


def tree(root):
    return {str(filename.relative_to(root)): deployment.sha256(filename) for filename in root.rglob("*.py")}


@pytest.fixture
def sources(tmp_path, monkeypatch):
    roots = {}
    for name in ("runtime", "vllm", "kvcached", "cloudpickle"):
        root = tmp_path / name
        root.mkdir()
        (root / "__init__.py").write_text("VALUE = 1\n")
        roots[name] = root
    native = tmp_path / "extension.so"
    native.write_bytes(b"cpu_fixture")
    launcher = tmp_path / "common.sh"
    launcher.write_text("true\n")
    manifest = dict(schema=1, configuration=copy.deepcopy(deployment.CONFIGURATION),
                    packages={name: dict(root=str(root), python_sha256=tree(root))
                              for name, root in roots.items() if name != "runtime"},
                    runtime_sha256=tree(roots["runtime"]),
                    native_sha256={str(native): deployment.sha256(native)},
                    launcher_sha256={str(launcher): deployment.sha256(launcher)}, versions={"vllm": "cpu_fixture"})
    monkeypatch.setattr(deployment, "host_clock_id", lambda: "cpu-clock")
    return manifest, roots


def test_full_tree_and_native_binding(sources):
    manifest, roots = sources
    result = deployment.verify_sources(manifest, roots["runtime"], imported_roots=roots)
    assert result["python_files"] == 4 and result["native_checked"] and result["clock_id"] == "cpu-clock"


@pytest.mark.parametrize("package", ["runtime", "vllm", "kvcached", "cloudpickle"])
@pytest.mark.parametrize("change", ["modify", "add", "delete"])
def test_every_python_source_mutation_rejects(sources, package, change):
    manifest, roots = sources
    filename = roots[package] / ("extra.py" if change == "add" else "__init__.py")
    if change == "delete":
        filename.unlink()
    else:
        filename.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="Source tree differs"):
        deployment.verify_sources(manifest, roots["runtime"], imported_roots=roots)


@pytest.mark.parametrize("field", ["native_sha256", "launcher_sha256"])
def test_native_and_launcher_mutations_reject(sources, field):
    manifest, roots = sources
    Path(next(iter(manifest[field]))).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Native/launcher"):
        deployment.verify_sources(manifest, roots["runtime"], imported_roots=roots)


def test_imported_root_must_be_declared(sources):
    manifest, roots = sources
    with pytest.raises(ValueError, match="source root differs"):
        deployment.verify_sources(manifest, roots["runtime"], imported_roots=dict(roots, vllm=roots["cloudpickle"]))


def test_manifest_digest_and_schema_binding(tmp_path, monkeypatch, sources):
    manifest, roots = sources
    filename = tmp_path / "manifest.json"
    filename.write_text(json.dumps(manifest))
    monkeypatch.setenv("XPLACE_CAUSAL_MANIFEST", str(filename))
    monkeypatch.setenv("XPLACE_CAUSAL_MANIFEST_SHA256", deployment.sha256(filename))
    assert deployment.read_manifest()[0] == manifest
    filename.write_text(json.dumps(dict(manifest, schema=2)))
    with pytest.raises(ValueError, match="checksum"):
        deployment.read_manifest()
    monkeypatch.setenv("XPLACE_CAUSAL_MANIFEST_SHA256", deployment.sha256(filename))
    with pytest.raises(ValueError, match="Incomplete"):
        deployment.read_manifest()


def worker_config():
    worker = SimpleNamespace()
    for key, value in deployment.CONFIGURATION.items():
        target = worker
        parts = key.split(".")
        for component in parts[:-1]:
            if not hasattr(target, component):
                setattr(target, component, SimpleNamespace())
            target = getattr(target, component)
        setattr(target, parts[-1], copy.deepcopy(value))
    return worker


def test_actual_configuration_with_enum_conversion():
    worker = worker_config()
    worker.compilation_config.mode = enum.IntEnum("Mode", {"NONE": 0}).NONE
    worker.compilation_config.cudagraph_mode = enum.Enum("Graph", {"FULL_DECODE_ONLY": 7}).FULL_DECODE_ONLY
    assert deployment.capture_configuration(worker) == deployment.CONFIGURATION


@pytest.mark.parametrize("key", list(deployment.CONFIGURATION))
def test_each_changed_actual_configuration_field_rejects(key):
    worker = worker_config()
    target = worker
    parts = key.split(".")
    for component in parts[:-1]:
        target = getattr(target, component)
    setattr(target, parts[-1], "unsupported")
    with pytest.raises(ValueError, match="Unsupported deployed"):
        deployment.capture_configuration(worker)


@pytest.mark.parametrize("change", [None, "clock", "manifest", "config", "status", "pid", "missing"])
def test_all_eight_workers_must_attest(tmp_path, monkeypatch, change):
    monkeypatch.setattr(deployment, "host_clock_id", lambda: "cpu-clock")
    for model in (1, 2):
        for rank in (0, 1):
            for lane in (0, 1):
                item = dict(manifest_sha256="a" * 64, clock_id="cpu-clock", configuration=deployment.CONFIGURATION,
                            status="verified", pid=100 + 4 * model + 2 * rank + lane)
                if (model, rank, lane) == (2, 1, 1):
                    if change == "missing":
                        continue
                    field = {"clock": "clock_id", "manifest": "manifest_sha256", "config": "configuration",
                             "status": "status", "pid": "pid"}.get(change)
                    if field:
                        item[field] = 104 if change == "pid" else "wrong"
                (tmp_path / f"worker-instance-{model}-dp{rank}-tp{lane}.json").write_text(
                    json.dumps(dict(causal_deployment=item)))
    if change:
        with pytest.raises((ValueError, FileNotFoundError)):
            deployment.verify_worker_attestations(tmp_path, "a" * 64)
    else:
        assert len(deployment.verify_worker_attestations(tmp_path, "a" * 64)["workers"]) == 8
