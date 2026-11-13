import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as element_tree

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "runtime"))
sys.path.insert(0, str(PROJECT / "tools"))

import analyze_scan_overhead as analysis
import graph_execution as graph
import imbalance_calibration as calibration
import scan_job

CLOCK = "linux-monotonic:archived-test-host:time:[123]"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def change(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    save(path, value)


def graph_pair(case, workers, started, completed):
    starts, stops = [], []
    for rank in (0, 1):
        initial, final = [], []
        for lane in (0, 1):
            identity = dict(model=case["model"], dp_rank=rank, tp_rank=lane, tp_size=2,
                            clock_id=CLOCK, graph_epoch="a" * 32, **workers[(case["model"], rank, lane)])
            descriptor = dict(mode="FULL", padded_tokens=case["capture_bucket"],
                              batch_requests=case["layout"][rank], uniform=True)
            events = [dict(sequence=1, at=started - 1, kind="start"),
                      dict(sequence=2, at=started, kind="selection_return", **descriptor,
                           unpadded_tokens=case["layout"][rank], microbatch=False, dp_padding_metadata_present=True),
                      dict(sequence=3, at=started, kind="replay_enter", call=1, **descriptor),
                      dict(sequence=4, at=completed, kind="replay_return", call=1),
                      dict(sequence=5, at=completed + 1, kind="stop")]
            common = dict(case=case["name"], identity=identity, event_limit=graph.EVENT_LIMIT,
                          byte_limit=graph.BYTE_LIMIT, source_sha256=graph.SOURCE_SHA256)
            initial.append(dict(common, kind="graph_start_v1", started_at=started - 1))
            final.append(dict(common, kind="graph_execution_v1", events=events, stopped=True,
                              attempted=5, retained=5, dropped=0, invalid=0,
                              bytes_retained=sum(len(json.dumps(event, separators=(",", ":")).encode()) for event in events),
                              gpu_completion_observed=False, added_cuda_sync=False))
        starts.append(initial)
        stops.append(final)
    common = dict(case=case["name"], status="received", errors=[], local_tasks_pending=[],
                  remote_rpc_may_continue=False, gpu_completion_observed=False)
    return dict(common, command="start", results=starts), dict(common, command="stop", results=stops)


def case_artifacts(root, case, mode, workers, started=100.0, scale=1.0):
    step = scale / 32
    first = started + scale / 2
    completed = first + step * (case["output_tokens"] - 1)
    rows = []
    for planned in analysis.expected_requests(case):
        row = {key: planned[key] for key in ("request_id", "model", "rank_initial", "prompt_tokens")}
        row.update(role="calibration", output_tokens=case["output_tokens"], arrived_at=started,
                   submitted_at=started, first_token_at=first, completed_at=completed, cadence_stride=32,
                   cadence=[[first + (count - 1) * step, count]
                            for count in [1, *range(32, case["output_tokens"] + 1, 32)]])
        rows.append(row)
    status = dict(case, status="measured", prepared_at=started - 1, started_at=started,
                  finished_at=completed + 1, step_s=step, preemptions=0,
                  window=dict(window_s=step * (case["output_tokens"] - 1) - 2,
                              samples=case["batch_size"], spread=[step, step]))
    if mode == "collect":
        status["graph_evidence"] = dict(valid=True)
        initial, final = graph_pair(case, workers, started, completed)
        save(root / f"{case['name']}-graph-start.json", initial)
        save(root / f"{case['name']}-graph-stop.json", final)
    values = dict(status=status, attempt=dict(case=case, requests=analysis.expected_requests(case), prepared_at=started - 1),
                  requests=rows, before=[[dict(preemptions_total=0)] * 2 for _ in range(2)],
                  after=[[dict(preemptions_total=0)] * 2 for _ in range(2)])
    for suffix, value in values.items():
        save(root / f"{case['name']}-{suffix}.json", value)
    return status


def fixture_bundle(root):
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    protocol = scan_job.protocol_core()
    protocol["configs"] = {}
    for block in (0, 1):
        for mode in ("off", "collect"):
            path = inputs / f"block{block}-{mode}.json"
            save(path, scan_job.overhead_config(block, mode))
            protocol["configs"][f"{block}-{mode}"] = dict(file=path.name, sha256=analysis.digest(path))
    save(inputs / "graph-overhead-protocol.json", protocol)
    save(inputs / "prompt-bank.json", {"fixture": True})
    for directory in ("runtime", "tests"):
        (root / directory).mkdir()
        (root / directory / "fixture.py").write_text("fixture = True\n")
    manifest = dict(runtime_sha256={"fixture.py": analysis.digest(root / "runtime/fixture.py")},
                    protocol_sha256=analysis.digest(inputs / "graph-overhead-protocol.json"),
                    prompt_bank_sha256=analysis.digest(inputs / "prompt-bank.json"),
                    configuration={"fixture": True}, versions={"fixture": "1"})
    save(inputs / "deployment-manifest.json", manifest)
    return manifest, protocol


def condition_artifacts(job, block, mode, index, manifest, protocol, manifest_sha):
    root = job / "out/scan-overhead-block" / mode
    root.mkdir(parents=True)
    inputs = job / "inputs"
    pid = 10000 + block * 1000 + index * 100
    command = [sys.executable, "/runtime/main.py", "--checkpoint", "fixed-model", "--prompt-bank", str(inputs / "prompt-bank.json"),
               "--phases", "dataplane,imbalance", "--policies", "no_move", "--num-ranks", "2", "--tensor-parallel-size", "2",
               "--enable-tp-migration", "--budget-margin-gib", "3", "--seed", "20261110", "--kv-mode", "prompt",
               "--reserve-extra-tokens", "256", "--max-inflight", "0", "0", "--drain-timeout-s", "210",
               "--output", str(root / "out"), "--imbalance-config", str(inputs / f"block{block}-{mode}.json")]
    process = dict(status="passed", returncode=0, timeout=False, forced_cleanup=False, pid=pid, command=command,
                   started_at=1000 + index * 500, finished_at=1400 + index * 500)
    save(root / "process.json", process)
    front = dict(status="verified", manifest_sha256=manifest_sha, native_checked=True, versions=manifest["versions"],
                 pid=pid, clock_id=CLOCK)
    for name in ("causal-frontend-attestation.json", "causal-final-attestation.json"):
        save(root / "out" / name, front)
    workers = {}
    for model in (0, 1):
        for rank in (0, 1):
            for lane in (0, 1):
                worker_pid = pid + 1 + model * 4 + rank * 2 + lane
                worker = dict(pid=worker_pid, gpu_uuid=f"GPU-{rank}-{lane}", gpu_name="NVIDIA A800 80GB",
                              service=f"instance-{model + 1}", dp_rank=rank, checkpoint="fixed-model",
                              parameter_bytes=1024, kv_num_blocks=1024, kv_block_size=16,
                              kv_runtime=dict(initialized=True, world_size=2), ipc_name=f"ipc-{model}-{rank}",
                              expert_ids=[list(range((rank * 2 + lane) * 32, (rank * 2 + lane + 1) * 32))],
                              groups=dict(dp=dict(rank=rank, world_size=2), tp=dict(rank=lane, world_size=2),
                                          ep=dict(rank=rank * 2 + lane, world_size=4)),
                              causal_deployment=dict(status="verified", manifest_sha256=manifest_sha,
                                                     configuration=manifest["configuration"], native_checked=False,
                                                     clock_id=CLOCK, pid=worker_pid))
                save(root / f"out/worker-instance-{model + 1}-dp{rank}-tp{lane}.json", worker)
                workers[(model, rank, lane)] = dict(pid=worker_pid, gpu_uuid=worker["gpu_uuid"])
    save(root / "out/causal-worker-attestations.json", dict(status="verified", clock_id=CLOCK,
         workers=[dict(model=key[0], dp_rank=key[1], tp_rank=key[2], pid=value["pid"]) for key, value in workers.items()]))
    save(root / "out/budget.json", dict(margin_gib=3.0, budget_gib=[32.0, 32.0], natural_budget_gib=[32.0, 32.0]))
    save(root / "out/calibration/dataplane.json", dict(status="passed"))
    save(root / "out/phases.json", dict(phases=[dict(phase="startup", status="passed", wall=process["started_at"] + 100),
                                              dict(phase="job", status="passed")]))
    config = scan_job.overhead_config(block, mode)
    cases = analysis.scan_cases(config, 2, 2)
    scan = root / "out/calibration/imbalance"
    save(scan / "plan.json", dict(config=config, cases=cases))
    save(scan / "summary.json", dict(status="passed", stop_reason="completed", failure=None, planned=4, completed=4,
                                     measured=4, remaining_cases=[], case_timeout_s=150, max_scan_s=900))
    results = [case_artifacts(scan, case, mode, workers, 100.0 + case_index * 40)
               for case_index, case in enumerate(cases)]
    save(scan / "results.json", results)
    save(scan / "case-statuses.json", results)
    save(scan / "warmup.json", case_artifacts(scan, analysis.warmup_case(cases), mode, workers, 50.0))
    usage = [dict(group=pid, at=process["started_at"] + offset, clock_ticks_per_s=100, page_bytes=4096,
                  read_failures=[], processes=[dict(pid=pid, start_ticks=1, cpu_ticks=offset + 1, rss_pages=100)])
             for offset in (0, 1, 2)]
    (root / "process-usage.jsonl").write_text("".join(json.dumps(row) + "\n" for row in usage))
    return dict(mode=mode, **process)


@pytest.fixture(scope="module")
def archive_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("scan-analysis-template")
    bundle = root / "bundle"
    manifest, protocol = fixture_bundle(bundle)
    manifest_sha = analysis.digest(bundle / "inputs/deployment-manifest.json")
    for block in (0, 1):
        job = root / f"job{block}"
        shutil.copytree(bundle, job)
        (job / "exit-status").write_text("0\n")
        suite = element_tree.Element("testsuite", tests="238", errors="0", failures="0", skipped="0")
        for index in range(238):
            element_tree.SubElement(suite, "testcase", classname="fixture", name=f"test_{index}")
        element_tree.ElementTree(suite).write(job / "startup-tests.xml")
        (job / "gpus.csv").write_text("".join(f"{rank * 2 + lane},GPU-{rank}-{lane},NVIDIA A800,driver,81920\n"
                                               for rank in (0, 1) for lane in (0, 1)))
        completed = [condition_artifacts(job, block, mode, index, manifest, protocol, manifest_sha)
                     for index, mode in enumerate(scan_job.BLOCKS[block])]
        save(job / "out/scan-overhead-block/status.json",
             dict(block=block, status="collected_not_overhead_accepted", not_executed=[],
                  planned=list(scan_job.BLOCKS[block]), started=list(scan_job.BLOCKS[block]), completed=completed,
                  protocol_sha256=manifest["protocol_sha256"], allocation_started_at=999,
                  inputs=dict(common=protocol["common_args"], checkpoint="fixed-model")))
        save(job / "out/scan-overhead-block/protocol.json", protocol)
    return root


@pytest.fixture
def archived(tmp_path, archive_template):
    root = tmp_path / "archive"
    shutil.copytree(archive_template, root)
    return root


def run_analysis(root, missing=False):
    jobs = {block: root / f"job{block}" for block in ((0,) if missing else (0, 1))}
    return analysis.analyze_jobs(jobs, root / "bundle", root / "report",
                                 analysis.digest(root / "bundle/inputs/deployment-manifest.json"),
                                 startup_ids=[("fixture", f"test_{index}") for index in range(238)])


def test_complete_artifacts_and_independent_constant_rate_oracle(archived, monkeypatch):
    monkeypatch.setattr(graph, "host_clock_id", lambda: pytest.fail("offline audit used analysis-host clock"))
    report = run_analysis(archived)
    assert report["overhead"]["passed"], report["blocks"]
    assert report["overhead"]["available_pairs"] == 8
    assert report["request_coverage"] == dict(planned=3072, valid=3072)
    assert len(report["warmups"]) == 4 and all(row["available"] for row in report["warmups"])
    for case in report["cases"]:
        assert case["metrics"] == dict(steady_step_s=1 / 32, mean_request_ttft_s=0.5, mean_request_tpot_s=1 / 32)
    assert not report["gpu_completion_observed"] and not report["prediction_gate_passed"]
    assert not report["system_gain_gate"] and not report["complete_93_point_scan_validated"]


def test_missing_block_keeps_every_planned_pair(archived, monkeypatch):
    monkeypatch.setattr(graph, "host_clock_id", lambda: CLOCK)
    report = run_analysis(archived, missing=True)
    assert not report["overhead"]["passed"]
    assert len(report["cases"]) == 16 and len(report["overhead"]["pairs"]) == 8
    assert all(not row["available"] for row in report["cases"] if row["block"] == 1)


@pytest.mark.parametrize("fault", ["request", "duplicate", "censored", "cleanup", "preempt", "nan", "overflow",
                                   "step", "cadence", "cleanup_results", "empty_incomplete", "graph_loss", "graph_clock", "graph_pid", "graph_lane",
                                   "graph_bytes", "graph_window", "graph_source", "graph_missing", "warmup"])
def test_bad_raw_case_cannot_pass(archived, monkeypatch, fault):
    monkeypatch.setattr(graph, "host_clock_id", lambda: CLOCK)
    root = archived / "job0/out/scan-overhead-block/collect/out/calibration/imbalance"
    case = analysis.scan_cases(scan_job.overhead_config(0, "collect"), 2, 2)[0]
    name = "warmup" if fault == "warmup" else case["name"]
    if fault in ("request", "warmup"):
        change(root / f"{name}-requests.json", lambda rows: rows.pop())
    elif fault == "duplicate":
        change(root / f"{name}-requests.json", lambda rows: rows.__setitem__(1, rows[0]))
    elif fault in ("censored", "step"):
        change(root / f"{name}-status.json", lambda row: row.update({"censored_at": 110} if fault == "censored" else {"step_s": 9}))
    elif fault == "cleanup":
        save(root / f"{name}-cleanup.json", dict(completed_late=True))
    elif fault in ("cleanup_results", "empty_incomplete"):
        suffix = "cleanup-results" if fault == "cleanup_results" else "incomplete"
        save(root / f"{name}-{suffix}.json", [])
    elif fault == "preempt":
        change(root / f"{name}-after.json", lambda rows: rows[case["model"]][0].update(preemptions_total=1))
    elif fault in ("nan", "overflow"):
        (root / f"{name}-requests.json").write_text('[{"request_id": ' + ("NaN" if fault == "nan" else "1e999") + '}]')
    elif fault == "cadence":
        change(root / f"{name}-requests.json", lambda rows: rows[0]["cadence"].reverse())
    elif fault == "graph_missing":
        (root / f"{name}-graph-stop.json").unlink()
    else:
        def mutate(record):
            row = record["results"][0][0]
            if fault == "graph_loss":
                row.update(dropped=1, attempted=6)
            elif fault == "graph_clock":
                row["identity"]["clock_id"] = "linux-monotonic:other:time:[2]"
            elif fault == "graph_pid":
                row["identity"]["pid"] += 1000
            elif fault == "graph_lane":
                record["results"][0].pop()
            elif fault == "graph_bytes":
                row["bytes_retained"] += 1
            elif fault == "graph_window":
                row["events"][-1]["at"] -= 3
            elif fault == "graph_source":
                row["source_sha256"] = {}
        change(root / f"{name}-graph-stop.json", mutate)
    report = run_analysis(archived)
    assert not report["overhead"]["passed"]
    assert len(report["cases"]) == 16 and len(report["overhead"]["pairs"]) == 8


@pytest.mark.parametrize("fault", ["source", "test_source", "startup", "order", "cleanup", "clock", "worker_pid",
                                   "gpu", "budget", "command", "warmup_summary", "usage", "protocol", "exit", "ep", "shard", "namespace"])
def test_bad_block_identity_cannot_pass(archived, monkeypatch, fault):
    monkeypatch.setattr(graph, "host_clock_id", lambda: CLOCK)
    job = archived / "job0"
    root = job / "out/scan-overhead-block/collect"
    if fault in ("source", "test_source"):
        (job / ("runtime" if fault == "source" else "tests") / "fixture.py").write_text("changed = True\n")
    elif fault == "startup":
        tree = element_tree.parse(job / "startup-tests.xml")
        tree.getroot()[0].set("name", "changed")
        tree.write(job / "startup-tests.xml")
    elif fault == "order":
        change(job / "out/scan-overhead-block/status.json", lambda row: row["started"].reverse())
    elif fault == "cleanup":
        change(root / "process.json", lambda row: row.update(forced_cleanup=True))
    elif fault == "clock":
        change(root / "out/causal-final-attestation.json", lambda row: row.update(clock_id="other"))
    elif fault in ("worker_pid", "gpu"):
        change(root / "out/worker-instance-1-dp0-tp0.json",
               lambda row: row.update(pid=888) if fault == "worker_pid" else row.update(gpu_uuid="other"))
    elif fault in ("ep", "shard", "namespace"):
        def mutate_worker(row):
            if fault == "ep":
                row["groups"]["ep"]["world_size"] = 2
            elif fault == "shard":
                row["expert_ids"][0].pop()
            else:
                row["ipc_name"] = "other"
        change(root / "out/worker-instance-1-dp0-tp0.json", mutate_worker)
    elif fault == "budget":
        change(root / "out/budget.json", lambda row: row.update(budget_gib=[31.0, 32.0]))
    elif fault == "command":
        change(root / "process.json", lambda row: row["command"].extend(["--max-inflight", "1", "1"]))
    elif fault == "warmup_summary":
        change(root / "out/calibration/imbalance/warmup.json", lambda row: row.update(status="failed"))
    elif fault == "usage":
        (root / "process-usage.jsonl").write_text('{"group": 1}\n')
    elif fault == "protocol":
        change(job / "inputs/graph-overhead-protocol.json", lambda row: row.update(cases_per_condition=3))
    elif fault == "exit":
        (job / "exit-status").write_text("1\n")
    report = run_analysis(archived)
    assert not report["overhead"]["passed"]
    assert report["blocks"]["0"]["status"] == "not_passed"


@pytest.mark.parametrize("losses,passed", [([0.05] * 8, True), ([0.05001] * 8, False),
                                          ([0.0] * 7 + [0.15], True), ([0.0] * 7 + [0.15001], False)])
def test_exact_thresholds_and_exceedances(losses, passed):
    rows = []
    for block in (0, 1):
        for index in range(4):
            for mode in ("off", "collect"):
                value = 1.0 if mode == "off" else 1.0 + losses[block * 4 + index]
                rows.append(dict(block=block, case=dict(name=f"case{index}"), mode=mode, available=True,
                                 request_identity=["fixed"], metrics={metric: value for metric in analysis.METRICS}))
    report = analysis.paired_overhead(rows, {str(block): dict(status="passed") for block in (0, 1)})
    assert report["passed"] is passed


@pytest.mark.parametrize("clock", [None, "", "local-mac", "linux-monotonic:missing-namespace"])
def test_missing_attested_clock_is_not_local_fallback(clock, monkeypatch):
    monkeypatch.setattr(graph, "host_clock_id", lambda: pytest.fail("local clock fallback"))
    with pytest.raises(ValueError, match="attested_clock_missing"):
        analysis.graph_metrics({}, {}, {}, {}, clock, 1, 2)


def test_nan_artifact_keeps_raw_hash(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"value":NaN}')
    reader = analysis.ArtifactReader(tmp_path)
    with pytest.raises(ValueError, match="nonfinite"):
        reader.load(path.name)
    assert reader.hashes[path.name] == analysis.digest(path)


@pytest.mark.parametrize("fault", ["units", "cpu", "pid", "group", "duplicate", "time", "empty"])
def test_process_usage_rejects_inconsistent_samples(tmp_path, fault):
    rows = [dict(group=10, at=20 + index, clock_ticks_per_s=100, page_bytes=4096, read_failures=[],
                 processes=[dict(pid=10, start_ticks=1, cpu_ticks=2 + index, rss_pages=1)]) for index in range(2)]
    if fault == "units":
        rows[1]["page_bytes"] = 8192
    elif fault == "cpu":
        rows[1]["processes"][0]["cpu_ticks"] = 0
    elif fault == "pid":
        for row in rows:
            row["processes"][0]["pid"] = 11
    elif fault == "group":
        rows[1]["group"] = 11
    elif fault == "duplicate":
        rows[1]["processes"] *= 2
    elif fault == "time":
        rows[1]["at"] = 0
    elif fault == "empty":
        rows = []
    filename = tmp_path / "usage.jsonl"
    filename.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        analysis.process_usage(filename, 10)


def test_completion_cadence_may_coalesce_and_repeat_final_count(tmp_path):
    case = analysis.scan_cases(scan_job.overhead_config(0, "off"), 2, 2)[0]
    case_artifacts(tmp_path, case, "off", {})
    rows = json.loads((tmp_path / f"{case['name']}-requests.json").read_text())
    for row in rows:
        row["cadence"] = [[100.25, 0], *row["cadence"][2:], row["cadence"][-1]]
    report = analysis.request_metrics(case, rows, 100.0)
    assert report["metrics"]["steady_step_s"] == 1 / 32


def test_actual_measure_layout_artifacts_are_accepted(tmp_path, monkeypatch):
    case = calibration.scan_cases(scan_job.overhead_config(0, "off"), 2, 2)[0]
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(calibration, "time", SimpleNamespace(monotonic=lambda: clock.now))

    class Controller:
        async def memory(self, **kwargs):
            return None

        async def summaries(self):
            return [[dict(preemptions_total=0)] * 2 for _ in range(2)]

    async def idle(controller):
        return None

    async def stream(engine, request_id, prompt, output_tokens, rank, model, role, arrived_at, progress):
        first = arrived_at + 0.5
        completed = first + (output_tokens - 1) / 32
        clock.now = completed + 1
        return dict(request_id=request_id, rank_initial=rank, model=model, role=role, prompt_tokens=len(prompt),
                    output_tokens=output_tokens, arrived_at=arrived_at, submitted_at=arrived_at,
                    first_token_at=first, completed_at=completed, cadence_stride=32,
                    cadence=[[first + (count - 1) / 32, count] for count in [1, *range(32, output_tokens + 1, 32)]])

    monkeypatch.setattr(calibration, "wait_idle", idle)
    monkeypatch.setattr(calibration, "fits_budget", lambda *args: True)
    monkeypatch.setattr(calibration, "stream", stream)
    prompts = [[1] * case["context_tokens"]] * case["batch_size"]
    asyncio.run(calibration.measure_layout([None, None], Controller(), case, prompts, tmp_path))
    report = analysis.analyze_case(tmp_path, case, "off", {}, CLOCK)
    assert report["available"], report["errors"]
    assert report["metrics"]["steady_step_s"] == 1 / 32


@pytest.mark.parametrize("missing,change_source", [(False, False), (True, False), (False, True)])
def test_complete_missing_and_unfrozen_dependency_cli(archived, missing, change_source):
    bundle = archived / "bundle"
    rules = dict(schema=1, deployment_manifest_sha256=analysis.digest(bundle / "inputs/deployment-manifest.json"),
                 deployment_protocol_sha256=analysis.digest(bundle / "inputs/graph-overhead-protocol.json"),
                 startup_expected=238, startup_case_ids=[("fixture", f"test_{index}") for index in range(238)],
                 analysis_sources=analysis.analysis_sources())
    if change_source:
        rules["analysis_sources"].pop("runtime/engine.py")
    path = archived / "analysis-protocol.json"
    save(path, rules)
    command = [sys.executable, str(PROJECT / "tools/analyze_scan_overhead.py"), "--bundle", str(bundle),
               "--analysis-protocol", str(path), "--manifest-sha256", rules["deployment_manifest_sha256"],
               "--output", str(archived / "cli-result")]
    for block in ((0,) if missing else (0, 1)):
        command.extend(["--job", f"{block}={archived / ('job' + str(block))}"])
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if change_source:
        assert result.returncode != 0 and "analysis_dependency_set_not_frozen" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        report = json.loads((archived / "cli-result/report.json").read_text())
        assert report["overhead"]["passed"] is not missing


def test_shared_deadline_rejects_late_start_even_if_condition_finishes(archived, monkeypatch):
    monkeypatch.setattr(graph, "host_clock_id", lambda: CLOCK)
    root = archived / "job0/out/scan-overhead-block"
    process = json.loads((root / "collect/process.json").read_text())
    process.update(started_at=2800, finished_at=3000)
    save(root / "collect/process.json", process)
    change(root / "collect/out/phases.json", lambda row: row["phases"][0].update(wall=2850))
    change(root / "status.json", lambda row: row["completed"].__setitem__(1, dict(mode="collect", **process)))
    report = run_analysis(archived)
    assert "condition_started_without_minimum_remaining_time" in report["blocks"]["0"]["errors"]
    assert not report["overhead"]["passed"]
