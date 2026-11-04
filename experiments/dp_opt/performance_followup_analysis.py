import ast
import inspect
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import amortized_campaign
import analyze_tail_slo
import public_poisson_followup as poisson
import public_workload_search as search
from archive_trace_job import verify_raw
from performance_numerics import digest, observe_maps, verify
from performance_public import adapted
from public_response_inputs import validate
from verify_workaware_profile import same_payload, verify_profile


SCORE_RULE = "equal_group_log_TTFT_gain_vs_no_move_plus_log_Conv_TPOT_gain_vs_random"
COST_SCOPE = "reuse_verified_native_N2_N3_N4_no_new_N5_N9_profiles_no_outside_envelope_forecasts"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    if path.exists():
        if read_json(path) != payload:
            raise ValueError("Preserve immutable followup evidence: " + str(path))
    else:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix="performance-followup-json-", delete=False) as output:
            output.write(json.dumps(payload, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
            temporary = Path(output.name)
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if read_json(path) != payload:
                    raise ValueError("Preserve immutable followup evidence: " + str(path))
        finally:
            temporary.unlink()
    return payload


def file_hashes(root):
    root = Path(root)
    paths = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Frozen copies cannot contain symbolic links")
    return {str(path.relative_to(root)): digest(path) for path in paths if path.is_file()}


def check_hashes(root, manifest):
    root = Path(root)
    for name, expected in manifest.items():
        path = root / name
        if (Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve()) or digest(path) != expected):
            raise ValueError("Frozen followup SHA differs: " + name)


def validate_inputs(inputs):
    manifest = validate(inputs)
    if manifest["dataset_label"] != "openthoughts3_closed":
        raise ValueError("Require the frozen structurally-closed original OpenThoughts3 cohort")
    if any(mark["prompt_tokens"] < 100000 or mark["output_tokens"] != 64
           for mark in read_json(inputs / "code-marks.json")):
        raise ValueError("SWE-smith must retain real >=100K inputs and original O64")
    return manifest


def native_names():
    return [f"native-{seed}-N{population}" for seed in search.SEEDS for population in search.POPULATIONS]


def parent_checkpoint(parent, native_job, allow_checkpoint_cancelled=False):
    previous = read_json(parent / "campaign-state.json")
    accounting = parent / f"accounting-{native_job}.txt"
    rows = [line.split("|") for line in accounting.read_text().splitlines()
            if line.split("|")[0].strip() == str(native_job)]
    if len(rows) != 1 or len(rows[0]) < 3:
        raise ValueError("Require the original native allocation's canonical terminal accounting")
    terminal = rows[0][1].strip().split()[0].rstrip("+")
    if previous.get("job") != native_job or previous.get("state_last_observed") != terminal:
        raise ValueError("Native coordinator and independent archived accounting disagree")
    if terminal == "CANCELLED" and not allow_checkpoint_cancelled:
        raise ValueError("CANCELLED native rescue requires explicit scientific-checkpoint authorization")
    if terminal not in ("COMPLETED", "TIMEOUT", "CANCELLED"):
        raise ValueError("Diagnose the existing native coordinator's failure; no automatic FAILED rescue")
    raw = parent / f"live-trace-{native_job}"
    raw_manifest = parent / f"raw-{native_job}.sha256"
    verify_raw(raw, raw_manifest.read_text())
    exit_status = (raw / "exit-status").read_text().strip() if (raw / "exit-status").exists() else None
    if terminal == "COMPLETED" and (exit_status != "0" or rows[0][2].strip() != "0:0"):
        raise ValueError("Do not reclassify the parent allocation's exit status")
    return {"native_job": native_job, "terminal": terminal, "raw_exit_status": exit_status,
            "accounting_exit_code": rows[0][2].strip(), "accounting_sha256": digest(accounting),
            "native_raw_manifest_sha256": digest(raw_manifest),
            "scientific_checkpoint_cancelled_authorized": allow_checkpoint_cancelled,
            "original_allocation_not_reclassified_as_success": True}


def native_configuration(phase, fit_sha256):
    config = read_json(phase / "config.json")
    expected = {"policies": list(search.POLICIES), "repeats": 1, "max_model_len": 131072,
                "cache_bytes": 25769803776, "kv_budget_mib": 10240, "request_timeout_s": 7200,
                "allocator_gpu_utilization": .98, "fairness_queue_threshold_s": 30,
                "workaware_optimizer": "robust", "workaware_fit_sha256": fit_sha256,
                "workaware_rate_priors": [.001, .012]}
    if any(config.get(field) != value for field, value in expected.items()):
        raise ValueError("Native case changed original policies, budgets, defaults or verified costs")
    placement_sha = digest(phase / "placement-fit.json")
    if not any(line.split()[0] == placement_sha for line in (phase / "source.sha256").read_text().splitlines()):
        raise ValueError("Native placement fit lacks executed-source SHA provenance")
    return {"movement_costs": {field: config[field] for field in ("move_fixed_s", "move_bandwidth_gib_s")},
            "placement_fit_sha256": placement_sha, "native_controller_seed": config["seed"]}


def verify_native_case(phase, frozen, controls, fit_sha256):
    if phase.name not in native_names() or digest(phase / "trace.json") != digest(frozen / (phase.name + ".json")):
        raise ValueError("Native rescue must retain the exact original frozen seed/population trace")
    if native_configuration(phase, fit_sha256) != controls:
        raise ValueError("Native case changed placement fit, migration controls or controller seed")
    if ((phase / "exit-status").read_text().strip() != "0"
            or read_json(phase / "summary.json").get("status") != "passed"):
        raise ValueError("Only complete exit0 native cases can supply measurement evidence")
    return verify(phase, frozen / "inputs/prompt-bank.json")


def prepare_workspace(root, parent, native_job, allow_checkpoint_cancelled=False, coordinator_exit=None):
    if not coordinator_exit or coordinator_exit.get("stopped") is not True:
        raise ValueError("Wait for the original local native coordinator to stop before rescue")
    checkpoint = parent_checkpoint(parent, native_job, allow_checkpoint_cancelled)
    raw = parent / f"live-trace-{native_job}"
    search.validate_frozen(raw / "frozen")
    validate_inputs(raw / "frozen/inputs")
    fit = verify_profile(raw / "public-context-calibration")
    if not same_payload(fit, read_json(raw / "public-context-fit.json")):
        raise ValueError("Native default-head costs do not reconstruct")
    fit_sha = digest(raw / "public-context-fit.json")
    configured = [raw / name for name in native_names() if all((raw / name / filename).exists()
                  for filename in ("config.json", "placement-fit.json", "source.sha256"))]
    if not configured:
        raise ValueError("Require archived native input configuration; do not guess movement costs or controller seed")
    first = configured[0]
    controls = native_configuration(first, fit_sha)
    if any(native_configuration(phase, fit_sha) != controls for phase in configured):
        raise ValueError("All native configurations must use the same placement fit and migration-cost controls")
    results, excluded = {}, {}
    for name in native_names():
        phase = raw / name
        status = (phase / "exit-status").read_text().strip() if (phase / "exit-status").exists() else None
        if status == "0":
            results[name] = verify_native_case(phase, raw / "frozen", controls, fit_sha)
        else:
            if (phase / "error.json").exists():
                raise ValueError("Native non-greedy error requires diagnosis, not checkpoint rescue")
            excluded[name] = {"raw_exit_status": status, "used_for_performance_or_rates": False,
                              "reason": "missing_or_incomplete_native_case"}
    independent = parent / "independent-native-performance.json"
    if independent.exists() and read_json(independent) != results:
        raise ValueError("All six or available native cases must independently reconstruct")
    if checkpoint["terminal"] == "COMPLETED" and excluded:
        raise ValueError("COMPLETED native parent did not independently supply all six cases")
    receipt = {**checkpoint, **controls, "coordinator_exit": coordinator_exit, "available_cases": results,
               "excluded_cases": excluded, "placement_fit_source_phase": first.name,
               "cost_scope": COST_SCOPE, "quality_equivalence_not_proven": True}
    workspace = root / "workspace"
    if workspace.exists():
        original = read_json(root / "native-parent-review.json")
        if {name: value for name, value in original.items() if name != "workspace_initial_sha256"} != receipt:
            raise ValueError("The archived native parent changed")
        check_hashes(workspace, original["workspace_initial_sha256"])
        if any(file_hashes(workspace / group) != file_hashes(raw / group)
               for group in ("frozen", "public-context-calibration", *results)):
            raise ValueError("Copied original native cases or inputs differ from their canonical raw SHA archive")
        return workspace
    with tempfile.TemporaryDirectory(prefix="performance-followup-", dir=root) as directory:
        building = Path(directory) / "workspace"
        shutil.copytree(raw / "frozen", building / "frozen")
        for name in results:
            shutil.copytree(raw / name, building / name)
        shutil.copytree(raw / "public-context-calibration", building / "public-context-calibration")
        shutil.copy2(raw / "public-context-fit.json", building / "public-context-fit.json")
        shutil.copy2(first / "placement-fit.json", building / "placement-fit.json")
        receipt["workspace_initial_sha256"] = file_hashes(building)
        write_json(root / "native-parent-review.json", receipt)
        building.rename(workspace)
    return workspace


def merge_native_case(root, entry, result):
    review = read_json(root / "native-parent-review.json")
    name = entry["plan"]["phase_name"]
    if name in review["available_cases"] or name not in native_names() or entry["plan"]["kind"] != "native":
        raise ValueError("Rescue cannot replace an already verified original native case")
    source = root / "stages" / entry["stage_id"] / f"live-trace-{entry['job']}" / name
    target = root / "workspace" / name
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix="performance-followup-native-", dir=root) as directory:
            building = Path(directory) / name
            shutil.copytree(source, building)
            building.rename(target)
    if file_hashes(target) != file_hashes(source):
        raise ValueError("Merged rescue case differs from its original raw archive")
    write_json(root / (name + "-provenance.json"), {"stage_id": entry["stage_id"], "job": entry["job"],
        "terminal": entry["terminal"], "verification": result, "case_sha256": file_hashes(source),
        "raw_manifest_sha256": digest(root / "stages" / entry["stage_id"] / f"raw-{entry['job']}.sha256")})


def finalize_native_workspace(root, parent, native_job, shard_verifier):
    workspace = root / "workspace"
    review = read_json(root / "native-parent-review.json")
    check_hashes(workspace, review["workspace_initial_sha256"])
    state = read_json(root / "campaign-state.json")
    cases, provenance = {}, {}
    controls = {field: review[field] for field in ("movement_costs", "placement_fit_sha256", "native_controller_seed")}
    for name in native_names():
        if name in review["available_cases"]:
            if file_hashes(workspace / name) != file_hashes(parent / f"live-trace-{native_job}" / name):
                raise ValueError("Original native case copy differs from its canonical raw allocation")
            cases[name] = review["available_cases"][name]
            provenance[name] = {"job": native_job, "terminal": review["terminal"],
                "raw_manifest_sha256": review["native_raw_manifest_sha256"],
                "case_sha256": file_hashes(workspace / name), "complete_case_exit0": True}
        else:
            entry = state["stages"].get(name)
            if not entry or entry["phase"] != "verified":
                raise ValueError("Do not freeze calibration before all six native cases independently verify")
            cases[name] = shard_verifier(root, entry)
            merge_native_case(root, entry, cases[name])
            provenance[name] = read_json(root / (name + "-provenance.json"))
        rebuilt = verify_native_case(workspace / name, workspace / "frozen", controls,
                                     digest(workspace / "public-context-fit.json"))
        if rebuilt != cases[name]:
            raise ValueError("Merged native case independent verification changed")
    frozen = workspace / "calibration-frozen"
    receipt_path = root / "native-provenance.json"
    pending = root / "native-rate-freeze.pending"
    if not receipt_path.exists() and not pending.exists():
        with tempfile.TemporaryDirectory(prefix="performance-followup-rate-freeze-", dir=root) as directory:
            building = Path(directory) / "workspace"
            building.mkdir()
            for name in ("frozen", *native_names()):
                shutil.copytree(workspace / name, building / name)
            with adapted(search):
                search.freeze_calibration(building)
            parent_freeze = parent / f"live-trace-{native_job}/calibration-frozen"
            compared = parent_freeze.exists() and set(review["available_cases"]) == set(native_names())
            if compared and file_hashes(building / "calibration-frozen") != file_hashes(parent_freeze):
                raise ValueError("GPU rate traces differ from the original measured-service freeze")
            receipt = {**controls, "native_job": native_job, "six_cases": cases, "case_provenance": provenance,
                "native_raw_manifest_sha256": review["native_raw_manifest_sha256"],
                "original_native_terminal": review["terminal"], "original_job_statuses_unchanged": True,
                "parent_calibration_compared_not_overwritten": compared, "cost_scope": COST_SCOPE,
                "workspace_initial_sha256": {name: sha for name, sha in file_hashes(workspace).items()
                    if name.split("/")[0] in ("frozen", *native_names(), "public-context-calibration",
                                              "public-context-fit.json", "placement-fit.json")},
                "calibration_frozen_sha256": file_hashes(building / "calibration-frozen"),
                "quality_equivalence_not_proven": True}
            ready = Path(directory) / "ready"
            ready.mkdir()
            (building / "calibration-frozen").rename(ready / "calibration-frozen")
            write_json(ready / "receipt.json", receipt)
            ready.rename(pending)
    if pending.exists():
        receipt = read_json(pending / "receipt.json")
        if receipt["six_cases"] != cases or receipt["case_provenance"] != provenance:
            raise ValueError("Pending native rate freeze provenance changed")
        write_json(receipt_path, receipt)
        if frozen.exists():
            if file_hashes(frozen) != receipt["calibration_frozen_sha256"]:
                raise ValueError("Preserve the already frozen GPU rate traces")
        else:
            if file_hashes(pending / "calibration-frozen") != receipt["calibration_frozen_sha256"]:
                raise ValueError("Pending GPU rate trace freeze SHA differs")
            (pending / "calibration-frozen").rename(frozen)
    receipt = read_json(receipt_path)
    expected = {**controls, "native_job": native_job, "original_native_terminal": review["terminal"],
                "native_raw_manifest_sha256": review["native_raw_manifest_sha256"], "cost_scope": COST_SCOPE,
                "original_job_statuses_unchanged": True}
    if any(receipt.get(field) != value for field, value in expected.items()):
        raise ValueError("Native controls, source allocation status or provenance changed")
    if receipt["six_cases"] != cases or receipt["case_provenance"] != provenance:
        raise ValueError("Native six-case provenance changed on resume")
    if not frozen.exists():
        raise ValueError("Interrupted rate-freeze install; explicit reconciliation required, never reroll rates")
    check_hashes(workspace, receipt["workspace_initial_sha256"])
    if file_hashes(frozen) != receipt["calibration_frozen_sha256"]:
        raise ValueError("Frozen rate traces changed after all-six-case calibration")
    if pending.exists():
        shutil.rmtree(pending)
    return receipt


def observation_tail_function(original, terminal):
    tree = ast.parse(inspect.getsource(original))
    counts = {"greedy": 0, "terminal": 0}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or len(node.body) != 1 or not isinstance(node.body[0], ast.Raise):
            continue
        error = node.body[0].exc
        if not isinstance(error, ast.Call) or len(error.args) != 1 or not isinstance(error.args[0], ast.Constant):
            continue
        if error.args[0].value == "Paired policies generated different output token IDs":
            node.body = [ast.Pass()]
            counts["greedy"] += 1
        elif error.args[0].value == "Tail comparison requires a successfully completed allocation":
            node.test = ast.UnaryOp(op=ast.Not(), operand=ast.Call(
                func=ast.Name(id="_followup_tail_terminal", ctx=ast.Load()),
                args=[ast.Name(id="root", ctx=ast.Load())], keywords=[]))
            counts["terminal"] += 1
    if counts != {"greedy": 1, "terminal": 1}:
        raise ValueError("Require exact tail gates; no structural or metric check can be weakened")
    namespace = {**original.__globals__, "_followup_tail_terminal": terminal}
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(original), "exec"), namespace)
    return namespace[original.__name__]


@contextmanager
def tail_observation(split_verifier=None):
    original = amortized_campaign.analyze

    def terminal(root):
        if (root / "split-provenance.json").exists():
            if split_verifier is None:
                raise ValueError("Split completion requires independent shard verification")
            split_verifier(root, None)
            return True
        return ((root / "exit-status").read_text().strip() == "0"
                and read_json(root / "summary.json")["status"] == "passed")

    transformed = observation_tail_function(original, terminal)

    def analyze(*args, **kwargs):
        result = transformed(*args, **kwargs)
        result["quality_equivalence_not_proven"] = True
        result["greedy_observation_only_lengths_arrivals_metrics_checks_preserved"] = True
        return result

    amortized_campaign.analyze = analyze
    try:
        yield
    finally:
        amortized_campaign.analyze = original


def analyze_calibration(root):
    workspace = root / "workspace"
    check_hashes(workspace, read_json(root / "native-provenance.json")["workspace_initial_sha256"])
    with adapted(search), tail_observation():
        result = search.analyze(workspace)
    write_json(root / "rate-analysis.json", result)
    return result


def evaluation_configuration(root):
    analysis = read_json(root / "rate-analysis.json")
    workspace = root / "workspace"
    grid = poisson.search_rates(analysis["calibration"], analysis["candidate_case"])
    return {"schema_version": 1, "grid": grid, "policies": list(poisson.POLICIES),
            "dataset_label": "openthoughts3_closed", "exploration_seed": poisson.EXPLORATION_SEED,
            "heldout_seed": poisson.HELDOUT_SEED, "exploration_s": poisson.EXPLORATION_S,
            "heldout_s": poisson.HELDOUT_S, "warmup_s": 600, "selection_rule": SCORE_RULE,
            "fit_sha256": digest(workspace / "public-context-fit.json"), "cost_scope": COST_SCOPE,
            "inputs_sha256": digest(workspace / "frozen/inputs/input-manifest.json"),
            "score_source_sha256": digest(poisson.__file__), "rate_analysis_sha256": digest(root / "rate-analysis.json")}


def freeze_evaluation(root):
    config = evaluation_configuration(root)
    frozen = root / "evaluation/frozen"
    if not frozen.exists():
        with tempfile.TemporaryDirectory(prefix="performance-followup-grid-", dir=root) as directory:
            building = Path(directory) / "frozen"
            shutil.copytree(root / "workspace/frozen/inputs", building / "inputs")
            for ordinal, setting in enumerate(config["grid"]):
                write_json(building / f"explore-{ordinal}.json", poisson.make_poisson(
                    building / "inputs", poisson.EXPLORATION_SEED, setting["conv_rate"],
                    poisson.EXPLORATION_S, setting["code_rate"]))
            shutil.copy2(building / "explore-0.json", building / "trace.json")
            config["trace_sha256"] = {f"explore-{ordinal}.json": digest(building / f"explore-{ordinal}.json")
                                      for ordinal in range(len(config["grid"]))}
            write_json(building / "evaluation-config.json", config)
            frozen.parent.mkdir(exist_ok=True)
            building.rename(frozen)
    return validate_evaluation(frozen)


def validate_evaluation(frozen):
    root = frozen.parent.parent
    config = read_json(frozen / "evaluation-config.json")
    expected = evaluation_configuration(root)
    validate_inputs(frozen / "inputs")
    if file_hashes(frozen / "inputs") != file_hashes(root / "workspace/frozen/inputs"):
        raise ValueError("Evaluation inputs differ from the native freeze")
    for ordinal, setting in enumerate(expected["grid"]):
        path = frozen / f"explore-{ordinal}.json"
        if read_json(path) != poisson.make_poisson(frozen / "inputs", poisson.EXPLORATION_SEED,
                setting["conv_rate"], poisson.EXPLORATION_S, setting["code_rate"]):
            raise ValueError("Do not change rates, marks, seeds or original response lengths")
    expected["trace_sha256"] = {f"explore-{ordinal}.json": digest(frozen / f"explore-{ordinal}.json")
                               for ordinal in range(len(expected["grid"]))}
    if config != expected or (frozen / "trace.json").read_bytes() != (frozen / "explore-0.json").read_bytes():
        raise ValueError("The predeclared grid or score changed")
    return config


def verified_native_profile(evaluation):
    workspace = evaluation.parent / "workspace"
    fit = verify_profile(workspace / "public-context-calibration")
    config = validate_evaluation(evaluation / "frozen")
    if (not same_payload(fit, read_json(workspace / "public-context-fit.json"))
            or digest(workspace / "public-context-fit.json") != config["fit_sha256"]):
        raise ValueError("All split policies require the identical verified default-head cost fit")
    return {"status": "native_default_head_costs_verified", "fit_sha256": config["fit_sha256"],
            "cost_scope": COST_SCOPE, "new_N5_N9_population_calibration_performed": False,
            "outside_population_context_envelope": "no_forecast_no_extrapolation"}


def join_phase(root, name, entries):
    phase = root / "evaluation" / name
    if set(entries) != set(poisson.POLICIES):
        raise ValueError("Require all seven archived policy allocations before joining a phase")
    sources = [{"policy": policy, "stage_id": entries[policy]["stage_id"]} for policy in poisson.POLICIES]
    first = entries[poisson.POLICIES[0]]
    first_phase = root / "stages" / first["stage_id"] / f"live-trace-{first['job']}" / name
    config = {**read_json(first_phase / "config.json"), "policies": list(poisson.POLICIES),
              "reference_policy": "no_move", "split_policy_view_not_an_allocation": True,
              "workaware_optimizer": "robust",
              "split_optimizer_by_policy": {policy: entry["plan"]["optimizer"] for policy, entry in entries.items()}}
    phase.mkdir(exist_ok=True)
    write_json(phase / "split-provenance.json", {"campaign_root": str(root), "sources": sources})
    write_json(phase / "config.json", config)
    write_json(phase / "summary.json", {"status": "split_policy_view_not_an_allocation",
                                         "original_job_statuses_unchanged": True})
    if not (phase / "exit-status").exists():
        (phase / "exit-status").write_text("NOT_AN_ALLOCATION\n")
    if not (phase / "trace.json").exists():
        shutil.copy2(first_phase / "trace.json", phase / "trace.json")
    for policy, entry in entries.items():
        source = root / "stages" / entry["stage_id"] / f"live-trace-{entry['job']}" / name / f"r0-{policy}"
        target = phase / f"r0-{policy}"
        if not target.is_symlink():
            target.symlink_to(source, target_is_directory=True)
        if target.resolve() != source.resolve():
            raise ValueError("Split policy view points to a different raw allocation")
    return phase


def verify_join(root, phase, shard_verifier):
    provenance = read_json(phase / "split-provenance.json")
    expected_policies = list(poisson.POLICIES)
    if (provenance["campaign_root"] != str(root)
            or [source["policy"] for source in provenance["sources"]] != expected_policies):
        raise ValueError("Split provenance must bind all seven policies to this campaign")
    state = read_json(root / "campaign-state.json")
    results, maps, raw_sha, optimizers = {}, {}, {}, {}
    references = None
    first_config = None
    for source in provenance["sources"]:
        policy, stage_id = source["policy"], source["stage_id"]
        entry = state["stages"][stage_id]
        if entry["plan"]["phase_name"] != phase.name or entry["plan"]["policy"] != policy:
            raise ValueError("Wrong stage, candidate or policy binding in aggregate")
        results[policy] = shard_verifier(root, entry)
        optimizers[policy] = entry["plan"]["optimizer"]
        archive = root / "stages" / stage_id
        actual = archive / f"live-trace-{entry['job']}" / phase.name
        if ((phase / f"r0-{policy}").resolve() != (actual / f"r0-{policy}").resolve()
                or digest(actual / "trace.json") != digest(phase / "trace.json")):
            raise ValueError("Cross-policy trace or raw provenance changed")
        config = read_json(actual / "config.json")
        if references is None:
            references = read_json(actual / "references.json")
            first_config = config
        maps[(0, policy)] = {record["request_id"]: record["token_ids"]
                             for record in read_json(actual / f"r0-{policy}/requests.json")}
        raw_sha[policy] = {"job": entry["job"], "manifest_sha256": digest(archive / f"raw-{entry['job']}.sha256")}
    expected_config = {**first_config, "policies": expected_policies, "reference_policy": "no_move",
                       "split_policy_view_not_an_allocation": True, "workaware_optimizer": "robust",
                       "split_optimizer_by_policy": optimizers}
    if read_json(phase / "config.json") != expected_config:
        raise ValueError("Aggregate configuration no longer reflects its verified raw shards")
    return {"status": "split_performance_phase_independently_verified", "shards": results,
            "raw_SHA_provenance": raw_sha, "cross_policy_greedy_observations": observe_maps(expected_config, maps, references),
            "quality_equivalence_not_proven": True, "original_job_statuses_unchanged": True}


@contextmanager
def split_public_analysis(root, shard_verifier):
    originals = {name: getattr(poisson, name) for name in ("verify", "validate_frozen", "verify_calibration")}

    def split_verify(phase, prompt_bank=None):
        return verify_join(root, Path(phase), shard_verifier)

    with adapted(poisson), tail_observation(split_verify):
        poisson.verify = split_verify
        poisson.validate_frozen = validate_evaluation
        poisson.verify_calibration = verified_native_profile
        try:
            yield
        finally:
            for name, original in originals.items():
                setattr(poisson, name, original)
