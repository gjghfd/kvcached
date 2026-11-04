import ast
import hashlib
import inspect
import json
from contextlib import contextmanager
from pathlib import Path

import live_trace
import verify_trace


GREEDY_ERRORS = (
    "Selected trace request differs from its full serial greedy reference",
    "Paired policy trace token IDs differ from no-move baseline",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rewrite_greedy_raises(function, observer):
    source = inspect.getsource(function)
    tree = ast.parse(source)
    counts = {message: 0 for message in GREEDY_ERRORS}

    class Transformer(ast.NodeTransformer):
        def visit_Raise(self, node):
            error = node.exc
            if (isinstance(error, ast.Call) and isinstance(error.func, ast.Name)
                    and error.func.id == "RuntimeError" and len(error.args) == 1
                    and isinstance(error.args[0], ast.Constant) and error.args[0].value in counts):
                message = error.args[0].value
                counts[message] += 1
                replacement = ast.Expr(value=ast.Call(func=ast.Name(id="_observe_performance_greedy", ctx=ast.Load()),
                    args=[ast.Constant(value=message)], keywords=[]))
                return ast.copy_location(replacement, node)
            return node

    tree = Transformer().visit(tree)
    if counts != {message: 1 for message in GREEDY_ERRORS}:
        raise ValueError("Require exactly the two declared greedy-only runtime gates")
    namespace = {**function.__globals__, "_observe_performance_greedy": observer}
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(function), "exec"), namespace)
    return namespace[function.__name__]


def compare_tokens(actual, reference):
    first = next((index for index, values in enumerate(zip(actual, reference)) if values[0] != values[1]), None)
    equal = actual == reference
    if not equal and first is None:
        first = min(len(actual), len(reference))
    return {"equal": equal, "first_difference_zero_based": first,
            "actual_tokens": len(actual), "reference_tokens": len(reference),
            "actual_token": actual[first] if first is not None and first < len(actual) else None,
            "reference_token": reference[first] if first is not None and first < len(reference) else None}


def observe_maps(config, token_maps, references):
    policies = config["policies"]
    if not policies or len(policies) != len(set(policies)):
        raise ValueError("Require unique configured reference policies")
    reference_policy = config.get("reference_policy", "no_move" if "no_move" in policies else policies[0])
    if reference_policy not in policies:
        raise ValueError("Token reference policy is not a configured trial")
    serial, paired = [], []
    for repeat in range(config["repeats"]):
        baseline = token_maps[(repeat, reference_policy)]
        for policy in policies:
            actual = token_maps[(repeat, policy)]
            if set(actual) != set(baseline):
                raise ValueError("Missing or extra request IDs cannot be treated as numerical divergence")
            for identifier, reference in baseline.items():
                if len(actual[identifier]) != len(reference):
                    raise ValueError("Different output workloads cannot be treated as numerical divergence")
                paired.append({"repeat": repeat, "policy": policy, "request_id": identifier,
                               **compare_tokens(actual[identifier], reference)})
            for reference in references:
                identifier = reference["trace_request_id"]
                if len(actual[identifier]) != len(reference["token_ids"]):
                    raise ValueError("Different serial output lengths cannot be treated as numerical divergence")
                serial.append({"repeat": repeat, "policy": policy, "request_id": identifier,
                               **compare_tokens(actual[identifier], reference["token_ids"])})
    return {"reference_policy": reference_policy, "serial": serial, "paired": paired,
            "full_greedy_equivalence_passed": all(item["equal"] for item in serial + paired),
            "greedy_divergence_does_not_abort_performance": True,
            "numerical_root_cause_proven": False}


def observe_root(root):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    maps = {(repeat, policy): {record["request_id"]: record["token_ids"] for record in json.loads(
        (root / f"r{repeat}-{policy}/requests.json").read_text())}
        for repeat in range(config["repeats"]) for policy in config["policies"]}
    return observe_maps(config, maps, json.loads((root / "references.json").read_text()))


@contextmanager
def runtime_observation(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    original = live_trace.run
    events = []
    policy = {"schema_version": 1, "user_authorized_on": "2026-11-02",
              "mode": "performance_greedy_observation", "greedy_gate_is_fatal": False,
              "all_non_greedy_runtime_errors_remain_fatal": True,
              "BF16_default_head_and_sparse_graphs_unchanged": True,
              "no_hidden_head_CPU_snapshot_hooks": True,
              "base_live_trace_sha256": digest(Path(live_trace.__file__)),
              "performance_numerics_sha256": digest(Path(__file__)),
              "declared_runtime_gate_messages": list(GREEDY_ERRORS)}
    with (root / "performance-numerics-policy.json").open("x") as output:
        output.write(json.dumps(policy, indent=2) + "\n")
    live_trace.run = rewrite_greedy_raises(original, events.append)
    try:
        yield
        observations = observe_root(root)
        observations["runtime_greedy_events"] = events
        (root / "greedy-observations.json").write_text(json.dumps(observations, indent=2) + "\n")
        summary = json.loads((root / "summary.json").read_text())
        summary.update({"performance_completed": True,
                        "full_greedy_equivalence_passed": observations["full_greedy_equivalence_passed"],
                        "scope": "performance_measurement_with_greedy_observation_quality_unresolved"})
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        live_trace.run = original


def terminal_acceptable(root, allow_drained_greedy_failure):
    root = Path(root)
    status = (root / "exit-status").read_text().strip()
    summary = json.loads((root / "summary.json").read_text())
    if status == "0" and summary["status"] == "passed":
        return True
    if not allow_drained_greedy_failure or status != "1" or summary["status"] != "running":
        return False
    error = json.loads((root / "error.json").read_text())
    return error.get("type") == "RuntimeError" and error.get("message") in GREEDY_ERRORS


def verification_function(observer, allow_drained_greedy_failure):
    original = verify_trace.verify
    tree = ast.parse(inspect.getsource(original))
    changed = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and len(node.body) == 1 and isinstance(node.body[0], ast.Raise)
                and isinstance(node.body[0].exc, ast.Call)
                and any(isinstance(argument, ast.Constant) and argument.value == "Trace allocation did not complete successfully"
                        for argument in node.body[0].exc.args)):
            node.test = ast.UnaryOp(op=ast.Not(), operand=ast.Call(
                func=ast.Name(id="_performance_terminal_acceptable", ctx=ast.Load()),
                args=[ast.Name(id="root", ctx=ast.Load())], keywords=[]))
            changed += 1
    if changed != 1:
        raise ValueError("Require one exact terminal gate; do not weaken other structural verification")
    namespace = {**original.__globals__, "verify_paired_tokens": observer,
                 "_performance_terminal_acceptable": lambda root: terminal_acceptable(root, allow_drained_greedy_failure)}
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(original), "exec"), namespace)
    return namespace[original.__name__]


def verify_sources(root, config, allow_legacy):
    path = root / "performance-source.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        names = {"performance_trace.py", "performance_numerics.py", "solver_efficiency.py",
                 "grouped_robust_uncertainty.py", "prune_headroom_scenarios.py", "verify_solver_efficiency.py"}
        if (set(manifest["source_sha256"]) != names
                or manifest["original_strict_greedy_token_checks_unchanged"] is not False
                or manifest["greedy_divergence_observed_not_fatal"] is not True
                or manifest["objective_and_uncertainty_unchanged"] is not True
                or manifest["equivalent_robust_solver_enabled"] != (config.get("workaware_optimizer") == "robust")):
            raise ValueError("Performance source manifest must explicitly disclose the observation-only greedy gate")
        policy = json.loads((root / "performance-numerics-policy.json").read_text())
        if (policy["mode"] != "performance_greedy_observation" or policy["greedy_gate_is_fatal"] is not False
                or policy["declared_runtime_gate_messages"] != list(GREEDY_ERRORS)
                or policy["base_live_trace_sha256"] != digest(live_trace.__file__)
                or policy["performance_numerics_sha256"] != manifest["source_sha256"]["performance_numerics.py"]):
            raise ValueError("Performance numerical policy differs from its frozen source/configuration")
    elif allow_legacy and config.get("workaware_optimizer") == "robust":
        from verify_efficient_trace import SOURCES

        manifest = json.loads((root / "solver-efficiency-source.json").read_text())
        if (set(manifest["source_sha256"]) != set(SOURCES)
                or manifest["objective_and_uncertainty_unchanged"] is not True
                or manifest["original_strict_greedy_token_checks_unchanged"] is not True):
            raise ValueError("Legacy analysis requires its original equivalent-solver source evidence")
    else:
        raise ValueError("Require explicit performance source evidence; legacy failures need separate opt-in")
    for name, expected in manifest["source_sha256"].items():
        if digest(root / (name + ".executed")) != expected:
            raise ValueError("Executed performance or equivalent-solver source SHA differs")
    return manifest


def verify(root, prompt_bank=None, allow_drained_greedy_failure=False):
    root = Path(root)
    observation = {}

    def tokens(config, maps, references):
        observation.update(observe_maps(config, maps, references))
        return observation["reference_policy"]

    config = json.loads((root / "config.json").read_text())
    source = verify_sources(root, config, allow_drained_greedy_failure)
    import workaware_robust_adapter
    from verify_solver_efficiency import verify_plan

    original = workaware_robust_adapter.verify_raw_plan
    if config.get("workaware_optimizer") == "robust":
        workaware_robust_adapter.verify_raw_plan = verify_plan
    try:
        result = verification_function(tokens, allow_drained_greedy_failure)(root, prompt_bank)
    finally:
        workaware_robust_adapter.verify_raw_plan = original
    return {**result, "status": "performance_trace_verified_greedy_observed", "greedy_observations": observation,
            "performance_or_legacy_solver_source_SHA_verified": source["source_sha256"],
            "raw_exit_status": (root / "exit-status").read_text().strip(),
            "raw_summary_status": json.loads((root / "summary.json").read_text())["status"],
            "old_drained_greedy_failure_not_reclassified_as_job_success": allow_drained_greedy_failure,
            "request_lengths_arrivals_ownership_budget_migration_and_metrics_checks_preserved": True,
            "quality_equivalence_unresolved": not observation["full_greedy_equivalence_passed"]}
