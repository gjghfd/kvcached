import argparse
import hashlib
import json
import math
import shlex
from pathlib import Path

from layout_cost_pilot import load_verified_lookup
from verify_admission_cost import verify


PARAMETER_ENV = {
    "wait_incoming_handshake_s": "DP_OPT_WAIT_INCOMING_HANDSHAKE_S",
    "wait_old_transaction_fixed_s": "DP_OPT_WAIT_OLD_TRANSACTION_FIXED_S",
    "wait_copy_seconds_per_gib": "DP_OPT_WAIT_COPY_SECONDS_PER_GIB",
    "wait_solve_floor_s": "DP_OPT_WAIT_SOLVE_FLOOR_S",
    "move_fixed_s": "DP_OPT_MOVE_FIXED_S",
    "move_bandwidth_gib_s": "DP_OPT_MOVE_BANDWIDTH_GIB_S",
}


def rounded_upper(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("Cannot freeze invalid observed control costs")
    return math.ceil(value * 100) / 100


def parameters(trials):
    selected = [trial for trial in trials if trial["case"]["mode"] == "preplan_boundary"]
    if sorted(trial["case"]["population"] for trial in selected) != [8, 12, 16]:
        raise ValueError("Require all three predeclared mixed-population boundary cases")
    decisions = [trial["decision"] for trial in selected]
    if any(decision["solve_s"] != 0 or decision["preplan_s"] <= 0 or not decision["moves"]
           for decision in decisions):
        raise ValueError("Cost calibration must solve outside pause and actually migrate")
    slope = rounded_upper(max(move["elapsed_s"] / (move["kv_bytes"] / 1024**3)
                              for decision in decisions for move in decision["moves"]))
    fixed = rounded_upper(max(decision["end_to_end_control_s"] - decision["pause_call_s"]
                              - sum(move["elapsed_s"] for move in decision["moves"]) for decision in decisions))
    if slope <= 0:
        raise ValueError("Require positive measured end-to-end copy slope")
    return {"wait_incoming_handshake_s": rounded_upper(max(decision["pause_call_s"] for decision in decisions)),
            "wait_old_transaction_fixed_s": fixed, "wait_copy_seconds_per_gib": slope,
            "wait_solve_floor_s": rounded_upper(max(decision["preplan_s"] for decision in decisions)),
            "move_fixed_s": 0, "move_bandwidth_gib_s": 1 / slope}


def load_control(path, digest):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("Measured control configuration differs from frozen SHA256")
    decision = json.loads(payload)
    if decision["status"] != "frozen_measured_boundary_control":
        raise ValueError("Missing frozen measured control provenance")
    if (set(decision["parameters"]) != set(PARAMETER_ENV)
            or any(not math.isfinite(value) or value < 0 for value in decision["parameters"].values())
            or decision["parameters"]["wait_copy_seconds_per_gib"] <= 0
            or not math.isclose(decision["parameters"]["move_bandwidth_gib_s"],
                                1 / decision["parameters"]["wait_copy_seconds_per_gib"])):
        raise ValueError("Invalid or inconsistent frozen control parameters")
    return decision


def freeze(admission_root, inputs, lookup_path, lookup_root):
    admission = verify(admission_root, inputs)
    digest = hashlib.sha256(lookup_path.read_bytes()).hexdigest()
    lookup = load_verified_lookup(lookup_path, digest, lookup_root)
    return {"status": "frozen_measured_boundary_control", "admission_source_job": 244279,
            "layout_source_job": 244371,
            "parameters": parameters(json.loads((admission_root / "admission-trials.json").read_text())),
            "rounding_rule": "predeclared_max_observed_then_ceil_0.01_seconds_no_trace_outcome_tuning",
            "layout_lookup_sha256": digest, "layout_source_files_sha256": lookup["source_files_sha256"],
            "admission_trials_sha256": hashlib.sha256((admission_root / "admission-trials.json").read_bytes()).hexdigest(),
            "admission_verification": admission,
            "policy_scope": "preplan_and_echo_shared_by_all_migrating_policies_wait_gate_only_ILP",
            "scope": "small_Conv_population_control_cost_not_Code_prefill_counterfactual_or_net_benefit"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--layout-lookup", type=Path, required=True)
    parser.add_argument("--layout-source-root", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    decision = freeze(options.admission_root, options.inputs, options.layout_lookup, options.layout_source_root)
    options.output.mkdir()
    config = options.output / "control-config.json"
    config.write_text(json.dumps(decision, indent=2) + "\n")
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    values = {name: str(decision["parameters"][key]) for key, name in PARAMETER_ENV.items()}
    values.update({"DP_OPT_LAYOUT_COST_LOOKUP": options.remote_root + "/layout-cost-lookup.json",
                   "DP_OPT_LAYOUT_COST_LOOKUP_SHA256": decision["layout_lookup_sha256"],
                   "DP_OPT_LAYOUT_COST_SOURCE_ROOT": options.remote_root + "/layout-cost-source",
                   "DP_OPT_MEASURED_CONTROL_CONFIG": options.remote_root + "/control-config.json",
                   "DP_OPT_MEASURED_CONTROL_CONFIG_SHA256": digest,
                   "DP_OPT_PREPLAN": "1", "DP_OPT_WAIT_AWARE": "1"})
    (options.output / "control-config.env").write_text("".join(
        "export " + name + "=" + shlex.quote(value) + "\n" for name, value in values.items()))
    print(json.dumps({"parameters": decision["parameters"], "config_sha256": digest,
                      "layout_sha256": decision["layout_lookup_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
