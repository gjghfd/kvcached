import copy
import hashlib
import json
import math
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from pathlib import Path
from threading import Lock

import verify_solver_efficiency as certificates


SOURCE_AT_IMPORT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
STRICT_CERTIFICATE_MESSAGES = frozenset({
    "Independent rational usage/risk certificate reconstruction differs",
    "Retained snapshot changed physical capacities, ownership, uncertainty or risk",
})
_active_scope = ContextVar("risk_portability_scope", default=None)
_pending_inputs = ContextVar("risk_portability_inputs", default=None)
_installation = None
_installation_lock = Lock()


def finite_decimal(value):
    if type(value) not in (int, float):
        raise ValueError("Risk portability requires finite nonnegative scalars")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError("Risk portability requires finite nonnegative scalars") from error
    if not math.isfinite(converted) or value < 0:
        raise ValueError("Risk portability requires finite nonnegative scalars")
    return Decimal.from_float(converted)


def bounded_risk(value, exact, label):
    actual = finite_decimal(value)
    if not isinstance(exact, Decimal) or not exact.is_finite() or exact < 0:
        raise ValueError("High-precision risk reference must remain finite and nonnegative")
    nearest = float(exact)
    if not math.isfinite(nearest):
        raise ValueError("High-precision risk reference must remain finite and nonnegative")
    bound = Decimal.from_float(4 * math.ulp(nearest))
    actual_numerator, actual_denominator = actual.as_integer_ratio()
    exact_numerator, exact_denominator = exact.as_integer_ratio()
    bound_numerator, bound_denominator = bound.as_integer_ratio()
    difference = abs(actual_numerator * exact_denominator - exact_numerator * actual_denominator)
    if difference * bound_denominator > bound_numerator * actual_denominator * exact_denominator:
        raise ValueError("High-precision risk bound rejected " + label)


def bounded_probability(value, exact, label):
    if (finite_decimal(value) > 1 or not isinstance(exact, Decimal)
            or not exact.is_finite() or not 0 <= exact <= 1):
        raise ValueError("Risk arrival probability must be in [0, 1]")
    bounded_risk(value, exact, label)


def risk_model(value):
    if type(value) is not int or value not in (0, 1):
        raise ValueError("Risk model must be a strict two-model integer index")
    return value


def decimal_probability(product):
    value = finite_decimal(product)
    with localcontext() as context:
        context.prec = 80
        if value >= 200:
            return Decimal(1)
        if value > 1:
            return Decimal(1) - (-value).exp()
        term = total = +value
        denominator = 2
        while term:
            term = -term * value / denominator
            updated = total + term
            if updated == total:
                break
            total = updated
            denominator += 1
        return total


def risk_primitives(profiles, horizon, cold_costs):
    if len(profiles) != 2 or len(cold_costs) != 2:
        raise ValueError("Require the actual two-model arrival and cold-cost inputs")
    finite_decimal(horizon)
    numerical = {}
    with localcontext() as context:
        context.prec = 80
        for model, profile in enumerate(profiles):
            rate = profile["rate_rps"]
            finite_decimal(rate)
            cold = finite_decimal(cold_costs[model])
            probability = decimal_probability(rate * horizon)
            numerical[model] = (probability, probability * cold)
    return numerical


def reconcile(original, logged, profiles, horizon, cold_costs):
    numerical = risk_primitives(profiles, horizon, cold_costs)
    with localcontext() as context:
        context.prec = 80
        templates, member_risks = {}, {}
        for scenario in logged:
            for guard in scenario["guards"]:
                model = risk_model(guard["model"])
                probability, cost = numerical[model]
                bounded_probability(guard["arrival_probability"], probability, "logged probability")
                bounded_risk(guard["unprotected_cost_s"], cost, "logged cold cost")
                values = (guard["arrival_probability"], guard["unprotected_cost_s"])
                if model in templates and templates[model] != values:
                    raise ValueError("Logged risk primitives disagree between scenarios")
                templates[model] = values
            for member in scenario["headroom_equivalence_proof"]["members"]:
                index, risk = member["original_index"], member["risk_cost_s"]
                if type(index) is not int or index < 0 or index >= len(original) or index in member_risks:
                    raise ValueError("Require exactly one logged risk for every original scenario")
                member_risks[index] = risk
        if set(member_risks) != set(range(len(original))):
            raise ValueError("Missing original scenario risk evidence")
        adjusted, changes, exact_risks = copy.deepcopy(original), [], {}
        for index, (source, target) in enumerate(zip(original, adjusted)):
            exact_risk = Decimal(0)
            for source_guard, target_guard in zip(source["guards"], target["guards"]):
                model = risk_model(source_guard["model"])
                probability, cost = numerical[model]
                bounded_probability(source_guard["arrival_probability"], probability, "replayed probability")
                bounded_risk(source_guard["unprotected_cost_s"], cost, "replayed cold cost")
                if source_guard["protected_rank"] is None:
                    exact_risk += cost
                if model not in templates:
                    raise ValueError("No logged probability witness for a replayed model")
                target_guard["arrival_probability"], target_guard["unprotected_cost_s"] = templates[model]
            bounded_risk(source["risk_cost_s"], exact_risk, "replayed scenario risk")
            bounded_risk(member_risks[index], exact_risk, "logged scenario risk")
            exact_risks[index] = exact_risk
            target["risk_cost_s"] = member_risks[index]
            if "placement_risk_cost_s" in target["snapshot"]:
                if source["snapshot"]["placement_risk_cost_s"] != source["risk_cost_s"]:
                    raise ValueError("Replayed snapshot and scenario risks differ")
                target["snapshot"]["placement_risk_cost_s"] = member_risks[index]
            if source != target:
                changes.append({"original_index": index, "replayed_risk_s": source["risk_cost_s"],
                                "logged_risk_s": member_risks[index], "high_precision_risk_s": str(exact_risk)})
        for scenario in logged:
            proof = scenario["headroom_equivalence_proof"]
            indices = proof["original_indices"]
            representative = proof["representative_index"]
            if (type(indices) is not list or not indices
                    or any(type(index) is not int or index not in exact_risks for index in indices)
                    or type(representative) is not int or representative not in indices):
                raise ValueError("Invalid original risk scenario indices")
            if exact_risks[representative] != min(exact_risks[index] for index in indices):
                raise ValueError("High-precision constant-risk dominance is not preserved")
    return adjusted, {"scope": "host_libm_risk_primitives_only_geometry_and_fraction_proofs_unchanged",
                      "decimal_precision": 80, "maximum_rounding_bound_ulps": 4, "changes": changes}


def canonical_sha256(value):
    return hashlib.sha256(certificates.exact_payload(value).encode()).hexdigest()


def validated_binding(binding):
    fields = {"root", "job", "stage_id", "phase", "trace_sha256", "raw_manifest_sha256",
              "archive_report_sha256", "log_sha256", "host_source_manifest_sha256"}
    if type(binding) is not dict or set(binding) != fields:
        raise ValueError("Require the explicit stage, archive, logs and host verification binding fields")
    if not isinstance(binding["root"], (str, Path)) or not Path(binding["root"]).is_absolute():
        raise ValueError("Risk verification root must be an absolute path")
    if type(binding["job"]) is not int or binding["job"] <= 0:
        raise ValueError("Risk verification job must be a positive strict integer")
    for name in ("stage_id", "phase"):
        if type(binding[name]) is not str or not re.fullmatch(r"[a-zA-Z0-9_-]+", binding[name]):
            raise ValueError("Invalid risk verification stage or phase")
    for name in ("trace_sha256", "raw_manifest_sha256", "archive_report_sha256", "host_source_manifest_sha256"):
        if type(binding[name]) is not str or not re.fullmatch(r"[0-9a-f]{64}", binding[name]):
            raise ValueError("Risk verification digests must be canonical SHA256 values")
    if (type(binding["log_sha256"]) is not dict or set(binding["log_sha256"]) != {"out", "err"}
            or any(type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in binding["log_sha256"].values())):
        raise ValueError("Risk verification requires both canonical job log SHA256 values")
    return {**copy.deepcopy(binding), "root": str(Path(binding["root"]).resolve())}


@dataclass(frozen=True)
class _Scope:
    binding: dict


@dataclass
class _CertificateInputs:
    installation: object
    scope: _Scope
    original: list
    original_sha256: str
    inputs: dict
    inputs_sha256: str
    consumed: bool = False
    lock: object = field(default_factory=Lock, repr=False)

    def claim(self):
        with self.lock:
            if self.consumed:
                return False
            self.consumed = True
            return True


def consume_inputs():
    pending = _pending_inputs.get()
    _pending_inputs.set(None)
    return pending if pending is not None and pending.claim() else None


@contextmanager
def verification_scope(binding):
    scope = _Scope(validated_binding(binding))
    consume_inputs()
    token = _active_scope.set(scope)
    try:
        yield copy.deepcopy(scope.binding)
    finally:
        consume_inputs()
        _active_scope.reset(token)


class _VerifierInstallation:
    def __init__(self, receipt_path):
        self.receipt_path = Path(receipt_path)
        self.backend = certificates.verify_workaware_robust_rebalance
        self.original_modes = self.backend.headroom_modes
        self.original_certificates = certificates.verify_certificates
        self.receipt_lock = Lock()
        self.active = True

    def __enter__(self):
        if not self.active:
            raise RuntimeError("Risk portability installation has already been uninstalled")
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.uninstall()

    def uninstall(self):
        global _installation
        with _installation_lock:
            if not self.active:
                return
            if (_installation is not self or self.backend.headroom_modes is not self.modes
                    or certificates.verify_certificates is not self.verifier):
                raise RuntimeError("Risk portability hooks changed outside their installation")
            self.backend.headroom_modes = self.original_modes
            certificates.verify_certificates = self.original_certificates
            self.active = False
            _installation = None
            consume_inputs()


def install_verifier(receipt_path):
    global _installation
    with _installation_lock:
        if _installation is not None:
            raise RuntimeError("Risk portability verifier is already installed; uninstall it explicitly")
        installation = _VerifierInstallation(receipt_path)

        def modes(snapshot, observed, profiles, horizon, cold):
            consume_inputs()
            scope = _active_scope.get()
            inputs = copy.deepcopy({"profiles": profiles, "horizon": horizon, "cold": cold}) if scope else None
            if inputs is not None:
                risk_primitives(inputs["profiles"], inputs["horizon"], inputs["cold"])
                inputs_sha256 = canonical_sha256(inputs)
            original = installation.original_modes(snapshot, observed, profiles, horizon, cold)
            if scope is not None and installation.active:
                _pending_inputs.set(_CertificateInputs(installation, scope, original,
                    canonical_sha256(original), inputs, inputs_sha256))
            return original

        def verifier(original, logged):
            pending = consume_inputs()
            scope = _active_scope.get()
            try:
                return installation.original_certificates(original, logged)
            except ValueError as error:
                if (str(error) not in STRICT_CERTIFICATE_MESSAGES or scope is None or pending is None
                        or not installation.active or pending.installation is not installation
                        or pending.scope is not scope or pending.original is not original
                        or pending.original_sha256 != canonical_sha256(original)):
                    raise
                strict_message = str(error)
            inputs = pending.inputs
            adjusted, receipt = reconcile(original, logged, inputs["profiles"], inputs["horizon"], inputs["cold"])
            result = installation.original_certificates(adjusted, logged)
            receipt.update({"strict_fraction_certificate_verified": True,
                            "strict_failure_message": strict_message,
                            "binding": copy.deepcopy(scope.binding),
                            "inputs": copy.deepcopy(inputs), "inputs_sha256": pending.inputs_sha256,
                            "source_at_import_sha256": SOURCE_AT_IMPORT_SHA256,
                            "source_sha256": SOURCE_AT_IMPORT_SHA256,
                            "original_scenarios_sha256": pending.original_sha256,
                            "logged_scenarios_sha256": canonical_sha256(logged)})
            with installation.receipt_lock, installation.receipt_path.open("a") as output:
                output.write(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
            return result

        installation.modes, installation.verifier = modes, verifier
        installation.backend.headroom_modes = modes
        certificates.verify_certificates = verifier
        _installation = installation
        return installation


def uninstall_verifier():
    installation = _installation
    if installation is not None:
        installation.uninstall()
