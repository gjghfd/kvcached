import copy
import importlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context, copy_context
from decimal import Decimal, localcontext
from pathlib import Path
from threading import Barrier

import pytest


@pytest.fixture
def portability(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    module = importlib.import_module("risk_certificate_portability")
    try:
        yield module
    finally:
        module.uninstall_verifier()


@pytest.fixture
def risk_case():
    probability = -math.expm1(-.013 * 100)
    cost = probability * 20
    source = {"guards": [{"model": 1, "protected_rank": None,
                           "arrival_probability": probability, "unprotected_cost_s": cost}],
              "risk_cost_s": cost, "snapshot": {"capacity_gib": [10, 10], "movable": []}}
    logged = {"guards": copy.deepcopy(source["guards"]), "headroom_equivalence_proof": {
        "original_indices": [0], "representative_index": 0,
        "members": [{"original_index": 0, "risk_cost_s": math.nextafter(cost, math.inf)}]}}
    return [source], [logged], [{"rate_rps": .001}, {"rate_rps": .013}], 100, [10, 20]


def test_one_ulp_risk_reconciles_without_touching_geometry(portability, risk_case):
    before = copy.deepcopy(risk_case[0])
    adjusted, receipt = portability.reconcile(*risk_case)
    assert adjusted[0]["snapshot"] == before[0]["snapshot"]
    assert adjusted[0]["risk_cost_s"] == risk_case[1][0]["headroom_equivalence_proof"]["members"][0]["risk_cost_s"]
    assert risk_case[0] == before
    assert receipt["decimal_precision"] == 80
    assert len(receipt["changes"]) == 1


@pytest.mark.parametrize("target", ["member", "probability", "cost", "source"])
def test_real_risk_tampering_is_rejected(portability, risk_case, target):
    if target == "member":
        risk_case[1][0]["headroom_equivalence_proof"]["members"][0]["risk_cost_s"] += .0001
    elif target == "probability":
        risk_case[1][0]["guards"][0]["arrival_probability"] += .0001
    elif target == "cost":
        risk_case[1][0]["guards"][0]["unprotected_cost_s"] += .0001
    else:
        risk_case[0][0]["risk_cost_s"] += .0001
    with pytest.raises(ValueError, match="High-precision"):
        portability.reconcile(*risk_case)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, "1", True])
def test_nonfinite_signed_and_non_numeric_values_rejected(portability, value):
    with pytest.raises(ValueError):
        portability.bounded_risk(value, Decimal(1), "test")


def test_missing_member_proof_is_fatal(portability, risk_case):
    risk_case[1][0]["headroom_equivalence_proof"]["members"] = []
    with pytest.raises(ValueError, match="Missing"):
        portability.reconcile(*risk_case)


@pytest.fixture
def stage_binding(tmp_path):
    return {"root": str(tmp_path.resolve()), "job": 279153, "stage_id": "native-2026103101-N4",
            "phase": "native-2026103101-N4", "trace_sha256": "a" * 64,
            "raw_manifest_sha256": "b" * 64, "archive_report_sha256": "c" * 64,
            "log_sha256": {"out": "d" * 64, "err": "e" * 64},
            "host_source_manifest_sha256": "f" * 64}


@pytest.fixture
def strict_case(portability):
    from prune_headroom_scenarios import prune_headroom_scenarios

    snapshot = {"new_request_count": 0, "new_gib": 0, "capacity_gib": [10., 10.],
                "initial_gib": [1., 0.], "movable": [{"id": "request", "model": 1,
                "source": 0, "memory_gib": 1.}], "work_segments": [{"uncertainty_request_s": 123.}]}
    rank = {"uniform_page_geometry": True, "block_size": 16, "page_block_capacity": 16,
            "page_bytes": 1048576}
    observed = [[copy.deepcopy(rank), copy.deepcopy(rank)] for model in range(2)]
    profiles, horizon, cold = [{"rate_rps": .001, "tokens": 16}, {"rate_rps": .013, "tokens": 16}], 100, [10, 20]
    original = portability.certificates.verify_workaware_robust_rebalance.headroom_modes(
        snapshot, observed, profiles, horizon, cold)
    pruned = prune_headroom_scenarios(original)
    logged = [{**entry, "snapshot": {**entry["snapshot"], "placement_risk_cost_s": entry["risk_cost_s"]},
               "headroom_equivalence_proof": proof, "original_scenario_count": pruned["proof"]["input_count"],
               "pruned_scenario_count": pruned["proof"]["output_count"]}
              for entry, proof in zip(pruned["scenarios"], pruned["proof"]["groups"])]
    portability.certificates.verify_certificates(original, logged)
    return {"snapshot": snapshot, "observed": observed, "profiles": profiles, "horizon": horizon,
            "cold": cold, "original": original, "logged": logged}


def shifted_certificate(case, message="retained"):
    logged = copy.deepcopy(case["logged"])
    if message == "retained":
        for entry in logged:
            for guard in entry["guards"]:
                for name in ("arrival_probability", "unprotected_cost_s"):
                    guard[name] = math.nextafter(guard[name], -math.inf)
    else:
        member = next(member for entry in logged for member in entry["headroom_equivalence_proof"]["members"]
                      if member["risk_cost_s"] > 0)
        member["risk_cost_s"] = math.nextafter(member["risk_cost_s"], math.inf)
    return logged


def reconstruct(portability, case):
    return portability.certificates.verify_workaware_robust_rebalance.headroom_modes(
        case["snapshot"], case["observed"], case["profiles"], case["horizon"], case["cold"])


@pytest.mark.parametrize("message", ["retained", "rational"])
def test_real_strict_certificate_messages_reconcile_then_run_original_gate(
        portability, strict_case, stage_binding, tmp_path, monkeypatch, message):
    strict = portability.certificates.verify_certificates
    logged = shifted_certificate(strict_case, message)
    expected_message = ("Retained snapshot changed physical capacities, ownership, uncertainty or risk"
                        if message == "retained" else "Independent rational usage/risk certificate reconstruction differs")
    with pytest.raises(ValueError, match=expected_message):
        strict(strict_case["original"], logged)
    calls = []

    def strict_spy(original, retained):
        calls.append(copy.deepcopy(original))
        return strict(original, retained)

    monkeypatch.setattr(portability.certificates, "verify_certificates", strict_spy)
    receipt_path = tmp_path / "receipt.jsonl"
    before = copy.deepcopy((strict_case, logged))
    with portability.install_verifier(receipt_path), portability.verification_scope(stage_binding):
        original = reconstruct(portability, strict_case)
        result = portability.certificates.verify_certificates(original, logged)
        assert original == strict_case["original"]
    assert len(calls) == 2
    assert len(result) == len(logged)
    assert (strict_case, logged) == before
    for source, adjusted in zip(calls[0], calls[1]):
        assert source["snapshot"] == adjusted["snapshot"]
    receipt = json.loads(receipt_path.read_text())
    assert receipt["binding"] == stage_binding
    assert receipt["strict_failure_message"] == expected_message
    assert receipt["strict_fraction_certificate_verified"] is True
    assert receipt["decimal_precision"] == 80
    assert receipt["maximum_rounding_bound_ulps"] == 4
    assert receipt["source_at_import_sha256"] == portability.SOURCE_AT_IMPORT_SHA256


@pytest.mark.parametrize("tamper", ["capacity", "capacity_ulp", "ownership", "uncertainty", "fraction"])
def test_true_geometry_and_fraction_tamper_still_fail_original_gate(
        portability, strict_case, stage_binding, tmp_path, tamper):
    logged = shifted_certificate(strict_case)
    if tamper == "capacity":
        logged[0]["snapshot"]["capacity_gib"][0] += 1
    elif tamper == "capacity_ulp":
        logged[0]["snapshot"]["capacity_gib"][0] = math.nextafter(10., math.inf)
    elif tamper == "ownership":
        logged[0]["snapshot"]["movable"][0]["source"] = 1
    elif tamper == "uncertainty":
        logged[0]["snapshot"]["work_segments"][0]["uncertainty_request_s"] = 0.
    else:
        logged[0]["headroom_equivalence_proof"]["members"][0]["rank_proofs"][0]["exact_upper"]["numerator"] += 1
    receipt_path = tmp_path / "receipt.jsonl"
    before = copy.deepcopy(logged)
    with portability.install_verifier(receipt_path), portability.verification_scope(stage_binding):
        original = reconstruct(portability, strict_case)
        with pytest.raises(ValueError):
            portability.certificates.verify_certificates(original, logged)
        assert portability._pending_inputs.get() is None
    assert logged == before
    assert not receipt_path.exists()


@pytest.mark.parametrize("value", [math.nextafter(0., -math.inf), math.nextafter(1., math.inf),
                                   2., True, float("nan"), float("inf")])
def test_probability_has_closed_unit_domain_even_inside_ulp_bound(portability, value):
    with pytest.raises(ValueError):
        portability.bounded_probability(value, Decimal(1 if value > 0 else 0), "probability")


@pytest.mark.parametrize("bank", [0, 1])
@pytest.mark.parametrize("model", [True, False, 0., 1., -1, 2, "1"])
def test_both_guard_banks_require_strict_model_indices(portability, risk_case, bank, model):
    risk_case[bank][0]["guards"][0]["model"] = model
    with pytest.raises(ValueError, match="strict two-model"):
        portability.reconcile(*risk_case)


@pytest.mark.parametrize("field,value", [("rate", float("inf")), ("rate", float("nan")), ("rate", True),
                                         ("rate", 10 ** 400), ("horizon", -.1), ("horizon", True),
                                         ("horizon", float("inf")), ("cold", float("nan")), ("cold", -1)])
def test_all_math_inputs_have_finite_nonnegative_domains(portability, risk_case, field, value):
    if field == "rate":
        risk_case[2][1]["rate_rps"] = value
    elif field == "horizon":
        risk_case = (*risk_case[:3], value, risk_case[4])
    else:
        risk_case[4][1] = value
    with pytest.raises(ValueError):
        portability.reconcile(*risk_case)


def test_float_product_overflow_is_rejected(portability):
    with pytest.raises(ValueError):
        portability.risk_primitives([{"rate_rps": 1e308}, {"rate_rps": 0}], 1e308, [1, 1])


@pytest.mark.parametrize("exact", [Decimal("NaN"), Decimal("Infinity"), Decimal(-1), Decimal("1e309")])
def test_decimal_reference_and_float_nearest_must_be_finite(portability, exact):
    with pytest.raises(ValueError):
        portability.bounded_risk(0., exact, "reference")


@pytest.mark.parametrize("product", [0., 1e-30, 1e-90, 1e-200, math.nextafter(0., math.inf), 1., 2., 200., 1e308])
def test_stable_decimal_probability_including_tiny_and_saturated_products(portability, product):
    actual = portability.decimal_probability(product)
    if product < 200:
        with localcontext() as context:
            context.prec = 600
            expected = Decimal(1) - (-Decimal.from_float(product)).exp()
        assert float(actual) == float(expected)
    else:
        assert actual == 1
    assert actual.is_finite() and 0 <= actual <= 1
    if product > 0:
        assert actual > 0


def test_tiny_rate_horizon_reconciles_without_one_minus_exp_cancellation(portability, risk_case):
    probability = -math.expm1(-1e-200)
    cost = probability * 20
    risk_case[2][1]["rate_rps"] = 1e-200
    risk_case = (*risk_case[:3], 1., risk_case[4])
    for bank in risk_case[:2]:
        bank[0]["guards"][0].update({"arrival_probability": probability, "unprotected_cost_s": cost})
    risk_case[0][0]["risk_cost_s"] = cost
    risk_case[1][0]["headroom_equivalence_proof"]["members"][0]["risk_cost_s"] = math.nextafter(cost, math.inf)
    adjusted, receipt = portability.reconcile(*risk_case)
    assert adjusted[0]["risk_cost_s"] > 0
    assert Decimal(receipt["changes"][0]["high_precision_risk_s"]) > 0


def test_risk_reference_preserves_binary_float_product_semantics(portability):
    with localcontext() as context:
        context.prec = 100
        binary_product = Decimal.from_float(.1 * .1)
        separate_product = Decimal.from_float(.1) * Decimal.from_float(.1)
        expected = Decimal(1) - (-binary_product).exp()
        different = Decimal(1) - (-separate_product).exp()
    actual = portability.risk_primitives([{"rate_rps": .1}, {"rate_rps": 0}], .1, [1, 0])[0][0]
    assert abs(actual - expected) < Decimal("1e-78")
    assert abs(actual - different) > Decimal("1e-20")


@pytest.mark.parametrize("nearest", [0., math.nextafter(0., math.inf), 1.])
def test_four_ulp_bound_is_not_widened(portability, nearest):
    value = nearest
    for step in range(4):
        value = math.nextafter(value, math.inf)
    portability.bounded_risk(value, Decimal.from_float(nearest), "four")
    with pytest.raises(ValueError, match="High-precision"):
        portability.bounded_risk(math.nextafter(value, math.inf), Decimal.from_float(nearest), "five")


@pytest.mark.parametrize("mode", ["outside", "exit", "strict_success", "wrong_object", "new_modes", "mutation"])
def test_missing_consumed_stale_or_cross_call_inputs_cannot_recover(
        portability, strict_case, stage_binding, tmp_path, mode):
    logged = shifted_certificate(strict_case)
    receipt_path = tmp_path / "receipt.jsonl"
    with portability.install_verifier(receipt_path):
        if mode == "outside":
            original = reconstruct(portability, strict_case)
            with portability.verification_scope(stage_binding), pytest.raises(ValueError):
                portability.certificates.verify_certificates(original, logged)
        elif mode == "exit":
            with portability.verification_scope(stage_binding):
                original = reconstruct(portability, strict_case)
            with pytest.raises(ValueError):
                portability.certificates.verify_certificates(original, logged)
        else:
            with portability.verification_scope(stage_binding):
                original = reconstruct(portability, strict_case)
                if mode == "strict_success":
                    portability.certificates.verify_certificates(original, strict_case["logged"])
                elif mode == "wrong_object":
                    with pytest.raises(ValueError):
                        portability.certificates.verify_certificates(copy.deepcopy(original), logged)
                elif mode == "new_modes":
                    newer = reconstruct(portability, strict_case)
                    with pytest.raises(ValueError):
                        portability.certificates.verify_certificates(original, logged)
                    original = newer
                else:
                    original[0]["guards"][0]["arrival_probability"] = math.nextafter(
                        original[0]["guards"][0]["arrival_probability"], math.inf)
                with pytest.raises(ValueError):
                    portability.certificates.verify_certificates(original, logged)
                assert portability._pending_inputs.get() is None
    assert not receipt_path.exists()


def test_recovered_certificate_consumes_inputs_exactly_once_across_copied_contexts(
        portability, strict_case, stage_binding, tmp_path):
    receipt_path = tmp_path / "receipt.jsonl"
    logged = shifted_certificate(strict_case)
    with portability.install_verifier(receipt_path), portability.verification_scope(stage_binding):
        original = reconstruct(portability, strict_case)
        inherited = copy_context()
        portability.certificates.verify_certificates(original, logged)
        with pytest.raises(ValueError):
            inherited.run(portability.certificates.verify_certificates, original, logged)
        with pytest.raises(ValueError):
            portability.certificates.verify_certificates(original, logged)
    assert len(receipt_path.read_text().splitlines()) == 1


def test_nested_context_restores_binding_but_never_restores_stale_inputs(
        portability, strict_case, stage_binding, tmp_path):
    receipt_path = tmp_path / "receipt.jsonl"
    inner = {**stage_binding, "job": 279154, "stage_id": "native-N3", "phase": "native-N3"}
    logged = shifted_certificate(strict_case)
    with portability.install_verifier(receipt_path), portability.verification_scope(stage_binding):
        stale = reconstruct(portability, strict_case)
        with portability.verification_scope(inner):
            with pytest.raises(ValueError):
                portability.certificates.verify_certificates(stale, logged)
            portability.certificates.verify_certificates(reconstruct(portability, strict_case), logged)
        with pytest.raises(ValueError):
            portability.certificates.verify_certificates(stale, logged)
        portability.certificates.verify_certificates(reconstruct(portability, strict_case), logged)
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    assert [receipt["binding"] for receipt in receipts] == [inner, stage_binding]
    assert portability._active_scope.get() is None and portability._pending_inputs.get() is None


def test_scope_and_installation_cleanup_after_exceptions(portability, strict_case, stage_binding, tmp_path):
    backend = portability.certificates.verify_workaware_robust_rebalance
    before = backend.headroom_modes, portability.certificates.verify_certificates
    with pytest.raises(RuntimeError, match="scope failure"):
        with portability.install_verifier(tmp_path / "receipt.jsonl"), portability.verification_scope(stage_binding):
            reconstruct(portability, strict_case)
            raise RuntimeError("scope failure")
    assert (backend.headroom_modes, portability.certificates.verify_certificates) == before
    assert portability._active_scope.get() is None and portability._pending_inputs.get() is None


def test_duplicate_install_rejected_and_explicit_uninstall_allows_new_epoch(portability, tmp_path):
    backend = portability.certificates.verify_workaware_robust_rebalance
    before = backend.headroom_modes, portability.certificates.verify_certificates
    first = portability.install_verifier(tmp_path / "first.jsonl")
    with pytest.raises(RuntimeError, match="already installed"):
        portability.install_verifier(tmp_path / "duplicate.jsonl")
    portability.uninstall_verifier()
    assert (backend.headroom_modes, portability.certificates.verify_certificates) == before
    with portability.install_verifier(tmp_path / "second.jsonl"):
        first.uninstall()
        assert backend.headroom_modes is not before[0]
    assert (backend.headroom_modes, portability.certificates.verify_certificates) == before


def test_receipt_binds_canonical_inputs_and_source_at_import_not_later_disk_contents(
        portability, strict_case, stage_binding, tmp_path, monkeypatch):
    inputs = copy.deepcopy({name: strict_case[name] for name in ("profiles", "horizon", "cold")})
    binding_before = copy.deepcopy(stage_binding)
    source_at_import = portability.SOURCE_AT_IMPORT_SHA256
    receipt_path = tmp_path / "receipt.jsonl"
    with portability.install_verifier(receipt_path), portability.verification_scope(stage_binding) as yielded:
        original = reconstruct(portability, strict_case)
        strict_case["profiles"][0]["rate_rps"] = 999.
        strict_case["cold"][0] = 999.
        stage_binding["log_sha256"]["out"] = "0" * 64
        yielded["job"] = 1
        monkeypatch.setattr(portability, "__file__", str(tmp_path / "nonexistent-new-source.py"))
        portability.certificates.verify_certificates(original, shifted_certificate(strict_case))
    receipt = json.loads(receipt_path.read_text())
    assert receipt["binding"] == binding_before
    assert receipt["inputs"] == inputs
    assert receipt["inputs_sha256"] == portability.canonical_sha256(inputs)
    reordered = {name: inputs[name] for name in ("cold", "horizon", "profiles")}
    assert portability.canonical_sha256(reordered) == receipt["inputs_sha256"]
    assert receipt["source_at_import_sha256"] == receipt["source_sha256"] == source_at_import
    assert portability.canonical_sha256({**inputs, "horizon": 101}) != receipt["inputs_sha256"]


@pytest.mark.parametrize("field,value", [("job", True), ("job", 1.), ("root", "relative"),
                                         ("stage_id", "../stage"), ("phase", "bad/phase"),
                                         ("trace_sha256", "not-a-sha"), ("raw_manifest_sha256", None),
                                         ("archive_report_sha256", "A" * 64),
                                         ("host_source_manifest_sha256", "1" * 63),
                                         ("log_sha256", {"out": "d" * 64})])
def test_scope_binding_rejects_unbound_or_malformed_evidence(portability, stage_binding, field, value):
    stage_binding[field] = value
    with pytest.raises(ValueError):
        with portability.verification_scope(stage_binding):
            pytest.fail("Malformed binding must not enter a scope")


def test_only_the_two_known_messages_can_enter_reconcile(
        portability, strict_case, stage_binding, tmp_path, monkeypatch):
    def unrelated_failure(original, logged):
        raise ValueError("Different certificate failure")

    def forbidden_reconcile(*args):
        pytest.fail("Unknown certificate failure must not reconcile")

    monkeypatch.setattr(portability.certificates, "verify_certificates", unrelated_failure)
    monkeypatch.setattr(portability, "reconcile", forbidden_reconcile)
    with portability.install_verifier(tmp_path / "receipt.jsonl"), portability.verification_scope(stage_binding):
        with pytest.raises(ValueError, match="Different certificate failure"):
            portability.certificates.verify_certificates(reconstruct(portability, strict_case), strict_case["logged"])
        assert portability._pending_inputs.get() is None


def test_contextvars_isolate_parallel_certificate_inputs(portability, strict_case, stage_binding, tmp_path):
    barrier = Barrier(2)
    receipt_path = tmp_path / "receipt.jsonl"

    def verify_job(job):
        binding = {**stage_binding, "job": job}
        with portability.verification_scope(binding):
            original = reconstruct(portability, strict_case)
            barrier.wait(timeout=10)
            return portability.certificates.verify_certificates(original, shifted_certificate(strict_case))

    with portability.install_verifier(receipt_path), ThreadPoolExecutor(max_workers=2) as workers:
        assert all(workers.map(verify_job, (279153, 279154)))
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    assert {receipt["binding"]["job"] for receipt in receipts} == {279153, 279154}
    assert portability._active_scope.get() is None and portability._pending_inputs.get() is None


def test_fresh_context_cannot_consume_another_scopes_pending_inputs(
        portability, strict_case, stage_binding, tmp_path):
    receipt_path = tmp_path / "receipt.jsonl"
    with portability.install_verifier(receipt_path), portability.verification_scope(stage_binding):
        original = reconstruct(portability, strict_case)
        with pytest.raises(ValueError):
            Context().run(portability.certificates.verify_certificates, original, shifted_certificate(strict_case))
        portability.certificates.verify_certificates(original, shifted_certificate(strict_case))
    assert len(receipt_path.read_text().splitlines()) == 1


def test_high_precision_dominance_is_not_relaxed(portability, strict_case):
    logged = shifted_certificate(strict_case)
    proof = logged[0]["headroom_equivalence_proof"]
    proof["representative_index"] = next(member["original_index"] for member in proof["members"]
                                         if member["risk_cost_s"] > 0)
    with pytest.raises(ValueError, match="constant-risk dominance"):
        portability.reconcile(strict_case["original"], logged, strict_case["profiles"],
                              strict_case["horizon"], strict_case["cold"])


def test_failed_modes_clear_previous_inputs(portability, strict_case, stage_binding, tmp_path):
    receipt_path = tmp_path / "receipt.jsonl"
    with portability.install_verifier(receipt_path), portability.verification_scope(stage_binding):
        original = reconstruct(portability, strict_case)
        invalid = copy.deepcopy(strict_case)
        invalid["profiles"][0]["rate_rps"] = float("nan")
        with pytest.raises(ValueError):
            reconstruct(portability, invalid)
        with pytest.raises(ValueError):
            portability.certificates.verify_certificates(original, shifted_certificate(strict_case))
        assert portability._pending_inputs.get() is None
    assert not receipt_path.exists()


def test_uninstall_invalidates_pending_inputs_in_copied_contexts(
        portability, strict_case, stage_binding, tmp_path):
    first = portability.install_verifier(tmp_path / "first.jsonl")
    with portability.verification_scope(stage_binding):
        original = reconstruct(portability, strict_case)
        inherited = copy_context()
        first.uninstall()
        with portability.install_verifier(tmp_path / "second.jsonl"):
            with pytest.raises(ValueError):
                inherited.run(portability.certificates.verify_certificates, original, shifted_certificate(strict_case))
    assert not (tmp_path / "first.jsonl").exists() and not (tmp_path / "second.jsonl").exists()


def test_probability_reference_must_also_be_in_closed_unit_domain(portability):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        portability.bounded_probability(1., Decimal("1.00000000000000000001"), "reference")
