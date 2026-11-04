import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def cost(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("workaware_cost")


def anchor():
    return {"coefficients": {"base_s": .01, "graph_batch_s": .001, "attention_token_s": .000001},
            "status": "raw_verified_workaware_cadence_fit", "source_verification": "raw_verified",
            "scope": "one_active_tenant_other_resident_idle", "capture_sizes": [1, 8, 16, 32, 64, 128, 256],
            "residual_bound_s": .002, "max_population": 9, "max_context_tokens": 20000}


def samples():
    return [{"counts": [population, 0], "context_tokens": [context, 0],
             "cadence_s": .018 + .000001 * context}
            for population in (5, 6, 7) for context in (1000, 2000)]


def test_sparse_constant_graph_cannot_refit_three_global_parameters(cost):
    with pytest.raises(ValueError, match="not identifiable"):
        cost.fit_cadence(samples())


def test_anchoring_preserves_coefficients_and_never_reduces_residual_guard(cost):
    verified = anchor()
    before = copy.deepcopy(verified)
    result = cost.anchored_cadence(samples(), verified)
    assert result["coefficients"] == verified["coefficients"]
    assert result["residual_bound_s"] == .002
    assert result["samples"] == 6 and verified == before
    assert result["coefficient_scope"].startswith("fixed_independently_verified")


def test_anchoring_retains_new_adverse_residuals(cost):
    measured = samples()
    measured[0]["cadence_s"] += .01
    result = cost.anchored_cadence(measured, anchor())
    assert result["residual_bound_s"] == pytest.approx(.01)


@pytest.mark.parametrize("field,value", (("status", "diagnostic"), ("source_verification", "unchecked"),
                                       ("scope", "joint_active_unverified")))
def test_anchor_requires_independent_raw_verification(cost, field, value):
    verified = anchor()
    verified[field] = value
    with pytest.raises(ValueError, match="raw-verified"):
        cost.anchored_cadence(samples(), verified)
