import copy
import importlib
import itertools
import math
from fractions import Fraction
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("prune_headroom_scenarios")


def snapshot():
    return {
        "counts": [[0, 0], [2, 0]], "initial_gib": [3.1875, .1875], "capacity_gib": [10., 10.],
        "final_capacity_gib": [10., 10.], "new_request_count": 0, "new_gib": 0.,
        "movable": [{"id": "first", "model": 1, "source": 0, "memory_gib": 1.59375, "move_cost_s": .2},
                    {"id": "second", "model": 1, "source": 0, "memory_gib": 1.59375, "move_cost_s": .3}],
        "source_pages": [{"id": "page", "rank": 0, "memory_gib": .09375, "owners": ["first"]}],
        "work_segments": [{"population": 2, "uncertainty_request_s": 4.,
                           "request_sources": {"first": 0, "second": 0}}],
        "migration_activation_cost_s": 2., "performance_cost_tables_s": [[0.], [0., 0., 0.]],
    }


def scenario(final=(10., 10.), risk=0., selected_snapshot=None):
    selected = snapshot() if selected_snapshot is None else copy.deepcopy(selected_snapshot)
    selected["final_capacity_gib"] = list(final)
    return {"snapshot": selected, "risk_cost_s": risk, "guards": [{"protected_rank": None}]}


def test_nine_scenarios_reduce_to_three_with_proof_and_original_indices(module):
    capacities = [(10., 10.), (8.40625, 10.), (10., 8.40625),
                  (.625, 10.), (.625, 10.), (.625, 8.40625),
                  (10., .625), (8.40625, .625), (10., .625)]
    risks = [6.9, 1.7, 1.7, 5.1, 0., 0., 5.1, 0., 0.]
    entries = [scenario(final, risk) for final, risk in zip(capacities, risks)]
    before = copy.deepcopy(entries)
    result = module.prune_headroom_scenarios(entries)
    assert entries == before
    assert result["proof"]["input_count"] == 9 and result["proof"]["output_count"] == 3
    groups = result["proof"]["groups"]
    assert [group["original_indices"] for group in groups] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert [group["representative_index"] for group in groups] == [1, 4, 7]
    assert [entry["risk_cost_s"] for entry in result["scenarios"]] == [1.7, 0., 0.]
    assert [entry["snapshot"]["final_capacity_gib"] for entry in result["scenarios"]] == [
        [10., 10.], [.625, 10.], [10., .625]]
    first = groups[0]["members"][0]["rank_proofs"]
    assert [proof["final_usage_upper_gib"] for proof in first] == [3.1875, 3.375]
    assert [proof["inbound_movable_ids"] for proof in first] == [[], ["first", "second"]]
    assert all(group["all_other_snapshot_parameters_identical"] for group in groups)


def test_truly_binding_final_constraints_do_not_merge(module):
    result = module.prune_headroom_scenarios([scenario((10., .5)), scenario((10., .75))])
    assert len(result["scenarios"]) == 2
    assert [entry["snapshot"]["final_capacity_gib"][1] for entry in result["scenarios"]] == [.5, .75]


@pytest.mark.parametrize("field", ["source_pages", "work_segments", "movable", "counts",
                                  "migration_activation_cost_s", "performance_cost_tables_s", "capacity_gib"])
def test_other_snapshot_parameter_differences_do_not_merge(module, field):
    first, second = scenario(), scenario()
    changed = second["snapshot"]
    if field == "source_pages":
        changed[field][0]["owners"].append("second")
    elif field == "work_segments":
        changed[field][0]["uncertainty_request_s"] = 5.
    elif field == "movable":
        changed[field][0]["move_cost_s"] = .4
    elif field == "counts":
        changed[field][1] = [3, 0]
    elif field == "migration_activation_cost_s":
        changed[field] = 3.
    elif field == "performance_cost_tables_s":
        changed[field][1][0] = 1.
    else:
        changed[field][0] = 11.
    assert len(module.prune_headroom_scenarios([first, second])["scenarios"]) == 2


def test_other_snapshot_parameter_numeric_types_are_not_silently_equated(module):
    first, second = scenario(), scenario()
    second["snapshot"]["migration_activation_cost_s"] = 2
    assert len(module.prune_headroom_scenarios([first, second])["scenarios"]) == 2


def test_dictionary_order_is_not_a_parameter_difference(module):
    first, second = scenario(), scenario()
    second["snapshot"] = dict(reversed(list(second["snapshot"].items())))
    assert len(module.prune_headroom_scenarios([first, second])["scenarios"]) == 1


def test_minimum_risk_representative_preserves_metadata_and_stable_tie(module):
    entries = [scenario(risk=3.), scenario(risk=1.), scenario(risk=1.)]
    for index, entry in enumerate(entries):
        entry["guards"][0]["protected_rank"] = index
    result = module.prune_headroom_scenarios(entries)
    assert result["scenarios"][0]["risk_cost_s"] == 1.
    assert result["scenarios"][0]["guards"] == entries[1]["guards"]
    assert result["proof"]["groups"][0]["representative_index"] == 1
    assert result["proof"]["groups"][0]["original_indices"] == [0, 1, 2]


def test_equal_risk_distinct_groups_follow_original_representative_order(module):
    entries = [scenario((10., .625), 0.), scenario((.625, 10.), 0.), scenario((10., 10.), 0.)]
    result = module.prune_headroom_scenarios(entries)
    assert [group["representative_index"] for group in result["proof"]["groups"]] == [0, 1, 2]


def test_embedded_snapshot_risk_is_checked_and_stripped_from_comparison(module):
    first, second, third = scenario(risk=4.), scenario(risk=1.), scenario(risk=2.)
    first["snapshot"]["placement_risk_cost_s"] = 4.
    second["snapshot"]["placement_risk_cost_s"] = 1.
    result = module.prune_headroom_scenarios([first, second, third])
    assert len(result["scenarios"]) == 1
    assert result["scenarios"][0]["snapshot"]["placement_risk_cost_s"] == 1.
    assert [member["snapshot_risk_checked"] for member in result["proof"]["groups"][0]["members"]] == [
        True, True, False]


def test_embedded_risk_mismatch_is_rejected(module):
    entry = scenario(risk=1.)
    entry["snapshot"]["placement_risk_cost_s"] = 2.
    with pytest.raises(ValueError, match="differs"):
        module.prune_headroom_scenarios([entry])


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["risk", "final", "initial", "memory", "uncertainty", "metadata"])
def test_nonfinite_values_are_rejected_everywhere(module, field, value):
    entry = scenario()
    if field == "risk":
        entry["risk_cost_s"] = value
    elif field == "final":
        entry["snapshot"]["final_capacity_gib"][0] = value
    elif field == "initial":
        entry["snapshot"]["initial_gib"][0] = value
    elif field == "memory":
        entry["snapshot"]["movable"][0]["memory_gib"] = value
    elif field == "uncertainty":
        entry["snapshot"]["work_segments"][0]["uncertainty_request_s"] = value
    else:
        entry["metadata"] = value
    with pytest.raises(ValueError, match="finite"):
        module.prune_headroom_scenarios([entry])


@pytest.mark.parametrize("fault", ["entry", "snapshot", "risk", "missing_final", "rank_count", "negative_final",
                                  "final_exceeds_copy_cap", "boolean_capacity", "movable", "source", "model",
                                  "missing_id", "missing_memory", "negative_memory", "duplicate_id", "new_count",
                                  "missing_new_count", "incoming_memory", "missing_new_gib", "unknown_object"])
def test_strict_structure_validation(module, fault):
    entry = scenario()
    selected = entry["snapshot"]
    if fault == "entry":
        entry = []
    elif fault == "snapshot":
        entry["snapshot"] = []
    elif fault == "risk":
        entry["risk_cost_s"] = True
    elif fault == "missing_final":
        selected.pop("final_capacity_gib")
    elif fault == "rank_count":
        selected["initial_gib"] = [0.]
    elif fault == "negative_final":
        selected["final_capacity_gib"][0] = -1.
    elif fault == "final_exceeds_copy_cap":
        selected["final_capacity_gib"][0] = 11.
    elif fault == "boolean_capacity":
        selected["capacity_gib"][0] = True
    elif fault == "movable":
        selected["movable"] = {}
    elif fault == "source":
        selected["movable"][0]["source"] = 2
    elif fault == "model":
        selected["movable"][0]["model"] = False
    elif fault == "missing_id":
        selected["movable"][0].pop("id")
    elif fault == "missing_memory":
        selected["movable"][0].pop("memory_gib")
    elif fault == "negative_memory":
        selected["movable"][0]["memory_gib"] = -1.
    elif fault == "duplicate_id":
        selected["movable"][1]["id"] = "first"
    elif fault == "new_count":
        selected["new_request_count"] = 1
    elif fault == "missing_new_count":
        selected.pop("new_request_count")
    elif fault == "incoming_memory":
        selected["new_gib"] = .1
    elif fault == "missing_new_gib":
        selected.pop("new_gib")
    else:
        selected["unknown"] = object()
    with pytest.raises(ValueError):
        module.prune_headroom_scenarios([entry])


def test_nonstring_dictionary_keys_and_cycles_are_rejected(module):
    entry = scenario()
    entry["snapshot"][5] = "invalid"
    with pytest.raises(ValueError, match="keys"):
        module.prune_headroom_scenarios([entry])
    cyclic = scenario()
    cyclic["snapshot"]["cycle"] = cyclic
    with pytest.raises(ValueError, match="cycles"):
        module.prune_headroom_scenarios([cyclic])


def test_empty_list_is_valid_but_other_containers_are_rejected(module):
    result = module.prune_headroom_scenarios([])
    assert result["scenarios"] == [] and result["proof"]["groups"] == []
    with pytest.raises(ValueError, match="list"):
        module.prune_headroom_scenarios(tuple())


def test_incoming_movable_upper_bound_does_not_subtract_outgoing_releases(module):
    selected = snapshot()
    selected["initial_gib"] = [.8, .5]
    selected["movable"] = [
        {"id": "outgoing", "model": 1, "source": 0, "memory_gib": .4},
        {"id": "incoming", "model": 1, "source": 1, "memory_gib": .2}]
    entry = scenario((.9, .9), selected_snapshot=selected)
    result = module.prune_headroom_scenarios([entry])
    normalized = result["scenarios"][0]["snapshot"]
    assert normalized["final_capacity_gib"][0] == .9
    proofs = result["proof"]["groups"][0]["members"][0]["rank_proofs"]
    assert proofs[0]["final_usage_upper_gib"] >= 1.
    assert proofs[0]["inbound_movable_ids"] == ["incoming"]
    assert not proofs[0]["nonbinding"]
    assert all(proof["source_releases_subtracted"] is False for proof in proofs)


def test_exact_bound_equality_normalizes_without_a_tolerance(module):
    entries = [scenario((3.1875, 3.375)), scenario()]
    result = module.prune_headroom_scenarios(entries)
    assert len(result["scenarios"]) == 1
    assert result["scenarios"][0]["snapshot"]["final_capacity_gib"] == [10., 10.]


def test_float_addition_rounding_down_cannot_prove_nonbinding(module):
    selected = snapshot()
    selected["initial_gib"] = [1., 0.]
    selected["capacity_gib"] = [2., 2.]
    selected["movable"] = [{"id": "tiny", "model": 1, "source": 1, "memory_gib": 2.**-54}]
    entries = [scenario((1., 2.), selected_snapshot=selected), scenario((2., 2.), selected_snapshot=selected)]
    assert 1. + 2.**-54 == 1.
    result = module.prune_headroom_scenarios(entries)
    assert len(result["scenarios"]) == 2
    proof = result["proof"]["groups"][0]["members"][0]["rank_proofs"][0]
    assert not proof["nonbinding"]
    assert proof["final_usage_upper_gib"] == math.nextafter(1., math.inf)
    assert Fraction(proof["exact_upper"]["numerator"], proof["exact_upper"]["denominator"]) > 1


def test_finite_inputs_with_unrepresentable_usage_bound_are_rejected(module):
    selected = snapshot()
    selected["initial_gib"] = [1.e308, 0.]
    selected["movable"] = [{"id": "huge", "model": 1, "source": 1, "memory_gib": 1.e308}]
    with pytest.raises(ValueError, match="representable"):
        module.prune_headroom_scenarios([scenario(selected_snapshot=selected)])


def test_requests_source_pages_copy_caps_and_uncertainty_are_preserved_and_detached(module):
    entry = scenario((8.40625, 8.40625))
    before = copy.deepcopy(entry)
    result = module.prune_headroom_scenarios([entry])
    normalized = result["scenarios"][0]["snapshot"]
    assert normalized["final_capacity_gib"] == [10., 10.]
    for name in entry["snapshot"]:
        if name != "final_capacity_gib":
            assert normalized[name] == entry["snapshot"][name]
    normalized["movable"][0]["memory_gib"] = 100.
    normalized["source_pages"][0]["owners"].clear()
    normalized["work_segments"][0]["uncertainty_request_s"] = 0.
    result["proof"]["groups"][0]["members"][0]["rank_proofs"][0]["initial_gib"] = 0.
    assert entry == before


def test_exhaustive_move_subsets_preserve_feasibility_and_minimum_risk(module):
    selected = snapshot()
    selected["initial_gib"] = [1., .5]
    selected["capacity_gib"] = [2., 2.]
    selected["movable"] = [
        {"id": "first", "model": 1, "source": 0, "memory_gib": .5},
        {"id": "second", "model": 1, "source": 1, "memory_gib": .25}]
    entries = [scenario((1.25, 1.), 3., selected), scenario((2., 2.), 1., selected),
               scenario((.75, 2.), 0., selected), scenario((2., .5), 0., selected)]
    result = module.prune_headroom_scenarios(entries)
    for group, normalized in zip(result["proof"]["groups"], result["scenarios"]):
        originals = [entries[index] for index in group["original_indices"]]
        for flags in itertools.product((False, True), repeat=len(selected["movable"])):
            final = [Fraction(value) for value in selected["initial_gib"]]
            peak = final.copy()
            for request, moved in zip(selected["movable"], flags):
                if moved:
                    source = request["source"]
                    destination = 1 - source
                    peak[destination] = max(peak[destination], final[destination] + Fraction(request["memory_gib"]))
                    final[destination] += Fraction(request["memory_gib"])
                    final[source] -= Fraction(request["memory_gib"])
            copy_fits = all(usage <= capacity for usage, capacity in zip(peak, selected["capacity_gib"]))
            feasible = lambda entry: copy_fits and all(
                usage <= limit for usage, limit in zip(final, entry["snapshot"]["final_capacity_gib"]))
            assert all(feasible(original) == feasible(normalized) for original in originals)
            if feasible(normalized):
                assert normalized["risk_cost_s"] == min(original["risk_cost_s"] for original in originals)
