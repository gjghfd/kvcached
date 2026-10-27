import importlib.util
import itertools
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def reference():
    if not hasattr(pytest.importorskip("scipy.optimize"), "milp"):
        pytest.skip("SciPy with milp is required for the CPU reference")
    path = Path(__file__).resolve().parents[1] / "experiments/dp_opt/ilp_reference.py"
    spec = importlib.util.spec_from_file_location("dp_opt_ilp_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def snapshot():
    path = Path(__file__).resolve().parents[1] / "experiments/dp_opt/docs/ilp-example.json"
    return json.loads(path.read_text())


def exhaustive_objective(snapshot, target, max_moves=None):
    best = None
    requests = snapshot["movable"]
    for bits in itertools.product((0, 1), repeat=len(requests)):
        selected = [request for request, moved in zip(requests, bits) if moved]
        if max_moves is not None and len(selected) > max_moves:
            continue
        for order in itertools.permutations(selected):
            memory = snapshot["initial_gib"].copy()
            moved_ids, released_ids = set(), set()
            reused_pools = {request.get("destination_pool") for request in order}
            counts = [population.copy() for population in snapshot["counts"]]
            counts[0][target] += 1
            valid = True
            for request in order:
                source, destination = request["source"], 1 - request["source"]
                memory[destination] += request["memory_gib"]
                if memory[destination] > snapshot["capacity_gib"][destination] + 1e-9:
                    valid = False
                    break
                moved_ids.add(request["id"])
                if snapshot.get("source_pages") is None:
                    memory[source] -= request["memory_gib"]
                else:
                    for page in snapshot["source_pages"]:
                        if (page["id"] not in released_ids and not page.get("pinned", False)
                                and (page.get("pool") is None or page["pool"] not in reused_pools)
                                and page["owners"] and set(page["owners"]) <= moved_ids):
                            memory[page["rank"]] -= page["memory_gib"]
                            released_ids.add(page["id"])
                counts[request["model"]][source] -= 1
                counts[request["model"]][destination] += 1
            memory[target] += snapshot["new_gib"]
            if not valid or any(used > capacity + 1e-9 for used, capacity in zip(memory, snapshot["capacity_gib"])):
                continue
            tables = snapshot.get("performance_cost_tables_s")
            joint_table = snapshot.get("joint_performance_costs_s")
            if joint_table is not None:
                score = joint_table[counts[0][0]][counts[1][0]]
            elif tables is not None:
                score = sum(table[counts[model][0]] for model, table in enumerate(tables))
            else:
                score = sum(cost * abs(population[0] - population[1])
                            for cost, population in zip(snapshot["imbalance_cost_s"], counts))
            score += sum(request["move_cost_s"] for request in order)
            score += snapshot.get("admission_cost_s", [0, 0])[target]
            best = score if best is None else min(score, best)
    return best


def test_rank_enumeration_matches_exhaustive_search(reference, snapshot):
    result = reference.enumerate_targets(snapshot)
    assert result["status"] == "optimal"
    assert result["winner"]["target"] == 0
    assert set(result["winner"]["moves"]) == {"A0", "B0"}
    assert result["winner"]["final_gib"] == [11, 11]
    for candidate in result["candidates"]:
        assert candidate["objective_s"] == pytest.approx(exhaustive_objective(snapshot, candidate["target"]))
        assert candidate["mip_gap"] == 0


@pytest.mark.parametrize("pinned,unknown,limit", ((False, False, 2), (True, False, 2),
                                                  (False, True, 2), (False, False, 1)))
def test_shared_source_page_requires_every_owner(reference, pinned, unknown, limit):
    snapshot = {"counts": [[0, 1], [2, 0]], "initial_gib": [3, 1], "capacity_gib": [3, 3],
                "new_gib": 1, "performance_cost_tables_s": [[0, 0, 0], [0, 0, 0]],
                "movable": [{"id": name, "model": 1, "source": 0, "memory_gib": 1,
                             "move_cost_s": 0.1} for name in ("B0", "B1")],
                "source_pages": [{"id": "shared", "rank": 0, "memory_gib": 2,
                                  "owners": ["B0", "B1"] + (["immovable"] if unknown else []),
                                  "pinned": pinned}]}
    result = reference.solve_candidate(snapshot, 0, max_moves=limit)
    expected = exhaustive_objective(snapshot, 0, max_moves=limit)
    if expected is None:
        assert result["status"] == "infeasible"
    else:
        assert result["objective_s"] == pytest.approx(expected)
        assert set(result["moves"]) == {"B0", "B1"}
        assert result["released_source_pages"] == ["shared"]
        assert result["final_gib"] == [2, 3]


def test_page_release_not_private_destination_charge(reference):
    snapshot = {"counts": [[1, 0], [0, 1]], "initial_gib": [4, 1], "capacity_gib": [4, 4],
                "new_gib": 2, "performance_cost_tables_s": [[0, 0, 0], [0, 0]],
                "movable": [{"id": "A", "model": 0, "source": 0, "memory_gib": 3, "move_cost_s": 0.1}],
                "source_pages": [{"id": "actual", "rank": 0, "memory_gib": 2, "owners": ["A"]}]}
    result = reference.solve_candidate(snapshot, 0)
    assert result["objective_s"] == pytest.approx(exhaustive_objective(snapshot, 0))
    assert result["final_gib"] == [4, 4]
    snapshot["new_gib"] = 3
    assert reference.solve_candidate(snapshot, 0)["status"] == "infeasible"


def test_incoming_request_cannot_make_original_owner_credit_unsafe(reference):
    snapshot = {"counts": [[1, 1], [0, 0]], "initial_gib": [2, 2], "capacity_gib": [3, 3],
                "new_gib": 2, "performance_cost_tables_s": [[1, 1, 0, 1], [0]],
                "movable": [{"id": "A0", "model": 0, "source": 0, "memory_gib": 1, "move_cost_s": 0.1,
                             "destination_pool": "A:1"},
                            {"id": "A1", "model": 0, "source": 1, "memory_gib": 1, "move_cost_s": 0.1,
                             "destination_pool": "A:0"}],
                "source_pages": [{"id": "page0", "rank": 0, "memory_gib": 1, "owners": ["A0"], "pool": "A:0"},
                                 {"id": "page1", "rank": 1, "memory_gib": 1, "owners": ["A1"], "pool": "A:1"}]}
    for target in (0, 1):
        result = reference.solve_candidate(snapshot, target)
        assert result["objective_s"] == pytest.approx(exhaustive_objective(snapshot, target))
        assert len(result["moves"]) == 1


def test_seeded_page_owner_plans_match_exhaustive_search(reference):
    generator = random.Random(20261019)
    for case in range(12):
        requests, pages = [], []
        counts = [[0, 0], [0, 0]]
        for index in range(4):
            model, source = generator.randrange(2), generator.randrange(2)
            counts[model][source] += 1
            requests.append({"id": f"request-{index}", "model": model, "source": source,
                             "memory_gib": generator.choice((1, 2)), "move_cost_s": 0.1,
                             "destination_pool": f"{model}:{1 - source}"})
        for model in (0, 1):
            for rank in (0, 1):
                owners = [request["id"] for request in requests
                          if request["model"] == model and request["source"] == rank]
                if owners:
                    pages.append({"id": f"page-{model}-{rank}", "rank": rank, "pool": f"{model}:{rank}",
                                  "memory_gib": len(owners), "owners": owners, "pinned": case % 5 == 0})
        initial = [1 + sum(page["memory_gib"] for page in pages if page["rank"] == rank) for rank in (0, 1)]
        snapshot = {"counts": counts, "initial_gib": initial,
                    "capacity_gib": [used + generator.randrange(4) for used in initial],
                    "new_gib": generator.choice((1, 2, 3)), "imbalance_cost_s": [0.2, 0.3],
                    "movable": requests, "source_pages": pages}
        for target in (0, 1):
            result = reference.solve_candidate(snapshot, target)
            expected = exhaustive_objective(snapshot, target)
            if expected is None:
                assert result["status"] == "infeasible"
            else:
                assert result["objective_s"] == pytest.approx(expected)


@pytest.mark.parametrize("capacity,initial,new,sizes,feasible", ((8, 7, 2, (2, 1), True), (20, 19, 2, (3, 2), False)))
def test_copy_order_and_deadlock(reference, snapshot, capacity, initial, new, sizes, feasible):
    snapshot.update(capacity_gib=[capacity] * 2, initial_gib=[initial] * 2, new_gib=new,
                    movable=[{"id": "outgoing_A", "model": 0, "source": 0, "memory_gib": sizes[0], "move_cost_s": 0.1},
                             {"id": "incoming_B", "model": 1, "source": 1, "memory_gib": sizes[1], "move_cost_s": 0.1}])
    result = reference.solve_candidate(snapshot, 0)
    expected = exhaustive_objective(snapshot, 0)
    if feasible:
        assert result["moves"] == ["incoming_B", "outgoing_A"]
        assert result["objective_s"] == pytest.approx(expected)
    else:
        assert result["status"] == "infeasible"
        assert expected is None


def test_no_movable_requests_fast_path(reference, snapshot):
    snapshot.update(movable=[], new_gib=1)
    result = reference.enumerate_targets(snapshot)
    assert result["status"] == "optimal"
    assert result["winner"]["target"] == 1
    assert result["winner"]["moves"] == []
    assert result["winner"]["imbalance_counts"] == [0, 0]


def test_move_limit_cannot_fake_capacity(reference, snapshot):
    result = reference.enumerate_targets(snapshot, max_moves=1)
    assert result["status"] == "no_feasible_plan"
    assert result["proven_infeasible"]
    assert all(exhaustive_objective(snapshot, target, max_moves=1) is None for target in (0, 1))


def test_invalid_snapshot_is_rejected(reference, snapshot):
    snapshot["movable"][0]["memory_gib"] = 50
    with pytest.raises(ValueError, match="exceeds initial"):
        reference.enumerate_targets(snapshot)


def test_seeded_small_instances_match_assignments_and_orders(reference, snapshot):
    generator = random.Random(42)
    for _ in range(24):
        snapshot["initial_gib"] = [generator.randint(5, 10) for rank in (0, 1)]
        snapshot["capacity_gib"] = [initial + generator.randint(0, 3) for initial in snapshot["initial_gib"]]
        snapshot["new_gib"] = generator.randint(1, 5)
        snapshot["imbalance_cost_s"] = [generator.uniform(0.01, 0.1) for model in (0, 1)]
        for request in snapshot["movable"]:
            request["memory_gib"] = generator.randint(1, 2)
            request["move_cost_s"] = generator.uniform(0.001, 0.1)
        limit = generator.randint(0, 4)
        for target in (0, 1):
            result = reference.solve_candidate(snapshot, target, max_moves=limit)
            expected = exhaustive_objective(snapshot, target, max_moves=limit)
            if expected is None:
                assert result["status"] == "infeasible"
            else:
                assert result["status"] == "optimal"
                assert result["objective_s"] == pytest.approx(expected)


def test_timeout_without_incumbent_is_not_infeasibility(reference, snapshot, monkeypatch):
    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "milp", lambda *args, **kwargs:
                        SimpleNamespace(status=1, x=None, message="Time limit reached"))
    result = reference.enumerate_targets(snapshot)
    assert result["status"] == "no_feasible_plan"
    assert not result["proven_infeasible"]
    assert all(candidate["status"] == "no_verified_incumbent" for candidate in result["candidates"])


def test_timeout_with_incumbent_is_not_global_optimality(reference, snapshot, monkeypatch):
    import scipy.optimize

    original = scipy.optimize.milp

    def limited(*args, **kwargs):
        result = original(*args, **kwargs)
        result.status = 1
        return result

    monkeypatch.setattr(scipy.optimize, "milp", limited)
    result = reference.enumerate_targets(snapshot)
    assert result["status"] == "feasible_not_proven"
    assert all(candidate["status"] == "feasible_time_limited" for candidate in result["candidates"])


def test_nonlinear_orientation_cost_table_changes_the_selected_rank(reference, snapshot):
    snapshot.update(movable=[], new_gib=1, performance_cost_tables_s=[[0.2] * 129, [0.3] * 129])
    snapshot["performance_cost_tables_s"][0][64] = 1
    result = reference.enumerate_targets(snapshot)
    assert result["winner"]["target"] == 0
    assert result["winner"]["objective_s"] == pytest.approx(0.5)
    assert result["winner"]["performance_cost_mode"] == "layout_table"


def test_seeded_layout_cost_tables_match_exhaustive_search(reference, snapshot):
    generator = random.Random(11)
    for _ in range(12):
        snapshot["performance_cost_tables_s"] = [[generator.uniform(0, 1) for count in range(129)] for model in (0, 1)]
        for target in (0, 1):
            result = reference.solve_candidate(snapshot, target)
            assert result["objective_s"] == pytest.approx(exhaustive_objective(snapshot, target))
            assert result["performance_cost_mode"] == "layout_table"


def test_partial_layout_cost_table_is_not_silently_extrapolated(reference, snapshot):
    snapshot["performance_cost_tables_s"] = [[0.1] * 128, [0.1] * 129]
    with pytest.raises(ValueError, match="cover every"):
        reference.enumerate_targets(snapshot)


def test_joint_cost_distinguishes_aligned_and_opposite_hot_ranks(reference, snapshot):
    snapshot.update(counts=[[1, 1], [2, 0]], movable=[], new_gib=1,
                    joint_performance_costs_s=[[3, 3, 3], [3, 3, 0.4], [3, 3, 2], [3, 3, 3]])
    result = reference.enumerate_targets(snapshot)
    assert result["winner"]["target"] == 1
    assert result["winner"]["objective_s"] == pytest.approx(0.4)
    assert result["winner"]["performance_cost_mode"] == "joint_layout_table"
    assert result["winner"]["migration_cost_s"] == 0
    assert [candidate["imbalance_counts"] for candidate in result["candidates"]] == [[1, 2], [1, 2]]


def test_seeded_joint_costs_match_assignments_and_orders(reference, snapshot):
    generator = random.Random(19)
    snapshot["counts"] = [[1, 1], [1, 1]]
    for _ in range(12):
        snapshot["joint_performance_costs_s"] = [[generator.uniform(0, 1) for count_b in range(3)]
                                               for count_a in range(4)]
        limit = generator.randint(0, 4)
        for target in (0, 1):
            result = reference.solve_candidate(snapshot, target, max_moves=limit)
            expected = exhaustive_objective(snapshot, target, max_moves=limit)
            if expected is None:
                assert result["status"] == "infeasible"
            else:
                assert result["objective_s"] == pytest.approx(expected)
                assert result["objective_s"] == pytest.approx(result["performance_cost_s"] + result["migration_cost_s"])


@pytest.mark.parametrize("joint_table,error", (([[0]], "cover every"),
                                             ([[0, 0, 0]] * 3 + [[0, 0, float("nan")]], "finite"),
                                             ([[0, 0, 0]] * 3 + [[0, 0, -1]], "nonnegative")))
def test_invalid_joint_cost_table_is_rejected(reference, snapshot, joint_table, error):
    snapshot.update(counts=[[1, 1], [1, 1]], joint_performance_costs_s=joint_table)
    with pytest.raises(ValueError, match=error):
        reference.enumerate_targets(snapshot)


def test_joint_and_separate_costs_cannot_be_double_counted(reference, snapshot):
    snapshot.update(performance_cost_tables_s=[[0] * 129, [0] * 129], joint_performance_costs_s=[[0]])
    with pytest.raises(ValueError, match="not both"):
        reference.enumerate_targets(snapshot)


def test_target_admission_cost_is_counted_in_objective_and_bound(reference, snapshot):
    snapshot.update(movable=[], new_gib=1, admission_cost_s=[0, 1])
    result = reference.enumerate_targets(snapshot)
    assert result["winner"]["target"] == 0
    for candidate in result["candidates"]:
        assert candidate["objective_s"] == pytest.approx(exhaustive_objective(snapshot, candidate["target"]))
        assert candidate["dual_bound_s"] == pytest.approx(candidate["objective_s"])
        assert candidate["objective_s"] == pytest.approx(candidate["performance_cost_s"]
                                                       + candidate["migration_cost_s"] + candidate["admission_cost_s"])


@pytest.mark.parametrize("admission_costs", ([0], [0, -1], [0, float("nan")]))
def test_invalid_admission_cost_is_rejected(reference, snapshot, admission_costs):
    snapshot["admission_cost_s"] = admission_costs
    with pytest.raises(ValueError, match="Admission costs"):
        reference.enumerate_targets(snapshot)


@pytest.mark.parametrize("mode", ("separate", "joint"))
def test_layout_lookup_does_not_require_rejected_affine_coefficients(reference, snapshot, mode):
    snapshot.pop("imbalance_cost_s")
    snapshot.update(counts=[[1, 1], [1, 1]], movable=[], new_gib=1)
    if mode == "separate":
        snapshot["performance_cost_tables_s"] = [[0.2] * 4, [0.3] * 3]
    else:
        snapshot["joint_performance_costs_s"] = [[0.5] * 3 for count in range(4)]
    assert reference.enumerate_targets(snapshot)["winner"]["objective_s"] == pytest.approx(0.5)


def test_timeout_gap_includes_target_admission_constant(reference, snapshot, monkeypatch):
    import scipy.optimize

    original = scipy.optimize.milp

    def limited(*args, **kwargs):
        result = original(*args, **kwargs)
        if result.x is not None:
            result.status = 1
            result.mip_dual_bound = result.fun - 0.02
        return result

    snapshot["admission_cost_s"] = [0.5, 1]
    monkeypatch.setattr(scipy.optimize, "milp", limited)
    result = reference.enumerate_targets(snapshot)
    assert result["status"] == "feasible_not_proven"
    for candidate in result["candidates"]:
        assert candidate["mip_gap"] == pytest.approx(0.02 / candidate["objective_s"])
