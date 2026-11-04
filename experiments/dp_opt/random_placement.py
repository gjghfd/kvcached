import hashlib
import random

from ilp_reference import replay_candidate, validate_snapshot


POLICY = "random_no_return"


def request_placement_seed(seed, repeat, request_id):
    payload = f"{seed}:{repeat}:{request_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def select_random_candidate(candidates, seed, selected_target=None):
    feasible = [candidate for candidate in candidates if "objective_s" in candidate]
    if selected_target is not None:
        feasible = [candidate for candidate in feasible if candidate["target"] == selected_target]
    direct = [candidate for candidate in feasible if not candidate["moves"]]
    feasible = direct or feasible
    targets = sorted({candidate["target"] for candidate in feasible})
    target = random.Random(seed).choice(targets) if targets else None
    winner = next((candidate for candidate in feasible if candidate["target"] == target), None)
    return winner, {"seed": seed, "feasible_targets": targets, "selected_target": target,
                    "selection": "uniform_target_not_objective_minimization", "release_rebalance": False}


def greedy_target_candidate(snapshot, target, max_moves):
    direct = replay_candidate(snapshot, {"target": target, "moves": []})
    if "objective_s" in direct:
        return direct
    requests = [request for request in snapshot["movable"] if request["source"] == target]
    pages = snapshot.get("source_pages")
    released = {request["id"]: (request["memory_gib"] if pages is None else sum(
        page["memory_gib"] for page in pages
        if not page.get("pinned", False) and page["owners"] == [request["id"]])) for request in requests}
    requests.sort(key=lambda request: (-released[request["id"]], request["memory_gib"], request["id"]))
    destination = 1 - target
    copied_gib, moves = 0, []
    for request in requests:
        if len(moves) >= max_moves:
            break
        if (snapshot["initial_gib"][destination] + copied_gib + request["memory_gib"]
                > snapshot["capacity_gib"][destination] + 1e-6):
            continue
        moves.append(request["id"])
        copied_gib += request["memory_gib"]
        candidate = replay_candidate(snapshot, {"target": target, "moves": moves})
        if "objective_s" in candidate:
            candidate["status"] = "greedy_feasible_not_optimal"
            candidate["scope"] = "one_way_largest_exclusive_release_first_shared_owner_safe_no_ILP"
            return candidate
    return {"target": target, "status": "no_greedy_certificate",
            "scope": "bounded_one_way_greedy_failure_not_allocator_or_ILP_infeasibility"}


def random_target_plan(snapshot, max_moves=2, seed=0):
    validate_snapshot(snapshot)
    if snapshot.get("new_request_count", 1) != 1:
        raise ValueError("Random no-return placement is only an admission policy")
    if (not isinstance(max_moves, int) or isinstance(max_moves, bool) or max_moves < 0
            or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError("Random placement needs nonnegative integer move limit and seed")
    candidates = [greedy_target_candidate(snapshot, target, max_moves) for target in (0, 1)]
    winner, selection = select_random_candidate(candidates, seed)
    return {"status": "greedy_feasible_not_optimal" if winner else "no_feasible_greedy_plan",
            "proven_infeasible": False, "candidates": candidates, "winner": winner,
            "random_selection": selection,
            "scope": "cross_tenant_random_feasible_DP_one_way_greedy_no_return_no_ILP"}
