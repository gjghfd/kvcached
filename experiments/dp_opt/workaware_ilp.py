import math

from workaware_cost import evaluate_segments


def validate_work_snapshot(snapshot):
    segments = snapshot.get("work_segments", [])
    risk = snapshot.get("placement_risk_cost_s", 0)
    if isinstance(risk, bool) or not math.isfinite(risk) or risk < 0:
        raise ValueError("Placement risk must be finite nonnegative request-seconds")
    if not segments:
        return
    if snapshot.get("new_request_count", 1) != 0:
        raise ValueError("Remaining-work calibration currently supports rebalancing, not new prefill")
    known, previous = {}, {}
    for segment in segments:
        model, population = segment["model"], segment["population"]
        if (model not in (0, 1) or type(population) is not int or population <= 0
                or type(segment["start_step"]) is not int or segment["start_step"] < 0
                or type(segment["end_step"]) is not int or segment["end_step"] <= segment["start_step"]
                or segment["steps"] != segment["end_step"] - segment["start_step"]
                or len(segment["graph_costs_s"]) != population + 1
                or len(segment["initial_counts"]) != 2
                or any(type(count) is not int or count < 0 for count in segment["initial_counts"])
                or sum(segment["initial_counts"]) != population
                or len(segment["request_sources"]) != population
                or set(segment["request_sources"]) != set(segment["attention_weights"])):
            raise ValueError("Invalid future decode segment population, events or cost table")
        values = segment["graph_costs_s"] + [segment["attention_difference_cost_s"], segment["uncertainty_request_s"]]
        values += [segment.get("attention_difference_limit_tokens", sum(segment["attention_weights"].values()))]
        if any(isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Future decode costs must be finite nonnegative request-seconds")
        sources = segment["request_sources"]
        if (any(source not in (0, 1) for source in sources.values())
                or [sum(source == rank for source in sources.values()) for rank in (0, 1)] != segment["initial_counts"]
                or any(isinstance(value, bool) or not math.isfinite(value) or value <= 0
                       for value in segment["attention_weights"].values())):
            raise ValueError("Invalid future attention weights or initial sources")
        for identifier, source in sources.items():
            if identifier in known and known[identifier] != (model, source):
                raise ValueError("Future segments change request identity/model/source")
            known[identifier] = (model, source)
        if model in previous:
            earlier = previous[model]
            if (segment["start_step"] != earlier["end_step"]
                    or not set(sources) <= set(earlier["request_sources"])):
                raise ValueError("Future decode segments must be contiguous and only drain current requests")
        elif segment["start_step"] != 0 or segment["initial_counts"] != snapshot["counts"][model]:
            raise ValueError("First future segment must describe the current full population")
        previous[model] = segment
    for request in snapshot["movable"]:
        if known.get(request["id"]) != (request["model"], request["source"]):
            raise ValueError("Movable request is absent or inconsistent with remaining-work forecast")


def allocate_work_variables(snapshot, dimensions):
    variables = []
    for segment in snapshot.get("work_segments", []):
        counts = list(range(dimensions, dimensions + segment["population"] + 1))
        difference = counts[-1] + 1
        variables.append({"counts": counts, "difference": difference})
        dimensions = difference + 1
    return variables, dimensions


def install_work_constraints(snapshot, variables, requests, stages, dimensions,
                             objective, upper, integrality, constraint):
    import numpy as np

    for segment, positions in zip(snapshot.get("work_segments", []), variables):
        counts, difference = positions["counts"], positions["difference"]
        integrality[counts] = 1
        objective[counts] = segment["graph_costs_s"]
        objective[difference] = segment["attention_difference_cost_s"]
        upper[difference] = segment.get("attention_difference_limit_tokens", sum(segment["attention_weights"].values()))
        row = np.zeros(dimensions)
        row[counts] = 1
        constraint(row, minimum=1, maximum=1)
        row = np.zeros(dimensions)
        row[counts] = range(len(counts))
        for index, request in enumerate(requests):
            if request["id"] in segment["request_sources"]:
                source = segment["request_sources"][request["id"]]
                row[index * stages:(index + 1) * stages] = 1 if source == 0 else -1
        constraint(row, minimum=segment["initial_counts"][0], maximum=segment["initial_counts"][0])
        initial_difference = sum((1 if source == 0 else -1) * segment["attention_weights"][identifier]
                                 for identifier, source in segment["request_sources"].items())
        for sign in (-1, 1):
            row = np.zeros(dimensions)
            row[difference] = 1
            for index, request in enumerate(requests):
                if request["id"] in segment["request_sources"]:
                    source = segment["request_sources"][request["id"]]
                    weight = segment["attention_weights"][request["id"]]
                    row[index * stages:(index + 1) * stages] = sign * (2 if source == 0 else -2) * weight
            constraint(row, minimum=sign * initial_difference)


def work_cost(snapshot, moved_ids):
    segments = snapshot.get("work_segments", [])
    result = evaluate_segments(segments, moved_ids) if segments else {
        "performance_cost_s": 0, "graph_cost_s": 0, "attention_cost_s": 0, "counts_by_segment": []}
    return {**result, "placement_risk_cost_s": snapshot.get("placement_risk_cost_s", 0)}
