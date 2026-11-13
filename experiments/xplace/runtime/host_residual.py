import bisect
from collections import Counter, defaultdict
import math

from reference_work import fingerprint


OUTCOMES = ("returned", "failed", "right_censored")


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {name}")
    return float(value)


def group_key(group):
    if not isinstance(group, (list, tuple)) or not group or any(
            type(value) not in (str, int) for value in group):
        raise ValueError("Group must contain explicit string/integer covariates")
    return fingerprint(list(group))


def fit_host_residual(episodes, *, source_id, minimum_at_risk, minimum_cases):
    """Fit time until host-scope return, including failed returns and right censoring."""
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("Training requires source identity")
    if any(type(value) is not int or value < 1 for value in (minimum_at_risk, minimum_cases)):
        raise ValueError("Minimum support must be positive integers")
    groups = defaultdict(list)
    identities = set()
    for episode in episodes:
        if (not isinstance(episode.get("id"), str) or not episode["id"] or episode["id"] in identities
                or not isinstance(episode.get("case"), str) or not episode["case"]
                or episode.get("outcome") not in OUTCOMES):
            raise ValueError("Invalid or duplicate training episode")
        identities.add(episode["id"])
        duration = nonnegative(episode.get("duration_s"), "episode duration")
        key = group_key(episode.get("group"))
        groups[key].append(dict(episode, duration_s=duration))
    tables = {}
    for key, rows in groups.items():
        times = defaultdict(list)
        cases = Counter(row["case"] for row in rows)
        for row in rows:
            times[row["duration_s"]].append(row)
        survival, failures, area, previous = 1.0, 0.0, 0.0, 0.0
        risk = len(rows)
        points = []
        for duration, ending in sorted(times.items()):
            area += survival * (duration - previous)
            failed = sum(row["outcome"] == "failed" for row in ending)
            returned = sum(row["outcome"] == "returned" for row in ending)
            failures += survival * failed / risk
            survival *= 1 - (returned + failed) / risk
            risk -= len(ending)
            for row in ending:
                cases[row["case"]] -= 1
                if cases[row["case"]] == 0:
                    del cases[row["case"]]
            points.append(dict(time_s=duration, survival=survival, failure_cdf=failures, area_s=area,
                               at_risk=risk, cases_at_risk=len(cases)))
            previous = duration
        tables[key] = dict(group=list(rows[0]["group"]), count=len(rows),
            cases=sorted({row["case"] for row in rows}), outcomes=dict(Counter(row["outcome"] for row in rows)),
            points=points)
    model = dict(kind="host_return_product_limit_v1", source_id=source_id, minimum_at_risk=minimum_at_risk,
                 minimum_cases=minimum_cases, groups=tables, episode_count=len(identities),
                 training_sha256=fingerprint(sorted(episodes, key=lambda row: row["id"])),
                 deployment_status="exploratory_host_return_only_not_device_or_token_progress")
    model["model_id"] = fingerprint(model)
    return model


class HostResidualModel:
    def __init__(self, model):
        content = {key: value for key, value in model.items() if key != "model_id"}
        if model.get("kind") != "host_return_product_limit_v1" or fingerprint(content) != model.get("model_id"):
            raise ValueError("Residual model changed or has unsupported kind")
        self.model_id = model["model_id"]
        self.minimum_at_risk = model["minimum_at_risk"]
        self.minimum_cases = model["minimum_cases"]
        self.tables = {}
        for key, table in model["groups"].items():
            self.tables[key] = dict(times=tuple(point["time_s"] for point in table["points"]),
                                   points=tuple(dict(point) for point in table["points"]), count=table["count"],
                                   cases=len(table["cases"]))

    def state(self, table, age):
        index = bisect.bisect_right(table["times"], age) - 1
        if index < 0:
            return dict(survival=1.0, failure_cdf=0.0, area_s=age, at_risk=table["count"],
                        cases_at_risk=table["cases"])
        point = table["points"][index]
        return dict(point, area_s=point["area_s"] + point["survival"] * (age - point["time_s"]))

    def predict(self, group, *, observed_age_s, observation_lag_s, restriction_s):
        """Condition on activity at sampling, not unobserved survival through receipt lag."""
        age = nonnegative(observed_age_s, "observed age")
        lag = nonnegative(observation_lag_s, "observation lag")
        horizon = nonnegative(restriction_s, "restriction")
        if horizon == 0 or not math.isfinite(age + lag + horizon):
            raise ValueError("Invalid restricted prediction interval")
        result = dict(status="unavailable", reason=None, model_id=self.model_id, group=list(group),
            observed_age_s=age, observation_lag_s=lag, restriction_s=horizon,
            restricted_mean_s=None, full_mean_s=None, prediction_gate_passed=False,
            target="host_return_not_device_completion_or_output_milestone")
        table = self.tables.get(group_key(group))
        if table is None:
            return dict(result, reason="unseen_group")
        sampled = self.state(table, age)
        result.update(at_risk=sampled["at_risk"], cases_at_risk=sampled["cases_at_risk"])
        if (sampled["survival"] <= 0 or sampled["at_risk"] < self.minimum_at_risk
                or sampled["cases_at_risk"] < self.minimum_cases):
            return dict(result, reason="insufficient_conditional_support")
        decision_age, stop_age = age + lag, age + lag + horizon
        last = table["points"][-1]
        if last["survival"] > 0 and stop_age > last["time_s"]:
            known_start = min(decision_age, last["time_s"])
            lower = (last["area_s"] - self.state(table, known_start)["area_s"]) / sampled["survival"]
            return dict(result, reason="unidentified_censored_tail", restricted_mean_lower_bound_s=max(0.0, lower))
        decision, stop = self.state(table, decision_age), self.state(table, stop_age)
        denominator = sampled["survival"]
        remaining = max(0.0, (stop["area_s"] - decision["area_s"]) / denominator)
        return dict(result, status="estimated_exploratory", restricted_mean_s=remaining,
            full_mean_s=remaining if stop["survival"] == 0 else None,
            already_returned_probability=max(0.0, 1 - decision["survival"] / denominator),
            unresolved_at_horizon_probability=stop["survival"] / denominator,
            failed_return_probability_by_horizon=(stop["failure_cdf"] - sampled["failure_cdf"]) / denominator,
            censored_support_assumption="independent_right_censoring_within_group_not_empirically_proved")
