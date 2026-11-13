"""Generality check of general (*_g) trials, protocol §21.4 item 5: which requests started time-multiplexing
episodes and migration admissions, which requests were moved, and how many episodes each long request opened.

usage: python3 tools/check_general.py TRIAL_DIR [TRIAL_DIR ...]   (TRIAL_DIR = .../out/trace/<segment>/<policy>)
"""
import json
import sys
from collections import Counter
from pathlib import Path


def check(trial):
    trial = Path(trial)
    requests = json.loads((trial / "requests.json").read_text())
    model_of = {request["request_id"]: request["model"] for request in requests}  # controller ids
    trace_model = {request["trace_request_id"]: request["model"] for request in requests}
    episodes = json.loads((trial / "episodes.json").read_text()) if (trial / "episodes.json").exists() else []
    decisions = json.loads((trial / "decisions.json").read_text())
    migrations = json.loads((trial / "migrations.json").read_text())
    triggers = Counter(trigger for episode in episodes for trigger in episode["triggers"])
    admissions = [item for item in decisions if item.get("path") == "admission" and item.get("submitted")]
    direct_episodes = [item for item in decisions if item.get("path") == "direct" and item.get("episode")]
    moved_model = Counter(model_of.get(move["request_id"].rsplit("-", 1)[0]) for move in migrations)
    long_requests = [request["request_id"] for request in requests if request["model"] == 0]
    return {
        "trial": f"{trial.parent.name}/{trial.name}",
        "episodes": len(episodes),
        "episode_tenants": dict(Counter(episode["model"] for episode in episodes)),
        "trigger_tenants": dict(Counter(model_of.get(trigger) for trigger in triggers)),
        "episodes_per_tenant0_request": [triggers.get(identifier, 0) for identifier in sorted(long_requests)],
        "migration_admissions_by_tenant": dict(Counter(trace_model.get(item["request_id"]) for item in admissions)),
        "direct_admissions_starting_episodes_by_tenant": dict(Counter(trace_model.get(item["request_id"])
                                                                      for item in direct_episodes)),
        "moved_by_tenant": dict(moved_model),
    }


def main():
    for path in sys.argv[1:]:
        print(json.dumps(check(path), ensure_ascii=False))


if __name__ == "__main__":
    main()
