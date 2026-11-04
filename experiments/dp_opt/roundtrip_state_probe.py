import hashlib
import json
import os
import sys
from pathlib import Path

import live_trace
from migration_roundtrip_control import run_state_controls


async def rate_pilot_adapter(engines, controller, args, trace, tokenizer):
    return await run_state_controls(engines, controller, args, trace)


def validate_environment(environment):
    if environment.get("DP_OPT_KV_TRANSPORT") != "dp_echo":
        raise ValueError("State control requires explicit DP_OPT_KV_TRANSPORT=dp_echo before engine initialization")


def main():
    if "--pilot" not in sys.argv:
        raise ValueError("State probe must use the independent pilot branch, not a benefit trace")
    validate_environment(os.environ)
    root = Path(sys.argv[sys.argv.index("--output") + 1])
    root.mkdir(parents=True, exist_ok=True)
    sources = {"roundtrip_state_probe.py": Path(__file__),
               "migration_roundtrip_control.py": Path(__file__).with_name("migration_roundtrip_control.py")}
    hashes = {}
    for name, path in sources.items():
        content = path.read_bytes()
        with (root / (name + ".executed")).open("xb") as output:
            output.write(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    with (root / "roundtrip-source.json").open("x") as output:
        output.write(json.dumps(hashes, indent=2) + "\n")
    live_trace.run_rate_pilot = rate_pilot_adapter
    live_trace.main()


if __name__ == "__main__":
    main()
