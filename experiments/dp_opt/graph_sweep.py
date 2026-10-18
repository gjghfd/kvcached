import argparse
import json
import subprocess
import sys
from pathlib import Path

from batch_sweep import case_metrics, confirmed_target
from single_model_benchmark import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_root / "motivation/graph-sweep"
    root.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "execution": "FULL_DECODE_ONLY, torch.compile disabled",
              "mechanism": "Standard DP max-rank CUDA-graph padding; not pure unpadded attention imbalance",
              "target_decode_loss": 0.30, "cases": []}
    save_json(root / "results.json", report)
    try:
        for requests in (128, 256, 32):
            name = f"graph-batch-{requests}-ctx1024"
            command = [sys.executable, str(Path(__file__).with_name("motivation.py")),
                       "--url", args.url, "--checkpoint", args.checkpoint,
                       "--evidence-root", str(args.evidence_root), "--case-name", name,
                       "--requests", str(requests), "--input-tokens", "1024",
                       "--output-tokens", "512", "--extremes-only",
                       "--expected-cudagraph-mode", "FULL"]
            print(json.dumps({"starting_graph_case": name}), flush=True)
            subprocess.run(command, check=True)
            result = json.loads((args.evidence_root / "motivation/cases" / name / "results.json").read_text())
            if result["status"] != "passed":
                raise RuntimeError("Unverified graph trials cannot count toward target")
            metrics = case_metrics(result)
            report["cases"].append({"name": name, "requests": requests, **metrics,
                                    "target_confirmed": confirmed_target(metrics)})
            save_json(root / "results.json", report)
        report["status"] = "passed"
        report["target_confirmed"] = any(case["target_confirmed"] for case in report["cases"])
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        save_json(root / "results.json", report)
        raise
    save_json(root / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
