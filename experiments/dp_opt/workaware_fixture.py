import argparse
import hashlib
import json
from pathlib import Path


def make_fixture(code_marks, conv_marks, population, prompt_bank_sha256):
    if population not in (2, 9):
        raise ValueError("Predeclared recovery fixtures use N2 and N9")
    conv = next(mark for mark in conv_marks if 2048 <= mark["prompt_tokens"] <= 3072)
    code = code_marks[0]
    requests = [{**conv, "request_id": f"recovery-{population}-conv-{index}", "model": 1,
                 "arrival_s": index * .01, "original_output_tokens": conv["output_tokens"], "output_tokens": 8192,
                 "synthetic_output_tokens": 8192, "source_fragment": "Azure-Conv-input-synthetic-long-output"}
                for index in range(population)]
    requests.append({**code, "request_id": f"recovery-{population}-code", "model": 0, "arrival_s": 20.})
    return {"max_model_len": 131072, "duration_s": 300., "requests": requests,
            "prompt_bank_sha256": prompt_bank_sha256, "seed": 20261026,
            "sampling": "predeclared_closed_population_recovery_mechanism_not_Poisson_final_trace",
            "evaluation_start_s": 0., "evaluation_duration_s": 300.,
            "synthetic_Conv_output_tokens": 8192}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, choices=(2, 9), required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve frozen recovery fixture")
    read = lambda name: json.loads((args.inputs / name).read_text())
    result = make_fixture(read("code-marks.json"), read("conv-marks.json"), args.population,
                          hashlib.sha256((args.inputs / "prompt-bank.json").read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
