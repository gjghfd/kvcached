import argparse
import csv
import hashlib
import io
import json
import math
import random
import statistics
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3] / "evidence/azure-trace-theory-20261021"
DEFAULT_OUTPUT = ROOT.parent / "azure-trace-32k-theory-20261021"
SEED = 20261021
WINDOWS = 24
WINDOW_BYTES = 262144
ASSETS = {"code": (436400338, 691989454), "conv": (436401563, 1135195393)}
SOURCE = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/"


def download(url, headers=None):
    request = urllib.request.Request(url, headers={"User-Agent": "dp-opt-trace-theory", **(headers or {})})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(WINDOW_BYTES + 1) if headers else response.read()
        return response.status, dict(response.headers), payload


def fetch_window(specification):
    dataset, index, start, size = specification
    end = start + WINDOW_BYTES - 1
    url = f"https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_{dataset}_1week.csv"
    code, headers, payload = download(url, {"Range": f"bytes={start}-{end}"})
    if code != 206 or len(payload) != WINDOW_BYTES or headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
        raise RuntimeError("Require bounded exact byte-range response")
    artifact = ROOT / "ranges" / f"{dataset}-{index:02d}.csv.fragment"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if artifact.exists():
        raise RuntimeError("Preserve existing source sample")
    artifact.write_bytes(payload)
    rows = []
    for fields in csv.reader(io.StringIO(payload.decode().split("\n", 1)[1].rsplit("\n", 1)[0])):
        if len(fields) != 3:
            raise ValueError("Unexpected Azure three-column schema")
        context, output = int(fields[1]), int(fields[2])
        rows.append((context, output))
    return {"dataset": dataset, "window": index, "start": start, "end": end, "url": url,
            "response_content_range": headers["Content-Range"], "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(), "rows": len(rows)}, rows


def summarize(rows):
    if not rows:
        return {"requests": 0}
    contexts = sorted(context for context, _ in rows)
    outputs = sorted(output for _, output in rows)
    def quantile(values, fraction):
        return values[round((len(values) - 1) * fraction)]
    return {"requests": len(rows), "mean_context": statistics.mean(contexts),
            "mean_output": statistics.mean(outputs),
            "context_quantiles": {str(fraction): quantile(contexts, fraction) for fraction in (.5, .9, .95, .99)},
            "output_quantiles": {str(fraction): quantile(outputs, fraction) for fraction in (.5, .9, .95, .99)},
            "mean_reserved_mib": statistics.mean(math.ceil((context + output) / 16) * 1.5 for context, output in rows),
            "mean_context_times_output": statistics.mean(context * output for context, output in rows),
            "decode_weighted_reserved_mib": sum(math.ceil((context + output) / 16) * 1.5 * output
                                                for context, output in rows) / sum(outputs),
            "service_assumptions": [{"decode_step_s": cadence, "prefill_s": prefill,
                                     "mean_resident_s": statistics.mean(prefill + cadence * output for _, output in rows),
                                     "mean_kv_mib_seconds": statistics.mean(math.ceil((context + output) / 16) * 1.5
                                                                            * (prefill + cadence * output)
                                                                            for context, output in rows)}
                                    for cadence, prefill in ((.035, .2), (.04, .2), (.05, .5))]}


def read_cached_sample(root):
    records = {dataset: [] for dataset in ASSETS}
    for entry in json.loads((root / "range-manifest.json").read_text()):
        artifact = root / "ranges" / f"{entry['dataset']}-{entry['window']:02d}.csv.fragment"
        payload = artifact.read_bytes()
        if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("Cached Azure fragment differs from its source manifest")
        rows = [(int(fields[1]), int(fields[2])) for fields in csv.reader(
            io.StringIO(payload.decode().split("\n", 1)[1].rsplit("\n", 1)[0]))]
        if len(rows) != entry["rows"]:
            raise ValueError("Cached Azure fragment row count differs from its source manifest")
        records[entry["dataset"]].extend(rows)
    return records


def analyze_lengths(records, max_model_len):
    if max_model_len <= 0:
        raise ValueError("Max model length must be positive")
    report = {"created_at": datetime.now().astimezone().isoformat(), "seed": SEED,
              "max_model_len": max_model_len,
              "scope": "CPU analytical source-length sample only, no GPU inference or trace-performance measurement",
              "sampling": "24 uniformly located byte windows in equal byte strata per dataset; full internal rows only, not exact uniform row reservoir",
              "datasets": {}}
    for dataset, rows in records.items():
        supported = [(context, output) for context, output in rows
                     if context > 0 and output > 0 and context + output <= max_model_len]
        long = [(context, output) for context, output in supported
                if context >= 2048 and 256 <= output <= 512]
        longer = [(context, output) for context, output in supported
                  if context >= 2048 and 512 <= output <= 1024]
        report["datasets"][dataset] = {"all": summarize(rows), "supported": summarize(supported),
                                      "long_slice": summarize(long), "longer_decode_slice": summarize(longer),
                                      "supported_fraction": len(supported) / len(rows),
                                      "long_slice_fraction": len(long) / len(rows),
                                      "longer_decode_fraction": len(longer) / len(rows),
                                      "max_total_tokens_in_sample": max(context + output for context, output in rows)}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-sample", type=Path)
    args = parser.parse_args()
    if args.max_model_len <= 0:
        parser.error("Max model length must be positive")
    if (args.output / "length-statistics.json").exists():
        parser.error("Preserve existing analysis; select a new output directory")
    if args.reuse_sample is not None:
        records = read_cached_sample(args.reuse_sample)
        sample_root = args.reuse_sample
    else:
        if list((ROOT / "ranges").glob("*.fragment")):
            parser.error("Preserve existing sample; use --reuse-sample")
        ROOT.mkdir(parents=True, exist_ok=True)
        for filename in ("AzureLLMInferenceDataset2024.md", "LICENSE"):
            _, _, text = download(SOURCE + filename)
            (ROOT / filename).write_bytes(text)
        _, _, metadata = download("https://api.github.com/repos/Azure/AzurePublicDataset/releases/tags/dataset-llm-2024")
        (ROOT / "release.json").write_bytes(metadata)
        _, _, notebook = download(SOURCE + "analysis/AzureLLMInferenceDataset2024.ipynb")
        (ROOT / "AzureLLMInferenceDataset2024.ipynb").write_bytes(notebook)
        rng = random.Random(SEED)
        specifications = []
        for dataset, (_, size) in ASSETS.items():
            for index in range(WINDOWS):
                lower = size * index // WINDOWS
                upper = size * (index + 1) // WINDOWS - WINDOW_BYTES
                specifications.append((dataset, index, rng.randint(lower, upper), size))
        records = {dataset: [] for dataset in ASSETS}
        manifest = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for metadata, rows in executor.map(fetch_window, specifications):
                manifest.append(metadata)
                records[metadata["dataset"]].extend(rows)
                print(metadata["dataset"], metadata["window"], metadata["rows"], flush=True)
        (ROOT / "range-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        sample_root = ROOT
    report = analyze_lengths(records, args.max_model_len)
    report["source_sample"] = str(sample_root.resolve())
    report["source_manifest_sha256"] = hashlib.sha256((sample_root / "range-manifest.json").read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "length-statistics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
