import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
import urllib.parse
from pathlib import Path

import dataset_viewer_snapshot as snapshot
import public_response_profile as profiler


SCOPE = {"sample_only": True, "not_full_population": True, "no_GPU_benefit": True,
         "natural_eos_verified": False, "source_git_revision_verified": False,
         "endpoint_supports_git_revision": False}
ALGORITHM = "stratified_full_page_sha256_v1_including_terminal_window"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def identity_sha(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def identity(dataset, population, seed=20261031, config="default", split="train",
             page_count=16, page_length=100, label="openr1"):
    if (type(population) is not int or type(seed) is not int
            or type(page_count) is not int or type(page_length) is not int
            or not 1 <= page_count <= 16 or not 1 <= page_length <= 100
            or population < page_count * page_length or not all((dataset, config, split, label))):
        raise ValueError("Require a bounded, disjoint full-page public sample")
    return {"dataset": dataset, "config": config, "split": split, "label": label,
            "population_count": population, "seed": seed, "page_count": page_count,
            "page_length": page_length, "selection_algorithm": ALGORITHM}


def select_pages(sample_identity):
    population = sample_identity["population_count"]
    length = sample_identity["page_length"]
    candidates = list(range(0, population - length + 1, length))
    if candidates[-1] != population - length:
        candidates.append(population - length)
    count = sample_identity["page_count"]
    selected = []
    for stratum in range(count):
        group = candidates[stratum * len(candidates) // count:(stratum + 1) * len(candidates) // count]
        offset = min(group, key=lambda value: (identity_sha({"identity": sample_identity,
                                                           "offset": value}), value))
        selected.append({"stratum": stratum, "offset": offset, "length": length,
                         "selection_sha256": identity_sha({"identity": sample_identity, "offset": offset})})
    if any(after["offset"] < before["offset"] + length for before, after in zip(selected, selected[1:])):
        raise ValueError("Fixed selected pages overlap; never redraw")
    return selected


def event(root, sample_identity, stage, **details):
    record = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "pid": os.getpid(),
              "cwd": os.getcwd(), "identity_sha256": identity_sha(sample_identity),
              "stage": stage, **details}
    with (root / "status.jsonl").open("a", encoding="utf-8") as target:
        target.write(canonical(record) + "\n")
        target.flush()
        os.fsync(target.fileno())
    print(canonical(record), flush=True)
    return record


def require_sha(root, item, prefix):
    path = (root / item[prefix + "path"]).resolve()
    if (root.resolve() not in path.parents or not path.is_file()
            or path.stat().st_size != item[prefix + "size"]
            or profiler.digest(path) != item[prefix + "sha256"]):
        raise ValueError("Missing, escaping or changed sample evidence file")
    return path


def source_entry(sample_identity, page, revision):
    return {"dataset": sample_identity["dataset"], "config": sample_identity["config"],
            "split": sample_identity["split"], "label": sample_identity["label"],
            "path": page["canonical_path"], "destination": page["canonical_path"],
            "revision": revision, "sha256": page["canonical_sha256"], "size": page["canonical_size"],
            "page_offset": page["offset"], "rows": page["length"],
            "source_row_semantics": "page_local_line_index; global_row=page_offset+source_row",
            "sample_identity_sha256": identity_sha(sample_identity), **SCOPE}


def freeze_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("Do not overwrite changed frozen sample metadata")
    else:
        write_json(path, value)


def download(root, sample_identity):
    root.mkdir(parents=True, exist_ok=True)
    selected = select_pages(sample_identity)
    freeze_json(root / "identity.json", {"identity": sample_identity,
                "identity_sha256": identity_sha(sample_identity), "selected_pages": selected, **SCOPE})
    if (root / "manifest.json").exists():
        verify(root)
        return json.loads((root / "manifest.json").read_text())
    pages = []
    for chosen in selected:
        offset, length = chosen["offset"], chosen["length"]
        raw_path = root / f"page-{offset:08d}.json"
        receipt_path = root / f"page-{offset:08d}-receipt.json"
        query = urllib.parse.urlencode({"dataset": sample_identity["dataset"],
                "config": sample_identity["config"], "split": sample_identity["split"],
                "offset": offset, "length": length})
        url = "https://datasets-server.huggingface.co/rows?" + query
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if any(receipt.get(key) != value for key, value in chosen.items()) or receipt["url"] != url:
                raise ValueError("Cached page belongs to another fixed selection")
            require_sha(root, receipt, "raw_")
            require_sha(root, receipt, "canonical_")
            snapshot.validate_page(json.loads(raw_path.read_text()), offset, length,
                                   sample_identity["population_count"])
            pages.append(receipt)
            event(root, sample_identity, "page_resumed", offset=offset)
            continue
        if raw_path.exists():
            raise ValueError("Unreceipted raw page retained; diagnose before recovery, never refetch it")
        for attempt in range(6):
            time.sleep(2)
            event(root, sample_identity, "page_fetch", offset=offset, attempt=attempt + 1, url=url)
            try:
                data = snapshot.fetch(url)
                break
            except Exception as error:
                delay = snapshot.retry_delay(error, attempt)
                event(root, sample_identity, "network_failure", offset=offset, attempt=attempt + 1,
                      error_type=type(error).__name__, error=str(error), retry_after_s=delay)
                if attempt == 5:
                    raise
                time.sleep(delay)
        with raw_path.open("xb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        page = json.loads(data)
        snapshot.validate_page(page, offset, length, sample_identity["population_count"])
        canonical_path = root / f"page-{offset:08d}.jsonl"
        with canonical_path.open("x", encoding="utf-8") as target:
            for row in page["rows"]:
                messages = row["row"].get("messages")
                if snapshot.original_messages(row["row"]) != messages:
                    raise ValueError("Require unchanged original messages and role strings")
                target.write(canonical({"source_row": row["row_idx"], "messages": messages}) + "\n")
        receipt = {**chosen, "url": url, "raw_path": raw_path.name,
                   "raw_sha256": profiler.digest(raw_path), "raw_size": raw_path.stat().st_size,
                   "canonical_path": canonical_path.name, "canonical_sha256": profiler.digest(canonical_path),
                   "canonical_size": canonical_path.stat().st_size}
        write_json(receipt_path, receipt)
        pages.append(receipt)
        event(root, sample_identity, "page_complete", offset=offset, rows=length,
              completed_pages=len(pages), raw_sha256=receipt["raw_sha256"])
    content_sha = identity_sha({"identity": sample_identity, "pages": pages})
    manifest = {"identity": sample_identity, "identity_sha256": identity_sha(sample_identity),
                "content_identity_sha256": content_sha, "revision": "viewer-sample:" + content_sha,
                "pages": pages, "sampled_rows": sum(page["length"] for page in pages), **SCOPE}
    sources = [source_entry(sample_identity, page, manifest["revision"]) for page in pages]
    freeze_json(root / "download-manifest.json", sources)
    freeze_json(root / "manifest.json", manifest)
    verify(root)
    return manifest


def verify(root):
    frozen = json.loads((root / "identity.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    sample_identity = frozen["identity"]
    if (sample_identity.get("selection_algorithm") != ALGORITHM
            or manifest["identity"] != sample_identity
            or any(frozen.get(key) != value or manifest.get(key) != value for key, value in SCOPE.items())):
        raise ValueError("Changed sample identity or unsupported maturity claim")
    serialized = json.dumps(sample_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    sample_sha = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if frozen["identity_sha256"] != sample_sha or manifest["identity_sha256"] != sample_sha:
        raise ValueError("Sample identity SHA mismatch")
    population, length, count = (sample_identity[key] for key in ("population_count", "page_length", "page_count"))
    if (type(population) is not int or type(length) is not int or type(count) is not int
            or not 1 <= count <= 16 or not 1 <= length <= 100 or population < count * length):
        raise ValueError("Invalid bounded sample dimensions")
    candidates = list(range(0, population - length + 1, length))
    candidates += [population - length] if candidates[-1] != population - length else []
    expected_pages = []
    for stratum in range(count):
        ranked = []
        for offset in candidates[stratum * len(candidates) // count:(stratum + 1) * len(candidates) // count]:
            encoded = json.dumps({"identity": sample_identity, "offset": offset}, ensure_ascii=False,
                                 sort_keys=True, separators=(",", ":")).encode("utf-8")
            ranked.append((hashlib.sha256(encoded).hexdigest(), offset))
        selection_sha, offset = min(ranked)
        expected_pages.append({"stratum": stratum, "offset": offset, "length": length,
                               "selection_sha256": selection_sha})
    if frozen["selected_pages"] != expected_pages or len(manifest["pages"]) != count:
        raise ValueError("Fixed page-hash selection changed")
    sources = json.loads((root / "download-manifest.json").read_text())
    if len(sources) != count:
        raise ValueError("Missing profile page sources")
    total_rows, previous_end = 0, -1
    for expected, item, source in zip(expected_pages, manifest["pages"], sources):
        if any(item.get(key) != value for key, value in expected.items()) or item["offset"] < previous_end:
            raise ValueError("Changed or overlapping original page offset")
        previous_end = item["offset"] + length
        raw_path = require_sha(root, item, "raw_")
        canonical_path = require_sha(root, item, "canonical_")
        if (raw_path.name != f"page-{item['offset']:08d}.json"
                or canonical_path.name != f"page-{item['offset']:08d}.jsonl"):
            raise ValueError("Original page offset missing from source path")
        query = urllib.parse.parse_qs(urllib.parse.urlparse(item["url"]).query)
        if (not item["url"].startswith("https://datasets-server.huggingface.co/rows?")
                or query != {key: [str(value)] for key, value in {
                    "dataset": sample_identity["dataset"], "config": sample_identity["config"],
                    "split": sample_identity["split"], "offset": item["offset"], "length": length}.items()}):
            raise ValueError("Official URL identity changed")
        page = json.loads(raw_path.read_text())
        if (page.get("partial") is not False or type(page.get("num_rows_total")) is not int
                or page["num_rows_total"] != population or len(page.get("rows", [])) != length):
            raise ValueError("Partial, missing or changed original population")
        with canonical_path.open(encoding="utf-8") as target:
            for line_index, row in enumerate(page["rows"]):
                global_row = item["offset"] + line_index
                if row.get("row_idx") != global_row or row.get("truncated_cells") != []:
                    raise ValueError("Reordered or truncated original row")
                messages = row["row"].get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("Missing original message strings")
                for message in messages:
                    if (message.get("role") not in ("user", "assistant", "system")
                            or not isinstance(message.get("content"), str)):
                        raise ValueError("Unsupported original messages")
                if row["row"].get("system") not in (None, ""):
                    raise ValueError("Unrepresented original system string")
                expected_row = {"source_row": global_row, "messages": messages}
                encoded_row = json.dumps(expected_row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if target.readline() != encoded_row + "\n":
                    raise ValueError("Canonical strings/global-row mapping differ from raw page")
                total_rows += 1
            if target.read():
                raise ValueError("Extra canonical source rows")
        expected_source = {"dataset": sample_identity["dataset"], "config": sample_identity["config"],
            "split": sample_identity["split"], "label": sample_identity["label"],
            "path": canonical_path.name, "destination": canonical_path.name,
            "revision": manifest["revision"], "size": canonical_path.stat().st_size,
            "sha256": hashlib.sha256(canonical_path.read_bytes()).hexdigest(),
            "page_offset": item["offset"], "rows": length,
            "source_row_semantics": "page_local_line_index; global_row=page_offset+source_row",
            "sample_identity_sha256": sample_sha, **SCOPE}
        if source != expected_source:
            raise ValueError("Profile source SHA/offset/sample identity mismatch")
    content_encoded = json.dumps({"identity": sample_identity, "pages": manifest["pages"]},
                                ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    content_sha = hashlib.sha256(content_encoded).hexdigest()
    if (manifest["content_identity_sha256"] != content_sha or manifest["revision"] != "viewer-sample:" + content_sha
            or manifest["sampled_rows"] != total_rows or total_rows != count * length):
        raise ValueError("Changed snapshot content identity or sample count")
    return {"rows": total_rows, "pages": count, "identity_sha256": sample_sha,
            "content_identity_sha256": content_sha, "all_message_strings_reconstructed": True,
            "global_row_mapping_verified": True, "all_cells_untruncated": True, **SCOPE}


def theoretical_pressure(joint):
    count = sum(joint.values())
    output_sum = sum(output * frequency for (prompt, output), frequency in joint.items())
    if not count or not output_sum:
        raise ValueError("No original output available for theoretical pressure")
    weighted = sum(math.ceil((prompt + output) / 1024) * 96 * output * frequency
                   for (prompt, output), frequency in joint.items())
    moment, mean_output = weighted / count, output_sum / count
    ratio = weighted / output_sum
    available = 2 * (1.2 * 10240 - 9600)
    return {"E_bO_mib_tokens": moment, "E_O_tokens": mean_output, "E_bO_over_E_O_mib": ratio,
            "B_mib_per_gpu": 10240, "code_mib_per_gpu": 9600, "physical_pressure_ratio": 1.2,
            "Conv_population_prior": available / ratio,
            "Conv_RPS_priors": {str(step): available / (moment * step) for step in (.015, .020, .030)},
            "b_definition": "96*ceil((P+O)/1024) independent request page-upper MiB",
            "population_formula": "lambda*t*E[O]=2*(1.2*B-code)/(E[bO]/E[O])",
            "scope": "sample_moment_output_only_service_prior_not_actual_KV_allocator_or_joint_calibration", **SCOPE}


def verify_profile(root, output, tokenizer_path, config_path):
    from jinja2 import Environment, StrictUndefined
    from tokenizers import Tokenizer

    sample_check = verify(root)
    manifest = json.loads((root / "manifest.json").read_text())
    sources = json.loads((root / "download-manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.padding is not None or tokenizer.truncation is not None:
        raise ValueError("Independent tokenizer must not pad or truncate")
    if (summary["tokenizer_sha256"] != profiler.digest(tokenizer_path)
            or summary["tokenizer_config_sha256"] != profiler.digest(config_path)):
        raise ValueError("Tokenizer identity changed")
    template = Environment(undefined=StrictUndefined).from_string(json.loads(config_path.read_text())["chat_template"])
    joint, flag_counts = collections.Counter(), collections.Counter()
    response_count, row_count = 0, 0
    audit_path = output / "independent-response-audit.jsonl"
    with (output / "response-length-records.jsonl").open() as records, audit_path.open("x") as audit:
        for page, source in zip(manifest["pages"], sources):
            for line_index, row in enumerate(json.loads((root / page["raw_path"]).read_text())["rows"]):
                raw, global_row = row["row"], page["offset"] + line_index
                row_count += 1
                generations, reasoning, finishes = (raw.get(key) for key in
                                                    ("generations", "is_reasoning_complete", "finish_reasons"))
                flag_counts["finish_reasons_null_rows" if finishes is None else "finish_reasons_present_rows"] += 1
                for flag in reasoning if isinstance(reasoning, list) else [None]:
                    flag_counts["is_reasoning_complete:" + str(flag)] += 1
                if isinstance(finishes, list):
                    for reason in finishes:
                        flag_counts["finish_reason:" + str(reason)] += 1
                messages = raw["messages"]
                for index, message in enumerate(messages):
                    if (index == 0 or message["role"] != "assistant"
                            or not any(previous["role"] == "user" for previous in messages[:index])):
                        continue
                    response = message["content"]
                    prepared = [{"role": previous["role"], "content": previous["content"],
                                 **({"tool_calls": []} if previous["role"] == "assistant" else {})}
                                for previous in messages[:index]]
                    prompt = template.render(messages=prepared, tools=None, add_generation_prompt=True)
                    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
                    output_tokens = len(tokenizer.encode(response, add_special_tokens=False).ids)
                    response_sha = hashlib.sha256(response.encode("utf-8")).hexdigest()
                    profile_identity = source["dataset"] + "@" + source["revision"] + "/" + source["path"]
                    selection_sha = hashlib.sha256(
                        f"{profile_identity}|{line_index}|{index}|{response_sha}".encode("utf-8")).hexdigest()
                    expected = {"dataset": source["dataset"], "revision": source["revision"],
                        "source_file": source["destination"], "source_row": line_index, "assistant_index": index,
                        "source_sha256": source["sha256"], "original_response_sha256": response_sha,
                        "selection_sha256": selection_sha, "original_response_has_think_tag": "<think>" in response,
                        "prompt_tokens": prompt_tokens, "output_tokens": output_tokens}
                    line = records.readline()
                    if not line or json.loads(line) != expected:
                        raise ValueError("Profile P/O/original-response SHA or page-local row mismatch")
                    matches = [ordinal for ordinal, generation in enumerate(generations or []) if generation == response]
                    flag_counts["response_exactly_matches_generation" if matches else "response_generation_unmatched"] += 1
                    if bool(matches) and isinstance(reasoning, list):
                        for ordinal in matches:
                            flag_counts["matched_reasoning_complete:" + str(reasoning[ordinal])] += 1
                    flag_counts["responses_with_open_think"] += int("<think>" in response)
                    flag_counts["responses_with_close_think"] += int("</think>" in response)
                    audit.write(canonical({**expected, "source_global_row": global_row,
                        "source_page_offset": page["offset"], "is_reasoning_complete": reasoning,
                        "finish_reasons": finishes, "generation_response_match_indices": matches}) + "\n")
                    joint[prompt_tokens, output_tokens] += 1
                    response_count += 1
        if records.read():
            raise ValueError("Profile has extra sampled response records")
    label = manifest["identity"]["label"]
    dataset_summary = summary["datasets"][label]
    from workload_profile import histogram_statistics

    prompts = collections.Counter()
    outputs = collections.Counter()
    for (prompt_tokens, output_tokens), frequency in joint.items():
        prompts[prompt_tokens] += frequency
        outputs[output_tokens] += frequency
    prompt_stats, output_stats = histogram_statistics(prompts), histogram_statistics(outputs)
    if (dataset_summary["sampled_rows"] != row_count or dataset_summary["invalid_rows"] != 0
            or dataset_summary["sources"] != sources
            or dataset_summary["all_sampled_responses"]["requests"] != response_count
            or dataset_summary["all_sampled_responses"]["prompt_tokens"] != prompt_stats
            or dataset_summary["all_sampled_responses"]["original_output_tokens"] != output_stats):
        raise ValueError("Profile silently excluded rows or disagrees with independent P/O")
    pressure = theoretical_pressure(joint)
    if (dataset_summary["all_sampled_responses"]["mean_page_upper_times_output_mib_tokens"]
            != pressure["E_bO_mib_tokens"]):
        raise ValueError("Profile page-output moment differs from independent arithmetic")
    return {"sample_verification": sample_check, "rows": row_count, "responses": response_count,
            "prompt_tokens": prompt_stats, "original_output_tokens": output_stats,
            "original_flags": dict(flag_counts), "theoretical_pressure": pressure,
            "all_original_response_SHAs_and_P_O_independently_verified": True,
            "audit_sha256": profiler.digest(audit_path), **SCOPE}


def run(args):
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sample_identity = identity(args.dataset, args.population, args.seed, args.config, args.split,
                               args.page_count, args.page_length, args.label)
    with (root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            event(root, sample_identity, "worker_started", command=sys.argv, executable=sys.executable,
                  source_sha256=profiler.digest(Path(__file__)))
            if args.info:
                info = json.loads(args.info.read_text())
                if (info.get("partial") is not False or info.get("failed") or info.get("pending")
                        or info["dataset_info"][args.config]["splits"][args.split]["num_examples"] != args.population):
                    raise ValueError("Official info does not confirm the frozen complete population")
                freeze_json(root / "official-info-identity.json",
                            {"path": str(args.info.resolve()), "sha256": profiler.digest(args.info)})
            manifest = download(root, sample_identity)
            freeze_json(root / "independent-sample-verification.json", verify(root))
            event(root, sample_identity, "sample_verified", rows=manifest["sampled_rows"],
                  content_identity_sha256=manifest["content_identity_sha256"])
            output = args.output.resolve()
            if not (output / "summary.json").exists():
                event(root, sample_identity, "profile_started", output=str(output), export_limit=64)
                profiler.profile(root, args.tokenizer, args.tokenizer_config, output, export_limit=64)
            check_path = output / "independent-profile-verification.json"
            if check_path.exists():
                raise ValueError("Profile already independently frozen; do not overwrite results")
            result = verify_profile(root, output, args.tokenizer, args.tokenizer_config)
            write_json(check_path, result)
            event(root, sample_identity, "complete", rows=result["rows"], responses=result["responses"],
                  verification_sha256=profiler.digest(check_path), output=str(output), **SCOPE)
        except BaseException as error:
            event(root, sample_identity, "failed", error_type=type(error).__name__, error=str(error),
                  traceback=traceback.format_exc(), fixed_selection_retained=True)
            raise


def launch(args):
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sample_identity = identity(args.dataset, args.population, args.seed, args.config, args.split,
                               args.page_count, args.page_length, args.label)
    with (root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run_number = 1
    while (root / f"run-{run_number:04d}.json").exists():
        run_number += 1
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", "run",
               "--destination", str(root), "--output", str(args.output.resolve()), "--dataset", args.dataset,
               "--config", args.config, "--split", args.split, "--label", args.label,
               "--population", str(args.population), "--seed", str(args.seed),
               "--page-count", str(args.page_count), "--page-length", str(args.page_length),
               "--tokenizer", str(args.tokenizer.resolve()), "--tokenizer-config", str(args.tokenizer_config.resolve())]
    if args.info:
        command += ["--info", str(args.info.resolve())]
    log_path = root / f"run-{run_number:04d}.log"
    with log_path.open("xb") as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 cwd=os.getcwd(), start_new_session=True)
    record = {"pid": child.pid, "cwd": os.getcwd(), "identity": sample_identity,
              "identity_sha256": identity_sha(sample_identity), "command": command, "log": str(log_path),
              "status_log": str(root / "status.jsonl"), "source_sha256": profiler.digest(Path(__file__)),
              "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **SCOPE}
    write_json(root / f"run-{run_number:04d}.json", record)
    print(json.dumps(record, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("launch", "run", "verify"), default="launch")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--label", default="openr1")
    parser.add_argument("--population", type=int, default=93733)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--page-count", type=int, default=16)
    parser.add_argument("--page-length", type=int, default=100)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--tokenizer-config", type=Path)
    parser.add_argument("--info", type=Path)
    args = parser.parse_args()
    if args.mode == "verify":
        print(json.dumps(verify(args.destination), indent=2))
        return
    if not all((args.output, args.tokenizer, args.tokenizer_config)):
        parser.error("Download/profile workers require output and original tokenizer paths")
    (launch if args.mode == "launch" else run)(args)


if __name__ == "__main__":
    main()
