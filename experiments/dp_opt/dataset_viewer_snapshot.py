import argparse
import datetime
import email.utils
import gzip
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from public_response_profile import digest, normalize


def validate_page(page, offset, length, total=None):
    count = page.get("num_rows_total")
    if page.get("partial") is not False or not isinstance(count, int) or count <= 0:
        raise ValueError("Require a complete public viewer dataset, not a partial export")
    if total is not None and count != total:
        raise ValueError("Viewer dataset row count changed while exporting")
    expected = min(length, count - offset)
    rows = page.get("rows", [])
    if expected <= 0 or len(rows) != expected:
        raise ValueError("Viewer page misses original rows")
    for ordinal, row in enumerate(rows):
        if row.get("row_idx") != offset + ordinal or row.get("truncated_cells") != []:
            raise ValueError("Reject truncated, missing or reordered public viewer cells")
        if not isinstance(row.get("row"), dict):
            raise ValueError("Require original structured viewer row")
    return count


def original_messages(row):
    messages = normalize(row)
    system = row.get("system")
    if system is not None:
        if not isinstance(system, str) or any(message["role"] == "system" for message in messages):
            raise ValueError("Require unambiguous original system text")
        if system:
            messages.insert(0, {"role": "system", "content": system})
    return messages


def fetch(url):
    request = urllib.request.Request(url, headers={"Accept": "application/json", "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
    return data


def retry_delay(error, attempt):
    if isinstance(error, urllib.error.HTTPError) and error.code == 429:
        fallback = min(120 * 2 ** attempt, 900)
        value = error.headers.get("Retry-After") if error.headers else None
        if value and value.isdigit():
            return max(fallback, int(value))
        if value:
            try:
                deadline = email.utils.parsedate_to_datetime(value)
                remaining = (deadline - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
                return max(fallback, remaining)
            except (TypeError, ValueError):
                pass
        return fallback
    return 5 * (attempt + 1)


def download(destination, dataset, config="default", split="train", maximum_rows=20000):
    destination.mkdir(parents=True, exist_ok=True)
    identity = {"dataset": dataset, "config": config, "split": split, "maximum_rows": maximum_rows,
                "endpoint_supports_git_revision": False}
    identity_path = destination / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Never reuse cached pages for a different public source")
    if not identity_path.exists():
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    if (destination / "manifest.json").exists():
        raise ValueError("Snapshot is already frozen; do not overwrite it")
    pages, total, offset = [], None, 0
    while total is None or offset < total:
        length = 100
        query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                        "offset": offset, "length": length})
        url = "https://datasets-server.huggingface.co/rows?" + query
        path = destination / f"page-{offset:08d}.json"
        if not path.exists():
            for attempt in range(6):
                try:
                    data = fetch(url)
                    page = json.loads(data)
                    count = validate_page(page, offset, length, total)
                    if count > maximum_rows:
                        raise ValueError("Viewer source exceeds the declared download bound")
                    temporary = path.with_suffix(".partial")
                    temporary.write_bytes(data)
                    temporary.replace(path)
                    break
                except ValueError:
                    raise
                except Exception as error:
                    delay = retry_delay(error, attempt)
                    print(json.dumps({"offset": offset, "attempt": attempt + 1, "error": str(error),
                                      "retry_after_s": delay}), flush=True)
                    if attempt == 5:
                        raise
                    time.sleep(delay)
        page = json.loads(path.read_text())
        total = validate_page(page, offset, length, total)
        if total > maximum_rows:
            raise ValueError("Cached viewer source exceeds the declared download bound")
        pages.append({"path": path.name, "url": url, "sha256": digest(path),
                      "offset": offset, "rows": len(page["rows"])})
        offset += len(page["rows"])
        print(json.dumps({"stage": "public_original_viewer_snapshot", "rows": offset, "total": total}), flush=True)
        if offset < total:
            time.sleep(2)
    source = destination / "source.jsonl"
    with source.open("x") as output:
        for item in pages:
            page_path = destination / item["path"]
            if digest(page_path) != item["sha256"]:
                raise ValueError("Raw viewer page changed before canonical export")
            for row in json.loads(page_path.read_text())["rows"]:
                output.write(json.dumps({"source_row": row["row_idx"], "messages": original_messages(row["row"])},
                                        ensure_ascii=False) + "\n")
    content_identity = hashlib.sha256(json.dumps(pages, sort_keys=True).encode()).hexdigest()
    manifest = {**identity, "snapshot_frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "rows": total, "pages": pages, "pages_identity_sha256": content_identity,
                "source": {"path": source.name, "size": source.stat().st_size, "sha256": digest(source)},
                "source_git_revision_verified": False, "original_system_text_preserved": True,
                "original_cells_untruncated": True, "natural_eos_verified": False,
                "scope": "official_viewer_content_snapshot_not_LFS_file_or_git_revision_verification"}
    with (destination / "manifest.json").open("x") as output:
        output.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--maximum-rows", type=int, default=20000)
    args = parser.parse_args()
    result = download(args.destination, args.dataset, maximum_rows=args.maximum_rows)
    print(json.dumps({"rows": result["rows"], "source": result["source"],
                      "source_git_revision_verified": result["source_git_revision_verified"]}, indent=2))


if __name__ == "__main__":
    main()
