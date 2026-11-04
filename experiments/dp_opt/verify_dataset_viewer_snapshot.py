import argparse
import hashlib
import json
from pathlib import Path


def file_digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(data)
    return hasher.hexdigest()


def contained(root, name):
    path = (root / name).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError("Snapshot file must stay inside its raw evidence root")
    return path


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("source_git_revision_verified") is not False
            or manifest.get("endpoint_supports_git_revision") is not False):
        raise ValueError("Viewer evidence does not verify the upstream Git/LFS revision")
    identity = hashlib.sha256(json.dumps(manifest["pages"], sort_keys=True).encode()).hexdigest()
    if identity != manifest["pages_identity_sha256"]:
        raise ValueError("Frozen viewer page manifest identity changed")
    source = contained(root, manifest["source"]["path"])
    if (source.stat().st_size != manifest["source"]["size"]
            or file_digest(source) != manifest["source"]["sha256"]):
        raise ValueError("Canonical source size/SHA changed")
    count = 0
    roles = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant", "system": "system"}
    with source.open() as canonical:
        for item in manifest["pages"]:
            path = contained(root, item["path"])
            if item["offset"] != count or file_digest(path) != item["sha256"]:
                raise ValueError("Original viewer page ordering/SHA changed")
            page = json.loads(path.read_text())
            if (page.get("partial") is not False or page["num_rows_total"] != manifest["rows"]
                    or len(page["rows"]) != item["rows"]):
                raise ValueError("Partial or changed original viewer page")
            for row in page["rows"]:
                if row["row_idx"] != count or row.get("truncated_cells") != []:
                    raise ValueError("Truncated or missing original row")
                raw = row["row"]
                conversations = next(raw[field] for field in ("messages", "conversations", "conversation") if field in raw)
                messages = []
                if raw.get("system"):
                    messages.append({"role": "system", "content": raw["system"]})
                for message in conversations:
                    role = roles[message.get("role", message.get("from"))]
                    content = message.get("content", message.get("value"))
                    if not isinstance(content, str):
                        raise ValueError("Original response is not unmodified text")
                    messages.append({"role": role, "content": content})
                expected = {"source_row": count, "messages": messages}
                line = canonical.readline()
                if not line or json.loads(line) != expected:
                    raise ValueError("Canonical messages differ from the original public source cells")
                count += 1
        if canonical.read() or count != manifest["rows"]:
            raise ValueError("Canonical export has extra or missing source rows")
    return {"rows": count, "pages_sha_verified": len(manifest["pages"]),
            "all_original_messages_independently_reconstructed": True,
            "all_cells_untruncated": True, "original_system_preserved": True,
            "source_sha256": manifest["source"]["sha256"], "page_manifest_identity_sha256": identity,
            "source_git_revision_verified": False, "natural_eos_verified": False,
            "research_benefit_verified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
