"""Build, upload, submit, poll and fetch xplace campaign jobs (runs locally; uses `ssh paracloud`)."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

XPLACE = Path(__file__).resolve().parents[1]
WORKSPACE = XPLACE.parents[2]
STAGING = WORKSPACE / "evidence" / "xplace-staging"
REMOTE_RELATIVE = "run/kvcached-paracloud/xplace"
REMOTE = f"$HOME/{REMOTE_RELATIVE}"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=15",
       "-o", "ServerAliveCountMax=4", "paracloud"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote(script, check=True, timeout=300):
    result = subprocess.run(SSH + ["bash -s"], input="set -euo pipefail\n" + script, text=True,
                            capture_output=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(f"remote failed ({result.returncode}): {result.stderr.strip()[-2000:]}")
    return result.stdout


def build(tag, inputs):
    root = STAGING / tag
    if root.exists():
        raise FileExistsError(f"Staging for {tag} already exists; tags are immutable")
    upload = root / "upload"
    for name in ("runtime", "tests", "inputs"):
        (upload / name).mkdir(parents=True)
    for path in sorted((XPLACE / "runtime").glob("*.py")):
        shutil.copy2(path, upload / "runtime" / path.name)
    for path in sorted((XPLACE / "tests").glob("*.py")):
        shutil.copy2(path, upload / "tests" / path.name)
    shutil.copy2(XPLACE / "xplace.sbatch", upload / "xplace.sbatch")
    for path in inputs:
        shutil.copy2(path, upload / "inputs" / path.name)
    files = sorted(path for path in upload.rglob("*") if path.is_file())
    manifest = "".join(f"{sha256(path)}  {path.relative_to(upload)}\n" for path in files)
    (upload / "MANIFEST.sha256").write_text(manifest)
    tarball = root / "upload.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        archive.add(upload, arcname="upload")
    record = {"tag": tag, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": len(files),
              "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(), "tarball_sha256": sha256(tarball)}
    (root / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def upload(tag):
    root = STAGING / tag
    record = json.loads((root / "build.json").read_text())
    remote(f'if [ -e "{REMOTE}/{tag}" ]; then echo "remote tag exists" >&2; exit 3; fi\n'
           f'mkdir -p "{REMOTE}/{tag}/logs"\n')
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(root / "upload.tar.gz"),
                    f"paracloud:{REMOTE_RELATIVE}/{tag}/upload.tar.gz"], check=True, timeout=1800)
    output = remote(f'cd "{REMOTE}/{tag}"\n'
                    f'test "$(sha256sum upload.tar.gz | cut -d" " -f1)" = "{record["tarball_sha256"]}"\n'
                    'tar xzf upload.tar.gz\n'
                    'chmod -R a-w upload\n'
                    'cd upload && sha256sum -c --quiet MANIFEST.sha256 && echo VERIFIED\n')
    if "VERIFIED" not in output:
        raise RuntimeError(f"Remote manifest verification failed: {output}")
    record["uploaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (root / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def submit(tag, env, time_limit, job_name, dependency=None, exclude=None):
    # sbatch --export splits on commas and cannot escape them; xplace.sbatch maps '+' back to ','.
    if any("+" in value for value in env.values()):
        raise ValueError("'+' is reserved as the comma placeholder in exported values")
    exports = ",".join(["ALL", f"XPLACE_TAG={tag}"]
                       + [f"{key}={value.replace(',', '+')}" for key, value in env.items()])
    depends = f"--dependency={dependency} " if dependency else ""
    excludes = f"--exclude={exclude} " if exclude else ""
    output = remote(f'cd "{REMOTE}/{tag}"\n'
                    f'sbatch --parsable --time={time_limit} --job-name={job_name} {depends}{excludes}'
                    f'--output="{REMOTE}/{tag}/logs/slurm-%j.out" --error="{REMOTE}/{tag}/logs/slurm-%j.err" '
                    f'--export={exports} upload/xplace.sbatch\n')
    job = output.strip().split(";")[0]
    record = {"tag": tag, "job": job, "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "env": env,
              "time_limit": time_limit, "job_name": job_name, "dependency": dependency, "exclude": exclude}
    (STAGING / tag / f"submission-{job}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def status(tag, job):
    return remote(f'squeue -h -j {job} -o "%i|%T|%M|%l|%R" 2>/dev/null || true\n'
                  f'sacct -n -X -j {job} -o JobID,State,ExitCode,Start,End,Elapsed,NodeList -P 2>/dev/null || true\n'
                  f'base="{REMOTE}/{tag}/job-{job}"\n'
                  'if [ -d "$base" ]; then\n'
                  '  cat "$base/exit-status" 2>/dev/null | sed "s/^/exit-status=/" || true\n'
                  '  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); '
                  '[print(p[\'phase\'], p[\'status\'], round(p[\'wall\'])) for p in d[\'phases\'][-6:]]" '
                  '"$base/out/phases.json" 2>/dev/null || true\n'
                  '  for f in "$base"/out/trace/*/progress.json; do [ -f "$f" ] && echo "$f: $(cat "$f")"; done\n'
                  '  tail -n 3 "$base/main.log" 2>/dev/null || true\n'
                  'fi\n', check=False)


def fetch(tag, job):
    destination = WORKSPACE / "evidence" / f"xplace-{tag}"
    destination.mkdir(parents=True, exist_ok=True)
    name = f"job-{job}.tar.gz"
    digest = remote(f'cd "{REMOTE}/{tag}"\n'
                    f'tar czf "{name}" "job-{job}" logs/slurm-{job}.out logs/slurm-{job}.err\n'
                    f'sha256sum "{name}" | cut -d" " -f1\n', timeout=1800).strip().splitlines()[-1]
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"paracloud:{REMOTE_RELATIVE}/{tag}/{name}",
                    str(destination / name)], check=True, timeout=3600)
    if sha256(destination / name) != digest:
        raise RuntimeError("Fetched archive checksum differs from the remote archive")
    with tarfile.open(destination / name) as archive:
        archive.extractall(destination)
    record = {"tag": tag, "job": job, "archive": name, "sha256": digest,
              "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (destination / f"fetch-{job}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("build")
    command.add_argument("tag")
    command.add_argument("--inputs", type=Path, nargs="*", default=[])
    command = sub.add_parser("upload")
    command.add_argument("tag")
    command = sub.add_parser("submit")
    command.add_argument("tag")
    command.add_argument("--time", required=True)
    command.add_argument("--job-name", default="xplace")
    command.add_argument("--dependency", help="e.g. afterany:<jobid>")
    command.add_argument("--exclude", help="comma-separated nodes to avoid, e.g. d1n41a23g03")
    command.add_argument("--env", nargs="*", default=[], help="KEY=VALUE pairs exported to the job")
    for name in ("status", "fetch"):
        command = sub.add_parser(name)
        command.add_argument("tag")
        command.add_argument("job")
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(args.tag, args.inputs), indent=2))
    elif args.command == "upload":
        print(json.dumps(upload(args.tag), indent=2))
    elif args.command == "submit":
        env = dict(item.split("=", 1) for item in args.env)
        print(json.dumps(submit(args.tag, env, args.time, args.job_name, args.dependency, args.exclude), indent=2))
    elif args.command == "status":
        print(status(args.tag, args.job))
    else:
        print(json.dumps(fetch(args.tag, args.job), indent=2))


if __name__ == "__main__":
    main()
