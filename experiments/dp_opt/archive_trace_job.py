import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path, PurePosixPath


def extract_raw(bundle, staging, manifest, job):
    pointers = []
    with tarfile.open(bundle) as contents:
        members = contents.getmembers()
        names = [str(PurePosixPath(member.name)) for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate raw archive member")
        regular = {str(PurePosixPath(member.name)) for member in members if member.isfile()}
        extracted = []
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not (staging / member.name).resolve().is_relative_to(staging.resolve()):
                raise ValueError("Unsafe raw archive member")
            if member.issym() and str(name) == "progress.json":
                target = PurePosixPath(member.linkname)
                anchor = ("run", "kvcached-paracloud", "state", f"live-trace-{job}")
                matches = [index for index in range(len(target.parts)) if target.parts[index:index + 4] == anchor]
                if not target.is_absolute() or ".." in target.parts or len(matches) != 1:
                    raise ValueError("Unsafe raw progress pointer")
                relative = PurePosixPath(*target.parts[matches[0] + 4:])
                if len(relative.parts) != 2 or relative.name != "progress.json" or str(relative) not in regular:
                    raise ValueError("Raw progress pointer lacks an archived regular target")
                pointers.append({"name": member.name, "remote_target": member.linkname,
                                 "archived_target": str(relative), "action": "recorded_not_extracted"})
                continue
            if not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe raw archive member")
            extracted.append(member)
        contents.extractall(staging, members=extracted)
    verify_raw(staging, manifest)
    return pointers


def verify_raw(destination, manifest):
    entries = []
    for line in manifest.splitlines():
        digest, filename = line.split(maxsplit=1)
        path = destination / filename
        if not path.resolve().is_relative_to(destination.resolve()) or path.is_symlink():
            raise ValueError("Unsafe raw manifest path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Downloaded evidence checksum mismatch: " + filename)
        entries.append(path.resolve())
    actual = {path.resolve() for path in destination.rglob("*") if path.is_file()}
    if not entries or len(set(entries)) != len(entries) or actual != set(entries):
        raise ValueError("Raw manifest file set differs from archive")


def archive_command(job, legacy_watcher_stopped=False):
    return f'''set -eu
accounting="$(sacct -n -X -j {job} --format=JobID,State,ExitCode,Submit,Start,End,Elapsed,NodeList,AllocTRES -P)"
printf '%s\\n' "$accounting"
state="$(printf '%s\\n' "$accounting" | awk -F'|' -v job={job} '$1 == job {{print $2}}')"
case "$state" in
    COMPLETED|FAILED|TIMEOUT|OUT_OF_MEMORY|CANCELLED*|NODE_FAIL|PREEMPTED) ;;
    *) printf 'Refuse to archive active job\\n' >&2; exit 2 ;;
esac
cd "$HOME/run/kvcached-paracloud/state/live-trace-{job}"
marker_mode=0
for marker in remote-query.expected remote-query.started remote-query.exit; do
    if [ -e "$marker" ] || [ -L "$marker" ]; then
        if [ ! -f "$marker" ] || [ -L "$marker" ]; then
            printf 'Unsafe shared watcher marker: %s\\n' "$marker" >&2
            exit 3
        fi
        marker_mode=1
    fi
done
barrier=no_watcher
if [ "$marker_mode" = 1 ]; then
    barrier=waiting
    for attempt in $(seq 1 20); do
        for marker in remote-query.expected remote-query.started remote-query.exit; do
            if {{ [ -e "$marker" ] || [ -L "$marker" ]; }} && {{ [ ! -f "$marker" ] || [ -L "$marker" ]; }}; then
                printf 'Unsafe shared watcher marker: %s\\n' "$marker" >&2
                exit 3
            fi
        done
        if [ -f remote-query.started ] && [ -f remote-query.exit ]; then
            if ! awk -F= -v job={job} '
                FNR == 1 {{ section++ }}
                {{
                    if (NF != 2 || $2 == "" || seen[section, $1]++) invalid=1
                    values[section, $1]=$2
                    counts[section]++
                }}
                END {{
                    if (invalid || counts[1] != 6 || counts[2] != 6) exit 1
                    if (values[1, "job"] != job || values[2, "job"] != job) exit 1
                    if (values[1, "hostname"] !~ /^[A-Za-z0-9._-]+$/) exit 1
                    if (values[1, "pid"] !~ /^[1-9][0-9]*$/) exit 1
                    if (values[1, "time"] == "" || values[2, "time"] == "") exit 1
                    if (values[1, "run_id"] == "") exit 1
                    if (values[1, "hostname"] != values[2, "hostname"] ||
                        values[1, "pid"] != values[2, "pid"] ||
                        values[1, "run_id"] != values[2, "run_id"]) exit 1
                    if (values[1, "exit_status"] != "running" ||
                        values[2, "exit_status"] !~ /^[0-9]+$/ ||
                        values[2, "exit_status"] > 255) exit 1
                }}' remote-query.started remote-query.exit; then
                printf 'Invalid or mismatched shared watcher markers; preserve evidence\\n' >&2
                exit 3
            fi
            barrier=shared_markers
            break
        fi
        sleep 20
    done
    if [ "$barrier" = waiting ]; then
        printf 'Shared watcher exit barrier incomplete; preserve evidence and retry later\\n' >&2
        exit 3
    fi
elif [ -e remote-query.log ] || [ -L remote-query.log ]; then
    if [ {int(legacy_watcher_stopped)} != 1 ]; then
        printf 'Legacy watcher has no shared exit markers: independently confirm it stopped, then use --legacy-watcher-stopped; host-local process lookup is not a barrier\\n' >&2
        exit 3
    fi
    printf 'Legacy compatibility: caller attests watcher stopped independently; full before/bundle/after SHA verification remains required\\n' >&2
    barrier=legacy_watcher_stopped_attestation
fi
printf '\\nWATCHER_BARRIER\\n%s\\nRAW_MANIFEST\\n' "$barrier"
find . -type f -print0 | sort -z | xargs -0 sha256sum
'''


def write_once(path, text):
    try:
        with path.open("x") as output:
            output.write(text)
    except FileExistsError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=int)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--legacy-watcher-stopped", action="store_true",
                        help="Attest a markerless legacy watcher was independently confirmed stopped; not via login-host-local process lookup")
    args = parser.parse_args()
    if args.job <= 0:
        parser.error("Require positive Slurm job ID")
    args.archive.mkdir(parents=True, exist_ok=True)
    destination = args.archive / f"live-trace-{args.job}"
    if destination.exists() and not args.resume:
        parser.error("Preserve existing raw archive")
    attempt = Path(tempfile.mkdtemp(
        prefix=f"archive-{args.job}-attempt-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-",
        dir=args.archive))
    attempt_id = attempt.name.removeprefix(f"archive-{args.job}-attempt-")
    options = ["-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3"]
    remote = f"$HOME/run/kvcached-paracloud/state/live-trace-{args.job}"
    try:
        result = subprocess.run(["ssh", *options, "paracloud", archive_command(args.job, args.legacy_watcher_stopped)],
                                text=True, capture_output=True, timeout=540)
    except subprocess.TimeoutExpired as error:
        for filename, text in (("query.stdout", error.stdout), ("query.stderr", error.stderr)):
            write_once(attempt / filename, text.decode() if isinstance(text, bytes) else text or "")
        write_once(attempt / "accounting.txt", (attempt / "query.stdout").read_text().split("WATCHER_BARRIER\n", 1)[0])
        raise
    write_once(attempt / "query.stdout", result.stdout)
    write_once(attempt / "query.stderr", getattr(result, "stderr", ""))
    header, separator, manifest = result.stdout.partition("RAW_MANIFEST\n")
    accounting, barrier_separator, barrier = header.partition("WATCHER_BARRIER\n")
    write_once(attempt / "accounting.txt", accounting)
    if separator:
        write_once(attempt / "raw.sha256", manifest)
    result.check_returncode()
    barrier = barrier.strip()
    if not separator or not barrier_separator or barrier not in ("shared_markers", "no_watcher", "legacy_watcher_stopped_attestation"):
        raise ValueError("Missing remote manifest or explicit watcher barrier")
    state = accounting.strip().splitlines()[0].split("|")[1].split()[0]
    if state not in ("COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL", "PREEMPTED"):
        raise ValueError("Do not archive an active job")
    manifest_path = args.archive / f"raw-{args.job}.sha256"
    if manifest_path.exists() and manifest_path.read_text() != manifest:
        raise ValueError("Remote manifest changed; preserve prior expected manifest")
    resuming = destination.exists()
    pointers = []
    if resuming:
        if manifest_path.read_text() != manifest:
            raise ValueError("Remote manifest changed; preserve prior raw archive")
        verify_raw(destination, manifest)
        bundles = sorted(args.archive.glob(f"raw-{args.job}-*.tar.gz"))
        bundle = bundles[-1] if bundles else None
    else:
        bundle = args.bundle or args.archive / f"raw-{args.job}-{attempt_id}.tar.gz"
        if args.bundle is None:
            with bundle.open("xb") as output:
                subprocess.run(["ssh", *options, "paracloud", f'tar -C "{remote}" -czf - .'],
                               stdout=output, check=True, timeout=1800)
        staging = args.archive / f"download-{args.job}-{attempt_id}"
        staging.mkdir()
        pointers = extract_raw(bundle, staging, manifest, args.job)
        after = subprocess.run(["ssh", *options, "paracloud",
                                f'cd "{remote}"; find . -type f -print0 | sort -z | xargs -0 sha256sum'],
                               check=True, text=True, capture_output=True, timeout=540).stdout
        write_once(attempt / "raw-after.sha256", after)
        if after != manifest:
            raise ValueError("Remote manifest changed during archive; preserve attempt evidence")
        pointer_text = json.dumps(pointers, indent=2) + "\n"
        write_once(attempt / "raw-pointers.json", pointer_text)
    missing_logs, log_sha256 = [], {}
    for extension in ("out", "err"):
        exists = subprocess.run(["ssh", *options, "paracloud",
                                 f'test -f "$HOME/run/kvcached-paracloud/logs/live-trace-{args.job}.{extension}"'],
                                timeout=120)
        if exists.returncode == 1:
            missing_logs.append(extension)
            continue
        exists.check_returncode()
        checksum = subprocess.run(["ssh", *options, "paracloud",
                                   f'sha256sum "$HOME/run/kvcached-paracloud/logs/live-trace-{args.job}.{extension}"'],
                                  check=True, text=True, capture_output=True, timeout=120).stdout.split()[0]
        target = args.archive / f"live-trace-{args.job}.{extension}"
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
                raise ValueError("Preserve differing local job log: " + extension)
        else:
            temporary = args.archive / f"live-trace-{args.job}.{extension}.partial-{attempt_id}"
            subprocess.run(["scp", *options, f"paracloud:~/run/kvcached-paracloud/logs/live-trace-{args.job}.{extension}",
                            str(temporary)], check=True, timeout=180)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != checksum:
                raise ValueError("Downloaded job log checksum mismatch: " + extension)
            temporary.rename(target)
        log_sha256[extension] = checksum
    if not resuming:
        staging.rename(destination)
        write_once(args.archive / f"raw-{args.job}-pointers.json", pointer_text)
    write_once(args.archive / f"accounting-{args.job}.txt", accounting)
    write_once(manifest_path, manifest)
    report = {"job": args.job, "state": state, "raw_files_verified": len(manifest.splitlines()),
              "raw": str(destination), "bundle_preserved": str(bundle), "missing_job_logs": missing_logs,
              "log_sha256": log_sha256, "resumed_verified_raw": resuming,
              "attempt": str(attempt), "raw_manifest": str(attempt / "raw.sha256"),
              "watcher_barrier": barrier,
              "pointer_metadata": str(args.archive / f"raw-{args.job}-pointers.json")}
    write_once(attempt / "report.json", json.dumps(report, indent=2) + "\n")
    (args.archive / f"archive-{args.job}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
