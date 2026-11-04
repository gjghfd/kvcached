import tarfile
from pathlib import PurePosixPath

import archive_trace_job


CONTROL_TARGETS = frozenset(("state-control/pilot/state-control-progress.json", "plain-no-move/progress.json"))


def extract_control_raw(bundle, staging, manifest, job, allow_missing_failed_progress=False):
    pointers = []
    with tarfile.open(bundle) as contents:
        members = contents.getmembers()
        names = [str(PurePosixPath(member.name)) for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate control archive member")
        regular = {str(PurePosixPath(member.name)) for member in members if member.isfile()}
        exit_member = next((member for member in members if member.isfile()
                            and str(PurePosixPath(member.name)) == "exit-status"), None)
        failed = False
        if allow_missing_failed_progress and exit_member is not None and exit_member.size <= 32:
            value = contents.extractfile(exit_member).read().strip()
            failed = value.isdigit() and 0 < int(value) < 256
        extracted = []
        for member in members:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or ".." in name.parts
                    or not (staging / member.name).resolve().is_relative_to(staging.resolve())):
                raise ValueError("Unsafe control archive member")
            if member.issym() and str(name) == "progress.json":
                target = PurePosixPath(member.linkname)
                if ".." in target.parts:
                    raise ValueError("Unsafe control progress target")
                if target.is_absolute():
                    anchor = ("run", "kvcached-paracloud", "state", f"live-trace-{job}")
                    matches = [index for index in range(len(target.parts))
                               if target.parts[index:index + 4] == anchor]
                    if len(matches) != 1:
                        raise ValueError("Control progress target must belong to this job")
                    target = PurePosixPath(*target.parts[matches[0] + 4:])
                if str(target) not in CONTROL_TARGETS:
                    raise ValueError("Control progress target is not a predeclared phase path")
                present = str(target) in regular
                if not present and not failed:
                    raise ValueError("Successful control archive lacks its regular progress target")
                pointers.append({"name": member.name, "remote_target": member.linkname,
                    "archived_target": str(target), "target_present": present,
                    "action": "recorded_not_extracted" if present else
                              "absent_failed_job_target_recorded_not_fabricated"})
                continue
            if not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe control archive member")
            extracted.append(member)
        contents.extractall(staging, members=extracted)
    archive_trace_job.verify_raw(staging, manifest)
    return pointers


def main():
    original = archive_trace_job.extract_raw
    archive_trace_job.extract_raw = extract_control_raw
    try:
        archive_trace_job.main()
    finally:
        archive_trace_job.extract_raw = original


if __name__ == "__main__":
    main()
