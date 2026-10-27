#!/usr/bin/env bash
set -u
job="${1:?Specify Slurm job ID}"
[[ "$job" =~ ^[1-9][0-9]*$ ]] || exit 2
root="$HOME/run/kvcached-paracloud/state/live-trace-$job"
max_polls="${DP_OPT_REMOTE_MAX_POLLS:-1201}"
[[ "$max_polls" =~ ^[1-9][0-9]*$ ]] || exit 2
mkdir -p "$root" || exit 2
for marker in remote-query.started remote-query.exit; do
    [[ ! -e "$root/$marker" && ! -L "$root/$marker" ]] || exit 2
done
mkdir "$root/.remote-query-watcher.lock" || exit 2
for marker in remote-query.started remote-query.exit; do
    if [[ -e "$root/$marker" || -L "$root/$marker" ]]; then
        rmdir "$root/.remote-query-watcher.lock"
        exit 2
    fi
done
watcher_hostname="$(hostname)" || exit 2
started_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')" || exit 2
run_id="$watcher_hostname:$$:$started_at"
write_marker() {
    local name="$1" marker_time="$2" marker_status="$3" temporary
    temporary="$(mktemp "$root/.$name.XXXXXX")" || return 1
    printf 'job=%s\nhostname=%s\ntime=%s\npid=%s\nrun_id=%s\nexit_status=%s\n' \
        "$job" "$watcher_hostname" "$marker_time" "$$" "$run_id" "$marker_status" > "$temporary" || return 1
    mv "$temporary" "$root/$name"
}
finish() {
    local exit_status="$?" exited_at
    trap - EXIT
    exited_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')" || exit 2
    printf 'remote-query exit hostname=%s time=%s exit_status=%s\n' \
        "$watcher_hostname" "$exited_at" "$exit_status"
    exec > /dev/null 2>&1
    rmdir "$root/.remote-query-watcher.lock" || exit 2
    write_marker remote-query.exit "$exited_at" "$exit_status" || exit 2
    exit "$exit_status"
}
trap finish EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
write_marker remote-query.started "$started_at" running || exit 2
for iteration in $(seq 1 "$max_polls"); do
    date -Is
    accounting="$(sacct -n -X -j "$job" --format=JobID,State,ExitCode,Start,End,Elapsed,NodeList -P)"
    printf '%s\n' "$accounting"
    for filename in phase.json progress.json error.json exit-status capacity/error.json capacity/capacity-setting.json transfer/progress.json transfer/error.json transfer/transfer-summary.json calibration/progress.json calibration/error.json; do
        if [[ -f "$root/$filename" ]]; then
            printf '%s\n' "$filename"
            cat "$root/$filename"
        fi
    done
    for filename in "$root"/cases/case-*/progress.json "$root"/cases/case-*/error.json; do
        if [[ -f "$filename" ]]; then
            printf '%s\n' "$filename"
            cat "$filename"
        fi
    done
    state="$(printf '%s\n' "$accounting" | awk -F'|' -v job="$job" '$1 == job {print $2}')"
    case "$state" in
        COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED) exit 0 ;;
    esac
    sleep 300
done
exit 124
