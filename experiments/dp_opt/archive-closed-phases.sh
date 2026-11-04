#!/usr/bin/env bash
set -eu
job="${1:?Specify job ID}"
stage="${2:?Specify independent sync stage}"
[[ "$job" =~ ^[1-9][0-9]*$ ]] || exit 2
prefix="$HOME/run/kvcached-paracloud/sync/"
[[ "$stage" == "$prefix"* ]] || exit 2
name="${stage#"$prefix"}"
[[ "$name" =~ ^[A-Za-z0-9_-][A-Za-z0-9._-]*$ ]] || exit 2
root="$HOME/run/kvcached-paracloud/state/live-trace-$job"
max_polls="${DP_OPT_PHASE_ARCHIVE_MAX_POLLS:-1201}"
[[ "$max_polls" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ ! -e "$stage/phase-watch.started" && ! -L "$stage/phase-watch.started" ]] || exit 2
mkdir "$stage/.phase-watch.lock"
finish() {
    status="$?"
    trap - EXIT
    printf 'job=%s\nhostname=%s\npid=%s\ntime=%s\nexit_status=%s\n' \
        "$job" "$(hostname)" "$$" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$status" > "$stage/phase-watch.exit"
    rmdir "$stage/.phase-watch.lock"
    exit "$status"
}
trap finish EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'job=%s\nhostname=%s\npid=%s\ntime=%s\n' \
    "$job" "$(hostname)" "$$" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$stage/phase-watch.started"
pack_phase() {
    phase="$1"
    source="$root/$phase"
    [[ ! -e "$stage/$phase.ready" && ! -L "$stage/$phase.ready" ]] || return 0
    [[ -d "$source" && ! -L "$source" ]] || return 0
    if [[ "$phase" == composite-profile ]]; then
        [[ -f "$source/fit.json" && -f "$source/workaware-profile-sources.json" ]] || return 0
        [[ -f "$root/phase.json" && ! -L "$root/phase.json" ]] || return 0
        current="$(cat "$root/phase.json")"
        case "$current" in
            *'"phase":"cache-fixed-original"'*|*'"phase":"lower-code-rate"'*) ;;
            *) return 0 ;;
        esac
    else
        [[ -f "$source/exit-status" && ! -L "$source/exit-status" ]] || return 0
        status="$(cat "$source/exit-status")"
        [[ "$status" =~ ^[0-9]+$ && "$status" -le 255 ]] || return 2
    fi
    [[ -z "$(find "$source" -type l -print -quit)" ]] || return 2
    attempt="$(mktemp -d "$stage/phase-$phase.XXXXXX")"
    (cd "$source"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$attempt/raw-before.sha256"
    [[ -s "$attempt/raw-before.sha256" ]] || return 2
    nice -n 10 tar -C "$source" -czf "$attempt/raw.tar.gz.partial" .
    gzip -t "$attempt/raw.tar.gz.partial"
    [[ -z "$(find "$source" -type l -print -quit)" ]] || return 2
    (cd "$source"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$attempt/raw-after.sha256"
    cmp "$attempt/raw-before.sha256" "$attempt/raw-after.sha256"
    mv "$attempt/raw.tar.gz.partial" "$attempt/raw.tar.gz"
    digest="$(sha256sum "$attempt/raw.tar.gz" | awk '{print $1}')"
    printf 'attempt=%s\nbundle_sha256=%s\n' "$(basename "$attempt")" "$digest" > "$stage/.$phase.ready"
    mv "$stage/.$phase.ready" "$stage/$phase.ready"
    printf 'closed_phase_archive_ready phase=%s time=%s\n' "$phase" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}
for iteration in $(seq 1 "$max_polls"); do
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    for phase in calibration-n567 composite-profile cache-fixed-original lower-code-rate; do
        pack_phase "$phase"
    done
    if accounting="$(sacct -n -X -j "$job" --format=JobID,State,ExitCode -P)"; then
        printf '%s\n' "$accounting"
        state="$(printf '%s\n' "$accounting" | awk -F'|' -v job="$job" '$1 == job {print $2}')"
        case "$state" in
            COMPLETED|FAILED|TIMEOUT|OUT_OF_MEMORY|CANCELLED*|NODE_FAIL|PREEMPTED) exit 0 ;;
        esac
    fi
    sleep 300
done
exit 124
