import hashlib
import shlex
import pytest

import roundtrip_state_campaign


@pytest.fixture
def staged_inputs(tmp_path):
    repo = tmp_path / "repo"
    root = tmp_path / "evidence"
    files = {
        "trace.json": b'{"requests": []}',
        "inputs/prompt-bank.json": b'{"tokens": [1, 2, 3]}',
        "inputs/lengths.json": b'{"output": 16557}',
    }
    for name, data in files.items():
        path = root / "frozen" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return repo, root, {}, "/remote/home with spaces", 273701


def parent_manifest(root, home, job):
    entries = []
    for path in sorted((root / "frozen").rglob("*")):
        if path.is_file():
            name = str(path.relative_to(root / "frozen"))
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append(f"{expected}  {home}/run/kvcached-paracloud/state/live-trace-{job}/frozen/{name}")
    return "\n".join(entries)


def test_reuse_verified_inputs_uploads_only_sources(monkeypatch, staged_inputs):
    repo, root, state, home, job = staged_inputs
    stage = home + "/run/kvcached-paracloud/sync/roundtrip-state-test"
    commands, sync_calls = [], []

    def checked_ssh(command):
        commands.append(command)
        return parent_manifest(root, home, job) if len(commands) == 1 else ""

    def protected_sync(actual_repo, actual_root, actual_state, **kwargs):
        sync_calls.append((actual_repo, actual_root, actual_state, kwargs))
        return stage, home

    monkeypatch.setattr(roundtrip_state_campaign, "checked_ssh", checked_ssh)
    monkeypatch.setattr(roundtrip_state_campaign, "protected_sync", protected_sync)
    assert roundtrip_state_campaign.stage_parent_inputs(*staged_inputs) == (stage, home)
    assert sync_calls == [(repo, root, state, {
        "files": roundtrip_state_campaign.SYNC_FILES,
        "prefix": "roundtrip-state",
        "upload_files": (),
    })]
    assert len(commands) == 2
    assert 'test -z "$(squeue -h -u "$USER")"' in commands[1]
    reused = state["remote_parent_inputs_reused_after_SHA_verification"]
    assert reused["job"] == job
    assert len(reused["files"]) == 3
    for name, expected in reused["files"].items():
        original = shlex.quote(home + f"/run/kvcached-paracloud/state/live-trace-{job}/frozen/" + name)
        target = shlex.quote(stage + "/upload/frozen/" + name)
        assert f"cp -p {original} {target}" in commands[1]
        assert f'sha256sum {original} |' in commands[1]
        assert f'sha256sum {target} |' in commands[1]
        assert commands[1].count(f"= {expected}") == 2
    assert (root / "remote-parent-frozen-inputs.sha256").read_text() == parent_manifest(root, home, job) + "\n"


@pytest.mark.parametrize("bad_manifest", ["", "wrong_sha", "missing_file", "reordered"])
def test_parent_mismatch_refuses_runtime_install(monkeypatch, staged_inputs, bad_manifest):
    repo, root, state, home, job = staged_inputs
    expected = parent_manifest(root, home, job)
    variants = {
        "": "",
        "wrong_sha": "0" * 64 + expected[64:],
        "missing_file": "\n".join(expected.splitlines()[:-1]),
        "reordered": "\n".join(reversed(expected.splitlines())),
    }
    calls = []
    monkeypatch.setattr(roundtrip_state_campaign, "checked_ssh", lambda command: variants[bad_manifest])
    monkeypatch.setattr(roundtrip_state_campaign, "protected_sync", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(RuntimeError, match="Remote immutable parent inputs differ"):
        roundtrip_state_campaign.stage_parent_inputs(*staged_inputs)
    assert not calls
    assert "remote_parent_inputs_reused_after_SHA_verification" not in state


def test_failed_remote_copy_is_not_recorded_as_verified(monkeypatch, staged_inputs):
    repo, root, state, home, job = staged_inputs
    commands = []

    def checked_ssh(command):
        commands.append(command)
        if len(commands) == 1:
            return parent_manifest(root, home, job)
        raise RuntimeError("remote copy or SHA verification failed")

    monkeypatch.setattr(roundtrip_state_campaign, "checked_ssh", checked_ssh)
    monkeypatch.setattr(roundtrip_state_campaign, "protected_sync", lambda *args, **kwargs: (home + "/stage", home))
    with pytest.raises(RuntimeError, match="copy or SHA"):
        roundtrip_state_campaign.stage_parent_inputs(*staged_inputs)
    assert "remote_parent_inputs_reused_after_SHA_verification" not in state
