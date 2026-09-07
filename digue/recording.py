"""Audio capture, take-state files, watchdog, and orphan recovery."""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import subprocess

# The daemon enforces max-duration (200ms poll); the detached watchdog is only
# a safety killer for a SIGKILLed daemon, so it fires this much later. With an
# equal deadline the watchdog won the race (measured: at 20s the recorder was
# already dead when the daemon checked) and the daemon saw "died", not "limit".
WATCHDOG_GRACE_SECONDS = 5

# -- Recording ----------------------------------------------------------------


def _runtime_dir() -> Path:
    import tempfile

    configured = os.environ.get("XDG_RUNTIME_DIR")
    runtime_dir = Path(configured) if configured else Path(tempfile.gettempdir()) / f"digue-{os.getuid()}"
    runtime_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if runtime_dir.stat().st_uid != os.getuid():
        raise RuntimeError(f"Runtime directory is not owned by the current user: {runtime_dir}")
    if not configured:
        runtime_dir.chmod(0o700)
    return runtime_dir


def _write_state_file(path: Path, content: str) -> None:
    """Publishes a small state file in one step (temp sibling + rename).

    Path.write_text truncates before writing, so a concurrent toggle could read
    an empty file and conclude there is no daemon/recorder.
    """

    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(content)
    os.replace(temp_path, path)


def _rec_file(suffix: str = ".wav") -> Path:
    """Returns a unique recording path without creating the audio file."""
    import secrets

    from digue.audio import now_timestamp

    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return _runtime_dir() / f"digue-{now_timestamp()}-{os.getpid()}-{secrets.token_hex(4)}{suffix}"


def _pid_alive(pid: int) -> bool:

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _resolve_recorder(recorder: str) -> str:
    """Resolves "auto" to pw-record when present, else arecord."""
    import shutil

    if recorder != "auto":
        return recorder
    if shutil.which("pw-record"):
        return "pw-record"
    if shutil.which("arecord"):
        return "arecord"
    return "pw-record"


def _cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    path = base / "digue"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pw_record_supports_flac() -> bool:
    """True when this pw-record's libsndfile was built with the flac container.

    Result is cached under XDG_CACHE_HOME, keyed by the binary path and mtime,
    so a libsndfile upgrade is picked up and a missing binary is not.
    """
    import shutil
    import subprocess

    binary = shutil.which("pw-record")
    if binary is None:
        return False
    path = Path(binary)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return False
    cache_path = _cache_dir() / "pw-record-containers"
    cache_key = f"{path.resolve()}\n{mtime_ns}\n"
    try:
        cached = cache_path.read_text()
    except OSError:
        cached = ""
    if cached.startswith(cache_key):
        names = {line.strip() for line in cached[len(cache_key) :].splitlines() if line.strip()}
        return "flac" in names
    try:
        result = subprocess.run(
            [str(path), "--list-containers"],
            capture_output=True,
            timeout=5,
            check=False,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    containers: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        containers.append(stripped.split(":", 1)[0].strip())
    with contextlib.suppress(OSError):
        cache_path.write_text(cache_key + "\n".join(containers) + "\n")
    return "flac" in containers


def _live_recording_suffix(config: dict[str, dict[str, Any]]) -> str:
    """Suffix of the live take file: .flac when pw-record can write it, else .wav.

    Both are LIVE_RECORDING_SUFFIXES: the take state and the /proc fd scan
    accept exactly these.
    """
    if config["dictate"]["audio_format"] != "flac":
        return ".wav"
    if _resolve_recorder(config["dictate"]["recorder"]) != "pw-record":
        return ".wav"
    if _pw_record_supports_flac():
        return ".flac"
    return ".wav"


def recording_command(
    rec_file: str | Path,
    recorder: str = "auto",
    device: str = "",
    container: str | None = None,
) -> list[str]:
    """Builds the argv that records mono 16 kHz s16 audio to rec_file.

    recorder: "auto" (pw-record if available, else arecord), "pw-record", or "arecord".
    device: empty keeps the system default; otherwise pw-record --target / arecord -D.
    container: pw-record --container (e.g. "flac"); inferred from a .flac suffix.
    """
    recorder = _resolve_recorder(recorder)
    rec_path = Path(rec_file)
    if container is None and rec_path.suffix.lower() == ".flac":
        container = "flac"
    if recorder == "pw-record":
        argv = ["pw-record", "--rate", "16000", "--channels", "1", "--format", "s16"]
        if device:
            argv.extend(["--target", device])
        if container:
            argv.extend(["--container", container])
        argv.append(str(rec_path))
        return argv
    if recorder == "arecord":
        argv = ["arecord", "-f", "S16_LE", "-r", "16000", "-c", "1"]
        if device:
            argv.extend(["-D", device])
        argv.append(str(rec_path))
        return argv
    raise RuntimeError(f"Unknown recorder: {recorder}. Use 'auto', 'pw-record', or 'arecord'.")


def _popen_recorder(argv: list[str], *, start_new_session: bool = True) -> subprocess.Popen[bytes]:
    """Starts the recorder. Raises RuntimeError if it exits non-zero immediately."""
    import subprocess
    import time

    err_path = _runtime_dir() / f"digue-recorder-err-{os.getpid()}"
    with err_path.open("w") as err_file:
        try:
            recorder = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=err_file,
                start_new_session=start_new_session,
            )
        except Exception:
            err_path.unlink(missing_ok=True)
            raise
        exit_code = recorder.poll()
        if exit_code is None:
            deadline = time.monotonic() + 0.15
            while recorder.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            exit_code = recorder.poll()
    if isinstance(exit_code, int) and exit_code != 0:
        detail = err_path.read_text(errors="replace").strip() or f"exit code {exit_code}"
        err_path.unlink(missing_ok=True)
        compact = " ".join(detail.split())
        if len(compact) > 400:
            compact = compact[:400] + "..."
        raise RuntimeError(f"{Path(argv[0]).name} failed: {compact}")
    err_path.unlink(missing_ok=True)
    return recorder


@dataclass(frozen=True)
class RecordingProcesses:
    """Processes owned by a recording daemon; the watchdog may be disabled."""

    recorder: subprocess.Popen[bytes]
    watchdog: subprocess.Popen[bytes] | None
    rec_file: Path | None = None
    take_id: str | None = None


TAKE_STATE_VERSION = 1

# Suffixes a live take may have in the runtime dir: .wav, or .flac when
# pw-record writes the flac container natively (_live_recording_suffix).
LIVE_RECORDING_SUFFIXES = frozenset((".wav", ".flac"))

TAKE_STATES = ("starting", "recording", "delivering", "recovering")


@dataclass(frozen=True)
class TakeState:
    """Persistent identity and lifecycle state for one recording take."""

    version: int
    take_id: str
    created_at_ns: int
    state: str
    rec_file: Path
    daemon_pid: int
    daemon_starttime: int
    recorder_pid: int | None = None
    recorder_starttime: int | None = None
    recoverer_pid: int | None = None
    recoverer_starttime: int | None = None

    def __post_init__(self) -> None:
        import re

        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version != TAKE_STATE_VERSION:
            raise ValueError(f"Unsupported take state version: {self.version}")
        if not isinstance(self.take_id, str) or re.fullmatch(r"[0-9a-f]{16}", self.take_id) is None:
            raise ValueError("take_id must contain exactly 16 lowercase hexadecimal characters")
        integer_fields: dict[str, int] = {
            "created_at_ns": self.created_at_ns,
            "daemon_pid": self.daemon_pid,
            "daemon_starttime": self.daemon_starttime,
        }
        recorder: int | None = self.recorder_pid
        recorder_start: int | None = self.recorder_starttime
        recoverer: int | None = self.recoverer_pid
        recoverer_start: int | None = self.recoverer_starttime
        optional_integer_fields: dict[str, int | None] = {
            "recorder_pid": recorder,
            "recorder_starttime": recorder_start,
            "recoverer_pid": recoverer,
            "recoverer_starttime": recoverer_start,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for optional_name, optional_value in optional_integer_fields.items():
            is_invalid = optional_value is not None and (
                isinstance(optional_value, bool) or not isinstance(optional_value, int) or optional_value <= 0
            )
            if is_invalid:
                raise ValueError(f"{optional_name} must be a positive integer or null")
        if (recorder is None) != (recorder_start is None):
            raise ValueError("recorder_pid and recorder_starttime must both be set or null")
        if (recoverer is None) != (recoverer_start is None):
            raise ValueError("recoverer_pid and recoverer_starttime must both be set or null")
        if self.state not in TAKE_STATES:
            raise ValueError(f"Invalid take state: {self.state}")
        if self.state == "starting" and (recorder is not None or recoverer is not None):
            raise ValueError("starting takes cannot have a recorder or recoverer")
        if self.state in ("recording", "delivering") and (recorder is None or recoverer is not None):
            raise ValueError(f"{self.state} takes require a recorder and cannot have a recoverer")
        if self.state == "recovering" and recoverer is None:
            raise ValueError("recovering takes require a recoverer")
        runtime_dir = _runtime_dir().resolve()
        try:
            relative_rec_file = self.rec_file.resolve().relative_to(runtime_dir)
        except ValueError:
            raise ValueError(f"rec_file must be inside the runtime directory: {self.rec_file}") from None
        if (
            relative_rec_file.parent != Path(".")
            or not relative_rec_file.name.startswith("digue-")
            or relative_rec_file.suffix not in LIVE_RECORDING_SUFFIXES
        ):
            raise ValueError("rec_file must be named digue-*.wav or .flac directly inside the runtime directory")


def new_take_id() -> str:
    import secrets

    return secrets.token_hex(8)


def _take_state_file(take_id: str) -> Path:
    return _runtime_dir() / f"digue-take-{take_id}.json"


def _write_take_state(take: TakeState) -> Path:
    import json

    state_path = _take_state_file(take.take_id)
    _write_state_file(state_path, json.dumps(_take_state_payload(take), separators=(",", ":")))
    return state_path


def _read_take_state(state_path: Path) -> TakeState | None:
    """Reads strict take JSON; malformed state is reported and left untouched."""
    import json

    required_fields = {
        "version",
        "take_id",
        "created_at_ns",
        "state",
        "rec_file",
        "daemon_pid",
        "daemon_starttime",
        "recorder_pid",
        "recorder_starttime",
        "recoverer_pid",
        "recoverer_starttime",
    }
    try:
        raw = json.loads(state_path.read_text())
        if not isinstance(raw, dict) or set(raw) != required_fields:
            raise ValueError("take state must contain exactly the required fields")
        if not isinstance(raw["rec_file"], str):
            raise ValueError("rec_file must be a string")
        take = TakeState(
            version=raw["version"],
            take_id=raw["take_id"],
            created_at_ns=raw["created_at_ns"],
            state=raw["state"],
            rec_file=Path(raw["rec_file"]),
            daemon_pid=raw["daemon_pid"],
            daemon_starttime=raw["daemon_starttime"],
            recorder_pid=raw["recorder_pid"],
            recorder_starttime=raw["recorder_starttime"],
            recoverer_pid=raw["recoverer_pid"],
            recoverer_starttime=raw["recoverer_starttime"],
        )
        if state_path != _take_state_file(take.take_id):
            raise ValueError("take_id does not match the state filename")
        return take
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Unreadable take state at {state_path}: {exc}", file=sys.stderr)
        return None


def _take_states() -> list[TakeState]:
    states = []
    for state_path in _runtime_dir().glob("digue-take-*.json"):
        take = _read_take_state(state_path)
        if take is not None:
            states.append(take)
    return sorted(states, key=lambda take: (take.created_at_ns, take.take_id))


ORPHAN_MIN_AGE_SECONDS = 60


def _take_age_seconds(take: TakeState) -> float:
    import time

    return (time.time_ns() - take.created_at_ns) / 1e9


def _expire_orphan_starting(config: dict[str, dict[str, Any]], take: TakeState) -> Path | None:
    """Conservative recovery for a "starting" take whose daemon died (e.g. it
    was killed between publishing the state and registering the recorder).

    No /proc/*/fd scanning (complex, racy, and it yields no identity). Rules:
    while the daemon is alive or the state is younger than
    ORPHAN_MIN_AGE_SECONDS, nothing happens; after that, a missing or
    empty WAV expires together with its state, and a non-empty WAV is rescued
    (never transcribed/pasted automatically: the recorder may still be
    writing). Returns the rescued path or None.
    """

    from digue.audio import now_timestamp, rescue_recording
    from digue.notify import send_notification

    if _pid_alive(take.daemon_pid) and _process_starttime(take.daemon_pid) == str(take.daemon_starttime):
        return None
    if _take_age_seconds(take) < ORPHAN_MIN_AGE_SECONDS:
        return None
    state_file = _take_state_file(take.take_id)
    if not take.rec_file.exists() or take.rec_file.stat().st_size == 0:
        take.rec_file.unlink(missing_ok=True)
        state_file.unlink(missing_ok=True)
        return None
    timestamp = now_timestamp()
    rescued = rescue_recording(take.rec_file, config["dictate"]["audio_dir"], timestamp, take.take_id)
    if rescued is None:
        return None  # origin preserved, state kept: the next toggle retries
    # same metadata contract as the surplus rescue: the JSON goes with the audio
    _archive_rescued_take_state(take, config["dictate"]["audio_dir"], timestamp)
    send_notification(
        f"A recording whose daemon died while starting was recovered; audio kept at {rescued}",
        timeout_ms=10000,
    )
    return rescued


def _take_identity_alive(pid: int | None, starttime: int | None) -> bool:
    """True only when pid is alive AND is still the process that published the
    identity: a pid alone is not an identity (pids get recycled)."""
    if pid is None or starttime is None:
        return False
    return _pid_alive(pid) and _process_starttime(pid) == str(starttime)


def _take_is_orphan(take: TakeState) -> bool:
    """True when the take's owner is provably dead: the daemon for a take that
    is starting/recording/delivering, or the recoverer for a recovering one."""
    if take.state == "recovering":
        return not _take_identity_alive(take.recoverer_pid, take.recoverer_starttime)
    return not _take_identity_alive(take.daemon_pid, take.daemon_starttime)


def _claim_orphan_take() -> TakeState | None:
    """Claims the oldest orphan take for recovery by the current process.

    Must be called while holding the dictate lock (discovery + transition only,
    nothing slow). Returns the take in state "recovering" with this process as
    recoverer, or None when there is nothing to recover: a recovering take
    with a live recoverer is not an orphan, so a concurrent toggle claims
    nothing and starts a new take instead. A claim always means work: an
    orphan without a recorder identity (a "starting" take) is only claimed
    once it is old enough for the conservative expiry rules to act on it
    (ORPHAN_MIN_AGE_SECONDS), so a toggle never returns after claiming a take
    it could do nothing with.
    """
    import dataclasses

    for take in _take_states():
        if not _take_is_orphan(take):
            continue
        if take.recorder_pid is None and _take_age_seconds(take) < ORPHAN_MIN_AGE_SECONDS:
            continue
        recoverer_starttime = _process_starttime(os.getpid())
        if recoverer_starttime is None:
            return None
        claimed = dataclasses.replace(
            take,
            state="recovering",
            recoverer_pid=os.getpid(),
            recoverer_starttime=int(recoverer_starttime),
        )
        _write_take_state(claimed)
        return claimed
    return None


def _recover_claimed_take(config: dict[str, dict[str, Any]], take: TakeState) -> int:
    """Recovers a claimed orphan take; called outside the dictate lock.

    The recorder identity published by the dead daemon is revalidated before
    signaling: stop_recording_pid stops a live recorder and no-ops on a dead
    or recycled one, so both cases converge on the delivery flow -- unless
    the transcript for this take already exists in audio-dir, which proves the
    text was pasted (the daemon died during the archive): then only the audio
    is archived, so a take is never pasted twice. The claimed
    state is removed only after a terminal outcome (delivered, rescued, empty):
    retryable failures and unexpected exceptions preserve state and WAV, and
    the next toggle finds a recovering take with a dead recoverer and retries.
    An empty WAV is final here, whatever the take's age: stop_recording_pid
    leaves the recorder dead (or already gone) and unlinks an empty file, and
    the no-speech path archives it -- keeping the state would make every toggle
    reclaim a take that no longer has audio. Only a "starting" take, which has
    no recorder identity to trust or stop, gets the age-based expiry rules.
    """
    from digue.audio import _archive_recovered_take, _delivered_transcript
    from digue.dictate import TERMINAL_OUTCOMES, finish_dictation

    if take.recorder_pid is None:
        # A rescued starting take keeps its audio and warns the user; not a
        # failure of this toggle (the new take's exit code still dominates).
        _expire_orphan_starting(config, take)
        return 0
    rec_file = stop_recording_pid(take.recorder_pid, take.rec_file, expected_starttime=take.recorder_starttime)
    transcript = _delivered_transcript(Path(config["dictate"]["audio_dir"]), take.take_id)
    if rec_file is not None and transcript is not None:
        # the dead daemon had already pasted and saved the text (it died during
        # the archive): archive the audio, never paste twice
        result = _archive_recovered_take(config, rec_file, transcript)
    else:
        result = finish_dictation(config, rec_file, take_id=take.take_id)
    if result.outcome not in TERMINAL_OUTCOMES:
        return result.exit_code
    _take_state_file(take.take_id).unlink(missing_ok=True)
    return result.exit_code


def _take_state_payload(take: TakeState, state: str | None = None) -> dict[str, Any]:
    """JSON payload of a take state; `state` overrides the lifecycle state (the
    archived "rescued" metadata is written outside the runtime dir, where the
    strict parser never sees it)."""
    return {
        "version": take.version,
        "take_id": take.take_id,
        "created_at_ns": take.created_at_ns,
        "state": state if state is not None else take.state,
        "rec_file": str(take.rec_file),
        "daemon_pid": take.daemon_pid,
        "daemon_starttime": take.daemon_starttime,
        "recorder_pid": take.recorder_pid,
        "recorder_starttime": take.recorder_starttime,
        "recoverer_pid": take.recoverer_pid,
        "recoverer_starttime": take.recoverer_starttime,
    }


def _archive_rescued_take_state(take: TakeState, audio_dir: str | Path, timestamp: str) -> None:
    """Moves the take state JSON next to the rescued recording (state "rescued").

    The JSON is metadata of the recording, not a live take state: it is kept
    beside the audio, and clean removes it
    together with the recording of the same stem (never on its own)."""
    import json

    from digue.audio import _saved_stem, month_dir_for

    state_file = _take_state_file(take.take_id)
    if not state_file.exists():
        return
    target = Path(audio_dir) / month_dir_for(timestamp) / f"{_saved_stem(timestamp, take.take_id)}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_state_file(target, json.dumps(_take_state_payload(take, state="rescued"), separators=(",", ":")))
    state_file.unlink(missing_ok=True)


def _rescue_surplus_take(config: dict[str, dict[str, Any]], take: TakeState) -> Path | None:
    """Rescues one surplus orphan take's audio without transcribing it.

    A live recorder is stopped through its published identity first; the WAV
    goes to rescue_recording (exclusive name per take id) and the take state
    JSON is archived next to it. Returns the rescued path, or None when there
    was nothing to rescue (the state is then removed) or the rescue failed
    (the state is kept, so the next toggle retries)."""
    from digue.audio import now_timestamp, rescue_recording

    rec_file: Path | None = take.rec_file
    if take.recorder_pid is not None:
        rec_file = stop_recording_pid(take.recorder_pid, take.rec_file, expected_starttime=take.recorder_starttime)
    else:
        # no recorder identity to trust: the conservative starting rules apply
        return _expire_orphan_starting(config, take)
    if rec_file is None or not rec_file.exists() or rec_file.stat().st_size == 0:
        _take_state_file(take.take_id).unlink(missing_ok=True)
        return None
    audio_dir = Path(config["dictate"]["audio_dir"])
    timestamp = now_timestamp()
    rescued = rescue_recording(rec_file, audio_dir, timestamp, take.take_id)
    if rescued is None:
        return None
    _archive_rescued_take_state(take, audio_dir, timestamp)
    return rescued


def _rescue_surplus_orphans(config: dict[str, dict[str, Any]]) -> list[Path]:
    """Rescues the remaining orphan takes after the oldest one was delivered.

    Each take is claimed (under the dictate lock) and rescued one by one, so a
    concurrent toggle never races a claim. Only the oldest orphan is ever
    transcribed and pasted; the rest keep their audio and metadata. Stops at
    the first take whose state could not be removed (rescue failure), leaving
    it for the next toggle."""
    from digue.dictate import _dictate_lock

    rescued_paths: list[Path] = []
    while True:
        # This loop runs outside the toggle's lock (the rescue itself is slow
        # I/O), so each claim takes it: _claim_orphan_take requires the lock.
        with _dictate_lock():
            surplus = _claim_orphan_take()
        if surplus is None:
            return rescued_paths
        rescued = _rescue_surplus_take(config, surplus)
        if rescued is not None:
            rescued_paths.append(rescued)
        if _take_state_file(surplus.take_id).exists():
            return rescued_paths


def start_recording(config: dict[str, dict[str, Any]]) -> RecordingProcesses:
    """Starts the recorder and safety watchdog, returning their owned handles.

    The take identity is published before the recorder exists (starting), and
    the recorder identity is published before the watchdog is spawned
    (recording): publishing the state is two syscalls (~50 us) while spawning
    the watchdog is fork+exec (~10 ms), so identity -- the thing recovery knows
    how to act on -- is exposed the soonest. On a Popen failure the state is
    removed (no WAV, no process). The recorder runs in a new process group so
    it survives a killed daemon; the take state carries the recorder identity
    as the recovery contract for a later invocation, while the live daemon
    retains Popen handles so it can reap both children.
    """
    import dataclasses
    import time

    rec_file = _rec_file(_live_recording_suffix(config))
    take_id = new_take_id()
    max_duration = config["dictate"]["max_duration"]
    daemon_starttime = _process_starttime(os.getpid())
    if daemon_starttime is None:
        raise RuntimeError("Cannot identify the current process via /proc")
    state = TakeState(
        version=TAKE_STATE_VERSION,
        take_id=take_id,
        created_at_ns=time.time_ns(),
        state="starting",
        rec_file=rec_file,
        daemon_pid=os.getpid(),
        daemon_starttime=int(daemon_starttime),
    )
    _write_take_state(state)
    argv = recording_command(
        rec_file,
        recorder=config["dictate"]["recorder"],
        device=str(config["dictate"].get("device") or ""),
        container="flac" if rec_file.suffix.lower() == ".flac" else None,
    )
    try:
        recorder = _popen_recorder(argv)
    except Exception:
        _take_state_file(take_id).unlink(missing_ok=True)
        raise
    recorder_starttime = _process_starttime(recorder.pid)
    if recorder_starttime is None:
        # The recorder died before its identity could be read: without a
        # verifiable recorder identity the state cannot move to "recording".
        _take_state_file(take_id).unlink(missing_ok=True)
    else:
        _write_take_state(
            dataclasses.replace(
                state, state="recording", recorder_pid=recorder.pid, recorder_starttime=int(recorder_starttime)
            )
        )
    watchdog = _spawn_limit_watchdog(recorder.pid, max_duration) if max_duration > 0 else None
    return RecordingProcesses(recorder=recorder, watchdog=watchdog, rec_file=rec_file, take_id=take_id)


def _process_starttime(pid: int, stat_path: Path | None = None) -> str | None:
    """Returns Linux /proc starttime, which distinguishes recycled PIDs."""
    path = stat_path or Path(f"/proc/{pid}/stat")
    try:
        stat = path.read_text()
        fields_after_comm = stat[stat.rindex(")") + 2 :].split()
        return fields_after_comm[19]
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def _process_pgrp(pid: int, stat_path: Path | None = None) -> int | None:
    """Returns the process group id from /proc/<pid>/stat (field 5), or None."""
    path = stat_path or Path(f"/proc/{pid}/stat")
    try:
        stat = path.read_text()
        fields_after_comm = stat[stat.rindex(")") + 2 :].split()
        return int(fields_after_comm[2])
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def _recorder_identity_valid(pid: int, expected_starttime: str | int | None) -> bool:
    """True when pid is still the recorder about to be signaled.

    killpg assumes the pid still leads its process group, and a recycled pid
    could be an unrelated process of the same user: the /proc starttime catches
    recycling and pgrp == pid catches a process that no longer leads a group
    (the recorder is spawned with start_new_session=True, so pid == pgid). No
    expectation always validates -- the owner's unreaped Popen child cannot be
    recycled; recovery paths must pass an identity.
    """
    if expected_starttime is None:
        return True
    return _process_starttime(pid) == str(expected_starttime) and _process_pgrp(pid) == pid


def _spawn_limit_watchdog(pgid: int, max_duration: int) -> subprocess.Popen[bytes]:
    """Spawns an identity-checking safety killer and returns its handle.

    The detached child survives a SIGKILLed daemon. Before signaling, it checks
    the same identity as _recorder_identity_valid (Linux /proc starttime and
    pgrp == pid) so a stale watchdog cannot kill a recycled PGID. It
    sleeps WATCHDOG_GRACE_SECONDS past max_duration so the daemon, which polls,
    always reaches the limit first.
    """
    import subprocess

    starttime = _process_starttime(pgid)
    if starttime is None:
        raise RuntimeError(f"Cannot identify recorder process {pgid}")
    script = """import os
import signal
import sys
import time
from pathlib import Path

pid = int(sys.argv[1])
expected_starttime = sys.argv[2]
time.sleep(int(sys.argv[3]))
try:
    fields = Path(f"/proc/{pid}/stat").read_text()
    fields = fields[fields.rindex(")") + 2:].split()
    # same identity as _recorder_identity_valid: starttime (recycled pid) and
    # still the leader of its own process group (killpg target)
    if fields[19] == expected_starttime and int(fields[2]) == pid:
        os.killpg(pid, signal.SIGTERM)
except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, ValueError, IndexError):
    pass
"""
    return subprocess.Popen(
        [sys.executable, "-c", script, str(pgid), starttime, str(max_duration + WATCHDOG_GRACE_SECONDS)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _cancel_watchdog(watchdog: subprocess.Popen[bytes] | None) -> None:
    """Cancels and reaps a watchdog after the owning daemon finishes normally."""
    import subprocess

    if watchdog is None:
        return
    if watchdog.poll() is None:
        watchdog.terminate()
    try:
        watchdog.wait(timeout=5)
    except subprocess.TimeoutExpired:
        watchdog.kill()
        watchdog.wait(timeout=5)


def _wait_recorder_end_daemon(recorder: subprocess.Popen[bytes], max_duration: int) -> str:
    """Waits on the daemon's own recorder handle and reaps spontaneous exits.

    An exit at or past the limit is reported as "limit" whoever stopped the
    recorder (the watchdog may have), so the limit notification is never lost.
    """
    import time

    from digue.dictate import _got_sigint, _got_sigterm

    start = time.monotonic()
    while True:
        if recorder.poll() is not None:
            recorder.wait(timeout=0)
            if max_duration > 0 and time.monotonic() - start >= max_duration:
                return "limit"
            return "died"
        if _got_sigterm:
            return "manual"
        if _got_sigint:
            return "interrupted"
        if max_duration > 0 and time.monotonic() - start >= max_duration:
            return "limit"
        time.sleep(0.2)


def _process_is_zombie(pid: int) -> bool:
    """True if /proc reports the process as exited but not yet reaped (state Z)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat[stat.rindex(")") + 2 :].split()[0] == "Z"
    except (OSError, ValueError, IndexError):
        return False


def _group_alive(pid: int) -> bool:
    """True while the recorder group still has a running process.

    A zombie recorder (exited, not yet reaped by the daemon's Popen.wait)
    still answers signal 0, so it is checked explicitly: otherwise every stop
    escalated to SIGKILL and paid the full grace period.
    """

    try:
        os.killpg(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return not _process_is_zombie(pid)


def _recording_file_of(pid: int) -> Path | None:
    """Finds the audio file a recording PID is writing, via /proc/<pid>/fd.

    The recorder argv carries the target path, so scanning its open file
    descriptors is the single source of truth -- no state file can drift out
    of sync (a timestamped rec_file name regenerated at stop time once made
    stop_recording check a file the recorder never wrote). Returns None when
    the process is already gone (its descriptors are closed).
    """

    # readlink yields the kernel's resolved path: compare against the resolved
    # runtime dir (XDG_RUNTIME_DIR may be a symlink), as TakeState does
    runtime_dir = _runtime_dir().resolve()
    try:
        fd_links = list(Path(f"/proc/{pid}/fd").iterdir())
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    for fd_link in fd_links:
        try:
            target = Path(os.readlink(fd_link))
        except OSError:
            continue
        if (
            target.parent == runtime_dir
            and target.suffix in LIVE_RECORDING_SUFFIXES
            and target.name.startswith("digue-")
        ):
            return target
    return None


def _validate_recording_file(rec_file: Path | None) -> Path | None:
    """Returns a non-empty recording, removing an empty file when present."""
    if rec_file is None or not rec_file.exists() or rec_file.stat().st_size == 0:
        if rec_file is not None:
            rec_file.unlink(missing_ok=True)
        return None
    return rec_file


def stop_recording_pid(
    pid: int, rec_file: Path | None = None, expected_starttime: str | int | None = None
) -> Path | None:
    """Stops the recorder process group `pid` and returns its audio file or None.

    Used by the take's owner (the daemon that started this recorder). With
    overlapping takes each daemon stops only its own recorder -- there is no
    global recorder state to stop, so a stop can never act on another take.
    The owner passes the rec_file captured while the recorder was alive;
    without it, the /proc/<pid>/fd scan (_recording_file_of) is the only
    confirmation, and when it finds nothing (recorder already dead, its
    descriptors closed) there is nothing to deliver. expected_starttime
    (recovery paths) revalidates before every killpg that the pid still is the
    recorder (same /proc starttime and still a process-group leader); a
    diverged identity is never signaled and the validated WAV is returned.
    """
    import time

    if rec_file is None:
        rec_file = _recording_file_of(pid)

    for signal in (15, 9):  # SIGTERM, then SIGKILL if it does not exit
        if not _recorder_identity_valid(pid, expected_starttime):
            break
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal)
        # Poll instead of a fixed sleep: pw-record exits within milliseconds,
        # and this wait sits between the hotkey and the transcription.
        deadline = time.monotonic() + 0.5
        while _group_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        if not _group_alive(pid):
            break

    return _validate_recording_file(rec_file)


def _finish_owned_recorder(recorder: subprocess.Popen[bytes], rec_file: Path | None) -> Path | None:
    """Stops a live owned recorder or validates output from an already reaped one."""
    if recorder.poll() is None:
        result = stop_recording_pid(recorder.pid, rec_file)
        recorder.wait(timeout=5)
        return result
    return _validate_recording_file(rec_file)


def record_to(
    output_path: str | Path,
    seconds: float,
    config: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Records from the microphone into output_path for seconds, then returns it.

    Honors [dictate] recorder, device and (for a .flac path) native FLAC when
    pw-record supports it. Raises RuntimeError if the recorder exits at start
    (bad --target, missing PCM, ...). Does not transcribe or paste.
    """
    import time

    from digue.config import load_config

    config = load_config() if config is None else config
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = str(config["dictate"].get("device") or "")
    container = "flac" if output_path.suffix.lower() == ".flac" else None
    argv = recording_command(
        output_path,
        recorder=config["dictate"]["recorder"],
        device=device,
        container=container,
    )
    proc = _popen_recorder(argv, start_new_session=False)
    try:
        time.sleep(seconds)
    except BaseException:
        proc.terminate()
        raise
    proc.terminate()
    deadline = time.monotonic() + 1.0
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"recorder produced no audio at {output_path}")
    return output_path
