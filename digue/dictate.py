"""Dictation daemon, toggle, and delivery orchestration."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from digue.recording import TakeState


def _dictate_lock() -> Any:
    """Serializes short state transitions between concurrent toggle processes."""
    import fcntl

    from digue.recording import _runtime_dir

    @contextlib.contextmanager
    def locked() -> Any:
        lock_path = _runtime_dir() / "digue.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return locked()


def _daemon_pid_file() -> Path:
    """The daemon is the digue process that started the recording and waits for it.

    Content: "<pid> <state> <starttime>". state is "starting" while startup is
    reserved, "recording" while the recorder is alive (a second toggle should
    stop it), and "delivering" while the take is being delivered (a second
    toggle must NOT stop it -- it starts a new take instead). starttime is the
    /proc starttime of the daemon: a pid alone is not an identity (the file
    outlives a SIGKILLed daemon and the kernel reuses pids), and signaling a
    recycled pid would SIGTERM an unrelated process of the same user.
    """
    from digue.recording import _runtime_dir

    return _runtime_dir() / "digue-daemon.pid"


def _write_daemon_state(daemon_pid: int, state: str) -> None:
    from digue.recording import _process_starttime, _write_state_file

    starttime = _process_starttime(daemon_pid) or "?"
    _write_state_file(_daemon_pid_file(), f"{daemon_pid} {state} {starttime}")


def _daemon_state() -> tuple[int, str, str] | None:
    """Returns (pid, state, starttime) from the daemon file, or None.

    A file without the starttime (older format, or a truncated write) has no
    verifiable identity and is treated as absent.
    """
    daemon_file = _daemon_pid_file()
    try:
        pid_text, state, starttime = daemon_file.read_text().split()
        return int(pid_text), state, starttime
    except (OSError, ValueError):
        return None


def _daemon_alive(entry: tuple[int, str, str]) -> bool:
    """True only if the pid is alive AND is still the process that wrote the file.

    A zombie (exited, not yet reaped by its parent) answers signal 0 and keeps
    its starttime, but cannot stop anything: it counts as dead, like in
    _group_alive.
    """
    from digue.recording import _take_identity_alive

    pid, _state, starttime = entry
    return _take_identity_alive(pid, starttime)


def _remove_daemon_state(daemon_pid: int) -> bool:
    """Removes this daemon's state while locked; callers must not hold the dictate lock."""
    with _dictate_lock():
        daemon_file = _daemon_pid_file()
        try:
            current_pid = daemon_file.read_text().split()[0]
            if int(current_pid) != daemon_pid:
                return False
            daemon_file.unlink()
            return True
        except (OSError, ValueError, IndexError):
            return False


def _is_daemon_alive() -> bool:
    entry = _daemon_state()
    return entry is not None and _daemon_alive(entry)


_got_sigterm = False
# Set by the SIGINT handler when the user presses Ctrl+c in a terminal.

_got_sigint = False


def _on_sigterm(_signum: int, _frame: object) -> None:
    global _got_sigterm
    _got_sigterm = True


def _on_sigint(_signum: int, _frame: object) -> None:
    global _got_sigint
    _got_sigint = True


DeliveryOutcome = Literal["delivered", "rescued", "empty", "retryable_failure"]

TERMINAL_OUTCOMES = ("delivered", "rescued", "empty")


@dataclass(frozen=True)
class DeliveryResult:
    """Terminal result of a delivery flow: the outcome alone does not carry the
    exit code nor where a rescued recording was kept (paste failed but the
    transcript and audio were saved -> rescued, exit 1; nothing salvaged ->
    retryable_failure, which recovery retries on the next toggle)."""

    outcome: DeliveryOutcome
    exit_code: int
    rescued_path: Path | None = None


def finish_dictation(
    config: dict[str, dict[str, Any]], rec_file: Path | None, limit_reached: bool = False, take_id: str | None = None
) -> DeliveryResult:
    """Runs the full delivery flow (transcribe, paste, archive) for a stopped recording.

    Called by the daemon once the recorder is dead: manual stop (second toggle
    signaled the daemon, which stopped the recorder) or duration limit (the
    watchdog safety killer stopped it).
    """
    from digue.audio import _archive_recording, _write_transcript, now_timestamp, rescue_recording
    from digue.container import server_url
    from digue.delivery import normalize_pasted_text, send_text
    from digue.notify import _stderr_is_tty, send_notification
    from digue.transcribe import TRANSCRIPTION_TIMEOUT, transcribe

    if rec_file is None:
        send_notification("Empty or missing audio file", timeout_ms=5000)
        return DeliveryResult(outcome="empty", exit_code=1)

    audio_dir = Path(config["dictate"]["audio_dir"])
    timestamp = now_timestamp()
    rescued_path: Path | None = None

    def archive_audio() -> bool:
        """Runs the post-delivery archiving (copy + compression, the slow part).

        Returns False when the archive failed; the raw WAV is then rescued
        (moved) and the path is left in rescued_path for the caller to report
        (rescued vs retryable_failure)."""
        nonlocal rescued_path
        archived, rescued_path = _archive_recording(config, rec_file, timestamp, take_id)
        return archived

    # Ctrl+c leaves "^C" echoed on the current terminal line; the \r redraw in
    # send_notification() would write over it and leave stray glyphs ("v"). Start a fresh
    # line for the transcription status.
    if _stderr_is_tty():
        print(file=sys.stderr, flush=True)
    message = (
        f"Limit reached ({config['dictate']['max_duration']}s), transcribing..." if limit_reached else "Transcribing..."
    )
    send_notification(message)
    try:
        url = server_url(config)
        language = config["transcribe"]["language"]
        prompt = config["transcribe"].get("prompt") or None
        timeout = int(config["transcribe"].get("timeout", TRANSCRIPTION_TIMEOUT))
        text = normalize_pasted_text(transcribe(url, rec_file, language, prompt=prompt, timeout=timeout))
    except Exception as exc:
        archived = rescue_recording(rec_file, audio_dir, timestamp, take_id)
        send_notification(f"Transcription failed: {exc}", timeout_ms=10000)
        if archived:
            print(f"Recording kept at: {archived}", file=sys.stderr)
            return DeliveryResult(outcome="rescued", exit_code=1, rescued_path=archived)
        return DeliveryResult(outcome="retryable_failure", exit_code=1)

    if not text:
        try:
            _write_transcript(audio_dir, timestamp, text, take_id=take_id)
        except Exception as exc:
            send_notification(f"Failed to save transcript: {exc}", timeout_ms=10000)
            print(text, file=sys.stderr)
            archived = rescue_recording(rec_file, audio_dir, timestamp, take_id)
            if archived is not None:
                return DeliveryResult(outcome="rescued", exit_code=1, rescued_path=archived)
            return DeliveryResult(outcome="retryable_failure", exit_code=1)
        audio_kept = archive_audio()
        send_notification("No speech detected", timeout_ms=5000)
        if audio_kept or rescued_path is not None:
            return DeliveryResult(outcome="empty", exit_code=0, rescued_path=rescued_path)
        return DeliveryResult(outcome="retryable_failure", exit_code=1)

    try:
        send_text(
            text,
            display_server=config["dictate"]["display_server"],
            input_mode=config["dictate"]["input_mode"],
        )
    except Exception as exc:
        send_notification(f"Paste failed: {exc}", timeout_ms=10000)
        try:
            text_path = _write_transcript(audio_dir, timestamp, text, take_id=take_id)
            print(f"Transcription saved to: {text_path}", file=sys.stderr)
        except Exception as save_exc:
            send_notification(f"Failed to save transcript: {save_exc}", timeout_ms=10000)
            print(text, file=sys.stderr)
        # Nothing reached the user: the outcome is only terminal if the audio
        # left the runtime dir (archived or rescued); otherwise the take state
        # stays. The retry only archives: the .txt above is what a recovery
        # reads as "delivered" (_delivered_transcript), and the user was told
        # where the text is -- pasting it later into whatever window has the
        # focus would be worse than not pasting it.
        if archive_audio() or rescued_path is not None:
            return DeliveryResult(outcome="rescued", exit_code=1, rescued_path=rescued_path)
        return DeliveryResult(outcome="retryable_failure", exit_code=1)
    verb = "Pasted" if config["dictate"]["input_mode"] == "paste" else "Typed"
    send_notification(f"{verb} ({len(text)} chars)", timeout_ms=3000)

    # From here on every outcome is terminal: the text was pasted, and a
    # retryable_failure would make a recovery paste it a second time. A
    # failure to save the transcript or to archive the audio is still an
    # error (exit 1), reported as delivered/rescued.
    try:
        text_path = _write_transcript(audio_dir, timestamp, text, take_id=take_id)
    except Exception as exc:
        send_notification(f"Failed to save transcript: {exc}", timeout_ms=10000)
        print(text, file=sys.stderr)
        archived = rescue_recording(rec_file, audio_dir, timestamp, take_id)
        if archived is not None:
            return DeliveryResult(outcome="rescued", exit_code=1, rescued_path=archived)
        return DeliveryResult(outcome="delivered", exit_code=1)
    # Transcribing... is a \r-redrawn line (no newline); break before this one.
    if _stderr_is_tty():
        print(file=sys.stderr, flush=True)
    print(f"Dictation done ({len(text)} chars): {text_path}", file=sys.stderr)
    if not archive_audio():
        if rescued_path is not None:
            return DeliveryResult(outcome="rescued", exit_code=1, rescued_path=rescued_path)
        return DeliveryResult(outcome="delivered", exit_code=1)
    return DeliveryResult(outcome="delivered", exit_code=0)


def dictate_toggle(config: dict[str, dict[str, Any]]) -> int:
    """Toggle recording/transcription. Returns exit code.

    First call starts the recorder and stays alive as a daemon, waiting for
    the recording to end (manual stop via a second toggle, duration limit, or
    recorder crash) to run the delivery flow. A second call while the daemon
    is alive signals SIGTERM and exits immediately: the daemon does the work,
    so the keybinding feels instant. Killing the daemon (pkill digue) leaves
    the recorder alive -- the next toggle transcribes what kept recording and
    returns without starting a new take.
    """
    from digue.container import ensure_server, is_server_running, server_not_running_hint
    from digue.notify import _stderr_is_tty, notify_close, send_notification
    from digue.recording import (
        _cancel_watchdog,
        _claim_orphan_take,
        _consume_recorder_stderr,
        _finish_owned_recorder,
        _mark_take_delivering,
        _recording_file_of,
        _take_state_file,
        _wait_recorder_end_daemon,
        start_recording,
    )

    daemon_pid = os.getpid()
    with _dictate_lock():
        entry = _daemon_state()
        if entry is not None and _daemon_alive(entry):
            current_daemon_pid, daemon_state, _starttime = entry
            if daemon_state == "recording":
                with contextlib.suppress(OSError):
                    os.kill(current_daemon_pid, 15)  # SIGTERM: daemon stops recording and delivers
                return 0
            if daemon_state == "starting":
                # Startup is already owned by another toggle. It has no recorder
                # to stop yet, so signaling it would abort or orphan the take.
                # First use may take minutes (image pull, model download): say so.
                send_notification("Still starting the server; recording begins when it is ready", timeout_ms=3000)
                return 0
            # A delivering daemon owns its old take. A new recording may replace
            # the global state; the old daemon removes it only if it still owns it.
        # Only with no current recording does the toggle look at orphan takes:
        # an old orphan must never keep the user from stopping the live one.
        claimed = _claim_orphan_take()
        if claimed is None:
            _write_daemon_state(daemon_pid, "starting")

    if claimed is not None:
        # This toggle recovers and returns; it does not record. It publishes
        # no daemon state on purpose: a recovering take with a live recoverer
        # is like a delivering one, so a concurrent toggle starts a new take.
        # Recording here too would leave that new take and this one competing
        # for the same daemon state (two recorders, one stop). An unexpected
        # exception in the delivery flow propagates and preserves the claimed
        # state and WAV: the next toggle retries.
        return _recover_orphan_takes(config, claimed)

    try:
        result = ensure_server(config)
        if result is None and not is_server_running(config):
            send_notification(server_not_running_hint(config), timeout_ms=5000)
            _remove_daemon_state(daemon_pid)
            return 1
    except Exception as exc:
        # no recorder is involved yet: a missing binary here is docker (or
        # ffmpeg during the model download), never pw-record/arecord
        send_notification(f"Cannot start the server: {exc}", timeout_ms=10000)
        _remove_daemon_state(daemon_pid)
        return 1
    try:
        limit = config["dictate"]["max_duration"]
        message = (
            f"Recording... (max {limit}s, press again to stop)" if limit > 0 else "Recording... (press again to stop)"
        )
        send_notification(message)
        # running from a terminal: the user can also Ctrl+c to stop and transcribe.
        # send_notification() redraws its line without a trailing newline on a TTY, so this
        # starts with \n to sit on its own line.
        if _stderr_is_tty():
            print("\nPress Ctrl+c to stop recording and transcribe", file=sys.stderr, flush=True)

        # Install handlers before publishing the daemon as recording: a second
        # toggle must never hit the default SIGTERM action while the recorder lives.
        global _got_sigterm
        import signal

        signal.signal(signal.SIGTERM, _on_sigterm)
        # Ctrl+c in a terminal means "stop and transcribe": the daemon handles
        # SIGINT itself (the global KeyboardInterrupt handler would discard the
        # take and leave the recorder running).
        signal.signal(signal.SIGINT, _on_sigint)
        processes = start_recording(config)
    except FileNotFoundError as exc:
        send_notification(
            f"Recorder not found: {exc.filename}. Install it (pipewire for pw-record, alsa-utils for arecord)",
            timeout_ms=10000,
        )
        _remove_daemon_state(daemon_pid)
        return 1
    except Exception as exc:
        send_notification(f"Failed to start recording: {exc}", timeout_ms=10000)
        _remove_daemon_state(daemon_pid)
        return 1
    recorder_pid = processes.recorder.pid
    _write_daemon_state(daemon_pid, "recording")
    # capture the recording file while the recorder is alive: the fd scan is
    # deterministic here and identifies the file after an unexpected death.
    rec_file = processes.rec_file or _recording_file_of(recorder_pid)
    outcome = _wait_recorder_end_daemon(processes.recorder, limit)
    _got_sigterm = False
    _got_sigint = False
    # the take is complete: mark delivering BEFORE stopping the recorder, so a
    # concurrent toggle never lands in the kill window (it would be dropped:
    # SIGTERM on a daemon that is already delivering is ignored by the gate).
    _write_daemon_state(daemon_pid, "delivering")
    if processes.take_id is not None:
        _mark_take_delivering(processes.take_id)
    notify_close()
    rec_file = _finish_owned_recorder(processes.recorder, rec_file)
    _cancel_watchdog(processes.watchdog)
    recorder_detail = _consume_recorder_stderr(processes)
    try:
        if outcome == "died" and rec_file is None:
            # The recorder exited on its own (PipeWire restarted, device
            # vanished) and left no audio: its stderr is the whole story, and
            # "Empty or missing audio file" would hide it.
            send_notification(f"Recorder exited unexpectedly: {recorder_detail}", timeout_ms=10000)
            own_result = DeliveryResult(outcome="empty", exit_code=1)
        else:
            if outcome == "died":
                print(
                    f"Recorder exited unexpectedly ({recorder_detail}); transcribing what was recorded",
                    file=sys.stderr,
                )
            own_result = finish_dictation(config, rec_file, limit_reached=outcome == "limit", take_id=processes.take_id)
    finally:
        _remove_daemon_state(daemon_pid)
    # Same rule as the recovery: the take state goes only after a terminal
    # outcome. A retryable failure (server down and the rescue failed too) or
    # an unexpected exception keeps the state and the WAV, and the next toggle
    # claims the take -- the daemon file is gone, so the take reads as orphan.
    if processes.take_id is not None and own_result.outcome in TERMINAL_OUTCOMES:
        _take_state_file(processes.take_id).unlink(missing_ok=True)
    return own_result.exit_code


def _recover_orphan_takes(config: dict[str, dict[str, Any]], claimed: TakeState) -> int:
    """Delivers the claimed orphan and rescues the remaining ones. Returns the exit code.

    The server must be up for the delivery; a failure to reach it is reported
    the same way a recording start would report it, and the claimed state stays
    (retried by the next toggle). Surplus orphans are only rescued once the
    oldest take reached a terminal outcome: a retryable failure leaves the
    claimed state in place and the next toggle claims it again.
    """
    from digue.container import ensure_server, is_server_running, server_not_running_hint
    from digue.notify import send_notification
    from digue.recording import _recover_claimed_take, _rescue_surplus_orphans, _take_state_file

    try:
        result = ensure_server(config)
        if result is None and not is_server_running(config):
            send_notification(server_not_running_hint(config), timeout_ms=5000)
            return 1
    except Exception as exc:
        send_notification(f"Cannot recover the previous recording: {exc}", timeout_ms=5000)
        return 1
    exit_code = _recover_claimed_take(config, claimed)
    if not _take_state_file(claimed.take_id).exists():
        rescued_paths = _rescue_surplus_orphans(config)
        if rescued_paths:
            send_notification(
                f"{len(rescued_paths)} recordings rescued to {config['dictate']['audio_dir']}", timeout_ms=10000
            )
    return exit_code


def cmd_dictate(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    if args.prompt is not None:
        config["transcribe"]["prompt"] = args.prompt
    return dictate_toggle(config)
