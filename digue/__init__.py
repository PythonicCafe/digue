#!/usr/bin/env python3
"""Local speech-to-text dictation and transcription using whisper.cpp."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from digue.recording import TakeState

__version__ = "0.1.0"

DEFAULT_PORT = 8178
DEFAULT_LANGUAGE = "auto"
DEFAULT_MODELS = {"nvidia": "large-v3-turbo", "amd": "large-v3-turbo", "intel": "large-v3-turbo", "cpu": "small"}
AVAILABLE_MODELS = ("tiny", "base", "small", "medium", "large-v3-turbo", "large-v3")
DEFAULT_MAX_RECORD_SECONDS = 300
BENCHMARK_TRANSCRIPTION_TIMEOUT = 300
BENCHMARK_RUNS = 3


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
    from digue.recording import _pid_alive, _process_is_zombie, _process_starttime

    pid, _state, starttime = entry
    return _pid_alive(pid) and _process_starttime(pid) == starttime and not _process_is_zombie(pid)


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


# Set by the SIGTERM handler when a second toggle signals the daemon.
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
    # Invariant: once send_text succeeded the outcome is terminal (delivered
    # or rescued), never retryable_failure -- a retry would paste twice.


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
    from digue.transcribe import transcribe

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
        text = normalize_pasted_text(transcribe(url, rec_file, language, prompt=prompt))
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
        _finish_owned_recorder,
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
    notify_close()
    rec_file = _finish_owned_recorder(processes.recorder, rec_file)
    _cancel_watchdog(processes.watchdog)
    try:
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


# -- Benchmark ----------------------------------------------------------------


def _benchmark_run(url: str, audio_path: str | Path, language: str, runs: int) -> list[tuple[int, str]]:
    """Runs N transcription requests and returns list of (elapsed_ms, text)."""
    import time

    from digue.transcribe import transcribe

    with contextlib.suppress(Exception):
        transcribe(url, audio_path, language, timeout=BENCHMARK_TRANSCRIPTION_TIMEOUT)

    results = []
    for _run in range(runs):
        start = time.perf_counter()
        text = transcribe(url, audio_path, language, timeout=BENCHMARK_TRANSCRIPTION_TIMEOUT)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        results.append((elapsed_ms, text))
    return results


def run_benchmark(audio_path: str | Path, config: dict[str, dict[str, Any]]) -> None:
    """Benchmarks different backend/model combinations with the same audio."""
    from digue.container import (
        _wait_for_server,
        container_exists,
        create_container,
        download_model,
        preserve_container_for_benchmark,
        remove_container,
        resolve_backend,
        resolve_image,
        server_url,
    )

    models_dir = Path(config["server"]["data_dir"]) / "models"
    url = server_url(config)
    language = config["transcribe"]["language"]
    # The configured backend wins over detection: a forced "cpu" (with an
    # image override for a CPU the default image cannot run on) must not be
    # bypassed here.
    detected = resolve_backend(config)

    print("digue benchmark", file=sys.stderr)
    print(f"Audio: {audio_path}", file=sys.stderr)
    print(f"Backend: {detected}", file=sys.stderr)
    print(f"Runs per case: {BENCHMARK_RUNS}", file=sys.stderr)
    print(file=sys.stderr)

    for model in ("small", "large-v3-turbo"):
        model_path = models_dir / f"ggml-{model}.bin"
        if not model_path.exists():
            download_model(model, models_dir)
            print(file=sys.stderr)

    # Build test cases based on detected backend
    test_cases = [("CPU / small", "cpu", "small")]
    if detected != "cpu":
        test_cases.append((f"{detected.upper()} / small", detected, "small"))
    test_cases.append(("CPU / large-v3-turbo", "cpu", "large-v3-turbo"))
    if detected != "cpu":
        test_cases.append((f"{detected.upper()} / large-v3-turbo", detected, "large-v3-turbo"))

    all_results = []
    with preserve_container_for_benchmark():
        for label, backend, model in test_cases:
            print(f"=== {label} ===", file=sys.stderr)

            # The image override is global in config, but it only applies to
            # the backend the config resolved to; other cases fall back to
            # DOCKER_IMAGES (resolve_image uses the empty image).
            bench_server = {**config["server"]}
            if backend != detected:
                bench_server["image"] = ""
            bench_config = {**config, "server": bench_server, "models": {**config["models"], backend: model}}
            print(f"  Image: {resolve_image(backend, bench_config)}", file=sys.stderr)
            try:
                try:
                    create_container(bench_config, backend)
                except RuntimeError as exc:
                    print(f"  Skipped: {exc}", file=sys.stderr)
                    continue

                print("  Waiting for server...", file=sys.stderr, flush=True)
                if not _wait_for_server(config, verbose=True):
                    print("  Server failed to start, skipping", file=sys.stderr)
                    continue

                results = _benchmark_run(url, audio_path, language, BENCHMARK_RUNS)
                for idx, (elapsed_ms, text) in enumerate(results, 1):
                    print(f"  run {idx}: {elapsed_ms}ms", file=sys.stderr)
                if results:
                    avg_ms = sum(elapsed for elapsed, _ in results) // len(results)
                    print(f"  avg: {avg_ms}ms", file=sys.stderr)
                    print(f"  text: {results[-1][1]}", file=sys.stderr)
                    all_results.append((label, avg_ms))
                print(file=sys.stderr)
            finally:
                if container_exists():
                    remove_container()

    print(f"\n{'=' * 50}", file=sys.stderr)
    print("Summary", file=sys.stderr)
    print(f"{'=' * 50}", file=sys.stderr)
    for label, avg_ms in all_results:
        print(f"  {label:<35} {avg_ms}ms", file=sys.stderr)


def record_benchmark_audio(
    output_path: str | Path, duration_seconds: int = 10, config: dict[str, dict[str, Any]] | None = None
) -> None:
    """Records audio from microphone for benchmark, with the configured recorder."""
    import time

    from digue.recording import _popen_recorder, recording_command

    recorder = config["dictate"]["recorder"] if config else "auto"
    device = str(config["dictate"].get("device") or "") if config else ""
    print(f"Recording {duration_seconds}s from microphone...", file=sys.stderr)
    print("(speak something so there is content to transcribe)", file=sys.stderr)
    proc = _popen_recorder(
        recording_command(output_path, recorder=recorder, device=device),
        start_new_session=False,
    )
    time.sleep(duration_seconds)
    proc.terminate()
    time.sleep(0.5)
    print(f"Recorded: {output_path}", file=sys.stderr)


# -- CLI ----------------------------------------------------------------------


def _existing_dir(value: str) -> Path:
    """argparse type: validates that the path is an existing directory."""

    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory not found: {value}")
    return path


def _ensure_dir(value: str) -> Path:
    """argparse type: creates the directory if it doesn't exist."""

    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_parser() -> argparse.ArgumentParser:

    from digue.transcribe import RESPONSE_FORMATS

    parser = argparse.ArgumentParser(
        prog="digue",
        description="Local speech-to-text dictation and transcription using whisper.cpp",
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="path",
        help="Path to config.toml (default: ~/.config/digue/config.toml)",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"digue {__version__}",
    )
    parser.set_defaults(command=None)

    subparsers = parser.add_subparsers(dest="command", metavar="command")

    subparsers.add_parser("detect", help="Detect GPU backend and print it")

    sub_detect_language = subparsers.add_parser("detect-language", help="Detect the spoken language of an audio file")
    sub_detect_language.add_argument("audio", type=Path, help="Audio file to inspect")
    sub_detect_language.add_argument(
        "--json",
        action="store_true",
        help='Print {"language", "probability", "all"} JSON instead of just the code',
    )
    sub_detect_language.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show progress messages (ffmpeg conversion) on stderr",
    )

    sub_download = subparsers.add_parser("download", help="Download model(s)")
    sub_download.add_argument(
        "model",
        nargs="?",
        default=None,
        choices=AVAILABLE_MODELS,
        help=f"Model to download (default: auto-detect). Options: {', '.join(AVAILABLE_MODELS)}",
    )

    sub_server = subparsers.add_parser("server", help="Manage the whisper-server container")
    sub_server_sub = sub_server.add_subparsers(dest="server_action", metavar="action")
    # main() prints this help when no action is given (no default action).
    sub_server.set_defaults(server_parser=sub_server)

    sub_server_sub.add_parser("start", help="Start (or create) digue container")
    sub_server_sub.add_parser("stop", help="Stop digue container")
    sub_server_sub.add_parser("destroy", help="Stop and remove digue container")
    sub_server_sub.add_parser("status", help="Show server status")
    sub_dictate = subparsers.add_parser("dictate", help="Toggle recording/transcription")
    sub_dictate.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="Initial prompt to steer spelling of names/acronyms (overrides config transcribe.prompt)",
    )

    sub_transcribe = subparsers.add_parser("transcribe", help="Transcribe an audio file")
    sub_transcribe.add_argument("audio", type=Path, help="Audio file to transcribe")
    sub_transcribe.add_argument(
        "-o",
        "--output",
        metavar="path",
        default=None,
        help="Output file (default: stdout)",
    )
    sub_transcribe.add_argument(
        "-f",
        "--format",
        dest="response_format",
        choices=("vtt", "srt", "timestamps", "text"),
        default=None,
        help='Output format: "vtt", "srt", "timestamps" ([00:00:12] text lines) or "text" (plain). Default: config output-format, else "text"',
    )
    sub_transcribe.add_argument(
        "-l",
        "--language",
        default=None,
        help="Language code, e.g. pt, en (default: from config or auto)",
    )
    sub_transcribe.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="Initial prompt to steer spelling of names/acronyms (default: config transcribe.prompt)",
    )
    sub_transcribe.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show progress messages (conversion attempts etc.) on stderr",
    )

    sub_convert = subparsers.add_parser(
        "convert", help="Convert between subtitle/text formats (vtt, srt, timestamps, text)"
    )
    sub_convert.add_argument(
        "input",
        help='Input file, or "-" for stdin (then -f/--from-format is required)',
    )
    sub_convert.add_argument(
        "output",
        nargs="?",
        default=None,
        help='Output file (optional: "-" or omitted = stdout)',
    )
    sub_convert.add_argument(
        "-f",
        "--from-format",
        dest="from_format",
        choices=("vtt", "srt", "timestamps", "text"),
        default=None,
        help="Input format (required when input is -; otherwise guessed from the file extension)",
    )
    sub_convert.add_argument(
        "-t",
        "--to-format",
        dest="to_format",
        choices=("vtt", "srt", "timestamps", "text"),
        default=None,
        help="Output format: vtt, srt, timestamps, text (default: guessed from output extension, else text)",
    )

    sub_batch_transcribe = subparsers.add_parser(
        "batch-transcribe",
        help="Transcribe all audio files in a directory",
    )
    sub_batch_transcribe.add_argument("input_dir", type=_existing_dir, help="Directory with audio files")
    sub_batch_transcribe.add_argument("output_dir", type=_ensure_dir, help="Directory for transcription output")
    sub_batch_transcribe.add_argument(
        "-f",
        "--format",
        dest="response_format",
        choices=RESPONSE_FORMATS,
        default=None,
        help=f"Output format. Options: {', '.join(RESPONSE_FORMATS)} (default: config output-format)",
    )
    sub_batch_transcribe.add_argument(
        "-l",
        "--language",
        default=None,
        help="Language code, e.g. pt, en (default: from config or auto)",
    )

    sub_batch_simplify = subparsers.add_parser(
        "batch-simplify-vtt",
        help="Simplify all VTT files in a directory",
    )
    sub_batch_simplify.add_argument("input_dir", type=_existing_dir, help="Directory with VTT files")
    sub_batch_simplify.add_argument("output_dir", type=_ensure_dir, help="Directory for simplified output")

    sub_benchmark = subparsers.add_parser("benchmark", help="Compare backend performance")
    sub_benchmark.add_argument(
        "audio",
        nargs="?",
        type=Path,
        default=None,
        help="Audio file (records from microphone if not given)",
    )

    sub_config = subparsers.add_parser("config", help="Show or initialize the configuration")
    sub_config_sub = sub_config.add_subparsers(dest="config_action", metavar="action")
    # main() prints this help when no action is given (no default action).
    sub_config.set_defaults(config_parser=sub_config)

    sub_config_show = sub_config_sub.add_parser("show", help="Show the resolved configuration")
    sub_config_show.add_argument(
        "-f",
        "--format",
        dest="output_format",
        choices=("toml", "json"),
        default="toml",
        help="Output format (default: toml)",
    )

    sub_config_init = sub_config_sub.add_parser("init", help="Create the config file with commented defaults")
    sub_config_init.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite the config file if it already exists",
    )
    sub_config_init.add_argument(
        "-o",
        "--output",
        metavar="path",
        default=None,
        help="Where to write the config file (default: -c/--config path, else ~/.config/digue/config.toml)",
    )
    subparsers.add_parser("doctor", help="Check system dependencies and test Docker images")

    sub_clean = subparsers.add_parser("clean", help="Remove dictation recordings and/or transcripts")
    sub_clean.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Remove without asking for confirmation",
    )
    sub_clean.add_argument(
        "-w",
        "--what",
        choices=("recordings", "transcripts", "both"),
        default="both",
        help="What to remove (default: both)",
    )

    return parser


def cmd_dictate(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    if args.prompt is not None:
        config["transcribe"]["prompt"] = args.prompt
    return dictate_toggle(config)


def _format_extension(response_format: str) -> str:
    return {"text": ".txt", "vtt": ".vtt", "srt": ".srt", "timestamps": ".txt"}[response_format]


def cmd_benchmark(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:

    from digue.container import _is_remote
    from digue.recording import _runtime_dir

    if _is_remote(config):
        print("Error: benchmark creates local containers; it is not available with backend 'remote'", file=sys.stderr)
        return 1

    if args.audio:
        audio_path = args.audio
        if not audio_path.exists():
            print(f"Error: file not found: {audio_path}", file=sys.stderr)
            return 1
    else:
        # the runtime dir is private to the user; a fixed name in /tmp could be
        # a symlink planted by another local user
        audio_path = _runtime_dir() / "digue-bench.wav"
        record_benchmark_audio(audio_path, config=config)
        print(file=sys.stderr)

    run_benchmark(audio_path, config)
    return 0


def main() -> None:
    from digue.audio import cmd_clean
    from digue.config import _config_init, _config_path, cmd_config, load_config
    from digue.container import (
        DockerNotFoundError,
        cmd_detect,
        cmd_doctor,
        cmd_download,
        cmd_server_destroy,
        cmd_server_start,
        cmd_server_status,
        cmd_server_stop,
    )
    from digue.convert import cmd_convert
    from digue.notify import notify_close
    from digue.transcribe import cmd_batch_simplify_vtt, cmd_batch_transcribe, cmd_detect_language, cmd_transcribe

    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        # No default command on purpose: an accidental bare `digue` (wrong
        # keybinding, typo) would otherwise toggle recording out of nowhere.
        parser.print_help()
        sys.exit(1)

    command = args.command

    if command == "detect":
        sys.exit(cmd_detect(args))

    if command == "config":
        if args.config_action is None:
            # No default action, like the bare `digue`: show what is available.
            args.config_parser.print_help()
            sys.exit(1)
        if args.config_action == "init":
            sys.exit(_config_init(args))

    if command == "server" and args.server_action is None:
        # No default action, like the bare `digue`: show what is available.
        args.server_parser.print_help()
        sys.exit(1)

    try:
        config = load_config(args.config)
    except (OSError, ValueError) as exc:
        selected_path = Path(args.config).expanduser() if args.config else _config_path()
        print(f"Error: failed to load configuration {selected_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    commands = {
        "download": cmd_download,
        "server-start": cmd_server_start,
        "server-stop": cmd_server_stop,
        "server-destroy": cmd_server_destroy,
        "server-status": cmd_server_status,
        "dictate": cmd_dictate,
        "transcribe": cmd_transcribe,
        "detect-language": cmd_detect_language,
        "convert": cmd_convert,
        "batch-transcribe": cmd_batch_transcribe,
        "batch-simplify-vtt": cmd_batch_simplify_vtt,
        "benchmark": cmd_benchmark,
        "config": cmd_config,
        "clean": cmd_clean,
        "doctor": cmd_doctor,
    }

    handler = commands.get(f"{command}-{args.server_action}" if command == "server" else command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        sys.exit(handler(args, config))
    except DockerNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        notify_close()
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
