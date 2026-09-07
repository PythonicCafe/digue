"""Dictation audio archive: compress, save, rescue, and clean."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Any

from digue import DeliveryResult


def now_timestamp() -> str:
    """Shell-friendly timestamp for filenames: YYYYMMDD-HHMMSS (no ':' to escape)."""
    import datetime

    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def month_dir_for(timestamp: str) -> Path:
    """Returns the YYYY/MM relative path for a YYYYMMDD-HHMMSS timestamp.

    Derived from the timestamp itself (not from now()), so the .txt always
    lands beside the audio saved with the same timestamp even across midnight.
    """
    return Path(timestamp[:4]) / timestamp[4:6]


def _saved_stem(timestamp: str, take_id: str | None) -> str:
    """Stem for saved files: the take id suffix makes two takes that end in the
    same second unique; the exclusive write is the second line of defense."""
    return f"{timestamp}-{take_id}" if take_id else timestamp


# -- Dictation ------------------------------------------------------------------


def _compress_audio(rec_file: str | Path, audio_format: str, backend: str | None = None) -> Path:
    """Compresses a WAV recording in place. Returns the new path (rec_file swapped).

    audio_format: "wav" (no-op), "flac", or "opus".
    - flac: lossless, ~35% of WAV for speech, decodable by whisper-server natively
      (verified). Safe choice: the archive is bit-exact to what was transcribed.
    - opus: ~7% of WAV at 24 kbit/s (lossy). Speech quality is excellent, but the
      archive is not identical to the input; whisper-server rejects opus, so a
      retranscription goes through the ffmpeg fallback.
    Tries host ffmpeg first, then falls back to running ffmpeg inside the local
    container via stdin/stdout pipe when backend is not "remote" (any other
    value means local; only remote-or-not is looked at).

    The final name is reserved up front (exclusive creation): two takes can
    never overwrite each other's compressed file; a collision raises and the
    caller rescues the WAV. Whatever happens afterwards -- ffmpeg failure,
    timeout, a docker error -- the reservation and the temp file are dropped
    unless the compressed file was published, so a failure never leaves an
    empty .flac next to the WAV it kept.
    """
    if audio_format == "wav":
        return Path(rec_file)
    if audio_format not in ("flac", "opus"):
        raise KeyError(audio_format)

    rec_file = Path(rec_file)
    converted = rec_file.with_suffix(f".{audio_format}")
    temp_converted = converted.with_name(f".{converted.name}.{os.getpid()}.tmp")
    converted.touch(exist_ok=False)
    published = False
    try:
        published = _run_compression(rec_file, converted, temp_converted, audio_format, backend)
    finally:
        if not published:
            temp_converted.unlink(missing_ok=True)
            converted.unlink(missing_ok=True)  # drop the reservation, keep the WAV
    if not published:
        return rec_file
    rec_file.unlink(missing_ok=True)
    return converted


def _run_compression(
    rec_file: Path, converted: Path, temp_converted: Path, audio_format: str, backend: str | None
) -> bool:
    """Writes the compressed audio into temp_converted and publishes it as
    converted. Returns True when published; False (after a warning) when
    compression was not possible. Exceptions propagate to the caller."""
    import shutil
    import subprocess

    from digue.container import CONTAINER_NAME, container_status

    codec_args = {
        "flac": ["-c:a", "flac"],
        "opus": ["-c:a", "libopus", "-b:a", "24k"],
    }
    format_args = {
        "flac": ["-f", "flac"],
        "opus": ["-f", "ogg"],
    }

    if shutil.which("ffmpeg"):
        # The temp is ours alone (ffmpeg opens the path itself, so exclusivity
        # is the temp name); no -y: the final file is never ffmpeg's to overwrite.
        temp_converted.unlink(missing_ok=True)
        result = subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                str(rec_file),
                *codec_args[audio_format],
                *format_args[audio_format],
                str(temp_converted),
            ],
            capture_output=True,
            timeout=600,
        )
        if result.returncode == 0 and temp_converted.exists():
            os.replace(temp_converted, converted)
            return True
        print(
            f"Warning: ffmpeg failed to compress recording ({result.stderr.decode(errors='replace').strip()[:150]}); keeping WAV",
            file=sys.stderr,
        )
        return False

    if backend != "remote" and container_status() == "running":
        cmd = [
            "docker",
            "exec",
            "-i",
            CONTAINER_NAME,
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            *codec_args[audio_format],
            *format_args[audio_format],
            "pipe:1",
        ]
        result = subprocess.run(
            cmd,
            input=rec_file.read_bytes(),
            capture_output=True,
            timeout=600,
        )
        if result.returncode == 0 and result.stdout:
            temp_converted.unlink(missing_ok=True)
            with open(temp_converted, "xb") as temp_file:
                temp_file.write(result.stdout)
            os.replace(temp_converted, converted)
            return True
        print(
            f"Warning: ffmpeg failed to compress recording ({result.stderr.decode(errors='replace').strip()[:150]}); keeping WAV",
            file=sys.stderr,
        )
        return False

    print(
        f"Warning: ffmpeg not found, keeping the recording as WAV (install ffmpeg for {audio_format})",
        file=sys.stderr,
    )
    return False


def save_audio(
    rec_file: str | Path,
    audio_dir: str | Path,
    audio_format: str = "wav",
    timestamp: str | None = None,
    backend: str | None = None,
    take_id: str | None = None,
) -> tuple[Path, str]:
    """Copies audio to <audio_dir>/YYYY/MM/<timestamp>-<take_id>.<ext>. Returns (saved_path, timestamp).

    The timestamp comes from the caller (dictate_toggle generates it when the
    take stops, so the audio and its transcript share the same name even when
    archiving runs later). Without one, the current time is used. The copy is
    exclusive: a name collision raises instead of overwriting another take's
    file. audio_format "flac" or "opus" compresses the copy; the live recording
    file is kept as WAV and removed after saving.
    """
    audio_dir = Path(audio_dir)
    timestamp = timestamp or now_timestamp()
    month_dir = audio_dir / month_dir_for(timestamp)
    month_dir.mkdir(parents=True, exist_ok=True)
    source = Path(rec_file)
    source_suffix = source.suffix.lower() if source.suffix.lower() in {".wav", ".flac", ".opus"} else ".wav"
    saved = month_dir / f"{_saved_stem(timestamp, take_id)}{source_suffix}"
    _copy_file_exclusive(source, saved)
    if audio_format != "wav" and saved.suffix.lower() != f".{audio_format}":
        saved = _compress_audio(saved, audio_format, backend=backend)
    return saved, timestamp


def _copy_file_exclusive(source: Path, destination: Path) -> None:
    """Copies source to destination with exclusive creation ("xb"): a collision
    raises FileExistsError instead of silently overwriting another take's file."""
    import shutil

    with open(destination, "xb") as destination_file, source.open("rb") as source_file:
        shutil.copyfileobj(source_file, destination_file)


def rescue_recording(
    rec_file: str | Path, audio_dir: str | Path, timestamp: str, take_id: str | None = None
) -> Path | None:
    """Keeps a recording that could not be fully delivered. Never raises.

    Copies the WAV to <audio_dir>/YYYY/MM/<timestamp>-<take_id>.wav via an
    exclusive temp sibling + flush + fsync + exclusive link (runtime dir and
    audio-dir usually live on different filesystems, the destination must
    never be readable in a partial state, and an existing destination is
    never overwritten), then removes the origin -- only after the destination
    is valid. Any failure removes the temp, preserves the origin and reports
    on stderr.
    """
    import shutil

    rec_file = Path(rec_file)
    temp_archived: Path | None = None
    try:
        month_dir = Path(audio_dir) / month_dir_for(timestamp)
        month_dir.mkdir(parents=True, exist_ok=True)
        archived = month_dir / f"{_saved_stem(timestamp, take_id)}.wav"
        temp_archived = archived.with_name(f".{archived.name}.{os.getpid()}.tmp")
        with open(temp_archived, "xb") as temp_file, rec_file.open("rb") as source_file:
            shutil.copyfileobj(source_file, temp_file)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        # os.link fails with FileExistsError instead of replacing: the publish
        # step is as exclusive as the temp file (os.replace would clobber).
        os.link(temp_archived, archived)
        temp_archived.unlink()
        rec_file.unlink()
        return archived
    except Exception as rescue_exc:
        if temp_archived is not None:
            temp_archived.unlink(missing_ok=True)
        print(f"Failed to keep recording: {rescue_exc}; audio still at {rec_file}", file=sys.stderr)
        return None


def _write_transcript(audio_dir: Path, timestamp: str, text: str, take_id: str | None = None) -> Path:
    """Writes the transcript next to the recording: <audio_dir>/YYYY/MM/<timestamp>-<take_id>.txt.

    The month folder comes from the timestamp itself (not from now()), so the
    .txt always lands beside the audio saved with the same timestamp. The write
    is exclusive: a collision raises instead of overwriting another take's text.
    """
    text_path = audio_dir / month_dir_for(timestamp) / f"{_saved_stem(timestamp, take_id)}.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    with open(text_path, "xb") as text_file:
        text_file.write((text + "\n").encode())
    return text_path


def _archive_recording(
    config: dict[str, dict[str, Any]], rec_file: Path, timestamp: str, take_id: str | None
) -> tuple[bool, Path | None]:
    """Archives a delivered recording: copy + compression when save-audio is on
    (the slow part), then removes the live WAV. Returns (archived, rescued_path):
    on failure the raw WAV is rescued (moved) instead and the user is told."""
    from digue.container import _is_remote
    from digue.notify import send_notification

    audio_dir = Path(config["dictate"]["audio_dir"])
    try:
        if config["dictate"]["save_audio"]:
            # compression only needs remote-or-not: no hardware detection
            # (nvidia-smi/lspci) on every delivery
            save_audio(
                rec_file,
                audio_dir,
                config["dictate"].get("audio_format", "wav"),
                timestamp=timestamp,
                backend="remote" if _is_remote(config) else "local",
                take_id=take_id,
            )
        rec_file.unlink(missing_ok=True)
        return True, None
    except Exception as save_exc:
        rescued_path = rescue_recording(rec_file, audio_dir, timestamp, take_id)
        message = f"Failed to save audio: {save_exc}"
        if rescued_path:
            message += f"; uncompressed copy kept at {rescued_path}"
        send_notification(message, timeout_ms=10000)
        return False, rescued_path


def _delivered_transcript(audio_dir: Path, take_id: str) -> Path | None:
    """Returns the transcript already saved for a take, if any.

    finish_dictation writes the transcript right after pasting, so its
    presence proves the text reached the user: a recovery of a take whose
    daemon died afterwards (during the archive) must not paste it again.
    """
    return next(iter(sorted(audio_dir.glob(f"[0-9][0-9][0-9][0-9]/[0-9][0-9]/*-{take_id}.txt"))), None)


def _archive_recovered_take(config: dict[str, dict[str, Any]], rec_file: Path, transcript: Path) -> DeliveryResult:
    """Finishes a take whose text was already pasted and saved: archive only.

    The audio takes the transcript's timestamp and take id, so it lands next
    to the .txt. Every outcome is terminal (the text was delivered)."""

    from digue.notify import send_notification

    timestamp, _, take_id = transcript.stem.rpartition("-")
    send_notification("Recovering the previous recording (text already delivered)")
    # The daemon died mid-archive, so a partial .wav copy, an empty compressed
    # reservation or a .tmp of this same take may already sit next to the .txt.
    # Same stem means same take id: they are provably incomplete products of
    # this take, and the live WAV is the source of truth. Without this, the
    # exclusive archive collides and the good audio stays in the runtime dir.
    for leftover in transcript.parent.glob(f"*{transcript.stem}*"):
        if leftover.suffix not in (".txt", ".json"):
            leftover.unlink(missing_ok=True)
    archived, rescued_path = _archive_recording(config, rec_file, timestamp, take_id)
    if archived:
        return DeliveryResult(outcome="delivered", exit_code=0)
    if rescued_path is not None:
        return DeliveryResult(outcome="rescued", exit_code=1, rescued_path=rescued_path)
    return DeliveryResult(outcome="delivered", exit_code=1)


DICTATION_RECORDING_SUFFIXES = frozenset((".wav", ".flac", ".opus"))


def _dictation_files(audio_dir: Path, suffixes: frozenset[str]) -> list[Path]:
    """Lists <audio_dir>/YYYY/MM/<timestamp>[-<take_id>].<suffix> dictation files.

    Only that exact layout qualifies: audio-dir is user-configurable, and a
    recursive *.wav/*.flac/*.txt glob pointed at a music folder would remove
    the library. Both layouts are accepted: the pre-take-id stem
    (YYYYMMDD-HHMMSS, from now_timestamp()) and the current
    YYYYMMDD-HHMMSS-<16 hex chars>. Symlinks are skipped: clean unlinks what it
    lists, and deleting a symlink's target would destroy an outside file.
    """
    import re

    stem_pattern = re.compile(r"^\d{8}-\d{6}(-[0-9a-f]{16})?$")
    found = []
    for year_dir in audio_dir.glob("[0-9][0-9][0-9][0-9]"):
        for month_dir in year_dir.glob("[0-9][0-9]"):
            if not month_dir.is_dir():
                continue
            for path in month_dir.iterdir():
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and path.suffix.lower() in suffixes
                    and stem_pattern.match(path.stem)
                ):
                    found.append(path)
    return sorted(found)


def cmd_clean(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    """Removes dictation recordings and/or transcripts from the audio directory.

    Lists what it found and asks for confirmation; --force removes right away.
    --what selects what is removed: recordings, transcripts, or both (default).
    """

    audio_dir = Path(config["dictate"]["audio_dir"])
    if not audio_dir.exists():
        print(f"Audio directory does not exist: {audio_dir}", file=sys.stderr)
        return 0

    what = args.what
    recordings = _dictation_files(audio_dir, DICTATION_RECORDING_SUFFIXES) if what in ("recordings", "both") else []
    transcripts = _dictation_files(audio_dir, frozenset((".txt",))) if what in ("transcripts", "both") else []

    total_mb = sum(path.stat().st_size for path in recordings) / (1024 * 1024)
    print(f"Audio directory: {audio_dir}", file=sys.stderr)
    print(f"  Recordings: {len(recordings)} file(s), {total_mb:.1f} MB", file=sys.stderr)
    print(f"  Transcripts: {len(transcripts)} file(s)", file=sys.stderr)

    # A rescued take's .json is metadata of the recording, not its own
    # category: it is removed together with the recording of the same stem
    # (counted as one unit), never listed as a transcript, and a .json whose
    # recording is gone is preserved.
    recording_metadata = {path: path.with_suffix(".json") for path in recordings if path.with_suffix(".json").exists()}

    if not recordings and not transcripts:
        print("Nothing to remove.", file=sys.stderr)
        return 0

    total = len(recordings) + len(transcripts)
    if not args.force:
        for path in sorted(recordings):
            print(f"  {path.relative_to(audio_dir)}", file=sys.stderr)
            if path in recording_metadata:
                print(f"  {recording_metadata[path].relative_to(audio_dir)} (metadata)", file=sys.stderr)
        for path in sorted(transcripts):
            print(f"  {path.relative_to(audio_dir)}", file=sys.stderr)
        answer = input(f"Remove all {total} file(s)? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.", file=sys.stderr)
            return 1

    count = 0
    for path in recordings:
        path.unlink()
        count += 1
        metadata_path = recording_metadata.get(path)
        if metadata_path is not None:
            metadata_path.unlink()
    for path in transcripts:
        path.unlink()
        count += 1

    # Remove now-empty month/year directories (deepest first)
    for directory in sorted((parent for parent in audio_dir.rglob("*") if parent.is_dir()), reverse=True):
        with contextlib.suppress(OSError):
            directory.rmdir()
    with contextlib.suppress(OSError):
        audio_dir.rmdir()

    print(f"Removed {count} file(s).", file=sys.stderr)
    return 0
