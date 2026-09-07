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
    import subprocess

__version__ = "0.1.0"

DEFAULT_PORT = 8178
DEFAULT_LANGUAGE = "auto"
DEFAULT_MODELS = {"nvidia": "large-v3-turbo", "amd": "large-v3-turbo", "intel": "large-v3-turbo", "cpu": "small"}
AVAILABLE_MODELS = ("tiny", "base", "small", "medium", "large-v3-turbo", "large-v3")
DEFAULT_MAX_RECORD_SECONDS = 300
# The daemon enforces max-duration (200ms poll); the detached watchdog is only
# a safety killer for a SIGKILLed daemon, so it fires this much later. With an
# equal deadline the watchdog won the race (measured: at 20s the recorder was
# already dead when the daemon checked) and the daemon saw "died", not "limit".
WATCHDOG_GRACE_SECONDS = 5
# Container formats whisper-server decodes natively (miniaudio: RIFF/PCM, fLaC, MP3,
# Ogg/Vorbis, AIFF). Verified empirically against whisper-server (ghcr.io main-vulkan
# image, built with WHISPER_COMMON_FFMPEG=OFF): wav, flac, mp3, ogg-vorbis and aiff
# return HTTP 200; opus-in-ogg (WhatsApp voice notes), m4a/AAC, mp4, webm, mka and
# wma return HTTP 400. Everything else is converted with ffmpeg before upload.
NATIVE_FORMATS = frozenset((".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"))
RESPONSE_FORMATS = ("text", "vtt", "srt", "timestamps")
TRANSCRIPTION_TIMEOUT = 120
BENCHMARK_TRANSCRIPTION_TIMEOUT = 300
BENCHMARK_RUNS = 3
# whisper.cpp g_lang (src/whisper.cpp): full names the server's JSON "language"
# field may carry, mapped to the two-letter codes.
LANGUAGE_FULL_TO_CODE: dict[str, str] = {
    "afrikaans": "af",
    "albanian": "sq",
    "amharic": "am",
    "arabic": "ar",
    "armenian": "hy",
    "assamese": "as",
    "azerbaijani": "az",
    "basque": "eu",
    "belarusian": "be",
    "bengali": "bn",
    "bosnian": "bs",
    "breton": "br",
    "bulgarian": "bg",
    "cantonese": "yue",
    "catalan": "ca",
    "chinese": "zh",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "faroese": "fo",
    "finnish": "fi",
    "french": "fr",
    "galician": "gl",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "haitian creole": "ht",
    "hausa": "ha",
    "hawaiian": "haw",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "icelandic": "is",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jw",
    "kannada": "kn",
    "kazakh": "kk",
    "khmer": "km",
    "korean": "ko",
    "lao": "lo",
    "latin": "la",
    "latvian": "lv",
    "lingala": "ln",
    "lithuanian": "lt",
    "luxembourgish": "lb",
    "macedonian": "mk",
    "malagasy": "mg",
    "malay": "ms",
    "malayalam": "ml",
    "maltese": "mt",
    "maori": "mi",
    "marathi": "mr",
    "mongolian": "mn",
    "myanmar": "my",
    "nepali": "ne",
    "norwegian": "no",
    "nynorsk": "nn",
    "occitan": "oc",
    "pashto": "ps",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "romanian": "ro",
    "russian": "ru",
    "sanskrit": "sa",
    "serbian": "sr",
    "shona": "sn",
    "sindhi": "sd",
    "sinhala": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "somali": "so",
    "spanish": "es",
    "sundanese": "su",
    "swahili": "sw",
    "swedish": "sv",
    "tagalog": "tl",
    "tajik": "tg",
    "tamil": "ta",
    "tatar": "tt",
    "telugu": "te",
    "thai": "th",
    "tibetan": "bo",
    "turkish": "tr",
    "turkmen": "tk",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "welsh": "cy",
    "yiddish": "yi",
    "yoruba": "yo",
}
AUDIO_EXTENSIONS = frozenset(
    (
        ".mp3",
        ".m4a",
        ".aac",
        ".wav",
        ".flac",
        ".ogg",
        ".opus",
        ".wma",
        ".aiff",
        ".aif",
        ".mka",
        ".mp4",
        ".webm",
        ".oga",
    )
)
DEFAULT_LAST_CUE_DURATION_MS = 2_000


@dataclass(frozen=True)
class SubtitleCue:
    """A subtitle cue whose boundaries are integer milliseconds."""

    start_ms: int
    end_ms: int | None
    text: str


# -- HTTP helpers -------------------------------------------------------------


def _multipart_request(
    url: str, audio_data: bytes, fields: dict[str, str], timeout: int | float, filename: str = "audio.wav"
) -> str:
    """Sends a multipart/form-data POST request using only stdlib."""
    import time
    import urllib.request

    boundary = f"----digue{os.getpid()}{time.time_ns()}"

    parts = []
    for field_name, field_value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{field_name}"\r\n\r\n{field_value}\r\n')
    file_part = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n"
        f"\r\n"
    )
    footer = f"\r\n--{boundary}--\r\n"

    body = b"".join(part.encode() for part in parts) + file_part.encode() + audio_data + footer.encode()

    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    response = urllib.request.urlopen(request, timeout=timeout)
    try:
        return str(response.read().decode())
    finally:
        response.close()


# -- Transcription ------------------------------------------------------------


def _convert_to_wav(audio_path: Path) -> bytes:
    """Converts audio to 16 kHz mono WAV in memory using ffmpeg. Returns the bytes.

    Nothing is written to disk: ffmpeg writes to stdout, which is captured
    (16 kHz mono s16 is ~32 KB/s, so a 300 s recording tops out around 10 MB).
    """
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            f"Format {audio_path.suffix} is not supported by whisper-server and ffmpeg is not installed. "
            "Install it with: sudo apt install ffmpeg"
        )
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to convert {audio_path.name}: {result.stderr.decode().strip()[:200]}")
    return result.stdout


def _send_audio(
    url: str,
    audio_path: Path,
    language: str,
    response_format: str,
    timeout: int | float,
    audio_data: bytes | None = None,
    prompt: str | None = None,
) -> str:
    """Uploads a single audio file to the server and returns the stripped response.

    token_timestamps=false disables the server's max_len=60 segment wrapping,
    which breaks segments on token boundaries (mid-word, e.g. "trans|crevendo").
    Verified against whisper-server: with it disabled, text output comes as one
    line per natural segment.
    prompt, when set, is sent as the whisper initial prompt (steers spelling of
    names/acronyms).
    """
    data = audio_data if audio_data is not None else audio_path.read_bytes()
    # The server's default language is "en" (server.cpp): omitting the field
    # would transcribe everything as English. "auto" is passed as-is and makes
    # whisper detect the language in the same pass (no extra cost).
    fields = {"response_format": response_format, "token_timestamps": "false", "language": language or "auto"}
    if prompt:
        fields["prompt"] = prompt
    return _multipart_request(url, data, fields, timeout, filename=audio_path.name).strip()


def _language_code(full_name: str) -> str:
    """Maps a whisper full language name ("portuguese") to its code ("pt").

    Codes pass through unchanged; unknown names are lowercased as-is (the
    server may add languages before this map is updated).
    """
    name = full_name.strip().lower()
    if name in LANGUAGE_FULL_TO_CODE:
        return LANGUAGE_FULL_TO_CODE[name]
    return name


def detect_language(
    url: str, audio_path: Path, timeout: int | float = TRANSCRIPTION_TIMEOUT, verbose: bool = False
) -> str:
    """Detects the spoken language of an audio file. Returns the language code (e.g. "pt").

    The server only reports the language in verbose_json (plain json returns
    {"text":""} even with detect_language=true). detect_language=true makes
    whisper return right after the encoder pass, skipping text decoding.
    Formats the server cannot decode are converted in memory (upfront for
    unknown extensions, as a retry after HTTP 400), like transcribe. Progress
    messages follow the verbose flag (default off: quiet library use).
    """
    import json
    import urllib.error

    def request(audio_data: bytes | None) -> str:
        fields = {"response_format": "verbose_json", "detect_language": "true", "language": "auto"}
        return _multipart_request(
            url,
            audio_data if audio_data is not None else audio_path.read_bytes(),
            fields,
            timeout,
            filename=audio_path.name,
        )

    def parse(response: str) -> str:
        payload = json.loads(response)
        return _language_code(payload["detected_language"])

    if audio_path.suffix.lower() not in NATIVE_FORMATS:
        if verbose:
            print(f"Converting {audio_path.name} with ffmpeg...", file=sys.stderr, flush=True)
        return parse(request(_convert_to_wav(audio_path)))
    try:
        return parse(request(None))
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            raise
        if verbose:
            print(
                f"Server rejected {audio_path.name} (HTTP 400). Trying ffmpeg conversion...",
                file=sys.stderr,
                flush=True,
            )
        return parse(request(_convert_to_wav(audio_path)))


def language_probabilities(
    url: str, audio_path: Path, timeout: int | float = TRANSCRIPTION_TIMEOUT, verbose: bool = False
) -> dict[str, Any]:
    """Detects the language and returns {"detected": (code, probability), "all": {code: probability}}.

    Runs a full verbose_json request (the server computes the probability
    table from the encoder's first-token logits and reports it in the
    response; a full transcription pass also runs server-side). Progress
    messages follow the verbose flag (default off: quiet library use).
    """
    import json
    import urllib.error

    def request(audio_data: bytes | None) -> str:
        fields = {
            "response_format": "verbose_json",
            "language": "auto",
            "token_timestamps": "false",
            "no_language_probabilities": "false",
        }
        return _multipart_request(
            url,
            audio_data if audio_data is not None else audio_path.read_bytes(),
            fields,
            timeout,
            filename=audio_path.name,
        )

    def parse(response: str) -> dict[str, Any]:
        payload = json.loads(response)
        detected = _language_code(payload["detected_language"])
        all_probs = {
            _language_code(name): float(prob) for name, prob in payload.get("language_probabilities", {}).items()
        }
        return {
            "detected": (detected, float(payload["detected_language_probability"])),
            "all": all_probs,
        }

    if audio_path.suffix.lower() not in NATIVE_FORMATS:
        if verbose:
            print(f"Converting {audio_path.name} with ffmpeg...", file=sys.stderr, flush=True)
        return parse(request(_convert_to_wav(audio_path)))
    try:
        return parse(request(None))
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            raise
        if verbose:
            print(
                f"Server rejected {audio_path.name} (HTTP 400). Trying ffmpeg conversion...",
                file=sys.stderr,
                flush=True,
            )
        return parse(request(_convert_to_wav(audio_path)))


def _post_process_subtitle(content: str, response_format: str, max_line_length: int, max_lines: int) -> str:
    """Cleans subtitle cues: strips outer spaces and wraps long cue text.

    whisper cues start with a space (" Álvaro, ..."); subtitles should not.
    Cues longer than max_line_length * max_lines are wrapped word-aligned over
    up to max_lines lines (overflow stays on the last line, no truncation).
    The block structure (index line, timestamp line, text line) of VTT and SRT
    is preserved.
    """
    import re

    timestamp_pattern = re.compile(r"^\s*\S+\s+-->\s+\S+")
    index_pattern = re.compile(r"^\s*\d+\s*$")
    output: list[str] = []
    cue_texts: list[str] = []
    awaiting_cue_start = True

    def flush() -> None:
        if not cue_texts:
            return
        text = " ".join(cue_texts).strip()
        cue_texts.clear()
        if not text:
            return
        output.extend(_wrap_cue_lines(text, max_line_length, max_lines))

    for line in content.splitlines():
        stripped = line.strip()
        is_timestamp = bool(timestamp_pattern.match(line)) and "-->" in stripped
        is_index = awaiting_cue_start and bool(index_pattern.match(line)) and response_format == "srt"
        is_header = stripped in ("WEBVTT", "NOTE") or stripped.startswith("Kind:") or stripped.startswith("Language:")
        if is_timestamp or is_index or is_header:
            flush()
            output.append(stripped)
            awaiting_cue_start = False
        elif stripped:
            cue_texts.append(stripped)
        else:
            flush()
            if output and output[-1] != "":
                output.append("")
            awaiting_cue_start = True

    flush()
    # Collapse runs of blank lines (wrap may have added doubles)
    cleaned: list[str] = []
    for line in output:
        if line == "" and cleaned and cleaned[-1] == "":
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip() + "\n"


def transcribe(
    url: str,
    audio_path: str | Path,
    language: str = "auto",
    response_format: str = "text",
    timeout: int | float = TRANSCRIPTION_TIMEOUT,
    verbose: bool = False,
    prompt: str | None = None,
    max_line_length: int = 42,
    max_lines: int = 2,
    wrap_cues: bool = True,
) -> str:
    """Sends audio to the server and returns the response (text, VTT, or SRT).

    Formats the server cannot decode are converted to WAV with ffmpeg in memory
    (nothing is written to disk), either upfront (unknown extension) or as a
    fallback after an HTTP 400. Status messages (conversion attempts etc.) print
    only when verbose=True -- the CLI default is silent.

    The "text" format is normalized to a single line (whisper segments start
    with a space and the server joins them with newlines; the segment breaks
    carry no semantic value - use vtt/srt when timestamps are needed).
    prompt, when set, is sent as the whisper initial prompt (steers spelling of
    names/acronyms).
    VTT/SRT cues are space-stripped and wrapped word-aligned to max_line_length
    chars over max_lines lines.
    """
    import urllib.error

    audio_path = Path(audio_path)
    if audio_path.suffix.lower() not in NATIVE_FORMATS:
        if verbose:
            print(f"Converting {audio_path.name} with ffmpeg...", file=sys.stderr, flush=True)
        wav_data = _convert_to_wav(audio_path)
        result = _send_audio(url, audio_path, language, response_format, timeout, audio_data=wav_data, prompt=prompt)
        return _finalize_output(result, response_format, max_line_length, max_lines, wrap_cues)
    try:
        result = _send_audio(url, audio_path, language, response_format, timeout, prompt=prompt)
        return _finalize_output(result, response_format, max_line_length, max_lines, wrap_cues)
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            raise
        if verbose:
            print(
                f"Server rejected {audio_path.name} (HTTP 400). Trying ffmpeg conversion...",
                file=sys.stderr,
                flush=True,
            )
        wav_data = _convert_to_wav(audio_path)
        result = _send_audio(url, audio_path, language, response_format, timeout, audio_data=wav_data, prompt=prompt)
        return _finalize_output(result, response_format, max_line_length, max_lines, wrap_cues)


def _with_final_newline(text: str) -> str:
    """Ends file content with exactly one newline (subtitle results already carry one)."""
    return text if text.endswith("\n") else text + "\n"


def _finalize_output(
    result: str, response_format: str, max_line_length: int, max_lines: int, wrap_cues: bool = True
) -> str:
    """Applies format-specific normalization to the server response."""
    if response_format == "text":
        return normalize_pasted_text(result)
    if response_format in ("vtt", "srt"):
        if not wrap_cues:
            # Cues keep their single-line text (only stripped)
            return _post_process_subtitle(result, response_format, max_line_length=10**9, max_lines=1)
        return _post_process_subtitle(result, response_format, max_line_length, max_lines)
    return result


# -- VTT simplification -------------------------------------------------------


def _wrap_cue_lines(text: str, max_line_length: int, max_lines: int) -> list[str]:
    """Greedy word-wrap of a cue text into at most max_lines lines of max_line_length.

    Words that do not fit are never truncated: overflow stays on the last line.
    """
    words = text.split()
    if not words:
        return []
    if len(text) <= max_line_length:
        return [text]
    if max_lines <= 1:
        return [" ".join(words)]

    lines: list[str] = []
    current = ""
    for index, word in enumerate(words):
        candidate = f"{current} {word}" if current else word
        if len(candidate) > max_line_length and current:
            lines.append(current)
            if len(lines) == max_lines - 1:
                # Overflow: last allowed line carries all remaining words
                lines.append(" ".join(words[index:]))
                return lines
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _strip_vtt_tags(text: str) -> str:
    """Removes inline VTT timing tags (YouTube word-level captions).

    Strips tags like ``<00:00:01.440>``, ``<c>``, ``</c>``.
    """
    import re

    text = text.replace("<c>", "").replace("</c>", "")
    text = re.sub(r"<[\d:.]+>", "", text)
    return text.strip()


def simplify_vtt(content: str, keep_timestamps: bool = True) -> str:
    """Simplifies a VTT file to timestamped plain text, removing duplications.

    With keep_timestamps=False, returns the joined text without timestamps.
    The deduplication only collapses exact repeats (the YouTube "rolling caption"
    pattern repeats the full previous line verbatim); distinct cues with similar
    text are kept.
    """
    result_lines = []
    last_clean_text = ""
    current_timestamp = None

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "WEBVTT":
            continue
        if stripped.startswith("Kind:") or stripped.startswith("Language:"):
            continue
        if stripped.isdigit():
            continue
        if "-->" in stripped:
            start_time = stripped.split("-->")[0].strip()
            if "." in start_time:
                start_time = start_time.split(".")[0]
            current_timestamp = start_time
            continue

        if current_timestamp is None:
            continue

        clean = _strip_vtt_tags(stripped)
        if not clean:
            continue

        if clean == last_clean_text:
            continue

        result_lines.append(clean if not keep_timestamps else f"[{current_timestamp}] {clean}")
        last_clean_text = clean

    return "\n".join(result_lines)


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


def _rec_file(suffix: str = ".wav") -> Path:
    """Returns a unique recording path without creating the audio file."""
    import secrets

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
    """Suffix of the live take file: .flac when pw-record can write it, else .wav."""
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
            or not relative_rec_file.name.endswith(".wav")
        ):
            raise ValueError("rec_file must be named digue-*.wav directly inside the runtime directory")


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
        if target.parent == runtime_dir and target.suffix == ".wav" and target.name.startswith("digue-"):
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


# -- Clipboard ----------------------------------------------------------------


def detect_display_server() -> str | None:
    """Detects whether the session is Wayland or X11."""

    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return None


def send_text(text: str, display_server: str = "auto", input_mode: str = "paste") -> None:
    """Sends text to the focused window.

    input_mode "paste" copies to the clipboard and simulates Ctrl+V.
    input_mode "type" simulates keystrokes (useful in terminals, where the
    paste shortcut differs). Typing is slower and may drop characters in
    slow applications.
    Raises RuntimeError with actionable message on failure.

    The whole body runs under an exclusive flock ("digue-delivery.lock"):
    the clipboard is global, and two overlapping deliveries pasting within
    the same window would deliver one text twice and lose the other. The
    "type" mode has the sibling race (interleaved keystrokes into the
    focused window), so it is serialized too. This is a dedicated lock, not
    _dictate_lock: a delivery can take seconds (paste timeout is 5s) and
    must not block state transitions.
    """
    if display_server == "auto":
        detected = detect_display_server()
        if detected is None:
            raise RuntimeError("No DISPLAY or WAYLAND_DISPLAY set. Cannot access clipboard or send keystrokes.")
        display_server = detected

    with _delivery_lock():
        _send_text_locked(text, display_server, input_mode)


def _delivery_lock() -> Any:
    """Serializes deliveries (clipboard copy+paste or keystroke typing) between
    overlapping takes."""
    import fcntl

    @contextlib.contextmanager
    def locked() -> Any:
        lock_path = _runtime_dir() / "digue-delivery.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return locked()


def _send_text_locked(text: str, display_server: str, input_mode: str) -> None:
    """The delivery itself; the caller holds the delivery lock."""
    import subprocess

    if input_mode == "type":
        # The text goes through stdin, never argv: wtype rejects any unknown
        # -option ("Unknown parameter", it has no --no-newline flag) and
        # xdotool would parse a transcript starting with "-" as an option.
        if display_server == "wayland":
            type_cmd = ["wtype", "-"]
            type_pkg = "wtype"
        else:
            type_cmd = ["xdotool", "type", "--clearmodifiers", "--file", "-"]
            type_pkg = "xdotool"
        try:
            subprocess.run(type_cmd, input=text.encode(), capture_output=True, timeout=120, check=True)
        except FileNotFoundError:
            raise RuntimeError(f"{type_cmd[0]} not found. Install with: sudo apt install {type_pkg}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{type_cmd[0]} timed out. Is a {display_server} session running?")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"{type_cmd[0]} failed: {exc.stderr.decode().strip() if exc.stderr else 'unknown error'}"
            )
        return

    if display_server == "wayland":
        copy_cmd = ["wl-copy"]
        paste_cmd = ["wtype", "-M", "ctrl", "v", "-m", "ctrl"]
        copy_pkg = "wl-clipboard"
        paste_pkg = "wtype"
    else:
        copy_cmd = ["xclip", "-selection", "clipboard"]
        paste_cmd = ["xdotool", "key", "--clearmodifiers", "ctrl+v"]
        copy_pkg = "xclip"
        paste_pkg = "xdotool"

    try:
        subprocess.run(
            copy_cmd,
            input=text.encode(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(f"{copy_cmd[0]} not found. Install with: sudo apt install {copy_pkg}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{copy_cmd[0]} timed out. Is a {display_server} session running?")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{copy_cmd[0]} failed: {exc.stderr.decode().strip() if exc.stderr else 'unknown error'}")

    try:
        subprocess.run(paste_cmd, capture_output=True, timeout=5, check=True)
    except FileNotFoundError:
        raise RuntimeError(f"{paste_cmd[0]} not found. Install with: sudo apt install {paste_pkg}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{paste_cmd[0]} timed out. Is a {display_server} session running?")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{paste_cmd[0]} failed: {exc.stderr.decode().strip() if exc.stderr else 'unknown error'}")


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


def normalize_pasted_text(text: str) -> str:
    """Joins wrapped lines into a single clean line.

    Line breaks come from whisper segment boundaries (word-aligned once
    token_timestamps is disabled, see _send_audio), so joining with a single
    space is safe; mid-word splits do not occur anymore.
    """
    return " ".join(text.split())


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


def _dictate_lock() -> Any:
    """Serializes short state transitions between concurrent toggle processes."""
    import fcntl

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
    return _runtime_dir() / "digue-daemon.pid"


def _write_daemon_state(daemon_pid: int, state: str) -> None:
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


def finish_dictation(
    config: dict[str, dict[str, Any]], rec_file: Path | None, limit_reached: bool = False, take_id: str | None = None
) -> DeliveryResult:
    """Runs the full delivery flow (transcribe, paste, archive) for a stopped recording.

    Called by the daemon once the recorder is dead: manual stop (second toggle
    signaled the daemon, which stopped the recorder) or duration limit (the
    watchdog safety killer stopped it).
    """
    from digue.container import server_url
    from digue.notify import _stderr_is_tty, send_notification

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


def transcribe_file(
    audio_path: str | Path,
    config: dict[str, dict[str, Any]] | None = None,
    *,
    language: str | None = None,
    response_format: str | None = None,
    prompt: str | None = None,
    verbose: bool = False,
) -> str:
    """Ensures the server is up and transcribes audio_path. Returns the text.

    config None loads the user config. language / response_format / prompt
    default to the [transcribe] section. Raises RuntimeError if the server
    cannot be reached.
    """
    from digue.config import load_config
    from digue.container import ensure_server, is_server_running, server_not_running_hint, server_url

    config = load_config() if config is None else config
    ensure_server(config, silent=not verbose)
    if not is_server_running(config):
        raise RuntimeError(f"server is not running. {server_not_running_hint(config)}")
    transcribe_cfg = config["transcribe"]
    fmt = response_format or str(transcribe_cfg.get("output_format", "text"))
    wrap_subtitles = fmt != "timestamps"
    result = transcribe(
        server_url(config),
        audio_path,
        language or str(transcribe_cfg["language"]),
        "vtt" if fmt == "timestamps" else fmt,
        verbose=verbose,
        prompt=prompt if prompt is not None else (transcribe_cfg.get("prompt") or None),
        max_line_length=int(transcribe_cfg.get("max_line_length", 42)),
        max_lines=int(transcribe_cfg.get("max_lines", 2)),
        wrap_cues=wrap_subtitles,
    )
    if fmt == "timestamps":
        return _convert_content(result, "vtt", "timestamps")
    return result


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


def cmd_detect_language(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    """Detects the spoken language of an audio file (no transcription)."""
    from digue.container import ensure_server, is_server_running, server_not_running_hint, server_url

    audio_path: Path = args.audio
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        return 1
    if not audio_path.is_file():
        print(f"Error: not a file: {audio_path} (expected an audio file; got a directory?)", file=sys.stderr)
        return 1

    ensure_server(config, silent=True)
    if not is_server_running(config):
        print(f"Error: server is not running. {server_not_running_hint(config)}", file=sys.stderr)
        return 1

    url = server_url(config)
    try:
        if args.json:
            import json

            probs = language_probabilities(url, audio_path, verbose=args.verbose)
            detected_code, detected_prob = probs["detected"]
            print(json.dumps({"language": detected_code, "probability": detected_prob, "all": probs["all"]}, indent=2))
        else:
            print(detect_language(url, audio_path, verbose=args.verbose))
    except Exception as exc:
        print(f"Error: language detection failed: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_dictate(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    if args.prompt is not None:
        config["transcribe"]["prompt"] = args.prompt
    return dictate_toggle(config)


def cmd_transcribe(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:

    from digue.container import ensure_server, is_server_running, server_not_running_hint, server_url

    audio_path = args.audio
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        return 1
    if not audio_path.is_file():
        print(f"Error: not a file: {audio_path} (expected an audio file; got a directory?)", file=sys.stderr)
        return 1

    ensure_server(config, silent=True)
    if not is_server_running(config):
        print(f"Error: server is not running. {server_not_running_hint(config)}", file=sys.stderr)
        return 1

    language = args.language or config["transcribe"]["language"]
    prompt = args.prompt if args.prompt is not None else config["transcribe"].get("prompt", "")
    response_format = args.response_format or config["transcribe"].get("output_format", "text")
    max_line_length = int(config["transcribe"].get("max_line_length", 42))
    max_lines = int(config["transcribe"].get("max_lines", 2))
    url = server_url(config)

    # timestamps output is meant for reading on one screen: cues are not
    # wrapped, so each timestamp gets exactly one line with all its text.
    wrap_subtitles = response_format != "timestamps"
    try:
        result = transcribe(
            url,
            audio_path,
            language,
            "vtt" if response_format == "timestamps" else response_format,
            verbose=args.verbose,
            prompt=prompt,
            max_line_length=max_line_length,
            max_lines=max_lines,
            wrap_cues=wrap_subtitles,
        )

        if response_format == "timestamps":
            result = _convert_content(result, "vtt", "timestamps")

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(_with_final_newline(result))
            if args.verbose:
                print(f"Saved: {output_path}", file=sys.stderr)
        else:
            print(result)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _guess_format_from_extension(path: Path) -> str | None:
    """Guesses the content format from a file extension, or None if unknown."""
    return {
        ".vtt": "vtt",
        ".srt": "srt",
        ".txt": "timestamps",  # digue-generated timestamped or plain text
    }.get(path.suffix.lower())


def _parse_timestamped_text(content: str) -> list[tuple[str | None, str]]:
    """Parses digue-generated timestamped text into (timestamp, line) pairs.

    Lines matching "[HH:MM:SS] text" carry their timestamp; any other line is
    returned with timestamp None. Used by `digue convert` for timestamps->*
    conversions.
    """
    import re

    pairs: list[tuple[str | None, str]] = []
    pattern = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*(.*)$")
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = pattern.match(stripped)
        if match:
            pairs.append((match.group(1), match.group(2)))
        else:
            pairs.append((None, stripped))
    return pairs


def _parse_subtitle_timestamp(value: str) -> int:
    """Parses an SRT or VTT timestamp as integer milliseconds."""
    import re

    match = re.fullmatch(r"(?:(\d{2,}):)?(\d{2}):(\d{2})[.,](\d{3})", value)
    if not match:
        raise ValueError(f"invalid subtitle timestamp: {value}")
    raw_hours, raw_minutes, raw_seconds, raw_ms = match.groups()
    hours = int(raw_hours) if raw_hours is not None else 0
    minutes = int(raw_minutes)
    seconds = int(raw_seconds)
    milliseconds = int(raw_ms)
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid subtitle timestamp: {value}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1_000 + milliseconds


def _format_subtitle_timestamp(milliseconds: int, output_format: str) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    separator = "," if output_format == "srt" else "."
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}"


def _parse_subtitle_cues(content: str, input_format: str) -> list[SubtitleCue]:
    """Parses basic VTT/SRT cues, retaining boundaries and multiline text."""
    import re

    timing_pattern = re.compile(r"^(\S+)\s+-->\s+(\S+)(?:\s+.*)?$")
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cues: list[SubtitleCue] = []
    line_index = 0
    while line_index < len(lines):
        match = timing_pattern.match(lines[line_index].strip())
        if not match:
            line_index += 1
            continue
        start_ms = _parse_subtitle_timestamp(match.group(1))
        end_ms = _parse_subtitle_timestamp(match.group(2))
        if end_ms < start_ms:
            raise ValueError("subtitle cue ends before it starts")
        line_index += 1
        text_lines: list[str] = []
        while line_index < len(lines) and lines[line_index].strip():
            text_lines.append(
                _strip_vtt_tags(lines[line_index].strip()) if input_format == "vtt" else lines[line_index]
            )
            line_index += 1
        text = "\n".join(line for line in text_lines if line)
        if text:
            cues.append(SubtitleCue(start_ms=start_ms, end_ms=end_ms, text=text))
    if not cues and not _is_empty_subtitle(content, input_format):
        raise ValueError(f"input does not look like a {input_format.upper()} file")
    return cues


def _is_empty_subtitle(content: str, input_format: str) -> bool:
    """True for a subtitle file with no cues: blank, or a VTT with only its header.

    whisper-server answers a silent audio with a bare "WEBVTT" line, which is
    a valid, empty subtitle -- not malformed input.
    """
    stripped = content.strip()
    if not stripped:
        return True
    if input_format != "vtt":
        return False
    header, _, rest = stripped.partition("\n")
    if not header.upper().startswith("WEBVTT"):
        return False
    # After the header only metadata may follow: "Key: value" header lines,
    # then NOTE/STYLE/REGION blocks (blank-line separated). A block starting
    # with anything else is cue text that lost its timing line -- corrupt.
    blocks = [block for block in rest.strip().split("\n\n") if block.strip()]
    for index, block in enumerate(blocks):
        first_line = block.strip().split("\n", 1)[0].strip()
        if first_line.startswith(("NOTE", "STYLE", "REGION")):
            continue
        if index == 0 and all(":" in line for line in block.strip().splitlines()):
            continue
        return False
    return True


def _render_subtitle_cues(cues: list[SubtitleCue], output_format: str) -> str:
    lines = ["WEBVTT", ""] if output_format == "vtt" else []
    for cue_index, cue in enumerate(cues, 1):
        if cue.end_ms is None:
            raise ValueError("subtitle cue has no end time")
        if output_format == "srt":
            lines.append(str(cue_index))
        start = _format_subtitle_timestamp(cue.start_ms, output_format)
        end = _format_subtitle_timestamp(cue.end_ms, output_format)
        lines.extend((f"{start} --> {end}", cue.text, ""))
    return "\n".join(lines).rstrip() + "\n"


def _timestamp_pairs_to_cues(pairs: list[tuple[str | None, str]], output_format: str) -> list[SubtitleCue]:
    """Builds cues using the next start as end and two seconds for the last cue."""
    import re

    timestamp_pattern = re.compile(r"^(\d{2}):(\d{2}):(\d{2})$")
    raw_starts_and_text = [(timestamp, text) for timestamp, text in pairs if timestamp is not None]
    if not raw_starts_and_text:
        raise ValueError(f"cannot convert text without timestamps to {output_format.upper()}")
    starts_and_text: list[tuple[str, str]] = []
    for timestamp, text in raw_starts_and_text:
        if starts_and_text and starts_and_text[-1][0] == timestamp:
            starts_and_text[-1] = (timestamp, f"{starts_and_text[-1][1]} {text}".strip())
        else:
            starts_and_text.append((timestamp, text))
    starts: list[int] = []
    for timestamp, _text in starts_and_text:
        match = timestamp_pattern.fullmatch(timestamp or "")
        if not match:
            raise ValueError(f"invalid timestamp: {timestamp}")
        hours, minutes, seconds = (int(part) for part in match.groups())
        if minutes >= 60 or seconds >= 60:
            raise ValueError(f"invalid timestamp: {timestamp}")
        starts.append(((hours * 60 + minutes) * 60 + seconds) * 1_000)
    if any(next_start <= start for start, next_start in zip(starts, starts[1:], strict=False)):
        raise ValueError("timestamps must be strictly increasing")
    return [
        SubtitleCue(
            start_ms=start,
            end_ms=starts[index + 1] if index + 1 < len(starts) else start + DEFAULT_LAST_CUE_DURATION_MS,
            text=starts_and_text[index][1],
        )
        for index, start in enumerate(starts)
    ]


def _convert_content(content: str, from_format: str, to_format: str) -> str:
    """Converts content between vtt/srt/timestamps/text formats."""
    pairs: list[tuple[str | None, str]]
    if from_format in ("vtt", "srt"):
        cues = _parse_subtitle_cues(content, from_format)
        if to_format in ("vtt", "srt"):
            # Header-only VTT (silent audio) is an empty subtitle, not an error.
            return _render_subtitle_cues(cues, to_format)
        pairs = [
            (_format_subtitle_timestamp(cue.start_ms, "vtt").split(".")[0], " ".join(cue.text.split())) for cue in cues
        ]
    elif from_format == "timestamps":
        pairs = _parse_timestamped_text(content)
    else:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        pairs = [(None, line) for line in lines]
    if not pairs and to_format in ("vtt", "srt"):
        raise ValueError(f"input has no cues; cannot produce {to_format.upper()}")

    if to_format == "text":
        return normalize_pasted_text(" ".join(text for _timestamp, text in pairs))
    if to_format == "timestamps":
        return "\n".join(f"[{timestamp}] {text}" if timestamp else text for timestamp, text in pairs)
    if to_format in ("vtt", "srt"):
        return _render_subtitle_cues(_timestamp_pairs_to_cues(pairs, to_format), to_format)
    raise ValueError(f"unknown to-format: {to_format}")


def cmd_convert(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    """Converts between subtitle/text formats (vtt, srt, timestamps, text)."""

    from_format = args.from_format
    if args.input == "-":
        if not from_format:
            print("Error: -f/--from-format is required when input is -", file=sys.stderr)
            return 1
        content = sys.stdin.read()
        input_path: Path | None = None
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: file not found: {args.input}", file=sys.stderr)
            return 1
        if not input_path.is_file():
            print(
                f"Error: not a file: {input_path} (expected a subtitle or text file; got a directory?)",
                file=sys.stderr,
            )
            return 1
        if not from_format:
            from_format = _guess_format_from_extension(input_path)
            if not from_format:
                print(
                    f"Error: cannot guess input format from extension {input_path.suffix!r}; use -f/--from-format",
                    file=sys.stderr,
                )
                return 1
        content = input_path.read_text()

    output_is_stdout = not args.output or args.output == "-"
    to_format = args.to_format
    if not to_format:
        if not output_is_stdout and args.output:
            to_format = _guess_format_from_extension(Path(args.output))
        if not to_format:
            # No extension to guess from: plain text is the safest default
            to_format = "text"

    try:
        result = _convert_content(content, from_format, to_format)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if output_is_stdout:
        print(result, end="" if result.endswith("\n") else "\n")
        return 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result)
    print(f"Saved: {output_path}", file=sys.stderr)
    return 0


def _format_extension(response_format: str) -> str:
    return {"text": ".txt", "vtt": ".vtt", "srt": ".srt", "timestamps": ".txt"}[response_format]


def cmd_batch_transcribe(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    import time

    from digue.container import ensure_server, is_server_running, server_not_running_hint, server_url

    language = args.language or config["transcribe"]["language"]
    response_format = args.response_format or config["transcribe"]["output_format"]
    prompt = config["transcribe"]["prompt"]
    max_line_length = config["transcribe"]["max_line_length"]
    max_lines = config["transcribe"]["max_lines"]
    url = server_url(config)
    ext = _format_extension(response_format)

    audio_files = sorted(
        path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        print(f"No audio files found in {args.input_dir}", file=sys.stderr)
        return 1

    pending = []
    for audio_file in audio_files:
        output_file = args.output_dir / (audio_file.stem + ext)
        if not output_file.exists():
            pending.append((audio_file, output_file))

    skipped = len(audio_files) - len(pending)
    if skipped:
        print(f"Skipping {skipped} already transcribed file(s)", file=sys.stderr)
    if not pending:
        print("All files already transcribed", file=sys.stderr)
        return 0

    ensure_server(config, silent=True)
    if not is_server_running(config):
        print(f"Error: server is not running. {server_not_running_hint(config)}", file=sys.stderr)
        return 1

    succeeded = 0
    failed = 0
    wrap_subtitles = response_format not in ("timestamps", "text")
    for idx, (audio_file, output_file) in enumerate(pending, 1):
        print(f"[{idx}/{len(pending)}] {audio_file.name}...", file=sys.stderr, flush=True)
        start = time.perf_counter()
        temp_file = output_file.with_name(f".{output_file.name}.tmp")
        try:
            result = transcribe(
                url,
                audio_file,
                language,
                "vtt" if response_format == "timestamps" else response_format,
                prompt=prompt,
                max_line_length=max_line_length,
                max_lines=max_lines,
                wrap_cues=wrap_subtitles,
            )
            if response_format == "timestamps":
                result = _convert_content(result, "vtt", "timestamps")
            temp_file.write_text(_with_final_newline(result))
            temp_file.replace(output_file)
            succeeded += 1
            elapsed = time.perf_counter() - start
            print(f"  Saved: {output_file.name} ({elapsed:.1f}s)", file=sys.stderr)
        except Exception as exc:
            failed += 1
            temp_file.unlink(missing_ok=True)
            print(f"  Error: {exc}", file=sys.stderr)

    print(
        f"Done: {succeeded} succeeded, {failed} failed, {skipped} skipped. Output: {args.output_dir}", file=sys.stderr
    )
    return 1 if failed else 0


def cmd_batch_simplify_vtt(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    vtt_files = sorted(path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".vtt")
    if not vtt_files:
        print(f"No VTT files found in {args.input_dir}", file=sys.stderr)
        return 1

    pending = []
    for vtt_file in vtt_files:
        output_file = args.output_dir / (vtt_file.stem + ".txt")
        if not output_file.exists():
            pending.append((vtt_file, output_file))

    skipped = len(vtt_files) - len(pending)
    if skipped:
        print(f"Skipping {skipped} already simplified file(s)", file=sys.stderr)
    if not pending:
        print("All files already simplified", file=sys.stderr)
        return 0

    succeeded = 0
    failed = 0
    for idx, (vtt_file, output_file) in enumerate(pending, 1):
        print(f"[{idx}/{len(pending)}] {vtt_file.name}...", file=sys.stderr, flush=True)
        temp_file = output_file.with_name(f".{output_file.name}.tmp")
        try:
            content = vtt_file.read_text()
            result = simplify_vtt(content)
            temp_file.write_text(result + "\n")
            temp_file.replace(output_file)
            succeeded += 1
            print(f"  Saved: {output_file.name}", file=sys.stderr)
        except Exception as exc:
            failed += 1
            temp_file.unlink(missing_ok=True)
            print(f"  Error: {exc}", file=sys.stderr)

    print(
        f"Done: {succeeded} succeeded, {failed} failed, {skipped} skipped. Output: {args.output_dir}",
        file=sys.stderr,
    )
    return 1 if failed else 0


def cmd_benchmark(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:

    from digue.container import _is_remote

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


def main() -> None:
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
    from digue.notify import notify_close

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
