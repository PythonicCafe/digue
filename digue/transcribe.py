"""HTTP transcription, language detection, VTT helpers, and transcribe commands."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Container formats whisper-server decodes natively (miniaudio: RIFF/PCM, fLaC, MP3,
# Ogg/Vorbis, AIFF). Verified empirically against whisper-server (ghcr.io main-vulkan
# image, built with WHISPER_COMMON_FFMPEG=OFF): wav, flac, mp3, ogg-vorbis and aiff
# return HTTP 200; opus-in-ogg (WhatsApp voice notes), m4a/AAC, mp4, webm, mka and
# wma return HTTP 400. Everything else is converted with ffmpeg before upload.
NATIVE_FORMATS = frozenset((".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"))

RESPONSE_FORMATS = ("text", "vtt", "srt", "timestamps")

TRANSCRIPTION_TIMEOUT = 120

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
    from digue.delivery import normalize_pasted_text

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
    from digue.convert import _convert_content

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


def cmd_transcribe(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    from digue.container import ensure_server, is_server_running, server_not_running_hint, server_url
    from digue.convert import _convert_content

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


def cmd_batch_transcribe(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    import time

    from digue import _format_extension
    from digue.container import ensure_server, is_server_running, server_not_running_hint, server_url
    from digue.convert import _convert_content

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
