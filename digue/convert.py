"""Subtitle conversion between vtt, srt, timestamps, and text."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_LAST_CUE_DURATION_MS = 2_000


@dataclass(frozen=True)
class SubtitleCue:
    """A subtitle cue whose boundaries are integer milliseconds."""

    start_ms: int
    end_ms: int | None
    text: str


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

    from digue.transcribe import _strip_vtt_tags

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
    from digue.delivery import normalize_pasted_text

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
