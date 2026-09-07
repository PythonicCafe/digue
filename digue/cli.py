"""CLI parser, dispatch, and helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# -- CLI ----------------------------------------------------------------------


def _existing_dir(value: str) -> Path:
    """argparse type: validates that the path is an existing directory."""

    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory not found: {value}")
    return path


def _positive_int(value: str) -> int:
    """argparse type: an integer greater than zero."""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed}")
    return parsed


def create_parser() -> argparse.ArgumentParser:
    from digue import AVAILABLE_MODELS, __version__
    from digue.benchmark import BENCHMARK_MODELS, BENCHMARK_RUNS
    from digue.container import BACKENDS
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
    sub_batch_transcribe.add_argument(
        "output_dir", type=Path, help="Directory for transcription output (created if missing)"
    )
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
    sub_batch_simplify.add_argument(
        "output_dir", type=Path, help="Directory for simplified output (created if missing)"
    )

    sub_benchmark = subparsers.add_parser("benchmark", help="Compare backend and model performance")
    benchmark_audio = sub_benchmark.add_mutually_exclusive_group()
    benchmark_audio.add_argument(
        "audio",
        nargs="?",
        type=Path,
        default=None,
        help="Audio file (records 10s from the microphone if not given)",
    )
    benchmark_audio.add_argument(
        "--sample",
        action="store_true",
        help="Use the whisper.cpp JFK sample (downloaded once) instead of recording",
    )
    local_backends = [backend for backend in BACKENDS if backend != "remote"]
    sub_benchmark.add_argument(
        "-b",
        "--backends",
        nargs="+",
        choices=local_backends,
        metavar="BACKEND",
        default=None,
        help=f"Backends to test (default: the resolved backend plus cpu). Options: {', '.join(local_backends)}",
    )
    sub_benchmark.add_argument(
        "-m",
        "--models",
        nargs="+",
        choices=(*AVAILABLE_MODELS, "all"),
        metavar="MODEL",
        default=None,
        help=f"Models to test (default: {', '.join(BENCHMARK_MODELS)}). Options: {', '.join(AVAILABLE_MODELS)}, "
        'or "all" for every model',
    )
    sub_benchmark.add_argument(
        "-n",
        "--runs",
        type=_positive_int,
        default=BENCHMARK_RUNS,
        help=f"Timed runs per case, after one warm-up (default: {BENCHMARK_RUNS})",
    )
    sub_benchmark.add_argument("--json", action="store_true", help="Print the results as JSON on stdout")

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


def _format_extension(response_format: str) -> str:
    return {"text": ".txt", "vtt": ".vtt", "srt": ".srt", "timestamps": ".txt"}[response_format]


def main() -> None:
    from digue.audio import cmd_clean
    from digue.benchmark import cmd_benchmark
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
    from digue.dictate import cmd_dictate
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
