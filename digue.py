#!/usr/bin/env python3
"""Local speech-to-text dictation and transcription using whisper.cpp."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import subprocess

__version__ = "0.1.0"

CONTAINER_NAME = "digue"
NOTIFY_REPLACE_ID = 48271
NOTIFY_ID_SLOTS = 32  # concurrent takes: id = base + (pid % slots), so popups of
# overlapping dictations do not replace or close each other.
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
BACKENDS = ("nvidia", "amd", "intel", "cpu", "remote")
DOCKER_IMAGES = {
    "nvidia": "ghcr.io/ggml-org/whisper.cpp:main-cuda",
    "amd": "ghcr.io/ggml-org/whisper.cpp:main-vulkan",
    "intel": "ghcr.io/ggml-org/whisper.cpp:main-vulkan",
    # CPU default: main-vulkan without GPU devices. Falls back to CPU when no /dev/dri is mounted. Works on modern CPUs
    # (Meteor Lake etc.) where the main image crashes due to AMX initialization failure.
    # However, on older CPUs (Kaby Lake etc.) main-vulkan itself crashes with SIGILL (compiled with instructions the
    # CPU doesn't support). For those, override with: image = "ghcr.io/ggml-org/whisper.cpp:main"
    "cpu": "ghcr.io/ggml-org/whisper.cpp:main-vulkan",
}
HUGGINGFACE_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
HUGGINGFACE_VAD_URL = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin"
RESPONSE_FORMATS = ("text", "vtt", "srt", "timestamps")
SERVER_STARTUP_TIMEOUT = 180
TRANSCRIPTION_TIMEOUT = 120
BENCHMARK_TRANSCRIPTION_TIMEOUT = 300
DOWNLOAD_TIMEOUT = 60
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


# -- Config -------------------------------------------------------------------


def _config_path() -> Path:

    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "digue" / "config.toml"


def _default_config() -> dict[str, dict[str, Any]]:

    xdg = os.environ.get("XDG_DATA_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    data_dir = base / "digue"

    return {
        "server": {
            "port": DEFAULT_PORT,
            "data_dir": str(data_dir),
            "backend": "auto",
            "image": "",
            "bind_ip": "127.0.0.1",
            "remote_host": "",
        },
        "transcribe": {
            "language": DEFAULT_LANGUAGE,
            "prompt": "",
            "output_format": "text",
            "max_line_length": 42,
            "max_lines": 2,
        },
        "dictate": {
            "audio_dir": "",
            "display_server": "auto",
            "input_mode": "paste",
            "recorder": "auto",
            "max_duration": DEFAULT_MAX_RECORD_SECONDS,
            "save_audio": True,
            "audio_format": "flac",
        },
        "models": dict(DEFAULT_MODELS),
    }


def _merge_section(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Merges source keys into target dict; kebab-case keys override snake_case defaults."""
    for key, value in source.items():
        target[key.replace("-", "_")] = value


def _host_overrides(user_config: dict[str, Any]) -> dict[str, Any]:
    """Returns the [host.<this-hostname>] tables from the user config, or {}.

    Matches the exact hostname first; then tries without the domain part
    (e.g. "thinkpad" matches "thinkpad.local"). gethostname() is in-memory
    (microseconds), so calling it on every run adds no startup delay.
    """
    hosts = user_config.get("host")
    if not isinstance(hosts, dict):
        return {}
    import socket

    hostname = socket.gethostname()
    for candidate in (hostname, hostname.split(".")[0]):
        if candidate in hosts and isinstance(hosts[candidate], dict):
            override: dict[str, Any] = hosts[candidate]
            return override
    return {}


def _validate_config(config: dict[str, dict[str, Any]]) -> None:
    """Validates the fully merged configuration before commands consume it."""

    def require_type(section: str, key: str, expected_type: type[Any]) -> Any:
        value = config[section][key]
        if not isinstance(value, expected_type) or expected_type is int and isinstance(value, bool):
            raise ValueError(f"Invalid {section}.{key}: expected {expected_type.__name__}, got {type(value).__name__}")
        return value

    def require_choice(section: str, key: str, choices: tuple[str, ...]) -> None:
        value = require_type(section, key, str)
        if value not in choices:
            options = ", ".join(choices)
            raise ValueError(f"Invalid {section}.{key}: {value!r}; expected one of: {options}")

    require_choice("server", "backend", ("auto", *BACKENDS))
    port = require_type("server", "port", int)
    if not 1 <= port <= 65535:
        raise ValueError(f"Invalid server.port: {port}; expected an integer from 1 to 65535")
    for key in ("data_dir", "image", "bind_ip", "remote_host"):
        require_type("server", key, str)

    for key in ("language", "prompt"):
        require_type("transcribe", key, str)
    require_choice("transcribe", "output_format", ("text", "vtt", "srt", "timestamps"))
    for key in ("max_line_length", "max_lines"):
        value = require_type("transcribe", key, int)
        if value < 1:
            raise ValueError(f"Invalid transcribe.{key}: {value}; expected an integer greater than zero")

    require_type("dictate", "audio_dir", str)
    require_choice("dictate", "display_server", ("auto", "x11", "wayland"))
    require_choice("dictate", "input_mode", ("paste", "type"))
    require_choice("dictate", "recorder", ("auto", "pw-record", "arecord"))
    require_choice("dictate", "audio_format", ("wav", "flac", "opus"))
    require_type("dictate", "save_audio", bool)
    max_duration = require_type("dictate", "max_duration", int)
    if max_duration < 0:
        raise ValueError(f"Invalid dictate.max_duration: {max_duration}; expected zero or greater")

    for backend, model in config["models"].items():
        if backend in DEFAULT_MODELS and (not isinstance(model, str) or model not in AVAILABLE_MODELS):
            options = ", ".join(AVAILABLE_MODELS)
            raise ValueError(f"Invalid models.{backend}: {model!r}; expected one of: {options}")


def load_config(config_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Loads config from TOML file, falling back to defaults for missing keys.

    If the file defines [host.<hostname>][section] tables matching this machine
    (exact hostname, or the first dot component of it), those sections are
    merged on top of the global ones, which in turn override the defaults.
    """
    import tomllib

    config = _default_config()
    path = Path(config_path) if config_path else _config_path()

    if path.exists():
        with path.open("rb") as fobj:
            user_config: dict[str, Any] = tomllib.load(fobj)
        for section, defaults in config.items():
            if section in user_config:
                _merge_section(config[section], user_config[section])
        host_overrides = _host_overrides(user_config)
        for section, defaults in config.items():
            if section in host_overrides and isinstance(host_overrides[section], dict):
                _merge_section(config[section], host_overrides[section])

    _validate_config(config)

    if not config["dictate"]["audio_dir"]:
        config["dictate"]["audio_dir"] = str(Path(config["server"]["data_dir"]) / "audio")

    # Expand ~ in path values
    for key in ("data_dir",):
        config["server"][key] = str(Path(config["server"][key]).expanduser())
    for key in ("audio_dir",):
        config["dictate"][key] = str(Path(config["dictate"][key]).expanduser())

    return config


def model_for_backend(backend: str, config: dict[str, dict[str, Any]] | None = None) -> str:
    """Returns the model name for a given backend, respecting config overrides."""
    if config and "models" in config:
        models: dict[str, Any] = config["models"]
        return str(models.get(backend, DEFAULT_MODELS.get(backend, "small")))
    return DEFAULT_MODELS.get(backend, "small")


# -- Detection ----------------------------------------------------------------


def detect_backend() -> str:
    """Detects GPU backend: nvidia, amd, intel, or cpu."""
    import subprocess

    # NVIDIA discrete GPU (highest priority)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "nvidia"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # AMD with ROCm/KFD
    if Path("/dev/kfd").exists():
        return "amd"

    # Intel modern iGPU (Skylake+, excludes Broadwell/Haswell/older)
    try:
        result = subprocess.run(
            ["lspci"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        old_gpu_markers = ("Broadwell", "Haswell", "Ivy", "Sandy")
        for line in result.stdout.splitlines():
            if "VGA" in line.upper() and "Intel" in line and not any(marker in line for marker in old_gpu_markers):
                return "intel"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return "cpu"


# -- Container management -----------------------------------------------------


def _docker_run(args: list[str], timeout: int | float = 30) -> subprocess.CompletedProcess[str]:
    import subprocess

    return subprocess.run(
        ["docker"] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def container_exists() -> bool:
    """Returns True if the digue container exists (running or stopped)."""
    result = _docker_run(["inspect", "--format", "{{.State.Status}}", CONTAINER_NAME])
    return result.returncode == 0


def container_status() -> str | None:
    """Returns container status string ('running', 'exited', etc.) or None."""
    result = _docker_run(["inspect", "--format", "{{.State.Status}}", CONTAINER_NAME])
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def image_exists(image: str) -> bool:
    """Returns True if a Docker image exists locally."""
    result = _docker_run(["image", "inspect", image])
    return result.returncode == 0


def pull_image(image: str) -> None:
    """Pulls a Docker image if not present locally, showing download progress."""
    import subprocess

    if image_exists(image):
        return

    print(f"Pulling {image} (this may take a while on first run)...", file=sys.stderr, flush=True)
    result = subprocess.run(
        ["docker", "pull", image],
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to pull image: {image}")
    print(f"Pull complete: {image}", file=sys.stderr)


def resolve_backend(config: dict[str, dict[str, Any]]) -> str:
    """Returns the backend to use, respecting config override or auto-detecting."""
    configured = config["server"].get("backend", "auto")
    if configured != "auto":
        return str(configured)
    return detect_backend()


def resolve_image(backend: str, config: dict[str, dict[str, Any]]) -> str:
    """Returns the Docker image to use, respecting config override."""
    if backend == "remote":
        return ""
    configured = config["server"].get("image", "")
    if configured:
        return str(configured)
    return DOCKER_IMAGES[backend]


def create_container(config: dict[str, dict[str, Any]], backend: str | None = None) -> str:
    """Creates the digue container.

    Resolves backend and image from config (with auto-detection fallback).
    Downloads the model and pulls the Docker image if not present locally.
    """

    if backend is None:
        backend = resolve_backend(config)

    if backend == "remote":
        raise RuntimeError("backend 'remote' uses a remote server; there is no local container to create")

    port = config["server"]["port"]
    data_dir = Path(config["server"]["data_dir"])
    models_dir = data_dir / "models"
    model = model_for_backend(backend, config)
    image = resolve_image(backend, config)

    # Ensure model is downloaded before creating the container (avoids crash loop)
    model_path = models_dir / f"ggml-{model}.bin"
    if not model_path.exists():
        print(f"Model not found: {model_path.name}. Downloading...", file=sys.stderr, flush=True)
        download_model(model, models_dir, with_notification=True)

    pull_image(image)

    cmd = [
        "run",
        "-d",
        "--name",
        CONTAINER_NAME,
        "--restart",
        "unless-stopped",
        "-p",
        f"{config['server'].get('bind_ip', '127.0.0.1')}:{port}:8080",
    ]

    if backend == "nvidia":
        cmd += ["--gpus", "all"]
    elif backend == "amd":
        cmd += ["--device", "/dev/kfd", "--device", "/dev/dri"]
    elif backend == "intel":
        cmd += ["--device", "/dev/dri"]

    cmd += [
        "-v",
        f"{models_dir}:/models:ro",
        "--entrypoint",
        "whisper-server",
        image,
        "--model",
        f"/models/ggml-{model}.bin",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--vad",
        "--vad-model",
        "/models/ggml-silero-v6.2.0.bin",
    ]

    result = _docker_run(cmd, timeout=60)
    if result.returncode != 0:
        first_error_line = result.stderr.strip().splitlines()[0] if result.stderr.strip() else "unknown error"
        raise RuntimeError(f"Failed to create container: {first_error_line}")
    return backend


def _raise_for_docker_failure(result: subprocess.CompletedProcess[str], action: str) -> None:
    if result.returncode != 0:
        error = result.stderr.strip().splitlines()[0] if result.stderr.strip() else "unknown error"
        raise RuntimeError(f"Failed to {action}: {error}")


def remove_container() -> None:
    """Stops and removes the digue container."""
    result = _docker_run(["rm", "-f", CONTAINER_NAME])
    _raise_for_docker_failure(result, "remove container")


def start_container() -> bool:
    """Starts an existing stopped container."""
    result = _docker_run(["start", CONTAINER_NAME])
    _raise_for_docker_failure(result, "start container")
    return True


def stop_container() -> None:
    """Stops the running container."""
    result = _docker_run(["stop", CONTAINER_NAME], timeout=15)
    _raise_for_docker_failure(result, "stop container")


def _rename_container(old_name: str, new_name: str) -> None:
    result = _docker_run(["rename", old_name, new_name])
    _raise_for_docker_failure(result, "rename container")


@contextlib.contextmanager
def preserve_container_for_benchmark() -> Iterator[None]:
    """Makes room for benchmark containers, then restores the prior container and running state."""
    previous_status = container_status()
    backup_name = f"{CONTAINER_NAME}-benchmark-backup-{os.getpid()}"

    if previous_status is not None:
        if previous_status == "running":
            stop_container()
        try:
            _rename_container(CONTAINER_NAME, backup_name)
        except BaseException:
            if previous_status == "running":
                start_container()
            raise

    try:
        yield
    finally:
        try:
            if container_exists():
                remove_container()
        finally:
            if previous_status is not None:
                _rename_container(backup_name, CONTAINER_NAME)
                if previous_status == "running":
                    start_container()


# -- Notifications ------------------------------------------------------------

_notify_send_warned = False
_last_notify_len = 0


def notify(message: str, timeout_ms: int = 0) -> None:
    """Prints message to stderr AND sends a desktop notification.

    The notification stays visible until replaced by the next one (timeout_ms=0).
    Pass a timeout for messages that should auto-dismiss (success, errors).
    If notify-send is not installed, prints a one-time warning and continues.

    On a terminal the stderr line is redrawn (\\r, padded to erase a previous
    shorter message), so it coexists with single-line progress bars; on a
    captured stderr it is a plain line with \\n.
    """
    import subprocess

    global _notify_send_warned, _last_notify_len

    if _stderr_is_tty():
        padding = " " * max(0, _last_notify_len - len(message))
        print(f"\r[digue] {message}{padding}", end="", file=sys.stderr, flush=True)
        _last_notify_len = len(message)
    else:
        print(f"[digue] {message}", file=sys.stderr, flush=True)
        _last_notify_len = 0

    try:
        subprocess.run(
            [
                "notify-send",
                "-a",
                "digue",
                "--replace-id",
                str(NOTIFY_REPLACE_ID + os.getpid() % NOTIFY_ID_SLOTS),
                "-t",
                str(timeout_ms),
                "digue",
                message,
            ],
            capture_output=True,
            timeout=5,
            check=True,
        )
    except FileNotFoundError:
        if not _notify_send_warned:
            print(
                "Warning: notify-send not found. Install libnotify-bin for desktop notifications.",
                file=sys.stderr,
            )
            _notify_send_warned = True
    except subprocess.SubprocessError as exc:
        if not _notify_send_warned:
            print(
                f"Warning: notify-send failed ({type(exc).__name__}); desktop notifications unavailable.",
                file=sys.stderr,
            )
            _notify_send_warned = True


def notify_close() -> None:
    """Closes the current digue notification via D-Bus."""
    import subprocess

    with contextlib.suppress(subprocess.SubprocessError, FileNotFoundError):
        subprocess.run(
            [
                "gdbus",
                "call",
                "--session",
                "--dest",
                "org.freedesktop.Notifications",
                "--object-path",
                "/org/freedesktop/Notifications",
                "--method",
                "org.freedesktop.Notifications.CloseNotification",
                str(NOTIFY_REPLACE_ID + os.getpid() % NOTIFY_ID_SLOTS),
            ],
            capture_output=True,
            timeout=5,
        )


# -- Server -------------------------------------------------------------------


def server_host(config: dict[str, dict[str, Any]]) -> str:
    """Returns the host the server is probed on.

    With backend 'remote', returns the configured remote host (default
    127.0.0.1 for an SSH tunnel). For local backends, probes the address Docker
    binds, except that the wildcard bind is reached through loopback.
    """
    if resolve_backend(config) == "remote":
        return str(config["server"].get("remote_host") or "127.0.0.1")
    bind_ip = str(config["server"].get("bind_ip", "127.0.0.1"))
    return "127.0.0.1" if bind_ip == "0.0.0.0" else bind_ip


def server_url(config: dict[str, dict[str, Any]]) -> str:
    port = config["server"]["port"]
    return f"http://{server_host(config)}:{port}/inference"


def is_server_running(config: dict[str, dict[str, Any]]) -> bool:
    """Returns True if the server is responding to HTTP requests.

    For local backends this probes the configured bind address. A wildcard
    bind is reached through loopback because 0.0.0.0 is not a destination.
    """
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(f"http://{server_host(config)}:{config['server']['port']}/", timeout=1)
        return True
    except (urllib.error.URLError, OSError):
        return False


def _wait_for_server(config: dict[str, dict[str, Any]], verbose: bool = False) -> bool:
    """Waits for server to respond. Returns True if successful.

    When verbose=True, prints elapsed time to stderr every 10 seconds.
    """
    import time

    start = time.perf_counter()
    for attempt in range(SERVER_STARTUP_TIMEOUT):
        time.sleep(1)
        if is_server_running(config):
            if verbose:
                elapsed = time.perf_counter() - start
                print(f"\rServer ready ({elapsed:.0f}s)          ", file=sys.stderr)
            return True
        if verbose and attempt > 0 and attempt % 10 == 0:
            elapsed = time.perf_counter() - start
            print(f"\r  Waiting for model to load... {elapsed:.0f}s", end="", file=sys.stderr, flush=True)

    if verbose:
        print(file=sys.stderr)
    return False


def ensure_server(config: dict[str, dict[str, Any]], silent: bool = False) -> str | None:
    """Ensures server is running, creating/starting the container if needed.

    Returns the backend used, or None if the server was already running.
    With backend 'remote' no local container is ever touched; the server is
    expected to be reachable through an SSH tunnel.
    """
    if is_server_running(config):
        return None

    if resolve_backend(config) == "remote":
        host = config["server"].get("remote_host") or "127.0.0.1"
        if not silent:
            notify(
                f"Remote server {host}:{config['server']['port']} not responding. Is your tunnel active / host reachable?",
                timeout_ms=10000,
            )
        return None

    status = container_status()
    backend = None

    if status == "exited":
        if not silent:
            notify("Starting server...")
        start_container()
    elif status is None:
        backend = resolve_backend(config)
        if not silent:
            notify(f"Creating server ({backend})...")
        create_container(config, backend)
    else:
        if not silent:
            notify(f"Container in unexpected state: {status}", timeout_ms=5000)
        return None

    if _wait_for_server(config, verbose=not silent):
        if not silent:
            notify_close()
        return backend

    if not silent:
        notify("Server failed to start (see: docker logs digue)", timeout_ms=10000)
    return None


def server_not_running_hint(config: dict[str, dict[str, Any]]) -> str:
    """Returns the actionable hint shown when the server is not responding."""
    if resolve_backend(config) == "remote":
        host = config["server"].get("remote_host") or "127.0.0.1"
        if host == "127.0.0.1":
            return (
                f"Backend is 'remote': no local container to start. Forward port {config['server']['port']} with "
                f"ssh -NfL {config['server']['port']}:127.0.0.1:{config['server']['port']} user@host "
                "(see README, Remote access), or set server.remote-host to a LAN host."
            )
        return f"Backend is 'remote' and server {host}:{config['server']['port']} is not responding (see README, Remote access)."
    return "Run: digue start"


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


# -- Download -----------------------------------------------------------------


def _stderr_is_tty() -> bool:
    """Returns True if stderr is a terminal (dynamic progress makes sense).

    With captured/piped stderr, \\r has no visual effect and every update
    becomes a full line in the log -- hence the sparse-line mode in progress
    and notification prints.
    """
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


def _download_progress_hook(label: str, with_notification: bool = False) -> Any:
    """Returns a reporthook callback for urlretrieve that prints a progress bar.

    On a terminal, redraws one line with \\r. On a captured/piped stderr, prints
    one line every ~5% (no bar, no \\r), so logs stay readable.
    The downloaded MBs are padded to the total's width, so line size stays
    stable across digit rollovers (9.9 -> 10.0 MB).
    When with_notification=True, also updates the desktop notification (~2x/s);
    the notification text never carries the bar nor \\r.
    """
    import time

    last_notify_time = [0.0]
    last_reported_pct = [-1]
    tty = _stderr_is_tty()

    def hook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            total_mb_str = f"{total_mb:.1f}"
            width = len(total_mb_str)
            short_msg = f"{pct:3d}% {downloaded_mb:>{width}.1f}/{total_mb_str} MB"
            if tty:
                bar_width = 30
                filled = bar_width * pct // 100
                bar = "=" * filled + " " * (bar_width - filled)
                msg = f"{short_msg.split('%', 1)[0]}% [{bar}] {downloaded_mb:>{width}.1f}/{total_mb_str} MB"
                print(f"\r  {label}: {msg}", end="", file=sys.stderr, flush=True)
            else:
                # Sparse output for logs: one line every 5% (and at 100%)
                if pct == 100 and last_reported_pct[0] != 100:
                    last_reported_pct[0] = 100
                    print(f"  {label}: {short_msg}", file=sys.stderr, flush=True)
                elif pct >= last_reported_pct[0] + 5:
                    last_reported_pct[0] = pct
                    print(f"  {label}: {short_msg}", file=sys.stderr, flush=True)
        else:
            downloaded_mb = downloaded / (1024 * 1024)
            short_msg = f"{downloaded_mb:9.1f} MB"
            if tty:
                print(f"\r  {label}: {short_msg}", end="", file=sys.stderr, flush=True)
            else:
                print(f"  {label}: {short_msg}", file=sys.stderr, flush=True)

        if with_notification:
            now = time.monotonic()
            if now - last_notify_time[0] >= 0.5:
                # Notifications get no bar (meaningless outside a terminal) and
                # use the total's width for the downloaded value, so the message
                # size stays stable across digit rollovers.
                if total_size > 0:
                    total_mb_str = f"{total_mb:.1f}"
                    short_msg = f"{pct:3d}% {downloaded_mb:>{len(total_mb_str)}.1f}/{total_mb_str} MB"
                else:
                    short_msg = f"{downloaded_mb:9.1f} MB"
                notify(f"Downloading {label}... {short_msg}")
                last_notify_time[0] = now

    return hook


def _download_file(url: str, output: Path, label: str, with_notification: bool = False) -> None:
    """Downloads to a temporary sibling and atomically publishes the complete file."""
    import urllib.request

    part_path = output.with_suffix(f"{output.suffix}.part")
    hook = _download_progress_hook(label, with_notification)
    block_size = 1024 * 1024
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response, part_path.open("wb") as part_file:
            total_size = int(response.headers.get("Content-Length", -1))
            downloaded = 0
            hook(downloaded, 1, total_size)
            while block := response.read(block_size):
                part_file.write(block)
                downloaded += len(block)
                hook(downloaded, 1, total_size)
    except BaseException:
        # No resume support, so a partial file is only clutter next to the models.
        part_path.unlink(missing_ok=True)
        raise
    part_path.replace(output)


def download_model(model_name: str, models_dir: str | Path, with_notification: bool = False) -> None:
    """Downloads a whisper.cpp GGML model and the Silero VAD model."""

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"ggml-{model_name}.bin"
    vad_path = models_dir / "ggml-silero-v6.2.0.bin"

    for label, url, output in [
        (f"ggml-{model_name}.bin", f"{HUGGINGFACE_MODEL_URL}/ggml-{model_name}.bin", model_path),
        ("ggml-silero-v6.2.0.bin (VAD)", HUGGINGFACE_VAD_URL, vad_path),
    ]:
        if output.exists():
            size_mb = output.stat().st_size / (1024 * 1024)
            print(f"Already exists: {output} ({size_mb:.1f} MB)", file=sys.stderr)
            continue
        print(f"Downloading {label}...", file=sys.stderr, flush=True)
        _download_file(url, output, label, with_notification)
        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"\n  Saved: {output} ({size_mb:.1f} MB)", file=sys.stderr)


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


def _pid_file() -> Path:
    return _runtime_dir() / "digue.pid"


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


def _rec_file() -> Path:
    """Returns a unique recording path without creating the audio file."""
    import secrets

    return _runtime_dir() / f"digue-{now_timestamp()}-{os.getpid()}-{secrets.token_hex(4)}.wav"


def is_recording() -> bool:
    pid_file = _pid_file()
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return False
    return _pid_alive(pid)


def _pid_alive(pid: int) -> bool:

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def recording_command(rec_file: str | Path, recorder: str = "auto") -> list[str]:
    """Builds the argv that records mono 16 kHz s16 audio to rec_file.

    recorder: "auto" (pw-record if available, else arecord), "pw-record", or "arecord".
    """
    import shutil

    if recorder == "auto":
        if shutil.which("pw-record"):
            recorder = "pw-record"
        elif shutil.which("arecord"):
            recorder = "arecord"
        else:
            recorder = "pw-record"
    if recorder == "pw-record":
        return ["pw-record", "--rate", "16000", "--channels", "1", "--format", "s16", str(rec_file)]
    if recorder == "arecord":
        return ["arecord", "-f", "S16_LE", "-r", "16000", "-c", "1", str(rec_file)]
    raise RuntimeError(f"Unknown recorder: {recorder}. Use 'auto', 'pw-record', or 'arecord'.")


@dataclass(frozen=True)
class RecordingProcesses:
    """Processes owned by a recording daemon; the watchdog may be disabled."""

    recorder: subprocess.Popen[bytes]
    watchdog: subprocess.Popen[bytes] | None
    rec_file: Path | None = None


def start_recording(config: dict[str, dict[str, Any]]) -> RecordingProcesses:
    """Starts the recorder and safety watchdog, returning their owned handles.

    The recorder runs in a new process group so it survives a killed daemon.
    The PID file remains the recovery contract for a later invocation, while
    the live daemon retains Popen handles so it can reap both children.
    """
    import subprocess

    rec_file = _rec_file()
    pid_file = _pid_file()
    max_duration = config["dictate"]["max_duration"]
    argv = recording_command(rec_file, recorder=config["dictate"]["recorder"])
    recorder = subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _write_state_file(pid_file, str(recorder.pid))
    watchdog = _spawn_limit_watchdog(recorder.pid, max_duration) if max_duration > 0 else None
    return RecordingProcesses(recorder=recorder, watchdog=watchdog, rec_file=rec_file)


def _process_starttime(pid: int, stat_path: Path | None = None) -> str | None:
    """Returns Linux /proc starttime, which distinguishes recycled PIDs."""
    path = stat_path or Path(f"/proc/{pid}/stat")
    try:
        stat = path.read_text()
        fields_after_comm = stat[stat.rindex(")") + 2 :].split()
        return fields_after_comm[19]
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def _spawn_limit_watchdog(pgid: int, max_duration: int) -> subprocess.Popen[bytes]:
    """Spawns an identity-checking safety killer and returns its handle.

    The detached child survives a SIGKILLed daemon. Before signaling, it checks
    Linux /proc starttime so a stale watchdog cannot kill a recycled PGID. It
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
    stat = Path(f"/proc/{pid}/stat").read_text()
    current_starttime = stat[stat.rindex(")") + 2:].split()[19]
    if current_starttime == expected_starttime:
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

    runtime_dir = _runtime_dir()
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


def stop_recording_pid(pid: int, rec_file: Path | None = None) -> Path | None:
    """Stops the recorder process group `pid` and returns its audio file or None.

    Used by the take's owner (the daemon that started this recorder). Never
    touches the global pid file: with overlapping takes each daemon stops only
    its own recorder -- a global stop would kill another take's recorder. The
    owner passes the rec_file captured while the recorder was alive; without
    it, the newest-runtime-wav fallback runs (single-take recovery only: with
    concurrent takes it could grab another daemon's file).
    """
    import time

    if rec_file is None:
        rec_file = _recording_file_of(pid)

    for signal in (15, 9):  # SIGTERM, then SIGKILL if it does not exit
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal)
        # Poll instead of a fixed sleep: pw-record exits within milliseconds,
        # and this wait sits between the hotkey and the transcription.
        deadline = time.monotonic() + 0.5
        while _group_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        if not _group_alive(pid):
            break

    if rec_file is None:
        # fd scan found nothing (recorder already dead and descriptors closed);
        # fall back to the newest digue-*.wav left in the runtime dir. Only for
        # callers without a captured file: with concurrent takes this could
        # grab another daemon's recording.
        runtime_dir = _runtime_dir()
        candidates = sorted(runtime_dir.glob("digue-*.wav"), key=lambda path: path.stat().st_mtime)
        rec_file = candidates[-1] if candidates else None
    return _validate_recording_file(rec_file)


def _finish_owned_recorder(recorder: subprocess.Popen[bytes], rec_file: Path | None) -> Path | None:
    """Stops a live owned recorder or validates output from an already reaped one."""
    if recorder.poll() is None:
        result = stop_recording_pid(recorder.pid, rec_file)
        recorder.wait(timeout=5)
        return result
    return _validate_recording_file(rec_file)


def stop_recording() -> Path | None:
    """Stops the current recording (from the global pid file). Returns the audio file or None.

    Recovery/legacy path: the daemon stops its own recorder via
    stop_recording_pid; this reads the global pid file (last started recorder)
    for callers outside the daemon flow.
    """
    pid_file = _pid_file()

    if not pid_file.exists():
        return None

    pid = int(pid_file.read_text().strip())
    pid_file.unlink(missing_ok=True)
    return stop_recording_pid(pid)


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
    """
    import subprocess

    if display_server == "auto":
        detected = detect_display_server()
        if detected is None:
            raise RuntimeError("No DISPLAY or WAYLAND_DISPLAY set. Cannot access clipboard or send keystrokes.")
        display_server = detected

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
    container via stdin/stdout pipe when backend is not remote.
    """
    import shutil
    import subprocess

    if audio_format == "wav":
        return Path(rec_file)
    codec_args = {
        "flac": ["-c:a", "flac"],
        "opus": ["-c:a", "libopus", "-b:a", "24k"],
    }
    format_args = {
        "flac": ["-f", "flac"],
        "opus": ["-f", "ogg"],
    }
    if audio_format not in codec_args:
        raise KeyError(audio_format)

    rec_file = Path(rec_file)
    converted = rec_file.with_suffix(f".{audio_format}")

    if shutil.which("ffmpeg"):
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(rec_file), *codec_args[audio_format], str(converted)],
            capture_output=True,
            timeout=600,
        )
        if result.returncode != 0 or not converted.exists():
            print(
                f"Warning: ffmpeg failed to compress recording ({result.stderr.decode(errors='replace').strip()[:150]}); keeping WAV",
                file=sys.stderr,
            )
            converted.unlink(missing_ok=True)
            return rec_file
        rec_file.unlink(missing_ok=True)
        return converted

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
            converted.write_bytes(result.stdout)
            rec_file.unlink(missing_ok=True)
            return converted
        print(
            f"Warning: ffmpeg failed to compress recording ({result.stderr.decode(errors='replace').strip()[:150]}); keeping WAV",
            file=sys.stderr,
        )
        converted.unlink(missing_ok=True)
        return rec_file

    print(
        f"Warning: ffmpeg not found, keeping the recording as WAV (install ffmpeg for {audio_format})",
        file=sys.stderr,
    )
    return rec_file


def save_audio(
    rec_file: str | Path,
    audio_dir: str | Path,
    audio_format: str = "wav",
    timestamp: str | None = None,
    backend: str | None = None,
) -> tuple[Path, str]:
    """Copies audio to <audio_dir>/YYYY/MM/<timestamp>.<ext>. Returns (saved_path, timestamp).

    The timestamp comes from the caller (dictate_toggle generates it when the
    take stops, so the audio and its transcript share the same name even when
    archiving runs later). Without one, the current time is used.
    audio_format "flac" or "opus" compresses the copy; the live recording file
    is kept as WAV and removed after saving.
    """
    import shutil

    audio_dir = Path(audio_dir)
    timestamp = timestamp or now_timestamp()
    month_dir = audio_dir / month_dir_for(timestamp)
    month_dir.mkdir(parents=True, exist_ok=True)
    saved = month_dir / f"{timestamp}.wav"
    shutil.copy2(rec_file, saved)
    if audio_format != "wav":
        saved = _compress_audio(saved, audio_format, backend=backend)
    return saved, timestamp


def normalize_pasted_text(text: str) -> str:
    """Joins wrapped lines into a single clean line.

    Line breaks come from whisper segment boundaries (word-aligned once
    token_timestamps is disabled, see _send_audio), so joining with a single
    space is safe; mid-word splits do not occur anymore.
    """
    return " ".join(text.split())


def _write_transcript(audio_dir: Path, timestamp: str, text: str) -> Path:
    """Writes the transcript next to the recording: <audio_dir>/YYYY/MM/<timestamp>.txt.

    The month folder comes from the timestamp itself (not from now()), so the
    .txt always lands beside the audio saved with the same timestamp.
    """
    text_path = audio_dir / month_dir_for(timestamp) / f"{timestamp}.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text + "\n")
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


def _recorder_pid_file(daemon_pid: int) -> Path:
    """Per-take recorder pid file: this daemon owns exactly this recorder.

    With overlapping takes a global recorder pid file would make one daemon's
    wait/stop logic act on another daemon's recorder (seen in the wild: two
    daemons waiting on the same pid; one delivered "Empty or missing audio
    file" after the other overwrote the global pid file).
    """
    return _runtime_dir() / f"digue-recorder-{daemon_pid}.pid"


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
    """True only if the pid is alive AND is still the process that wrote the file."""
    pid, _state, starttime = entry
    return _pid_alive(pid) and _process_starttime(pid) == starttime


def _remove_daemon_state(daemon_pid: int) -> bool:
    """Removes the global state only while it still belongs to this daemon."""
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


def finish_dictation(config: dict[str, dict[str, Any]], rec_file: Path | None) -> int:
    """Runs the full delivery flow (transcribe, paste, archive) for a stopped recording.

    Called by the daemon once the recorder is dead: manual stop (second toggle
    signaled the daemon, which stopped the recorder) or duration limit (the
    watchdog safety killer stopped it).
    """

    if rec_file is None:
        notify("Empty or missing audio file", timeout_ms=5000)
        return 1

    audio_dir = Path(config["dictate"]["audio_dir"])
    timestamp = now_timestamp()

    def rescue_recording() -> Path | None:
        """Keeps the live recording when the take could not be fully delivered.

        Moves the raw WAV to <audio_dir>/YYYY/MM/<timestamp>.wav (shutil.move
        handles cross-filesystem) regardless of save-audio: that setting only
        skips the backup of a delivered take, and an undelivered one exists
        nowhere else. On a failed move, prints and leaves the file in the
        runtime dir. Never raises.
        """
        try:
            import shutil

            month_dir = audio_dir / month_dir_for(timestamp)
            month_dir.mkdir(parents=True, exist_ok=True)
            archived = month_dir / f"{timestamp}.wav"
            shutil.move(rec_file, archived)
            return archived
        except Exception as rescue_exc:
            print(f"Failed to keep recording: {rescue_exc}; audio still at {rec_file}", file=sys.stderr)
            return None

    def archive_audio() -> bool:
        """Runs the post-delivery archiving (copy + compression, the slow part)."""
        try:
            if config["dictate"]["save_audio"]:
                backend = resolve_backend(config)
                save_audio(
                    rec_file,
                    audio_dir,
                    config["dictate"].get("audio_format", "wav"),
                    timestamp=timestamp,
                    backend=backend,
                )
            rec_file.unlink(missing_ok=True)
            return True
        except Exception as save_exc:
            rescued = rescue_recording()
            message = f"Failed to save audio: {save_exc}"
            if rescued:
                message += f"; uncompressed copy kept at {rescued}"
            notify(message, timeout_ms=10000)
            return False

    # Ctrl+c leaves "^C" echoed on the current terminal line; the \r redraw in
    # notify() would write over it and leave stray glyphs ("v"). Start a fresh
    # line for the transcription status.
    if _stderr_is_tty():
        print(file=sys.stderr, flush=True)
    notify("Transcribing...")
    try:
        url = server_url(config)
        language = config["transcribe"]["language"]
        prompt = config["transcribe"].get("prompt") or None
        text = normalize_pasted_text(transcribe(url, rec_file, language, prompt=prompt))
    except Exception as exc:
        archived = rescue_recording()
        notify(f"Transcription failed: {exc}", timeout_ms=10000)
        if archived:
            print(f"Recording kept at: {archived}", file=sys.stderr)
        return 1

    if not text:
        try:
            _write_transcript(audio_dir, timestamp, text)
        except Exception as exc:
            notify(f"Failed to save transcript: {exc}", timeout_ms=10000)
            print(text, file=sys.stderr)
            rescue_recording()
            return 1
        archive_audio()
        notify("No speech detected", timeout_ms=5000)
        return 0

    try:
        send_text(
            text,
            display_server=config["dictate"]["display_server"],
            input_mode=config["dictate"]["input_mode"],
        )
    except Exception as exc:
        notify(f"Paste failed: {exc}", timeout_ms=10000)
        try:
            text_path = _write_transcript(audio_dir, timestamp, text)
            print(f"Transcription saved to: {text_path}", file=sys.stderr)
        except Exception as save_exc:
            notify(f"Failed to save transcript: {save_exc}", timeout_ms=10000)
            print(text, file=sys.stderr)
        archive_audio()
        return 1
    notify_close()

    try:
        text_path = _write_transcript(audio_dir, timestamp, text)
    except Exception as exc:
        notify(f"Failed to save transcript: {exc}", timeout_ms=10000)
        print(text, file=sys.stderr)
        rescue_recording()
        return 1
    # Transcribing... is a \r-redrawn line (no newline); break before this one.
    if _stderr_is_tty():
        print(file=sys.stderr, flush=True)
    print(f"Dictation done ({len(text)} chars): {text_path}", file=sys.stderr)
    return 0 if archive_audio() else 1


def dictate_toggle(config: dict[str, dict[str, Any]]) -> int:
    """Toggle recording/transcription. Returns exit code.

    First call starts the recorder and stays alive as a daemon, waiting for
    the recording to end (manual stop via a second toggle, duration limit, or
    recorder crash) to run the delivery flow. A second call while the daemon
    is alive signals SIGTERM and exits immediately: the daemon does the work,
    so the keybinding feels instant. Killing the daemon (pkill digue) leaves
    the recorder alive -- the next toggle transcribes what kept recording.
    """

    daemon_pid = os.getpid()
    daemon_file = _daemon_pid_file()
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
                notify("Still starting the server; recording begins when it is ready", timeout_ms=3000)
                return 0
            # A delivering daemon owns its old take. A new recording may replace
            # the global state; the old daemon removes it only if it still owns it.
        elif is_recording():
            # Recorder alive but no daemon at all (the daemon was killed, e.g.
            # pkill digue): recover -- stop and deliver what kept recording.
            rec_file = stop_recording()
            daemon_file.unlink(missing_ok=True)
            return finish_dictation(config, rec_file)
        _write_daemon_state(daemon_pid, "starting")

    try:
        result = ensure_server(config)
        if result is None and not is_server_running(config):
            notify(server_not_running_hint(config), timeout_ms=5000)
            _remove_daemon_state(daemon_pid)
            return 1
        limit = config["dictate"]["max_duration"]
        message = (
            f"Recording... (max {limit}s, press again to stop)" if limit > 0 else "Recording... (press again to stop)"
        )
        notify(message)
        # running from a terminal: the user can also Ctrl+c to stop and transcribe.
        # notify() redraws its line without a trailing newline on a TTY, so this
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
        notify(
            f"Recorder not found: {exc.filename}. Install it (pipewire for pw-record, alsa-utils for arecord)",
            timeout_ms=10000,
        )
        _remove_daemon_state(daemon_pid)
        return 1
    except Exception as exc:
        notify(f"Failed to start recording: {exc}", timeout_ms=5000)
        _remove_daemon_state(daemon_pid)
        return 1
    recorder_pid = processes.recorder.pid
    recorder_file = _recorder_pid_file(daemon_pid)
    _write_state_file(recorder_file, str(recorder_pid))
    _write_daemon_state(daemon_pid, "recording")
    # capture the recording file while the recorder is alive: the fd scan is
    # deterministic here; after death the fallback could grab another
    # concurrent take's file (overlap scenario C).
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
    recorder_file.unlink(missing_ok=True)
    if _pid_file().exists() and _pid_file().read_text().strip() == str(recorder_pid):
        _pid_file().unlink(missing_ok=True)
    if outcome == "limit":
        notify(f"Recording stopped: {limit}s limit reached", timeout_ms=5000)
    try:
        return finish_dictation(config, rec_file)
    finally:
        _remove_daemon_state(daemon_pid)


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

            bench_config = {**config, "models": {**config["models"], backend: model}}
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
    import subprocess
    import time

    recorder = config["dictate"]["recorder"] if config else "auto"
    print(f"Recording {duration_seconds}s from microphone...", file=sys.stderr)
    print("(speak something so there is content to transcribe)", file=sys.stderr)
    proc = subprocess.Popen(
        recording_command(output_path, recorder=recorder),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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


CONFIG_TEMPLATE = """\
# -- Server -------------------------------------------------------------------
[server]
# port = 8178                   # host port for the whisper-server container
# bind-ip = "127.0.0.1"         # IP Docker binds the port to; 127.0.0.1 = local only.
                                #   Set to a LAN IP to expose it to that network
                                #   (the server has no authentication; prefer SSH tunnels)
# data-dir = ""                 # where models are stored
                                #   (default: $XDG_DATA_HOME/digue -- XDG_DATA_HOME is often
                                #   unset, in which case: ~/.local/share/digue)
# backend = "auto"              # "auto" (detect GPU), "nvidia", "amd", "intel", "cpu",
                                # or "remote" (server on another machine via SSH tunnel)
# remote-host = ""              # for backend = "remote": the server host (LAN IP,
                                #   hostname, or empty = 127.0.0.1 via SSH tunnel)
# image = ""                    # override Docker image (see README for compatibility matrix)

# -- Transcription (defaults for transcribe, batch-transcribe and dictate) -----
[transcribe]
# language = "auto"             # language for transcription: "auto", "pt", "en", etc.
# prompt = ""                   # initial prompt to steer spelling of names/acronyms,
                                #   e.g. "KINAI, Turicas, Pythonic Café"
# output-format = "text"        # transcribe output: "vtt", "srt", "timestamps"
                                #   ([00:00:12] text lines) or "text" (plain)
# max-line-length = 42          # subtitle cue wrapping (vtt/srt): max chars per line
# max-lines = 2                 # max lines per cue when wrapping

# -- Dictation ----------------------------------------------------------------
[dictate]
# audio-dir = ""                # where recordings are saved (default: <data-dir>/audio/YYYY/MM)
# display-server = "auto"       # "auto" (detect), "x11", or "wayland"
# input-mode = "paste"          # "paste" (clipboard + Ctrl+V) or "type" (simulate
                                #   keystrokes; use "type" in terminals)
# save-audio = true             # save the recording as a backup
# audio-format = "flac"         # format of the saved recording: "flac" (lossless,
                                #   ~35% of WAV; default), "opus" (~7%, lossy 24 kbit/s)
                                #   or "wav". Requires ffmpeg for flac/opus
# max-duration = 300            # stop recording after N seconds (0 = unlimited)
# recorder = "auto"             # "auto" (pw-record or arecord), "pw-record", or "arecord"

# -- Models per backend -------------------------------------------------------
[models]
# nvidia = "large-v3-turbo"
# amd = "large-v3-turbo"
# intel = "large-v3-turbo"
# cpu = "small"
# Available models: tiny, base, small, medium, large-v3-turbo, large-v3

# -- Per-host overrides (version this file in your dotfiles) -------------------
# [host.<hostname>][section] tables override the global sections of the same
# name on that machine only (defaults < global < host). The hostname matches
# exactly, or without the domain part (thinkpad matches thinkpad.local).
# Example:
#
# [host.minideb.server]
# backend = "amd"
#
# [host.thinkpad.server]
# backend = "cpu"
#
# [host.thinkpad.dictate]
# max-duration = 120
"""


def _config_example() -> str:
    """Returns the config template (all settings documented, defaults commented)."""
    return CONFIG_TEMPLATE


def _config_as_toml(config: dict[str, dict[str, Any]]) -> str:
    """Renders the resolved config as TOML (section by section, strings quoted).

    Keys use the kebab-case spelling of the template and README (data-dir),
    so the output can be pasted back into config.toml as documented.
    """
    lines = []
    for section, values in config.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            elif isinstance(value, int):
                rendered = str(value)
            else:
                rendered = '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'
            lines.append(f"{key.replace('_', '-')} = {rendered}")
        lines.append("")
    return "\n".join(lines)


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

    subparsers.add_parser("start", help="Start (or create) digue container")
    subparsers.add_parser("stop", help="Stop digue container")
    subparsers.add_parser("destroy", help="Stop and remove digue container")
    subparsers.add_parser("status", help="Show server status")
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


def cmd_detect(args: argparse.Namespace) -> int:
    backend = detect_backend()
    print(backend)
    return 0


def cmd_detect_language(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    """Detects the spoken language of an audio file (no transcription)."""
    ensure_server(config, silent=True)
    if not is_server_running(config):
        print(f"Error: server is not running. {server_not_running_hint(config)}", file=sys.stderr)
        return 1

    audio_path: Path = args.audio
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        return 1
    if not audio_path.is_file():
        print(f"Error: not a file: {audio_path} (expected an audio file; got a directory?)", file=sys.stderr)
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


def cmd_download(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:

    models_dir = Path(config["server"]["data_dir"]) / "models"

    if args.model:
        model = args.model
    else:
        backend = resolve_backend(config)
        if backend == "remote":
            print(
                "Backend is 'remote': models are stored on the remote machine. "
                "Use 'digue download <model>' to force a specific model.",
                file=sys.stderr,
            )
            return 1
        model = model_for_backend(backend, config)
        print(f"Backend: {backend}, downloading model: {model}", file=sys.stderr)

    download_model(model, models_dir)
    return 0


def cmd_start(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    if is_server_running(config):
        print("server is already running", file=sys.stderr)
        return 0

    if resolve_backend(config) == "remote":
        print(server_not_running_hint(config), file=sys.stderr)
        return 1

    status = container_status()
    try:
        if status == "exited":
            print("Starting existing container...", file=sys.stderr, flush=True)
            start_container()
        elif status is None:
            backend = resolve_backend(config)
            print(f"Creating container ({backend})...", file=sys.stderr, flush=True)
            create_container(config, backend)
        elif status == "running":
            print("Container running but server not responding, waiting...", file=sys.stderr, flush=True)
        else:
            print(f"Container in unexpected state: {status}", file=sys.stderr)
            return 1
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if _wait_for_server(config, verbose=True):
        return 0

    print(f"Server failed to start. Check: docker logs {CONTAINER_NAME}", file=sys.stderr)
    return 1


def cmd_stop(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    if resolve_backend(config) == "remote":
        print("Backend is 'remote': there is no local container to stop", file=sys.stderr)
        return 1
    try:
        stop_container()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Server stopped", file=sys.stderr)
    return 0


def cmd_destroy(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    if resolve_backend(config) == "remote":
        print("Backend is 'remote': there is no local container to remove", file=sys.stderr)
        return 1
    try:
        remove_container()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Container removed", file=sys.stderr)
    return 0


def cmd_status(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    port = config["server"]["port"]
    if resolve_backend(config) == "remote":
        http_ok = is_server_running(config)
        host = config["server"].get("remote_host") or "127.0.0.1"
        print(
            f"digue: remote backend, {'responding' if http_ok else 'not responding'} on {host}:{port}", file=sys.stderr
        )
        return 0 if http_ok else 1
    status = container_status()
    if status is None:
        print("Container does not exist", file=sys.stderr)
        return 1
    http_ok = is_server_running(config)
    label = f"{status}, {'responding' if http_ok else 'not responding'} on port {port}"
    print(f"digue: {label}", file=sys.stderr)
    return 0 if http_ok else 1


def cmd_dictate(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    if args.prompt is not None:
        config["transcribe"]["prompt"] = args.prompt
    return dictate_toggle(config)


def cmd_transcribe(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:

    ensure_server(config, silent=True)
    if not is_server_running(config):
        print(f"Error: server is not running. {server_not_running_hint(config)}", file=sys.stderr)
        return 1

    audio_path = args.audio
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        return 1
    if not audio_path.is_file():
        print(f"Error: not a file: {audio_path} (expected an audio file; got a directory?)", file=sys.stderr)
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
            if not cues:
                raise ValueError(f"input has no cues; cannot produce {to_format.upper()}")
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

    ensure_server(config, silent=True)
    if not is_server_running(config):
        print(f"Error: server is not running. {server_not_running_hint(config)}", file=sys.stderr)
        return 1

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

    if resolve_backend(config) == "remote":
        print("Error: benchmark creates local containers; it is not available with backend 'remote'", file=sys.stderr)
        return 1

    if args.audio:
        audio_path = args.audio
        if not audio_path.exists():
            print(f"Error: file not found: {audio_path}", file=sys.stderr)
            return 1
    else:
        audio_path = Path("/tmp/digue-bench.wav")
        record_benchmark_audio(audio_path, config=config)
        print(file=sys.stderr)

    run_benchmark(audio_path, config)
    return 0


def cmd_config(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    import json

    if args.output_format == "toml":
        print(_config_as_toml(config), end="")
    else:
        print(json.dumps(config, default=str, ensure_ascii=False, indent=2))
    return 0


def _config_init(args: argparse.Namespace) -> int:
    """Creates the config file from the template. Refuses to overwrite without --force."""

    selected_path = args.output or vars(args).get("config")
    path = Path(selected_path).expanduser() if selected_path else _config_path()
    if path.exists() and not args.force:
        print(f"Error: config file already exists: {path}", file=sys.stderr)
        print("Use --force to overwrite it.", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_config_example())
    print(f"Config created: {path}", file=sys.stderr)
    if args.force:
        print("(existing file was overwritten)", file=sys.stderr)
    return 0


DICTATION_RECORDING_SUFFIXES = frozenset((".wav", ".flac", ".opus"))


def _dictation_files(audio_dir: Path, suffixes: frozenset[str]) -> list[Path]:
    """Lists <audio_dir>/YYYY/MM/<timestamp>.<suffix> files created by dictation.

    Only that exact layout qualifies: audio-dir is user-configurable, and a
    recursive *.wav/*.flac/*.txt glob pointed at a music folder would remove
    the library. The stem is the now_timestamp() format, YYYYMMDD-HHMMSS.
    """
    import re

    stem_pattern = re.compile(r"^\d{8}-\d{6}$")
    found = []
    for year_dir in audio_dir.glob("[0-9][0-9][0-9][0-9]"):
        for month_dir in year_dir.glob("[0-9][0-9]"):
            if not month_dir.is_dir():
                continue
            for path in month_dir.iterdir():
                if path.is_file() and path.suffix.lower() in suffixes and stem_pattern.match(path.stem):
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

    if not recordings and not transcripts:
        print("Nothing to remove.", file=sys.stderr)
        return 0

    total = len(recordings) + len(transcripts)
    if not args.force:
        for path in sorted(recordings + transcripts):
            print(f"  {path.relative_to(audio_dir)}", file=sys.stderr)
        answer = input(f"Remove all {total} file(s)? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.", file=sys.stderr)
            return 1

    count = 0
    for path in recordings:
        path.unlink()
        count += 1
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


def cmd_doctor(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    """Checks system dependencies and tests which Docker images work."""
    import shutil
    import subprocess

    ok_mark = "OK"
    fail_mark = "FAIL"
    skip_mark = "SKIP"

    print("=== System dependencies ===", file=sys.stderr)
    tools = {
        "docker": "Docker runtime",
        "pw-record": "PipeWire audio recording (dictation)",
        "arecord": "ALSA audio recording (dictation fallback, alsa-utils)",
        "notify-send": "Desktop notifications (libnotify-bin)",
        "xclip": "X11 clipboard",
        "xdotool": "X11 key simulation",
        "wl-copy": "Wayland clipboard (wl-clipboard)",
        "wtype": "Wayland key simulation",
        "lspci": "GPU detection (pciutils)",
        "nvidia-smi": "NVIDIA GPU detection",
        "vulkaninfo": "Vulkan verification (vulkan-tools)",
        "ffmpeg": "Optional: convert audio formats the server cannot decode",
    }
    for tool, description in tools.items():
        found = shutil.which(tool)
        status = ok_mark if found else skip_mark
        location = found or "not found"
        print(f"  [{status}] {tool}: {location} -- {description}", file=sys.stderr)

    print("\n=== GPU detection ===", file=sys.stderr)
    detected = detect_backend()
    resolved = resolve_backend(config)
    print(f"  Auto-detected: {detected}", file=sys.stderr)
    if resolved != detected:
        print(f"  Config override: {resolved}", file=sys.stderr)
    image = resolve_image(resolved, config) or "(remote server)"
    print(f"  Image: {image}", file=sys.stderr)

    print("\n=== Docker images ===", file=sys.stderr)
    if resolved == "remote":
        print("  Skipped (backend is 'remote'; the server runs on another machine)", file=sys.stderr)
    else:
        test_images = [
            ("ghcr.io/ggml-org/whisper.cpp:main", "CPU (main)"),
            ("ghcr.io/ggml-org/whisper.cpp:main-vulkan", "Vulkan/CPU (main-vulkan)"),
        ]
        if shutil.which("nvidia-smi"):
            test_images.append(("ghcr.io/ggml-org/whisper.cpp:main-cuda", "CUDA (main-cuda)"))

        for test_image, label in test_images:
            print(f"  Testing {label}...", end="", file=sys.stderr, flush=True)
            if not image_exists(test_image):
                print(f" [{skip_mark}] not pulled", file=sys.stderr)
                continue
            try:
                result = subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--entrypoint",
                        "whisper-server",
                        test_image,
                        "--help",
                    ],
                    capture_output=True,
                    timeout=15,
                )
                if result.returncode == 0:
                    print(f" [{ok_mark}]", file=sys.stderr)
                else:
                    exit_info = f"exit {result.returncode}"
                    if result.returncode > 128:
                        sig = result.returncode - 128
                        exit_info = f"signal {sig} (exit {result.returncode})"
                        if sig == 4:
                            exit_info += " -- SIGILL: CPU does not support this image's instructions"
                    print(f" [{fail_mark}] {exit_info}", file=sys.stderr)
            except subprocess.TimeoutExpired:
                print(f" [{fail_mark}] timed out", file=sys.stderr)

    print("\n=== Models ===", file=sys.stderr)
    models_dir = Path(config["server"]["data_dir"]) / "models"
    if models_dir.exists():
        for model_file in sorted(models_dir.glob("ggml-*.bin")):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  {model_file.name} ({size_mb:.1f} MB)", file=sys.stderr)
    else:
        print(f"  Models directory not found: {models_dir}", file=sys.stderr)

    print("\n=== Config ===", file=sys.stderr)
    selected_path = vars(args).get("config")
    config_path = Path(selected_path).expanduser() if selected_path else _config_path()
    if config_path.exists():
        print(f"  {config_path}", file=sys.stderr)
    else:
        print(f"  {config_path} (not found; using defaults)", file=sys.stderr)
    print(f"  Backend: {resolved}", file=sys.stderr)
    if resolved == "remote":
        print("  Model: (on the remote machine)", file=sys.stderr)
    else:
        print(f"  Model: {model_for_backend(resolved, config)}", file=sys.stderr)
    print(f"  Image: {image}", file=sys.stderr)
    print(f"  Language: {config['transcribe']['language']}", file=sys.stderr)

    return 0


def main() -> None:
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

    try:
        config = load_config(args.config)
    except (OSError, ValueError) as exc:
        selected_path = Path(args.config).expanduser() if args.config else _config_path()
        print(f"Error: failed to load configuration {selected_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    commands = {
        "download": cmd_download,
        "start": cmd_start,
        "stop": cmd_stop,
        "destroy": cmd_destroy,
        "status": cmd_status,
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

    handler = commands.get(command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        sys.exit(handler(args, config))
    except KeyboardInterrupt:
        notify_close()
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
