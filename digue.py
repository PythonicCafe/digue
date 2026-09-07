#!/usr/bin/env python3
"""Local speech-to-text dictation and transcription using whisper.cpp."""

__version__ = "0.1.0"

import argparse
import sys

CONTAINER_NAME = "digue"
NOTIFY_REPLACE_ID = 48271
DEFAULT_PORT = 8178
DEFAULT_LANGUAGE = "auto"
DEFAULT_MODELS = {"nvidia": "large-v3-turbo", "amd": "large-v3-turbo", "intel": "large-v3-turbo", "cpu": "small"}
AVAILABLE_MODELS = ("tiny", "base", "small", "medium", "large-v3-turbo", "large-v3")
BACKENDS = ("nvidia", "amd", "intel", "cpu")
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
RESPONSE_FORMATS = ("text", "vtt", "srt")
SERVER_STARTUP_TIMEOUT = 180
TRANSCRIPTION_TIMEOUT = 120
BENCHMARK_TRANSCRIPTION_TIMEOUT = 300
BENCHMARK_RUNS = 3
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


# -- Config -------------------------------------------------------------------


def _config_path():
    import os
    from pathlib import Path

    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "digue" / "config.toml"


def _default_config():
    import os
    from pathlib import Path

    xdg = os.environ.get("XDG_DATA_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    data_dir = base / "digue"

    return {
        "server": {
            "port": DEFAULT_PORT,
            "data_dir": str(data_dir),
            "backend": "auto",
            "image": "",
        },
        "dictation": {
            "language": DEFAULT_LANGUAGE,
            "audio_dir": "",
            "display_server": "auto",
        },
        "models": dict(DEFAULT_MODELS),
    }


def load_config(config_path=None):
    """Loads config from TOML file, falling back to defaults for missing keys."""
    from pathlib import Path

    config = _default_config()
    path = Path(config_path) if config_path else _config_path()

    if path.exists():
        import tomllib

        with path.open("rb") as fobj:
            user_config = tomllib.load(fobj)
        for section, defaults in config.items():
            if section in user_config:
                for key in defaults:
                    toml_key = key.replace("_", "-")
                    if toml_key in user_config[section]:
                        config[section][key] = user_config[section][toml_key]
                    elif key in user_config[section]:
                        config[section][key] = user_config[section][key]

    if not config["dictation"]["audio_dir"]:
        config["dictation"]["audio_dir"] = str(Path(config["server"]["data_dir"]) / "audio")

    # Expand ~ in path values
    for key in ("data_dir",):
        config["server"][key] = str(Path(config["server"][key]).expanduser())
    for key in ("audio_dir",):
        config["dictation"][key] = str(Path(config["dictation"][key]).expanduser())

    return config


def model_for_backend(backend, config=None):
    """Returns the model name for a given backend, respecting config overrides."""
    if config and "models" in config:
        return config["models"].get(backend, DEFAULT_MODELS.get(backend, "small"))
    return DEFAULT_MODELS.get(backend, "small")


# -- Detection ----------------------------------------------------------------


def detect_backend():
    """Detects GPU backend: nvidia, amd, intel, or cpu."""
    import subprocess
    from pathlib import Path

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
            if "VGA" in line.upper() and "Intel" in line:
                if not any(marker in line for marker in old_gpu_markers):
                    return "intel"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return "cpu"


# -- Container management -----------------------------------------------------


def _docker_run(args, timeout=30):
    import subprocess

    return subprocess.run(
        ["docker"] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def container_exists():
    """Returns True if the digue container exists (running or stopped)."""
    result = _docker_run(["inspect", "--format", "{{.State.Status}}", CONTAINER_NAME])
    return result.returncode == 0


def container_status():
    """Returns container status string ('running', 'exited', etc.) or None."""
    result = _docker_run(["inspect", "--format", "{{.State.Status}}", CONTAINER_NAME])
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def image_exists(image):
    """Returns True if a Docker image exists locally."""
    result = _docker_run(["image", "inspect", image])
    return result.returncode == 0


def pull_image(image):
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


def resolve_backend(config):
    """Returns the backend to use, respecting config override or auto-detecting."""
    configured = config["server"].get("backend", "auto")
    if configured != "auto":
        return configured
    return detect_backend()


def resolve_image(backend, config):
    """Returns the Docker image to use, respecting config override."""
    configured = config["server"].get("image", "")
    if configured:
        return configured
    return DOCKER_IMAGES[backend]


def create_container(config, backend=None):
    """Creates the digue container.

    Resolves backend and image from config (with auto-detection fallback).
    Downloads the model and pulls the Docker image if not present locally.
    """
    from pathlib import Path

    if backend is None:
        backend = resolve_backend(config)

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
        f"127.0.0.1:{port}:8080",
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
        raise RuntimeError(f"Failed to create container: {result.stderr.strip()}")
    return backend


def remove_container():
    """Stops and removes the digue container."""
    _docker_run(["rm", "-f", CONTAINER_NAME])


def start_container():
    """Starts an existing stopped container."""
    result = _docker_run(["start", CONTAINER_NAME])
    return result.returncode == 0


def stop_container():
    """Stops the running container."""
    _docker_run(["stop", CONTAINER_NAME], timeout=15)


# -- Notifications ------------------------------------------------------------

_notify_send_warned = False


def notify(message, timeout_ms=0):
    """Prints message to stderr AND sends a desktop notification.

    The notification stays visible until replaced by the next one (timeout_ms=0).
    Pass a timeout for messages that should auto-dismiss (success, errors).
    If notify-send is not installed, prints a one-time warning and continues.
    """
    import subprocess

    global _notify_send_warned

    print(f"[digue] {message}", file=sys.stderr, flush=True)

    try:
        subprocess.run(
            [
                "notify-send",
                "-a",
                "digue",
                "--replace-id",
                str(NOTIFY_REPLACE_ID),
                "-t",
                str(timeout_ms),
                "Whisper",
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
    except subprocess.SubprocessError:
        pass


def notify_close():
    """Closes the current digue notification via D-Bus."""
    import subprocess

    try:
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
                str(NOTIFY_REPLACE_ID),
            ],
            capture_output=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        pass


# -- Server -------------------------------------------------------------------


def server_url(config):
    port = config["server"]["port"]
    return f"http://localhost:{port}/inference"


def is_server_running(config):
    """Returns True if server is responding to HTTP requests."""
    import urllib.error
    import urllib.request

    port = config["server"]["port"]
    try:
        urllib.request.urlopen(f"http://localhost:{port}/", timeout=1)
        return True
    except (urllib.error.URLError, OSError):
        return False


def _wait_for_server(config, verbose=False):
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


def ensure_server(config, silent=False):
    """Ensures server is running, creating/starting the container if needed.

    Returns the backend used, or None if the server was already running.
    """
    if is_server_running(config):
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


# -- HTTP helpers -------------------------------------------------------------


def _multipart_request(url, audio_data, fields, timeout):
    """Sends a multipart/form-data POST request using only stdlib."""
    import os
    import time
    import urllib.request

    boundary = f"----digue{os.getpid()}{time.time_ns()}"

    parts = []
    for field_name, field_value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{field_name}"\r\n\r\n{field_value}\r\n')
    file_part = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
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
    return response.read().decode()


# -- Transcription ------------------------------------------------------------


def transcribe(url, audio_path, language="auto", response_format="text", timeout=TRANSCRIPTION_TIMEOUT):
    """Sends audio to the server and returns the response (text, VTT, or SRT)."""
    audio_data = audio_path.read_bytes()
    fields = {"response_format": response_format}
    if language and language != "auto":
        fields["language"] = language
    return _multipart_request(url, audio_data, fields, timeout).strip()


# -- VTT simplification -------------------------------------------------------


def _strip_vtt_tags(text):
    """Removes inline VTT timing tags (YouTube word-level captions).

    Strips tags like ``<00:00:01.440>``, ``<c>``, ``</c>``.
    """
    import re

    text = text.replace("<c>", "").replace("</c>", "")
    text = re.sub(r"<[\d:.]+>", "", text)
    return text.strip()


def simplify_vtt(content):
    """Simplifies a VTT file to timestamped plain text, removing duplications."""
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
        if last_clean_text and last_clean_text.startswith(clean):
            continue

        result_lines.append(f"[{current_timestamp}] {clean}")
        last_clean_text = clean

    return "\n".join(result_lines)


# -- Download -----------------------------------------------------------------


def _download_progress_hook(label, with_notification=False):
    """Returns a reporthook callback for urlretrieve that prints a progress bar.

    When with_notification=True, also updates the desktop notification (~2x/s).
    """
    import time

    last_notify_time = [0.0]  # mutable for closure

    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            bar_width = 30
            filled = bar_width * pct // 100
            bar = "=" * filled + " " * (bar_width - filled)
            msg = f"{pct:3d}% [{bar}] {downloaded_mb:.1f}/{total_mb:.1f} MB"
            print(f"\r  {label}: {msg}", end="", file=sys.stderr, flush=True)
        else:
            downloaded_mb = downloaded / (1024 * 1024)
            msg = f"{downloaded_mb:.1f} MB"
            print(f"\r  {label}: {msg}", end="", file=sys.stderr, flush=True)

        if with_notification:
            now = time.monotonic()
            if now - last_notify_time[0] >= 0.5:
                notify(f"Downloading {label}... {msg}")
                last_notify_time[0] = now

    return hook


def download_model(model_name, models_dir, with_notification=False):
    """Downloads a whisper.cpp GGML model and the Silero VAD model."""
    import urllib.request
    from pathlib import Path

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
        urllib.request.urlretrieve(url, output, reporthook=_download_progress_hook(label, with_notification))
        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"\n  Saved: {output} ({size_mb:.1f} MB)", file=sys.stderr)


# -- Recording ----------------------------------------------------------------


def _runtime_dir():
    import os
    from pathlib import Path

    return Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))


def _pid_file():
    return _runtime_dir() / "digue.pid"


def _rec_file():
    return _runtime_dir() / "digue.wav"


def is_recording():
    pid_file = _pid_file()
    if not pid_file.exists():
        return False
    pid = int(pid_file.read_text().strip())
    return _pid_alive(pid)


def _pid_alive(pid):
    import os

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_recording():
    """Starts pw-record in background, returns the PID."""
    import subprocess

    rec_file = _rec_file()
    pid_file = _pid_file()
    proc = subprocess.Popen(
        ["pw-record", "--rate", "16000", "--channels", "1", "--format", "s16", str(rec_file)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pid_file.write_text(str(proc.pid))
    return proc.pid


def stop_recording():
    """Stops recording, returns the path to the audio file or None."""
    import os
    import time

    pid_file = _pid_file()
    rec_file = _rec_file()

    if not pid_file.exists():
        return None

    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, 15)  # SIGTERM
    except OSError:
        pass
    time.sleep(0.5)
    pid_file.unlink(missing_ok=True)

    if not rec_file.exists() or rec_file.stat().st_size == 0:
        rec_file.unlink(missing_ok=True)
        return None
    return rec_file


# -- Clipboard ----------------------------------------------------------------


def detect_display_server():
    """Detects whether the session is Wayland or X11."""
    import os

    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return None


def paste_text(text, display_server="auto"):
    """Copies text to clipboard and simulates Ctrl+V.

    Uses xclip+xdotool on X11, wl-copy+wtype on Wayland.
    Raises RuntimeError with actionable message on failure.
    """
    import subprocess

    if display_server == "auto":
        display_server = detect_display_server()

    if display_server is None:
        raise RuntimeError("No DISPLAY or WAYLAND_DISPLAY set. Cannot access clipboard.")

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
        subprocess.run(paste_cmd, capture_output=True, timeout=5)
    except FileNotFoundError:
        raise RuntimeError(f"{paste_cmd[0]} not found. Install with: sudo apt install {paste_pkg}")


# -- Dictate ------------------------------------------------------------------


def save_audio(rec_file, audio_dir):
    """Copies audio to timestamped file in audio_dir. Returns (saved_path, timestamp)."""
    import datetime
    import shutil
    from pathlib import Path

    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    saved = audio_dir / f"{timestamp}.wav"
    shutil.copy2(rec_file, saved)
    return saved, timestamp


def dictate_toggle(config):
    """Toggle recording/transcription. Returns exit code."""
    from pathlib import Path

    if is_recording():
        rec_file = stop_recording()
        if rec_file is None:
            notify("Empty or missing audio file", timeout_ms=5000)
            return 1

        audio_dir = config["dictation"]["audio_dir"]
        saved, timestamp = save_audio(rec_file, audio_dir)

        notify("Transcribing...")
        try:
            url = server_url(config)
            language = config["dictation"]["language"]
            text = transcribe(url, rec_file, language)
        except Exception as exc:
            notify(f"Transcription failed: {exc}", timeout_ms=10000)
            return 1
        finally:
            rec_file.unlink(missing_ok=True)

        text_path = Path(audio_dir) / f"{timestamp}.txt"
        text_path.write_text(text + "\n")

        if not text:
            notify("No speech detected", timeout_ms=5000)
            return 0

        try:
            paste_text(text, display_server=config["dictation"]["display_server"])
        except Exception as exc:
            notify(f"Paste failed: {exc}", timeout_ms=10000)
            print(f"Transcription saved to: {text_path}", file=sys.stderr)
            return 1
        notify(f"Pasted ({len(text)} chars)", timeout_ms=5000)
        return 0
    else:
        try:
            result = ensure_server(config)
            if result is None and not is_server_running(config):
                notify("Could not start whisper-server", timeout_ms=5000)
                return 1
            start_recording()
        except FileNotFoundError:
            notify("pw-record not found. Install with: sudo apt install pipewire", timeout_ms=10000)
            return 1
        except Exception as exc:
            notify(f"Failed to start recording: {exc}", timeout_ms=5000)
            return 1
        notify("Recording... (press again to stop)")
        return 0


# -- Benchmark ----------------------------------------------------------------


def _benchmark_run(url, audio_path, language, runs):
    """Runs N transcription requests and returns list of (elapsed_ms, text)."""
    import time

    try:
        transcribe(url, audio_path, language, timeout=BENCHMARK_TRANSCRIPTION_TIMEOUT)
    except Exception:
        pass

    results = []
    for _run in range(runs):
        start = time.perf_counter()
        text = transcribe(url, audio_path, language, timeout=BENCHMARK_TRANSCRIPTION_TIMEOUT)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        results.append((elapsed_ms, text))
    return results


def run_benchmark(audio_path, config):
    """Benchmarks different backend/model combinations with the same audio."""
    import time
    from pathlib import Path

    models_dir = Path(config["server"]["data_dir"]) / "models"
    url = server_url(config)
    language = config["dictation"]["language"]
    detected = detect_backend()

    print("digue benchmark", file=sys.stderr)
    print(f"Audio: {audio_path}", file=sys.stderr)
    print(f"Detected backend: {detected}", file=sys.stderr)
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
    for label, backend, model in test_cases:
        print(f"=== {label} ===", file=sys.stderr)

        remove_container()
        time.sleep(2)

        bench_config = {**config, "models": {**config["models"], backend: model}}
        try:
            create_container(bench_config, backend)
        except RuntimeError as exc:
            print(f"  Skipped: {exc}", file=sys.stderr)
            continue

        print("  Waiting for server...", file=sys.stderr, flush=True)
        if not _wait_for_server(config, verbose=True):
            print("  Server failed to start, skipping", file=sys.stderr)
            remove_container()
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

        remove_container()

    # Restore default container
    print("Restoring default container...", file=sys.stderr, flush=True)
    create_container(config)
    _wait_for_server(config, verbose=True)

    print(f"\n{'=' * 50}", file=sys.stderr)
    print("Summary", file=sys.stderr)
    print(f"{'=' * 50}", file=sys.stderr)
    for label, avg_ms in all_results:
        print(f"  {label:<35} {avg_ms}ms", file=sys.stderr)


def record_benchmark_audio(output_path, duration_seconds=10):
    """Records audio from microphone for benchmark."""
    import subprocess
    import time

    print(f"Recording {duration_seconds}s from microphone...", file=sys.stderr)
    print("(speak something so there is content to transcribe)", file=sys.stderr)
    proc = subprocess.Popen(
        ["pw-record", "--rate", "16000", "--channels", "1", "--format", "s16", str(output_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(duration_seconds)
    proc.terminate()
    time.sleep(0.5)
    print(f"Recorded: {output_path}", file=sys.stderr)


# -- CLI ----------------------------------------------------------------------


def _existing_dir(value):
    """argparse type: validates that the path is an existing directory."""
    from pathlib import Path

    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory not found: {value}")
    return path


def _ensure_dir(value):
    """argparse type: creates the directory if it doesn't exist."""
    from pathlib import Path

    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_parser():
    from pathlib import Path

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
    parser.set_defaults(command=None)

    subparsers = parser.add_subparsers(dest="command", metavar="command")

    subparsers.add_parser("detect", help="Detect GPU backend and print it")

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
    subparsers.add_parser("dictate", help="Toggle recording/transcription (default when no command given)")

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
        choices=RESPONSE_FORMATS,
        default="text",
        help=f"Output format. Options: {', '.join(RESPONSE_FORMATS)} (default: text)",
    )
    sub_transcribe.add_argument(
        "-l",
        "--language",
        default=None,
        help="Language code, e.g. pt, en (default: from config or auto)",
    )

    sub_simplify = subparsers.add_parser("simplify-vtt", help="Simplify a VTT file to timestamped plain text")
    sub_simplify.add_argument("input", help="Input VTT file (use - for stdin)")
    sub_simplify.add_argument(
        "-o",
        "--output",
        metavar="path",
        default=None,
        help="Output file (default: stdout)",
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
        default="vtt",
        help=f"Output format. Options: {', '.join(RESPONSE_FORMATS)} (default: vtt)",
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

    subparsers.add_parser("config", help="Show current configuration as JSON")
    subparsers.add_parser("doctor", help="Check system dependencies and test Docker images")

    return parser


def cmd_detect(args):
    backend = detect_backend()
    print(backend)
    return 0


def cmd_download(args, config):
    from pathlib import Path

    models_dir = Path(config["server"]["data_dir"]) / "models"

    if args.model:
        model = args.model
    else:
        backend = resolve_backend(config)
        model = model_for_backend(backend, config)
        print(f"Backend: {backend}, downloading model: {model}", file=sys.stderr)

    download_model(model, models_dir)
    return 0


def cmd_start(args, config):
    if is_server_running(config):
        print("server is already running", file=sys.stderr)
        return 0

    status = container_status()
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

    if _wait_for_server(config, verbose=True):
        return 0

    print("Server failed to start. Check: docker logs whisper-server", file=sys.stderr)
    return 1


def cmd_stop(args, config):
    stop_container()
    print("Server stopped", file=sys.stderr)
    return 0


def cmd_destroy(args, config):
    remove_container()
    print("Container removed", file=sys.stderr)
    return 0


def cmd_status(args, config):
    status = container_status()
    port = config["server"]["port"]
    if status is None:
        print("Container does not exist", file=sys.stderr)
        return 1
    http_ok = is_server_running(config)
    label = f"{status}, {'responding' if http_ok else 'not responding'} on port {port}"
    print(f"digue: {label}", file=sys.stderr)
    return 0 if http_ok else 1


def cmd_dictate(args, config):
    return dictate_toggle(config)


def cmd_transcribe(args, config):
    from pathlib import Path

    ensure_server(config, silent=True)
    if not is_server_running(config):
        print("Error: server is not running. Run: digue start", file=sys.stderr)
        return 1

    audio_path = args.audio
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        return 1

    language = args.language or config["dictation"]["language"]
    url = server_url(config)

    print(f"Transcribing {audio_path.name}...", file=sys.stderr, flush=True)
    result = transcribe(url, audio_path, language, args.response_format)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result + "\n")
        print(f"Saved: {output_path}", file=sys.stderr)
    else:
        print(result)
    return 0


def cmd_simplify_vtt(args, config):
    from pathlib import Path

    if args.input == "-":
        content = sys.stdin.read()
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: file not found: {input_path}", file=sys.stderr)
            return 1
        content = input_path.read_text()

    result = simplify_vtt(content)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result + "\n")
        print(f"Saved: {output_path}", file=sys.stderr)
    else:
        print(result)
    return 0


def _format_extension(response_format):
    return {"text": ".txt", "vtt": ".vtt", "srt": ".srt"}[response_format]


def cmd_batch_transcribe(args, config):
    import time

    ensure_server(config, silent=True)
    if not is_server_running(config):
        print("Error: server is not running. Run: digue start", file=sys.stderr)
        return 1

    language = args.language or config["dictation"]["language"]
    url = server_url(config)
    ext = _format_extension(args.response_format)

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

    for idx, (audio_file, output_file) in enumerate(pending, 1):
        print(f"[{idx}/{len(pending)}] {audio_file.name}...", file=sys.stderr, flush=True)
        start = time.perf_counter()
        try:
            result = transcribe(url, audio_file, language, args.response_format)
            output_file.write_text(result + "\n")
            elapsed = time.perf_counter() - start
            print(f"  Saved: {output_file.name} ({elapsed:.1f}s)", file=sys.stderr)
        except Exception as exc:
            print(f"  Error: {exc}", file=sys.stderr)
            continue

    print(f"Done. Output: {args.output_dir}", file=sys.stderr)
    return 0


def cmd_batch_simplify_vtt(args, config):
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

    for idx, (vtt_file, output_file) in enumerate(pending, 1):
        print(f"[{idx}/{len(pending)}] {vtt_file.name}...", file=sys.stderr, flush=True)
        content = vtt_file.read_text()
        result = simplify_vtt(content)
        output_file.write_text(result + "\n")
        print(f"  Saved: {output_file.name}", file=sys.stderr)

    print(f"Done. Output: {args.output_dir}", file=sys.stderr)
    return 0


def cmd_benchmark(args, config):
    from pathlib import Path

    if args.audio:
        audio_path = args.audio
        if not audio_path.exists():
            print(f"Error: file not found: {audio_path}", file=sys.stderr)
            return 1
    else:
        audio_path = Path("/tmp/digue-bench.wav")
        record_benchmark_audio(audio_path)
        print(file=sys.stderr)

    run_benchmark(audio_path, config)
    return 0


def cmd_config(args, config):
    import json

    print(json.dumps(config, indent=2))
    return 0


def cmd_doctor(args, config):
    """Checks system dependencies and tests which Docker images work."""
    import shutil
    import subprocess
    from pathlib import Path

    ok_mark = "OK"
    fail_mark = "FAIL"
    skip_mark = "SKIP"

    print("=== System dependencies ===", file=sys.stderr)
    tools = {
        "docker": "Docker runtime",
        "pw-record": "PipeWire audio recording (dictation)",
        "notify-send": "Desktop notifications (libnotify-bin)",
        "xclip": "X11 clipboard",
        "xdotool": "X11 key simulation",
        "wl-copy": "Wayland clipboard (wl-clipboard)",
        "wtype": "Wayland key simulation",
        "lspci": "GPU detection (pciutils)",
        "nvidia-smi": "NVIDIA GPU detection",
        "vulkaninfo": "Vulkan verification (vulkan-tools)",
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
    image = resolve_image(resolved, config)
    print(f"  Image: {image}", file=sys.stderr)

    print("\n=== Docker images ===", file=sys.stderr)
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
    config_path = _config_path()
    if config_path.exists():
        print(f"  {config_path}", file=sys.stderr)
    else:
        print("  No config file (using defaults)", file=sys.stderr)
    print(f"  Backend: {resolved}", file=sys.stderr)
    print(f"  Model: {model_for_backend(resolved, config)}", file=sys.stderr)
    print(f"  Image: {image}", file=sys.stderr)
    print(f"  Language: {config['dictation']['language']}", file=sys.stderr)

    return 0


def main():
    parser = create_parser()
    args = parser.parse_args()

    command = args.command
    if command is None:
        command = "dictate"

    if command == "detect":
        sys.exit(cmd_detect(args))

    config = load_config(args.config)

    commands = {
        "download": cmd_download,
        "start": cmd_start,
        "stop": cmd_stop,
        "destroy": cmd_destroy,
        "status": cmd_status,
        "dictate": cmd_dictate,
        "transcribe": cmd_transcribe,
        "simplify-vtt": cmd_simplify_vtt,
        "batch-transcribe": cmd_batch_transcribe,
        "batch-simplify-vtt": cmd_batch_simplify_vtt,
        "benchmark": cmd_benchmark,
        "config": cmd_config,
        "doctor": cmd_doctor,
    }

    handler = commands.get(command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        sys.exit(handler(args, config))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
