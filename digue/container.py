"""Docker container lifecycle, model download, and whisper-server control."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import subprocess

from digue.config import model_for_backend

CONTAINER_NAME = "digue-whisper.cpp"

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

VAD_MODEL_FILENAME = "ggml-silero-v6.2.0.bin"

HUGGINGFACE_VAD_URL = f"https://huggingface.co/ggml-org/whisper-vad/resolve/main/{VAD_MODEL_FILENAME}"

SERVER_STARTUP_TIMEOUT = 180

DOWNLOAD_TIMEOUT = 60


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


DOCKER_NOT_FOUND = (
    'docker not found. Install Docker (https://docs.docker.com/engine/install/) or use backend = "remote"'
)


class DockerNotFoundError(RuntimeError):
    """The docker binary is missing: every local-backend command needs it."""


def _docker_run(args: list[str], timeout: int | float = 30) -> subprocess.CompletedProcess[str]:
    import subprocess

    try:
        return subprocess.run(
            ["docker"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise DockerNotFoundError(DOCKER_NOT_FOUND) from exc


def resolve_container_name(config: dict[str, dict[str, Any]]) -> str:
    """Docker name for the whisper-server container (`server.container-name`).

    Every docker call takes the name explicitly (no CONTAINER_NAME fallback):
    a default that silently ignored the config is how the ffmpeg fallback in
    `audio` kept talking to a container that no longer had that name.
    """
    name = str(config["server"].get("container_name") or "")
    return name if name else CONTAINER_NAME


def container_exists(name: str) -> bool:
    """Returns True if the container exists (running or stopped)."""
    result = _docker_run(["inspect", "--format", "{{.State.Status}}", name])
    return result.returncode == 0


def container_status(name: str) -> str | None:
    """Returns container status string ('running', 'exited', etc.) or None."""
    result = _docker_run(["inspect", "--format", "{{.State.Status}}", name])
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def container_image(name: str) -> str | None:
    """Returns the image the container was created from, or None."""
    result = _docker_run(["inspect", "--format", "{{.Config.Image}}", name])
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def _image_mismatch(config: dict[str, dict[str, Any]]) -> tuple[str, str] | None:
    """(current, configured) when the existing container was created from an image other than the one the config
    resolves to now, else None.

    `docker start` reuses the container's original image: a new `image` in the config (or `server start --image`) would
    otherwise silently do nothing until the container is destroyed by hand.
    """
    current = container_image(resolve_container_name(config))
    if current is None:
        return None
    configured = resolve_image(resolve_backend(config), config)
    if not configured or current == configured:
        return None
    return current, configured


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
    try:
        result = subprocess.run(
            ["docker", "pull", image],
            timeout=600,
        )
    except FileNotFoundError as exc:
        raise DockerNotFoundError(DOCKER_NOT_FOUND) from exc
    if result.returncode != 0:
        raise RuntimeError(f"Failed to pull image: {image}")
    print(f"Pull complete: {image}", file=sys.stderr)


def _is_remote(config: dict[str, dict[str, Any]]) -> bool:
    """Returns whether the validated configuration selects a remote server."""
    return bool(config["server"]["backend"] == "remote")


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


def create_container(
    config: dict[str, dict[str, Any]], backend: str | None = None, *, with_notification: bool = False
) -> str:
    """Creates the digue container.

    Resolves backend and image from config (with auto-detection fallback).  Downloads the model and pulls the Docker
    image if not present locally.  Desktop notifications are for the dictation hotkey path only.
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

    # Ensure models are downloaded before creating the container (avoids crash loop)
    model_path = models_dir / f"ggml-{model}.bin"
    vad_path = models_dir / VAD_MODEL_FILENAME
    if not model_path.exists() or not vad_path.exists():
        missing = model_path if not model_path.exists() else vad_path
        print(f"Model not found: {missing.name}. Downloading...", file=sys.stderr, flush=True)
        download_model(model, models_dir, with_notification=with_notification)

    pull_image(image)

    cmd = [
        "run",
        "-d",
        "--name",
        resolve_container_name(config),
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
        f"/models/{VAD_MODEL_FILENAME}",
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


def remove_container(name: str) -> None:
    """Stops and removes the container."""
    result = _docker_run(["rm", "-f", name])
    _raise_for_docker_failure(result, "remove container")


def start_container(name: str) -> bool:
    """Starts an existing stopped container."""
    result = _docker_run(["start", name])
    _raise_for_docker_failure(result, "start container")
    return True


def stop_container(name: str) -> None:
    """Stops the running container."""
    result = _docker_run(["stop", name], timeout=15)
    _raise_for_docker_failure(result, "stop container")


def _rename_container(old_name: str, new_name: str) -> None:
    result = _docker_run(["rename", old_name, new_name])
    _raise_for_docker_failure(result, "rename container")


@contextlib.contextmanager
def preserve_container_for_benchmark(container: str) -> Iterator[None]:
    """Makes room for benchmark containers, then restores the prior container and running state."""
    previous_status = container_status(container)
    backup_name = f"{container}-benchmark-backup-{os.getpid()}"

    if previous_status is not None:
        if previous_status == "running":
            stop_container(container)
        try:
            _rename_container(container, backup_name)
        except BaseException:
            if previous_status == "running":
                start_container(container)
            raise

    try:
        yield
    finally:
        try:
            if container_exists(container):
                remove_container(container)
        finally:
            if previous_status is not None:
                _rename_container(backup_name, container)
                if previous_status == "running":
                    start_container(container)


# Server


def server_host(config: dict[str, dict[str, Any]]) -> str:
    """Returns the host the server is probed on.

    With backend 'remote', returns the configured remote host (default
    127.0.0.1 for an SSH tunnel). For local backends, probes the address Docker
    binds, except that the wildcard bind is reached through loopback.
    """
    if _is_remote(config):
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

    When verbose=True, prints elapsed time to stderr every 10 seconds: one
    redrawn line on a TTY, plain lines on a captured stderr (the same
    contract as the download progress and `send_notification`).
    """
    import time

    from digue.notify import _stderr_is_tty

    tty = _stderr_is_tty()
    start = time.perf_counter()
    for attempt in range(SERVER_STARTUP_TIMEOUT):
        time.sleep(1)
        if is_server_running(config):
            if verbose:
                elapsed = time.perf_counter() - start
                prefix = "\r" if tty else ""
                padding = " " * 10 if tty else ""
                print(f"{prefix}Server ready ({elapsed:.0f}s){padding}", file=sys.stderr)
            return True
        if verbose and attempt > 0 and attempt % 10 == 0:
            elapsed = time.perf_counter() - start
            message = f"  Waiting for model to load... {elapsed:.0f}s"
            if tty:
                print(f"\r{message}", end="", file=sys.stderr, flush=True)
            else:
                print(message, file=sys.stderr, flush=True)

    if verbose and tty:
        print(file=sys.stderr)
    return False


def ensure_server(config: dict[str, dict[str, Any]], silent: bool = False) -> str | None:
    """Ensures server is running, creating/starting the container if needed.

    Returns the backend used, or None if the server was already running.
    With backend 'remote' no local container is ever touched; the server is
    expected to be reachable through an SSH tunnel.
    """

    from digue.notify import notify_close, send_notification

    if is_server_running(config):
        return None

    if _is_remote(config):
        host = config["server"].get("remote_host") or "127.0.0.1"
        if not silent:
            send_notification(
                f"Remote server {host}:{config['server']['port']} not responding. Is your tunnel active / host reachable?",
                timeout_ms=10000,
            )
        return None

    name = resolve_container_name(config)
    status = container_status(name)
    backend = None

    if status is not None:
        mismatch = _image_mismatch(config)
        if mismatch is not None:
            # Not recreated here: this runs behind a dictation hotkey, and a
            # multi-GB pull is not what a keypress asked for. server start does.
            current, configured = mismatch
            print(
                f"Warning: container runs {current} but the config selects {configured}; "
                "run `digue server start` to recreate it",
                file=sys.stderr,
            )
    if status == "exited":
        if not silent:
            send_notification("Starting server...")
        start_container(name)
    elif status == "running":
        if not silent:
            send_notification("Server starting...")
    elif status is None:
        backend = resolve_backend(config)
        if not silent:
            send_notification(f"Creating server ({backend})...")
        create_container(config, backend, with_notification=not silent)
    else:
        if not silent:
            send_notification(f"Container in unexpected state: {status}", timeout_ms=5000)
        return None

    if _wait_for_server(config, verbose=not silent):
        if not silent:
            notify_close()
        return backend

    if not silent:
        send_notification(f"Server failed to start (see: docker logs {name})", timeout_ms=10000)
    return None


def server_not_running_hint(config: dict[str, dict[str, Any]]) -> str:
    """Returns the actionable hint shown when the server is not responding."""
    if _is_remote(config):
        host = config["server"].get("remote_host") or "127.0.0.1"
        if host == "127.0.0.1":
            return (
                f"Backend is 'remote': no local container to start. Forward port {config['server']['port']} with "
                f"ssh -NfL {config['server']['port']}:127.0.0.1:{config['server']['port']} user@host "
                "(see README, Remote access), or set server.remote-host to a LAN host."
            )
        return f"Backend is 'remote' and server {host}:{config['server']['port']} is not responding (see README, Remote access)."
    return "Run: digue server start"


# Download


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

    from digue.notify import _stderr_is_tty, send_notification

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
                send_notification(f"Downloading {label}... {short_msg}")
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
    vad_path = models_dir / VAD_MODEL_FILENAME

    for label, url, output in [
        (f"ggml-{model_name}.bin", f"{HUGGINGFACE_MODEL_URL}/ggml-{model_name}.bin", model_path),
        (f"{VAD_MODEL_FILENAME} (VAD)", HUGGINGFACE_VAD_URL, vad_path),
    ]:
        if output.exists():
            size_mb = output.stat().st_size / (1024 * 1024)
            print(f"Already exists: {output} ({size_mb:.1f} MB)", file=sys.stderr)
            continue
        print(f"Downloading {label}...", file=sys.stderr, flush=True)
        _download_file(url, output, label, with_notification)
        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"\n  Saved: {output} ({size_mb:.1f} MB)", file=sys.stderr)


def cmd_detect(args: argparse.Namespace) -> int:
    backend = detect_backend()
    print(backend)
    return 0


def cmd_models(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    """Lists every model `digue` accepts, with size, whether it is already
    downloaded, and a `*` on the one this machine's backend resolves to."""
    from digue import AVAILABLE_MODELS
    from digue.benchmark import MODEL_SIZES_MB

    models_dir = Path(config["server"]["data_dir"]) / "models"
    backend = resolve_backend(config)
    selected = model_for_backend(backend, config) if backend != "remote" else None
    width = max(len(name) for name in AVAILABLE_MODELS)
    for name in AVAILABLE_MODELS:
        marker = "*" if name == selected else " "
        downloaded = "  downloaded" if (models_dir / f"ggml-{name}.bin").exists() else ""
        print(f"{marker} {name:<{width}}  {MODEL_SIZES_MB[name]:>5} MB{downloaded}")
    if selected is not None:
        print(f"\n* selected for backend {backend} ([models] {backend})", file=sys.stderr)
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


def cmd_server_start(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    """Starts (or creates) the container. `--image` overrides `server.image`
    for this run; a container created from another image than the one the
    config now selects is removed and recreated, since `docker start` would
    keep the old image."""
    if _is_remote(config):
        if is_server_running(config):
            print("server is already running", file=sys.stderr)
            return 0
        print(server_not_running_hint(config), file=sys.stderr)
        return 1

    image_override = getattr(args, "image", None)
    if image_override:
        config["server"]["image"] = image_override

    name = resolve_container_name(config)
    status = container_status(name)
    try:
        mismatch = _image_mismatch(config) if status is not None else None
        if mismatch is not None:
            current, configured = mismatch
            print(f"Container runs {current}; recreating with {configured}...", file=sys.stderr, flush=True)
            remove_container(name)
            status = None
        elif is_server_running(config):
            print("server is already running", file=sys.stderr)
            return 0
        if status == "exited":
            print("Starting existing container...", file=sys.stderr, flush=True)
            start_container(name)
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

    print(f"Server failed to start. Check: docker logs {name}", file=sys.stderr)
    return 1


def cmd_server_stop(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    if _is_remote(config):
        print("Backend is 'remote': there is no local container to stop", file=sys.stderr)
        return 1
    try:
        stop_container(resolve_container_name(config))
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Server stopped", file=sys.stderr)
    return 0


def cmd_server_destroy(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    if _is_remote(config):
        print("Backend is 'remote': there is no local container to remove", file=sys.stderr)
        return 1
    try:
        remove_container(resolve_container_name(config))
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Container removed", file=sys.stderr)
    return 0


def cmd_server_status(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    port = config["server"]["port"]
    if _is_remote(config):
        http_ok = is_server_running(config)
        host = config["server"].get("remote_host") or "127.0.0.1"
        print(
            f"digue: remote backend, {'responding' if http_ok else 'not responding'} on {host}:{port}", file=sys.stderr
        )
        return 0 if http_ok else 1
    status = container_status(resolve_container_name(config))
    if status is None:
        print("Container does not exist", file=sys.stderr)
        return 1
    http_ok = is_server_running(config)
    label = f"{status}, {'responding' if http_ok else 'not responding'} on port {port}"
    print(f"digue: {label}", file=sys.stderr)
    return 0 if http_ok else 1


def cmd_doctor(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    """Checks system dependencies and tests which Docker images work."""
    import shutil
    import subprocess

    from digue.config import _config_path

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

    print("\n=== Capture device ===", file=sys.stderr)
    configured = str(config["dictate"].get("device") or "")
    print(f"  Configured: {configured or '(system default)'}", file=sys.stderr)
    if shutil.which("arecord"):
        try:
            listed = subprocess.run(
                ["arecord", "-l"],
                capture_output=True,
                timeout=5,
                check=False,
                text=True,
            )
            for line in listed.stdout.splitlines():
                if line.strip():
                    print(f"  {line}", file=sys.stderr)
        except (OSError, subprocess.TimeoutExpired):
            print("  arecord -l failed", file=sys.stderr)
    if shutil.which("pw-record"):
        print(
            "  pw-record --target NAME (node name or serial). This build has no --list-targets;",
            file=sys.stderr,
        )
        print("  find sources with: pactl list sources short   or   wpctl status", file=sys.stderr)

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
        if image not in {known for known, _label in test_images}:
            # a custom image from the config is the one that matters most here
            test_images.append((image, "configured image"))

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
