"""Clipboard paste/type delivery to the focused window."""

from __future__ import annotations

import contextlib
import os
from typing import Any

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

    from digue.recording import _runtime_dir

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


def normalize_pasted_text(text: str) -> str:
    """Joins wrapped lines into a single clean line.

    Line breaks come from whisper segment boundaries (word-aligned once
    token_timestamps is disabled, see _send_audio), so joining with a single
    space is safe; mid-word splits do not occur anymore.
    """
    return " ".join(text.split())
