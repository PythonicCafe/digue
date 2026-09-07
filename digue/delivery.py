"""Clipboard paste/type delivery to the focused window."""

from __future__ import annotations

import contextlib
import os
from typing import Any


def detect_display_server() -> str | None:
    """Detects whether the session is Wayland or X11."""

    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return None


PASTE_KEYS = ("ctrl+v", "ctrl+shift+v", "shift+insert")


def send_text(text: str, display_server: str = "auto", input_mode: str = "paste", paste_key: str = "ctrl+v") -> None:
    """Sends text to the focused window.

    input_mode "paste" copies to the clipboard and simulates paste_key (PASTE_KEYS): "ctrl+v" (GUI apps; terminals
    ignore it), "ctrl+shift+v" (terminals and browsers; GTK/Qt apps ignore it, LibreOffice opens Paste Special) or
    "shift+insert" (the X11-wide paste: terminals, GTK, Qt, browsers, LibreOffice, VS Code). xterm, urxvt and
    alacritty paste the PRIMARY selection on Shift+Insert, so that key also fills PRIMARY.
    input_mode "type" simulates keystrokes (useful in terminals, where the paste shortcut differs). Typing is slower
    and may drop characters in slow applications.
    Raises RuntimeError with actionable message on failure.

    The whole body runs under an exclusive flock ("digue-delivery.lock"): the clipboard is global, and two overlapping
    deliveries pasting within the same window would deliver one text twice and lose the other. The "type" mode has the
    sibling race (interleaved keystrokes into the focused window), so it is serialized too. This is a dedicated lock,
    not _dictate_lock: a delivery can take seconds (paste timeout is 5s) and must not block state transitions.
    """
    if display_server == "auto":
        detected = detect_display_server()
        if detected is None:
            raise RuntimeError("No DISPLAY or WAYLAND_DISPLAY set. Cannot access clipboard or send keystrokes.")
        display_server = detected

    if paste_key not in PASTE_KEYS:
        raise ValueError(f"Unknown paste key {paste_key!r}; expected one of {', '.join(PASTE_KEYS)}")
    with _delivery_lock():
        _send_text_locked(text, display_server, input_mode, paste_key)


def _delivery_lock() -> Any:
    """Serializes deliveries (clipboard copy+paste or keystroke typing) between overlapping takes."""
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


def _paste_commands(display_server: str, paste_key: str) -> tuple[list[list[str]], list[str]]:
    """(copy commands, paste command) for a paste key on X11 or Wayland.

    The copy commands fill the clipboard, plus the PRIMARY selection for "shift+insert" (what xterm, urxvt and
    alacritty paste on that key; gnome-terminal and kitty paste the clipboard). wtype types a bare argument as text,
    so the named key goes through -k (xkb name); xdotool takes the xkb name in the chord itself.
    """
    modifiers = paste_key.split("+")[:-1]
    key = paste_key.split("+")[-1]
    key_name = "Insert" if key == "insert" else key
    if display_server == "wayland":
        copies = [["wl-copy"]]
        if paste_key == "shift+insert":
            copies.append(["wl-copy", "--primary"])
        paste = ["wtype"]
        for modifier in modifiers:
            paste += ["-M", modifier]
        paste += ["-k", key_name] if key == "insert" else [key_name]
        for modifier in reversed(modifiers):
            paste += ["-m", modifier]
        return copies, paste
    copies = [["xclip", "-selection", "clipboard"]]
    if paste_key == "shift+insert":
        copies.append(["xclip", "-selection", "primary"])
    return copies, ["xdotool", "key", "--clearmodifiers", "+".join([*modifiers, key_name])]


def _send_text_locked(text: str, display_server: str, input_mode: str, paste_key: str = "ctrl+v") -> None:
    """The delivery itself; the caller holds the delivery lock."""
    import subprocess

    if input_mode == "type":
        # The text goes through stdin, never argv: wtype rejects any unknown -option ("Unknown parameter", it has no
        # --no-newline flag) and xdotool would parse a transcript starting with "-" as an option.
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

    copy_cmds, paste_cmd = _paste_commands(display_server, paste_key)
    if display_server == "wayland":
        copy_pkg = "wl-clipboard"
        paste_pkg = "wtype"
    else:
        copy_pkg = "xclip"
        paste_pkg = "xdotool"

    for copy_cmd in copy_cmds:
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
            raise RuntimeError(
                f"{copy_cmd[0]} failed: {exc.stderr.decode().strip() if exc.stderr else 'unknown error'}"
            )

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

    Line breaks come from whisper segment boundaries (word-aligned once token_timestamps is disabled, see _send_audio),
    so joining with a single space is safe; mid-word splits do not occur anymore.
    """
    return " ".join(text.split())
