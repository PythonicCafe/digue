"""Desktop notifications and TTY-aware stderr progress."""

from __future__ import annotations

import contextlib
import os
import sys

NOTIFY_REPLACE_ID = 48271
NOTIFY_ID_SLOTS = 32  # concurrent takes: id = base + (pid % slots), so popups of
# overlapping dictations do not replace or close each other.
_notify_send_warned = False
_last_notify_len = 0


def send_notification(message: str, timeout_ms: int = 0) -> None:
    """Prints message to stderr AND sends a desktop notification.

    The notification stays visible until replaced by the next one (timeout_ms=0).
    Pass a timeout for messages that should auto-dismiss (success, errors).
    If notify-send is not installed, prints a one-time warning and continues.

    On a terminal the stderr line is redrawn (\\r, padded to erase a previous shorter message), so it coexists with
    single-line progress bars; on a captured stderr it is a plain line with \\n.
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
                str(notification_id()),
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


def notification_id(pid: int | None = None) -> int:
    """The notification slot of a digue process (this one by default)."""
    return NOTIFY_REPLACE_ID + (os.getpid() if pid is None else pid) % NOTIFY_ID_SLOTS


def notify_close(pid: int | None = None) -> None:
    """Closes a digue notification via D-Bus: this process's slot, or the slot of another (usually dead) daemon whose
    pid is known."""
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
                str(notification_id(pid)),
            ],
            capture_output=True,
            timeout=5,
        )


def _stderr_is_tty() -> bool:
    """Returns True if stderr is a terminal (dynamic progress makes sense).

    With captured/piped stderr, \\r has no visual effect and every update becomes a full line in the log -- hence the
    sparse-line mode in progress and notification prints.
    """
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
