"""Shared pytest fixtures."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def isolate_runtime_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    return runtime_dir


@pytest.fixture(autouse=True)
def block_desktop_notifications(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Create fake `notify-send` and `gdbus` on a fake path so the real ones are not called in tests

    No test may reach the real notify-send/gdbus: tests that assert on notifications mock
    `digue.notify.send_notification` explicitly; this autouse fixture only stops the ones that exercise the real
    delivery flow from popping desktop notifications on the developer's machine.  `digue.notify` imports subprocess
    lazily inside its functions (so patching the module attribute does not work); the block goes through PATH instead:
    no-op stubs shadow the real binaries while /usr/bin:/bin stays available for the real processes some tests spawn
    (sleep, sys.executable).
    """
    fake_bin = tmp_path / "no-notifications"
    fake_bin.mkdir()
    for name in ("notify-send", "gdbus"):
        stub = fake_bin / name
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:/usr/bin:/bin")
