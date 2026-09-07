"""Shared pytest fixtures."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_runtime_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    return runtime_dir
