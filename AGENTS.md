# AGENTS.md

> Single-file Python CLI tool for local speech-to-text using whisper.cpp in Docker. Stdlib only, no pip dependencies.

## Commands

- Test: `pytest tests/ -v --tb=short`
- Lint: `ruff check . --fix && ruff format --line-length 120`
- Run: `python digue.py` (toggle dictation), `python digue.py detect` etc.

## Conventions

- **Stdlib only.** No external runtime dependencies. `tomllib` (3.11+), `urllib.request`, `subprocess`, `pathlib`.
- **Single file.** All logic lives in `digue.py`. Do not split into modules.
- **English everywhere.** README, docstrings, comments, CLI help text, notifications -- all English.
- **`pathlib.Path` always.** Never `os.path`.
- **Modern type hints.** `str | None`, `list[Path]` -- not `Optional`, `List`.
- **No single-char variables** except `_` in unpacking.
- **Lazy imports.** Only `argparse` and `sys` at module level. Everything else inside functions.
- **Errors are visible.** `notify()` always prints to stderr AND tries desktop notification. Never silently swallow errors.

## Architecture decisions

- Docker images have CPU instruction compatibility issues. `main` crashes on Meteor Lake (AMX), `main-vulkan` crashes on Kaby Lake (SIGILL). The config allows overriding `image` per machine. See `DOCKER_IMAGES` dict and comments.
- `notify()` uses `--replace-id` with a fixed ID so each notification replaces the previous one. Default timeout is 0 (stays until replaced). Only success/error messages get timeouts.
- `xclip` must be called with `stdout=DEVNULL, stderr=DEVNULL` (not `capture_output=True`) because it forks a background process that inherits pipes and causes timeout.
- `create_container` auto-downloads the model if missing, to avoid Docker crash loops.

## Testing

- Tests use `pytest` with `unittest.mock` for subprocess/network calls. No real Docker or network in tests.
- Test classes group related tests (`class TestDetectBackend:`, `class TestNotify:`).
- All `create_container` tests must mock `pull_image` too, or they'll attempt real Docker pulls.

If you find a wrong assumption in this file during a session, suggest the correction.
