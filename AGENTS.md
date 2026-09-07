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

- Native server formats: wav, flac, mp3, ogg/Vorbis, aiff (verified empirically against whisper-server built with WHISPER_COMMON_FFMPEG=OFF; opus-in-ogg, m4a/AAC, mp4, webm, mka, wma fail with HTTP 400). Unsupported formats are converted in memory (ffmpeg writes to stdout pipe, nothing hits the disk), upfront for known-bad extensions and as a retry after HTTP 400.
- Every request sends `token_timestamps=false`: without it, the server enables token timestamps for text format, which triggers whisper's max_len=60 segment wrapping on token boundaries -- this split words in half across lines ("trans"/"crevendo"). Verified: with the flag, output comes as natural segments.
- `backend = "remote"` means the server runs on another machine behind an SSH tunnel. `digue` must never create, start, or stop a local container for it; all container management commands refuse and point to the tunnel.
- Dictation recording runs in its own session (`start_new_session=True`) and is stopped via `os.killpg`, so a killed `digue` never leaves an unbounded `pw-record` behind. The `max-duration` limit is enforced by a detached watchdog process (`sh -c 'sleep N; kill -TERM -PGID; notify-send ...'`), not by a thread: the watchdog must survive digue being killed and notify the user when it fires. pw-record has no `--duration` flag (checked man page); arecord has no limit at all.
- Default data dir is `$XDG_DATA_HOME/digue` (`~/.local/share/digue`); XDG_DATA_HOME is often unset, so the fallback to `~/.local/share` matters -- do not assume it is set.
- Docker images have CPU instruction compatibility issues. `main` crashes on Meteor Lake (AMX), `main-vulkan` crashes on Kaby Lake (SIGILL). The config allows overriding `image` per machine. See `DOCKER_IMAGES` dict and comments.
- `notify()` uses `--replace-id` with a fixed ID so each notification replaces the previous one. Default timeout is 0 (stays until replaced). Only success/error messages get timeouts.
- `xclip` must be called with `stdout=DEVNULL, stderr=DEVNULL` (not `capture_output=True`) because it forks a background process that inherits pipes and causes timeout.
- `create_container` auto-downloads the model if missing, to avoid Docker crash loops.

## Testing

- Tests use `pytest` with `unittest.mock` for subprocess/network calls. No real Docker or network in tests.
- Test classes group related tests (`class TestDetectBackend:`, `class TestNotify:`).
- All `create_container` tests must mock `pull_image` too, or they'll attempt real Docker pulls.

If you find a wrong assumption in this file during a session, suggest the correction.
