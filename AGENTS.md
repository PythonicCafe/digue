# AGENTS.md

> Single-file Python CLI tool for local speech-to-text using whisper.cpp in Docker. Stdlib only, no pip dependencies.

## Commands

- Test: `pytest tests/ -v --tb=short` (or `make test`)
- Type check: `mypy` (strict, configured in pyproject; or `make mypy`)
- Lint: `ruff check . --fix && ruff format --line-length 120` (or `make lint`)
- Run: `python digue.py dictate` (toggle dictation), `python digue.py detect` etc.
- Smoke test after building a wheel: `make smoke-wheel`. It installs the wheel into a temporary virtual environment, with no runtime dependencies, and runs the installed `digue --version` and `digue config show` entrypoint.
- All of the above have `make` targets (`make help`); `make check` runs lint-check + mypy + test.

## Conventions

- **Stdlib only.** No external runtime dependencies. `tomllib` (3.11+), `urllib.request`, `subprocess`, `pathlib`.
- **Single file.** All logic lives in `digue.py`. Do not split into modules.
- **English everywhere.** README, docstrings, comments, CLI help text, notifications, commit messages -- all English.
- **`pathlib.Path` always.** Never `os.path`.
- **Modern type hints.** `str | None`, `list[Path]` -- not `Optional`, `List`.
- **No single-char variables** except `_` in unpacking.
- **Lazy imports.** Module level has only `argparse`, `collections.abc`, `contextlib`, `dataclasses`, `os`, `pathlib`, `sys` and `typing` (the test suite pins this list). Everything else is imported inside the function that uses it, and a function never re-imports a module-level name (`os`, `contextlib`, `Path` were re-imported 42 times before the test caught it).
- **Type hints everywhere; `mypy --strict` must stay clean.** Config dicts are `dict[str, dict[str, Any]]` (values are TOML-parsed scalars; `Any` beats `object` because `dict` is invariant and these values flow into `str`/`int` params).
- **Errors are visible.** `notify()` always prints to stderr AND tries desktop notification. Never silently swallow errors.
- **User-facing failures notify, never traceback.** A failure inside a dictation/transcription flow (save, transcribe, paste) reports via `notify(..., timeout_ms=...)` and returns exit code 1; raw tracebacks are for bugs only.

## Architecture decisions

- Native server formats: wav, flac, mp3, ogg/Vorbis, aiff (verified empirically against whisper-server built with WHISPER_COMMON_FFMPEG=OFF; opus-in-ogg, m4a/AAC, mp4, webm, mka, wma fail with HTTP 400). Unsupported formats are converted in memory (ffmpeg writes to stdout pipe, nothing hits the disk), upfront for known-bad extensions and as a retry after HTTP 400.
- Every request sends `token_timestamps=false`: without it, the server enables token timestamps for text format, which triggers whisper's max_len=60 segment wrapping on token boundaries -- this split words in half across lines ("trans"/"crevendo"). Verified: with the flag, output comes as natural segments.
- `backend = "remote"` means the server runs on another machine: via SSH tunnel (default, `remote-host` empty = 127.0.0.1) or directly on the LAN (`remote-host` set to a host/IP; `port` stays `server.port`). `digue` must never create, start, or stop a local container for it; all container management commands refuse. When the remote host is the default (tunnel), errors suggest the ssh command; with a LAN host, they show host:port instead.
- Dictation runs as a daemon: the first `digue dictate` starts the recorder and stays alive waiting (200ms poll); the second toggle signals SIGTERM and exits instantly (the daemon stops the recorder and delivers: transcribe -> paste -> txt+flac). In a terminal, Ctrl+c does the same (the daemon installs its own SIGINT handler -- the global KeyboardInterrupt handler would discard the take and leave the recorder running), and stderr announces "Press Ctrl+c to stop recording and transcribe". The daemon enforces the duration limit itself and replaces the "Recording..." popup with "Limit reached (<seconds>s), transcribing..." by id (per-take id = base + pid % 32, so overlapping takes never replace or close each other's popups). The detached watchdog is only a safety killer (kill, no notify, no re-run) for a SIGKILLed daemon, and it sleeps `WATCHDOG_GRACE_SECONDS` past the limit: with an equal deadline it won the race against the 200ms poll (measured at 20s) and the daemon saw "died" instead of "limit"; the daemon also reports any recorder exit at or past the limit as "limit". After the recorder stops, the SIGINT/SIGTERM handlers stay installed on purpose: a second Ctrl+c or a `pkill digue` during delivery is ignored (the flags are only read while recording), so a take is never dropped mid-delivery; only SIGKILL aborts it (documented in the README, Recording). Overlapping takes are supported: while a daemon delivers (daemon pid file state "delivering"), a new toggle starts a fresh take; each daemon stops only its own recorder (`stop_recording_pid`, capture the rec_file while the recorder is alive). A recorder alive with no daemon at all (pkill digue) is recovered by the next toggle (stop + deliver what kept recording). The daemon file stores `<pid> <state> <starttime>`: a pid is only an identity together with its /proc starttime (the file outlives a SIGKILLed daemon and pids get recycled), so a toggle never signals a pid whose starttime differs; a file without starttime is treated as absent. State files (daemon, recorder pid) are published with `_write_state_file` (temp sibling + rename): `Path.write_text` truncates first, and a toggle reading that empty window would see "no daemon" and stop the live daemon's recorder. pw-record has no `--duration` flag (checked man page); arecord has no limit at all.
- Saved dictation audio is compressed with `audio-format` (default `flac`): flac is lossless and natively decodable by whisper-server, so retranscription never needs the ffmpeg fallback; opus is ~7% of WAV but lossy and forces the fallback. If compression fails (or ffmpeg is missing), keep the WAV and warn -- never leave a partial compressed file in its place. Delivery comes first: transcription and paste/type run on the live WAV, and the archiving (copy + ffmpeg compression, the slow part) runs only after the text was delivered; if anything fails after the recording stopped, the raw WAV is rescued (moved) to `<audio_dir>/YYYY/MM/<timestamp>.wav` so no take is lost -- `save-audio = false` does not change this: it only skips the backup of a delivered take.
- Default data dir is `$XDG_DATA_HOME/digue` (`~/.local/share/digue`); XDG_DATA_HOME is often unset, so the fallback to `~/.local/share` matters -- do not assume it is set.
- Config is split by role: `[transcribe]` is shared by transcribe, batch-transcribe and dictate for every applicable transcription-time option (language, prompt, output-format, cue wrapping); CLI options override it. `[dictate]` holds capture/delivery options (audio-dir, recorder, input-mode, ...). No option lives in the wrong section.
- Per-machine config lives in `[host.<hostname>][section]` tables (defaults < global < host, hostname matched with or without its domain part); one config.toml can be versioned in dotfiles for all machines. `gethostname()` is in-memory (~4us), safe to call on every run.
- Docker images have CPU instruction compatibility issues. `main` crashes on Meteor Lake (AMX), `main-vulkan` crashes on Kaby Lake (SIGILL). The config allows overriding `image` per machine. See `DOCKER_IMAGES` dict and comments.
- Notifications have a lifecycle contract: each process uses slot `NOTIFY_REPLACE_ID + pid % NOTIFY_ID_SLOTS`, and every progress notification must eventually be replaced or closed without affecting overlapping takes. Successful dictation ends with `notify_close()` (path goes to stderr instead of a transient "Pasted" notification); `main()` calls `notify_close()` on `KeyboardInterrupt`. Failures replace progress with a timed error (5-10s). Only a `kill -9` can leave one stuck; the README manual fix closes all slots.
- `notify()` uses `--replace-id` with the process slot so each take replaces only its own notification. Default timeout is 0 (stays until replaced). Only success/error messages get timeouts.
- `xclip` must be called with `stdout=DEVNULL, stderr=DEVNULL` (not `capture_output=True`) because it forks a background process that inherits pipes and causes timeout.
- Terminal progress output must be TTY-aware: `_stderr_is_tty()` gates everything. On a TTY, progress redraws one line with `\r` (bar + notification share the line via redraw). On captured/piped stderr, `\r` does nothing visually -- each print would become a huge line in logs -- so progress prints sparse plain lines (one per ~5%) with no bar, and `notify()` prints plain lines with `\n`. Never assume stderr is a TTY; every progress/notification print must handle both paths.
- `create_container` auto-downloads the model if missing, to avoid Docker crash loops.
- `doctor` never pulls Docker images; compatibility tests run only for images already present locally.
- Publishing: version has a single source of truth, `digue.__version__` (bump it before building). `make build` clears `dist/` first; before uploading, run `make build-check`, which also smoke-tests the installed wheel (see Commands).

## Testing

- Tests use `pytest` with `unittest.mock` for subprocess/network calls. No real Docker or network in tests.
- Test classes group related tests (`class TestDetectBackend:`, `class TestNotify:`).
- All `create_container` tests must mock `download_model` and `pull_image` too, or they'll attempt real network downloads or Docker pulls.
- Tests that mock `save_audio` must set `return_value=("path", "timestamp")` -- callers unpack the tuple.

If you find a wrong assumption in this file during a session, suggest the correction.
