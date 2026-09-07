# digue

Local speech-to-text dictation and transcription using [whisper.cpp](https://github.com/ggml-org/whisper.cpp). A stdlib-only Python package with zero pip dependencies.

Press a keybinding to start recording, press again to stop. The transcribed text is pasted into the focused window. Also works as a CLI for transcribing audio/video files and simplifying VTT subtitles.

## How it works

`digue` manages a whisper-server Docker container with automatic GPU detection. On first run it detects your hardware, downloads the model, pulls the right Docker image, and creates the container. Subsequent runs just start/stop it.

Supported backends:

| Backend | Docker image | Acceleration | Detection |
|---|---|---|---|
| `nvidia` | `main-cuda` | CUDA | `nvidia-smi` responds |
| `amd` | `main-vulkan` | Vulkan (RADV) | `/dev/kfd` exists |
| `intel` | `main-vulkan` | Vulkan (ANV) | Intel iGPU in `lspci` (Skylake+) |
| `cpu` | `main-vulkan` (no GPU device) | CPU fallback | everything else |


## Docker image compatibility

Not all Docker images work on all CPUs. `digue doctor` tests compatibility for images that are already present locally; it does not pull images. Pull an image with `docker pull IMAGE` first if you want it included in the test.

| CPU generation | `main` | `main-vulkan` | `main-cuda` |
|---|---|---|---|
| Kaby Lake (7th gen, 2016) | OK | SIGILL (exit 132) | n/a |
| Meteor Lake (Core Ultra, 2024) | AMX crash | OK | n/a |
| AMD Ryzen 7000+ | untested | OK | n/a |
| NVIDIA GPU (any CPU) | varies | varies | OK |

If the default image crashes on your CPU, override it in the config:

```toml
[server]
backend = "cpu"
image = "ghcr.io/ggml-org/whisper.cpp:main"
```


## Performance

On a Ryzen 7 8745HS with Radeon 780M (`amd` backend, Vulkan/RADV), GPU transcription of the whisper.cpp JFK sample is 5-11x faster than CPU. `large-v3-turbo` and `medium` both take about 1 s on this GPU; `small` is quicker (~0.4 s) but the transcript has no punctuation or capitalization. The default (`large-v3-turbo` on `amd`) is the right choice for this machine: same cost as `medium`, larger model.

Times are wall-clock, three runs after a warm-up, `main-vulkan` image:

| Model | GPU (`amd`) | CPU | GPU speedup |
|---|---|---|---|
| `small` | 0.42 s | 2.22 s | 5.3x |
| `medium` | 1.01 s | 6.96 s | 6.9x |
| `large-v3-turbo` | 0.99 s | 11.02 s | 11.1x |

One machine, not a ranking of AMD iGPUs. Re-run with `digue benchmark --sample -b amd cpu -m small medium large-v3-turbo`.


## System requirements

- GNU/Linux only. Tested on Debian trixie. Should work on Ubuntu 22.04+, Fedora 38+, Arch. Not compatible with macOS or Windows.
- Python 3.11+ (for `tomllib`). No pip packages needed at runtime.
- Docker is required for running the whisper-server container (`apt install docker.io && usermod -aG docker $USER`, then log out and back in)
- For NVIDIA GPUs, also install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- For audio recording (dictation only), PipeWire is needed (`apt install pipewire`) or ALSA (`apt install alsa-utils`) as a fallback
- For desktop notifications (dictation only), `notify-send` is needed (`apt install libnotify-bin`)
- For clipboard and paste (dictation only), `xclip` and `xdotool` on X11 or `wl-clipboard` and `wtype` on Wayland (`apt install xclip xdotool` / `apt install wl-clipboard wtype`). See "Text output" for the `input-mode = "type"` alternative.
  - `digue` auto-detects X11 or Wayland via `$DISPLAY` / `$WAYLAND_DISPLAY`. You can force it with `display-server` in the config.
- For GPU detection (optional): `apt install pciutils vulkan-tools mesa-vulkan-drivers`
- For audio formats the server cannot decode (optional): `apt install ffmpeg`. Natively supported: wav, flac, mp3, ogg/Vorbis, aiff (see "Audio formats").

## Installation

`digue` is on PyPI. The recommended way to install a command-line application is `pipx`, which puts it in an isolated virtual environment and exposes the `digue` command on your `PATH` without touching the Python managed by your distribution:

```bash
pipx install digue
```

On Debian/Ubuntu, `pipx` itself comes from the distribution (`sudo apt install pipx`). `pipx` places its launchers in `~/.local/bin`; if that directory is already on your `PATH` (it is on most desktop setups), nothing else is needed. If `digue` is not found after installing, either:

1. run `pipx ensurepath` once -- note that it edits your shell configuration files (`~/.bashrc`, `~/.zshrc`, ...) to add `~/.local/bin` to `PATH`; or
2. add `~/.local/bin` to `PATH` yourself, in whatever way your shell is configured.

Point your window manager keybinding at `~/.local/bin/digue` directly; no `bash -c` or activation script is needed. The launcher is not a standalone executable: it still uses the Python interpreter and environment managed by `pipx`.

Plain `pip install digue` also works inside a virtual environment you manage yourself. Avoid `sudo pip install` and `pip install --user` on modern Debian/Ubuntu: PEP 668 marks the distribution Python as externally managed, so pip refuses them, and bypassing that protection (`--break-system-packages`) can break system tools.

### Without pipx or pip

`digue` has no runtime dependencies beyond Python 3.11+, so the package directory can simply be copied somewhere and run with `python3 -m digue`. Clone the repository into a temporary directory, copy the `digue/` package to `~/.local/opt/`, and create a small launcher in `~/.local/bin/`:

```bash
git clone --depth 1 https://github.com/turicas/digue.git /tmp/digue
mkdir -p ~/.local/opt ~/.local/bin
cp -r /tmp/digue/digue ~/.local/opt/digue
cat > ~/.local/bin/digue <<'EOF'
#!/bin/sh
PYTHONPATH="$HOME/.local/opt${PYTHONPATH:+:$PYTHONPATH}" exec python3 -m digue "$@"
EOF
chmod +x ~/.local/bin/digue
rm -rf /tmp/digue
```

The launcher must use `python3 -m digue` (with the parent directory on `PYTHONPATH`): running `python3 ~/.local/opt/digue/cli.py` directly does not work, because the package uses absolute `digue.*` imports and Python puts the script's own directory, not its parent, on `sys.path`. To update, repeat the clone and the `cp` (remove the old `~/.local/opt/digue` first). The same `PATH` note as above applies to `~/.local/bin`.

To work on the code instead, clone it anywhere and run `python3 -m digue ...` from the checkout; `pipx install -e .` exposes that checkout as the `digue` command.

### First run

The Docker image (~1-3 GB) and model (~0.5-1.6 GB) are downloaded automatically on first use. This can take several minutes. The download progress is shown both in the terminal and as a desktop notification.


## Keybinding setup

### i3 / sway

Add to `~/.config/i3/config` or `~/.config/sway/config`:

```
bindsym $mod+Shift+d exec --no-startup-id $HOME/.local/bin/digue dictate
```

Note: `~` does not work in i3/sway config. Use `$HOME` or the full path.

### GNOME

```bash
# Create the shortcut
gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings \
    "['/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/digue/']"
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/digue/ \
    name 'Digue dictation'
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/digue/ \
    command "$HOME/.local/bin/digue dictate"
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/digue/ \
    binding '<Super><Shift>d'
```

Or via Settings -> Keyboard -> Custom Shortcuts.

### KDE Plasma

Settings -> Shortcuts -> Custom Shortcuts -> Edit -> New -> Global Shortcut -> Command/URL:

- Trigger: `Super+Shift+D`
- Action: `/home/YOUR_USER/.local/bin/digue dictate`

### XFCE

Settings -> Keyboard -> Application Shortcuts -> Add:

- Command: `/home/YOUR_USER/.local/bin/digue dictate`
- Shortcut: `Super+Shift+D`

### Unity

System Settings -> Keyboard -> Shortcuts -> Custom Shortcuts -> `+`:

- Name: `Digue dictation`
- Command: `/home/YOUR_USER/.local/bin/digue dictate`
- Click "Disabled", press `Super+Shift+D`


## CLI usage

```bash
# Global options
digue --version                      # print the digue version
digue -c ./config.toml status        # use a custom config file

# Dictation
digue dictate                        # toggle recording/transcription
digue dictate -p "KINAI, Turicas"    # override the shared transcription prompt

# Server management
digue detect                         # print detected backend
digue detect-language audio.mp3      # print detected language code
digue detect-language audio.mp3 --json  # code, probability, and all probabilities
digue detect-language audio.mp3 -v   # show conversion progress on stderr
digue download                       # download model for detected backend
digue download small                 # download a specific model
digue server start                   # start (or create) server container
digue server stop                    # stop server container
digue server destroy                 # stop and remove container
digue server status                  # show server status

# Dictation storage
digue clean                          # list recordings/transcripts, ask, remove all
digue clean -f                       # remove without asking
digue clean -w recordings            # remove only recordings (also: transcripts, both)

# File transcription
digue transcribe audio.mp3                       # text to stdout (silent)
digue transcribe audio.mp3 -v                    # show progress messages
digue transcribe interview.mp4 -f vtt -o out.vtt # VTT from video
digue transcribe audio.mp3 -f srt -o out.srt     # SRT to file
digue transcribe audio.mp3 -f timestamps -o out.txt  # [00:00:12] text lines
digue transcribe audio.mp3 -l pt                 # force language
digue transcribe audio.mp3 -p "KINAI, Turicas"   # hint names/acronyms

# Format conversion (no server needed)
digue convert a.vtt b.txt                        # VTT -> timestamps (.txt is inferred as timestamps)
digue convert a.vtt b.txt -t text                # VTT -> plain text
digue convert a.vtt                              # VTT -> plain text on stdout
digue convert -f vtt - b.txt                     # VTT on stdin -> timestamps in b.txt
digue convert -f vtt -                           # VTT on stdin -> plain text on stdout
digue convert -f vtt - -t timestamps -           # VTT on stdin -> timestamps on stdout
digue convert a.srt b.vtt                        # SRT -> VTT
# Formats accepted by -f/--from-format and -t/--to-format: vtt, srt, timestamps, text

# Batch operations
digue batch-transcribe ./audios ./transcriptions              # use shared output-format (text by default)
digue batch-transcribe ./audios ./transcriptions -f vtt       # override with VTT (also: srt, text)
digue batch-simplify-vtt ./transcriptions ./simplified        # all VTT -> text

# Diagnostics
digue config                         # show current config as JSON
digue config show                    # resolved config as TOML (default)
digue config show -f json            # resolved config as JSON
digue config init                    # create the config file with commented defaults
digue config init -f                 # overwrite the config file
digue config init -o path            # create the config file at a custom path
digue -c path config init            # create the config at the global -c path
digue doctor                         # check dependencies, test locally present images
digue benchmark                      # quick: resolved backend + cpu, small + large-v3-turbo (records 10s from mic)
digue benchmark audio.wav            # same, with an existing audio file
digue benchmark --sample             # same, with the whisper.cpp JFK sample (downloaded once)
digue benchmark --sample -m all      # every model on the default backends
digue benchmark -b amd cpu -m medium -n 5 --json   # pick backends/models/runs; JSON results on stdout
```


## Configuration

All settings have sensible defaults. The config file is optional; `digue config init` writes it with every setting documented (defaults commented out).

Unknown sections, keys, or `[models]` backends are rejected when the config is loaded -- including inside `[host.<hostname>]` tables and for every host, not just the current machine, since the file is versioned in dotfiles. If you are upgrading from an earlier version, a config file that relied on misspelled or otherwise ignored keys will now fail with an error naming the offending entry (with a suggestion when the spelling is close to a valid key).

Create `~/.config/digue/config.toml` (or `$XDG_CONFIG_HOME/digue/config.toml`):

```toml
# Only the sections and keys documented below are accepted (each key in
# kebab-case or snake_case, not both spellings at once); anything else --
# including inside [host.<hostname>] tables -- is rejected when the config
# is loaded. `digue config show` prints the resolved settings for this machine.

# -- Server -------------------------------------------------------------------
[server]
# port = 8178                   # host port for the whisper-server container
# bind-ip = "127.0.0.1"         # IP Docker binds the port to; 127.0.0.1 = local only.
                                #   Set to a LAN IP to expose it to that network
                                #   (the server has no authentication; prefer SSH tunnels)
# data-dir = ""                 # where models are stored
                                #   (default: $XDG_DATA_HOME/digue -- XDG_DATA_HOME is often
                                #   unset, in which case: ~/.local/share/digue)
# backend = "auto"              # "auto" (detect GPU), "nvidia", "amd", "intel", "cpu",
                                # or "remote" (server on another machine via SSH tunnel)
# remote-host = ""              # for backend = "remote": the server host (LAN IP,
                                #   hostname, or empty = 127.0.0.1 via SSH tunnel)
# image = ""                    # override Docker image (see README for compatibility matrix)

# -- Transcription (defaults for transcribe, batch-transcribe and dictate) -----
[transcribe]
# language = "auto"             # language for transcription: "auto", "pt", "en", etc.
# prompt = ""                   # initial prompt to steer spelling of names/acronyms,
                                #   e.g. "KINAI, Turicas, Pythonic Café"
# output-format = "text"        # transcribe output: "vtt", "srt", "timestamps"
                                #   ([00:00:12] text lines) or "text" (plain)
# max-line-length = 42          # subtitle cue wrapping (vtt/srt): max chars per line
# max-lines = 2                 # max lines per cue when wrapping
# timeout = 600                 # seconds to wait for the server's answer (the server
                                #   only replies after transcribing the whole file:
                                #   long files on CPU need more)

# -- Dictation ----------------------------------------------------------------
[dictate]
# audio-dir = ""                # where recordings are saved (default: <data-dir>/audio/YYYY/MM)
# display-server = "auto"       # "auto" (detect), "x11", or "wayland"
# input-mode = "paste"          # "paste" (clipboard + Ctrl+V) or "type" (simulate
                                #   keystrokes; use "type" in terminals)
# save-audio = true             # save the recording as a backup
# audio-format = "flac"         # format of the saved recording: "flac" (lossless,
                                #   ~35% of WAV; default), "opus" (~7%, lossy 24 kbit/s)
                                #   or "wav". pw-record writes flac natively when
                                #   libsndfile has the container; otherwise ffmpeg
                                #   compresses a WAV (arecord always needs this)
# max-duration = 300            # stop recording after N seconds (0 = unlimited)
# recorder = "auto"             # "auto" (pw-record or arecord), "pw-record", or "arecord"
# device = ""                   # capture source; empty = system default.
                                #   pw-record: --target NAME (node name or serial)
                                #   arecord: -D NAME (PCM; arecord -l lists cards)

# -- Models per backend -------------------------------------------------------
[models]
# nvidia = "large-v3-turbo"
# amd = "large-v3-turbo"
# intel = "large-v3-turbo"
# cpu = "small"
# Available models: tiny, base, small, medium, large-v3-turbo, large-v3

# -- Per-host overrides (version this file in your dotfiles) -------------------
# [host.<hostname>][section] tables override the global sections of the same
# name on that machine only (defaults < global < host). The hostname matches
# exactly, or without the domain part on either side (thinkpad matches
# "thinkpad.local" and vice versa; two tables differing only by domain are
# rejected as ambiguous).
# Hostnames containing dots must be quoted, or TOML parses each dot as a
# nested table and the file is rejected: [host."minipc.local".server]
# Example:
#
# [host.minideb.server]
# backend = "amd"
#
# [host.thinkpad.server]
# backend = "cpu"
#
# [host.thinkpad.dictate]
# max-duration = 120
```


Paths support `~` (expanded to home directory).

## Shared and per-host configuration

`[transcribe]` is the real shared configuration for `transcribe`, `batch-transcribe`, and `dictate`: `language`, `prompt`, `output-format`, `max-line-length`, `max-lines`, and `timeout` are inherited by all applicable commands. CLI options override those values. `[dictate]` contains only capture and delivery settings.

One config file can drive all your machines: version it in your dotfiles and add a `[host.<hostname>][section]` table per machine. Inside the host table, use the same section names as the top level (`server`, `transcribe`, `dictate`, `models`); keys there override the global sections when the hostname matches, and global keys you did not override are still inherited. The hostname is read with `gethostname()` (an in-memory call, microseconds) - it does not delay the dictation hotkey. Run `digue config show` on each machine to confirm what was resolved. Hostnames containing dots must be quoted (`[host."minipc.local".server]`); unquoted, TOML parses each dot as a nested table and the file is rejected.

## Config command

```bash
digue config init            # create ~/.config/digue/config.toml with commented defaults
digue config init -f         # overwrite an existing config file
digue config init -o path    # create the config file at a custom path
digue config show            # resolved configuration (defaults + file + host overrides), as TOML
digue config show -f json    # same, as JSON
```


## Audio formats

Verified against `whisper-server` (the `ghcr.io/ggml-org/whisper.cpp` images decode with miniaudio and are built without its own ffmpeg fallback): natively supported formats are **wav, flac, mp3, ogg/Vorbis and aiff**.

Formats the server rejects (HTTP 400) are converted to 16 kHz mono WAV with **ffmpeg**, entirely in memory (the converted audio is never written to disk). This covers, among others: **ogg/Opus** (WhatsApp voice notes), **m4a/AAC**, mp4, webm, mka, wma, opus. Conversion happens either upfront (extension known to be unsupported) or as a retry after an HTTP 400. ffmpeg is optional:

```bash
sudo apt install ffmpeg   # optional, only needed for formats the server cannot decode
```

Without ffmpeg, unsupported formats produce a clear error instead of a raw HTTP 400.

## Recording

Dictation uses PipeWire's `pw-record` by default and falls back to ALSA's `arecord` when PipeWire is not available (`recorder = "auto"`). Set `device` to a capture source (`pw-record --target NAME`, `arecord -D NAME`); empty keeps the system default. `pw-record` has no `--list-targets` -- list sources with `pactl list sources short` or `wpctl status`; `arecord -l` lists ALSA cards (`digue doctor` prints both). Needed packages:

```bash
sudo apt install pipewire        # default recorder (pw-record)
sudo apt install alsa-utils      # fallback recorder (arecord)
```

The first `digue dictate` invocation stays alive as the recording daemon. Pressing the keybinding again sends it a stop signal and returns immediately; the original process stops the recorder, transcribes, delivers the text, and archives the take. When run in a terminal, Ctrl+c also stops and transcribes. There is no global recording state: each take is tracked by its own state file in `$XDG_RUNTIME_DIR` (`digue-take-<take_id>.json`), which follows the cycle `starting` (recording file reserved) -> `recording` (recorder pid and /proc starttime published) -> `recovering` (the daemon died and the next toggle claimed the take). The owner itself is tracked by the daemon file (`digue-daemon.pid`: `starting` -> `recording` -> `delivering`); while it says `delivering` the daemon ignores further Ctrl+c and SIGTERM, and a new take may start. Recovery: a recorder still alive is stopped through its published identity and its audio delivered; a dead recorder's WAV goes straight to delivery, unless its transcript was already saved (the daemon died while archiving), in which case only the audio is archived and nothing is pasted again; an empty WAV is discarded with its state (a take without audio has nothing left to recover). The toggle that recovers a take delivers it and returns without starting a new recording (press again to record); while it is delivering, another press starts a new take normally. A state file that cannot be parsed is reported as unreadable and left untouched (the WAV is kept). A take whose delivery failed in a retryable way keeps its state and WAV, and the next `digue dictate` retries it; only `kill -9` aborts a delivery, and then the raw WAV and the take state stay in `$XDG_RUNTIME_DIR` for recovery. Because each take carries its own identity, a recovery never signals another take's recorder. A new take may start while an earlier one is still being delivered.

The recorder runs in its own process group, so it can survive a killed daemon. The daemon normally enforces `max-duration` (default 300s, set `0` for unlimited), stops the recorder, and reports that the limit was reached. A detached watchdog is only a safety killer: if the daemon is killed abruptly, it stops the recorder a few seconds after the limit but does not notify or transcribe. The next `digue dictate` recovers and delivers an orphaned recording, and returns without starting a new one.

The recording is saved as a backup next to the `.txt` transcript, compressed with `audio-format` (default `flac`: lossless, ~35% of the WAV size; `opus`: ~7%, lossy 24 kbit/s; `wav`: no compression). When `audio-format = "flac"` and `pw-record --list-containers` lists `flac`, the live take is already FLAC (no ffmpeg). `arecord` cannot write FLAC, and a `pw-record` without libFLAC still records WAV then compresses with ffmpeg (host, then the local container); without ffmpeg digue keeps the WAV and warns. Set `save-audio = false` to keep only the transcript (a take that fails to transcribe or paste is still kept as WAV, since it was delivered nowhere). A saved `.flac` is decodable by whisper-server natively; a saved `.opus` goes through the ffmpeg fallback if you run `digue transcribe` on it.

If the recorder exits at start (unknown `--target`, missing PCM, missing binary), digue notifies and prints the recorder's own error so the device name can be fixed.

## Text output

The transcribed text is joined into a single line before being sent to the focused window. Line breaks in the server output are segment boundaries; with `token_timestamps=false` (sent by digue on every request) the server no longer wraps segments at 60 characters, which was splitting words in half (`trans` / `crevendo`).

How the text lands on screen is controlled by `input-mode` -- and the choice matters most in terminals:

- **Terminal (X11)**: the paste shortcut is `Ctrl+Shift+V`, not `Ctrl+V` -- so `input-mode = "paste"` **does not work in terminals** (the simulated Ctrl+V does nothing). Use `input-mode = "type"` if you dictate into a terminal; note typing is slower (~12 ms/char) and may drop characters in slow apps.
- **Regular GUI apps** (editors, browsers): both modes work; `paste` is the recommended default (instant, atomic).
- **Wayland**: `wtype -` types text read from stdin and `wtype -M ctrl v` simulates the shortcut; same terminal caveat applies to terminals on Wayland.

Rule of thumb: `paste` everywhere, except when the target window is a terminal -- then `type`.

```bash
sudo apt install xclip xdotool       # X11
sudo apt install wl-clipboard wtype  # Wayland
```

## Remote access via SSH tunnel

The server binds to `127.0.0.1` (see `bind-ip` in the config) and is not exposed to the network. To use a remote machine's server (e.g. offloading from a laptop to a desktop):

```bash
ssh -NfL 8178:127.0.0.1:8178 user@desktop
```

Then set `backend = "remote"` in the client's config:

```toml
[server]
backend = "remote"
```

With the tunnel active, `digue` works normally on the client. The `remote` backend also tells `digue` to never create, start, or stop a local container: `digue server start`, `server stop`, and `server destroy` refuse to run, `digue server status` only checks the port, and a failed transcription points you to the tunnel instead of suggesting `digue server start`. Without this setting, `digue` would try to spin up a local container if it could not reach the port.

To manage the container itself, run the commands (`digue download`, `server start`, `server destroy`) on the remote machine.

If the server is already reachable on your network (no tunnel needed), point `digue` straight at it:

```toml
[server]
backend = "remote"
remote-host = "desktop.lan"    # or a LAN IP, e.g. 10.0.0.5
# port = 8178                  # the port the remote server listens on
```

On the server machine, keep the container reachable from the network by setting `bind-ip` to a LAN IP (the default `127.0.0.1` only accepts local connections; the server has no authentication, so only do this on a network you trust). `digue config show` on the client tells you which host:port is being probed.

For a persistent tunnel, add to `~/.ssh/config`:

```
Host digue-remote
    HostName <desktop-ip>
    User <user>
    LocalForward 8178 127.0.0.1:8178
    ServerAliveInterval 30
    ExitOnForwardFailure yes
```

## Troubleshooting

Run `digue doctor` to check dependencies and config, and to test compatible Docker images that are already downloaded. Images reported as `SKIP not pulled` are not tested or downloaded.

Common issues:

- **Server crashes in a loop (exit 132)**: the Docker image uses CPU instructions your processor doesn't support (SIGILL). Override the image in your config. See "Docker image compatibility" above.
- **"AMX is not ready to be used!"**: the `main` image fails on Meteor Lake CPUs inside Docker. Use `main-vulkan` (the default for `cpu` backend) or the `intel` backend.
- **`xclip` times out**: `xclip` forks a background process that inherits pipes. This is handled internally. If it still fails, check that `$DISPLAY` is set (run from a graphical terminal, not SSH).
- **No desktop notifications**: install `libnotify-bin`. All messages also print to stderr.
- **Config syntax error**: `digue` fails with a `tomllib` parse error pointing at the line - fix `~/.config/digue/config.toml` (or run `digue config init -f` to start over).
- **"Recorder not found"**: install PipeWire (`apt install pipewire`, for `pw-record`) or ALSA (`apt install alsa-utils`, for `arecord`).
- **A dictation notification got stuck**: concurrent dictations use 32 notification slots, from ID 48271 through 48302, based on the daemon PID. Successful delivery replaces the progress popup with a 3s Pasted/Typed toast; errors replace it with a notification that expires in 5-10s. `kill -9` may leave one behind. Click it, or close all digue slots with:

```bash
for id in $(seq 48271 48302); do
    gdbus call --session \
        --dest org.freedesktop.Notifications \
        --object-path /org/freedesktop/Notifications \
        --method org.freedesktop.Notifications.CloseNotification "$id" >/dev/null
done
```

## Python API

The CLI is the main interface; the same module works as a library (no daemon, no paste):

```python
import digue

config = digue.load_config()  # default path, or load_config("path/to/config.toml")
text = digue.transcribe_file("meeting.wav")  # starts the server if needed
digue.record_to("take.flac", seconds=8, config=config)
```

`transcribe_file` uses `[transcribe]` (language, prompt, output-format). `record_to` uses `[dictate]` `recorder` / `device`, and writes FLAC natively when the path ends in `.flac` and `pw-record` supports that container; otherwise (arecord, or a `pw-record` without the flac container) it records WAV under the same stem, says so on stderr, and returns the `.wav` path -- always use the returned path. A recorder that exits at start raises `RuntimeError` with its stderr.

## Tests

```bash
pip install pytest pytest-cov ruff mypy
make test            # or: pytest tests/ -v --cov=digue --cov-report=term-missing
make mypy            # mypy --strict over digue/
make lint            # ruff check --fix + format
make check           # lint-check + mypy + test in one go
```

## Publishing to PyPI

```bash
make build-check     # build sdist+wheel and validate with twine
make publish         # upload (requires credentials)
```

Bump `__version__` in `digue/__init__.py` before building (the package version comes from it).

## Audio storage

Every dictation is saved as a `<YYYYMMDD-HHMMSS>-<take_id>.txt` transcript plus the recording (compressed per `audio-format`, default `flac`) unless `save-audio = false`, under `<data-dir>/audio/YYYY/MM/` (one folder per month). The take id (16 hex chars) keeps overlapping takes that end in the same second from overwriting each other's files; older files without the id (plain `<YYYYMMDD-HHMMSS>.<ext>`) are still recognized. The timestamp has no colons, so filenames are shell-friendly to complete. These are kept as backup and not cleaned up automatically; `digue clean` lists what there is and removes it after confirmation (`-f` skips the confirmation, and empty month directories are removed too). Only files in the dictation layout are touched (`YYYY/MM/<timestamp>[-<take_id>].wav|flac|opus|txt`), so anything else living under `audio-dir` is left alone.

## License

[MIT](LICENSE)
