# digue

Local speech-to-text dictation and transcription using [whisper.cpp](https://github.com/ggml-org/whisper.cpp). A single Python file with zero pip dependencies.

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

Not all Docker images work on all CPUs. Use `digue doctor` to test which images work on your machine.

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


## System requirements

- GNU/Linux only. Tested on Debian trixie. Should work on Ubuntu 22.04+, Fedora 38+, Arch. Not compatible with macOS or Windows.
- Python 3.11+ (for `tomllib`). No pip packages needed at runtime.
- Docker is required for running the whisper-server container (`apt install docker.io && usermod -aG docker $USER`, then log out and back in)
- For NVIDIA GPUs, also install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- For audio recording (dictation only), PipeWire is needed (`apt install pipewire`)
- For desktop notifications (dictation only), `notify-send` is needed (`apt install libnotify-bin`)
- For clipboard and paste (dictation only), `xclip` and `xdotool` on X11 or `wl-clipboard` and `wtype` on Wayland.
  - `digue` auto-detects X11 or Wayland via `$DISPLAY` / `$WAYLAND_DISPLAY`. You can force it with `display-server` in the config.
- For GPU detection (optional): `apt install pciutils vulkan-tools mesa-vulkan-drivers`

## Installation

Install from PyPI:

```bash
pip install digue
```

Or manually clone and symlink:

```bash
mkdir -p ~/software/
git clone https://github.com/turicas/digue.git ~/software/digue
cd ~/software/digue/

# Detect backend, download model, pull Docker image, test server
python3 digue.py doctor
python3 digue.py download
python3 digue.py start
python3 digue.py stop

# Install to PATH (symlink)
mkdir -p ~/.local/bin/
ln -s "~software/digue/digue.py" ~/.local/bin/digue
```

### First run

The Docker image (~1-3 GB) and model (~0.5-1.6 GB) are downloaded automatically on first use. This can take several minutes. The download progress is shown both in the terminal and as a desktop notification.


## Keybinding setup

### i3 / sway

Add to `~/.config/i3/config` or `~/.config/sway/config`:

```
bindsym $mod+Shift+d exec --no-startup-id $HOME/.local/bin/digue
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
    command "$HOME/.local/bin/digue"
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/digue/ \
    binding '<Super><Shift>d'
```

Or via Settings -> Keyboard -> Custom Shortcuts.

### KDE Plasma

Settings -> Shortcuts -> Custom Shortcuts -> Edit -> New -> Global Shortcut -> Command/URL:

- Trigger: `Super+Shift+D`
- Action: `/home/YOUR_USER/.local/bin/digue`

### XFCE

Settings -> Keyboard -> Application Shortcuts -> Add:

- Command: `/home/YOUR_USER/.local/bin/digue`
- Shortcut: `Super+Shift+D`

### Unity

System Settings -> Keyboard -> Shortcuts -> Custom Shortcuts -> `+`:

- Name: `Digue dictation`
- Command: `/home/YOUR_USER/.local/bin/digue`
- Click "Disabled", press `Super+Shift+D`


## CLI usage

```bash
# Dictation (default when no command given)
digue                                # toggle recording/transcription
digue dictate

# Server management
digue detect                         # print detected backend
digue download                       # download model for detected backend
digue download small                 # download a specific model
digue start                          # start (or create) server container
digue stop                           # stop server container
digue destroy                        # stop and remove container
digue status                         # show server status

# File transcription
digue transcribe audio.mp3                       # text to stdout
digue transcribe interview.mp4 -f vtt -o out.vtt # VTT from video
digue transcribe audio.mp3 -f srt -o out.srt     # SRT to file
digue transcribe audio.mp3 -l pt                 # force language

# VTT simplification (no server needed)
digue simplify-vtt input.vtt                      # to stdout
digue simplify-vtt input.vtt -o output.txt        # to file
cat input.vtt | digue simplify-vtt -              # from stdin

# Batch operations
digue batch-transcribe ./audios ./transcriptions              # all media -> VTT
digue batch-transcribe ./audios ./transcriptions -f text      # all media -> text
digue batch-simplify-vtt ./transcriptions ./simplified        # all VTT -> text

# Diagnostics
digue config                         # show current config as JSON
digue doctor                         # check dependencies, test Docker images
digue benchmark                      # compare backends (records from mic)
digue benchmark audio.wav            # benchmark with existing audio
```


## Configuration

All settings have sensible defaults. The config file is optional.

Create `~/.config/digue/config.toml` (or `$XDG_CONFIG_HOME/digue/config.toml`):

```toml
# Server
[server]
port = 8178                     # host port for the whisper-server container
# data-dir = "~/digue/data"     # where models are stored (default: ./data next to digue.py)
# backend = "auto"              # "auto" (detect GPU), "nvidia", "amd", "intel", or "cpu"
# image = ""                    # override Docker image; leave empty for auto-selection
                                #   NVIDIA: ghcr.io/ggml-org/whisper.cpp:main-cuda
                                #   AMD/Intel: ghcr.io/ggml-org/whisper.cpp:main-vulkan
                                #   CPU fallback: ghcr.io/ggml-org/whisper.cpp:main-vulkan (no GPU device)
                                #   Older Intel CPUs (Kaby Lake etc.): ghcr.io/ggml-org/whisper.cpp:main

# Dictation
[dictation]
language = "auto"               # language for transcription: "auto", "pt", "en" etc.
# audio-dir = ""                # where recordings are saved (default: <data-dir>/../audio)
# display-server = "auto"       # "auto" (detect), "x11", or "wayland"
                                #   X11 uses: xclip + xdotool
                                #   Wayland uses: wl-copy + wtype

# Models per backend
[models]                        # available: tiny, base, small, medium, large-v3-turbo, large-v3
nvidia = "large-v3-turbo"       # best quality, fast on dedicated GPU
amd = "large-v3-turbo"          # good on AMD iGPUs with Vulkan
intel = "large-v3-turbo"        # use "small" or "medium" on weaker Intel iGPUs
cpu = "small"                   # lighter model for CPU-only machines
```

Paths support `~` (expanded to home directory).


## Remote access via SSH tunnel

The server binds to `127.0.0.1` and is not exposed to the network. To use a remote machine's server (e.g. offloading from a laptop to a desktop):

```bash
ssh -NfL 8178:127.0.0.1:8178 user@desktop
```

With the tunnel active, `digue` works normally on the client -- the default URL already points to `localhost:8178`.

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

Run `digue doctor` to check dependencies, test Docker images, and verify config.

Common issues:

- **Server crashes in a loop (exit 132)**: the Docker image uses CPU instructions your processor doesn't support (SIGILL). Override the image in your config. See "Docker image compatibility" above.
- **"AMX is not ready to be used!"**: the `main` image fails on Meteor Lake CPUs inside Docker. Use `main-vulkan` (the default for `cpu` backend) or the `intel` backend.
- **`xclip` times out**: `xclip` forks a background process that inherits pipes. This is handled internally. If it still fails, check that `$DISPLAY` is set (run from a graphical terminal, not SSH).
- **No desktop notifications**: install `libnotify-bin`. All messages also print to stderr.

## Tests

```bash
pip install pytest pytest-cov ruff
pytest tests/ -v --cov=digue --cov-report=term-missing
ruff check . --fix && ruff format --line-length 120
```

## Audio storage

Every dictation is saved as a timestamped `.wav` + `.txt` pair in the audio directory (default: `<data-dir>/../audio/`). These are kept as backup and not cleaned up automatically.

## License

[MIT](LICENSE)
