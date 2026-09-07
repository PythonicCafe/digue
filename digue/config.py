"""Configuration loading, validation, and the config command."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from digue import (
    AVAILABLE_MODELS,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_RECORD_SECONDS,
    DEFAULT_MODELS,
    DEFAULT_PORT,
    DEFAULT_TRANSCRIPTION_TIMEOUT,
)


def _config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "digue" / "config.toml"


def _default_config() -> dict[str, dict[str, Any]]:
    xdg = os.environ.get("XDG_DATA_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    data_dir = base / "digue"

    return {
        "server": {
            "port": DEFAULT_PORT,
            "data_dir": str(data_dir),
            "backend": "auto",
            "image": "",
            "bind_ip": "127.0.0.1",
            "remote_host": "",
            "container_name": "digue-whisper.cpp",
        },
        "transcribe": {
            "language": DEFAULT_LANGUAGE,
            "prompt": "",
            "output_format": "text",
            "max_line_length": 42,
            "max_lines": 2,
            "timeout": DEFAULT_TRANSCRIPTION_TIMEOUT,
        },
        "dictate": {
            "audio_dir": "",
            "display_server": "auto",
            "input_mode": "paste",
            "recorder": "auto",
            "device": "",
            "max_duration": DEFAULT_MAX_RECORD_SECONDS,
            "save_audio": True,
            "audio_format": "flac",
        },
        "models": dict(DEFAULT_MODELS),
    }


def _merge_section(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Merges source keys into target dict; kebab-case keys override snake_case defaults."""
    for key, value in source.items():
        target[key.replace("-", "_")] = value


def _host_overrides(user_config: dict[str, Any]) -> dict[str, Any]:
    """Returns the [host.<this-hostname>] tables from the user config, or {}.

    Matched with or without the domain part, in both directions: the exact hostname first, then the hostname without
    its domain as an exact table name, then a table name whose own first label equals the short hostname (a dotfile
    written with the FQDN, a machine whose gethostname() returns the short name). Two tables matching only by short
    name is ambiguous and rejected. gethostname() is in-memory (microseconds), so calling it on every run adds no
    startup delay.
    """
    hosts = user_config.get("host")
    if not isinstance(hosts, dict):
        return {}
    import socket

    hostname = socket.gethostname()
    short_hostname = hostname.split(".")[0]
    for candidate in (hostname, short_hostname):
        if candidate in hosts and isinstance(hosts[candidate], dict):
            override: dict[str, Any] = hosts[candidate]
            return override
    by_short_name = [
        name for name, table in hosts.items() if isinstance(table, dict) and name.split(".")[0] == short_hostname
    ]
    if len(by_short_name) > 1:
        listing = ", ".join(f'"{name}"' for name in by_short_name)
        raise ValueError(f'Hostname "{hostname}" is ambiguous between [host] tables {listing}; use the exact name')
    if by_short_name:
        short_override: dict[str, Any] = hosts[by_short_name[0]]
        return short_override
    return {}


def validate_container_name(name: str) -> None:
    """Docker container names: [a-zA-Z0-9][a-zA-Z0-9_.-]* (docker silently truncates anything else).

    Used by the config validation and by `server start -n`, which writes the name straight into the config and would
    otherwise reach docker inspect/create unvalidated.
    """
    import re

    if not name or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name) is None:
        raise ValueError(
            f"Invalid server.container_name: {name!r}; expected a non-empty Docker container name "
            "([a-zA-Z0-9][a-zA-Z0-9_.-]*)"
        )


def _validate_config(config: dict[str, dict[str, Any]]) -> None:
    """Validates the fully merged configuration before commands consume it."""

    def require_type(section: str, key: str, expected_type: type[Any]) -> Any:
        value = config[section][key]
        if not isinstance(value, expected_type) or expected_type is int and isinstance(value, bool):
            raise ValueError(f"Invalid {section}.{key}: expected {expected_type.__name__}, got {type(value).__name__}")
        return value

    def require_choice(section: str, key: str, choices: tuple[str, ...]) -> None:
        value = require_type(section, key, str)
        if value not in choices:
            options = ", ".join(choices)
            raise ValueError(f"Invalid {section}.{key}: {value!r}; expected one of: {options}")

    from digue.container import BACKENDS

    require_choice("server", "backend", ("auto", *BACKENDS))
    port = require_type("server", "port", int)
    if not 1 <= port <= 65535:
        raise ValueError(f"Invalid server.port: {port}; expected an integer from 1 to 65535")
    for key in ("data_dir", "image", "bind_ip", "remote_host", "container_name"):
        require_type("server", key, str)
    validate_container_name(str(config["server"]["container_name"]))

    for key in ("language", "prompt"):
        require_type("transcribe", key, str)
    require_choice("transcribe", "output_format", ("text", "vtt", "srt", "timestamps"))
    for key in ("max_line_length", "max_lines", "timeout"):
        value = require_type("transcribe", key, int)
        if value < 1:
            raise ValueError(f"Invalid transcribe.{key}: {value}; expected an integer greater than zero")

    require_type("dictate", "audio_dir", str)
    require_choice("dictate", "display_server", ("auto", "x11", "wayland"))
    require_choice("dictate", "input_mode", ("paste", "type"))
    require_choice("dictate", "recorder", ("auto", "pw-record", "arecord"))
    require_choice("dictate", "audio_format", ("wav", "flac", "opus"))
    require_type("dictate", "device", str)
    require_type("dictate", "save_audio", bool)
    max_duration = require_type("dictate", "max_duration", int)
    if max_duration < 0:
        raise ValueError(f"Invalid dictate.max_duration: {max_duration}; expected zero or greater")

    for backend, model in config["models"].items():
        if backend in DEFAULT_MODELS and (not isinstance(model, str) or model not in AVAILABLE_MODELS):
            options = ", ".join(AVAILABLE_MODELS)
            raise ValueError(f"Invalid models.{backend}: {model!r}; expected one of: {options}")


def _check_section_keys(section: str, table: dict[str, Any]) -> None:
    """Validates key names (and models values) of a raw config section.

    Accepts kebab-case and snake_case spellings (merge normalizes later) but rejects unknown keys, models backends or
    models outside AVAILABLE_MODELS, and keys that collide after normalizing hyphens.
    """
    defaults = _default_config()[section]
    seen: dict[str, str] = {}
    for key, value in table.items():
        canonical = key.replace("-", "_")
        if canonical in seen:
            raise ValueError(
                f'Conflicting keys in [{section}]: "{seen[canonical]}" and "{key}" '
                "differ only by kebab-case/snake_case spelling"
            )
        seen[canonical] = key
        if section == "models":
            if canonical not in DEFAULT_MODELS:
                backends = ", ".join(DEFAULT_MODELS)
                raise ValueError(f'Unknown models key "{key}"; valid backends: {backends}')
            if not isinstance(value, str) or value not in AVAILABLE_MODELS:
                options = ", ".join(AVAILABLE_MODELS)
                raise ValueError(f"Invalid models.{key}: {value!r}; expected one of: {options}")
        elif canonical not in defaults:
            import difflib

            # the file format is kebab-case (see CONFIG_TEMPLATE): suggest and list the names the user would actually
            # write
            valid_keys = [name.replace("_", "-") for name in defaults]
            matches = difflib.get_close_matches(canonical.replace("_", "-"), valid_keys, n=1)
            suggestion = f'; did you mean "{matches[0]}"?' if matches else ""
            raise ValueError(f'Unknown {section} key "{key}"{suggestion}; valid keys: {", ".join(valid_keys)}')


def _validate_host_config(hosts: dict[str, Any]) -> None:
    """Validates every [host.<hostname>] table, not just the current machine's.

    The config file is versioned in dotfiles and shared across machines, so a typo under another host must fail here
    too.
    """
    for hostname, sections in hosts.items():
        if not isinstance(sections, dict):
            raise ValueError(f"[host.{hostname}] must be a table of configuration sections")
        for name, value in sections.items():
            if name not in _default_config():
                example_section = "server"
                if isinstance(value, dict):
                    example_section = next((key for key in value if key in _default_config()), "server")
                raise ValueError(
                    f'Unknown host subsection "{name}"; if the hostname contains dots, '
                    f'quote it: [host."{hostname}.{name}".{example_section}]'
                )
            if not isinstance(value, dict):
                raise ValueError(f"Section [host.{hostname}.{name}] must be a table of key = value pairs")
            _check_section_keys(name, value)


def _validate_config_structure(user_config: dict[str, Any]) -> None:
    """Validates section and key names of the raw TOML, before any merging."""
    for name, value in user_config.items():
        if name == "host":
            if not isinstance(value, dict):
                raise ValueError("[host] must be a table of per-host configuration tables")
            _validate_host_config(value)
            continue
        if name not in _default_config():
            valid = ", ".join((*_default_config(), "host"))
            raise ValueError(f'Unknown section "{name}"; valid sections: {valid}')
        if not isinstance(value, dict):
            kind = type(value).__name__
            raise ValueError(f"Section [{name}] must be a table of key = value pairs, got {kind}")
        _check_section_keys(name, value)


def apply_cli_overrides(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> None:
    """Applies CLI options that override config values, with the same validation the TOML file gets.

    Every CLI-to-config override lives here so a value typed on the command line cannot bypass the checks the file
    goes through (the choices are also enforced by argparse, but -n historically was not, and "already running" hid it).
    An option absent from the parsed Namespace (tests with partial mocks, subcommands without the flag) keeps the
    loaded config.
    """
    candidates = {
        ("server", "container_name"): getattr(args, "container_name", None),
        ("server", "image"): getattr(args, "image", None),
        ("transcribe", "language"): getattr(args, "language", None),
        ("transcribe", "prompt"): getattr(args, "prompt", None),
        ("transcribe", "output_format"): getattr(args, "response_format", None),
    }
    overrides = {(section, key): value for (section, key), value in candidates.items() if isinstance(value, str)}
    merged = {section: dict(values) for section, values in config.items()}
    for (section, key), value in overrides.items():
        merged[section][key] = value
    _validate_config(merged)
    for (section, key), value in overrides.items():
        config[section][key] = value


def load_config(config_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Loads config from TOML file, falling back to defaults for missing keys.

    If the file defines [host.<hostname>][section] tables matching this machine (exact hostname, or the first dot
    component of it), those sections are merged on top of the global ones, which in turn override the defaults.
    """
    import tomllib

    config = _default_config()
    path = Path(config_path) if config_path else _config_path()

    if path.exists():
        with path.open("rb") as fobj:
            user_config: dict[str, Any] = tomllib.load(fobj)
        _validate_config_structure(user_config)
        for section, defaults in config.items():
            if section in user_config:
                _merge_section(config[section], user_config[section])
        host_overrides = _host_overrides(user_config)
        for section, defaults in config.items():
            if section in host_overrides and isinstance(host_overrides[section], dict):
                _merge_section(config[section], host_overrides[section])

    _validate_config(config)

    if not config["dictate"]["audio_dir"]:
        config["dictate"]["audio_dir"] = str(Path(config["server"]["data_dir"]) / "audio")

    # Expand ~ in path values
    for key in ("data_dir",):
        config["server"][key] = str(Path(config["server"][key]).expanduser())
    for key in ("audio_dir",):
        config["dictate"][key] = str(Path(config["dictate"][key]).expanduser())

    return config


def model_for_backend(backend: str, config: dict[str, dict[str, Any]] | None = None) -> str:
    """Returns the model name for a given backend, respecting config overrides."""
    if config and "models" in config:
        models: dict[str, Any] = config["models"]
        return str(models.get(backend, DEFAULT_MODELS.get(backend, "small")))
    return DEFAULT_MODELS.get(backend, "small")


CONFIG_TEMPLATE = """\
# Only the sections and keys documented below are accepted (each key in kebab-case or snake_case, not both spellings at
# once); anything else -- including inside [host.<hostname>] tables -- is rejected when the config is loaded.
# `digue config show` prints the resolved settings for this machine.

[server]
# port = 8178                           # Host port for the whisper-server container
# bind-ip = "127.0.0.1"                 # IP Docker binds the port to; 127.0.0.1 = local only.
                                        #   Set to a LAN IP to expose it to that network (the server has no
                                        #   authentication; prefer SSH tunnels)
# data-dir = ""                         # Where models are stored (default: $XDG_DATA_HOME/digue -- XDG_DATA_HOME is
                                        #   often unset, in which case: ~/.local/share/digue)
# backend = "auto"                      # "auto" (detect GPU), "nvidia", "amd", "intel", "cpu", or "remote" (server on
                                        #   another machine via SSH tunnel)
# remote-host = ""                      # For backend = "remote": the server host (LAN IP, hostname, or empty =
                                        #   127.0.0.1 via SSH tunnel)
# image = ""                            # Docker image for whisper-server; empty = the backend's
                                        #   default (see README for the compatibility matrix), e.g.
                                        #   "ghcr.io/ggml-org/whisper.cpp:main" for CPUs where main-vulkan crashes.
                                        #   `digue server start` recreates a container created from another image
# container-name = "digue-whisper.cpp"  # Docker container name (`digue server start -n` overrides)

[transcribe] # Defaults for transcribe, batch-transcribe and dictate
# language = "auto"                     # Language for transcription: "auto", "pt", "en", etc.
# prompt = ""                           # Initial prompt to steer spelling of names/acronyms, e.g. "Turicas, Pythonic"
# output-format = "text"                # Transcribe output: "vtt", "srt", "timestamps" ([00:00:12] text lines) or
                                        #   "text" (plain)
# max-line-length = 42                  # Subtitle cue wrapping (vtt/srt): max chars per line
# max-lines = 2                         # Max lines per cue when wrapping
# timeout = 600                         # Seconds to wait for the server's answer (the server only replies after
                                        #   transcribing the whole file: long files on CPU need more)

[dictate]
# audio-dir = ""                        # Where recordings are saved (default: <data-dir>/audio/YYYY/MM)
# display-server = "auto"               # "auto" (detect), "x11", or "wayland"
# input-mode = "paste"                  # "paste" (clipboard + Ctrl+V) or "type" (simulate keystrokes; use "type" in
                                        #   terminals)
# save-audio = true                     # Save the recording as a backup
# audio-format = "flac"                 # Format of the saved recording: "flac" (lossless, ~35% of WAV; default),
                                        #   "opus" (~7%, lossy 24 kbit/s) or "wav".
                                        #   pw-record writes flac natively when libsndfile has the container; otherwise
                                        #  ffmpeg compresses a WAV (arecord always needs this)
# max-duration = 300                    # Stop recording after N seconds (0 = unlimited)
# recorder = "auto"                     # "auto" (pw-record or arecord), "pw-record", or "arecord"
# device = ""                           # Capture source; empty = system default.
                                        #   pw-record: --target NAME (node name or serial)
                                        #   arecord: -D NAME (PCM; arecord -l lists cards)

[models]
# nvidia = "large-v3-turbo-q8_0"
# amd = "large-v3-turbo-q8_0"
# intel = "large-v3-turbo-q8_0"
# cpu = "small-q8_0"
# Available models (multilingual): tiny, base, small, medium, large-v1, large-v2, large-v3, large-v3-turbo.
# Quantized copies (smaller, usually faster on CPU; q5 loses a little accuracy): tiny-q8_0, tiny-q5_1, base-q8_0,
# base-q5_1, small-q8_0, small-q5_1, medium-q8_0, medium-q5_0, large-v2-q8_0, large-v2-q5_0, large-v3-q5_0,
# large-v3-turbo-q8_0, large-v3-turbo-q5_0. English-only: tiny.en, base.en, small.en, medium.en and their -q8_0 / -q5_x
# copies. See section "Quantized models" on README or run `digue models`.

# Per-host overrides:
# [host.<hostname>][section] tables override the global sections of the same name on that machine only
# (defaults < global < host). The hostname matches exactly, or without the domain part on either side (thinkpad matches
# "thinkpad.local" and vice versa; two tables differing only by domain are rejected as ambiguous).
# Hostnames containing dots must be quoted, or TOML parses each dot as a nested table and the file is rejected:
# [host."minipc.local".server]
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
"""


def _config_example() -> str:
    """Returns the config template (all settings documented, defaults commented)."""
    return CONFIG_TEMPLATE


def _config_as_toml(config: dict[str, dict[str, Any]]) -> str:
    """Renders the resolved config as TOML (section by section, strings quoted).

    Keys use the kebab-case spelling of the template and README (data-dir), so the output can be pasted back into
    config.toml as documented.
    """
    lines = []
    for section, values in config.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            elif isinstance(value, int):
                rendered = str(value)
            else:
                rendered = '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'
            lines.append(f"{key.replace('_', '-')} = {rendered}")
        lines.append("")
    return "\n".join(lines)


def cmd_config(args: argparse.Namespace, config: dict[str, dict[str, Any]]) -> int:
    import json

    if args.output_format == "toml":
        print(_config_as_toml(config), end="")
    else:
        print(json.dumps(config, default=str, ensure_ascii=False, indent=2))
    return 0


def _config_init(args: argparse.Namespace) -> int:
    """Creates the config file from the template. Refuses to overwrite without --force."""

    selected_path = args.output or vars(args).get("config")
    path = Path(selected_path).expanduser() if selected_path else _config_path()
    if path.exists() and not args.force:
        print(f"Error: config file already exists: {path}", file=sys.stderr)
        print("Use --force to overwrite it.", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_config_example())
    print(f"Config created: {path}", file=sys.stderr)
    if args.force:
        print("(existing file was overwritten)", file=sys.stderr)
    return 0
