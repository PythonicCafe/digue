"""Tests for configuration loading, validation, host overrides, and the config command."""

import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import digue

# -- Config -------------------------------------------------------------------


class TestDefaultConfig:
    def test_has_required_sections(self):
        config = digue._default_config()
        assert "server" in config
        assert "dictate" in config
        assert "models" in config

    def test_default_port(self):
        config = digue._default_config()
        assert config["server"]["port"] == 8178

    def test_default_models_include_nvidia(self):
        config = digue._default_config()
        assert "nvidia" in config["models"]
        assert config["models"]["nvidia"] == "large-v3-turbo"


class TestLoadConfig:
    def test_returns_defaults_when_no_file(self, tmp_path):
        config = digue.load_config(tmp_path / "nonexistent.toml")
        assert config["server"]["port"] == 8178

    def test_reads_toml_overrides(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent("""\
            [server]
            port = 9999

            [models]
            nvidia = "medium"
            cpu = "tiny"
        """)
        )
        config = digue.load_config(config_path)
        assert config["server"]["port"] == 9999
        assert config["models"]["nvidia"] == "medium"
        assert config["models"]["cpu"] == "tiny"
        assert config["models"]["amd"] == "large-v3-turbo"  # untouched

    def test_kebab_case_keys(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text('[server]\ndata-dir = "/opt/data"\n')
        config = digue.load_config(config_path)
        assert config["server"]["data_dir"] == "/opt/data"

    def test_main_reports_invalid_toml_without_traceback(self, tmp_path, capsys):
        config_path = tmp_path / "invalid.toml"
        config_path.write_text("[server\nport = 8178\n")

        with (
            patch.object(sys, "argv", ["digue", "--config", str(config_path), "config", "show"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            digue.main()

        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "Error: failed to load configuration" in err
        assert "Traceback" not in err

    def test_expands_tilde_in_paths(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent("""\
            [server]
            data-dir = "~/whisper/data"

            [dictate]
            audio-dir = "~/whisper-audio"
        """)
        )
        config = digue.load_config(config_path)
        assert "~" not in config["server"]["data_dir"]
        assert "~" not in config["dictate"]["audio_dir"]
        assert config["server"]["data_dir"].endswith("whisper/data")
        assert config["dictate"]["audio_dir"].endswith("whisper-audio")

    def test_custom_docker_image_is_allowed(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text('[server]\nimage = "registry.example/my-whisper:custom"\n')
        assert digue.load_config(config_path)["server"]["image"] == "registry.example/my-whisper:custom"


class TestModelForBackend:
    def test_returns_default_without_config(self):
        assert digue.model_for_backend("nvidia") == "large-v3-turbo"
        assert digue.model_for_backend("cpu") == "small"

    def test_respects_config_override(self):
        config = {"models": {"nvidia": "medium"}}
        assert digue.model_for_backend("nvidia", config) == "medium"

    def test_unknown_backend_falls_back_to_small(self):
        assert digue.model_for_backend("unknown") == "small"


class TestHostOverrides:
    def test_applies_matching_host(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent("""\
            [host.thinkpad.server]
            backend = "cpu"

            [host.thinkpad.dictate]
            max-duration = 42
        """)
        )
        with patch("socket.gethostname", return_value="thinkpad"):
            config = digue.load_config(config_path)
        assert config["server"]["backend"] == "cpu"
        assert config["dictate"]["max_duration"] == 42

    def test_matches_hostname_without_domain(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent("""\
            [host.laptop.server]
            backend = "remote"
        """)
        )
        with patch("socket.gethostname", return_value="laptop.company.com"):
            config = digue.load_config(config_path)
        assert config["server"]["backend"] == "remote"

    def test_ignores_other_hosts(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent("""\
            [host.desktop.server]
            backend = "remote"
        """)
        )
        with patch("socket.gethostname", return_value="thinkpad"):
            config = digue.load_config(config_path)
        assert config["server"]["backend"] == "auto"

    def test_host_overrides_beat_global(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent("""\
            [server]
            backend = "amd"

            [host.thinkpad.server]
            backend = "cpu"
        """)
        )
        with patch("socket.gethostname", return_value="thinkpad"):
            config = digue.load_config(config_path)
        assert config["server"]["backend"] == "cpu"

    def test_global_fills_what_host_does_not_override(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent("""\
            [server]
            backend = "amd"
            port = 9000

            [host.thinkpad.server]
            backend = "cpu"
        """)
        )
        with patch("socket.gethostname", return_value="thinkpad"):
            config = digue.load_config(config_path)
        assert config["server"]["backend"] == "cpu"
        assert config["server"]["port"] == 9000

    def test_host_models_section(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent("""\
            [models]
            cpu = "small"

            [host.thinkpad.models]
            cpu = "medium"
        """)
        )
        with patch("socket.gethostname", return_value="thinkpad"):
            config = digue.load_config(config_path)
        assert config["models"]["cpu"] == "medium"
        assert config["models"]["nvidia"] == "large-v3-turbo"

    def test_unknown_keys_in_host_section_are_rejected(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent("""\
            [host.thinkpad.server]
            no-such-key = true
        """)
        )
        with patch("socket.gethostname", return_value="thinkpad"), pytest.raises(ValueError, match="no-such-key"):
            digue.load_config(config_path)

    def test_gethostname_called_once_with_host_section(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent("""\
            [host.laptop.server]
            backend = "cpu"
        """)
        )
        with patch("socket.gethostname", return_value="laptop") as mock_hostname:
            digue.load_config(config_path)
        assert mock_hostname.call_count == 1


class TestConfigStructureValidation:
    """Section and key names are validated on the raw TOML, before merging,
    so a typo fails loudly even inside [host.<x>] tables that belong to other
    machines (the file is versioned in dotfiles and shared across them)."""

    def write_config(self, tmp_path, toml):
        config_path = tmp_path / "config.toml"
        config_path.write_text(textwrap.dedent(toml))
        return config_path

    def load_with_hostname(self, tmp_path, toml, hostname="thinkpad"):
        config_path = self.write_config(tmp_path, toml)
        with patch("socket.gethostname", return_value=hostname):
            return digue.load_config(config_path)

    def test_unknown_top_level_section_lists_valid_ones(self, tmp_path):
        config_path = self.write_config(tmp_path, '[serv]\nbackend = "cpu"\n')
        with pytest.raises(ValueError) as excinfo:
            digue.load_config(config_path)
        message = str(excinfo.value)
        assert '"serv"' in message
        for name in ("server", "transcribe", "dictate", "models", "host"):
            assert name in message

    def test_host_is_a_valid_top_level_section(self, tmp_path):
        config = self.load_with_hostname(tmp_path, '[host.desktop.server]\nbackend = "cpu"\n')
        assert config["server"]["backend"] == "auto"

    def test_section_value_must_be_a_table(self, tmp_path):
        config_path = self.write_config(tmp_path, 'server = "cpu"\n')
        with pytest.raises(ValueError, match=r"\[server\].*must be a table"):
            digue.load_config(config_path)

    def test_unknown_key_suggests_close_match(self, tmp_path):
        config_path = self.write_config(tmp_path, '[server]\nbackends = "cpu"\n')
        with pytest.raises(ValueError, match=r'Unknown server key "backends".*did you mean "backend"'):
            digue.load_config(config_path)

    def test_unknown_key_without_match_lists_valid_keys(self, tmp_path):
        config_path = self.write_config(tmp_path, "[server]\nxyzzy = 1\n")
        with pytest.raises(ValueError) as excinfo:
            digue.load_config(config_path)
        message = str(excinfo.value)
        assert '"xyzzy"' in message
        assert "port" in message

    def test_suggestions_and_valid_keys_use_the_kebab_case_of_the_file_format(self, tmp_path):
        """The file and `config init` template are kebab-case; suggesting the
        internal snake_case name (max_duration) would send the user to write a
        key that only works by accident."""
        config_path = self.write_config(tmp_path, "[dictate]\nmax-durations = 5\n")
        with pytest.raises(ValueError) as excinfo:
            digue.load_config(config_path)
        message = str(excinfo.value)
        assert 'did you mean "max-duration"' in message
        assert "audio-dir" in message and "save-audio" in message
        assert "max_duration" not in message and "audio_dir" not in message

    def test_snake_and_kebab_keys_are_both_accepted(self, tmp_path):
        config = self.load_with_hostname(
            tmp_path,
            """\
            [server]
            data_dir = "/opt/d1"
            bind-ip = "0.0.0.0"

            [dictate]
            max-duration = 42
            input_mode = "type"
        """,
        )
        assert config["server"]["data_dir"] == "/opt/d1"
        assert config["server"]["bind_ip"] == "0.0.0.0"
        assert config["dictate"]["max_duration"] == 42
        assert config["dictate"]["input_mode"] == "type"

    def test_kebab_and_snake_collision_after_normalization_is_rejected(self, tmp_path):
        config_path = self.write_config(tmp_path, '[server]\ndata-dir = "/a"\ndata_dir = "/b"\n')
        with pytest.raises(ValueError, match=r'Conflicting.*"data-dir".*"data_dir"'):
            digue.load_config(config_path)

    def test_models_rejects_unknown_backends(self, tmp_path):
        config_path = self.write_config(tmp_path, '[models]\nremote = "small"\n')
        with pytest.raises(ValueError, match=r'Unknown models key "remote"'):
            digue.load_config(config_path)

    def test_models_rejects_values_outside_available_models(self, tmp_path):
        config_path = self.write_config(tmp_path, '[models]\ncpu = "giant"\n')
        with pytest.raises(ValueError, match=r"Invalid models\.cpu.*giant"):
            digue.load_config(config_path)

    def test_models_accepts_valid_backends_and_values(self, tmp_path):
        config = self.load_with_hostname(tmp_path, '[models]\nnvidia = "tiny"\ncpu = "large-v3"\n')
        assert config["models"]["nvidia"] == "tiny"
        assert config["models"]["cpu"] == "large-v3"

    def test_host_keys_are_validated_for_other_hosts(self, tmp_path):
        with pytest.raises(ValueError, match=r'Unknown server key "no-such-key"'):
            self.load_with_hostname(tmp_path, "[host.desktop.server]\nno-such-key = true\n")

    def test_host_models_are_validated_for_other_hosts(self, tmp_path):
        with pytest.raises(ValueError, match=r"Invalid models\.cpu.*giant"):
            self.load_with_hostname(tmp_path, '[host.desktop.models]\ncpu = "giant"\n')

    def test_unquoted_dotted_hostname_hints_quoting_without_asserting_cause(self, tmp_path):
        """[host.thinkpad.local.server] unquoted parses as hostname "thinkpad"
        with subsection "local"; the message suggests the quoting fix but must
        not claim the hostname was the actual problem."""
        config_path = self.write_config(tmp_path, '[host.thinkpad.local.server]\nbackend = "cpu"\n')
        with pytest.raises(ValueError) as excinfo:
            digue.load_config(config_path)
        message = str(excinfo.value)
        assert 'Unknown host subsection "local"' in message
        assert "if the hostname contains dots, quote it" in message
        assert '[host."thinkpad.local".server]' in message

    def test_host_unknown_subsection_without_dots_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match=r'Unknown host subsection "serves"'):
            self.load_with_hostname(tmp_path, '[host.desktop.serves]\nbackend = "cpu"\n')

    def test_host_subsection_value_must_be_a_table(self, tmp_path):
        with pytest.raises(ValueError, match=r"host\.desktop\.server.*must be a table"):
            self.load_with_hostname(tmp_path, '[host.desktop]\nserver = "cpu"\n')

    def test_main_exits_1_without_traceback_on_invalid_config(self, tmp_path, capsys):
        config_path = self.write_config(tmp_path, '[serv]\nbackend = "cpu"\n')
        with (
            patch.object(sys, "argv", ["digue", "--config", str(config_path), "config", "show"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            digue.main()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "Error: failed to load configuration" in err
        assert "Traceback" not in err


class TestServerHost:
    @pytest.mark.parametrize(
        ("bind_ip", "expected"),
        (("192.0.2.10", "192.0.2.10"), ("0.0.0.0", "127.0.0.1")),
    )
    def test_local_server_host_matches_reachable_bind_address(self, bind_ip, expected):
        config = digue._default_config()
        config["server"]["bind_ip"] = bind_ip
        assert digue.server_host(config) == expected


class TestConfigValueValidation:
    """Resolved values (enums, ranges, types) are checked after merge.
    Unknown keys belong in TestConfigStructureValidation."""

    @pytest.mark.parametrize(
        ("toml", "message"),
        (
            ('[server]\nbackend = "invalid"\n', "server.backend"),
            ("[server]\nport = 0\n", "server.port"),
            ("[server]\nport = true\n", "server.port"),
            ("[dictate]\nmax-duration = -1\n", "dictate.max_duration"),
            ('[dictate]\nrecorder = "invalid"\n', "dictate.recorder"),
            ('[dictate]\ndisplay-server = "invalid"\n', "dictate.display_server"),
            ('[dictate]\ninput-mode = "invalid"\n', "dictate.input_mode"),
            ('[dictate]\naudio-format = "invalid"\n', "dictate.audio_format"),
            ('[dictate]\nsave-audio = "yes"\n', "dictate.save_audio"),
            ("[server]\ndata-dir = 42\n", "server.data_dir"),
            ("[dictate]\ndevice = 42\n", "dictate.device"),
            ('[transcribe]\noutput-format = "invalid"\n', "transcribe.output_format"),
        ),
    )
    def test_load_config_validates_resolved_values(self, tmp_path, toml, message):
        config_path = tmp_path / "config.toml"
        config_path.write_text(toml)
        with pytest.raises(ValueError, match=message):
            digue.load_config(config_path)


# -- CLI parser ---------------------------------------------------------------


class TestConfigCommand:
    def test_show_toml_includes_sections(self, capsys):
        config = digue._default_config()
        args = MagicMock()
        args.output_format = "toml"
        assert digue.cmd_config(args, config) == 0
        out = capsys.readouterr().out
        assert "[server]" in out
        assert "[dictate]" in out
        assert "port = 8178" in out
        assert 'backend = "auto"' in out

    def test_show_toml_uses_the_documented_kebab_case_keys(self, capsys):
        """The template and README spell keys as data-dir, max-duration...;
        config show printed data_dir, so its output did not match the format
        it documents. It must also be valid TOML that loads back unchanged."""
        import tomllib

        config = digue._default_config()
        args = MagicMock()
        args.output_format = "toml"
        digue.cmd_config(args, config)
        out = capsys.readouterr().out

        assert "data-dir = " in out
        assert "max-duration = " in out
        assert "output-format = " in out
        assert "_" not in "".join(line.split("=")[0] for line in out.splitlines() if "=" in line)
        parsed = tomllib.loads(out)
        assert {key.replace("-", "_"): value for key, value in parsed["dictate"].items()} == config["dictate"]

    def test_show_json_unchanged(self, capsys):
        config = digue._default_config()
        args = MagicMock()
        args.output_format = "json"
        assert digue.cmd_config(args, config) == 0
        output = json.loads(capsys.readouterr().out)
        assert output["server"]["port"] == 8178

    def test_show_resolves_host_overrides(self, tmp_path, capsys):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent("""\
            [host.thinkpad.server]
            backend = "cpu"
        """)
        )
        with patch("socket.gethostname", return_value="thinkpad"):
            config = digue.load_config(config_path)
        args = MagicMock()
        args.output_format = "toml"
        digue.cmd_config(args, config)
        assert 'backend = "cpu"' in capsys.readouterr().out

    def test_init_creates_file(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(digue, "_config_path", lambda: tmp_path / "digue" / "config.toml")
        args = MagicMock()
        args.force = False
        args.output = None
        assert digue._config_init(args) == 0
        created = tmp_path / "digue" / "config.toml"
        assert created.exists()
        assert "[server]" in created.read_text()
        assert "Config created" in capsys.readouterr().err

    def test_init_refuses_existing_without_force(self, tmp_path, capsys, monkeypatch):
        existing = tmp_path / "config.toml"
        existing.write_text("# my custom config")
        monkeypatch.setattr(digue, "_config_path", lambda: existing)
        args = MagicMock()
        args.force = False
        args.output = None
        assert digue._config_init(args) == 1
        assert existing.read_text() == "# my custom config"
        assert "already exists" in capsys.readouterr().err

    def test_init_force_overwrites(self, tmp_path, monkeypatch):
        existing = tmp_path / "config.toml"
        existing.write_text("# old")
        monkeypatch.setattr(digue, "_config_path", lambda: existing)
        args = MagicMock()
        args.force = True
        args.output = None
        assert digue._config_init(args) == 0
        assert "[server]" in existing.read_text()

    def test_init_output_path(self, tmp_path, capsys):
        target = tmp_path / "custom" / "digue.toml"
        args = MagicMock()
        args.force = False
        args.output = str(target)
        assert digue._config_init(args) == 0
        assert target.exists()
        assert "[server]" in target.read_text()

    def test_init_uses_args_config_path(self, tmp_path):
        target = tmp_path / "selected.toml"
        args = MagicMock(config=str(target), output=None, force=False)
        assert digue._config_init(args) == 0
        assert target.exists()

    def test_example_config_is_valid_toml(self, tmp_path):
        import tomllib

        example = digue._config_example()
        parsed = tomllib.loads(example)
        assert "models" in parsed  # sections exist; all keys stay commented


class TestCmdConfig:
    def test_prints_json(self, capsys):
        config = digue._default_config()
        result = digue.cmd_config(MagicMock(output_format="json"), config)
        assert result == 0
        output = json.loads(capsys.readouterr().out)
        assert output["server"]["port"] == 8178
        assert "nvidia" in output["models"]


class TestConfigTemplateSync:
    def test_readme_config_block_matches_config_init_template(self):
        """Regression: the README config example must stay in sync with `digue config init`."""
        readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
        section = readme.split("## Configuration", 1)[1]
        fence_start = section.index("```toml") + len("```toml\n")
        fence_end = section.index("```", fence_start)
        readme_block = section[fence_start:fence_end].strip()
        assert readme_block == digue._config_example().strip()
