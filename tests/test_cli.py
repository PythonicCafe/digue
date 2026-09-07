"""Tests for the CLI parser, module-level imports, and argparse helpers."""

import argparse
import ast
import textwrap
from pathlib import Path

import pytest

import digue


class TestCreateParser:
    def test_all_subcommands_parse(self):
        parser = digue.create_parser()
        for cmd in ("detect", "download", "dictate", "config", "benchmark"):
            args = parser.parse_args([cmd])
            assert args.command == cmd
        for action in ("start", "stop", "destroy", "status"):
            args = parser.parse_args(["server", action])
            assert args.command == "server" and args.server_action == action

    def test_version_flag(self, capsys):
        parser = digue.create_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--version"])
        assert excinfo.value.code == 0
        assert digue.__version__ in capsys.readouterr().out

    def test_convert_subcommand(self):
        parser = digue.create_parser()
        args = parser.parse_args(["convert", "a.vtt", "b.txt"])
        assert args.command == "convert"
        assert args.input == "a.vtt"
        assert args.output == "b.txt"
        assert args.from_format is None
        assert args.to_format is None

    @pytest.mark.parametrize("output_format", ("vtt", "srt"))
    def test_convert_to_format_accepts_subtitle_formats(self, output_format):
        parser = digue.create_parser()
        args = parser.parse_args(["convert", "a.txt", "--to-format", output_format])
        assert args.to_format == output_format

    def test_convert_to_format_help_lists_all_real_formats(self, capsys):
        parser = digue.create_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["convert", "--help"])
        assert excinfo.value.code == 0
        assert "vtt, srt, timestamps, text" in capsys.readouterr().out

    def test_transcribe_subcommand(self, tmp_path):
        parser = digue.create_parser()
        args = parser.parse_args(["transcribe", "test.wav", "-f", "vtt", "-o", "out.vtt"])
        assert args.command == "transcribe"
        assert args.response_format == "vtt"

    def test_custom_config_file_applies_to_commands(self, tmp_path):
        # -c/--config is a global option (before the subcommand) and must be
        # honored by every command that reads config.
        config_path = tmp_path / "custom.toml"
        config_path.write_text(
            textwrap.dedent("""\
            [server]
            port = 9999

            [transcribe]
            language = "it"
        """)
        )
        config = digue.load_config(config_path)
        assert config["server"]["port"] == 9999
        assert config["transcribe"]["language"] == "it"

        # argparse-level: the global option precedes the subcommand
        parser = digue.create_parser()
        args = parser.parse_args(["-c", str(config_path), "transcribe", "audio.wav"])
        assert args.config == str(config_path)

    def test_custom_config_before_subcommand_only(self, tmp_path):
        # After the subcommand, -c belongs to the subcommand (argparse default)
        parser = digue.create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["transcribe", "-c", "/tmp/x.toml", "audio.wav"])

    def test_bare_digue_exits_without_running_anything(self, capsys, monkeypatch):
        # No default command: a bare `digue` (wrong keybinding, typo) must show
        # help instead of toggling recording out of nowhere.
        monkeypatch.setattr("sys.argv", ["digue"])
        with pytest.raises(SystemExit) as excinfo:
            digue.main()
        assert excinfo.value.code == 1
        assert "usage" in capsys.readouterr().out.lower()

    def test_bare_config_shows_its_help_instead_of_assuming_an_action(self, capsys, monkeypatch, tmp_path):
        # `digue config` used to dump JSON (while `config show` defaults to
        # TOML); with no action it must show the config subcommand help.
        monkeypatch.setattr("sys.argv", ["digue", "-c", str(tmp_path / "none.toml"), "config"])
        with pytest.raises(SystemExit) as excinfo:
            digue.main()
        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "usage: digue config" in out
        assert "show" in out and "init" in out
        assert '"server"' not in out


class TestModuleImports:
    def _top_level_imports(self, tree):
        names = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.update(f"{node.module}.{alias.name}" for alias in node.names)
        return names

    def test_functions_do_not_reimport_module_level_names(self):
        """AGENTS.md: lazy imports live inside functions, and only what the
        module level does not already provide. os, contextlib and Path are
        module-level, so their 42 local re-imports were dead weight."""
        tree = ast.parse(Path(digue.__file__).read_text())
        top_level = self._top_level_imports(tree)
        duplicated = []
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef):
                continue
            for node in ast.walk(function):
                if isinstance(node, ast.Import):
                    duplicated.extend(
                        f"{function.name}:{alias.name}" for alias in node.names if alias.name in top_level
                    )
                elif isinstance(node, ast.ImportFrom):
                    duplicated.extend(
                        f"{function.name}:{node.module}.{alias.name}"
                        for alias in node.names
                        if f"{node.module}.{alias.name}" in top_level
                    )
        assert duplicated == []

    def test_module_level_imports_match_agents_md(self):
        tree = ast.parse(Path(digue.__file__).read_text())
        modules = {name.split(".")[0] for name in self._top_level_imports(tree)} - {"__future__"}
        assert modules == {"argparse", "collections", "contextlib", "dataclasses", "os", "pathlib", "sys", "typing"}


class TestExistingDir:
    def test_valid_dir(self, tmp_path):
        result = digue._existing_dir(str(tmp_path))
        assert result == tmp_path

    def test_invalid_dir(self):
        with pytest.raises(argparse.ArgumentTypeError, match="not found"):
            digue._existing_dir("/nonexistent/path")


class TestFormatExtension:
    def test_all_formats(self):
        assert digue._format_extension("text") == ".txt"
        assert digue._format_extension("vtt") == ".vtt"
        assert digue._format_extension("srt") == ".srt"
        assert digue._format_extension("timestamps") == ".txt"
