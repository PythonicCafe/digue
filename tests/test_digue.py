"""Tests for digue.py."""

import argparse
import ast
import math
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmark_models
import digue


class TestBenchmarkTempFiles:
    def test_recorded_benchmark_audio_lives_in_the_private_runtime_dir(self, tmp_path):
        """A fixed name in /tmp could be a symlink planted by another local
        user; the runtime dir is 0700 and owned by the user."""
        config = digue._default_config()
        config["server"]["backend"] = "cpu"
        args = argparse.Namespace(audio=None)

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.record_benchmark_audio") as mock_record,
            patch("digue.run_benchmark"),
        ):
            assert digue.cmd_benchmark(args, config) == 0

        assert mock_record.call_args[0][0].parent == tmp_path

    def test_benchmark_models_sample_lives_in_the_private_runtime_dir(self, tmp_path):
        with patch("digue._runtime_dir", return_value=tmp_path):
            assert benchmark_models.sample_path().parent == tmp_path


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


class TestBenchmarkContainerState:
    def test_absent_container_is_only_cleaned_up(self):
        with (
            patch("digue.container_status", return_value=None),
            patch("digue.container_exists", return_value=True),
            patch("digue.remove_container") as mock_remove,
            patch("digue._rename_container") as mock_rename,
            patch("digue.stop_container") as mock_stop,
            patch("digue.start_container") as mock_start,
            digue.preserve_container_for_benchmark(),
        ):
            pass

        mock_remove.assert_called_once_with()
        mock_rename.assert_not_called()
        mock_stop.assert_not_called()
        mock_start.assert_not_called()

    def test_stopped_container_is_restored_stopped_after_exception(self):
        statuses = iter(["exited"])
        existence = iter([True])
        with (
            patch("digue.container_status", side_effect=lambda: next(statuses)),
            patch("digue.container_exists", side_effect=lambda: next(existence)),
            patch("digue.remove_container") as mock_remove,
            patch("digue._rename_container") as mock_rename,
            patch("digue.stop_container") as mock_stop,
            patch("digue.start_container") as mock_start,
            patch("digue.os.getpid", return_value=123),
            pytest.raises(KeyboardInterrupt),
            digue.preserve_container_for_benchmark(),
        ):
            raise KeyboardInterrupt

        assert mock_rename.call_args_list == [
            ((digue.CONTAINER_NAME, "digue-benchmark-backup-123"),),
            (("digue-benchmark-backup-123", digue.CONTAINER_NAME),),
        ]
        mock_remove.assert_called_once_with()
        mock_stop.assert_not_called()
        mock_start.assert_not_called()

    def test_running_container_is_stopped_then_restored_running(self):
        with (
            patch("digue.container_status", return_value="running"),
            patch("digue.container_exists", return_value=False),
            patch("digue._rename_container") as mock_rename,
            patch("digue.stop_container") as mock_stop,
            patch("digue.start_container") as mock_start,
            patch("digue.os.getpid", return_value=456),
            digue.preserve_container_for_benchmark(),
        ):
            pass

        mock_stop.assert_called_once_with()
        assert mock_rename.call_args_list == [
            ((digue.CONTAINER_NAME, "digue-benchmark-backup-456"),),
            (("digue-benchmark-backup-456", digue.CONTAINER_NAME),),
        ]
        mock_start.assert_called_once_with()


class TestRunBenchmarkLanguage:
    def test_language_default_comes_from_transcribe_section(self, tmp_path, capsys):
        config = digue._default_config()
        with (
            patch("digue.download_model"),
            patch("digue.preserve_container_for_benchmark"),
            patch("digue.container_exists", return_value=True),
            patch("digue.remove_container"),
            patch("digue.create_container"),
            patch("digue._wait_for_server", return_value=True),
            patch("digue._benchmark_run", return_value=[]),
            patch("digue.detect_backend", return_value="cpu"),
        ):
            digue.run_benchmark(tmp_path / "no-audio.wav", config)
        err = capsys.readouterr().err
        assert "digue benchmark" in err

    def test_removes_benchmark_container_when_transcription_is_interrupted(self, tmp_path):
        config = digue._default_config()
        with (
            patch("digue.download_model"),
            patch("digue.preserve_container_for_benchmark"),
            patch("digue.container_exists", return_value=True),
            patch("digue.remove_container") as mock_remove,
            patch("digue.create_container"),
            patch("digue._wait_for_server", return_value=True),
            patch("digue._benchmark_run", side_effect=KeyboardInterrupt),
            patch("digue.detect_backend", return_value="cpu"),
            pytest.raises(KeyboardInterrupt),
        ):
            digue.run_benchmark(tmp_path / "audio.wav", config)

        mock_remove.assert_called_once_with()


class TestBenchmarkRespectsConfig:
    def test_backend_override_is_honored(self, tmp_path):
        """A forced backend (e.g. cpu with image = main on a Kaby Lake iGPU)
        must not be bypassed by hardware detection: the GPU cases would run
        with an image the machine cannot execute."""
        config = digue._default_config()
        config["server"]["backend"] = "cpu"
        with (
            patch("digue.download_model"),
            patch("digue.preserve_container_for_benchmark"),
            patch("digue.container_exists", return_value=False),
            patch("digue.create_container") as mock_create,
            patch("digue._wait_for_server", return_value=True),
            patch("digue._benchmark_run", return_value=[]),
            patch("digue.detect_backend", return_value="intel"),
        ):
            digue.run_benchmark(tmp_path / "audio.wav", config)

        backends = [call.args[1] for call in mock_create.call_args_list]
        assert backends == ["cpu", "cpu"]

    def test_custom_image_applies_only_to_the_resolved_backend(self, tmp_path, capsys):
        """server.image is a single global override that only makes sense for
        the backend the config resolved to (e.g. image "main" pinned for a
        Kaby Lake CPU): other cases must fall back to DOCKER_IMAGES."""
        config = digue._default_config()
        config["server"]["backend"] = "amd"
        config["server"]["image"] = "x"
        with (
            patch("digue.download_model"),
            patch("digue.preserve_container_for_benchmark"),
            patch("digue.container_exists", return_value=False),
            patch("digue.create_container") as mock_create,
            patch("digue._wait_for_server", return_value=True),
            patch("digue._benchmark_run", return_value=[]),
            patch("digue.detect_backend", return_value="amd"),
        ):
            digue.run_benchmark(tmp_path / "audio.wav", config)

        images = {backend: [] for backend in ("cpu", "amd")}
        for create_call in mock_create.call_args_list:
            bench_config, backend = create_call.args
            images[backend].append(digue.resolve_image(backend, bench_config))
            assert bench_config["server"]["image"] == ("x" if backend == "amd" else "")
        assert images == {"cpu": [digue.DOCKER_IMAGES["cpu"]] * 2, "amd": ["x", "x"]}
        err = capsys.readouterr().err
        assert f"Image: {digue.DOCKER_IMAGES['cpu']}" in err
        assert "Image: x" in err

    @patch("time.sleep")
    @patch("subprocess.Popen")
    def test_microphone_recording_uses_the_configured_recorder(self, mock_popen, mock_sleep, tmp_path):
        config = digue._default_config()
        config["dictate"]["recorder"] = "arecord"

        digue.record_benchmark_audio(tmp_path / "bench.wav", config=config)

        assert mock_popen.call_args.args[0][0] == "arecord"


class TestBenchmarkModels:
    def test_case_removes_container_when_transcription_is_interrupted(self, tmp_path):
        config = digue._default_config()
        config["server"]["data_dir"] = str(tmp_path)
        with (
            patch("benchmark_models.digue.create_container"),
            patch("benchmark_models.digue._wait_for_server", return_value=True),
            patch("benchmark_models.digue.transcribe", side_effect=KeyboardInterrupt),
            patch("benchmark_models.digue.container_exists", return_value=True),
            patch("benchmark_models.digue.remove_container") as mock_remove,
            pytest.raises(KeyboardInterrupt),
        ):
            benchmark_models.benchmark_case(config, "cpu", "small")

        mock_remove.assert_called_once_with()

    def test_main_preserves_previous_container_on_interrupt(self):
        config = digue._default_config()
        manager = MagicMock()
        manager.__enter__.return_value = None
        manager.__exit__.return_value = False
        with (
            patch("benchmark_models.create_parser") as mock_parser,
            patch("benchmark_models.download_sample"),
            patch("benchmark_models.digue.load_config", return_value=config),
            patch("benchmark_models.digue.detect_backend", return_value="cpu"),
            patch("benchmark_models.digue.preserve_container_for_benchmark", return_value=manager),
            patch("benchmark_models.benchmark_case", side_effect=KeyboardInterrupt),
        ):
            mock_parser.return_value.parse_args.return_value = argparse.Namespace(
                backends=["cpu"], models=["small"], runs=1
            )
            benchmark_models.main()

        manager.__exit__.assert_called_once()


class TestBenchmarkModelsConfig:
    def test_main_rejects_remote_before_downloading_sample(self, capsys):
        config = digue._default_config()
        config["server"]["backend"] = "remote"
        with (
            patch("benchmark_models.create_parser") as mock_parser,
            patch("benchmark_models.download_sample") as mock_download,
            patch("benchmark_models.digue.load_config", return_value=config),
        ):
            mock_parser.return_value.parse_args.return_value = argparse.Namespace(
                backends=None, models=["small"], runs=1
            )
            result = benchmark_models.main()

        assert result == 1
        mock_download.assert_not_called()
        assert "remote" in capsys.readouterr().err

    def test_main_uses_resolved_backend_for_case_selection(self, capsys):
        """A forced backend in config wins over detection, exactly like
        run_benchmark: the auto list must follow resolve_backend."""
        config = digue._default_config()
        config["server"]["backend"] = "cpu"
        with (
            patch("benchmark_models.create_parser") as mock_parser,
            patch("benchmark_models.download_sample"),
            patch("benchmark_models.digue.load_config", return_value=config),
            patch("benchmark_models.digue.detect_backend", return_value="intel"),
            patch("benchmark_models.digue.preserve_container_for_benchmark"),
            patch("benchmark_models.benchmark_case", return_value=None),
        ):
            mock_parser.return_value.parse_args.return_value = argparse.Namespace(
                backends=None, models=["small"], runs=1
            )
            assert benchmark_models.main() == 0

        err = capsys.readouterr().err
        assert "Backends: cpu\n" in err
        assert "intel" not in err

    def test_case_clears_custom_image_for_other_backends(self, tmp_path):
        config = digue._default_config()
        config["server"]["backend"] = "cpu"
        config["server"]["image"] = "x"
        config["server"]["data_dir"] = str(tmp_path)
        with (
            patch("benchmark_models.digue.create_container") as mock_create,
            patch("benchmark_models.digue._wait_for_server", return_value=True),
            patch("benchmark_models.digue.transcribe", return_value="hello"),
            patch("benchmark_models.digue.container_exists", return_value=False),
        ):
            assert benchmark_models.benchmark_case(config, "intel", "small") is not None

        bench_config, backend = mock_create.call_args.args
        assert backend == "intel"
        assert bench_config["server"]["image"] == ""
        assert digue.resolve_image(backend, bench_config) == digue.DOCKER_IMAGES["intel"]

    def test_case_keeps_custom_image_for_resolved_backend(self, tmp_path):
        config = digue._default_config()
        config["server"]["backend"] = "cpu"
        config["server"]["image"] = "x"
        config["server"]["data_dir"] = str(tmp_path)
        with (
            patch("benchmark_models.digue.create_container") as mock_create,
            patch("benchmark_models.digue._wait_for_server", return_value=True),
            patch("benchmark_models.digue.transcribe", return_value="hello"),
            patch("benchmark_models.digue.container_exists", return_value=False),
        ):
            assert benchmark_models.benchmark_case(config, "cpu", "small") is not None

        bench_config, backend = mock_create.call_args.args
        assert backend == "cpu"
        assert bench_config["server"]["image"] == "x"

    def test_models_argument_rejects_unknown_model(self, capsys):
        with pytest.raises(SystemExit):
            benchmark_models.create_parser().parse_args(["--models", "giant"])

        assert "invalid choice" in capsys.readouterr().err

    def test_runs_rejects_non_positive_values(self):
        parser = benchmark_models.create_parser()
        for bad in ("0", "-1"):
            with pytest.raises(SystemExit):
                parser.parse_args(["--runs", bad])

        assert parser.parse_args(["--runs", "2"]).runs == 2
        assert parser.parse_args([]).runs == benchmark_models.RUNS


class TestCmdConvert:
    def _make_vtt(self, tmp_path):
        vtt_file = tmp_path / "a.vtt"
        vtt_file.write_text(
            "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nHello\n\n2\n00:00:01.000 --> 00:00:02.000\nWorld\n"
        )
        return vtt_file

    def test_vtt_to_txt_file(self, tmp_path, capsys):
        self._make_vtt(tmp_path)
        args = MagicMock()
        args.input = str(tmp_path / "a.vtt")
        args.output = str(tmp_path / "b.txt")
        args.from_format = None
        args.to_format = None
        config = digue._default_config()
        result = digue.cmd_convert(args, config)
        assert result == 0
        assert "Saved:" in capsys.readouterr().err
        content = (tmp_path / "b.txt").read_text()
        assert "[00:00:00] Hello" in content
        assert "[00:00:01] World" in content

    def test_vtt_to_stdout_with_to_format(self, tmp_path, capsys):
        self._make_vtt(tmp_path)
        args = MagicMock()
        args.input = str(tmp_path / "a.vtt")
        args.output = None
        args.from_format = None
        args.to_format = "text"
        config = digue._default_config()
        result = digue.cmd_convert(args, config)
        assert result == 0
        assert capsys.readouterr().out.strip() == "Hello World"

    def test_stdin_requires_from_format(self, capsys):
        args = MagicMock()
        args.input = "-"
        args.output = "b.txt"
        args.from_format = None
        args.to_format = None
        config = digue._default_config()
        result = digue.cmd_convert(args, config)
        assert result == 1
        assert "from-format is required" in capsys.readouterr().err

    def test_directory_input_gives_clear_error(self, tmp_path, capsys):
        args = MagicMock()
        args.input = str(tmp_path)
        args.output = None
        args.from_format = "vtt"
        args.to_format = "text"
        config = digue._default_config()
        result = digue.cmd_convert(args, config)
        assert result == 1
        err = capsys.readouterr().err
        assert "not a file" in err
        assert "Traceback" not in err

    def test_stdin_with_from_format(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr("sys.stdin", __import__("io").StringIO("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi\n"))
        args = MagicMock()
        args.input = "-"
        args.output = str(tmp_path / "b.txt")
        args.from_format = "vtt"
        args.to_format = None
        config = digue._default_config()
        result = digue.cmd_convert(args, config)
        assert result == 0
        assert "[00:00:00] Hi" in (tmp_path / "b.txt").read_text()

    def test_unknown_extension_requires_from_format(self, tmp_path, capsys):
        weird = tmp_path / "subtitles.xyz"
        weird.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi\n")
        args = MagicMock()
        args.input = str(weird)
        args.output = None
        args.from_format = None
        args.to_format = "text"
        config = digue._default_config()
        result = digue.cmd_convert(args, config)
        assert result == 1
        assert "from-format" in capsys.readouterr().err

    def test_txt_to_stdout_without_to_format_defaults_to_text(self, tmp_path, capsys):
        # .txt input (timestamps content), no -t, stdout: defaults to text
        source = tmp_path / "a.txt"
        source.write_text("[00:00:00] Hello\n[00:00:01] World\n")
        args = MagicMock()
        args.input = str(source)
        args.output = None
        args.from_format = None
        args.to_format = None
        config = digue._default_config()
        result = digue.cmd_convert(args, config)
        assert result == 0
        assert capsys.readouterr().out.strip() == "Hello World"

    def test_timestamps_to_vtt(self, tmp_path, capsys):
        source = tmp_path / "a.txt"
        source.write_text("[00:00:01] Hello\n[00:00:02] World\n")
        args = MagicMock()
        args.input = str(source)
        args.output = None
        args.from_format = None
        args.to_format = "vtt"
        config = digue._default_config()
        result = digue.cmd_convert(args, config)
        assert result == 0
        out = capsys.readouterr().out
        assert out.startswith("WEBVTT")
        assert "00:00:01.000 --> 00:00:02.000" in out
        assert "00:00:02.000 --> 00:00:04.000" in out
        assert "Hello" in out

    def test_multiline_vtt_cues_roundtrip_through_timestamps(self):
        vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nLinha 1\nLinha 2\n\n00:00:04.000 --> 00:00:06.000\nLinha 3\n"
        timestamps = digue._convert_content(vtt, "vtt", "timestamps")
        assert "[00:00:01] Linha 1 Linha 2" in timestamps
        result_vtt = digue._convert_content(timestamps, "timestamps", "vtt")
        assert "00:00:01.000 --> 00:00:04.000" in result_vtt
        assert "Linha 1 Linha 2" in result_vtt

    def test_srt_to_vtt_preserves_times_milliseconds_and_multiline_cues(self):
        content = (
            "1\n00:00:01,234 --> 00:00:03,456\nFirst line\nSecond line\n\n"
            "2\n01:02:03,007 --> 01:02:05,089\nAnother cue\n"
        )

        result = digue._convert_content(content, "srt", "vtt")

        assert result == (
            "WEBVTT\n\n"
            "00:00:01.234 --> 00:00:03.456\nFirst line\nSecond line\n\n"
            "01:02:03.007 --> 01:02:05.089\nAnother cue\n"
        )

    def test_vtt_to_srt_preserves_times_milliseconds_and_multiline_cues(self):
        content = (
            "WEBVTT\n\n"
            "intro\n00:00:00.125 --> 00:00:02.750 align:start\nHello\nworld\n\n"
            "00:01:03.004 --> 00:01:04.999\nLast cue\n"
        )

        result = digue._convert_content(content, "vtt", "srt")

        assert result == (
            "1\n00:00:00,125 --> 00:00:02,750\nHello\nworld\n\n2\n00:01:03,004 --> 00:01:04,999\nLast cue\n"
        )

    def test_timestamp_end_is_next_start_and_last_cue_has_two_second_duration(self):
        content = "[00:00:01] First\n[00:00:03] Last\n"

        result = digue._convert_content(content, "timestamps", "srt")

        assert "00:00:01,000 --> 00:00:03,000" in result
        assert "00:00:03,000 --> 00:00:05,000" in result

    def test_text_without_timestamps_cannot_become_vtt(self, tmp_path, capsys):
        source = tmp_path / "a.txt"
        source.write_text("just plain text\nanother line\n")
        args = MagicMock()
        args.input = str(source)
        args.output = None
        args.from_format = "text"
        args.to_format = "vtt"
        config = digue._default_config()
        result = digue.cmd_convert(args, config)
        assert result == 1
        assert "cannot convert" in capsys.readouterr().err

    def test_srt_roundtrip(self, tmp_path, capsys):
        srt_file = tmp_path / "a.srt"
        srt_file.write_text("1\n00:00:00,320 --> 00:00:01,000\nHello\n\n")
        args = MagicMock()
        args.input = str(srt_file)
        args.output = None
        args.from_format = None
        args.to_format = "text"
        config = digue._default_config()
        result = digue.cmd_convert(args, config)
        assert result == 0
        assert "Hello" in capsys.readouterr().out


class TestParseSubtitleTimestamp:
    def test_parses_three_part_timestamp(self):
        assert digue._parse_subtitle_timestamp("01:02:03.456") == 3723456
        assert digue._parse_subtitle_timestamp("01:02:03,456") == 3723456

    def test_parses_two_part_vtt_timestamp_without_hours(self):
        assert digue._parse_subtitle_timestamp("02:03.456") == 123456
        assert digue._parse_subtitle_timestamp("00:05.100") == 5100

    def test_invalid_timestamps_raise_value_error(self):
        with pytest.raises(ValueError, match="invalid subtitle timestamp"):
            digue._parse_subtitle_timestamp("invalid")
        with pytest.raises(ValueError, match="invalid subtitle timestamp"):
            digue._parse_subtitle_timestamp("00:60:00.000")


class TestCmdClean:
    def _make_audio_files(self, audio_dir):
        month = audio_dir / "2026" / "09"
        month.mkdir(parents=True)
        (month / "20260901-100000.flac").write_bytes(b"audio")
        (month / "20260901-100000.txt").write_text("transcript")
        (month / "20260902-110000.flac").write_bytes(b"audio")
        return month

    def _args(self, force=False, what="both"):
        args = MagicMock()
        args.force = force
        args.what = what
        return args

    def test_lists_and_asks_without_force(self, tmp_path, capsys, monkeypatch):
        audio_dir = tmp_path / "audio"
        self._make_audio_files(audio_dir)
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)
        monkeypatch.setattr("builtins.input", lambda prompt: "n")

        result = digue.cmd_clean(self._args(), config)

        assert result == 1
        err = capsys.readouterr().err
        assert "2 file(s)" in err  # recordings
        assert "1 file(s)" in err  # transcripts
        assert "Aborted" in err
        assert (audio_dir / "2026" / "09" / "20260901-100000.flac").exists()

    def test_removes_on_confirmation(self, tmp_path, capsys, monkeypatch):
        audio_dir = tmp_path / "audio"
        self._make_audio_files(audio_dir)
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)
        monkeypatch.setattr("builtins.input", lambda prompt: "y")

        result = digue.cmd_clean(self._args(), config)

        assert result == 0
        assert list(audio_dir.rglob("*.flac")) == []
        assert list(audio_dir.rglob("*.txt")) == []

    def test_force_removes_without_asking(self, tmp_path, capsys, monkeypatch):
        audio_dir = tmp_path / "audio"
        self._make_audio_files(audio_dir)
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        def fail_input(prompt):
            raise AssertionError("input() must not be called with --force")

        monkeypatch.setattr("builtins.input", fail_input)

        result = digue.cmd_clean(self._args(force=True), config)

        assert result == 0
        assert list(audio_dir.rglob("*.flac")) == []

    def test_what_recordings_keeps_transcripts(self, tmp_path, capsys):
        audio_dir = tmp_path / "audio"
        self._make_audio_files(audio_dir)
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = digue.cmd_clean(self._args(force=True, what="recordings"), config)

        assert result == 0
        assert list(audio_dir.rglob("*.flac")) == []
        assert list(audio_dir.rglob("*.txt"))

    def test_what_transcripts_keeps_recordings(self, tmp_path, capsys):
        audio_dir = tmp_path / "audio"
        self._make_audio_files(audio_dir)
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = digue.cmd_clean(self._args(force=True, what="transcripts"), config)

        assert result == 0
        assert list(audio_dir.rglob("*.txt")) == []
        assert list(audio_dir.rglob("*.flac"))

    def test_removes_empty_month_directories(self, tmp_path, capsys):
        audio_dir = tmp_path / "audio"
        self._make_audio_files(audio_dir)
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        digue.cmd_clean(self._args(force=True), config)

        assert not (audio_dir / "2026" / "09").exists()

    def test_only_removes_files_that_look_like_dictations(self, tmp_path, capsys):
        """audio-dir is user-configurable: pointed at ~/Music, an rglob over
        *.wav/*.flac/*.txt would wipe a music library. Only
        <audio_dir>/YYYY/MM/<timestamp>.{wav,flac,opus,txt} qualifies."""
        audio_dir = tmp_path / "audio"
        month = self._make_audio_files(audio_dir)
        foreign = [
            audio_dir / "song.flac",
            month / "2026-09-03T12:00:00.flac",  # not the now_timestamp() layout
            audio_dir / "notes.txt",
            month / "interview.wav",
            audio_dir / "2026" / "backup.txt",
            audio_dir / "albums" / "2026" / "09" / "20260901-100000.flac",
        ]
        foreign[-1].parent.mkdir(parents=True)
        for path in foreign:
            path.write_bytes(b"keep me")
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = digue.cmd_clean(self._args(force=True), config)

        assert result == 0
        assert all(path.exists() for path in foreign)
        assert not list(month.glob("2026090*"))
        assert "Removed 3 file(s)" in capsys.readouterr().err

    def test_regular_files_named_like_year_or_month_do_not_crash(self, tmp_path, capsys):
        audio_dir = tmp_path / "audio"
        self._make_audio_files(audio_dir)
        (audio_dir / "2025").write_text("a file, not a year directory")
        (audio_dir / "2026" / "08").write_text("a file, not a month directory")
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = digue.cmd_clean(self._args(force=True), config)

        assert result == 0
        assert "Removed 3 file(s)" in capsys.readouterr().err
        assert (audio_dir / "2025").exists()
        assert (audio_dir / "2026" / "08").exists()

    def test_confirmation_lists_the_files_it_will_remove(self, tmp_path, capsys, monkeypatch):
        audio_dir = tmp_path / "audio"
        self._make_audio_files(audio_dir)
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)
        monkeypatch.setattr("builtins.input", lambda prompt: "n")

        digue.cmd_clean(self._args(), config)

        err = capsys.readouterr().err
        assert "2026/09/20260901-100000.flac" in err
        assert "2026/09/20260901-100000.txt" in err
        assert "2026/09/20260902-110000.flac" in err

    def test_rescued_json_is_removed_with_its_recording_as_one_unit(self, tmp_path, capsys):
        """A rescued take's .json is metadata of the recording: it is removed
        together with the recording of the same stem and counted as one unit,
        never as its own category."""
        audio_dir = tmp_path / "audio"
        month = audio_dir / "2026" / "09"
        month.mkdir(parents=True)
        (month / "20260904-120000-0123456789abcdef.wav").write_bytes(b"audio")
        (month / "20260904-120000-0123456789abcdef.json").write_text('{"state": "rescued"}')
        (month / "20260905-130000-aaaaaaaaaaaaaaaa.wav").write_bytes(b"audio")
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = digue.cmd_clean(self._args(force=True), config)

        assert result == 0
        assert not list(month.glob("*0123456789abcdef*"))
        assert not list(month.glob("*aaaaaaaaaaaaaaaa*"))
        assert "Removed 2 file(s)" in capsys.readouterr().err

    def test_json_without_recording_is_preserved(self, tmp_path, capsys):
        """clean is not a general metadata collector: a .json whose recording
        is gone stays untouched."""
        audio_dir = tmp_path / "audio"
        month = audio_dir / "2026" / "09"
        month.mkdir(parents=True)
        json_path = month / "20260904-120000-0123456789abcdef.json"
        json_path.write_text('{"state": "rescued"}')
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = digue.cmd_clean(self._args(force=True), config)

        assert result == 0
        assert "Nothing to remove" in capsys.readouterr().err
        assert json_path.exists()

    def test_json_is_never_removed_as_transcript(self, tmp_path, capsys):
        audio_dir = tmp_path / "audio"
        month = audio_dir / "2026" / "09"
        month.mkdir(parents=True)
        json_path = month / "20260904-120000-0123456789abcdef.json"
        json_path.write_text('{"state": "rescued"}')
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = digue.cmd_clean(self._args(force=True, what="transcripts"), config)

        assert result == 0
        assert "Nothing to remove" in capsys.readouterr().err
        assert json_path.exists()

    def test_nothing_to_remove(self, tmp_path, capsys):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = digue.cmd_clean(self._args(), config)

        assert result == 0
        assert "Nothing to remove" in capsys.readouterr().err

    def test_missing_audio_dir(self, tmp_path, capsys):
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "nonexistent")

        result = digue.cmd_clean(self._args(), config)

        assert result == 0


class TestTimestampFormat:
    def test_format_has_no_colon_or_dash_in_date(self):
        """Filenames must be shell-friendly: YYYYMMDD-HHMMSS (no ':' to escape)."""
        timestamp = digue.now_timestamp()
        assert len(timestamp) == 15
        assert timestamp[8] == "-"
        assert ":" not in timestamp
        assert "-" not in timestamp[:8]

    def test_month_directory_uses_timestamp_not_wall_clock(self):
        """The YYYY/MM path comes from the timestamp, so audio and .txt land together."""
        assert digue.month_dir_for("20260904-123456") == Path("2026") / "09"


class TestSaveAudio:
    def test_copies_with_timestamp_in_month_directory(self, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"wav data")
        audio_dir = tmp_path / "audio"
        saved, timestamp = digue.save_audio(rec_file, audio_dir)
        assert saved.exists()
        # <audio_dir>/YYYY/MM/<timestamp>.wav
        assert saved.parent == audio_dir / timestamp[:4] / timestamp[4:6]
        assert timestamp in saved.name


class TestRescueRecording:
    def test_rescue_never_overwrites_an_existing_destination(self, tmp_path):
        """The temp file is exclusive, but publishing with os.replace would
        still clobber a destination that already exists; the rescue contract
        is never to overwrite another take's file."""
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"new audio")
        timestamp = "20260905-101500"
        existing = tmp_path / "audio" / digue.month_dir_for(timestamp) / f"{timestamp}-0123456789abcdef.wav"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"old audio")

        rescued = digue.rescue_recording(rec_file, tmp_path / "audio", timestamp, "0123456789abcdef")

        assert rescued is None
        assert existing.read_bytes() == b"old audio"
        assert rec_file.read_bytes() == b"new audio"
        assert sorted(path.name for path in existing.parent.iterdir()) == [existing.name]

    def test_moves_wav_with_take_id_and_only_then_removes_origin(self, tmp_path):
        rec_file = tmp_path / "digue-rec.wav"
        rec_file.write_bytes(b"audio")
        audio_dir = tmp_path / "audio"

        rescued = digue.rescue_recording(rec_file, audio_dir, "20260904-120000", "0123456789abcdef")

        archived = audio_dir / "2026" / "09" / "20260904-120000-0123456789abcdef.wav"
        assert rescued == archived
        assert archived.read_bytes() == b"audio"
        assert not rec_file.exists()

    def test_copy_failure_preserves_origin_and_returns_none(self, tmp_path, capsys):
        rec_file = tmp_path / "digue-rec.wav"  # never created: the copy must fail
        audio_dir = tmp_path / "audio"

        result = digue.rescue_recording(rec_file, audio_dir, "20260904-120000", "0123456789abcdef")

        assert result is None
        assert "Failed to keep recording" in capsys.readouterr().err

    def test_publish_failure_preserves_origin_and_removes_temp(self, tmp_path, capsys):
        rec_file = tmp_path / "digue-rec.wav"
        rec_file.write_bytes(b"audio")
        audio_dir = tmp_path / "audio"

        with patch("os.link", side_effect=OSError("disk full")):
            result = digue.rescue_recording(rec_file, audio_dir, "20260904-120000", "0123456789abcdef")

        month_dir = audio_dir / "2026" / "09"
        assert result is None
        assert rec_file.exists()
        assert not any(path.name.startswith(".") for path in month_dir.iterdir())
        assert "Failed to keep recording" in capsys.readouterr().err


class TestTakeIdInSavedNames:
    """Two takes ending in the same second used to overwrite each other silently
    (<YYYYMMDD-HHMMSS>.<ext>): the take id makes saved names unique, and every
    write is exclusive, so a collision can never drop a file."""

    def test_take_id_goes_into_audio_and_transcript_names(self, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"wav data")
        audio_dir = tmp_path / "audio"

        saved, _ = digue.save_audio(rec_file, audio_dir, timestamp="20260904-120000", take_id="0123456789abcdef")
        text_path = digue._write_transcript(audio_dir, "20260904-120000", "hello", take_id="0123456789abcdef")

        assert saved.name == "20260904-120000-0123456789abcdef.wav"
        assert text_path.name == "20260904-120000-0123456789abcdef.txt"

    def test_two_takes_in_the_same_second_generate_four_distinct_files(self, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"wav data")
        audio_dir = tmp_path / "audio"

        saved_a, _ = digue.save_audio(rec_file, audio_dir, timestamp="20260904-120000", take_id="0" * 16)
        text_a = digue._write_transcript(audio_dir, "20260904-120000", "a", take_id="0" * 16)
        saved_b, _ = digue.save_audio(rec_file, audio_dir, timestamp="20260904-120000", take_id="f" * 16)
        text_b = digue._write_transcript(audio_dir, "20260904-120000", "b", take_id="f" * 16)

        assert len({saved_a, saved_b, text_a, text_b}) == 4
        assert all(path.exists() for path in (saved_a, saved_b, text_a, text_b))

    def test_saving_never_overwrites_an_existing_file(self, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"new data")
        audio_dir = tmp_path / "audio"
        month_dir = audio_dir / "2026" / "09"
        month_dir.mkdir(parents=True)
        existing = month_dir / "20260904-120000.wav"
        existing.write_bytes(b"original")

        with pytest.raises(FileExistsError):
            digue.save_audio(rec_file, audio_dir, timestamp="20260904-120000")

        assert existing.read_bytes() == b"original"

    def test_transcript_write_is_exclusive(self, tmp_path):
        audio_dir = tmp_path / "audio"
        month_dir = audio_dir / "2026" / "09"
        month_dir.mkdir(parents=True)
        existing = month_dir / "20260904-120000.txt"
        existing.write_text("original\n")

        with pytest.raises(FileExistsError):
            digue._write_transcript(audio_dir, "20260904-120000", "hello")

        assert existing.read_text() == "original\n"


class TestDictationFiles:
    def test_accepts_old_and_new_layout_and_ignores_other_files(self, tmp_path):
        audio_dir = tmp_path / "audio"
        month_dir = audio_dir / "2026" / "09"
        month_dir.mkdir(parents=True)
        old_wav = month_dir / "20260904-120000.wav"
        new_flac = month_dir / "20260904-120001-0123456789abcdef.flac"
        new_txt = month_dir / "20260904-120001-0123456789abcdef.txt"
        music = month_dir / "01 - Song.wav"
        notes = month_dir / "random.txt"
        for path in (old_wav, new_flac, new_txt, music, notes):
            path.write_bytes(b"x")

        recordings = digue._dictation_files(audio_dir, digue.DICTATION_RECORDING_SUFFIXES)
        transcripts = digue._dictation_files(audio_dir, frozenset((".txt",)))

        assert recordings == [old_wav, new_flac]
        assert transcripts == [new_txt]

    def test_symlinks_are_ignored(self, tmp_path):
        audio_dir = tmp_path / "audio"
        month_dir = audio_dir / "2026" / "09"
        month_dir.mkdir(parents=True)
        target = tmp_path / "elsewhere.wav"
        target.write_bytes(b"x")
        (month_dir / "20260904-120000.wav").symlink_to(target)

        assert digue._dictation_files(audio_dir, digue.DICTATION_RECORDING_SUFFIXES) == []


class TestCompressAudio:
    def test_wav_is_noop(self, tmp_path):
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"data")
        assert digue._compress_audio(rec, "wav") == rec

    def test_flac_compresses_and_removes_wav(self, tmp_path):
        import shutil
        import wave

        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not available")
        rate = 16000
        samples = b"".join(
            int(8000 * math.sin(2 * math.pi * 440 * i / rate)).to_bytes(2, "little", signed=True) for i in range(rate)
        )
        rec = tmp_path / "rec.wav"
        with wave.open(str(rec), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(samples)
        wav_size = rec.stat().st_size

        result = digue._compress_audio(rec, "flac")

        assert result.suffix == ".flac"
        assert result.exists()
        assert not rec.exists()
        assert result.stat().st_size < wav_size

    def test_unknown_format_raises(self, tmp_path):
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"data")
        with pytest.raises(KeyError):
            digue._compress_audio(rec, "mp3")

    @patch("digue.container_status", return_value=None)
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_local_ffmpeg_writes_temp_in_final_directory_and_replaces(
        self, mock_run, mock_which, mock_status, tmp_path
    ):
        import subprocess

        rec = tmp_path / "20260904-120000-0123456789abcdef.wav"
        rec.write_bytes(b"wav-data")
        temp_paths = []

        def fake_run(cmd, **kwargs):
            assert "-y" not in cmd
            temp_path = Path(cmd[-1])
            temp_paths.append(temp_path)
            temp_path.write_bytes(b"flac-data")
            return subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")

        mock_run.side_effect = fake_run

        result = digue._compress_audio(rec, "flac")

        assert result == tmp_path / "20260904-120000-0123456789abcdef.flac"
        assert result.read_bytes() == b"flac-data"
        assert not rec.exists()
        assert temp_paths[0].parent == tmp_path
        assert temp_paths[0].name.startswith(".") and temp_paths[0].name.endswith(".tmp")
        assert [path.name for path in tmp_path.iterdir() if path.is_file()] == [result.name]

    @patch("digue.container_status", return_value=None)
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_local_ffmpeg_failure_removes_temp_and_reservation_keeps_wav(
        self, mock_run, mock_which, mock_status, tmp_path
    ):
        import subprocess

        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"error")
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"wav-data")

        result = digue._compress_audio(rec, "flac")

        assert result == rec
        assert rec.exists()
        assert [path.name for path in tmp_path.iterdir() if path.is_file()] == [rec.name]

    @pytest.mark.parametrize("failure", [subprocess.TimeoutExpired(cmd="ffmpeg", timeout=600), OSError("cannot fork")])
    def test_unexpected_ffmpeg_error_releases_the_reservation_and_keeps_the_wav(self, failure, tmp_path):
        """The final name is reserved up front (exclusive touch); an exception
        out of subprocess.run must not leave that empty .flac next to the WAV."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        rec = audio_dir / "rec.wav"
        rec.write_bytes(b"data")

        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("subprocess.run", side_effect=failure),
            pytest.raises(type(failure)),
        ):
            digue._compress_audio(rec, "flac")

        assert rec.read_bytes() == b"data"
        assert sorted(path.name for path in audio_dir.iterdir()) == ["rec.wav"]

    def test_unexpected_container_error_releases_the_reservation_and_keeps_the_wav(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        rec = audio_dir / "rec.wav"
        rec.write_bytes(b"data")

        with (
            patch("shutil.which", return_value=None),
            patch("digue.container_status", return_value="running"),
            patch("subprocess.run", side_effect=OSError("docker gone")),
            pytest.raises(OSError),
        ):
            digue._compress_audio(rec, "flac", backend="cpu")

        assert rec.read_bytes() == b"data"
        assert sorted(path.name for path in audio_dir.iterdir()) == ["rec.wav"]

    def test_compression_never_overwrites_an_existing_destination(self, tmp_path):
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"wav-data")
        existing = tmp_path / "rec.flac"
        existing.write_bytes(b"original")

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"), pytest.raises(FileExistsError):
            digue._compress_audio(rec, "flac")

        assert existing.read_bytes() == b"original"
        assert rec.exists()

    @patch("digue.container_status", return_value="running")
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_container_fallback_when_host_ffmpeg_missing(self, mock_run, mock_which, mock_status, tmp_path):
        import subprocess

        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"flac-data", stderr=b"")
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"wav-data")

        result = digue._compress_audio(rec, "flac", backend="amd")

        assert result == tmp_path / "rec.flac"
        assert result.read_bytes() == b"flac-data"
        assert not rec.exists()
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[:5] == ["docker", "exec", "-i", digue.CONTAINER_NAME, "ffmpeg"]
        assert "-c:a" in cmd and "flac" in cmd
        assert mock_run.call_args[1]["input"] == b"wav-data"

    @patch("digue.container_status", return_value="running")
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_container_fallback_opus(self, mock_run, mock_which, mock_status, tmp_path):
        import subprocess

        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"opus-data", stderr=b"")
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"wav-data")

        result = digue._compress_audio(rec, "opus")

        assert result == tmp_path / "rec.opus"
        assert result.read_bytes() == b"opus-data"
        assert not rec.exists()
        cmd = mock_run.call_args[0][0]
        assert cmd[:5] == ["docker", "exec", "-i", digue.CONTAINER_NAME, "ffmpeg"]
        assert "-c:a" in cmd and "libopus" in cmd
        assert "-f" in cmd and "ogg" in cmd

    @patch("digue.container_status", return_value="running")
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_container_fallback_skipped_if_remote_backend(self, mock_run, mock_which, mock_status, tmp_path):
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"wav-data")

        result = digue._compress_audio(rec, "flac", backend="remote")

        assert result == rec
        assert rec.exists()
        mock_run.assert_not_called()

    @patch("digue.container_status", return_value=None)
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_container_fallback_skipped_if_container_not_running(self, mock_run, mock_which, mock_status, tmp_path):
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"wav-data")

        result = digue._compress_audio(rec, "flac")

        assert result == rec
        assert rec.exists()
        mock_run.assert_not_called()

    @patch("digue.container_status", return_value="running")
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_container_fallback_failure_keeps_wav(self, mock_run, mock_which, mock_status, tmp_path):
        import subprocess

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b"conversion error"
        )
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"wav-data")

        result = digue._compress_audio(rec, "flac")

        assert result == rec
        assert rec.exists()
        assert not (tmp_path / "rec.flac").exists()


class TestSaveAudioConfig:
    def test_default_saves_audio(self):
        config = digue._default_config()
        assert config["dictate"]["save_audio"] is True

    def test_loads_kebab_key(self, tmp_path):
        import textwrap

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent("""\
            [dictate]
            save-audio = false
        """)
        )
        config = digue.load_config(config_path)
        assert config["dictate"]["save_audio"] is False

    @patch("digue._compress_audio")
    def test_save_audio_passes_backend(self, mock_compress, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        audio_dir = tmp_path / "audio"
        mock_compress.return_value = audio_dir / "2026/01/test.flac"

        digue.save_audio(rec_file, audio_dir, audio_format="flac", timestamp="2026-01-01T00-00-00", backend="amd")

        mock_compress.assert_called_once()
        assert mock_compress.call_args[1]["backend"] == "amd"

    @patch("digue.send_text")
    @patch("digue.transcribe", return_value="hello")
    @patch("digue.save_audio", return_value=("saved.flac", "2026-01-01T00:00:00"))
    def test_finish_dictation_passes_resolved_backend(self, mock_save, mock_transcribe, mock_send, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        config = digue._default_config()
        config["server"]["backend"] = "remote"
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")

        with patch("digue.detect_backend") as mock_detect:
            digue.finish_dictation(config, rec_file)

        mock_save.assert_called_once()
        assert mock_save.call_args[1]["backend"] == "remote"
        mock_detect.assert_not_called()  # only remote-or-not matters here: no nvidia-smi/lspci per delivery

    @patch("digue.send_text")
    @patch("digue.transcribe", return_value="hello")
    @patch("digue.save_audio", return_value=("saved.flac", "2026-01-01T00:00:00"))
    def test_finish_dictation_does_not_detect_hardware_for_a_local_backend(
        self, mock_save, mock_transcribe, mock_send, tmp_path
    ):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")

        with patch("digue.detect_backend") as mock_detect:
            digue.finish_dictation(config, rec_file)

        mock_detect.assert_not_called()
        assert mock_save.call_args[1]["backend"] != "remote"

    @patch("digue.send_text")
    @patch("digue.transcribe", return_value="hello")
    @patch("digue.save_audio")
    def test_save_audio_false_skips_wav_but_writes_txt(self, mock_save, mock_transcribe, mock_send, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        audio_dir = tmp_path / "audio"
        config = digue._default_config()
        config["dictate"]["save_audio"] = False
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = digue.finish_dictation(config, rec_file)

        assert result.exit_code == 0
        mock_save.assert_not_called()
        txt_files = list(audio_dir.rglob("*.txt"))
        assert len(txt_files) == 1
        assert txt_files[0].read_text().strip() == "hello"


# -- Batch commands -----------------------------------------------------------


class TestSilentAudioTimestamps:
    """whisper-server answers a silent file with a bare "WEBVTT" header (no
    cues); the timestamps format is built from that VTT and must not fail."""

    def test_convert_header_only_vtt_to_timestamps_gives_empty_text(self):
        assert digue._convert_content("WEBVTT\n\n", "vtt", "timestamps") == ""

    def test_convert_header_only_vtt_to_text_gives_empty_text(self):
        assert digue._convert_content("WEBVTT\n", "vtt", "text") == ""

    def test_convert_header_only_vtt_to_srt_still_fails(self):
        with pytest.raises(ValueError, match="no cues"):
            digue._convert_content("WEBVTT\n\n", "vtt", "srt")

    def test_non_subtitle_content_is_still_rejected(self):
        with pytest.raises(ValueError, match="does not look like a VTT"):
            digue._convert_content("just some prose\n", "vtt", "timestamps")

    def test_header_followed_by_garbage_is_rejected(self):
        """A WEBVTT header does not make everything after it an empty file."""
        with pytest.raises(ValueError, match="does not look like a VTT"):
            digue._convert_content("WEBVTT\n\nsome text without any timing line\n", "vtt", "timestamps")

    def test_header_with_metadata_blocks_only_is_empty(self):
        content = (
            "WEBVTT\nKind: captions\nLanguage: pt\n\nNOTE\nmade by digue\nsecond line\n\nSTYLE\n::cue { color: red }\n"
        )
        assert digue._convert_content(content, "vtt", "timestamps") == ""

    @patch("digue.transcribe", return_value="WEBVTT\n")
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    def test_cmd_transcribe_timestamps_on_silent_audio_succeeds(
        self, mock_running, mock_ensure, mock_transcribe, tmp_path, capsys
    ):
        audio = tmp_path / "silence.wav"
        audio.write_bytes(b"audio")
        args = MagicMock()
        args.audio = audio
        args.output = None
        args.response_format = "timestamps"
        args.language = None
        args.prompt = None
        args.verbose = False

        result = digue.cmd_transcribe(args, digue._default_config())

        captured = capsys.readouterr()
        assert result == 0
        assert captured.out == "\n"
        assert "does not look like" not in captured.err

    @patch("digue.transcribe", return_value="WEBVTT\n")
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    def test_batch_transcribe_timestamps_on_silent_audio_is_not_a_failure(
        self, mock_running, mock_ensure, mock_transcribe, tmp_path
    ):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (input_dir / "silence.mp3").write_bytes(b"audio")
        args = MagicMock()
        args.input_dir = input_dir
        args.output_dir = output_dir
        args.response_format = "timestamps"
        args.language = None

        result = digue.cmd_batch_transcribe(args, digue._default_config())

        assert result == 0
        assert (output_dir / "silence.txt").read_text() == "\n"


class TestCmdBatchTranscribeInput:
    @patch("digue.ensure_server")
    @patch("digue.is_server_running")
    def test_empty_input_does_not_start_server(self, mock_running, mock_ensure, tmp_path, capsys):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        args = MagicMock(input_dir=input_dir, output_dir=output_dir, language=None, response_format=None)

        assert digue.cmd_batch_transcribe(args, digue._default_config()) == 1

        assert "No audio files" in capsys.readouterr().err
        mock_ensure.assert_not_called()
        mock_running.assert_not_called()

    @patch("digue.ensure_server")
    @patch("digue.is_server_running")
    def test_completed_input_does_not_start_server(self, mock_running, mock_ensure, tmp_path, capsys):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        (input_dir / "audio.wav").write_bytes(b"audio")
        (output_dir / "audio.txt").write_text("done\n")
        args = MagicMock(input_dir=input_dir, output_dir=output_dir, language=None, response_format=None)

        assert digue.cmd_batch_transcribe(args, digue._default_config()) == 0

        assert "All files already transcribed" in capsys.readouterr().err
        mock_ensure.assert_not_called()
        mock_running.assert_not_called()


class TestCmdBatchSimplifyVtt:
    def test_simplifies_all_vtt_files(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (input_dir / "a.vtt").write_text("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nHello\n")
        (input_dir / "b.vtt").write_text("WEBVTT\n\n1\n00:00:00.000 --> 00:00:02.000\nWorld\n")
        (input_dir / "ignore.txt").write_text("not a vtt")

        args = MagicMock()
        args.input_dir = input_dir
        args.output_dir = output_dir
        config = digue._default_config()

        result = digue.cmd_batch_simplify_vtt(args, config)
        assert result == 0
        assert (output_dir / "a.txt").exists()
        assert (output_dir / "b.txt").exists()
        assert "[00:00:00] Hello" in (output_dir / "a.txt").read_text()

    def test_skips_already_simplified(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (input_dir / "a.vtt").write_text("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nHello\n")
        (output_dir / "a.txt").write_text("already done")

        args = MagicMock()
        args.input_dir = input_dir
        args.output_dir = output_dir
        config = digue._default_config()

        result = digue.cmd_batch_simplify_vtt(args, config)
        assert result == 0
        assert (output_dir / "a.txt").read_text() == "already done"  # not overwritten

    def test_returns_1_when_no_files(self, tmp_path):
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        args = MagicMock()
        args.input_dir = input_dir
        args.output_dir = output_dir
        config = digue._default_config()

        result = digue.cmd_batch_simplify_vtt(args, config)
        assert result == 1

    def test_reports_failures_and_leaves_no_partial_output(self, tmp_path, capsys):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (input_dir / "ok.vtt").write_text("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nHello\n")
        (input_dir / "bad.vtt").write_text("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nWorld\n")

        args = MagicMock()
        args.input_dir = input_dir
        args.output_dir = output_dir
        config = digue._default_config()

        with patch("digue.simplify_vtt", side_effect=[RuntimeError("broken vtt"), "second"]):
            result = digue.cmd_batch_simplify_vtt(args, config)

        assert result == 1
        assert not (output_dir / "bad.txt").exists()
        assert not list(output_dir.glob("*.tmp"))
        assert "1 succeeded, 1 failed" in capsys.readouterr().err


class TestCmdBatchTranscribe:
    @patch("digue.transcribe", return_value="transcribed text")
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.ensure_server")
    def test_transcribes_audio_files(self, mock_ensure, mock_running, mock_transcribe, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (input_dir / "a.mp3").write_bytes(b"audio")
        (input_dir / "b.wav").write_bytes(b"audio")
        (input_dir / "readme.txt").write_text("not audio")

        args = MagicMock()
        args.input_dir = input_dir
        args.output_dir = output_dir
        args.response_format = "vtt"
        args.language = None
        config = digue._default_config()

        result = digue.cmd_batch_transcribe(args, config)
        assert result == 0
        assert (output_dir / "a.vtt").exists()
        assert (output_dir / "b.vtt").exists()
        assert not (output_dir / "readme.vtt").exists()

    @patch("digue.transcribe", return_value="text")
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.ensure_server")
    def test_skips_already_transcribed(self, mock_ensure, mock_running, mock_transcribe, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (input_dir / "done.mp3").write_bytes(b"audio")
        (output_dir / "done.vtt").write_text("already done")
        (input_dir / "new.mp3").write_bytes(b"audio")

        args = MagicMock()
        args.input_dir = input_dir
        args.output_dir = output_dir
        args.response_format = "vtt"
        args.language = None
        config = digue._default_config()

        result = digue.cmd_batch_transcribe(args, config)
        assert result == 0
        assert mock_transcribe.call_count == 1  # only new.mp3
        assert (output_dir / "done.vtt").read_text() == "already done"

    @patch("digue.transcribe", side_effect=["first", RuntimeError("request failed")])
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.ensure_server")
    def test_returns_failure_reports_counts_and_leaves_no_partial_output(
        self, mock_ensure, mock_running, mock_transcribe, tmp_path, capsys
    ):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (input_dir / "a.mp3").write_bytes(b"audio")
        (input_dir / "b.mp3").write_bytes(b"audio")

        args = MagicMock(
            input_dir=input_dir,
            output_dir=output_dir,
            response_format="text",
            language=None,
        )

        result = digue.cmd_batch_transcribe(args, digue._default_config())

        assert result == 1
        assert (output_dir / "a.txt").read_text() == "first\n"
        assert not (output_dir / "b.txt").exists()
        assert not list(output_dir.glob("*.tmp"))
        error = capsys.readouterr().err
        assert "1 succeeded, 1 failed, 0 skipped" in error

    @patch("pathlib.Path.replace", side_effect=OSError("replace failed"))
    @patch("digue.transcribe", return_value="partial")
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.ensure_server")
    def test_failed_replace_does_not_leave_output_or_temp(
        self, mock_ensure, mock_running, mock_transcribe, mock_replace, tmp_path
    ):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (input_dir / "a.mp3").write_bytes(b"audio")
        args = MagicMock(input_dir=input_dir, output_dir=output_dir, response_format="text", language=None)

        assert digue.cmd_batch_transcribe(args, digue._default_config()) == 1
        assert not (output_dir / "a.txt").exists()
        assert not list(output_dir.glob("*.tmp"))

    @patch("digue.transcribe", return_value="WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n")
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.ensure_server")
    def test_supports_timestamps_format(self, mock_ensure, mock_running, mock_transcribe, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        (input_dir / "test.mp3").write_bytes(b"audio")

        config = digue._default_config()
        config["transcribe"]["output_format"] = "timestamps"
        args = MagicMock(input_dir=input_dir, output_dir=output_dir, response_format=None, language=None)

        assert digue.cmd_batch_transcribe(args, config) == 0
        output_file = output_dir / "test.txt"
        assert output_file.exists()
        assert output_file.read_text().strip() == "[00:00:00] Hello"
        assert mock_transcribe.call_args[0][3] == "vtt"


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
