"""Tests for digue.py."""

import argparse
import ast
import contextlib
import json
import math
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmark_models
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


class TestModelForBackend:
    def test_returns_default_without_config(self):
        assert digue.model_for_backend("nvidia") == "large-v3-turbo"
        assert digue.model_for_backend("cpu") == "small"

    def test_respects_config_override(self):
        config = {"models": {"nvidia": "medium"}}
        assert digue.model_for_backend("nvidia", config) == "medium"

    def test_unknown_backend_falls_back_to_small(self):
        assert digue.model_for_backend("unknown") == "small"


# -- Detection ----------------------------------------------------------------


class TestDetectBackend:
    @patch("subprocess.run")
    def test_nvidia_when_nvidia_smi_works(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 3060\n")
        assert digue.detect_backend() == "nvidia"

    @patch("subprocess.run", side_effect=FileNotFoundError)
    @patch("pathlib.Path.exists", return_value=True)
    def test_amd_when_kfd_exists(self, mock_exists, mock_run):
        assert digue.detect_backend() == "amd"

    @patch("pathlib.Path.exists", return_value=False)
    @patch("subprocess.run")
    def test_intel_meteor_lake(self, mock_run, mock_exists):
        # First call: nvidia-smi fails. Second call: lspci returns Intel.
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout=""),  # nvidia-smi
            MagicMock(stdout="00:02.0 VGA compatible controller: Intel Corporation Meteor Lake-P [Intel Graphics]\n"),
        ]
        assert digue.detect_backend() == "intel"

    @patch("pathlib.Path.exists", return_value=False)
    @patch("subprocess.run")
    def test_cpu_for_broadwell(self, mock_run, mock_exists):
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout=""),
            MagicMock(
                stdout="00:02.0 VGA compatible controller: Intel Corporation Broadwell-U GT2 [HD Graphics 5500]\n"
            ),
        ]
        assert digue.detect_backend() == "cpu"

    @patch("pathlib.Path.exists", return_value=False)
    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_cpu_when_no_tools(self, mock_run, mock_exists):
        assert digue.detect_backend() == "cpu"


# -- Container management ----------------------------------------------------


class TestContainerExists:
    @patch("digue._docker_run")
    def test_true_when_inspect_succeeds(self, mock_docker):
        mock_docker.return_value = MagicMock(returncode=0, stdout="running\n")
        assert digue.container_exists() is True

    @patch("digue._docker_run")
    def test_false_when_inspect_fails(self, mock_docker):
        mock_docker.return_value = MagicMock(returncode=1, stdout="")
        assert digue.container_exists() is False


class TestContainerStatus:
    @patch("digue._docker_run")
    def test_returns_status(self, mock_docker):
        mock_docker.return_value = MagicMock(returncode=0, stdout="exited\n")
        assert digue.container_status() == "exited"

    @patch("digue._docker_run")
    def test_returns_none_when_missing(self, mock_docker):
        mock_docker.return_value = MagicMock(returncode=1)
        assert digue.container_status() is None


class TestCreateContainer:
    @patch("digue.download_model")
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_nvidia_uses_gpus_flag(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = digue._default_config()
        digue.create_container(config, "nvidia")
        cmd = mock_docker.call_args[0][0]
        assert "--gpus" in cmd
        assert "all" in cmd
        assert any("main-cuda" in arg for arg in cmd)

    @patch("digue.download_model")
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_amd_uses_kfd_device(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = digue._default_config()
        digue.create_container(config, "amd")
        cmd = mock_docker.call_args[0][0]
        assert "/dev/kfd" in cmd

    @patch("digue.download_model")
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_cpu_has_no_device_flags(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = digue._default_config()
        digue.create_container(config, "cpu")
        cmd = mock_docker.call_args[0][0]
        assert "--device" not in cmd
        assert "--gpus" not in cmd

    @patch("digue.download_model")
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_raises_on_failure(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=1, stderr="permission denied")
        config = digue._default_config()
        with pytest.raises(RuntimeError, match="permission denied"):
            digue.create_container(config, "cpu")

    @patch("digue.download_model")
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_binds_to_localhost(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = digue._default_config()
        digue.create_container(config, "cpu")
        cmd = mock_docker.call_args[0][0]
        port_binding = [arg for arg in cmd if "8178" in arg and "127.0.0.1" in arg]
        assert port_binding, "Port must bind to 127.0.0.1"

    @patch("digue.download_model")
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_binds_to_configured_ip(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = digue._default_config()
        config["server"]["bind_ip"] = "192.168.1.10"
        digue.create_container(config, "cpu")
        cmd = mock_docker.call_args[0][0]
        assert "192.168.1.10:8178:8080" in cmd

    @patch("digue.download_model")
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_calls_pull_image(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = digue._default_config()
        digue.create_container(config, "cpu")
        mock_pull.assert_called_once_with("ghcr.io/ggml-org/whisper.cpp:main-vulkan")

    @patch("digue.download_model")
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_downloads_model_when_missing(self, mock_docker, mock_pull, mock_download, tmp_path):
        mock_docker.return_value = MagicMock(returncode=0)
        config = digue._default_config()
        config["server"]["data_dir"] = str(tmp_path)
        digue.create_container(config, "cpu")
        mock_download.assert_called_once_with("small", tmp_path / "models", with_notification=True)

    @patch("digue.download_model")
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_skips_download_when_model_exists(self, mock_docker, mock_pull, mock_download, tmp_path):
        mock_docker.return_value = MagicMock(returncode=0)
        config = digue._default_config()
        config["server"]["data_dir"] = str(tmp_path)
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "ggml-small.bin").write_bytes(b"dummy")
        (models_dir / digue.VAD_MODEL_FILENAME).write_bytes(b"dummy")
        digue.create_container(config, "cpu")
        mock_download.assert_not_called()

    @patch("digue.download_model")
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_downloads_when_vad_is_missing(self, mock_docker, mock_pull, mock_download, tmp_path):
        mock_docker.return_value = MagicMock(returncode=0)
        config = digue._default_config()
        config["server"]["data_dir"] = str(tmp_path)
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "ggml-small.bin").write_bytes(b"dummy")

        digue.create_container(config, "cpu")

        mock_download.assert_called_once_with("small", models_dir, with_notification=True)


class TestImageExists:
    @patch("digue._docker_run")
    def test_true_when_image_present(self, mock_docker):
        mock_docker.return_value = MagicMock(returncode=0)
        assert digue.image_exists("test:latest") is True

    @patch("digue._docker_run")
    def test_false_when_image_missing(self, mock_docker):
        mock_docker.return_value = MagicMock(returncode=1)
        assert digue.image_exists("test:latest") is False


class TestPullImage:
    @patch("digue.image_exists", return_value=True)
    @patch("subprocess.run")
    def test_skips_when_exists(self, mock_run, mock_exists):
        digue.pull_image("test:latest")
        mock_run.assert_not_called()

    @patch("digue.image_exists", return_value=False)
    @patch("subprocess.run", return_value=MagicMock(returncode=0))
    def test_pulls_when_missing(self, mock_run, mock_exists):
        digue.pull_image("test:latest")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["docker", "pull", "test:latest"]

    @patch("digue.image_exists", return_value=False)
    @patch("subprocess.run", return_value=MagicMock(returncode=1))
    def test_raises_on_failure(self, mock_run, mock_exists):
        with pytest.raises(RuntimeError, match="Failed to pull"):
            digue.pull_image("test:latest")


class TestDownloadProgressHook:
    def test_formats_percentage(self, capsys):
        hook = digue._download_progress_hook("test.bin")
        hook(50, 1024 * 1024, 100 * 1024 * 1024)  # 50MB of 100MB
        output = capsys.readouterr().err
        assert "50%" in output
        assert "test.bin" in output

    def test_terminal_line_has_carriage_return_and_bar(self, capsys):
        with patch.object(digue, "_stderr_is_tty", return_value=True):
            hook = digue._download_progress_hook("test.bin")
            hook(25, 1024 * 1024, 100 * 1024 * 1024)  # 25%
        output = capsys.readouterr().err
        assert output.startswith("\r")
        assert "[" in output and "]" in output

    def test_non_terminal_prints_sparse_lines_without_bar(self, capsys):
        with patch.object(digue, "_stderr_is_tty", return_value=False):
            hook = digue._download_progress_hook("test.bin")
            for block in range(0, 100 * 1024 * 1024, 8 * 1024 * 1024):
                hook(block, 1, 100 * 1024 * 1024)
            hook(100 * 1024 * 1024, 1, 100 * 1024 * 1024)  # final block
        output = capsys.readouterr().err
        lines = [line for line in output.splitlines() if line.strip()]
        assert all("[" not in line for line in lines)  # no bar
        assert len(lines) <= 22  # ~5% steps + 100%, not one line per block
        assert lines[-1].endswith("100.0/100.0 MB")

    def test_fixed_width_across_digit_rollover(self, capsys):
        # 9.9 MB -> 10.0 MB must keep the line exactly the same length
        with patch.object(digue, "_stderr_is_tty", return_value=True):
            hook = digue._download_progress_hook("test.bin")
            hook(0, 1, 20 * 1024 * 1024)  # 0.0 MB of 20.0 MB
            first = capsys.readouterr().err
            hook(10380902, 1, 20 * 1024 * 1024)  # 9.9 MB
            at99 = capsys.readouterr().err
            hook(10485760, 1, 20 * 1024 * 1024)  # 10.0 MB
            at100 = capsys.readouterr().err
        assert len(first.rstrip("\r")) == len(at99.rstrip("\r")) == len(at100.rstrip("\r"))

    def test_notification_message_has_no_bar_and_fixed_width(self):
        messages = []
        with (
            patch("digue.notify", side_effect=lambda message, timeout_ms=0: messages.append(message)),
            patch("time.monotonic", side_effect=[1.0, 2.0, 3.0, 4.0]),
        ):
            hook = digue._download_progress_hook("test.bin", with_notification=True)
            # 97.1 MB and 102.5 MB of 465.0 MB: digit rollover must keep width
            hook(101816944, 1, 465 * 1024 * 1024)
            hook(107479040, 1, 465 * 1024 * 1024)
        assert len(messages) == 2
        assert all("\r" not in message for message in messages)
        assert all("[" not in message for message in messages)
        assert len(messages[0]) == len(messages[1])

    def test_unknown_content_length_can_notify(self):
        messages = []
        with (
            patch("digue.notify", side_effect=lambda message, timeout_ms=0: messages.append(message)),
            patch("time.monotonic", return_value=1.0),
        ):
            hook = digue._download_progress_hook("test.bin", with_notification=True)
            hook(1, 1024 * 1024, -1)
        assert messages == ["Downloading test.bin...       1.0 MB"]

    def test_notify_prints_carriage_return_on_tty(self, capsys):
        with patch.object(digue, "_stderr_is_tty", return_value=True):
            digue.notify("status message")
        err = capsys.readouterr().err
        assert err.startswith("\r")
        assert "[digue] status message" in err

    def test_notify_prints_plain_line_when_not_tty(self, capsys):
        with patch.object(digue, "_stderr_is_tty", return_value=False):
            digue.notify("status message")
        err = capsys.readouterr().err
        assert err.startswith("[digue] status message\n")


class TestDownloadModel:
    def test_downloads_to_part_with_timeout_then_replaces_atomically(self, tmp_path):
        model_path = tmp_path / "ggml-small.bin"
        vad_path = tmp_path / "ggml-silero-v6.2.0.bin"
        vad_path.write_bytes(b"vad")
        response = MagicMock()
        response.headers = {"Content-Length": "5"}
        response.read.side_effect = [b"model", b""]
        response.__enter__.return_value = response

        def check_destination_is_not_visible(*args, **kwargs):
            assert not model_path.exists()
            return response

        with patch("urllib.request.urlopen", side_effect=check_destination_is_not_visible) as mock_urlopen:
            digue.download_model("small", tmp_path)

        assert model_path.read_bytes() == b"model"
        assert not model_path.with_suffix(".bin.part").exists()
        assert mock_urlopen.call_args.kwargs["timeout"] == digue.DOWNLOAD_TIMEOUT

    def test_partial_file_is_not_treated_as_ready_model(self, tmp_path):
        part_path = tmp_path / "ggml-small.bin.part"
        part_path.write_bytes(b"partial")
        (tmp_path / "ggml-silero-v6.2.0.bin").write_bytes(b"vad")
        response = MagicMock()
        response.headers = {}
        response.read.side_effect = [b"complete", b""]
        response.__enter__.return_value = response

        with patch("urllib.request.urlopen", return_value=response) as mock_urlopen:
            digue.download_model("small", tmp_path)

        mock_urlopen.assert_called_once()
        assert (tmp_path / "ggml-small.bin").read_bytes() == b"complete"

    def test_interrupted_download_leaves_no_part_file(self, tmp_path):
        """A failed download left ggml-*.bin.part behind in the models dir."""
        response = MagicMock()
        response.headers = {"Content-Length": "10"}
        response.read.side_effect = [b"half", OSError("connection reset")]
        response.__enter__.return_value = response

        with patch("urllib.request.urlopen", return_value=response), pytest.raises(OSError, match="reset"):
            digue._download_file("http://example/model.bin", tmp_path / "ggml-small.bin", "ggml-small.bin")

        assert not (tmp_path / "ggml-small.bin").exists()
        assert not (tmp_path / "ggml-small.bin.part").exists()


# -- Notifications -----------------------------------------------------------


class TestNotify:
    @patch("subprocess.run")
    def test_sends_with_replace_id(self, mock_run):
        digue.notify("test", timeout_ms=5000)
        cmd = mock_run.call_args[0][0]
        assert "--replace-id" in cmd
        replace_id = int(cmd[cmd.index("--replace-id") + 1])
        assert digue.NOTIFY_REPLACE_ID <= replace_id < digue.NOTIFY_REPLACE_ID + digue.NOTIFY_ID_SLOTS

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_prints_warning_when_notify_send_missing(self, mock_run, capsys):
        digue._notify_send_warned = False
        digue.notify("test")
        err = capsys.readouterr().err
        assert "notify-send not found" in err

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_warns_only_once(self, mock_run, capsys):
        digue._notify_send_warned = False
        digue.notify("first")
        digue.notify("second")
        err = capsys.readouterr().err
        assert err.count("notify-send not found") == 1

    @patch("subprocess.run")
    def test_always_prints_to_stderr(self, mock_run, capsys):
        digue.notify("hello world")
        err = capsys.readouterr().err
        assert "hello world" in err

    @patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "notify-send"))
    def test_failing_notify_send_warns_once(self, mock_run, capsys):
        """A failed notification must be visible, not silently swallowed."""
        digue._notify_send_warned = False
        digue.notify("first")
        digue.notify("second")
        err = capsys.readouterr().err
        assert err.count("notify-send failed") == 1

    def test_response_is_closed_after_request(self, tmp_path):
        response = MagicMock()
        response.read.return_value = b"ok"
        with patch("urllib.request.urlopen", return_value=response):
            digue._multipart_request("http://x/inference", b"audio", {"model": "m"}, timeout=10)
        # The urllib stream must be released even if later processing fails.
        response.close.assert_called_once()


class TestNotifyClose:
    @patch("subprocess.run")
    def test_calls_gdbus(self, mock_run):
        digue.notify_close()
        cmd = mock_run.call_args[0][0]
        assert "gdbus" in cmd
        assert any("CloseNotification" in arg for arg in cmd)


# -- Server -------------------------------------------------------------------


class TestIsServerRunning:
    @patch("urllib.request.urlopen")
    def test_true_when_responds(self, mock_urlopen):
        config = digue._default_config()
        assert digue.is_server_running(config) is True

    @patch("urllib.request.urlopen", side_effect=OSError)
    def test_false_when_refused(self, mock_urlopen):
        config = digue._default_config()
        assert digue.is_server_running(config) is False


# -- Transcription ------------------------------------------------------------


class TestTranscribe:
    @patch("urllib.request.urlopen")
    def test_returns_stripped_text(self, mock_urlopen, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake wav data")

        mock_response = MagicMock()
        mock_response.read.return_value = b"  hello world  \n"
        mock_urlopen.return_value = mock_response

        result = digue.transcribe(
            "http://localhost:8178/inference",
            audio_file,
            "en",
        )
        assert result == "hello world"

    @patch("urllib.request.urlopen")
    def test_sends_language_field(self, mock_urlopen, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = b"text"
        mock_urlopen.return_value = mock_response

        digue.transcribe("http://localhost:8178/inference", audio_file, "pt")
        request = mock_urlopen.call_args[0][0]
        assert b"language" in request.data
        assert b"pt" in request.data

    @patch("urllib.request.urlopen")
    def test_sends_language_auto_explicitly(self, mock_urlopen, tmp_path):
        """Regression: the server's default language is "en"; "auto" must be sent
        explicitly so whisper detects the language instead of forcing English."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = b"text"
        mock_urlopen.return_value = mock_response

        digue.transcribe("http://localhost:8178/inference", audio_file, "auto")
        request = mock_urlopen.call_args[0][0]
        assert b'name="language"' in request.data
        assert b"auto" in request.data

    @patch("urllib.request.urlopen")
    def test_text_format_joins_segments_into_single_line(self, mock_urlopen, tmp_path):
        # Whisper segments start with a space and the server joins them with
        # newlines; text output must be one clean line.
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = b"First segment here.\n Second one.\n Third one.\n"
        mock_urlopen.return_value = mock_response

        result = digue.transcribe("http://localhost:8178/inference", audio_file, "pt")
        assert result == "First segment here. Second one. Third one."

    @patch("urllib.request.urlopen")
    def test_status_messages_are_silent_by_default(self, mock_urlopen, tmp_path, capsys):
        # The 400 retry path is a normal condition: no status output unless verbose.
        import urllib.error

        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = b"text"
        error = urllib.error.HTTPError("http://x", 400, "Bad Request", {}, None)
        mock_urlopen.side_effect = [error, mock_response]

        with patch("digue._convert_to_wav", return_value=b"wav"):
            result = digue.transcribe("http://localhost:8178/inference", audio_file, "pt")
        assert result == "text"
        assert capsys.readouterr().err == ""

    @patch("urllib.request.urlopen")
    def test_status_messages_print_when_verbose(self, mock_urlopen, tmp_path, capsys):
        import urllib.error

        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = b"text"
        error = urllib.error.HTTPError("http://x", 400, "Bad Request", {}, None)
        mock_urlopen.side_effect = [error, mock_response]

        with patch("digue._convert_to_wav", return_value=b"wav"):
            result = digue.transcribe("http://localhost:8178/inference", audio_file, "pt", verbose=True)
        assert result == "text"
        err = capsys.readouterr().err
        assert "HTTP 400" in err
        assert "Trying ffmpeg conversion" in err

    @patch("urllib.request.urlopen")
    def test_vtt_cues_are_stripped_and_wrapped(self, mock_urlopen, tmp_path):
        # whisper cues start with a space; vtt output must be stripped, and
        # long cues wrapped into max_lines lines of max_line_length
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b"WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n "
            b"uma frase bem comprida que passa do limite de quarenta e dois caracteres\n"
        )
        mock_urlopen.return_value = mock_response

        result = digue.transcribe("http://x", audio_file, "pt", response_format="vtt")
        assert "WEBVTT" in result
        for line in result.splitlines():
            if "-->" in line or line.startswith("WEBVTT"):
                continue
            assert line == line.strip()  # no leading/trailing spaces
            assert len(line) <= 42

    @patch("urllib.request.urlopen")
    def test_srt_cues_are_stripped_and_keep_index(self, mock_urlopen, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = b"1\n00:00:00,000 --> 00:00:01,000\n cue com espaco\n\n"
        mock_urlopen.return_value = mock_response

        result = digue.transcribe("http://x", audio_file, "pt", response_format="srt")
        lines = result.splitlines()
        assert lines[0] == "1"
        assert "-->" in lines[1]
        assert lines[2] == "cue com espaco"  # stripped

    def test_srt_number_in_cue_text_is_not_treated_as_index(self):
        srt = (
            "1\n"
            "00:00:00,000 --> 00:00:01,000\n"
            "Ano de\n"
            "1984\n"
            "no Brasil\n\n"
            "2\n"
            "00:00:02,000 --> 00:00:03,000\n"
            "Segunda fala\n"
        )
        result = digue._post_process_subtitle(srt, "srt", max_line_length=42, max_lines=2)
        lines = result.splitlines()
        assert lines[0] == "1"
        assert lines[1] == "00:00:00,000 --> 00:00:01,000"
        assert lines[2] == "Ano de 1984 no Brasil"
        assert lines[3] == ""
        assert lines[4] == "2"

    @patch("urllib.request.urlopen")
    def test_timestamps_format_from_server_vtt(self, mock_urlopen, tmp_path):
        # -f timestamps: transcribe() gets the server VTT (already stripped),
        # then cmd_transcribe converts it; distinct cues must all appear
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b"WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n Primeira frase.\n\n"
            b"00:00:02.100 --> 00:00:04.000\n Segunda frase com mais conteudo.\n"
        )
        mock_urlopen.return_value = mock_response

        with patch.object(digue, "_stderr_is_tty", return_value=True):
            result = digue.transcribe("http://x", audio_file, "pt", response_format="vtt")
        converted = digue._convert_content(result, "vtt", "timestamps")
        assert "[00:00:00] Primeira frase." in converted
        assert "[00:00:02] Segunda frase com mais conteudo." in converted

    @patch("urllib.request.urlopen")
    def test_timestamps_format_joins_same_instant(self, mock_urlopen, tmp_path):
        # -f timestamps: one line per timestamp with ALL its text (no wrap);
        # transcribe requests wrapped=off so cues stay single-line and the
        # conversion yields one line per timestamp.
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = b"WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n uma frase bem comprida que passa do limite de quarenta e dois caracteres\n"
        mock_urlopen.return_value = mock_response

        # Simulate the cmd_transcribe path: vtt with wrap disabled, then convert
        with patch.object(digue, "_stderr_is_tty", return_value=True):
            vtt = digue.transcribe("http://x", audio_file, "pt", response_format="vtt", wrap_cues=False)
        result = digue._convert_content(vtt, "vtt", "timestamps")
        lines = [line for line in result.splitlines() if line.strip()]
        assert len(lines) == 1  # same timestamp, single line
        assert lines[0] == "[00:00:00] uma frase bem comprida que passa do limite de quarenta e dois caracteres"

    @patch("urllib.request.urlopen")
    def test_vtt_format_keeps_lines(self, mock_urlopen, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = b"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n First segment\n"
        mock_urlopen.return_value = mock_response

        result = digue.transcribe("http://localhost:8178/inference", audio_file, "pt", response_format="vtt")
        assert "WEBVTT" in result
        assert "\n" in result


# -- Ffmpeg fallback ----------------------------------------------------------


class TestTranscribeFfmpegFallback:
    @patch("digue._send_audio")
    def test_native_format_sends_directly(self, mock_send, tmp_path):
        mock_send.return_value = "ok"
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"data")
        assert digue.transcribe("http://x", audio) == "ok"
        mock_send.assert_called_once()

    @patch("digue._convert_to_wav", return_value=b"wav-bytes")
    @patch("digue._send_audio")
    def test_converts_unknown_extension_upfront(self, mock_send, mock_convert, tmp_path):
        audio = tmp_path / "file.amr"
        audio.write_bytes(b"data")
        mock_send.return_value = "text"
        result = digue.transcribe("http://x", audio)
        assert result == "text"
        mock_convert.assert_called_once_with(audio)

    @patch("digue._convert_to_wav", return_value=b"wav-bytes")
    @patch("digue._send_audio")
    def test_retries_after_http_400(self, mock_send, mock_convert, tmp_path):
        import urllib.error

        audio = tmp_path / "file.ogg"
        audio.write_bytes(b"data")
        error = urllib.error.HTTPError("http://x", 400, "Bad Request", {}, None)
        mock_send.side_effect = [error, "converted text"]

        result = digue.transcribe("http://x", audio)

        assert result == "converted text"
        assert mock_send.call_count == 2
        mock_convert.assert_called_once_with(audio)

    @patch("shutil.which", return_value=None)
    @patch("digue._send_audio")
    def test_400_without_ffmpeg_raises(self, mock_send, mock_which, tmp_path):
        import urllib.error

        audio = tmp_path / "file.ogg"
        audio.write_bytes(b"data")
        mock_send.side_effect = urllib.error.HTTPError("http://x", 400, "Bad Request", {}, None)
        with pytest.raises(RuntimeError, match="ffmpeg is not installed"):
            digue.transcribe("http://x", audio)

    @patch("shutil.which", return_value=None)
    @patch("digue._send_audio")
    def test_unknown_extension_without_ffmpeg_raises(self, mock_send, mock_which, tmp_path):
        audio = tmp_path / "file.amr"
        audio.write_bytes(b"data")
        with pytest.raises(RuntimeError, match="ffmpeg is not installed"):
            digue.transcribe("http://x", audio)

    def test_conversion_returns_bytes_not_file(self, tmp_path):
        # _convert_to_wav must work in memory: returns bytes, writes nothing
        audio = tmp_path / "tone.ogg"
        audio.write_bytes(b"x")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b"wav", stderr=b"")
            result = digue._convert_to_wav(audio)
            argv = mock_run.call_args[0][0]
        assert result == b"wav"
        assert "pipe:1" in argv
        assert not list(tmp_path.glob("digue-*.wav"))  # no temp files on disk


class TestSendAudioTokenTimestamps:
    @patch("digue._multipart_request", return_value="text")
    def test_always_sends_token_timestamps_false(self, mock_multipart, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"data")
        digue._send_audio("http://x", audio, "en", "text", 10)
        fields = mock_multipart.call_args[0][2]
        assert fields["token_timestamps"] == "false"

    @patch("digue._multipart_request", return_value="text")
    def test_always_sends_language_even_when_auto(self, mock_multipart, tmp_path):
        """Regression: the server's default language is "en" (server.cpp); omitting
        the field made every transcription English. "auto" must be sent as-is so
        whisper detects the language in the same pass."""
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"data")
        digue._send_audio("http://x", audio, "auto", "text", 10)
        fields = mock_multipart.call_args[0][2]
        assert fields["language"] == "auto"

    @patch("digue._multipart_request", return_value="text")
    def test_sends_language_when_fixed(self, mock_multipart, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"data")
        digue._send_audio("http://x", audio, "pt", "text", 10)
        fields = mock_multipart.call_args[0][2]
        assert fields["language"] == "pt"

    @patch("digue._multipart_request", return_value="text")
    def test_sends_original_filename(self, mock_multipart, tmp_path):
        audio = tmp_path / "tone_opus.ogg"
        audio.write_bytes(b"data")
        digue._send_audio("http://x", audio, "auto", "text", 10, audio_data=b"converted")
        filename = mock_multipart.call_args[1]["filename"]
        assert filename == "tone_opus.ogg"
        audio_data = mock_multipart.call_args[0][1]
        assert audio_data == b"converted"


# -- VTT simplification ------------------------------------------------------


class TestWrapCueLines:
    def test_short_text_returns_single_line(self):
        assert digue._wrap_cue_lines("hello world", 42, 2) == ["hello world"]

    def test_empty_text_returns_empty_list(self):
        assert digue._wrap_cue_lines("", 42, 2) == []
        assert digue._wrap_cue_lines("   ", 42, 2) == []

    def test_wraps_at_word_boundary(self):
        text = "uma frase bem comprida que passa do limite de caracteres"
        result = digue._wrap_cue_lines(text, 35, 2)
        assert len(result) == 2
        assert " ".join(result) == text
        assert len(result[0]) <= 35

    def test_overflow_preserves_all_words_on_last_line(self):
        text = (
            "uma frase bem comprida que passa do limite de quarenta e dois caracteres "
            "e continua por mais uma linha inteira de texto"
        )
        result = digue._wrap_cue_lines(text, 42, 2)
        assert len(result) == 2
        assert " ".join(result) == text


class TestStripVttTags:
    def test_removes_c_tags(self):
        text = "Hey<00:00:00.440><c> everyone,</c><00:00:00.960><c> I'm</c>"
        assert digue._strip_vtt_tags(text) == "Hey everyone, I'm"

    def test_plain_text_unchanged(self):
        assert digue._strip_vtt_tags("Hello world") == "Hello world"

    def test_empty_and_whitespace(self):
        assert digue._strip_vtt_tags("") == ""
        assert digue._strip_vtt_tags("   ") == ""


class TestSimplifyVtt:
    def test_standard_vtt(self):
        vtt = (
            "WEBVTT\n\n"
            "1\n00:00:00.320 --> 00:00:02.000\nHello everyone.\n\n"
            "2\n00:00:02.000 --> 00:00:05.000\nWelcome to the talk.\n\n"
        )
        result = digue.simplify_vtt(vtt)
        assert result == "[00:00:00] Hello everyone.\n[00:00:02] Welcome to the talk."

    def test_youtube_rolling_pattern(self):
        vtt = (
            "WEBVTT\nKind: captions\nLanguage: en\n\n"
            "00:00:00.320 --> 00:00:01.990 align:start position:0%\n \n"
            "Hey<00:00:00.440><c> everyone,</c><00:00:00.960><c> I'm</c><00:00:01.120><c> Ishaan</c>\n\n"
            "00:00:01.990 --> 00:00:02.000 align:start position:0%\nHey everyone, I'm Ishaan\n \n\n"
            "00:00:02.000 --> 00:00:03.710 align:start position:0%\nHey everyone, I'm Ishaan\n"
            "and<00:00:02.480><c> today</c><00:00:03.000><c> I'm</c><00:00:03.120><c> going</c>\n\n"
        )
        result = digue.simplify_vtt(vtt)
        lines = result.splitlines()
        assert len(lines) == 2
        assert lines[0] == "[00:00:00] Hey everyone, I'm Ishaan"
        assert lines[1] == "[00:00:02] and today I'm going"

    def test_skips_youtube_headers(self):
        vtt = "WEBVTT\nKind: captions\nLanguage: en\n\n00:00:00.000 --> 00:00:01.000\nHello\n"
        result = digue.simplify_vtt(vtt)
        assert "Kind:" not in result
        assert result == "[00:00:00] Hello"

    def test_strips_milliseconds(self):
        vtt = "WEBVTT\n\n1\n00:01:23.456 --> 00:01:25.000\nTest line\n"
        result = digue.simplify_vtt(vtt)
        assert result == "[00:01:23] Test line"


# -- Recording ----------------------------------------------------------------


class TestRecording:
    @patch("digue._pid_file")
    def test_is_recording_false_when_no_pid_file(self, mock_pid_file, tmp_path):
        mock_pid_file.return_value = tmp_path / "nonexistent.pid"
        assert digue.is_recording() is False

    @patch("digue._pid_alive", return_value=False)
    @patch("digue._pid_file")
    def test_is_recording_false_when_pid_dead(self, mock_pid_file, mock_alive, tmp_path):
        pid_file = tmp_path / "whisper.pid"
        pid_file.write_text("99999")
        mock_pid_file.return_value = pid_file
        assert digue.is_recording() is False

    @patch("digue._pid_file")
    def test_is_recording_false_when_pid_file_is_empty(self, mock_pid_file, tmp_path):
        """A truncated pid file (writer interrupted) must not crash the toggle."""
        pid_file = tmp_path / "digue.pid"
        pid_file.write_text("")
        mock_pid_file.return_value = pid_file
        assert digue.is_recording() is False


class TestTakeState:
    def make_state(self, tmp_path, **changes):
        values = {
            "version": digue.TAKE_STATE_VERSION,
            "take_id": "0123456789abcdef",
            "created_at_ns": 123,
            "state": "starting",
            "rec_file": tmp_path / "digue-recording.wav",
            "daemon_pid": 42,
            "daemon_starttime": 99,
            "recorder_pid": None,
            "recorder_starttime": None,
            "recoverer_pid": None,
            "recoverer_starttime": None,
        }
        values.update(changes)
        with patch("digue._runtime_dir", return_value=tmp_path):
            return digue.TakeState(**values)

    def test_round_trip_and_ordering(self, tmp_path):
        newer = self.make_state(tmp_path, take_id="ffffffffffffffff", created_at_ns=124)
        older = self.make_state(tmp_path)
        with patch("digue._runtime_dir", return_value=tmp_path):
            digue._write_take_state(newer)
            digue._write_take_state(older)
            assert digue._read_take_state(tmp_path / "digue-take-0123456789abcdef.json") == older
            assert digue._take_states() == [older, newer]

    @pytest.mark.parametrize(
        "changes",
        [
            {"version": 2},
            {"take_id": "not-hex"},
            {"created_at_ns": 0},
            {"state": "unknown"},
            {"rec_file": Path("/outside/digue-recording.wav")},
            {"rec_file": Path("PLACEHOLDER")},
            {"recorder_pid": 1},
            {"state": "recording"},
            {
                "state": "recording",
                "recorder_pid": 1,
                "recorder_starttime": 2,
                "recoverer_pid": 3,
                "recoverer_starttime": 4,
            },
            {"state": "recovering"},
        ],
    )
    def test_rejects_invalid_combinations(self, tmp_path, changes):
        if changes.get("rec_file") == Path("PLACEHOLDER"):
            changes["rec_file"] = tmp_path / "other.wav"
        with pytest.raises(ValueError):
            self.make_state(tmp_path, **changes)

    def test_accepts_each_valid_state(self, tmp_path):
        self.make_state(tmp_path)
        self.make_state(tmp_path, state="recording", recorder_pid=10, recorder_starttime=20)
        self.make_state(tmp_path, state="delivering", recorder_pid=10, recorder_starttime=20)
        self.make_state(tmp_path, state="recovering", recoverer_pid=30, recoverer_starttime=40)
        self.make_state(
            tmp_path,
            state="recovering",
            recorder_pid=10,
            recorder_starttime=20,
            recoverer_pid=30,
            recoverer_starttime=40,
        )

    def test_truncated_state_is_reported_and_never_removed(self, tmp_path, capsys):
        state_path = tmp_path / "digue-take-0123456789abcdef.json"
        state_path.write_text('{"version":')
        wav_path = tmp_path / "digue-recording.wav"
        wav_path.write_bytes(b"audio")

        with patch("digue._runtime_dir", return_value=tmp_path):
            assert digue._take_states() == []

        assert state_path.exists()
        assert wav_path.exists()
        assert str(state_path) in capsys.readouterr().err


class TestStateFilesAreWrittenAtomically:
    """Path.write_text truncates before writing: a concurrent toggle reading in
    between sees an empty file. For the daemon file that reads as "no daemon",
    so the toggle falls back to is_recording() and stops the live daemon's
    recorder; both then deliver the same take. The state must appear in one
    step (temp sibling + rename), never through a truncating open."""

    @staticmethod
    def _truncating_writes(monkeypatch):
        opened = []
        real_open = Path.open

        def spy_open(self, mode="r", *args, **kwargs):
            if "w" in mode:
                opened.append(self)
            return real_open(self, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", spy_open)
        return opened

    def test_daemon_state_is_never_truncated_in_place(self, tmp_path, monkeypatch):
        import os

        opened = self._truncating_writes(monkeypatch)
        daemon_file = tmp_path / "digue-daemon.pid"
        with patch("digue._runtime_dir", return_value=tmp_path):
            digue._write_daemon_state(os.getpid(), "recording")

        assert daemon_file not in opened
        assert daemon_file.read_text().split()[:2] == [str(os.getpid()), "recording"]

    @patch("digue._spawn_limit_watchdog")
    @patch("subprocess.Popen")
    def test_recorder_pid_file_is_never_truncated_in_place(self, mock_popen, mock_watchdog, tmp_path, monkeypatch):
        mock_popen.return_value = MagicMock(pid=777)
        opened = self._truncating_writes(monkeypatch)
        pid_file = tmp_path / "digue.pid"
        with patch("digue._runtime_dir", return_value=tmp_path):
            digue.start_recording(digue._default_config())

        assert pid_file not in opened
        assert pid_file.read_text() == "777"


class TestRecordingCommand:
    @patch("shutil.which")
    def test_auto_prefers_pw_record(self, mock_which, tmp_path):
        mock_which.return_value = "/usr/bin/pw-record"
        argv = digue.recording_command(tmp_path / "rec.wav", recorder="auto")
        assert argv[0] == "pw-record"

    @patch("shutil.which")
    def test_auto_falls_back_to_arecord(self, mock_which, tmp_path):
        mock_which.side_effect = lambda name: None if name == "pw-record" else "/usr/bin/arecord"
        argv = digue.recording_command(tmp_path / "rec.wav", recorder="auto")
        assert argv[0] == "arecord"

    def test_pw_record_argv(self, tmp_path):
        argv = digue.recording_command(tmp_path / "rec.wav", recorder="pw-record")
        assert argv[0] == "pw-record"
        assert "--rate" in argv and "16000" in argv
        assert argv[-1].endswith("rec.wav")

    def test_arecord_argv(self, tmp_path):
        argv = digue.recording_command(tmp_path / "rec.wav", recorder="arecord")
        assert argv[0] == "arecord"
        assert "16000" in argv

    def test_unknown_recorder_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="Unknown recorder"):
            digue.recording_command(tmp_path / "rec.wav", recorder="sox")


class TestStartRecording:
    @patch("digue._pid_file")
    @patch("subprocess.Popen")
    def test_returns_owned_recorder_handle(self, mock_popen, mock_pid_file, tmp_path):
        recorder = MagicMock(pid=1234)
        mock_popen.return_value = recorder
        mock_pid_file.return_value = tmp_path / "digue.pid"
        config = digue._default_config()
        config["dictate"]["max_duration"] = 0

        processes = digue.start_recording(config)

        assert processes.recorder is recorder
        assert processes.watchdog is None
        assert processes.rec_file is not None
        assert processes.rec_file.name.startswith("digue-")
        assert processes.rec_file.suffix == ".wav"
        assert mock_popen.call_args[1].get("start_new_session") is True

    @patch("digue._spawn_limit_watchdog")
    @patch("digue._pid_file")
    @patch("subprocess.Popen")
    def test_max_duration_returns_watchdog_handle(self, mock_popen, mock_pid_file, mock_watchdog, tmp_path):
        recorder = MagicMock(pid=777)
        watchdog = MagicMock(pid=778)
        mock_popen.return_value = recorder
        mock_watchdog.return_value = watchdog
        mock_pid_file.return_value = tmp_path / "digue.pid"
        config = digue._default_config()
        config["dictate"]["max_duration"] = 300

        processes = digue.start_recording(config)

        assert processes.recorder is recorder
        assert processes.watchdog is watchdog
        mock_watchdog.assert_called_once_with(777, 300)


class TestStartRecordingPublishesTakeState:
    """The take identity is published before the recorder exists and gains the
    recorder identity before the watchdog is spawned: publishing the state is
    two syscalls (~50 us) while spawning the watchdog is fork+exec (~10 ms), so
    a recorder with identity (which recovery knows how to stop) is the state
    that is exposed the soonest."""

    @patch("digue._spawn_limit_watchdog")
    @patch("digue._pid_file")
    @patch("subprocess.Popen")
    def test_publishes_starting_before_popen(self, mock_popen, mock_pid_file, mock_watchdog, tmp_path):
        import os

        states_at_popen = []
        recorder = MagicMock(pid=os.getpid())

        def fake_popen(*args, **kwargs):
            states_at_popen.extend(digue._take_states())
            return recorder

        mock_popen.side_effect = fake_popen
        mock_pid_file.return_value = tmp_path / "digue.pid"
        config = digue._default_config()
        config["dictate"]["max_duration"] = 0

        with patch("digue._runtime_dir", return_value=tmp_path):
            processes = digue.start_recording(config)

        assert [take.state for take in states_at_popen] == ["starting"]
        assert processes.take_id == states_at_popen[0].take_id
        assert processes.rec_file == states_at_popen[0].rec_file

    @patch("digue._pid_file")
    @patch("subprocess.Popen")
    def test_publishes_recording_with_identity_before_watchdog(self, mock_popen, mock_pid_file, tmp_path):
        import os

        states_at_watchdog = []
        recorder = MagicMock(pid=os.getpid())

        def fake_watchdog(pgid, max_duration):
            states_at_watchdog.extend(digue._take_states())
            return MagicMock(pid=9999)

        mock_popen.return_value = recorder
        mock_pid_file.return_value = tmp_path / "digue.pid"
        config = digue._default_config()
        config["dictate"]["max_duration"] = 300

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._spawn_limit_watchdog", side_effect=fake_watchdog),
        ):
            processes = digue.start_recording(config)

        assert processes.take_id is not None
        assert [take.state for take in states_at_watchdog] == ["recording"]
        take = states_at_watchdog[0]
        assert take.recorder_pid == recorder.pid
        assert take.recorder_starttime == int(digue._process_starttime(os.getpid()))

    @patch("digue._pid_file")
    @patch("subprocess.Popen", side_effect=FileNotFoundError("pw-record"))
    def test_popen_failure_removes_state(self, mock_popen, mock_pid_file, tmp_path):
        mock_pid_file.return_value = tmp_path / "digue.pid"
        config = digue._default_config()

        with patch("digue._runtime_dir", return_value=tmp_path), pytest.raises(FileNotFoundError):
            digue.start_recording(config)

        assert list(tmp_path.glob("digue-take-*.json")) == []

    def test_toggle_publishes_take_state_and_removes_it_after_delivery(self, tmp_path):
        import os

        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        config["dictate"]["max_duration"] = 0
        recorder = MagicMock(pid=os.getpid(), poll=lambda: 0)
        finish_take_ids = []

        def fake_finish(_config, rec_file, limit_reached=False, take_id=None):
            finish_take_ids.append(take_id)
            assert [take.state for take in digue._take_states()] == ["recording"]
            return digue.DeliveryResult(outcome="delivered", exit_code=0)

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.ensure_server"),
            patch("digue.is_server_running", return_value=True),
            patch("digue.is_recording", return_value=False),
            patch("subprocess.Popen", return_value=recorder),
            patch("digue._pid_file", return_value=tmp_path / "digue.pid"),
            patch("digue._wait_recorder_end_daemon", return_value="ended"),
            patch("digue.finish_dictation", side_effect=fake_finish),
            patch("digue.notify"),
            patch("digue.notify_close"),
            patch("signal.signal"),
        ):
            assert digue.dictate_toggle(config) == 0

        assert len(finish_take_ids) == 1 and finish_take_ids[0] is not None
        assert list(tmp_path.glob("digue-take-*.json")) == []


class TestOrphanStartingTake:
    """A daemon that died between publishing 'starting' and registering the
    recorder leaves an orphaned state. Recovery is conservative (no /proc fd
    scanning): recent states are left alone, a missing/empty WAV expires with
    its state, and a non-empty WAV is rescued (never transcribed/pasted
    automatically -- the recorder may still be writing)."""

    def make_starting_take(self, tmp_path, age_seconds, rec_file=None, daemon_pid=999999, daemon_starttime=1):
        import time

        with patch("digue._runtime_dir", return_value=tmp_path):
            take = digue.TakeState(
                version=digue.TAKE_STATE_VERSION,
                take_id="0123456789abcdef",
                created_at_ns=time.time_ns() - int(age_seconds * 1e9),
                state="starting",
                rec_file=rec_file or tmp_path / "digue-recording.wav",
                daemon_pid=daemon_pid,
                daemon_starttime=daemon_starttime,
            )
            digue._write_take_state(take)
        return take

    def test_expired_orphan_without_wav_is_removed(self, tmp_path):

        take = self.make_starting_take(tmp_path, age_seconds=digue.ORPHAN_STARTING_MIN_AGE_SECONDS + 1)
        config = digue._default_config()

        with patch("digue._runtime_dir", return_value=tmp_path), patch("digue._pid_alive", return_value=False):
            rescued = digue._expire_orphan_starting(config, take)

        assert rescued is None
        assert list(tmp_path.glob("digue-take-*.json")) == []

    def test_expired_orphan_with_wav_is_rescued(self, tmp_path):

        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        take = self.make_starting_take(tmp_path, age_seconds=digue.ORPHAN_STARTING_MIN_AGE_SECONDS + 1)
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")

        with patch("digue._runtime_dir", return_value=tmp_path), patch("digue._pid_alive", return_value=False):
            rescued = digue._expire_orphan_starting(config, take)

        assert rescued is not None and rescued.read_bytes() == b"audio"
        assert rescued.name.endswith("-0123456789abcdef.wav")
        assert not rec_file.exists()
        assert list(tmp_path.glob("digue-take-*.json")) == []

    def test_recent_orphan_is_left_alone(self, tmp_path):
        take = self.make_starting_take(tmp_path, age_seconds=1)
        config = digue._default_config()

        with patch("digue._runtime_dir", return_value=tmp_path), patch("digue._pid_alive", return_value=False):
            assert digue._expire_orphan_starting(config, take) is None

        assert len(list(tmp_path.glob("digue-take-*.json"))) == 1

    def test_orphan_with_alive_daemon_is_left_alone(self, tmp_path):

        take = self.make_starting_take(
            tmp_path, age_seconds=digue.ORPHAN_STARTING_MIN_AGE_SECONDS + 1, daemon_starttime=555
        )
        config = digue._default_config()

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._pid_alive", return_value=True),
            patch("digue._process_starttime", return_value="555"),
        ):
            assert digue._expire_orphan_starting(config, take) is None

        assert len(list(tmp_path.glob("digue-take-*.json"))) == 1

    def test_toggle_rescues_expired_orphan_starting_take(self, tmp_path):
        import os

        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        self.make_starting_take(tmp_path, age_seconds=digue.ORPHAN_STARTING_MIN_AGE_SECONDS + 1, rec_file=rec_file)
        recorder = MagicMock(pid=os.getpid(), poll=lambda: 0)

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._pid_alive", lambda pid: pid == os.getpid()),
            patch("digue.ensure_server"),
            patch("digue.is_server_running", return_value=True),
            patch("digue.is_recording", return_value=False),
            patch("subprocess.Popen", return_value=recorder),
            patch("digue._pid_file", return_value=tmp_path / "digue.pid"),
            patch("digue._wait_recorder_end_daemon", return_value="ended"),
            patch("digue.finish_dictation", return_value=digue.DeliveryResult(outcome="delivered", exit_code=0)),
            patch("digue.notify"),
            patch("digue.notify_close"),
            patch("signal.signal"),
        ):
            assert digue.dictate_toggle(config) == 0

        rescued = list((tmp_path / "audio").rglob("*-0123456789abcdef.wav"))
        assert len(rescued) == 1
        assert list(tmp_path.glob("digue-take-*.json")) == []


class TestOrphanTakeClaim:
    """Orphan takes (daemon provably dead, or a recovering take whose recoverer
    died) are claimed for recovery under the dictate lock: exactly one claimer
    wins, the transition to "recovering" carries the claimer's identity, and an
    old orphan never blocks stopping the current recording."""

    def make_take(self, tmp_path, take_id="0123456789abcdef", created_at_ns=100, **changes):
        import time

        values = {
            "version": digue.TAKE_STATE_VERSION,
            "take_id": take_id,
            "created_at_ns": created_at_ns or time.time_ns(),
            "state": "recording",
            "rec_file": tmp_path / "digue-recording.wav",
            "daemon_pid": 999999,
            "daemon_starttime": 1,
            "recorder_pid": 555,
            "recorder_starttime": 666,
        }
        values.update(changes)
        with patch("digue._runtime_dir", return_value=tmp_path):
            take = digue.TakeState(**values)
            digue._write_take_state(take)
        return take

    def test_claims_the_oldest_orphan_and_records_the_recoverer_identity(self, tmp_path):
        import os

        newer = self.make_take(tmp_path, take_id="ffffffffffffffff", created_at_ns=200)
        oldest = self.make_take(tmp_path)

        with patch("digue._runtime_dir", return_value=tmp_path):
            claimed = digue._claim_orphan_take()

        assert claimed is not None
        assert claimed.take_id == oldest.take_id
        assert claimed.state == "recovering"
        assert claimed.recorder_pid == 555
        assert claimed.recoverer_pid == os.getpid()
        assert claimed.recoverer_starttime == int(digue._process_starttime(os.getpid()))
        with patch("digue._runtime_dir", return_value=tmp_path):
            states = {take.take_id: take for take in digue._take_states()}
        assert states[oldest.take_id].state == "recovering"
        assert states[newer.take_id].state == "recording"

    def test_take_with_live_daemon_is_not_claimed(self, tmp_path):
        import os

        self.make_take(tmp_path, daemon_pid=os.getpid(), daemon_starttime=int(digue._process_starttime(os.getpid())))

        with patch("digue._runtime_dir", return_value=tmp_path):
            assert digue._claim_orphan_take() is None

    def test_recovering_take_with_live_recoverer_is_not_claimed(self, tmp_path):
        import os

        self.make_take(
            tmp_path,
            state="recovering",
            recorder_pid=None,
            recorder_starttime=None,
            recoverer_pid=os.getpid(),
            recoverer_starttime=int(digue._process_starttime(os.getpid())),
        )

        with patch("digue._runtime_dir", return_value=tmp_path):
            assert digue._claim_orphan_take() is None

    def test_recovering_take_with_dead_recoverer_is_claimed(self, tmp_path):
        self.make_take(
            tmp_path,
            state="recovering",
            recorder_pid=None,
            recorder_starttime=None,
            recoverer_pid=999999,
            recoverer_starttime=1,
        )

        with patch("digue._runtime_dir", return_value=tmp_path):
            claimed = digue._claim_orphan_take()

        assert claimed is not None
        assert claimed.state == "recovering"
        assert claimed.recoverer_pid == os.getpid()

    def test_concurrent_recoverers_exactly_one_claims(self, tmp_path):
        """Two recoverers racing for the same orphan: the dictate lock makes
        the second one see a recovering take with a live recoverer (the first
        claimer, this same process) and claim nothing."""

        self.make_take(tmp_path)
        first_in_lock = threading.Event()
        release_first = threading.Event()
        claimed = []

        def recover_first():
            with digue._dictate_lock():
                first_in_lock.set()
                assert release_first.wait(2)
                claimed.append(digue._claim_orphan_take())

        def recover_second():
            assert first_in_lock.wait(2)
            release_first.set()
            with digue._dictate_lock():
                claimed.append(digue._claim_orphan_take())

        with patch("digue._runtime_dir", return_value=tmp_path):
            thread_first = threading.Thread(target=recover_first)
            thread_second = threading.Thread(target=recover_second)
            thread_first.start()
            assert first_in_lock.wait(2)
            thread_second.start()
            release_first.set()
            thread_first.join(timeout=2)
            thread_second.join(timeout=2)

        assert not thread_first.is_alive()
        assert not thread_second.is_alive()
        assert claimed[0] is not None
        assert claimed[1] is None

    def test_toggle_signals_current_recording_and_leaves_orphan_alone(self, tmp_path):
        """A second press must stop the current recording even with an orphan
        take waiting: an old orphan never blocks the toggle."""
        daemon_file = tmp_path / "digue-daemon.pid"
        daemon_file.write_text("4242 recording 555")
        self.make_take(tmp_path)
        config = digue._default_config()

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._pid_alive", return_value=True),
            patch("digue._process_starttime", return_value="555"),
            patch("digue.notify"),
            patch("os.kill") as mock_kill,
        ):
            assert digue.dictate_toggle(config) == 0

        mock_kill.assert_called_once_with(4242, 15)
        with patch("digue._runtime_dir", return_value=tmp_path):
            states = digue._take_states()
        assert len(states) == 1 and states[0].state == "recording"

    def test_toggle_recovers_the_claimed_orphan_then_starts_a_new_take(self, tmp_path):
        import os

        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        self.make_take(tmp_path)
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        config["dictate"]["max_duration"] = 0
        recorder = MagicMock(pid=os.getpid(), poll=lambda: 0)
        finish_calls = []

        def fake_finish(_config, file, limit_reached=False, take_id=None):
            finish_calls.append((file, take_id))
            return digue.DeliveryResult(outcome="delivered", exit_code=0)

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._pid_alive", lambda pid: pid == os.getpid()),
            patch("digue.ensure_server"),
            patch("digue.is_server_running", return_value=True),
            patch("digue.stop_recording_pid", return_value=rec_file) as mock_stop,
            patch("digue.finish_dictation", side_effect=fake_finish),
            patch("subprocess.Popen", return_value=recorder),
            patch("digue._pid_file", return_value=tmp_path / "digue.pid"),
            patch("digue._wait_recorder_end_daemon", return_value="ended"),
            patch("digue.notify"),
            patch("digue.notify_close"),
            patch("signal.signal"),
        ):
            assert digue.dictate_toggle(config) == 0

        assert [finish_call[1] for finish_call in finish_calls] == ["0123456789abcdef", finish_calls[1][1]]
        assert finish_calls[0][0] == rec_file
        mock_stop.assert_called_once()
        assert list(tmp_path.glob("digue-take-*.json")) == []

    def test_toggle_during_slow_recovery_starts_a_new_take(self, tmp_path):
        """A recovering take with a live recoverer is like a delivering one:
        the toggle does not touch it and starts a new take."""
        import os

        self.make_take(
            tmp_path,
            state="recovering",
            recorder_pid=None,
            recorder_starttime=None,
            recoverer_pid=os.getpid(),
            recoverer_starttime=int(digue._process_starttime(os.getpid())),
        )
        config = digue._default_config()

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._pid_alive", lambda pid: pid == os.getpid()),
            patch("digue.ensure_server", side_effect=RuntimeError("abort startup")),
            patch("digue.notify"),
            patch("os.kill") as mock_kill,
        ):
            assert digue.dictate_toggle(config) == 1

        mock_kill.assert_not_called()
        with patch("digue._runtime_dir", return_value=tmp_path):
            states = digue._take_states()
        assert len(states) == 1
        assert states[0].state == "recovering"
        assert states[0].recoverer_pid == os.getpid()


class TestSpawnLimitWatchdog:
    @patch("digue._process_starttime", return_value="98765")
    @patch("subprocess.Popen")
    def test_watchdog_validates_process_identity_before_killing(self, mock_popen, mock_starttime):
        watchdog = MagicMock(pid=5000)
        mock_popen.return_value = watchdog

        result = digue._spawn_limit_watchdog(4242, 300)

        argv = mock_popen.call_args[0][0]
        script = argv[2]
        assert argv[-3:] == ["4242", "98765", str(300 + digue.WATCHDOG_GRACE_SECONDS)]
        assert "/proc/{pid}/stat" in script
        assert "os.killpg(pid, signal.SIGTERM)" in script
        assert result is watchdog
        assert mock_popen.call_args[1].get("start_new_session") is True

    @patch("digue._process_starttime", return_value="98765")
    @patch("subprocess.Popen")
    def test_watchdog_fires_only_after_the_daemon_limit(self, mock_popen, mock_starttime):
        """The daemon polls every 200ms and drifts; a watchdog sleeping exactly
        max_duration wins the race (measured: at 20s the recorder was already
        dead when the daemon checked), so the daemon saw "died" and never
        notified the limit. The watchdog is a safety net and must fire later."""
        digue._spawn_limit_watchdog(4242, 20)

        sleep_seconds = int(mock_popen.call_args[0][0][-1])
        assert sleep_seconds >= 20 + 2

    def test_process_starttime_parses_comm_with_spaces_and_parentheses(self, tmp_path):
        stat = tmp_path / "stat"
        stat.write_text("4242 (odd name) value) S " + " ".join(str(value) for value in range(4, 30)))

        assert digue._process_starttime(4242, stat_path=stat) == "22"


class TestWaitRecorderEndDaemon:
    def test_polls_owned_process_and_collects_spontaneous_exit(self):
        recorder = MagicMock()
        recorder.poll.side_effect = [None, 7]

        with patch("time.monotonic", side_effect=[0.0, 0.1, 0.3]), patch("time.sleep"):
            outcome = digue._wait_recorder_end_daemon(recorder, 300)

        assert outcome == "died"
        recorder.wait.assert_called_once_with(timeout=0)

    def test_sigterm_returns_manual(self):
        recorder = MagicMock()
        recorder.poll.return_value = None
        with patch("time.monotonic", side_effect=[0.0, 0.5]), patch("time.sleep"):
            digue._got_sigterm = True
            try:
                assert digue._wait_recorder_end_daemon(recorder, 300) == "manual"
            finally:
                digue._got_sigterm = False

    def test_exit_after_the_limit_counts_as_limit(self):
        """If the watchdog killed the recorder first, the outcome is still the
        duration limit, not a spontaneous death (the notification depends on it)."""
        recorder = MagicMock()
        recorder.poll.side_effect = [None, -15]

        with patch("time.monotonic", side_effect=[0.0, 0.1, 300.2]), patch("time.sleep"):
            outcome = digue._wait_recorder_end_daemon(recorder, 300)

        assert outcome == "limit"


class TestFinishOwnedRecorder:
    def test_spontaneously_exited_recorder_is_not_signaled_again(self, tmp_path):
        rec_file = tmp_path / "take.wav"
        rec_file.write_bytes(b"audio")
        recorder = MagicMock(pid=777)
        recorder.poll.return_value = 1

        with patch("digue.stop_recording_pid") as mock_stop:
            result = digue._finish_owned_recorder(recorder, rec_file)

        assert result == rec_file
        mock_stop.assert_not_called()


class TestCancelWatchdog:
    def test_normal_stop_terminates_and_collects_watchdog(self):
        watchdog = MagicMock()
        watchdog.poll.return_value = None

        digue._cancel_watchdog(watchdog)

        watchdog.terminate.assert_called_once_with()
        watchdog.wait.assert_called_once_with(timeout=5)

    def test_no_watchdog_is_a_noop(self):
        digue._cancel_watchdog(None)


class TestStopRecording:
    @patch("os.killpg")
    @patch("digue._group_alive", return_value=False)
    @patch("digue._pid_file")
    def test_kills_process_group_and_returns_file(self, mock_pid_file, mock_alive, mock_killpg, tmp_path):
        pid_file = tmp_path / "digue.pid"
        pid_file.write_text("4242")
        rec_file = tmp_path / "digue-20260904T120000.wav"
        rec_file.write_bytes(b"audio data")
        mock_pid_file.return_value = pid_file

        with patch.object(digue, "_recording_file_of", return_value=rec_file) as mock_source:
            result = digue.stop_recording()

        assert result == rec_file
        mock_source.assert_called_once_with(4242)
        mock_killpg.assert_called_once_with(4242, 15)
        assert not pid_file.exists()

    @patch("digue._pid_file")
    def test_returns_none_without_pid_file(self, mock_pid_file, tmp_path):
        mock_pid_file.return_value = tmp_path / "nope.pid"
        assert digue.stop_recording() is None

    def test_stop_returns_as_soon_as_the_group_is_gone(self, tmp_path):
        """The stop slept a fixed 0.5s before checking the group, and the
        exited-but-unreaped recorder (a zombie still answers signal 0) made it
        escalate to SIGKILL and sleep again: a full second on every dictation
        (measured 1.00s). A recorder that exits on SIGTERM within
        milliseconds must be reported gone within milliseconds."""
        import time

        recorder = subprocess.Popen(["sleep", "60"], start_new_session=True)
        rec_file = tmp_path / "take.wav"
        rec_file.write_bytes(b"audio")
        try:
            start = time.perf_counter()
            result = digue.stop_recording_pid(recorder.pid, rec_file)
            elapsed = time.perf_counter() - start
        finally:
            recorder.wait(timeout=5)

        assert result == rec_file
        assert elapsed < 0.3


class TestStopRecordingPidIdentity:
    """killpg assumes the pid is still the process-group leader; a recycled pid
    could belong to an unrelated process. Before each signal, both the /proc
    starttime and pgrp == pid are revalidated; on divergence the recorder is
    not signaled and the validated WAV is returned."""

    def make_rec_file(self, tmp_path):
        rec_file = tmp_path / "take.wav"
        rec_file.write_bytes(b"audio")
        return rec_file

    @patch("digue._group_alive", return_value=False)
    @patch("os.killpg")
    @patch("digue._process_starttime", return_value="999")
    def test_starttime_divergence_skips_killpg(self, mock_starttime, mock_killpg, mock_alive, tmp_path):
        rec_file = self.make_rec_file(tmp_path)

        result = digue.stop_recording_pid(4242, rec_file, expected_starttime="111")

        mock_killpg.assert_not_called()
        assert result == rec_file

    @patch("digue._group_alive", return_value=False)
    @patch("os.killpg")
    @patch("digue._process_pgrp", return_value=5151)
    @patch("digue._process_starttime", return_value="111")
    def test_pgrp_mismatch_skips_killpg(self, mock_starttime, mock_pgrp, mock_killpg, mock_alive, tmp_path):
        rec_file = self.make_rec_file(tmp_path)

        result = digue.stop_recording_pid(4242, rec_file, expected_starttime="111")

        mock_killpg.assert_not_called()
        assert result == rec_file

    @patch("digue._group_alive", return_value=False)
    @patch("os.killpg")
    @patch("digue._process_pgrp", return_value=4242)
    @patch("digue._process_starttime", return_value="111")
    def test_matching_identity_signals_the_group(self, mock_starttime, mock_pgrp, mock_killpg, mock_alive, tmp_path):
        rec_file = self.make_rec_file(tmp_path)

        result = digue.stop_recording_pid(4242, rec_file, expected_starttime="111")

        mock_killpg.assert_called_once_with(4242, 15)
        assert result == rec_file

    @patch("digue._group_alive", return_value=True)
    @patch("os.killpg")
    @patch("digue._process_pgrp", return_value=4242)
    @patch("digue._process_starttime", side_effect=["111", "999"])
    def test_identity_is_rechecked_before_sigkill(self, mock_starttime, mock_pgrp, mock_killpg, mock_alive, tmp_path):
        rec_file = self.make_rec_file(tmp_path)

        with patch("time.monotonic", side_effect=[0.0, 0.1, 0.6]):
            result = digue.stop_recording_pid(4242, rec_file, expected_starttime="111")

        assert list(mock_killpg.call_args_list) == [call(4242, 15)]
        assert result == rec_file


# -- Remote backend -----------------------------------------------------------


class TestContainerFailures:
    @pytest.mark.parametrize(
        ("function", "args"),
        [
            (digue.start_container, ["start", digue.CONTAINER_NAME]),
            (digue.stop_container, ["stop", digue.CONTAINER_NAME]),
            (digue.remove_container, ["rm", "-f", digue.CONTAINER_NAME]),
        ],
    )
    @patch("digue._docker_run")
    def test_container_commands_raise_on_failure(self, mock_docker, function, args):
        mock_docker.return_value = MagicMock(returncode=1, stderr="docker failed\n")

        with pytest.raises(RuntimeError, match="docker failed"):
            function()

        assert mock_docker.call_args.args[0] == args

    @patch("digue._wait_for_server")
    @patch("digue.start_container", side_effect=RuntimeError("start failed"))
    @patch("digue.container_status", return_value="exited")
    @patch("digue.is_server_running", return_value=False)
    def test_ensure_server_does_not_wait_after_start_failure(
        self, mock_running, mock_status, mock_start, mock_wait, capsys
    ):
        with pytest.raises(RuntimeError, match="start failed"):
            digue.ensure_server(digue._default_config(), silent=True)

        mock_wait.assert_not_called()

    @pytest.mark.parametrize("silent", (False, True))
    @patch("digue.notify_close")
    @patch("digue.notify")
    @patch("digue.create_container")
    @patch("digue.start_container")
    @patch("digue._wait_for_server", return_value=True)
    @patch("digue.container_status", return_value="running")
    @patch("digue.is_server_running", return_value=False)
    def test_ensure_server_waits_for_running_container(
        self,
        mock_running,
        mock_status,
        mock_wait,
        mock_start,
        mock_create,
        mock_notify,
        mock_notify_close,
        silent,
    ):
        config = digue._default_config()

        assert digue.ensure_server(config, silent=silent) is None

        mock_wait.assert_called_once_with(config, verbose=not silent)
        mock_start.assert_not_called()
        mock_create.assert_not_called()
        if silent:
            mock_notify.assert_not_called()
            mock_notify_close.assert_not_called()
        else:
            mock_notify.assert_called_once_with("Server starting...")
            mock_notify_close.assert_called_once_with()

    @patch("digue.notify")
    @patch("digue._wait_for_server", return_value=False)
    @patch("digue.container_status", return_value="running")
    @patch("digue.is_server_running", return_value=False)
    def test_ensure_server_reports_running_container_timeout(self, mock_running, mock_status, mock_wait, mock_notify):
        config = digue._default_config()

        assert digue.ensure_server(config) is None

        assert mock_notify.call_args_list == [
            call("Server starting..."),
            call("Server failed to start (see: docker logs digue)", timeout_ms=10000),
        ]

    @pytest.mark.parametrize(
        ("status", "expected_backend"),
        [("exited", None), (None, "cpu")],
    )
    @patch("digue.resolve_backend", return_value="cpu")
    @patch("digue.create_container")
    @patch("digue.start_container")
    @patch("digue._wait_for_server", return_value=True)
    @patch("digue.container_status")
    @patch("digue.is_server_running", return_value=False)
    def test_ensure_server_starts_or_creates_container(
        self,
        mock_running,
        mock_status,
        mock_wait,
        mock_start,
        mock_create,
        mock_resolve,
        status,
        expected_backend,
    ):
        mock_status.return_value = status
        config = digue._default_config()

        assert digue.ensure_server(config, silent=True) == expected_backend

        if status == "exited":
            mock_start.assert_called_once_with()
            mock_create.assert_not_called()
        else:
            mock_start.assert_not_called()
            mock_create.assert_called_once_with(config, "cpu")

    @patch("digue.notify")
    @patch("digue._wait_for_server")
    @patch("digue.container_status", return_value="paused")
    @patch("digue.is_server_running", return_value=False)
    def test_ensure_server_rejects_invalid_container_state(self, mock_running, mock_status, mock_wait, mock_notify):
        assert digue.ensure_server(digue._default_config()) is None

        mock_notify.assert_called_once_with("Container in unexpected state: paused", timeout_ms=5000)
        mock_wait.assert_not_called()

    @patch("digue.stop_container", side_effect=RuntimeError("stop failed"))
    def test_cmd_stop_does_not_report_false_success(self, mock_stop, capsys):
        assert digue.cmd_stop(MagicMock(), digue._default_config()) == 1
        assert "Error: stop failed" in capsys.readouterr().err

    @patch("digue.remove_container", side_effect=RuntimeError("remove failed"))
    def test_cmd_destroy_does_not_report_false_success(self, mock_remove, capsys):
        assert digue.cmd_destroy(MagicMock(), digue._default_config()) == 1
        assert "Error: remove failed" in capsys.readouterr().err

    @patch("digue.start_container", side_effect=RuntimeError("start failed"))
    @patch("digue.container_status", return_value="exited")
    @patch("digue.is_server_running", return_value=False)
    def test_cmd_start_does_not_report_false_success(self, mock_running, mock_status, mock_start, capsys):
        assert digue.cmd_start(MagicMock(), digue._default_config()) == 1
        assert "Error: start failed" in capsys.readouterr().err

    @patch("digue._wait_for_server", return_value=False)
    @patch("digue.start_container")
    @patch("digue.container_status", return_value="exited")
    @patch("digue.is_server_running", return_value=False)
    def test_cmd_start_timeout_hint_names_the_digue_container(
        self, mock_running, mock_status, mock_start, mock_wait, capsys
    ):
        assert digue.cmd_start(MagicMock(), digue._default_config()) == 1
        err = capsys.readouterr().err
        assert f"docker logs {digue.CONTAINER_NAME}" in err
        assert "whisper-server" not in err


class TestRemoteBackend:
    def test_resolve_backend_from_config(self):
        config = digue._default_config()
        config["server"]["backend"] = "remote"
        assert digue.resolve_backend(config) == "remote"

    @patch("digue.is_server_running", return_value=False)
    @patch("digue.container_status")
    def test_ensure_server_remote_never_touches_container(self, mock_status, mock_running):
        config = digue._default_config()
        config["server"]["backend"] = "remote"
        assert digue.ensure_server(config, silent=True) is None
        mock_status.assert_not_called()

    @patch("digue._docker_run")
    def test_create_container_remote_raises(self, mock_docker):
        config = digue._default_config()
        with pytest.raises(RuntimeError, match="remote"):
            digue.create_container(config, "remote")
        mock_docker.assert_not_called()

    def test_hint_remote_mentions_tunnel(self):
        config = digue._default_config()
        config["server"]["backend"] = "remote"
        hint = digue.server_not_running_hint(config)
        assert "ssh -NfL" in hint
        assert "8178" in hint

    def test_hint_local_suggests_start(self):
        config = digue._default_config()
        assert digue.server_not_running_hint(config) == "Run: digue start"

    @patch("digue.is_server_running", return_value=True)
    def test_cmd_status_remote(self, mock_running, capsys):
        config = digue._default_config()
        config["server"]["backend"] = "remote"
        assert digue.cmd_status(MagicMock(), config) == 0
        assert "remote" in capsys.readouterr().err

    @patch("digue.is_server_running", return_value=False)
    def test_cmd_status_remote_not_responding(self, mock_running, capsys):
        config = digue._default_config()
        config["server"]["backend"] = "remote"
        assert digue.cmd_status(MagicMock(), config) == 1


class TestRemoteHost:
    @patch("urllib.request.urlopen")
    @patch("digue.detect_backend")
    def test_server_probes_do_not_detect_auto_backend(self, mock_detect_backend, mock_urlopen):
        config = digue._default_config()

        digue.is_server_running(config)
        digue.server_url(config)
        digue.server_not_running_hint(config)

        mock_detect_backend.assert_not_called()

    def test_server_host_localhost_for_local_backends(self):
        config = digue._default_config()
        config["server"]["remote_host"] = "10.0.0.5"
        assert digue.server_host(config) == "127.0.0.1"

    def test_server_host_default_tunnel(self):
        config = digue._default_config()
        config["server"]["backend"] = "remote"
        assert digue.server_host(config) == "127.0.0.1"

    def test_server_host_remote_lan(self):
        config = digue._default_config()
        config["server"]["backend"] = "remote"
        config["server"]["remote_host"] = "10.0.0.5"
        assert digue.server_host(config) == "10.0.0.5"

    def test_server_url_uses_remote_host(self):
        config = digue._default_config()
        config["server"]["backend"] = "remote"
        config["server"]["remote_host"] = "desktop.lan"
        assert digue.server_url(config) == "http://desktop.lan:8178/inference"

    @patch("urllib.request.urlopen")
    def test_is_server_running_probes_remote_host(self, mock_urlopen):
        config = digue._default_config()
        config["server"]["backend"] = "remote"
        config["server"]["remote_host"] = "10.0.0.5"
        digue.is_server_running(config)
        url = mock_urlopen.call_args[0][0]
        assert url == "http://10.0.0.5:8178/"

    def test_hint_remote_lan_mentions_host(self):
        config = digue._default_config()
        config["server"]["backend"] = "remote"
        config["server"]["remote_host"] = "10.0.0.5"
        hint = digue.server_not_running_hint(config)
        assert "10.0.0.5" in hint
        assert "ssh -NfL" not in hint

    def test_hint_remote_tunnel_keeps_ssh(self):
        config = digue._default_config()
        config["server"]["backend"] = "remote"
        hint = digue.server_not_running_hint(config)
        assert "ssh -NfL" in hint

    @patch("digue.is_server_running", return_value=False)
    def test_cmd_status_remote_shows_host(self, mock_running, capsys):
        config = digue._default_config()
        config["server"]["backend"] = "remote"
        config["server"]["remote_host"] = "10.0.0.5"
        digue.cmd_status(MagicMock(), config)
        assert "10.0.0.5:8178" in capsys.readouterr().err


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

    def test_unknown_keys_in_host_section_are_ignored(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent("""\
            [host.thinkpad.server]
            no-such-key = true
        """)
        )
        with patch("socket.gethostname", return_value="thinkpad"):
            config = digue.load_config(config_path)
        assert config["server"]["backend"] == "auto"

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

    def test_example_config_is_valid_toml(self, tmp_path):
        import tomllib

        example = digue._config_example()
        parsed = tomllib.loads(example)
        assert "models" in parsed  # sections exist; all keys stay commented


class TestCreateParser:
    def test_all_subcommands_parse(self):
        parser = digue.create_parser()
        for cmd in ("detect", "download", "start", "stop", "destroy", "status", "dictate", "config", "benchmark"):
            args = parser.parse_args([cmd])
            assert args.command == cmd

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


# -- Command handlers ---------------------------------------------------------


class TestCmdDetect:
    @patch("digue.detect_backend", return_value="nvidia")
    def test_prints_backend(self, mock_detect, capsys):
        result = digue.cmd_detect(MagicMock())
        assert result == 0
        assert capsys.readouterr().out.strip() == "nvidia"


class TestCmdConfig:
    def test_prints_json(self, capsys):
        config = digue._default_config()
        result = digue.cmd_config(MagicMock(output_format="json"), config)
        assert result == 0
        output = json.loads(capsys.readouterr().out)
        assert output["server"]["port"] == 8178
        assert "nvidia" in output["models"]


class TestRuntimeIsolation:
    def test_state_paths_use_isolated_runtime_dir(self):
        runtime_dir = Path(os.environ["XDG_RUNTIME_DIR"])

        assert digue._runtime_dir() == runtime_dir
        assert digue._pid_file().parent == runtime_dir
        assert digue._daemon_pid_file().parent == runtime_dir
        assert digue._recorder_pid_file(123).parent == runtime_dir
        with digue._dictate_lock():
            assert (runtime_dir / "digue.lock").exists()

    def test_toggle_does_not_read_state_outside_isolated_runtime(self, tmp_path):
        """A stale recorder state outside the isolated runtime must not be recovered."""
        sentinel_dir = tmp_path / "sentinel"
        sentinel_dir.mkdir()
        (sentinel_dir / "digue-daemon.pid").write_text("4242 recording 555")
        config = digue._default_config()

        with (
            patch("digue._process_starttime", return_value="555"),
            patch("digue.ensure_server", side_effect=RuntimeError("stop after state lookup")),
            patch("digue.is_recording", return_value=False),
            patch("digue.notify"),
            patch("os.kill") as mock_kill,
        ):
            assert digue.dictate_toggle(config) == 1

        mock_kill.assert_not_called()


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


class TestConfigTemplateSync:
    def test_readme_config_block_matches_config_init_template(self):
        """Regression: the README config example must stay in sync with `digue config init`."""
        readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
        section = readme.split("## Configuration", 1)[1]
        fence_start = section.index("```toml") + len("```toml\n")
        fence_end = section.index("```", fence_start)
        readme_block = section[fence_start:fence_end].strip()
        assert readme_block == digue._config_example().strip()


class TestDetectLanguage:
    def test_detect_language_uses_verbose_json_payload(self, tmp_path):
        """The server only reports the language in verbose_json (plain json returns
        {"text":""} even with detect_language=true)."""
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"data")
        payload = {
            "detected_language": "portuguese",
            "detected_language_probability": 0.999,
            "language_probabilities": {},
        }
        with (
            patch("digue._multipart_request", return_value=json.dumps(payload)) as mock_request,
            patch("digue.NATIVE_FORMATS", new=frozenset({".wav"})),
        ):
            result = digue.detect_language("http://x", audio, timeout=10)
        assert result == "pt"
        fields = mock_request.call_args[0][2]
        assert fields["detect_language"] == "true"
        assert fields["response_format"] == "verbose_json"

    def test_detect_language_accepts_code_from_server(self, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"data")
        payload = {"detected_language": "pt", "detected_language_probability": 0.9, "language_probabilities": {}}
        with patch("digue._multipart_request", return_value=json.dumps(payload)):
            assert digue.detect_language("http://x", audio, timeout=10) == "pt"

    def test_detect_language_converts_unsupported_format_upfront(self, tmp_path):
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"m4a!")
        payload = {"detected_language": "en", "detected_language_probability": 0.9, "language_probabilities": {}}
        with (
            patch("digue._multipart_request", return_value=json.dumps(payload)) as mock_request,
            patch("digue._convert_to_wav", return_value=b"wav") as mock_convert,
        ):
            assert digue.detect_language("http://x", audio, timeout=10) == "en"
        mock_convert.assert_called_once()
        assert mock_request.call_args[0][1] == b"wav"

    def test_detect_language_retries_with_ffmpeg_after_400(self, tmp_path):
        import urllib.error

        audio = tmp_path / "a.ogg"
        audio.write_bytes(b"ogg!")
        payload = {"detected_language": "en", "detected_language_probability": 0.9, "language_probabilities": {}}
        with (
            patch(
                "digue._multipart_request",
                side_effect=[urllib.error.HTTPError("url", 400, "Bad", {}, None), json.dumps(payload)],
            ) as mock_request,
            patch("digue._convert_to_wav", return_value=b"wav"),
        ):
            assert digue.detect_language("http://x", audio, timeout=10) == "en"
        assert mock_request.call_count == 2

    def test_language_probabilities_via_verbose_json(self, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"data")
        payload = {
            "detected_language": "Portuguese",
            "detected_language_probability": 0.999,
            "language_probabilities": {"pt": 0.999, "en": 0.0005},
        }
        with patch("digue._multipart_request", return_value=json.dumps(payload)):
            probs = digue.language_probabilities("http://x", audio, timeout=10)
        assert probs["detected"] == ("pt", 0.999)
        assert probs["all"] == {"pt": 0.999, "en": 0.0005}


class TestCmdDetectLanguage:
    @patch("digue.detect_language", return_value="pt")
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    def test_prints_language_code(self, mock_running, mock_ensure, mock_detect, tmp_path, capsys):
        config = digue._default_config()
        args = MagicMock()
        args.audio = tmp_path / "a.wav"
        args.audio.write_bytes(b"data")
        args.json = False

        result = digue.cmd_detect_language(args, config)

        assert result == 0
        assert capsys.readouterr().out.strip() == "pt"

    @patch("digue.language_probabilities", return_value={"detected": ("pt", 0.999), "all": {"pt": 0.999}})
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    def test_json_output_has_detected_and_all(self, mock_running, mock_ensure, mock_probs, tmp_path, capsys):
        config = digue._default_config()
        args = MagicMock()
        args.audio = tmp_path / "a.wav"
        args.audio.write_bytes(b"data")
        args.json = True

        result = digue.cmd_detect_language(args, config)

        assert result == 0
        output = json.loads(capsys.readouterr().out)
        assert output["language"] == "pt"
        assert output["probability"] == 0.999
        assert output["all"] == {"pt": 0.999}

    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    def test_missing_file_gives_clear_error(self, mock_running, mock_ensure, tmp_path, capsys):
        config = digue._default_config()
        args = MagicMock()
        args.audio = tmp_path / "nope.wav"
        args.json = False

        result = digue.cmd_detect_language(args, config)

        assert result == 1
        assert "not found" in capsys.readouterr().err
        mock_ensure.assert_not_called()
        mock_running.assert_not_called()


class TestDetectLanguageVerbose:
    @patch("digue._multipart_request", side_effect=RuntimeError("stop"))
    def test_conversion_message_respects_verbose_false(self, mock_request, tmp_path, capsys):
        """Regression: the ffmpeg-conversion notice is progress output; without
        --verbose the stderr stays clean."""
        audio = tmp_path / "a.ogg"
        audio.write_bytes(b"ogg")
        with patch("digue._convert_to_wav", return_value=b"wav"), pytest.raises(RuntimeError):
            digue.detect_language("http://x", audio, timeout=10, verbose=False)
        assert "ffmpeg" not in capsys.readouterr().err

    @patch("digue._multipart_request", side_effect=RuntimeError("stop"))
    def test_conversion_message_shown_with_verbose_true(self, mock_request, tmp_path, capsys):
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"m4a")
        with patch("digue._convert_to_wav", return_value=b"wav"), pytest.raises(RuntimeError):
            digue.detect_language("http://x", audio, timeout=10, verbose=True)
        assert "ffmpeg" in capsys.readouterr().err

    @patch("digue._multipart_request", side_effect=RuntimeError("stop"))
    def test_language_probabilities_message_respects_verbose_false(self, mock_request, tmp_path, capsys):
        audio = tmp_path / "a.ogg"
        audio.write_bytes(b"ogg")
        with patch("digue._convert_to_wav", return_value=b"wav"), pytest.raises(RuntimeError):
            digue.language_probabilities("http://x", audio, timeout=10, verbose=False)
        assert "ffmpeg" not in capsys.readouterr().err

    def test_cmd_detect_language_passes_verbose(self, tmp_path):
        config = digue._default_config()
        args = MagicMock()
        args.audio = tmp_path / "a.wav"
        args.audio.write_bytes(b"data")
        args.json = False
        args.verbose = False
        with (
            patch("digue.detect_language", return_value="pt") as mock_detect,
            patch("digue.ensure_server"),
            patch("digue.is_server_running", return_value=True),
        ):
            assert digue.cmd_detect_language(args, config) == 0
        assert mock_detect.call_args[1]["verbose"] is False


class TestDictateDaemon:
    def test_concurrent_first_toggles_start_only_one_recorder(self, tmp_path):
        """A second toggle must observe the first toggle's startup reservation."""
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        first_in_startup = threading.Event()
        release_first = threading.Event()
        ensure_calls = 0
        results = []
        errors = []

        def ensure_server(_config):
            nonlocal ensure_calls
            ensure_calls += 1
            if ensure_calls == 1:
                first_in_startup.set()
                assert release_first.wait(timeout=2)

        def run_toggle():
            try:
                results.append(digue.dictate_toggle(config))
            except BaseException as exc:
                errors.append(exc)

        recorder = MagicMock(pid=777, poll=lambda: 0)
        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.ensure_server", side_effect=ensure_server),
            patch("digue.is_server_running", return_value=True),
            patch("digue.is_recording", return_value=False),
            patch(
                "digue.start_recording",
                return_value=digue.RecordingProcesses(recorder=recorder, watchdog=None),
            ) as mock_start,
            patch("digue._recording_file_of", return_value=tmp_path / "take.wav"),
            patch("digue._wait_recorder_end_daemon", return_value="ended"),
            patch("digue.finish_dictation", return_value=digue.DeliveryResult(outcome="delivered", exit_code=0)),
            patch("signal.signal"),
            patch("os.kill") as mock_kill,
        ):
            first = threading.Thread(target=run_toggle)
            first.start()
            assert first_in_startup.wait(timeout=2)
            second = threading.Thread(target=run_toggle)
            second.start()
            second.join(timeout=2)
            assert not second.is_alive()
            release_first.set()
            first.join(timeout=2)
            assert not first.is_alive()

        assert errors == []
        assert results == [0, 0]
        mock_start.assert_called_once_with(config)
        assert all(call.args[1] == 0 for call in mock_kill.call_args_list)

    def test_dictate_toggle_works_when_stderr_has_no_isatty(self, tmp_path, monkeypatch):
        class NonFileStderr:
            def write(self, _message):
                pass

            def flush(self):
                pass

        monkeypatch.setattr(sys, "stderr", NonFileStderr())
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        recorder = MagicMock(pid=777, poll=lambda: 0)
        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.ensure_server"),
            patch("digue.is_server_running", return_value=True),
            patch("digue.is_recording", return_value=False),
            patch("digue.start_recording", return_value=digue.RecordingProcesses(recorder=recorder, watchdog=None)),
            patch("digue._recording_file_of", return_value=tmp_path / "take.wav"),
            patch("digue._wait_recorder_end_daemon", return_value="ended"),
            patch("digue.finish_dictation", return_value=digue.DeliveryResult(outcome="delivered", exit_code=0)),
            patch("digue.notify"),
            patch("signal.signal"),
        ):
            assert digue.dictate_toggle(config) == 0

    def test_startup_failure_clears_own_reservation(self, tmp_path):
        config = digue._default_config()
        daemon_file = tmp_path / "digue-daemon.pid"

        def fail_after_reservation(_config):
            import os

            assert daemon_file.read_text() == f"{os.getpid()} starting"
            raise RuntimeError("boom")

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.ensure_server"),
            patch("digue.is_server_running", return_value=True),
            patch("digue.is_recording", return_value=False),
            patch("digue.start_recording", side_effect=fail_after_reservation),
            patch("digue.notify"),
            patch("signal.signal"),
        ):
            assert digue.dictate_toggle(config) == 1

        assert not daemon_file.exists()

    def test_startup_failure_does_not_clear_new_daemon_state(self, tmp_path):
        config = digue._default_config()
        daemon_file = tmp_path / "digue-daemon.pid"

        def replace_reservation_then_fail(_config):
            import os

            assert daemon_file.read_text().split()[:2] == [str(os.getpid()), "starting"]
            daemon_file.write_text("4242 recording 1")
            raise RuntimeError("boom")

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.ensure_server", side_effect=replace_reservation_then_fail),
            patch("digue.is_recording", return_value=False),
            patch("digue.notify"),
        ):
            assert digue.dictate_toggle(config) == 1

        assert daemon_file.read_text() == "4242 recording 1"

    @patch("digue.stop_recording", return_value=None)
    @patch("digue.is_recording", return_value=True)
    def test_second_toggle_signals_daemon_and_exits_fast(self, mock_recording, mock_stop, tmp_path, capsys):
        """The second dictate sends SIGTERM to the daemon and exits immediately;
        the daemon (not this process) runs the transcription flow."""
        daemon_pid = tmp_path / "digue-daemon.pid"
        daemon_pid.write_text("4242 recording 555")

        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        with (
            patch("digue._daemon_pid_file", return_value=daemon_pid),
            patch("digue._pid_alive", return_value=True),
            patch("digue._process_starttime", return_value="555"),
            patch("os.kill") as mock_kill,
            patch("digue.stop_recording") as mock_stop_in_toggle,
        ):
            result = digue.dictate_toggle(config)

        assert result == 0
        mock_kill.assert_called_once_with(4242, 15)
        # the toggle must NOT run the transcription logic itself
        mock_stop_in_toggle.assert_not_called()

    def test_daemon_pid_file_removed_when_daemon_dead(self, tmp_path):
        """A stale daemon pid file (crashed daemon) must not block a new recording."""
        daemon_pid = tmp_path / "digue-daemon.pid"
        daemon_pid.write_text("4242 recording 555")
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        with (
            patch("digue._daemon_pid_file", return_value=daemon_pid),
            patch("digue._pid_alive", return_value=False),
            patch("digue.is_recording", return_value=False),
            patch("digue.ensure_server", return_value=None),
            patch("digue.is_server_running", return_value=True),
            patch(
                "digue.start_recording",
                return_value=digue.RecordingProcesses(recorder=MagicMock(pid=777, poll=lambda: 0), watchdog=None),
            ) as mock_start,
            patch("digue.stop_recording", return_value=None),
        ):
            result = digue.dictate_toggle(config)

        assert result == 1
        assert not daemon_pid.exists()
        mock_start.assert_called_once()

    def test_remove_daemon_state_reads_and_unlinks_while_locked(self, tmp_path):
        daemon_file = tmp_path / "digue-daemon.pid"
        daemon_file.write_text("99 delivering 555")
        lock_active = False
        operations = []

        @contextlib.contextmanager
        def tracked_lock():
            nonlocal lock_active
            lock_active = True
            operations.append("enter")
            try:
                yield
            finally:
                operations.append("exit")
                lock_active = False

        original_read_text = Path.read_text
        original_unlink = Path.unlink

        def tracked_read_text(path, *args, **kwargs):
            assert lock_active
            operations.append("read")
            return original_read_text(path, *args, **kwargs)

        def tracked_unlink(path, *args, **kwargs):
            assert lock_active
            operations.append("unlink")
            return original_unlink(path, *args, **kwargs)

        with (
            patch("digue._dictate_lock", side_effect=tracked_lock),
            patch("digue._daemon_pid_file", return_value=daemon_file),
            patch.object(Path, "read_text", tracked_read_text),
            patch.object(Path, "unlink", tracked_unlink),
        ):
            assert digue._remove_daemon_state(99) is True

        assert operations == ["enter", "read", "unlink", "exit"]

    def test_remove_daemon_state_serializes_with_state_publication(self, tmp_path):
        daemon_file = tmp_path / "digue-daemon.pid"
        daemon_file.write_text("99 delivering 555")
        remover_locked = threading.Event()
        release_remover = threading.Event()
        publisher_started = threading.Event()
        errors = []

        original_read_text = Path.read_text

        def pause_after_lock(path, *args, **kwargs):
            result = original_read_text(path, *args, **kwargs)
            if path == daemon_file:
                remover_locked.set()
                assert release_remover.wait(1)
            return result

        def remove_state():
            try:
                assert digue._remove_daemon_state(99) is True
            except BaseException as exc:
                errors.append(exc)

        def publish_state():
            try:
                assert remover_locked.wait(1)
                publisher_started.set()
                with digue._dictate_lock():
                    digue._write_daemon_state(100, "recording")
            except BaseException as exc:
                errors.append(exc)

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._process_starttime", return_value="777"),
            patch.object(Path, "read_text", pause_after_lock),
        ):
            remover = threading.Thread(target=remove_state)
            publisher = threading.Thread(target=publish_state)
            remover.start()
            assert remover_locked.wait(1)
            publisher.start()
            assert publisher_started.wait(1)
            publisher.join(timeout=0.05)
            assert publisher.is_alive()
            release_remover.set()
            remover.join(timeout=1)
            publisher.join(timeout=1)
            assert not remover.is_alive()
            assert not publisher.is_alive()

        assert errors == []
        assert daemon_file.read_text() == "100 recording 777"

    def test_old_daemon_does_not_remove_new_daemon_state(self, tmp_path):
        """A delivering take may finish while a newer take is recording."""
        daemon_file = tmp_path / "digue-daemon.pid"
        daemon_file.write_text("100 recording 555")

        with patch("digue._daemon_pid_file", return_value=daemon_file):
            removed = digue._remove_daemon_state(99)

        assert removed is False
        assert daemon_file.read_text() == "100 recording 555"

    def test_remove_daemon_state_tolerates_an_empty_file(self, tmp_path):
        daemon_file = tmp_path / "digue-daemon.pid"
        daemon_file.write_text("")

        with patch("digue._daemon_pid_file", return_value=daemon_file):
            assert digue._remove_daemon_state(99) is False

        assert daemon_file.exists()

    def test_daemon_removes_only_its_own_state(self, tmp_path):
        daemon_file = tmp_path / "digue-daemon.pid"
        daemon_file.write_text("99 delivering 555")

        with patch("digue._daemon_pid_file", return_value=daemon_file):
            removed = digue._remove_daemon_state(99)

        assert removed is True
        assert not daemon_file.exists()

    def _toggle_against_state(self, tmp_path, state_text):
        """Runs a toggle against a daemon state file; startup is aborted right
        after the state check so only the signaling decision matters."""
        daemon_file = tmp_path / "digue-daemon.pid"
        daemon_file.write_text(state_text)
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.is_recording", return_value=False),
            patch("digue.ensure_server", side_effect=RuntimeError("abort startup")),
            patch("digue.notify"),
            patch("os.kill") as mock_kill,
        ):
            digue.dictate_toggle(config)
        return [call.args for call in mock_kill.call_args_list if call.args[1] == 15]

    def test_toggle_does_not_signal_a_recycled_daemon_pid(self, tmp_path):
        """The daemon file may outlive its daemon (SIGKILL, crash before the
        try/finally). If the pid was reused by an unrelated process of the
        same user, a toggle must not SIGTERM it: liveness alone is not identity.
        The test process itself plays the unrelated process."""
        import os

        signals = self._toggle_against_state(tmp_path, f"{os.getpid()} recording")

        assert signals == []

    def test_toggle_does_not_signal_when_starttime_differs(self, tmp_path):
        import os

        signals = self._toggle_against_state(tmp_path, f"{os.getpid()} recording 1")

        assert signals == []

    def test_toggle_signals_the_daemon_whose_identity_matches(self, tmp_path):
        import os

        starttime = digue._process_starttime(os.getpid())
        signals = self._toggle_against_state(tmp_path, f"{os.getpid()} recording {starttime}")

        assert signals == [(os.getpid(), 15)]

    def test_toggle_during_startup_tells_the_user_instead_of_silently_exiting(self, tmp_path):
        """ensure_server can take minutes on first use (image pull, model
        download). A second press during that window must give feedback."""
        import os

        daemon_file = tmp_path / "digue-daemon.pid"
        daemon_file.write_text(f"{os.getpid()} starting {digue._process_starttime(os.getpid())}")
        config = digue._default_config()
        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.notify") as mock_notify,
            patch("os.kill") as mock_kill,
        ):
            result = digue.dictate_toggle(config)

        assert result == 0
        assert [call.args for call in mock_kill.call_args_list if call.args[1] != 0] == []
        assert mock_notify.call_count == 1
        assert "starting" in mock_notify.call_args.args[0].lower()
        assert mock_notify.call_args.kwargs.get("timeout_ms", 0) > 0

    def test_daemon_state_records_the_process_starttime(self, tmp_path):
        import os

        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        seen = []

        def capture_state(_config):
            seen.append((tmp_path / "digue-daemon.pid").read_text())
            raise RuntimeError("abort startup")

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.is_recording", return_value=False),
            patch("digue.ensure_server", side_effect=capture_state),
            patch("digue.notify"),
        ):
            digue.dictate_toggle(config)

        assert seen == [f"{os.getpid()} starting {digue._process_starttime(os.getpid())}"]

    def test_recording_filename_is_unique_within_same_second(self):
        with patch("digue.now_timestamp", return_value="20260904-120000"):
            first = digue._rec_file()
            second = digue._rec_file()

        assert first != second


class TestDictateInterrupt:
    @patch("digue.stop_recording_pid", return_value=None)
    @patch("digue.is_recording", side_effect=[False, True, True])
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.start_recording")
    def test_sigint_during_daemon_wait_stops_and_delivers(
        self, mock_start, mock_running, mock_ensure, mock_recording, mock_stop_pid, tmp_path, capsys
    ):
        """Ctrl+c (SIGINT) in a terminal dictation must stop the recording and
        deliver the take, not discard it (the global KeyboardInterrupt handler
        must not win: the daemon installs its own SIGINT handler)."""
        import signal

        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        handlers = []
        recorder = MagicMock(pid=777)
        recorder.poll.return_value = None
        mock_start.return_value = digue.RecordingProcesses(recorder=recorder, watchdog=None)

        def fake_wait(_recorder, _limit):
            # simulate Ctrl+c arriving during the wait
            digue._on_sigint(signal.SIGINT, None)
            return "interrupted"

        with (
            patch("digue._recording_file_of", return_value=tmp_path / "take.wav"),
            patch("digue._wait_recorder_end_daemon", side_effect=fake_wait),
            patch(
                "digue.finish_dictation", return_value=digue.DeliveryResult(outcome="delivered", exit_code=0)
            ) as mock_finish,
            patch("signal.signal", side_effect=lambda sig, handler: handlers.append((sig, handler))),
        ):
            digue._pid_file().write_text("777")  # the fake recorder's pid
            result = digue.dictate_toggle(config)

        assert result == 0
        mock_stop_pid.assert_called_once()
        mock_finish.assert_called_once()
        # the daemon must have installed its own SIGINT handler
        installed = dict(handlers)
        assert signal.SIGINT in installed
        assert installed[signal.SIGINT] is digue._on_sigint
        assert not digue._daemon_pid_file().exists()

    def test_wait_recorder_end_interrupted_outcome(self):
        recorder = MagicMock()
        recorder.poll.return_value = None
        with patch("time.monotonic", side_effect=[0.0, 0.5]), patch("time.sleep"):
            digue._on_sigint(2, None)
            try:
                assert digue._wait_recorder_end_daemon(recorder, 300) == "interrupted"
            finally:
                digue._got_sigint = False


class TestCmdDoctor:
    def test_prints_resolved_config_without_crashing(self, capsys):
        config = digue._default_config()
        with (
            patch("digue.image_exists", return_value=False),
            patch("shutil.which", return_value=None),
        ):
            result = digue.cmd_doctor(MagicMock(), config)
        assert result == 0
        err = capsys.readouterr().err
        assert "Language: auto" in err

    def test_lists_both_supported_recorders(self, capsys):
        """arecord is the documented fallback recorder; doctor only checked pw-record."""
        config = digue._default_config()
        with (
            patch("digue.image_exists", return_value=False),
            patch("shutil.which", return_value=None),
        ):
            digue.cmd_doctor(MagicMock(), config)
        err = capsys.readouterr().err
        assert "] pw-record:" in err
        assert "] arecord:" in err


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


class TestDetectDisplayServer:
    def test_wayland(self, monkeypatch):
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.delenv("DISPLAY", raising=False)
        assert digue.detect_display_server() == "wayland"

    def test_x11(self, monkeypatch):
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")
        assert digue.detect_display_server() == "x11"

    def test_wayland_takes_priority(self, monkeypatch):
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setenv("DISPLAY", ":0")
        assert digue.detect_display_server() == "wayland"

    def test_none_when_no_display(self, monkeypatch):
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        assert digue.detect_display_server() is None


class TestSendText:
    def test_raises_when_no_display(self, monkeypatch):
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        with pytest.raises(RuntimeError, match="No DISPLAY"):
            digue.send_text("hello", display_server="auto")

    @patch("subprocess.run")
    def test_x11_uses_xclip(self, mock_run):
        digue.send_text("hello", display_server="x11")
        cmds = [call[0][0] for call in mock_run.call_args_list]
        assert cmds[0][0] == "xclip"
        assert cmds[1][0] == "xdotool"

    @patch("subprocess.run")
    def test_wayland_uses_wl_copy(self, mock_run):
        digue.send_text("hello", display_server="wayland")
        cmds = [call[0][0] for call in mock_run.call_args_list]
        assert cmds[0][0] == "wl-copy"
        assert cmds[1][0] == "wtype"

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_missing_tool_gives_install_hint(self, mock_run):
        with pytest.raises(RuntimeError, match="sudo apt install"):
            digue.send_text("hello", display_server="x11")

    @patch("subprocess.run")
    def test_type_x11_uses_xdotool_type_reading_stdin(self, mock_run):
        digue.send_text("olá mundo", display_server="x11", input_mode="type")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["xdotool", "type", "--clearmodifiers", "--file", "-"]
        assert mock_run.call_args[1]["input"] == "olá mundo".encode()

    @patch("subprocess.run")
    def test_type_wayland_uses_wtype_reading_stdin(self, mock_run):
        # wtype has no --no-newline flag: any unknown -option makes it fail with
        # "Unknown parameter" (checked main.c of atx/wtype, the Debian package).
        digue.send_text("olá mundo", display_server="wayland", input_mode="type")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["wtype", "-"]
        assert "--no-newline" not in cmd
        assert mock_run.call_args[1]["input"] == "olá mundo".encode()

    @pytest.mark.parametrize("display_server", ["x11", "wayland"])
    @patch("subprocess.run")
    def test_type_text_starting_with_dash_never_lands_in_argv(self, mock_run, display_server):
        digue.send_text("- item one", display_server=display_server, input_mode="type")
        cmd = mock_run.call_args[0][0]
        assert "- item one" not in cmd
        assert mock_run.call_args[1]["input"] == b"- item one"

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired("xdotool", 120))
    def test_type_command_timeout_raises(self, mock_run):
        with pytest.raises(RuntimeError, match="xdotool timed out"):
            digue.send_text("hello", display_server="x11", input_mode="type")

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_type_missing_tool_gives_install_hint(self, mock_run):
        with pytest.raises(RuntimeError, match="sudo apt install"):
            digue.send_text("hello", display_server="x11", input_mode="type")

    @patch("subprocess.run")
    def test_paste_command_failure_raises(self, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=0),
            subprocess.CalledProcessError(1, "xdotool", stderr=b"no window"),
        ]

        with pytest.raises(RuntimeError, match="xdotool failed: no window"):
            digue.send_text("hello", display_server="x11")

    @patch("subprocess.run")
    def test_paste_command_timeout_raises(self, mock_run):
        mock_run.side_effect = [MagicMock(returncode=0), subprocess.TimeoutExpired("xdotool", 5)]

        with pytest.raises(RuntimeError, match="xdotool timed out"):
            digue.send_text("hello", display_server="x11")

    def test_input_mode_default_is_paste(self):
        config = digue._default_config()
        assert config["dictate"]["input_mode"] == "paste"


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

    def test_replace_failure_preserves_origin_and_removes_temp(self, tmp_path, capsys):
        rec_file = tmp_path / "digue-rec.wav"
        rec_file.write_bytes(b"audio")
        audio_dir = tmp_path / "audio"

        with patch("os.replace", side_effect=OSError("cross-device")):
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
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.stop_recording")
    @patch("digue.is_recording", return_value=True)
    @patch("digue.save_audio", return_value=("saved.flac", "2026-01-01T00:00:00"))
    def test_finish_dictation_passes_resolved_backend(
        self, mock_save, mock_recording, mock_stop, mock_running, mock_ensure, mock_transcribe, mock_send, tmp_path
    ):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        mock_stop.return_value = rec_file
        config = digue._default_config()
        config["server"]["backend"] = "remote"
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")

        digue.finish_dictation(config, mock_stop.return_value)

        mock_save.assert_called_once()
        assert mock_save.call_args[1]["backend"] == "remote"

    @patch("digue.send_text")
    @patch("digue.transcribe", return_value="hello")
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.stop_recording")
    @patch("digue.is_recording", return_value=True)
    @patch("digue.save_audio")
    def test_save_audio_false_skips_wav_but_writes_txt(
        self, mock_save, mock_recording, mock_stop, mock_running, mock_ensure, mock_transcribe, mock_send, tmp_path
    ):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        mock_stop.return_value = rec_file
        audio_dir = tmp_path / "audio"
        config = digue._default_config()
        config["dictate"]["save_audio"] = False
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = digue.finish_dictation(config, mock_stop.return_value)

        assert result.exit_code == 0
        mock_save.assert_not_called()
        txt_files = list(audio_dir.rglob("*.txt"))
        assert len(txt_files) == 1
        assert txt_files[0].read_text().strip() == "hello"


class TestDeliveryResult:
    """finish_dictation reports a typed terminal outcome instead of a bare int:
    "a function returned" alone does not say whether the take was delivered,
    rescued, empty, or failed in a way a later toggle may retry."""

    def make_config(self, tmp_path):
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        return config

    def make_rec_file(self, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        return rec_file

    def test_delivered_with_exit_zero(self, tmp_path):
        rec_file = self.make_rec_file(tmp_path)
        config = self.make_config(tmp_path)

        with (
            patch("digue.send_text"),
            patch("digue.transcribe", return_value="hello"),
        ):
            result = digue.finish_dictation(config, rec_file)

        assert result.outcome == "delivered"
        assert result.exit_code == 0
        assert result.rescued_path is None

    def test_empty_with_exit_zero(self, tmp_path):
        rec_file = self.make_rec_file(tmp_path)
        config = self.make_config(tmp_path)

        with patch("digue.send_text"), patch("digue.transcribe", return_value=""):
            result = digue.finish_dictation(config, rec_file)

        assert result.outcome == "empty"
        assert result.exit_code == 0

    def test_missing_recording_is_empty_with_exit_one(self, tmp_path):
        config = self.make_config(tmp_path)

        with patch("digue.notify") as mock_notify:
            result = digue.finish_dictation(config, None)

        assert result.outcome == "empty"
        assert result.exit_code == 1
        mock_notify.assert_called_once()

    def test_transcription_failure_rescues_the_recording(self, tmp_path):
        rec_file = self.make_rec_file(tmp_path)
        config = self.make_config(tmp_path)

        with (
            patch("digue.send_text"),
            patch("digue.transcribe", side_effect=RuntimeError("server down")),
        ):
            result = digue.finish_dictation(config, rec_file)

        assert result.outcome == "rescued"
        assert result.exit_code == 1
        assert result.rescued_path is not None
        assert result.rescued_path.read_bytes() == b"audio"
        assert not rec_file.exists()

    def test_transcription_failure_without_rescue_is_retryable(self, tmp_path):
        rec_file = self.make_rec_file(tmp_path)
        config = self.make_config(tmp_path)

        with (
            patch("digue.send_text"),
            patch("digue.transcribe", side_effect=RuntimeError("server down")),
            patch("digue.rescue_recording", return_value=None),
        ):
            result = digue.finish_dictation(config, rec_file)

        assert result.outcome == "retryable_failure"
        assert result.exit_code == 1
        assert result.rescued_path is None
        assert rec_file.exists()

    def test_paste_failure_keeps_txt_and_audio(self, tmp_path):
        rec_file = self.make_rec_file(tmp_path)
        config = self.make_config(tmp_path)

        with (
            patch("digue.send_text", side_effect=RuntimeError("no display")),
            patch("digue.transcribe", return_value="hello"),
        ):
            result = digue.finish_dictation(config, rec_file)

        assert result.outcome == "rescued"
        assert result.exit_code == 1
        month = tmp_path / "audio" / digue.month_dir_for(digue.now_timestamp())
        assert (month / f"{digue.now_timestamp()}.txt").exists() or list(month.glob("*.txt"))
        assert list(month.glob("*.wav"))

    def test_archive_failure_rescues_the_raw_recording(self, tmp_path):
        rec_file = self.make_rec_file(tmp_path)
        config = self.make_config(tmp_path)

        with (
            patch("digue.send_text"),
            patch("digue.transcribe", return_value="hello"),
            patch("digue.save_audio", side_effect=OSError("disk full")),
        ):
            result = digue.finish_dictation(config, rec_file)

        assert result.outcome == "rescued"
        assert result.exit_code == 1
        assert result.rescued_path is not None
        assert result.rescued_path.read_bytes() == b"audio"

    def test_archive_failure_without_rescue_is_retryable(self, tmp_path):
        rec_file = self.make_rec_file(tmp_path)
        config = self.make_config(tmp_path)

        with (
            patch("digue.send_text"),
            patch("digue.transcribe", return_value="hello"),
            patch("digue.save_audio", side_effect=OSError("disk full")),
            patch("digue.rescue_recording", return_value=None),
        ):
            result = digue.finish_dictation(config, rec_file)

        assert result.outcome == "retryable_failure"
        assert result.exit_code == 1
        assert rec_file.exists()

    @pytest.mark.parametrize("exit_code", [0, 1])
    def test_toggle_maps_the_result_to_the_exit_code(self, exit_code, tmp_path):
        import os

        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        config["dictate"]["max_duration"] = 0
        recorder = MagicMock(pid=os.getpid(), poll=lambda: 0)
        result = digue.DeliveryResult(outcome="rescued", exit_code=exit_code)

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.ensure_server"),
            patch("digue.is_server_running", return_value=True),
            patch("digue.is_recording", return_value=False),
            patch("subprocess.Popen", return_value=recorder),
            patch("digue._pid_file", return_value=tmp_path / "digue.pid"),
            patch("digue._wait_recorder_end_daemon", return_value="ended"),
            patch("digue.finish_dictation", return_value=result),
            patch("digue.notify"),
            patch("digue.notify_close"),
            patch("signal.signal"),
        ):
            assert digue.dictate_toggle(config) == exit_code


class TestDictateArchivesAfterDelivery:
    @patch("digue.notify")
    @patch("digue.send_text")
    @patch("digue.transcribe", return_value="hello")
    def test_limit_warning_remains_in_transcription_progress(self, mock_transcribe, mock_send, mock_notify, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")

        assert digue.finish_dictation(config, rec_file, limit_reached=True).exit_code == 0

        first_message = mock_notify.call_args_list[0].args[0]
        assert "Limit reached" in first_message
        assert "transcribing" in first_message

    @patch("digue.send_text")
    @patch("digue.transcribe", return_value="hello")
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.stop_recording")
    @patch("digue.is_recording", return_value=True)
    @patch("digue.save_audio", return_value=("saved.flac", "2026-01-01T00:00:00"))
    def test_transcribes_and_pastes_before_archiving(
        self, mock_save, mock_recording, mock_stop, mock_running, mock_ensure, mock_transcribe, mock_send, tmp_path
    ):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        mock_stop.return_value = rec_file
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")

        order = MagicMock()
        order.attach_mock(mock_transcribe, "transcribe")
        order.attach_mock(mock_send, "send_text")
        order.attach_mock(mock_save, "save_audio")

        rec_file_result = mock_stop.return_value
        result = digue.finish_dictation(config, rec_file_result)

        assert result.exit_code == 0
        assert [call_record[0] for call_record in order.mock_calls] == ["transcribe", "send_text", "save_audio"]

    @patch("digue.send_text")
    @patch("digue.transcribe", side_effect=RuntimeError("server down"))
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.stop_recording")
    @patch("digue.is_recording", return_value=True)
    def test_transcribe_failure_archives_recording(
        self, mock_recording, mock_stop, mock_running, mock_ensure, mock_transcribe, mock_send, tmp_path, capsys
    ):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        mock_stop.return_value = rec_file
        audio_dir = tmp_path / "audio"
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        rec_file_result = mock_stop.return_value
        result = digue.finish_dictation(config, rec_file_result)

        assert result.exit_code == 1
        assert not rec_file.exists()
        saved = list(audio_dir.rglob("*.wav"))
        assert len(saved) == 1
        assert saved[0].read_bytes() == b"audio"
        assert "Transcription failed" in capsys.readouterr().err

    @patch("digue.send_text")
    @patch("digue.transcribe", side_effect=RuntimeError("server down"))
    def test_transcribe_failure_keeps_recording_even_with_save_audio_off(
        self, mock_transcribe, mock_send, tmp_path, capsys
    ):
        """save-audio only skips the backup of a delivered take; a failed take
        is not delivered anywhere, so dropping it would lose the audio."""
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        audio_dir = tmp_path / "audio"
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)
        config["dictate"]["save_audio"] = False

        result = digue.finish_dictation(config, rec_file)

        assert result.exit_code == 1
        saved = list(audio_dir.rglob("*.wav"))
        assert len(saved) == 1
        assert saved[0].read_bytes() == b"audio"
        assert not rec_file.exists()
        assert "Recording kept at" in capsys.readouterr().err

    @patch("digue.save_audio", side_effect=OSError("disk full"))
    @patch("digue.send_text")
    @patch("digue.transcribe", return_value="hello")
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.stop_recording")
    @patch("digue.is_recording", return_value=True)
    def test_save_audio_failure_after_paste_keeps_uncompressed_recording(
        self,
        mock_recording,
        mock_stop,
        mock_running,
        mock_ensure,
        mock_transcribe,
        mock_send,
        mock_save,
        tmp_path,
        capsys,
    ):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        mock_stop.return_value = rec_file
        audio_dir = tmp_path / "audio"
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        rec_file_result = mock_stop.return_value
        result = digue.finish_dictation(config, rec_file_result)

        assert result.exit_code == 1
        assert not rec_file.exists()
        saved = list(audio_dir.rglob("*.wav"))
        assert len(saved) == 1
        assert saved[0].read_bytes() == b"audio"
        txts = list(audio_dir.rglob("*.txt"))
        assert len(txts) == 1
        err = capsys.readouterr().err
        assert "Failed to save audio" in err
        assert txts[0].read_text().strip() == "hello"


class TestDictateAudioTranscriptPairing:
    def test_audio_and_transcript_share_timestamp(self, tmp_path):
        """Regression: the .txt and the audio must share the same timestamp even when
        archiving runs after transcription (crossing a second boundary must not
        split the pair)."""

        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        with (
            patch("digue.now_timestamp", side_effect=["20260904-120000", "20260904-120005"]),
            patch("digue.send_text"),
            patch("digue.transcribe", return_value="hello"),
            patch("digue.ensure_server"),
            patch("digue.is_server_running", return_value=True),
            patch("digue.stop_recording", return_value=rec_file) as mock_stop,
            patch("digue.is_recording", return_value=True),
        ):
            result = digue.finish_dictation(config, mock_stop.return_value)

        assert result.exit_code == 0
        month = tmp_path / "audio" / "2026" / "09"
        assert {path.name for path in month.iterdir()} == {"20260904-120000.txt", "20260904-120000.wav"}


class TestNormalizePastedText:
    def test_collapses_newlines_and_spaces(self):
        assert digue.normalize_pasted_text("olá\n\nmundo \t aqui") == "olá mundo aqui"

    def test_joins_whisper_wrapped_lines(self):
        sample = (
            "Eu queria fazer um teste aqui e aí por algum motivo o trans\n"
            "crevendo não sumiu.\n"
            "Então tem que ver o que está acontecendo aqui para ele não\n"
            "estar desaparecendo."
        )
        assert digue.normalize_pasted_text(sample) == (
            "Eu queria fazer um teste aqui e aí por algum motivo o trans crevendo não sumiu. "
            "Então tem que ver o que está acontecendo aqui para ele não estar desaparecendo."
        )

    def test_strips_edges(self):
        assert digue.normalize_pasted_text("  text  ") == "text"

    def test_empty(self):
        assert digue.normalize_pasted_text("") == ""


class TestNotifyLifecycle:
    @patch("digue.send_text")
    @patch("digue.transcribe", return_value="hello")
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.stop_recording")
    @patch("digue.is_recording", return_value=True)
    @patch("digue.save_audio", return_value=("saved.flac", "2026-01-01T00:00:00"))
    @patch("digue.notify_close")
    def test_successful_dictation_closes_notification(
        self,
        mock_close,
        mock_save,
        mock_recording,
        mock_stop,
        mock_running,
        mock_ensure,
        mock_transcribe,
        mock_send,
        tmp_path,
    ):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        mock_stop.return_value = rec_file
        audio_dir = tmp_path / "audio"
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        rec_file_result = mock_stop.return_value
        result = digue.finish_dictation(config, rec_file_result)

        assert result.exit_code == 0
        mock_close.assert_called_once()

    @patch("digue.send_text", side_effect=RuntimeError("no display"))
    @patch("digue.transcribe", return_value="hello")
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.stop_recording")
    @patch("digue.is_recording", return_value=True)
    @patch("digue.save_audio", return_value=("saved.flac", "2026-01-01T00:00:00"))
    def test_paste_failure_notifies_with_timeout(
        self,
        mock_save,
        mock_recording,
        mock_stop,
        mock_running,
        mock_ensure,
        mock_transcribe,
        mock_send,
        tmp_path,
        capsys,
    ):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        mock_stop.return_value = rec_file
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")

        rec_file_result = mock_stop.return_value
        result = digue.finish_dictation(config, rec_file_result)

        assert result.exit_code == 1
        err = capsys.readouterr().err
        assert "Paste failed" in err
        assert "Transcription saved to" in err

    @patch("digue.send_text")
    @patch("digue.transcribe", return_value="hello")
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.save_audio", side_effect=OSError("disk full"))
    @patch("digue.stop_recording")
    @patch("digue.is_recording", return_value=True)
    def test_save_failure_notifies_and_does_not_crash(
        self,
        mock_recording,
        mock_stop,
        mock_save,
        mock_running,
        mock_ensure,
        mock_transcribe,
        mock_send,
        tmp_path,
        capsys,
    ):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        mock_stop.return_value = rec_file
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")

        rec_file_result = mock_stop.return_value
        result = digue.finish_dictation(config, rec_file_result)

        assert result.exit_code == 1
        err = capsys.readouterr().err
        assert "Failed to save audio" in err
        assert "disk full" in err

    @patch("digue.send_text")
    @patch("digue.transcribe", return_value="hello")
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.save_audio", return_value=("saved.flac", "2026-01-01T00:00:00"))
    @patch("digue.stop_recording")
    @patch("digue.is_recording", return_value=True)
    def test_transcript_write_failure_notifies_and_prints_text(
        self,
        mock_recording,
        mock_stop,
        mock_save,
        mock_running,
        mock_ensure,
        mock_transcribe,
        mock_send,
        tmp_path,
        capsys,
    ):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        mock_stop.return_value = rec_file
        config = digue._default_config()
        config["dictate"]["save_audio"] = False
        # point audio_dir at a file so the transcript write fails
        blocker = tmp_path / "blocker"
        blocker.write_text("not a dir")
        config["dictate"]["audio_dir"] = str(blocker)

        rec_file_result = mock_stop.return_value
        result = digue.finish_dictation(config, rec_file_result)

        assert result.exit_code == 1
        err = capsys.readouterr().err
        assert "Failed to save transcript" in err
        assert "hello" in err


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


class TestOutputFilesEndWithOneNewline:
    VTT = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi\n"

    @patch("digue.transcribe", return_value=VTT)
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    def test_transcribe_output_file(self, mock_running, mock_ensure, mock_transcribe, tmp_path):
        """VTT/SRT results already end with a newline (see _post_process_subtitle);
        -o appended another one, ending every subtitle file with a blank line."""
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"audio")
        args = MagicMock(audio=audio, output=str(tmp_path / "a.vtt"), response_format="vtt")
        args.language = args.prompt = None
        args.verbose = False

        assert digue.cmd_transcribe(args, digue._default_config()) == 0

        content = (tmp_path / "a.vtt").read_text()
        assert content == self.VTT

    @patch("digue.transcribe", return_value=VTT)
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    def test_batch_transcribe_output_file(self, mock_running, mock_ensure, mock_transcribe, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        (input_dir / "a.mp3").write_bytes(b"audio")
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        args = MagicMock(input_dir=input_dir, output_dir=output_dir, response_format="vtt", language=None)

        assert digue.cmd_batch_transcribe(args, digue._default_config()) == 0

        assert (output_dir / "a.vtt").read_text() == self.VTT

    @patch("digue.transcribe", return_value="hello")
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    def test_text_output_still_gets_its_newline(self, mock_running, mock_ensure, mock_transcribe, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"audio")
        args = MagicMock(audio=audio, output=str(tmp_path / "a.txt"), response_format="text")
        args.language = args.prompt = None
        args.verbose = False

        digue.cmd_transcribe(args, digue._default_config())

        assert (tmp_path / "a.txt").read_text() == "hello\n"


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


class TestCmdTranscribeInput:
    @patch("digue.send_text")
    @patch("digue.transcribe", return_value="text")
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    def test_directory_input_gives_clear_error_not_ffmpeg(
        self, mock_running, mock_ensure, mock_transcribe, mock_send, tmp_path, capsys
    ):
        """Regression: a directory passed to transcribe must fail early with a clear error."""
        config = digue._default_config()
        args = MagicMock()
        args.audio = tmp_path
        args.output = None
        args.response_format = None
        args.language = None
        args.prompt = None
        args.verbose = False

        result = digue.cmd_transcribe(args, config)

        assert result == 1
        err = capsys.readouterr().err
        assert "not a file" in err or "is a directory" in err.lower()
        mock_transcribe.assert_not_called()
        mock_ensure.assert_not_called()
        mock_running.assert_not_called()

    @patch("digue.ensure_server")
    @patch("digue.is_server_running")
    def test_missing_input_does_not_start_server(self, mock_running, mock_ensure, tmp_path, capsys):
        args = MagicMock(audio=tmp_path / "missing.wav")

        assert digue.cmd_transcribe(args, digue._default_config()) == 1

        assert "not found" in capsys.readouterr().err
        mock_ensure.assert_not_called()
        mock_running.assert_not_called()

    @pytest.mark.parametrize("failure", [RuntimeError("request failed"), OSError("disk full")])
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    def test_operational_failure_returns_1_without_traceback(
        self, mock_running, mock_ensure, failure, tmp_path, capsys
    ):
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"audio")
        output_path = tmp_path / "transcript.txt"
        args = MagicMock(
            audio=audio_path,
            output=output_path,
            response_format="text",
            language=None,
            prompt=None,
            verbose=False,
        )
        config = digue._default_config()

        transcribe_result = (
            patch("digue.transcribe", side_effect=failure)
            if isinstance(failure, RuntimeError)
            else patch("digue.transcribe", return_value="text")
        )
        write_result = (
            patch("pathlib.Path.write_text", side_effect=failure)
            if isinstance(failure, OSError)
            else patch("pathlib.Path.write_text")
        )
        with transcribe_result, write_result:
            result = digue.cmd_transcribe(args, config)

        assert result == 1
        error = capsys.readouterr().err
        assert f"Error: {failure}" in error
        assert "Traceback" not in error


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


class TestConfigurationNetworkAndSharedDefaults:
    @pytest.mark.parametrize(
        ("bind_ip", "expected"),
        (("192.0.2.10", "192.0.2.10"), ("0.0.0.0", "127.0.0.1")),
    )
    def test_local_server_host_matches_reachable_bind_address(self, bind_ip, expected):
        config = digue._default_config()
        config["server"]["bind_ip"] = bind_ip
        assert digue.server_host(config) == expected

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
            ('[models]\ncpu = "invalid"\n', "models.cpu"),
            ('[transcribe]\noutput-format = "invalid"\n', "transcribe.output_format"),
        ),
    )
    def test_load_config_validates_resolved_values(self, tmp_path, toml, message):
        config_path = tmp_path / "config.toml"
        config_path.write_text(toml)
        with pytest.raises(ValueError, match=message):
            digue.load_config(config_path)

    def test_custom_docker_image_is_allowed(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text('[server]\nimage = "registry.example/my-whisper:custom"\n')
        assert digue.load_config(config_path)["server"]["image"] == "registry.example/my-whisper:custom"

    def test_config_init_uses_global_config_path(self, tmp_path):
        target = tmp_path / "selected.toml"
        args = MagicMock(config=str(target), output=None, force=False)
        assert digue._config_init(args) == 0
        assert target.exists()

    def test_doctor_prints_selected_config_path(self, tmp_path, capsys):
        target = tmp_path / "selected.toml"
        target.write_text("")
        args = MagicMock(config=str(target))
        with patch("digue.image_exists", return_value=False), patch("shutil.which", return_value=None):
            digue.cmd_doctor(args, digue.load_config(target))
        assert str(target) in capsys.readouterr().err

    @patch("digue.transcribe", return_value="text")
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.ensure_server")
    def test_transcribe_uses_config_prompt_when_cli_absent(self, mock_ensure, mock_running, mock_transcribe, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"audio")
        args = MagicMock(audio=audio, language=None, response_format=None, verbose=False, prompt=None, output=None)
        config = digue._default_config()
        config["transcribe"]["prompt"] = "Pythonic Café"
        assert digue.cmd_transcribe(args, config) == 0
        assert mock_transcribe.call_args.kwargs["prompt"] == "Pythonic Café"

    @patch("digue.transcribe", return_value="WEBVTT\n")
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.ensure_server")
    def test_batch_uses_config_format_prompt_and_wrapping(self, mock_ensure, mock_running, mock_transcribe, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        (input_dir / "audio.wav").write_bytes(b"audio")
        args = MagicMock(input_dir=input_dir, output_dir=output_dir, response_format=None, language=None)
        config = digue._default_config()
        config["transcribe"].update(output_format="srt", prompt="names", max_line_length=50, max_lines=3)
        assert digue.cmd_batch_transcribe(args, config) == 0
        assert (output_dir / "audio.srt").exists()
        assert mock_transcribe.call_args.args[3] == "srt"
        assert mock_transcribe.call_args.kwargs == {
            "prompt": "names",
            "max_line_length": 50,
            "max_lines": 3,
            "wrap_cues": True,
        }


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
