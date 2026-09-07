"""Tests for digue.py."""

import argparse
import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import digue

# -- Config -------------------------------------------------------------------


class TestDefaultConfig:
    def test_has_required_sections(self):
        config = digue._default_config()
        assert "server" in config
        assert "dictation" in config
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

    def test_expands_tilde_in_paths(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent("""\
            [server]
            data-dir = "~/whisper/data"

            [dictation]
            audio-dir = "~/whisper-audio"
        """)
        )
        config = digue.load_config(config_path)
        assert "~" not in config["server"]["data_dir"]
        assert "~" not in config["dictation"]["audio_dir"]
        assert config["server"]["data_dir"].endswith("whisper/data")
        assert config["dictation"]["audio_dir"].endswith("whisper-audio")


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
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_nvidia_uses_gpus_flag(self, mock_docker, mock_pull):
        mock_docker.return_value = MagicMock(returncode=0)
        config = digue._default_config()
        digue.create_container(config, "nvidia")
        cmd = mock_docker.call_args[0][0]
        assert "--gpus" in cmd
        assert "all" in cmd
        assert any("main-cuda" in arg for arg in cmd)

    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_amd_uses_kfd_device(self, mock_docker, mock_pull):
        mock_docker.return_value = MagicMock(returncode=0)
        config = digue._default_config()
        digue.create_container(config, "amd")
        cmd = mock_docker.call_args[0][0]
        assert "/dev/kfd" in cmd

    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_cpu_has_no_device_flags(self, mock_docker, mock_pull):
        mock_docker.return_value = MagicMock(returncode=0)
        config = digue._default_config()
        digue.create_container(config, "cpu")
        cmd = mock_docker.call_args[0][0]
        assert "--device" not in cmd
        assert "--gpus" not in cmd

    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_raises_on_failure(self, mock_docker, mock_pull):
        mock_docker.return_value = MagicMock(returncode=1, stderr="permission denied")
        config = digue._default_config()
        with pytest.raises(RuntimeError, match="permission denied"):
            digue.create_container(config, "cpu")

    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_binds_to_localhost(self, mock_docker, mock_pull):
        mock_docker.return_value = MagicMock(returncode=0)
        config = digue._default_config()
        digue.create_container(config, "cpu")
        cmd = mock_docker.call_args[0][0]
        port_binding = [arg for arg in cmd if "8178" in arg and "127.0.0.1" in arg]
        assert port_binding, "Port must bind to 127.0.0.1"

    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_calls_pull_image(self, mock_docker, mock_pull):
        mock_docker.return_value = MagicMock(returncode=0)
        config = digue._default_config()
        digue.create_container(config, "cpu")
        mock_pull.assert_called_once_with("ghcr.io/ggml-org/whisper.cpp:main-vulkan")


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


# -- Notifications -----------------------------------------------------------


class TestNotify:
    @patch("subprocess.run")
    def test_sends_with_replace_id(self, mock_run):
        digue.notify("test", timeout_ms=5000)
        cmd = mock_run.call_args[0][0]
        assert "--replace-id" in cmd
        assert str(digue.NOTIFY_REPLACE_ID) in cmd

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
    def test_skips_language_for_auto(self, mock_urlopen, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = b"text"
        mock_urlopen.return_value = mock_response

        digue.transcribe("http://localhost:8178/inference", audio_file, "auto")
        request = mock_urlopen.call_args[0][0]
        assert b"language" not in request.data


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
        error = urllib.error.HTTPError("http://x", 400, "Bad Request", None, None)
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
        mock_send.side_effect = urllib.error.HTTPError("http://x", 400, "Bad Request", None, None)
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
    def test_sends_original_filename(self, mock_multipart, tmp_path):
        audio = tmp_path / "tone_opus.ogg"
        audio.write_bytes(b"data")
        digue._send_audio("http://x", audio, "auto", "text", 10, audio_data=b"converted")
        filename = mock_multipart.call_args[1]["filename"]
        assert filename == "tone_opus.ogg"
        audio_data = mock_multipart.call_args[0][1]
        assert audio_data == b"converted"


# -- VTT simplification ------------------------------------------------------


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
    def test_starts_in_new_session(self, mock_popen, mock_pid_file, tmp_path):
        mock_popen.return_value = MagicMock(pid=1234)
        mock_pid_file.return_value = tmp_path / "digue.pid"
        config = digue._default_config()

        pid = digue.start_recording(config)

        assert pid == 1234
        assert mock_popen.call_args[1].get("start_new_session") is True

    @patch("digue._spawn_limit_watchdog")
    @patch("digue._pid_file")
    @patch("subprocess.Popen")
    def test_max_duration_spawns_watchdog(self, mock_popen, mock_pid_file, mock_watchdog, tmp_path):
        mock_popen.return_value = MagicMock(pid=777)
        mock_pid_file.return_value = tmp_path / "digue.pid"
        config = digue._default_config()
        config["dictation"]["max_duration"] = 300

        digue.start_recording(config)

        mock_watchdog.assert_called_once_with(777, 300)

    @patch("digue._spawn_limit_watchdog")
    @patch("digue._pid_file")
    @patch("subprocess.Popen")
    def test_zero_max_duration_spawns_no_watchdog(self, mock_popen, mock_pid_file, mock_watchdog, tmp_path):
        mock_popen.return_value = MagicMock(pid=777)
        mock_pid_file.return_value = tmp_path / "digue.pid"
        config = digue._default_config()
        config["dictation"]["max_duration"] = 0

        digue.start_recording(config)

        mock_watchdog.assert_not_called()


class TestSpawnLimitWatchdog:
    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/bin/notify-send")
    def test_watchdog_kills_group_and_notifies(self, mock_which, mock_popen):
        digue._spawn_limit_watchdog(4242, 300)
        script = mock_popen.call_args[0][0][2]
        assert "sleep 300" in script
        assert "kill -TERM -4242" in script
        assert "notify-send" in script
        assert mock_popen.call_args[1].get("start_new_session") is True

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value=None)
    def test_watchdog_without_notify_send_falls_back_to_stderr(self, mock_which, mock_popen):
        digue._spawn_limit_watchdog(4242, 300)
        script = mock_popen.call_args[0][0][2]
        assert "notify-send" not in script
        assert "kill -TERM -4242" in script


class TestStopRecording:
    @patch("os.killpg")
    @patch("digue._group_alive", return_value=False)
    @patch("digue._rec_file")
    @patch("digue._pid_file")
    def test_kills_process_group_and_returns_file(
        self, mock_pid_file, mock_rec_file, mock_alive, mock_killpg, tmp_path
    ):
        pid_file = tmp_path / "digue.pid"
        pid_file.write_text("4242")
        rec_file = tmp_path / "digue.wav"
        rec_file.write_bytes(b"audio data")
        mock_pid_file.return_value = pid_file
        mock_rec_file.return_value = rec_file

        result = digue.stop_recording()

        assert result == rec_file
        mock_killpg.assert_called_once_with(4242, 15)
        assert not pid_file.exists()

    @patch("digue._pid_file")
    def test_returns_none_without_pid_file(self, mock_pid_file, tmp_path):
        mock_pid_file.return_value = tmp_path / "nope.pid"
        assert digue.stop_recording() is None


# -- Remote backend -----------------------------------------------------------


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


# -- CLI parser ---------------------------------------------------------------


class TestCreateParser:
    def test_all_subcommands_parse(self):
        parser = digue.create_parser()
        for cmd in ("detect", "download", "start", "stop", "destroy", "status", "dictate", "config", "benchmark"):
            args = parser.parse_args([cmd])
            assert args.command == cmd

    def test_simplify_vtt_subcommand(self, tmp_path):
        parser = digue.create_parser()
        args = parser.parse_args(["simplify-vtt", "test.vtt"])
        assert args.command == "simplify-vtt"
        assert args.input == "test.vtt"

    def test_transcribe_subcommand(self, tmp_path):
        parser = digue.create_parser()
        args = parser.parse_args(["transcribe", "test.wav", "-f", "vtt", "-o", "out.vtt"])
        assert args.command == "transcribe"
        assert args.response_format == "vtt"

    def test_no_args_defaults_to_none(self):
        parser = digue.create_parser()
        args = parser.parse_args([])
        assert args.command is None


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
        result = digue.cmd_config(MagicMock(), config)
        assert result == 0
        output = json.loads(capsys.readouterr().out)
        assert output["server"]["port"] == 8178
        assert "nvidia" in output["models"]


class TestCmdSimplifyVtt:
    def test_outputs_simplified_text(self, tmp_path, capsys):
        vtt_file = tmp_path / "test.vtt"
        vtt_file.write_text("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nHello\n")
        args = MagicMock()
        args.input = str(vtt_file)
        args.output = None
        config = digue._default_config()
        result = digue.cmd_simplify_vtt(args, config)
        assert result == 0
        assert capsys.readouterr().out.strip() == "[00:00:00] Hello"

    def test_writes_to_file(self, tmp_path):
        vtt_file = tmp_path / "test.vtt"
        vtt_file.write_text("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nHello\n")
        output_file = tmp_path / "output.txt"
        args = MagicMock()
        args.input = str(vtt_file)
        args.output = str(output_file)
        config = digue._default_config()
        result = digue.cmd_simplify_vtt(args, config)
        assert result == 0
        assert output_file.read_text().strip() == "[00:00:00] Hello"

    def test_reads_stdin(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", __import__("io").StringIO("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nHi\n"))
        args = MagicMock()
        args.input = "-"
        args.output = None
        config = digue._default_config()
        result = digue.cmd_simplify_vtt(args, config)
        assert result == 0
        assert capsys.readouterr().out.strip() == "[00:00:00] Hi"


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


class TestPasteText:
    def test_raises_when_no_display(self, monkeypatch):
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        with pytest.raises(RuntimeError, match="No DISPLAY"):
            digue.paste_text("hello", display_server="auto")

    @patch("subprocess.run")
    def test_x11_uses_xclip(self, mock_run):
        digue.paste_text("hello", display_server="x11")
        cmds = [call[0][0] for call in mock_run.call_args_list]
        assert cmds[0][0] == "xclip"
        assert cmds[1][0] == "xdotool"

    @patch("subprocess.run")
    def test_wayland_uses_wl_copy(self, mock_run):
        digue.paste_text("hello", display_server="wayland")
        cmds = [call[0][0] for call in mock_run.call_args_list]
        assert cmds[0][0] == "wl-copy"
        assert cmds[1][0] == "wtype"

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_missing_tool_gives_install_hint(self, mock_run):
        with pytest.raises(RuntimeError, match="sudo apt install"):
            digue.paste_text("hello", display_server="x11")


class TestSaveAudio:
    def test_copies_with_timestamp(self, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"wav data")
        audio_dir = tmp_path / "audio"
        saved, timestamp = digue.save_audio(rec_file, audio_dir)
        assert saved.exists()
        assert saved.parent == audio_dir
        assert timestamp in saved.name


class TestSaveAudioConfig:
    def test_default_saves_audio(self):
        config = digue._default_config()
        assert config["dictation"]["save_audio"] is True

    def test_loads_kebab_key(self, tmp_path):
        import textwrap

        config_path = tmp_path / "config.toml"
        config_path.write_text(textwrap.dedent("""\
            [dictation]
            save-audio = false
        """))
        config = digue.load_config(config_path)
        assert config["dictation"]["save_audio"] is False

    @patch("digue.paste_text")
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
        config["dictation"]["save_audio"] = False
        config["dictation"]["audio_dir"] = str(audio_dir)

        result = digue.dictate_toggle(config)

        assert result == 0
        mock_save.assert_not_called()
        txt_files = list(audio_dir.glob("*.txt"))
        assert len(txt_files) == 1
        assert txt_files[0].read_text().strip() == "hello"


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


# -- Batch commands -----------------------------------------------------------


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
