"""Tests for backend detection, Docker container lifecycle, model download, and remote server."""

import sys
from unittest.mock import MagicMock, call, patch

import pytest

import digue
from digue.config import _default_config, load_config

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


# -- Command handlers ---------------------------------------------------------


class TestCmdDetect:
    @patch("digue.detect_backend", return_value="nvidia")
    def test_prints_backend(self, mock_detect, capsys):
        result = digue.cmd_detect(MagicMock())
        assert result == 0
        assert capsys.readouterr().out.strip() == "nvidia"


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


class TestDockerMissing:
    """Without the docker binary, every local-backend command used to die with
    a FileNotFoundError traceback, and `dictate` blamed the recorder."""

    def test_docker_run_raises_a_named_error(self):
        with (
            patch("subprocess.run", side_effect=FileNotFoundError(2, "No such file", "docker")),
            pytest.raises(digue.DockerNotFoundError, match="docker not found"),
        ):
            digue._docker_run(["ps"])

    def test_pull_image_raises_a_named_error(self):
        with (
            patch("digue.image_exists", return_value=False),
            patch("subprocess.run", side_effect=FileNotFoundError(2, "No such file", "docker")),
            pytest.raises(digue.DockerNotFoundError, match="docker not found"),
        ):
            digue.pull_image("ghcr.io/example/image")

    def test_main_reports_missing_docker_without_traceback(self, tmp_path, capsys):
        config_path = tmp_path / "config.toml"
        config_path.write_text('[server]\nbackend = "cpu"\n')
        with (
            patch.object(sys, "argv", ["digue", "--config", str(config_path), "server", "status"]),
            patch("digue.is_server_running", return_value=False),
            patch("subprocess.run", side_effect=FileNotFoundError(2, "No such file", "docker")),
            pytest.raises(SystemExit) as exc_info,
        ):
            digue.main()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "Error: docker not found" in err
        assert "Traceback" not in err

    def test_dictate_blames_docker_not_the_recorder(self, tmp_path):
        config = _default_config()
        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.ensure_server", side_effect=digue.DockerNotFoundError(digue.DOCKER_NOT_FOUND)),
            patch("digue.send_notification") as mock_notify,
        ):
            assert digue.dictate_toggle(config) == 1

        message = mock_notify.call_args.args[0]
        assert "docker not found" in message
        assert "Recorder not found" not in message
        assert not (tmp_path / "digue-daemon.pid").exists()


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
        config = _default_config()
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
        config = _default_config()
        digue.create_container(config, "amd")
        cmd = mock_docker.call_args[0][0]
        assert "/dev/kfd" in cmd

    @patch("digue.download_model")
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_cpu_has_no_device_flags(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        digue.create_container(config, "cpu")
        cmd = mock_docker.call_args[0][0]
        assert "--device" not in cmd
        assert "--gpus" not in cmd

    @patch("digue.download_model")
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_raises_on_failure(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=1, stderr="permission denied")
        config = _default_config()
        with pytest.raises(RuntimeError, match="permission denied"):
            digue.create_container(config, "cpu")

    @patch("digue.download_model")
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_binds_to_localhost(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        digue.create_container(config, "cpu")
        cmd = mock_docker.call_args[0][0]
        port_binding = [arg for arg in cmd if "8178" in arg and "127.0.0.1" in arg]
        assert port_binding, "Port must bind to 127.0.0.1"

    @patch("digue.download_model")
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_binds_to_configured_ip(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        config["server"]["bind_ip"] = "192.168.1.10"
        digue.create_container(config, "cpu")
        cmd = mock_docker.call_args[0][0]
        assert "192.168.1.10:8178:8080" in cmd

    @patch("digue.download_model")
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_calls_pull_image(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        digue.create_container(config, "cpu")
        mock_pull.assert_called_once_with("ghcr.io/ggml-org/whisper.cpp:main-vulkan")

    @patch("digue.download_model")
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_downloads_model_when_missing(self, mock_docker, mock_pull, mock_download, tmp_path):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        config["server"]["data_dir"] = str(tmp_path)
        digue.create_container(config, "cpu")
        mock_download.assert_called_once_with("small", tmp_path / "models", with_notification=True)

    @patch("digue.download_model")
    @patch("digue.pull_image")
    @patch("digue._docker_run")
    def test_skips_download_when_model_exists(self, mock_docker, mock_pull, mock_download, tmp_path):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
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
        config = _default_config()
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
            patch("digue.send_notification", side_effect=lambda message, timeout_ms=0: messages.append(message)),
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
            patch("digue.send_notification", side_effect=lambda message, timeout_ms=0: messages.append(message)),
            patch("time.monotonic", return_value=1.0),
        ):
            hook = digue._download_progress_hook("test.bin", with_notification=True)
            hook(1, 1024 * 1024, -1)
        assert messages == ["Downloading test.bin...       1.0 MB"]

    def test_notify_prints_carriage_return_on_tty(self, capsys):
        with patch.object(digue, "_stderr_is_tty", return_value=True):
            digue.send_notification("status message")
        err = capsys.readouterr().err
        assert err.startswith("\r")
        assert "[digue] status message" in err

    def test_notify_prints_plain_line_when_not_tty(self, capsys):
        with patch.object(digue, "_stderr_is_tty", return_value=False):
            digue.send_notification("status message")
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


# -- Server -------------------------------------------------------------------


class TestIsServerRunning:
    @patch("urllib.request.urlopen")
    def test_true_when_responds(self, mock_urlopen):
        config = _default_config()
        assert digue.is_server_running(config) is True

    @patch("urllib.request.urlopen", side_effect=OSError)
    def test_false_when_refused(self, mock_urlopen):
        config = _default_config()
        assert digue.is_server_running(config) is False


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
            digue.ensure_server(_default_config(), silent=True)

        mock_wait.assert_not_called()

    @pytest.mark.parametrize("silent", (False, True))
    @patch("digue.notify_close")
    @patch("digue.send_notification")
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
        config = _default_config()

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

    @patch("digue.send_notification")
    @patch("digue._wait_for_server", return_value=False)
    @patch("digue.container_status", return_value="running")
    @patch("digue.is_server_running", return_value=False)
    def test_ensure_server_reports_running_container_timeout(self, mock_running, mock_status, mock_wait, mock_notify):
        config = _default_config()

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
        config = _default_config()

        assert digue.ensure_server(config, silent=True) == expected_backend

        if status == "exited":
            mock_start.assert_called_once_with()
            mock_create.assert_not_called()
        else:
            mock_start.assert_not_called()
            mock_create.assert_called_once_with(config, "cpu")

    @patch("digue.send_notification")
    @patch("digue._wait_for_server")
    @patch("digue.container_status", return_value="paused")
    @patch("digue.is_server_running", return_value=False)
    def test_ensure_server_rejects_invalid_container_state(self, mock_running, mock_status, mock_wait, mock_notify):
        assert digue.ensure_server(_default_config()) is None

        mock_notify.assert_called_once_with("Container in unexpected state: paused", timeout_ms=5000)
        mock_wait.assert_not_called()

    @patch("digue.stop_container", side_effect=RuntimeError("stop failed"))
    def test_cmd_server_stop_does_not_report_false_success(self, mock_stop, capsys):
        assert digue.cmd_server_stop(MagicMock(), _default_config()) == 1
        assert "Error: stop failed" in capsys.readouterr().err

    @patch("digue.remove_container", side_effect=RuntimeError("remove failed"))
    def test_cmd_server_destroy_does_not_report_false_success(self, mock_remove, capsys):
        assert digue.cmd_server_destroy(MagicMock(), _default_config()) == 1
        assert "Error: remove failed" in capsys.readouterr().err

    @patch("digue.start_container", side_effect=RuntimeError("start failed"))
    @patch("digue.container_status", return_value="exited")
    @patch("digue.is_server_running", return_value=False)
    def test_cmd_server_start_does_not_report_false_success(self, mock_running, mock_status, mock_start, capsys):
        assert digue.cmd_server_start(MagicMock(), _default_config()) == 1
        assert "Error: start failed" in capsys.readouterr().err

    @patch("digue._wait_for_server", return_value=False)
    @patch("digue.start_container")
    @patch("digue.container_status", return_value="exited")
    @patch("digue.is_server_running", return_value=False)
    def test_cmd_server_start_timeout_hint_names_the_digue_container(
        self, mock_running, mock_status, mock_start, mock_wait, capsys
    ):
        assert digue.cmd_server_start(MagicMock(), _default_config()) == 1
        err = capsys.readouterr().err
        assert f"docker logs {digue.CONTAINER_NAME}" in err
        assert "whisper-server" not in err


class TestRemoteBackend:
    def test_resolve_backend_from_config(self):
        config = _default_config()
        config["server"]["backend"] = "remote"
        assert digue.resolve_backend(config) == "remote"

    @patch("digue.is_server_running", return_value=False)
    @patch("digue.container_status")
    def test_ensure_server_remote_never_touches_container(self, mock_status, mock_running):
        config = _default_config()
        config["server"]["backend"] = "remote"
        assert digue.ensure_server(config, silent=True) is None
        mock_status.assert_not_called()

    @patch("digue._docker_run")
    def test_create_container_remote_raises(self, mock_docker):
        config = _default_config()
        with pytest.raises(RuntimeError, match="remote"):
            digue.create_container(config, "remote")
        mock_docker.assert_not_called()

    def test_hint_remote_mentions_tunnel(self):
        config = _default_config()
        config["server"]["backend"] = "remote"
        hint = digue.server_not_running_hint(config)
        assert "ssh -NfL" in hint
        assert "8178" in hint

    def test_hint_local_suggests_start(self):
        config = _default_config()
        assert digue.server_not_running_hint(config) == "Run: digue server start"

    @patch("digue.is_server_running", return_value=True)
    def test_cmd_server_status_remote(self, mock_running, capsys):
        config = _default_config()
        config["server"]["backend"] = "remote"
        assert digue.cmd_server_status(MagicMock(), config) == 0
        assert "remote" in capsys.readouterr().err

    @patch("digue.is_server_running", return_value=False)
    def test_cmd_server_status_remote_not_responding(self, mock_running, capsys):
        config = _default_config()
        config["server"]["backend"] = "remote"
        assert digue.cmd_server_status(MagicMock(), config) == 1


class TestRemoteHost:
    @patch("urllib.request.urlopen")
    @patch("digue.detect_backend")
    def test_server_probes_do_not_detect_auto_backend(self, mock_detect_backend, mock_urlopen):
        config = _default_config()

        digue.is_server_running(config)
        digue.server_url(config)
        digue.server_not_running_hint(config)

        mock_detect_backend.assert_not_called()

    def test_server_host_localhost_for_local_backends(self):
        config = _default_config()
        config["server"]["remote_host"] = "10.0.0.5"
        assert digue.server_host(config) == "127.0.0.1"

    def test_server_host_default_tunnel(self):
        config = _default_config()
        config["server"]["backend"] = "remote"
        assert digue.server_host(config) == "127.0.0.1"

    def test_server_host_remote_lan(self):
        config = _default_config()
        config["server"]["backend"] = "remote"
        config["server"]["remote_host"] = "10.0.0.5"
        assert digue.server_host(config) == "10.0.0.5"

    def test_server_url_uses_remote_host(self):
        config = _default_config()
        config["server"]["backend"] = "remote"
        config["server"]["remote_host"] = "desktop.lan"
        assert digue.server_url(config) == "http://desktop.lan:8178/inference"

    @patch("urllib.request.urlopen")
    def test_is_server_running_probes_remote_host(self, mock_urlopen):
        config = _default_config()
        config["server"]["backend"] = "remote"
        config["server"]["remote_host"] = "10.0.0.5"
        digue.is_server_running(config)
        url = mock_urlopen.call_args[0][0]
        assert url == "http://10.0.0.5:8178/"

    def test_hint_remote_lan_mentions_host(self):
        config = _default_config()
        config["server"]["backend"] = "remote"
        config["server"]["remote_host"] = "10.0.0.5"
        hint = digue.server_not_running_hint(config)
        assert "10.0.0.5" in hint
        assert "ssh -NfL" not in hint

    def test_hint_remote_tunnel_keeps_ssh(self):
        config = _default_config()
        config["server"]["backend"] = "remote"
        hint = digue.server_not_running_hint(config)
        assert "ssh -NfL" in hint

    @patch("digue.is_server_running", return_value=False)
    def test_cmd_server_status_remote_shows_host(self, mock_running, capsys):
        config = _default_config()
        config["server"]["backend"] = "remote"
        config["server"]["remote_host"] = "10.0.0.5"
        digue.cmd_server_status(MagicMock(), config)
        assert "10.0.0.5:8178" in capsys.readouterr().err


class TestCmdDoctor:
    def test_prints_resolved_config_without_crashing(self, capsys):
        config = _default_config()
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
        config = _default_config()
        with (
            patch("digue.image_exists", return_value=False),
            patch("shutil.which", return_value=None),
        ):
            digue.cmd_doctor(MagicMock(), config)
        err = capsys.readouterr().err
        assert "] pw-record:" in err
        assert "] arecord:" in err

    def test_prints_selected_config_path(self, tmp_path, capsys):
        target = tmp_path / "selected.toml"
        target.write_text("")
        args = MagicMock(config=str(target))
        with patch("digue.image_exists", return_value=False), patch("shutil.which", return_value=None):
            digue.cmd_doctor(args, load_config(target))
        assert str(target) in capsys.readouterr().err
