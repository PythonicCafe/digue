"""Tests for backend detection, Docker container lifecycle, model download, and remote server."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from digue import cli as cli_mod
from digue import container as container_mod
from digue import dictate as dictate_mod
from digue import notify as notify_mod
from digue.config import _default_config, apply_cli_overrides, load_config

_real_container_image = container_mod.container_image


@pytest.fixture(autouse=True)
def existing_container_has_the_configured_image(monkeypatch):
    """Tests that mock `container_status` never ran `docker inspect` for the
    image; keep them that way (no docker in tests). Tests about the image
    check override this with their own patch."""
    monkeypatch.setattr(container_mod, "container_image", lambda name=None: None)


# Detection


class TestDetectBackend:
    @patch("subprocess.run")
    def test_nvidia_when_nvidia_smi_works(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 3060\n")
        assert container_mod.detect_backend() == "nvidia"

    @patch("subprocess.run", side_effect=FileNotFoundError)
    @patch("pathlib.Path.exists", return_value=True)
    def test_amd_when_kfd_exists(self, mock_exists, mock_run):
        assert container_mod.detect_backend() == "amd"

    @patch("pathlib.Path.exists", return_value=False)
    @patch("subprocess.run")
    def test_intel_meteor_lake(self, mock_run, mock_exists):
        # First call: nvidia-smi fails. Second call: lspci returns Intel.
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout=""),  # nvidia-smi
            MagicMock(stdout="00:02.0 VGA compatible controller: Intel Corporation Meteor Lake-P [Intel Graphics]\n"),
        ]
        assert container_mod.detect_backend() == "intel"

    @patch("pathlib.Path.exists", return_value=False)
    @patch("subprocess.run")
    def test_cpu_for_broadwell(self, mock_run, mock_exists):
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout=""),
            MagicMock(
                stdout="00:02.0 VGA compatible controller: Intel Corporation Broadwell-U GT2 [HD Graphics 5500]\n"
            ),
        ]
        assert container_mod.detect_backend() == "cpu"

    @patch("pathlib.Path.exists", return_value=False)
    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_cpu_when_no_tools(self, mock_run, mock_exists):
        assert container_mod.detect_backend() == "cpu"


# Command handlers


class TestCmdDetect:
    @patch("digue.container.detect_backend", return_value="nvidia")
    def test_prints_backend(self, mock_detect, capsys):
        result = container_mod.cmd_detect(MagicMock())
        assert result == 0
        assert capsys.readouterr().out.strip() == "nvidia"


# Container management


class TestContainerExists:
    @patch("digue.container._docker_run")
    def test_true_when_inspect_succeeds(self, mock_docker):
        mock_docker.return_value = MagicMock(returncode=0, stdout="running\n")
        assert container_mod.container_exists(container_mod.CONTAINER_NAME) is True

    @patch("digue.container._docker_run")
    def test_false_when_inspect_fails(self, mock_docker):
        mock_docker.return_value = MagicMock(returncode=1, stdout="")
        assert container_mod.container_exists(container_mod.CONTAINER_NAME) is False


class TestDockerMissing:
    """Without the docker binary, every local-backend command used to die with
    a FileNotFoundError traceback, and `dictate` blamed the recorder."""

    def test_docker_run_raises_a_named_error(self):
        with (
            patch("subprocess.run", side_effect=FileNotFoundError(2, "No such file", "docker")),
            pytest.raises(container_mod.DockerNotFoundError, match="docker not found"),
        ):
            container_mod._docker_run(["ps"])

    def test_pull_image_raises_a_named_error(self):
        with (
            patch("digue.container.image_exists", return_value=False),
            patch("subprocess.run", side_effect=FileNotFoundError(2, "No such file", "docker")),
            pytest.raises(container_mod.DockerNotFoundError, match="docker not found"),
        ):
            container_mod.pull_image("ghcr.io/example/image")

    def test_main_reports_missing_docker_without_traceback(self, tmp_path, capsys):
        config_path = tmp_path / "config.toml"
        config_path.write_text('[server]\nbackend = "cpu"\n')
        with (
            patch.object(sys, "argv", ["digue", "--config", str(config_path), "server", "status"]),
            patch("digue.container.is_server_running", return_value=False),
            patch("subprocess.run", side_effect=FileNotFoundError(2, "No such file", "docker")),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_mod.main()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "Error: docker not found" in err
        assert "Traceback" not in err

    def test_dictate_blames_docker_not_the_recorder(self, tmp_path):
        config = _default_config()
        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch(
                "digue.container.ensure_server",
                side_effect=container_mod.DockerNotFoundError(container_mod.DOCKER_NOT_FOUND),
            ),
            patch("digue.notify.send_notification") as mock_notify,
        ):
            assert dictate_mod.dictate_toggle(config) == 1

        message = mock_notify.call_args.args[0]
        assert "docker not found" in message
        assert "Recorder not found" not in message
        assert not (tmp_path / "digue-daemon.pid").exists()


class TestContainerStatus:
    @patch("digue.container._docker_run")
    def test_returns_status(self, mock_docker):
        mock_docker.return_value = MagicMock(returncode=0, stdout="exited\n")
        assert container_mod.container_status(container_mod.CONTAINER_NAME) == "exited"

    @patch("digue.container._docker_run")
    def test_returns_none_when_missing(self, mock_docker):
        mock_docker.return_value = MagicMock(returncode=1)
        assert container_mod.container_status(container_mod.CONTAINER_NAME) is None


class TestCreateContainer:
    @patch("digue.container.download_model")
    @patch("digue.container.pull_image")
    @patch("digue.container._docker_run")
    def test_nvidia_uses_gpus_flag(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        container_mod.create_container(config, "nvidia")
        cmd = mock_docker.call_args[0][0]
        assert "--gpus" in cmd
        assert "all" in cmd
        assert any("main-cuda" in arg for arg in cmd)
        assert cmd[cmd.index("--name") + 1] == container_mod.CONTAINER_NAME == "digue-whisper.cpp"

    @patch("digue.container.download_model")
    @patch("digue.container.pull_image")
    @patch("digue.container._docker_run")
    def test_amd_uses_kfd_device(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        container_mod.create_container(config, "amd")
        cmd = mock_docker.call_args[0][0]
        assert "/dev/kfd" in cmd

    @patch("digue.container.download_model")
    @patch("digue.container.pull_image")
    @patch("digue.container._docker_run")
    def test_cpu_has_no_device_flags(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        container_mod.create_container(config, "cpu")
        cmd = mock_docker.call_args[0][0]
        assert "--device" not in cmd
        assert "--gpus" not in cmd

    @patch("digue.container.download_model")
    @patch("digue.container.pull_image")
    @patch("digue.container._docker_run")
    def test_raises_on_failure(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=1, stderr="permission denied")
        config = _default_config()
        with pytest.raises(RuntimeError, match="permission denied"):
            container_mod.create_container(config, "cpu")

    @patch("digue.container.download_model")
    @patch("digue.container.pull_image")
    @patch("digue.container._docker_run")
    def test_binds_to_localhost(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        container_mod.create_container(config, "cpu")
        cmd = mock_docker.call_args[0][0]
        port_binding = [arg for arg in cmd if "8178" in arg and "127.0.0.1" in arg]
        assert port_binding, "Port must bind to 127.0.0.1"

    @patch("digue.container.download_model")
    @patch("digue.container.pull_image")
    @patch("digue.container._docker_run")
    def test_binds_to_configured_ip(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        config["server"]["bind_ip"] = "192.168.1.10"
        container_mod.create_container(config, "cpu")
        cmd = mock_docker.call_args[0][0]
        assert "192.168.1.10:8178:8080" in cmd

    @patch("digue.container.download_model")
    @patch("digue.container.pull_image")
    @patch("digue.container._docker_run")
    def test_uses_configured_container_name(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        config["server"]["container_name"] = "whisper-lab"
        container_mod.create_container(config, "cpu")
        cmd = mock_docker.call_args[0][0]
        assert cmd[cmd.index("--name") + 1] == "whisper-lab"

    @patch("digue.container.download_model")
    @patch("digue.container.pull_image")
    @patch("digue.container._docker_run")
    def test_calls_pull_image(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        container_mod.create_container(config, "cpu")
        mock_pull.assert_called_once_with("ghcr.io/ggml-org/whisper.cpp:main-vulkan")

    @patch("digue.container.download_model")
    @patch("digue.container.pull_image")
    @patch("digue.container._docker_run")
    def test_downloads_model_when_missing(self, mock_docker, mock_pull, mock_download, tmp_path):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        config["server"]["data_dir"] = str(tmp_path)
        container_mod.create_container(config, "cpu")
        mock_download.assert_called_once_with("small-q8_0", tmp_path / "models", with_notification=False)

    @patch("digue.container.download_model")
    @patch("digue.container.pull_image")
    @patch("digue.container._docker_run")
    def test_skips_download_when_model_exists(self, mock_docker, mock_pull, mock_download, tmp_path):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        config["server"]["data_dir"] = str(tmp_path)
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "ggml-small-q8_0.bin").write_bytes(b"dummy")
        (models_dir / container_mod.VAD_MODEL_FILENAME).write_bytes(b"dummy")
        container_mod.create_container(config, "cpu")
        mock_download.assert_not_called()

    @patch("digue.container.download_model")
    @patch("digue.container.pull_image")
    @patch("digue.container._docker_run")
    def test_downloads_when_vad_is_missing(self, mock_docker, mock_pull, mock_download, tmp_path):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        config["server"]["data_dir"] = str(tmp_path)
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "ggml-small-q8_0.bin").write_bytes(b"dummy")

        container_mod.create_container(config, "cpu")

        mock_download.assert_called_once_with("small-q8_0", models_dir, with_notification=False)

    @patch("digue.container.download_model")
    @patch("digue.container.pull_image")
    @patch("digue.container._docker_run")
    def test_passes_resolved_thread_count(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        config["server"]["threads"] = 6
        container_mod.create_container(config, "cpu")
        cmd = mock_docker.call_args[0][0]
        assert cmd[cmd.index("--threads") + 1] == "6"

    @patch("digue.container.download_model")
    @patch("digue.container.pull_image")
    @patch("digue.container._docker_run")
    def test_auto_threads_use_physical_core_count(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        with patch("digue.container._physical_core_count", return_value=8):
            container_mod.create_container(config, "cpu")
            cmd = mock_docker.call_args[0][0]
            assert cmd[cmd.index("--threads") + 1] == "8"

    @patch("digue.container.download_model")
    @patch("digue.container.pull_image")
    @patch("digue.container._docker_run")
    def test_auto_threads_fall_back_to_server_default_when_detection_fails(self, mock_docker, mock_pull, mock_download):
        mock_docker.return_value = MagicMock(returncode=0)
        config = _default_config()
        with (
            patch("digue.container._physical_core_count", side_effect=OSError),
            patch("digue.container._cpuinfo_core_count", side_effect=ValueError),
            patch("os.sched_getaffinity", side_effect=AttributeError),
            patch("os.cpu_count", return_value=12),
        ):
            container_mod.create_container(config, "cpu")
            cmd = mock_docker.call_args[0][0]
            assert cmd[cmd.index("--threads") + 1] == "4"


class TestResolveThreads:
    def test_explicit_config_value_wins(self):
        config = _default_config()
        config["server"]["threads"] = 2
        assert container_mod.resolve_threads(config) == 2

    def write_cpu(self, cpu_root, index, core_cpus=None, siblings=None):
        topology = cpu_root / f"cpu{index}" / "topology"
        topology.mkdir(parents=True)
        if core_cpus is not None:
            (topology / "core_cpus_list").write_text(core_cpus)
        if siblings is not None:
            (topology / "thread_siblings_list").write_text(siblings)

    def test_counts_unique_core_cpus_lists(self):
        """4 physical cores with SMT 2x: the kernel repeats each sibling list across the 2 logical CPUs, so
        unique lists = cores. The Ryzen 8745HS looks like this (cpu0: "0,8" ... cpu15: "7,15")."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lists = ["0,8", "1,9", "2,10", "3,11", "4,12", "5,13", "6,14", "7,15"]
            for index in range(16):
                self.write_cpu(root, index, core_cpus=lists[index % 8])
            assert container_mod._physical_core_count(cpu_root=root) == 8

    def test_thread_siblings_list_is_the_fallback_in_sysfs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(4):
                self.write_cpu(root, index, siblings=f"{index % 2},{index % 2 + 2}")
            assert container_mod._physical_core_count(cpu_root=root) == 2

    def test_sysfs_without_topology_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cpu0").mkdir()
            with pytest.raises(OSError):
                container_mod._physical_core_count(cpu_root=root)

    def test_cpuinfo_counts_pairs_not_core_ids(self):
        """core_id repeats across sockets: 2 sockets x 2 cores = 4, not 2 (the psutil historical bug)."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cpuinfo = Path(tmp) / "cpuinfo"
            cpuinfo.write_text(
                "processor\t: 0\nphysical id\t: 0\ncore id\t\t: 0\n"
                "processor\t: 1\nphysical id\t: 0\ncore id\t\t: 1\n"
                "processor\t: 2\nphysical id\t: 1\ncore id\t\t: 0\n"
                "processor\t: 3\nphysical id\t: 1\ncore id\t\t: 1\n"
            )
            assert container_mod._cpuinfo_core_count(cpuinfo_path=cpuinfo) == 4

    def test_cpuinfo_counts_smt_siblings_once(self):
        """Ryzen 8745HS shape: 16 logical entries, core ids 0-11 repeated under SMT = 12 physical cores."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cpuinfo = Path(tmp) / "cpuinfo"
            lines = []
            for index in range(16):
                lines.append(f"processor\t: {index}\nphysical id\t: 0\ncore id\t\t: {index % 12}\n")
            cpuinfo.write_text("".join(lines))
            assert container_mod._cpuinfo_core_count(cpuinfo_path=cpuinfo) == 12

    def test_auto_prefers_sysfs_over_cpuinfo(self):
        config = _default_config()
        with (
            patch("digue.container._physical_core_count", return_value=8) as mock_sysfs,
            patch("digue.container._cpuinfo_core_count", side_effect=AssertionError("should not be called")),
        ):
            assert container_mod.resolve_threads(config) == 8
            mock_sysfs.assert_called_once()

    def test_auto_falls_back_to_cpuinfo_when_sysfs_empty(self):
        config = _default_config()
        with (
            patch("digue.container._physical_core_count", side_effect=OSError),
            patch("digue.container._cpuinfo_core_count", return_value=12) as mock_cpuinfo,
        ):
            assert container_mod.resolve_threads(config) == 12
            mock_cpuinfo.assert_called_once()

    def test_auto_falls_back_to_affinity_mask_without_topology(self):
        config = _default_config()
        with (
            patch("digue.container._physical_core_count", side_effect=OSError),
            patch("digue.container._cpuinfo_core_count", side_effect=ValueError),
            patch("os.sched_getaffinity", return_value=set(range(16))),
        ):
            assert container_mod.resolve_threads(config) == 16

    def test_auto_without_any_detection_caps_at_4(self):
        """No sysfs, no cpuinfo, no sched_getaffinity (macOS, non-Linux): the server default of min(4, ncpu)."""
        config = _default_config()
        with (
            patch("digue.container._physical_core_count", side_effect=OSError),
            patch("digue.container._cpuinfo_core_count", side_effect=ValueError),
            patch("os.sched_getaffinity", side_effect=AttributeError),
            patch("os.cpu_count", return_value=32),
        ):
            assert container_mod.resolve_threads(config) == 4


class TestCmdModels:
    def test_lists_every_available_model_and_marks_downloaded(self, capsys, tmp_path):
        from digue import AVAILABLE_MODELS

        config = _default_config()
        config["server"]["data_dir"] = str(tmp_path)
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "ggml-small.bin").write_bytes(b"x")

        config["server"]["backend"] = "cpu"  # resolves to small-q8_0: the starred line

        assert container_mod.cmd_models(MagicMock(), config) == 0

        out = capsys.readouterr().out
        by_name = {line[2:].split()[0]: line for line in out.splitlines() if line.strip()}
        assert list(by_name) == list(AVAILABLE_MODELS)
        assert "downloaded" in by_name["small"]
        assert "downloaded" not in by_name["tiny"]
        assert "74 MB" in by_name["tiny"]
        assert "547 MB" in by_name["large-v3-turbo-q5_0"]
        assert by_name["small-q8_0"].startswith("* ")
        assert all(line.startswith("  ") for name, line in by_name.items() if name != "small-q8_0")


class TestImageExists:
    @patch("digue.container._docker_run")
    def test_true_when_image_present(self, mock_docker):
        mock_docker.return_value = MagicMock(returncode=0)
        assert container_mod.image_exists("test:latest") is True

    @patch("digue.container._docker_run")
    def test_false_when_image_missing(self, mock_docker):
        mock_docker.return_value = MagicMock(returncode=1)
        assert container_mod.image_exists("test:latest") is False


class TestPullImage:
    @patch("digue.container.image_exists", return_value=True)
    @patch("subprocess.run")
    def test_skips_when_exists(self, mock_run, mock_exists):
        container_mod.pull_image("test:latest")
        mock_run.assert_not_called()

    @patch("digue.container.image_exists", return_value=False)
    @patch("subprocess.run", return_value=MagicMock(returncode=0))
    def test_pulls_when_missing(self, mock_run, mock_exists):
        container_mod.pull_image("test:latest")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["docker", "pull", "test:latest"]

    @patch("digue.container.image_exists", return_value=False)
    @patch("subprocess.run", return_value=MagicMock(returncode=1))
    def test_raises_on_failure(self, mock_run, mock_exists):
        with pytest.raises(RuntimeError, match="Failed to pull"):
            container_mod.pull_image("test:latest")


class TestDownloadProgressHook:
    def test_formats_percentage(self, capsys):
        hook = container_mod._download_progress_hook("test.bin")
        hook(50, 1024 * 1024, 100 * 1024 * 1024)  # 50MB of 100MB
        output = capsys.readouterr().err
        assert "50%" in output
        assert "test.bin" in output

    def test_terminal_line_has_carriage_return_and_bar(self, capsys):
        with patch("digue.notify._stderr_is_tty", return_value=True):
            hook = container_mod._download_progress_hook("test.bin")
            hook(25, 1024 * 1024, 100 * 1024 * 1024)  # 25%
        output = capsys.readouterr().err
        assert output.startswith("\r")
        assert "[" in output and "]" in output

    def test_non_terminal_prints_sparse_lines_without_bar(self, capsys):
        with patch("digue.notify._stderr_is_tty", return_value=False):
            hook = container_mod._download_progress_hook("test.bin")
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
        with patch("digue.notify._stderr_is_tty", return_value=True):
            hook = container_mod._download_progress_hook("test.bin")
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
            patch("digue.notify.send_notification", side_effect=lambda message, timeout_ms=0: messages.append(message)),
            patch("time.monotonic", side_effect=[1.0, 2.0, 3.0, 4.0]),
        ):
            hook = container_mod._download_progress_hook("test.bin", with_notification=True)
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
            patch("digue.notify.send_notification", side_effect=lambda message, timeout_ms=0: messages.append(message)),
            patch("time.monotonic", return_value=1.0),
        ):
            hook = container_mod._download_progress_hook("test.bin", with_notification=True)
            hook(1, 1024 * 1024, -1)
        assert messages == ["Downloading test.bin...       1.0 MB"]

    def test_notify_prints_carriage_return_on_tty(self, capsys):
        with patch("digue.notify._stderr_is_tty", return_value=True):
            notify_mod.send_notification("status message")
        err = capsys.readouterr().err
        assert err.startswith("\r")
        assert "[digue] status message" in err

    def test_notify_prints_plain_line_when_not_tty(self, capsys):
        with patch("digue.notify._stderr_is_tty", return_value=False):
            notify_mod.send_notification("status message")
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
            container_mod.download_model("small", tmp_path)

        assert model_path.read_bytes() == b"model"
        assert not model_path.with_suffix(".bin.part").exists()
        assert mock_urlopen.call_args.kwargs["timeout"] == container_mod.DOWNLOAD_TIMEOUT

    def test_partial_file_is_not_treated_as_ready_model(self, tmp_path):
        part_path = tmp_path / "ggml-small.bin.part"
        part_path.write_bytes(b"partial")
        (tmp_path / "ggml-silero-v6.2.0.bin").write_bytes(b"vad")
        response = MagicMock()
        response.headers = {}
        response.read.side_effect = [b"complete", b""]
        response.__enter__.return_value = response

        with patch("urllib.request.urlopen", return_value=response) as mock_urlopen:
            container_mod.download_model("small", tmp_path)

        mock_urlopen.assert_called_once()
        assert (tmp_path / "ggml-small.bin").read_bytes() == b"complete"

    def test_truncated_body_is_not_published(self, tmp_path):
        """Regression: http.client's read(amt) returns b"" when the server
        closes early (no IncompleteRead), so a 10-byte Content-Length answered
        with 4 bytes ended the loop normally and the short file became the
        model -- and a crash-looping container."""
        response = MagicMock()
        response.headers = {"Content-Length": "10"}
        response.read.side_effect = [b"half", b""]
        response.__enter__.return_value = response

        with (
            patch("urllib.request.urlopen", return_value=response),
            pytest.raises(RuntimeError, match="ended early: 4 of 10 bytes"),
        ):
            container_mod._download_file("http://example/model.bin", tmp_path / "ggml-small.bin", "ggml-small.bin")

        assert not (tmp_path / "ggml-small.bin").exists()
        assert not (tmp_path / "ggml-small.bin.part").exists()

    def test_interrupted_download_leaves_no_part_file(self, tmp_path):
        """A failed download left ggml-*.bin.part behind in the models dir."""
        response = MagicMock()
        response.headers = {"Content-Length": "10"}
        response.read.side_effect = [b"half", OSError("connection reset")]
        response.__enter__.return_value = response

        with patch("urllib.request.urlopen", return_value=response), pytest.raises(OSError, match="reset"):
            container_mod._download_file("http://example/model.bin", tmp_path / "ggml-small.bin", "ggml-small.bin")

        assert not (tmp_path / "ggml-small.bin").exists()
        assert not (tmp_path / "ggml-small.bin.part").exists()


# Server


class TestIsServerRunning:
    @patch("urllib.request.urlopen")
    def test_true_when_responds(self, mock_urlopen):
        config = _default_config()
        assert container_mod.is_server_running(config) is True

    @patch("urllib.request.urlopen", side_effect=OSError)
    def test_false_when_refused(self, mock_urlopen):
        config = _default_config()
        assert container_mod.is_server_running(config) is False


class TestWaitForServerOutput:
    """Same contract as every other progress print: \\r redraws only on a
    TTY; a captured stderr (hotkey daemon, journal) gets plain lines."""

    def run_wait(self, tty, capsys):
        self.answers = iter([False] * 51 + [True])
        with (
            patch("digue.container.is_server_running", side_effect=lambda _config: next(self.answers)),
            patch("digue.notify._stderr_is_tty", return_value=tty),
            patch("time.sleep"),
            patch("time.perf_counter", side_effect=range(1000)),
        ):
            assert container_mod._wait_for_server(_default_config(), verbose=True) is True
        return capsys.readouterr().err

    def test_captured_stderr_gets_plain_lines(self, capsys):
        err = self.run_wait(tty=False, capsys=capsys)
        assert "\r" not in err
        assert err.endswith("\n")
        assert "Waiting for model to load" in err and "Server ready" in err

    def test_tty_redraws_the_same_line(self, capsys):
        err = self.run_wait(tty=True, capsys=capsys)
        assert "\r  Waiting for model to load" in err
        assert "\rServer ready" in err

    def test_check_past_deadline_does_not_crash(self):
        """A slow check (up to the HTTP timeout) can push the clock past the deadline: the final sleep must not
        receive a negative duration (time.sleep raises ValueError on negative values)."""
        with (
            patch("digue.container.is_server_running", return_value=False),
            patch("digue.notify._stderr_is_tty", return_value=False),
            patch("time.sleep") as mock_sleep,
            patch("time.perf_counter", side_effect=[0.0, 179.9, 180.1, 180.1, 180.1]),
        ):
            assert container_mod._wait_for_server(_default_config()) is False
        mock_sleep.assert_not_called()


class TestServerHost:
    @pytest.mark.parametrize(
        ("bind_ip", "expected"),
        (("192.0.2.10", "192.0.2.10"), ("0.0.0.0", "127.0.0.1")),
    )
    def test_local_server_host_matches_reachable_bind_address(self, bind_ip, expected):
        config = _default_config()
        config["server"]["bind_ip"] = bind_ip
        assert container_mod.server_host(config) == expected


# Remote backend


class TestContainerFailures:
    @pytest.mark.parametrize(
        ("function", "args"),
        [
            (container_mod.start_container, ["start", container_mod.CONTAINER_NAME]),
            (container_mod.stop_container, ["stop", container_mod.CONTAINER_NAME]),
            (container_mod.remove_container, ["rm", "-f", container_mod.CONTAINER_NAME]),
        ],
    )
    @patch("digue.container._docker_run")
    def test_container_commands_raise_on_failure(self, mock_docker, function, args):
        mock_docker.return_value = MagicMock(returncode=1, stderr="docker failed\n")

        with pytest.raises(RuntimeError, match="docker failed"):
            function(container_mod.CONTAINER_NAME)

        assert mock_docker.call_args.args[0] == args

    @patch("digue.container._wait_for_server")
    @patch("digue.container.start_container", side_effect=RuntimeError("start failed"))
    @patch("digue.container.container_status", return_value="exited")
    @patch("digue.container.is_server_running", return_value=False)
    def test_ensure_server_does_not_wait_after_start_failure(
        self, mock_running, mock_status, mock_start, mock_wait, capsys
    ):
        with pytest.raises(RuntimeError, match="start failed"):
            container_mod.ensure_server(_default_config(), silent=True)

        mock_wait.assert_not_called()

    @pytest.mark.parametrize("silent", (False, True))
    @patch("digue.notify.notify_close")
    @patch("digue.notify.send_notification")
    @patch("digue.container.create_container")
    @patch("digue.container.start_container")
    @patch("digue.container._wait_for_server", return_value=True)
    @patch("digue.container.container_status", return_value="running")
    @patch("digue.container.is_server_running", return_value=False)
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

        assert container_mod.ensure_server(config, silent=silent) is None

        mock_wait.assert_called_once_with(config, verbose=not silent)
        mock_start.assert_not_called()
        mock_create.assert_not_called()
        if silent:
            mock_notify.assert_not_called()
            mock_notify_close.assert_not_called()
        else:
            mock_notify.assert_called_once_with("Server starting...")
            mock_notify_close.assert_called_once_with()

    @patch("digue.notify.send_notification")
    @patch("digue.container._wait_for_server", return_value=False)
    @patch("digue.container.container_status", return_value="running")
    @patch("digue.container.is_server_running", return_value=False)
    def test_ensure_server_reports_running_container_timeout(self, mock_running, mock_status, mock_wait, mock_notify):
        config = _default_config()

        assert container_mod.ensure_server(config) is None

        assert mock_notify.call_args_list == [
            call("Server starting..."),
            call(f"Server failed to start (see: docker logs {container_mod.CONTAINER_NAME})", timeout_ms=10000),
        ]

    @pytest.mark.parametrize(
        ("status", "expected_backend"),
        [("exited", None), (None, "cpu")],
    )
    @patch("digue.container.resolve_backend", return_value="cpu")
    @patch("digue.container.create_container")
    @patch("digue.container.start_container")
    @patch("digue.container._wait_for_server", return_value=True)
    @patch("digue.container.container_status")
    @patch("digue.container.is_server_running", return_value=False)
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

        assert container_mod.ensure_server(config, silent=True) == expected_backend

        if status == "exited":
            mock_start.assert_called_once_with(container_mod.CONTAINER_NAME)
            mock_create.assert_not_called()
        else:
            mock_start.assert_not_called()
            mock_create.assert_called_once_with(config, "cpu", with_notification=False)

    @patch("digue.notify.send_notification")
    @patch("digue.container._wait_for_server")
    @patch("digue.container.container_status", return_value="paused")
    @patch("digue.container.is_server_running", return_value=False)
    def test_ensure_server_rejects_invalid_container_state(self, mock_running, mock_status, mock_wait, mock_notify):
        assert container_mod.ensure_server(_default_config()) is None

        mock_notify.assert_called_once_with("Container in unexpected state: paused", timeout_ms=5000)
        mock_wait.assert_not_called()

    @patch("digue.container.stop_container", side_effect=RuntimeError("stop failed"))
    def test_cmd_server_stop_does_not_report_false_success(self, mock_stop, capsys):
        assert container_mod.cmd_server_stop(MagicMock(), _default_config()) == 1
        assert "Error: stop failed" in capsys.readouterr().err

    @patch("digue.container.remove_container", side_effect=RuntimeError("remove failed"))
    def test_cmd_server_destroy_does_not_report_false_success(self, mock_remove, capsys):
        assert container_mod.cmd_server_destroy(MagicMock(), _default_config()) == 1
        assert "Error: remove failed" in capsys.readouterr().err

    @patch("digue.container.start_container", side_effect=RuntimeError("start failed"))
    @patch("digue.container.container_status", return_value="exited")
    @patch("digue.container.is_server_running", return_value=False)
    def test_cmd_server_start_does_not_report_false_success(self, mock_running, mock_status, mock_start, capsys):
        assert container_mod.cmd_server_start(MagicMock(), _default_config()) == 1
        assert "Error: start failed" in capsys.readouterr().err

    @patch("digue.container._wait_for_server", return_value=False)
    @patch("digue.container.start_container")
    @patch("digue.container.container_status", return_value="exited")
    @patch("digue.container.is_server_running", return_value=False)
    def test_cmd_server_start_timeout_hint_names_the_digue_container(
        self, mock_running, mock_status, mock_start, mock_wait, capsys
    ):
        assert container_mod.cmd_server_start(MagicMock(), _default_config()) == 1
        err = capsys.readouterr().err
        assert f"docker logs {container_mod.CONTAINER_NAME}" in err
        assert "whisper-server" not in err


class TestServerStartImage:
    """`server start --image`, and the recreation of a container created
    from another image than the one the config now selects."""

    def test_parser_accepts_image(self):
        args = cli_mod.create_parser().parse_args(["server", "start", "--image", "ghcr.io/ggml-org/whisper.cpp:main"])
        assert args.server_action == "start"
        assert args.image == "ghcr.io/ggml-org/whisper.cpp:main"
        assert cli_mod.create_parser().parse_args(["server", "start"]).image is None

    @patch("digue.container._wait_for_server", return_value=True)
    @patch("digue.container.create_container")
    @patch("digue.container.remove_container")
    @patch("digue.container.container_image", return_value="ghcr.io/ggml-org/whisper.cpp:main-vulkan")
    @patch("digue.container.container_status", return_value="exited")
    @patch("digue.container.is_server_running", return_value=False)
    def test_image_option_recreates_a_container_from_another_image(
        self, mock_running, mock_status, mock_image, mock_remove, mock_create, mock_wait, capsys
    ):
        config = _default_config()
        config["server"]["backend"] = "cpu"
        args = MagicMock(image="ghcr.io/ggml-org/whisper.cpp:main")

        assert container_mod.cmd_server_start(args, config) == 0

        mock_remove.assert_called_once()
        mock_create.assert_called_once_with(config, "cpu")
        assert config["server"]["image"] == "ghcr.io/ggml-org/whisper.cpp:main"
        err = capsys.readouterr().err
        assert "recreating with ghcr.io/ggml-org/whisper.cpp:main" in err

    @patch("digue.container._wait_for_server", return_value=True)
    @patch("digue.container.create_container")
    @patch("digue.container.remove_container")
    @patch("digue.container.container_image", return_value="ghcr.io/ggml-org/whisper.cpp:main-vulkan")
    @patch("digue.container.container_status", return_value="running")
    @patch("digue.container.is_server_running", return_value=True)
    def test_config_image_change_recreates_even_a_running_server(
        self, mock_running, mock_status, mock_image, mock_remove, mock_create, mock_wait
    ):
        """`docker start` keeps the original image, so a new `image` in the
        config did nothing until the container was destroyed by hand."""
        config = _default_config()
        config["server"]["backend"] = "cpu"
        config["server"]["image"] = "ghcr.io/ggml-org/whisper.cpp:main"

        assert container_mod.cmd_server_start(MagicMock(image=None), config) == 0

        mock_remove.assert_called_once()
        mock_create.assert_called_once_with(config, "cpu")

    @patch("digue.container.create_container")
    @patch("digue.container.remove_container")
    @patch("digue.container.container_image", return_value="ghcr.io/ggml-org/whisper.cpp:main-vulkan")
    @patch("digue.container.container_status", return_value="running")
    @patch("digue.container.is_server_running", return_value=True)
    def test_matching_image_is_left_alone(
        self, mock_running, mock_status, mock_image, mock_remove, mock_create, capsys
    ):
        config = _default_config()
        config["server"]["backend"] = "cpu"  # resolves to main-vulkan, the container's image

        assert container_mod.cmd_server_start(MagicMock(image=None), config) == 0

        mock_remove.assert_not_called()
        mock_create.assert_not_called()
        assert "already running" in capsys.readouterr().err

    @patch("digue.container.container_image", return_value="ghcr.io/ggml-org/whisper.cpp:main-vulkan")
    @patch("digue.container.container_status", return_value="running")
    @patch("digue.container.is_server_running", return_value=True)
    def test_remote_backend_ignores_the_image_check(self, mock_running, mock_status, mock_image, capsys):
        config = _default_config()
        config["server"]["backend"] = "remote"
        assert container_mod.cmd_server_start(MagicMock(image="x"), config) == 0
        mock_status.assert_not_called()

    @patch("digue.container._wait_for_server", return_value=True)
    @patch("digue.container.start_container")
    @patch("digue.container.remove_container")
    @patch("digue.container.container_image", return_value="ghcr.io/ggml-org/whisper.cpp:main-vulkan")
    @patch("digue.container.container_status", return_value="exited")
    @patch("digue.container.is_server_running", return_value=False)
    def test_ensure_server_only_warns_about_a_mismatch(
        self, mock_running, mock_status, mock_image, mock_remove, mock_start, mock_wait, capsys
    ):
        """Behind the dictation hotkey a multi-GB pull is not what a keypress
        asked for: the mismatch is reported and `server start` recreates."""
        config = _default_config()
        config["server"]["backend"] = "cpu"
        config["server"]["image"] = "ghcr.io/ggml-org/whisper.cpp:main"

        container_mod.ensure_server(config, silent=True)

        mock_remove.assert_not_called()
        mock_start.assert_called_once()
        err = capsys.readouterr().err
        assert "runs ghcr.io/ggml-org/whisper.cpp:main-vulkan" in err
        assert "digue server start" in err

    @patch("digue.container._docker_run")
    def test_container_image_reads_docker_inspect(self, mock_docker, monkeypatch):
        monkeypatch.setattr(container_mod, "container_image", _real_container_image)
        mock_docker.return_value = MagicMock(returncode=0, stdout="ghcr.io/ggml-org/whisper.cpp:main\n")
        assert container_mod.container_image("digue-whisper.cpp") == "ghcr.io/ggml-org/whisper.cpp:main"
        assert mock_docker.call_args.args[0] == [
            "inspect",
            "--format",
            "{{.Config.Image}}",
            container_mod.CONTAINER_NAME,
        ]
        mock_docker.return_value = MagicMock(returncode=1, stdout="")
        assert container_mod.container_image("digue-whisper.cpp") is None

    @patch("digue.container._wait_for_server", return_value=True)
    @patch("digue.container.create_container")
    @patch("digue.container.container_status", return_value=None)
    @patch("digue.container.is_server_running", return_value=False)
    def test_container_name_option_is_stored_on_the_config(self, mock_running, mock_status, mock_create, mock_wait):
        config = _default_config()
        config["server"]["backend"] = "cpu"
        args = MagicMock(image=None, container_name="whisper-lab")
        apply_cli_overrides(args, config)

        assert container_mod.cmd_server_start(args, config) == 0

        assert config["server"]["container_name"] == "whisper-lab"
        mock_status.assert_called_once_with("whisper-lab")
        mock_create.assert_called_once_with(config, "cpu")

    @patch("digue.container._wait_for_server", return_value=True)
    @patch("digue.container.create_container")
    @patch("digue.container.container_status", return_value=None)
    @patch("digue.container.is_server_running", return_value=False)
    def test_container_name_option_with_an_invalid_docker_name_is_rejected(
        self, mock_running, mock_status, mock_create, mock_wait
    ):
        """`-n` goes through apply_cli_overrides like every CLI override; without that check it only got validated
        when it came from the TOML, and `server start -n 'bad name'` answered "already running" with a live server."""
        config = _default_config()
        config["server"]["backend"] = "cpu"
        config["server"]["container_name"] = "digue-whisper.cpp"
        args = MagicMock(image=None, container_name="bad name")

        with pytest.raises(ValueError, match="Invalid server.container_name"):
            apply_cli_overrides(args, config)

        assert config["server"]["container_name"] == "digue-whisper.cpp"


class TestRemoteBackend:
    def test_resolve_backend_from_config(self):
        config = _default_config()
        config["server"]["backend"] = "remote"
        assert container_mod.resolve_backend(config) == "remote"

    @patch("digue.container.is_server_running", return_value=False)
    @patch("digue.container.container_status")
    def test_ensure_server_remote_never_touches_container(self, mock_status, mock_running):
        config = _default_config()
        config["server"]["backend"] = "remote"
        assert container_mod.ensure_server(config, silent=True) is None
        mock_status.assert_not_called()

    @patch("digue.container._docker_run")
    def test_create_container_remote_raises(self, mock_docker):
        config = _default_config()
        with pytest.raises(RuntimeError, match="remote"):
            container_mod.create_container(config, "remote")
        mock_docker.assert_not_called()

    def test_hint_remote_mentions_tunnel(self):
        config = _default_config()
        config["server"]["backend"] = "remote"
        hint = container_mod.server_not_running_hint(config)
        assert "ssh -NfL" in hint
        assert "8178" in hint

    def test_hint_local_suggests_start(self):
        config = _default_config()
        assert container_mod.server_not_running_hint(config) == "Run: digue server start"

    @patch("digue.container.is_server_running", return_value=True)
    def test_cmd_server_status_remote(self, mock_running, capsys):
        config = _default_config()
        config["server"]["backend"] = "remote"
        assert container_mod.cmd_server_status(MagicMock(), config) == 0
        assert "remote" in capsys.readouterr().err

    @patch("digue.container.is_server_running", return_value=False)
    def test_cmd_server_status_remote_not_responding(self, mock_running, capsys):
        config = _default_config()
        config["server"]["backend"] = "remote"
        assert container_mod.cmd_server_status(MagicMock(), config) == 1


class TestRemoteHost:
    @patch("urllib.request.urlopen")
    @patch("digue.container.detect_backend")
    def test_server_probes_do_not_detect_auto_backend(self, mock_detect_backend, mock_urlopen):
        config = _default_config()

        container_mod.is_server_running(config)
        container_mod.server_url(config)
        container_mod.server_not_running_hint(config)

        mock_detect_backend.assert_not_called()

    def test_server_host_localhost_for_local_backends(self):
        config = _default_config()
        config["server"]["remote_host"] = "10.0.0.5"
        assert container_mod.server_host(config) == "127.0.0.1"

    def test_server_host_default_tunnel(self):
        config = _default_config()
        config["server"]["backend"] = "remote"
        assert container_mod.server_host(config) == "127.0.0.1"

    def test_server_host_remote_lan(self):
        config = _default_config()
        config["server"]["backend"] = "remote"
        config["server"]["remote_host"] = "10.0.0.5"
        assert container_mod.server_host(config) == "10.0.0.5"

    def test_server_url_uses_remote_host(self):
        config = _default_config()
        config["server"]["backend"] = "remote"
        config["server"]["remote_host"] = "desktop.lan"
        assert container_mod.server_url(config) == "http://desktop.lan:8178/inference"

    @patch("urllib.request.urlopen")
    def test_is_server_running_probes_remote_host(self, mock_urlopen):
        config = _default_config()
        config["server"]["backend"] = "remote"
        config["server"]["remote_host"] = "10.0.0.5"
        container_mod.is_server_running(config)
        url = mock_urlopen.call_args[0][0]
        assert url == "http://10.0.0.5:8178/"

    def test_hint_remote_lan_mentions_host(self):
        config = _default_config()
        config["server"]["backend"] = "remote"
        config["server"]["remote_host"] = "10.0.0.5"
        hint = container_mod.server_not_running_hint(config)
        assert "10.0.0.5" in hint
        assert "ssh -NfL" not in hint

    def test_hint_remote_tunnel_keeps_ssh(self):
        config = _default_config()
        config["server"]["backend"] = "remote"
        hint = container_mod.server_not_running_hint(config)
        assert "ssh -NfL" in hint

    @patch("digue.container.is_server_running", return_value=False)
    def test_cmd_server_status_remote_shows_host(self, mock_running, capsys):
        config = _default_config()
        config["server"]["backend"] = "remote"
        config["server"]["remote_host"] = "10.0.0.5"
        container_mod.cmd_server_status(MagicMock(), config)
        assert "10.0.0.5:8178" in capsys.readouterr().err


class TestCmdDoctor:
    def test_prints_resolved_config_without_crashing(self, capsys):
        config = _default_config()
        with (
            patch("digue.container.image_exists", return_value=False),
            patch("shutil.which", return_value=None),
        ):
            result = container_mod.cmd_doctor(MagicMock(), config)
        assert result == 0
        err = capsys.readouterr().err
        assert "Language: auto" in err

    def test_lists_both_supported_recorders(self, capsys):
        """arecord is the documented fallback recorder; doctor only checked pw-record."""
        config = _default_config()
        with (
            patch("digue.container.image_exists", return_value=False),
            patch("shutil.which", return_value=None),
        ):
            container_mod.cmd_doctor(MagicMock(), config)
        err = capsys.readouterr().err
        assert "] pw-record:" in err
        assert "] arecord:" in err

    def test_tests_a_custom_configured_image_too(self, capsys):
        """A custom `image` (a local build, another tag) is the one whose
        compatibility matters; it must not be skipped because it is not in
        the built-in list."""
        config = _default_config()
        config["server"]["backend"] = "cpu"
        config["server"]["image"] = "localhost/whisper:custom"
        with (
            patch("digue.container.image_exists", return_value=False),
            patch("shutil.which", return_value=None),
            patch("digue.container.detect_backend", return_value="cpu"),
        ):
            container_mod.cmd_doctor(MagicMock(), config)
        err = capsys.readouterr().err
        assert "Testing configured image..." in err
        assert "Image: localhost/whisper:custom" in err

    def test_prints_selected_config_path(self, tmp_path, capsys):
        target = tmp_path / "selected.toml"
        target.write_text("")
        args = MagicMock(config=str(target))
        with patch("digue.container.image_exists", return_value=False), patch("shutil.which", return_value=None):
            container_mod.cmd_doctor(args, load_config(target))
        assert str(target) in capsys.readouterr().err
