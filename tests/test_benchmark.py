"""Tests for digue benchmark and benchmark_models."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

import benchmark_models
import digue
from digue import container as container_mod
from digue.config import _default_config


class TestBenchmarkTempFiles:
    def test_recorded_benchmark_audio_lives_in_the_private_runtime_dir(self, tmp_path):
        """A fixed name in /tmp could be a symlink planted by another local
        user; the runtime dir is 0700 and owned by the user."""
        config = _default_config()
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


class TestBenchmarkContainerState:
    def test_absent_container_is_only_cleaned_up(self):
        with (
            patch("digue.container.container_status", return_value=None),
            patch("digue.container.container_exists", return_value=True),
            patch("digue.container.remove_container") as mock_remove,
            patch("digue.container._rename_container") as mock_rename,
            patch("digue.container.stop_container") as mock_stop,
            patch("digue.container.start_container") as mock_start,
            container_mod.preserve_container_for_benchmark(),
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
            patch("digue.container.container_status", side_effect=lambda: next(statuses)),
            patch("digue.container.container_exists", side_effect=lambda: next(existence)),
            patch("digue.container.remove_container") as mock_remove,
            patch("digue.container._rename_container") as mock_rename,
            patch("digue.container.stop_container") as mock_stop,
            patch("digue.container.start_container") as mock_start,
            patch("digue.os.getpid", return_value=123),
            pytest.raises(KeyboardInterrupt),
            container_mod.preserve_container_for_benchmark(),
        ):
            raise KeyboardInterrupt

        assert mock_rename.call_args_list == [
            ((container_mod.CONTAINER_NAME, "digue-benchmark-backup-123"),),
            (("digue-benchmark-backup-123", container_mod.CONTAINER_NAME),),
        ]
        mock_remove.assert_called_once_with()
        mock_stop.assert_not_called()
        mock_start.assert_not_called()

    def test_running_container_is_stopped_then_restored_running(self):
        with (
            patch("digue.container.container_status", return_value="running"),
            patch("digue.container.container_exists", return_value=False),
            patch("digue.container._rename_container") as mock_rename,
            patch("digue.container.stop_container") as mock_stop,
            patch("digue.container.start_container") as mock_start,
            patch("digue.os.getpid", return_value=456),
            container_mod.preserve_container_for_benchmark(),
        ):
            pass

        mock_stop.assert_called_once_with()
        assert mock_rename.call_args_list == [
            ((container_mod.CONTAINER_NAME, "digue-benchmark-backup-456"),),
            (("digue-benchmark-backup-456", container_mod.CONTAINER_NAME),),
        ]
        mock_start.assert_called_once_with()


class TestRunBenchmarkLanguage:
    def test_language_default_comes_from_transcribe_section(self, tmp_path, capsys):
        config = _default_config()
        with (
            patch("digue.container.download_model"),
            patch("digue.container.preserve_container_for_benchmark"),
            patch("digue.container.container_exists", return_value=True),
            patch("digue.container.remove_container"),
            patch("digue.container.create_container"),
            patch("digue.container._wait_for_server", return_value=True),
            patch("digue._benchmark_run", return_value=[]),
            patch("digue.container.detect_backend", return_value="cpu"),
        ):
            digue.run_benchmark(tmp_path / "no-audio.wav", config)
        err = capsys.readouterr().err
        assert "digue benchmark" in err

    def test_removes_benchmark_container_when_transcription_is_interrupted(self, tmp_path):
        config = _default_config()
        with (
            patch("digue.container.download_model"),
            patch("digue.container.preserve_container_for_benchmark"),
            patch("digue.container.container_exists", return_value=True),
            patch("digue.container.remove_container") as mock_remove,
            patch("digue.container.create_container"),
            patch("digue.container._wait_for_server", return_value=True),
            patch("digue._benchmark_run", side_effect=KeyboardInterrupt),
            patch("digue.container.detect_backend", return_value="cpu"),
            pytest.raises(KeyboardInterrupt),
        ):
            digue.run_benchmark(tmp_path / "audio.wav", config)

        mock_remove.assert_called_once_with()


class TestBenchmarkRespectsConfig:
    def test_backend_override_is_honored(self, tmp_path):
        """A forced backend (e.g. cpu with image = main on a Kaby Lake iGPU)
        must not be bypassed by hardware detection: the GPU cases would run
        with an image the machine cannot execute."""
        config = _default_config()
        config["server"]["backend"] = "cpu"
        with (
            patch("digue.container.download_model"),
            patch("digue.container.preserve_container_for_benchmark"),
            patch("digue.container.container_exists", return_value=False),
            patch("digue.container.create_container") as mock_create,
            patch("digue.container._wait_for_server", return_value=True),
            patch("digue._benchmark_run", return_value=[]),
            patch("digue.container.detect_backend", return_value="intel"),
        ):
            digue.run_benchmark(tmp_path / "audio.wav", config)

        backends = [recorded_call.args[1] for recorded_call in mock_create.call_args_list]
        assert backends == ["cpu", "cpu"]

    def test_custom_image_applies_only_to_the_resolved_backend(self, tmp_path, capsys):
        """server.image is a single global override that only makes sense for
        the backend the config resolved to (e.g. image "main" pinned for a
        Kaby Lake CPU): other cases must fall back to DOCKER_IMAGES."""
        config = _default_config()
        config["server"]["backend"] = "amd"
        config["server"]["image"] = "x"
        with (
            patch("digue.container.download_model"),
            patch("digue.container.preserve_container_for_benchmark"),
            patch("digue.container.container_exists", return_value=False),
            patch("digue.container.create_container") as mock_create,
            patch("digue.container._wait_for_server", return_value=True),
            patch("digue._benchmark_run", return_value=[]),
            patch("digue.container.detect_backend", return_value="amd"),
        ):
            digue.run_benchmark(tmp_path / "audio.wav", config)

        images = {backend: [] for backend in ("cpu", "amd")}
        for create_call in mock_create.call_args_list:
            bench_config, backend = create_call.args
            images[backend].append(container_mod.resolve_image(backend, bench_config))
            assert bench_config["server"]["image"] == ("x" if backend == "amd" else "")
        assert images == {"cpu": [container_mod.DOCKER_IMAGES["cpu"]] * 2, "amd": ["x", "x"]}
        err = capsys.readouterr().err
        assert f"Image: {container_mod.DOCKER_IMAGES['cpu']}" in err
        assert "Image: x" in err

    @patch("time.sleep")
    @patch("subprocess.Popen")
    def test_microphone_recording_uses_the_configured_recorder(self, mock_popen, mock_sleep, tmp_path):
        config = _default_config()
        config["dictate"]["recorder"] = "arecord"

        digue.record_benchmark_audio(tmp_path / "bench.wav", config=config)

        assert mock_popen.call_args.args[0][0] == "arecord"


class TestBenchmarkModels:
    def test_case_removes_container_when_transcription_is_interrupted(self, tmp_path):
        config = _default_config()
        config["server"]["data_dir"] = str(tmp_path)
        with (
            patch("benchmark_models.container_mod.create_container"),
            patch("benchmark_models.container_mod._wait_for_server", return_value=True),
            patch("benchmark_models.digue.transcribe", side_effect=KeyboardInterrupt),
            patch("benchmark_models.container_mod.container_exists", return_value=True),
            patch("benchmark_models.container_mod.remove_container") as mock_remove,
            pytest.raises(KeyboardInterrupt),
        ):
            benchmark_models.benchmark_case(config, "cpu", "small")

        mock_remove.assert_called_once_with()

    def test_main_preserves_previous_container_on_interrupt(self):
        config = _default_config()
        manager = MagicMock()
        manager.__enter__.return_value = None
        manager.__exit__.return_value = False
        with (
            patch("benchmark_models.create_parser") as mock_parser,
            patch("benchmark_models.download_sample"),
            patch("benchmark_models.load_config", return_value=config),
            patch("benchmark_models.container_mod.detect_backend", return_value="cpu"),
            patch("benchmark_models.container_mod.preserve_container_for_benchmark", return_value=manager),
            patch("benchmark_models.benchmark_case", side_effect=KeyboardInterrupt),
        ):
            mock_parser.return_value.parse_args.return_value = argparse.Namespace(
                backends=["cpu"], models=["small"], runs=1
            )
            benchmark_models.main()

        manager.__exit__.assert_called_once()


class TestBenchmarkModelsConfig:
    def test_main_rejects_remote_before_downloading_sample(self, capsys):
        config = _default_config()
        config["server"]["backend"] = "remote"
        with (
            patch("benchmark_models.create_parser") as mock_parser,
            patch("benchmark_models.download_sample") as mock_download,
            patch("benchmark_models.load_config", return_value=config),
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
        config = _default_config()
        config["server"]["backend"] = "cpu"
        with (
            patch("benchmark_models.create_parser") as mock_parser,
            patch("benchmark_models.download_sample"),
            patch("benchmark_models.load_config", return_value=config),
            patch("benchmark_models.container_mod.detect_backend", return_value="intel"),
            patch("benchmark_models.container_mod.preserve_container_for_benchmark"),
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
        config = _default_config()
        config["server"]["backend"] = "cpu"
        config["server"]["image"] = "x"
        config["server"]["data_dir"] = str(tmp_path)
        with (
            patch("benchmark_models.container_mod.create_container") as mock_create,
            patch("benchmark_models.container_mod._wait_for_server", return_value=True),
            patch("benchmark_models.digue.transcribe", return_value="hello"),
            patch("benchmark_models.container_mod.container_exists", return_value=False),
        ):
            assert benchmark_models.benchmark_case(config, "intel", "small") is not None

        bench_config, backend = mock_create.call_args.args
        assert backend == "intel"
        assert bench_config["server"]["image"] == ""
        assert container_mod.resolve_image(backend, bench_config) == container_mod.DOCKER_IMAGES["intel"]

    def test_case_keeps_custom_image_for_resolved_backend(self, tmp_path):
        config = _default_config()
        config["server"]["backend"] = "cpu"
        config["server"]["image"] = "x"
        config["server"]["data_dir"] = str(tmp_path)
        with (
            patch("benchmark_models.container_mod.create_container") as mock_create,
            patch("benchmark_models.container_mod._wait_for_server", return_value=True),
            patch("benchmark_models.digue.transcribe", return_value="hello"),
            patch("benchmark_models.container_mod.container_exists", return_value=False),
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
