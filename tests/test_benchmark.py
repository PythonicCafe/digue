"""Tests for digue benchmark."""

import argparse
import json
from unittest.mock import MagicMock, patch

import pytest

import digue
from digue import benchmark as benchmark_mod
from digue import container as container_mod
from digue.config import _default_config


class TestBenchmarkTempFiles:
    def test_recorded_benchmark_audio_lives_in_the_private_runtime_dir(self, tmp_path):
        """A fixed name in /tmp could be a symlink planted by another local
        user; the runtime dir is 0700 and owned by the user."""
        config = _default_config()
        config["server"]["backend"] = "cpu"
        args = argparse.Namespace(audio=None, sample=False, backends=None, models=None, runs=3, json=False)

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.benchmark.record_benchmark_audio") as mock_record,
            patch("digue.benchmark.run_benchmark"),
        ):
            assert benchmark_mod.cmd_benchmark(args, config) == 0

        assert mock_record.call_args[0][0].parent == tmp_path

    def test_sample_lives_in_the_private_runtime_dir(self, tmp_path):
        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            assert benchmark_mod.sample_path().parent == tmp_path


class TestCmdBenchmarkOptions:
    def make_args(self, **overrides):
        values = {"audio": None, "sample": False, "backends": None, "models": None, "runs": 3, "json": False}
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_sample_downloads_the_jfk_audio_instead_of_recording(self, tmp_path):
        config = _default_config()
        config["server"]["backend"] = "cpu"
        with (
            patch("digue.benchmark.download_sample", return_value=tmp_path / "jfk.wav") as mock_download,
            patch("digue.benchmark.record_benchmark_audio") as mock_record,
            patch("digue.benchmark.run_benchmark", return_value=[]) as mock_run,
        ):
            assert benchmark_mod.cmd_benchmark(self.make_args(sample=True), config) == 0

        mock_download.assert_called_once_with()
        mock_record.assert_not_called()
        assert mock_run.call_args.args[0] == tmp_path / "jfk.wav"

    def test_options_are_forwarded_and_all_expands_to_every_model(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"x")
        config = _default_config()
        config["server"]["backend"] = "cpu"
        args = self.make_args(audio=audio, backends=["amd", "cpu"], models=["all"], runs=2)
        with patch("digue.benchmark.run_benchmark", return_value=[]) as mock_run:
            assert benchmark_mod.cmd_benchmark(args, config) == 0

        mock_run.assert_called_once_with(
            audio, config, backends=["amd", "cpu"], models=list(digue.AVAILABLE_MODELS), runs=2
        )

    def test_json_prints_the_results_on_stdout(self, tmp_path, capsys):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"x")
        config = _default_config()
        config["server"]["backend"] = "cpu"
        results = [{"backend": "cpu", "model": "small", "avg_ms": 5, "runs_ms": [5], "text": "t"}]
        with patch("digue.benchmark.run_benchmark", return_value=results):
            assert benchmark_mod.cmd_benchmark(self.make_args(audio=audio, json=True), config) == 0

        assert json.loads(capsys.readouterr().out) == results

    def test_without_json_stdout_stays_empty(self, tmp_path, capsys):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"x")
        config = _default_config()
        config["server"]["backend"] = "cpu"
        with patch("digue.benchmark.run_benchmark", return_value=[{"backend": "cpu"}]):
            assert benchmark_mod.cmd_benchmark(self.make_args(audio=audio), config) == 0

        assert capsys.readouterr().out == ""

    def test_interrupt_still_prints_json_and_exits_130(self, tmp_path, capsys):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"x")
        config = _default_config()
        config["server"]["backend"] = "cpu"
        with patch("digue.benchmark.run_benchmark", side_effect=KeyboardInterrupt):
            assert benchmark_mod.cmd_benchmark(self.make_args(audio=audio, json=True), config) == 130

        assert json.loads(capsys.readouterr().out) == []


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

        mock_remove.assert_called_once_with(container_mod.CONTAINER_NAME)
        mock_rename.assert_not_called()
        mock_stop.assert_not_called()
        mock_start.assert_not_called()

    def test_stopped_container_is_restored_stopped_after_exception(self):
        statuses = iter(["exited"])
        existence = iter([True])
        with (
            patch("digue.container.container_status", side_effect=lambda *_args, **_kwargs: next(statuses)),
            patch("digue.container.container_exists", side_effect=lambda *_args, **_kwargs: next(existence)),
            patch("digue.container.remove_container") as mock_remove,
            patch("digue.container._rename_container") as mock_rename,
            patch("digue.container.stop_container") as mock_stop,
            patch("digue.container.start_container") as mock_start,
            patch("digue.container.os.getpid", return_value=123),
            pytest.raises(KeyboardInterrupt),
            container_mod.preserve_container_for_benchmark(),
        ):
            raise KeyboardInterrupt

        assert mock_rename.call_args_list == [
            ((container_mod.CONTAINER_NAME, f"{container_mod.CONTAINER_NAME}-benchmark-backup-123"),),
            ((f"{container_mod.CONTAINER_NAME}-benchmark-backup-123", container_mod.CONTAINER_NAME),),
        ]
        mock_remove.assert_called_once_with(container_mod.CONTAINER_NAME)
        mock_stop.assert_not_called()
        mock_start.assert_not_called()

    def test_running_container_is_stopped_then_restored_running(self):
        with (
            patch("digue.container.container_status", return_value="running"),
            patch("digue.container.container_exists", return_value=False),
            patch("digue.container._rename_container") as mock_rename,
            patch("digue.container.stop_container") as mock_stop,
            patch("digue.container.start_container") as mock_start,
            patch("digue.container.os.getpid", return_value=456),
            container_mod.preserve_container_for_benchmark(),
        ):
            pass

        mock_stop.assert_called_once_with(container_mod.CONTAINER_NAME)
        assert mock_rename.call_args_list == [
            ((container_mod.CONTAINER_NAME, f"{container_mod.CONTAINER_NAME}-benchmark-backup-456"),),
            ((f"{container_mod.CONTAINER_NAME}-benchmark-backup-456", container_mod.CONTAINER_NAME),),
        ]
        mock_start.assert_called_once_with(container_mod.CONTAINER_NAME)


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
            patch("digue.benchmark._benchmark_run", return_value=[]),
            patch("digue.container.detect_backend", return_value="cpu"),
        ):
            benchmark_mod.run_benchmark(tmp_path / "no-audio.wav", config)
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
            patch("digue.benchmark._benchmark_run", side_effect=KeyboardInterrupt),
            patch("digue.container.detect_backend", return_value="cpu"),
            pytest.raises(KeyboardInterrupt),
        ):
            benchmark_mod.run_benchmark(tmp_path / "audio.wav", config)

        # the case removes its own container before the interrupt propagates
        mock_remove.assert_called_once_with(container_mod.CONTAINER_NAME)


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
            patch("digue.benchmark._benchmark_run", return_value=[]),
            patch("digue.container.detect_backend", return_value="intel"),
        ):
            benchmark_mod.run_benchmark(tmp_path / "audio.wav", config)

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
            patch("digue.benchmark._benchmark_run", return_value=[]),
            patch("digue.container.detect_backend", return_value="amd"),
        ):
            benchmark_mod.run_benchmark(tmp_path / "audio.wav", config)

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

        benchmark_mod.record_benchmark_audio(tmp_path / "bench.wav", config=config)

        assert mock_popen.call_args.args[0][0] == "arecord"


class TestBenchmarkCase:
    """One backend+model case: the container is created with the case config
    (model swapped in, image override only for the resolved backend), waited
    on with that same config, and always removed afterwards."""

    def run_case(self, config, backend, model, tmp_path, transcribe=None):
        config["server"]["data_dir"] = str(tmp_path)
        with (
            patch("digue.container.create_container") as mock_create,
            patch("digue.container._wait_for_server", return_value=True) as mock_wait,
            patch("digue.transcribe.transcribe", side_effect=transcribe or (lambda *_a, **_k: "hello")),
            patch("digue.container.container_exists", return_value=True),
            patch("digue.container.remove_container") as mock_remove,
        ):
            result = benchmark_mod.benchmark_case(config, backend, model, tmp_path / "audio.wav", runs=2)
        return result, mock_create, mock_wait, mock_remove

    def test_case_removes_container_when_transcription_is_interrupted(self, tmp_path):
        config = _default_config()
        config["server"]["data_dir"] = str(tmp_path)
        with (
            patch("digue.container.create_container"),
            patch("digue.container._wait_for_server", return_value=True),
            patch("digue.transcribe.transcribe", side_effect=KeyboardInterrupt),
            patch("digue.container.container_exists", return_value=True),
            patch("digue.container.remove_container") as mock_remove,
            pytest.raises(KeyboardInterrupt),
        ):
            benchmark_mod.benchmark_case(config, "cpu", "small", tmp_path / "audio.wav")

        mock_remove.assert_called_once_with(container_mod.CONTAINER_NAME)

    def test_case_clears_custom_image_for_other_backends(self, tmp_path):
        config = _default_config()
        config["server"]["backend"] = "cpu"
        config["server"]["image"] = "x"

        result, mock_create, _wait, _remove = self.run_case(config, "intel", "small", tmp_path)

        assert result is not None
        bench_config, backend = mock_create.call_args.args
        assert backend == "intel"
        assert bench_config["server"]["image"] == ""
        assert bench_config["models"]["intel"] == "small"
        assert container_mod.resolve_image(backend, bench_config) == container_mod.DOCKER_IMAGES["intel"]

    def test_case_keeps_custom_image_for_resolved_backend(self, tmp_path):
        config = _default_config()
        config["server"]["backend"] = "cpu"
        config["server"]["image"] = "x"

        result, mock_create, _wait, _remove = self.run_case(config, "cpu", "small", tmp_path)

        assert result is not None
        bench_config, backend = mock_create.call_args.args
        assert backend == "cpu"
        assert bench_config["server"]["image"] == "x"

    def test_case_waits_on_the_case_config_it_created(self, tmp_path):
        """The wait probed the caller's config while the container was
        created from the case config; they share the port today, but the
        probe must follow the container it is waiting for."""
        config = _default_config()
        config["server"]["backend"] = "cpu"

        _result, mock_create, mock_wait, mock_remove = self.run_case(config, "cpu", "large-v3-turbo", tmp_path)

        assert mock_wait.call_args.args[0] is mock_create.call_args.args[0]
        mock_remove.assert_called_once_with(container_mod.CONTAINER_NAME)

    def test_case_result_carries_timings_and_text(self, tmp_path):
        config = _default_config()
        config["server"]["backend"] = "cpu"

        result, *_rest = self.run_case(config, "cpu", "small", tmp_path)

        assert result is not None
        assert result["backend"] == "cpu" and result["model"] == "small"
        assert len(result["runs_ms"]) == 2 and result["text"] == "hello"
        assert result["avg_ms"] == sum(result["runs_ms"]) // 2

    def test_transcription_error_skips_the_case(self, tmp_path, capsys):
        config = _default_config()
        config["server"]["backend"] = "cpu"

        def fail(*_args, **_kwargs):
            raise RuntimeError("HTTP 500")

        result, *_rest = self.run_case(config, "cpu", "small", tmp_path, transcribe=fail)

        assert result is None
        assert "Skipped: transcription failed: HTTP 500" in capsys.readouterr().err


class TestRunBenchmarkCases:
    def run(self, config, **kwargs):
        seen = []

        def fake_case(_config, backend, model, _audio, runs):
            seen.append((backend, model, runs))
            return {"backend": backend, "model": model, "avg_ms": 1, "runs_ms": [1], "text": "t"}

        with (
            patch("digue.container.preserve_container_for_benchmark"),
            patch("digue.container.detect_backend", return_value="intel"),
            patch("digue.benchmark.benchmark_case", side_effect=fake_case),
        ):
            results = benchmark_mod.run_benchmark("audio.wav", config, **kwargs)
        return seen, results

    def test_defaults_use_the_resolved_backend_plus_cpu_and_the_quick_models(self):
        config = _default_config()
        config["server"]["backend"] = "cpu"

        seen, results = self.run(config)

        assert seen == [("cpu", "small", benchmark_mod.BENCHMARK_RUNS), ("cpu", "large-v3-turbo", 3)]
        assert [result["model"] for result in results] == ["small", "large-v3-turbo"]

    def test_explicit_backends_models_and_runs(self):
        seen, _results = self.run(_default_config(), backends=["amd", "cpu"], models=["medium"], runs=1)

        assert seen == [("amd", "medium", 1), ("cpu", "medium", 1)]

    def test_interrupt_prints_the_partial_summary_restores_the_container_and_propagates(self, capsys):
        manager = MagicMock()
        manager.__enter__.return_value = None
        manager.__exit__.return_value = False
        answers = iter([{"backend": "cpu", "model": "small", "avg_ms": 5, "runs_ms": [5], "text": "t"}])

        def case_then_interrupt(*_args, **_kwargs):
            try:
                return next(answers)
            except StopIteration:
                raise KeyboardInterrupt from None

        with (
            patch("digue.container.preserve_container_for_benchmark", return_value=manager),
            patch("digue.container.detect_backend", return_value="cpu"),
            patch("digue.benchmark.benchmark_case", side_effect=case_then_interrupt),
            pytest.raises(KeyboardInterrupt),
        ):
            benchmark_mod.run_benchmark("audio.wav", _default_config(), backends=["cpu"], models=["small", "medium"])

        manager.__exit__.assert_called_once()
        err = capsys.readouterr().err
        assert "Summary" in err and err.index("Summary") < err.rindex("cpu / small")

    def test_missing_models_are_announced_with_sizes(self, tmp_path, capsys):
        config = _default_config()
        config["server"]["data_dir"] = str(tmp_path)
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "ggml-small.bin").write_bytes(b"x")

        self.run(config, backends=["cpu"], models=["small", "medium"])

        err = capsys.readouterr().err
        assert "Missing models (will download, ~1463 MB total): medium (~1463 MB)" in err
        assert "small (~" not in err
