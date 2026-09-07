"""Tests for batch-transcribe input handling and conversion."""

from unittest.mock import MagicMock, patch

import digue


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
