"""Tests for digue.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import digue


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
