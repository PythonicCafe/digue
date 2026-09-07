"""Tests for the library helpers (transcribe_file, record_to)."""

from unittest.mock import MagicMock, patch

import pytest

from digue import recording as recording_mod
from digue import transcribe as transcribe_mod
from digue.config import _default_config


class TestTranscribeFile:
    @patch("digue.transcribe.transcribe", return_value="hello")
    @patch("digue.container.is_server_running", return_value=True)
    @patch("digue.container.ensure_server")
    def test_uses_config_and_returns_text(self, mock_ensure, mock_running, mock_transcribe, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"audio")
        config = _default_config()
        config["transcribe"]["prompt"] = "KINAI"
        config["transcribe"]["language"] = "pt"

        assert transcribe_mod.transcribe_file(audio, config) == "hello"

        mock_ensure.assert_called_once()
        assert mock_transcribe.call_args.args[2] == "pt"
        assert mock_transcribe.call_args.kwargs["prompt"] == "KINAI"

    @patch("digue.container.ensure_server")
    @patch("digue.container.is_server_running", return_value=False)
    def test_raises_when_server_down(self, mock_running, mock_ensure, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"audio")
        with pytest.raises(RuntimeError, match="server is not running"):
            transcribe_mod.transcribe_file(audio, _default_config())


class TestRecordTo:
    def test_uses_recording_command_and_returns_path(self, tmp_path):
        output = tmp_path / "take.wav"
        config = _default_config()
        config["dictate"]["recorder"] = "arecord"
        config["dictate"]["device"] = "hw:2,0"
        proc = MagicMock()
        proc.poll.return_value = 0

        def fake_popen(argv, stdout=None, stderr=None, start_new_session=False):
            output.write_bytes(b"audio")
            return proc

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch("time.sleep"),
        ):
            result = recording_mod.record_to(output, seconds=1.5, config=config)

        assert result == output
        assert output.read_bytes() == b"audio"
        proc.terminate.assert_called()

    def test_propagates_recorder_startup_error(self, tmp_path):
        output = tmp_path / "take.wav"
        config = _default_config()
        config["dictate"]["recorder"] = "pw-record"
        dead = MagicMock()
        dead.poll.return_value = 1

        def fake_popen(argv, stdout=None, stderr=None, start_new_session=False):
            if stderr is not None:
                stderr.write("can't find node\n")
                stderr.flush()
            return dead

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            pytest.raises(RuntimeError, match="pw-record failed: can't find node"),
        ):
            recording_mod.record_to(output, seconds=1, config=config)
