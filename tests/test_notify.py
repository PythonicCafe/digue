"""Tests for desktop notifications and their lifecycle."""

import subprocess
from unittest.mock import MagicMock, patch

from digue import dictate as dictate_mod
from digue import notify as notify_mod
from digue import transcribe as transcribe_mod
from digue.config import _default_config

# -- Notifications -----------------------------------------------------------


class TestNotify:
    @patch("subprocess.run")
    def test_sends_with_replace_id(self, mock_run):
        notify_mod.send_notification("test", timeout_ms=5000)
        cmd = mock_run.call_args[0][0]
        assert "--replace-id" in cmd
        replace_id = int(cmd[cmd.index("--replace-id") + 1])
        assert notify_mod.NOTIFY_REPLACE_ID <= replace_id < notify_mod.NOTIFY_REPLACE_ID + notify_mod.NOTIFY_ID_SLOTS

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_prints_warning_when_notify_send_missing(self, mock_run, capsys):
        notify_mod._notify_send_warned = False
        notify_mod.send_notification("test")
        err = capsys.readouterr().err
        assert "notify-send not found" in err

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_warns_only_once(self, mock_run, capsys):
        notify_mod._notify_send_warned = False
        notify_mod.send_notification("first")
        notify_mod.send_notification("second")
        err = capsys.readouterr().err
        assert err.count("notify-send not found") == 1

    @patch("subprocess.run")
    def test_always_prints_to_stderr(self, mock_run, capsys):
        notify_mod.send_notification("hello world")
        err = capsys.readouterr().err
        assert "hello world" in err

    @patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "notify-send"))
    def test_failing_notify_send_warns_once(self, mock_run, capsys):
        """A failed notification must be visible, not silently swallowed."""
        notify_mod._notify_send_warned = False
        notify_mod.send_notification("first")
        notify_mod.send_notification("second")
        err = capsys.readouterr().err
        assert err.count("notify-send failed") == 1

    def test_response_is_closed_after_request(self, tmp_path):
        response = MagicMock()
        response.read.return_value = b"ok"
        with patch("urllib.request.urlopen", return_value=response):
            transcribe_mod._multipart_request("http://x/inference", b"audio", {"model": "m"}, timeout=10)
        # The urllib stream must be released even if later processing fails.
        response.close.assert_called_once()


class TestNotifyClose:
    @patch("subprocess.run")
    def test_calls_gdbus(self, mock_run):
        notify_mod.notify_close()
        cmd = mock_run.call_args[0][0]
        assert "gdbus" in cmd
        assert any("CloseNotification" in arg for arg in cmd)


class TestNotifyLifecycle:
    @patch("digue.notify.send_notification")
    @patch("digue.delivery.send_text")
    @patch("digue.transcribe.transcribe", return_value="hello")
    @patch("digue.audio.save_audio", return_value=("saved.flac", "2026-01-01T00:00:00"))
    @patch("digue.notify.notify_close")
    def test_successful_dictation_notifies_pasted(
        self, mock_close, mock_save, mock_transcribe, mock_send, mock_notify, tmp_path
    ):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        audio_dir = tmp_path / "audio"
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = dictate_mod.finish_dictation(config, rec_file)

        assert result.exit_code == 0
        mock_close.assert_not_called()
        last_notify = mock_notify.call_args_list[-1]
        assert last_notify.args == ("Pasted (5 chars)",)
        assert last_notify.kwargs == {"timeout_ms": 3000}

    @patch("digue.notify.send_notification")
    @patch("digue.delivery.send_text")
    @patch("digue.transcribe.transcribe", return_value="hello")
    @patch("digue.audio.save_audio", return_value=("saved.flac", "2026-01-01T00:00:00"))
    @patch("digue.notify.notify_close")
    def test_successful_dictation_notifies_typed(
        self, mock_close, mock_save, mock_transcribe, mock_send, mock_notify, tmp_path
    ):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        config["dictate"]["input_mode"] = "type"

        result = dictate_mod.finish_dictation(config, rec_file)

        assert result.exit_code == 0
        mock_close.assert_not_called()
        last_notify = mock_notify.call_args_list[-1]
        assert last_notify.args == ("Typed (5 chars)",)
        assert last_notify.kwargs == {"timeout_ms": 3000}

    @patch("digue.delivery.send_text", side_effect=RuntimeError("no display"))
    @patch("digue.transcribe.transcribe", return_value="hello")
    @patch("digue.audio.save_audio", return_value=("saved.flac", "2026-01-01T00:00:00"))
    def test_paste_failure_notifies_with_timeout(self, mock_save, mock_transcribe, mock_send, tmp_path, capsys):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")

        result = dictate_mod.finish_dictation(config, rec_file)

        assert result.exit_code == 1
        err = capsys.readouterr().err
        assert "Paste failed" in err
        assert "Transcription saved to" in err

    @patch("digue.delivery.send_text")
    @patch("digue.transcribe.transcribe", return_value="hello")
    @patch("digue.audio.save_audio", side_effect=OSError("disk full"))
    def test_save_failure_notifies_and_does_not_crash(self, mock_save, mock_transcribe, mock_send, tmp_path, capsys):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")

        result = dictate_mod.finish_dictation(config, rec_file)

        assert result.exit_code == 1
        err = capsys.readouterr().err
        assert "Failed to save audio" in err
        assert "disk full" in err

    @patch("digue.delivery.send_text")
    @patch("digue.transcribe.transcribe", return_value="hello")
    @patch("digue.audio.save_audio", return_value=("saved.flac", "2026-01-01T00:00:00"))
    def test_transcript_write_failure_notifies_and_prints_text(
        self, mock_save, mock_transcribe, mock_send, tmp_path, capsys
    ):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        config = _default_config()
        config["dictate"]["save_audio"] = False
        # point audio_dir at a file so the transcript write fails
        blocker = tmp_path / "blocker"
        blocker.write_text("not a dir")
        config["dictate"]["audio_dir"] = str(blocker)

        result = dictate_mod.finish_dictation(config, rec_file)

        assert result.exit_code == 1
        err = capsys.readouterr().err
        assert "Failed to save transcript" in err
        assert "hello" in err

    def test_transcript_write_failure_without_rescue_is_terminal(self, tmp_path, capsys):
        """The text was already pasted: a retryable outcome would make the
        recovery paste it again, so a failed .txt with a failed rescue is
        still terminal (delivered, exit 1)."""
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        config = _default_config()
        blocker = tmp_path / "blocker"
        blocker.write_text("not a dir")
        config["dictate"]["audio_dir"] = str(blocker)  # .txt write and rescue both fail

        with (
            patch("digue.delivery.send_text") as mock_send,
            patch("digue.transcribe.transcribe", return_value="hello"),
            patch("digue.notify.send_notification"),
        ):
            result = dictate_mod.finish_dictation(config, rec_file)

        mock_send.assert_called_once()
        assert result.outcome == "delivered"
        assert result.exit_code == 1
        assert rec_file.exists()
        assert "hello" in capsys.readouterr().err
