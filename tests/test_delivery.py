"""Tests for send_text, delivery lock, paste normalization, and post-delivery archive."""

import os
import subprocess
import threading
from unittest.mock import MagicMock, patch

import pytest

from digue import audio as audio_mod
from digue import delivery as delivery_mod
from digue import dictate as dictate_mod
from digue.config import _default_config


class TestSendText:
    def test_raises_when_no_display(self, monkeypatch):
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        with pytest.raises(RuntimeError, match="No DISPLAY"):
            delivery_mod.send_text("hello", display_server="auto")

    @patch("subprocess.run")
    def test_x11_uses_xclip(self, mock_run):
        delivery_mod.send_text("hello", display_server="x11")
        cmds = [recorded_call[0][0] for recorded_call in mock_run.call_args_list]
        assert cmds[0][0] == "xclip"
        assert cmds[1][0] == "xdotool"

    @patch("subprocess.run")
    def test_wayland_uses_wl_copy(self, mock_run):
        delivery_mod.send_text("hello", display_server="wayland")
        cmds = [recorded_call[0][0] for recorded_call in mock_run.call_args_list]
        assert cmds[0][0] == "wl-copy"
        assert cmds[1][0] == "wtype"

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_missing_tool_gives_install_hint(self, mock_run):
        with pytest.raises(RuntimeError, match="sudo apt install"):
            delivery_mod.send_text("hello", display_server="x11")

    @patch("subprocess.run")
    def test_type_x11_uses_xdotool_type_reading_stdin(self, mock_run):
        delivery_mod.send_text("olá mundo", display_server="x11", input_mode="type")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["xdotool", "type", "--clearmodifiers", "--file", "-"]
        assert mock_run.call_args[1]["input"] == "olá mundo".encode()

    @patch("subprocess.run")
    def test_type_wayland_uses_wtype_reading_stdin(self, mock_run):
        # wtype has no --no-newline flag: any unknown -option makes it fail with
        # "Unknown parameter" (checked main.c of atx/wtype, the Debian package).
        delivery_mod.send_text("olá mundo", display_server="wayland", input_mode="type")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["wtype", "-"]
        assert "--no-newline" not in cmd
        assert mock_run.call_args[1]["input"] == "olá mundo".encode()

    @pytest.mark.parametrize("display_server", ["x11", "wayland"])
    @patch("subprocess.run")
    def test_type_text_starting_with_dash_never_lands_in_argv(self, mock_run, display_server):
        delivery_mod.send_text("- item one", display_server=display_server, input_mode="type")
        cmd = mock_run.call_args[0][0]
        assert "- item one" not in cmd
        assert mock_run.call_args[1]["input"] == b"- item one"

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired("xdotool", 120))
    def test_type_command_timeout_raises(self, mock_run):
        with pytest.raises(RuntimeError, match="xdotool timed out"):
            delivery_mod.send_text("hello", display_server="x11", input_mode="type")

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_type_missing_tool_gives_install_hint(self, mock_run):
        with pytest.raises(RuntimeError, match="sudo apt install"):
            delivery_mod.send_text("hello", display_server="x11", input_mode="type")

    @patch("subprocess.run")
    def test_paste_command_failure_raises(self, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=0),
            subprocess.CalledProcessError(1, "xdotool", stderr=b"no window"),
        ]

        with pytest.raises(RuntimeError, match="xdotool failed: no window"):
            delivery_mod.send_text("hello", display_server="x11")

    @patch("subprocess.run")
    def test_paste_command_timeout_raises(self, mock_run):
        mock_run.side_effect = [MagicMock(returncode=0), subprocess.TimeoutExpired("xdotool", 5)]

        with pytest.raises(RuntimeError, match="xdotool timed out"):
            delivery_mod.send_text("hello", display_server="x11")

    def test_input_mode_default_is_paste(self):
        config = _default_config()
        assert config["dictate"]["input_mode"] == "paste"


class TestSendTextDeliveryLock:
    def test_concurrent_deliveries_are_serialized(self, tmp_path):
        """The clipboard is global: two overlapping deliveries must not
        interleave copy+paste, or one text is pasted twice and the other is
        lost (the .txt archives both, the clipboard keeps only the last)."""
        release = {"first": threading.Event(), "second": threading.Event()}
        reached = {"first": threading.Event(), "second": threading.Event()}
        current = threading.local()

        def slow_run(cmd, *args, **kwargs):
            if cmd[0] == "xclip":
                reached[current.which].set()
                release[current.which].wait(timeout=5)
            return MagicMock()

        def deliver(which):
            current.which = which
            delivery_mod.send_text("text")

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.delivery.detect_display_server", return_value="x11"),
            patch("subprocess.run", side_effect=slow_run),
        ):
            first = threading.Thread(target=deliver, args=("first",))
            first.start()
            assert reached["first"].wait(timeout=5)

            second = threading.Thread(target=deliver, args=("second",))
            second.start()
            # D2 must block before its copy while D1 holds the delivery lock.
            assert not reached["second"].wait(timeout=0.3)
            release["first"].set()
            first.join(timeout=5)
            # D1 released the lock: D2 now reaches its own copy.
            assert reached["second"].wait(timeout=5)
            release["second"].set()
            second.join(timeout=5)
        assert not second.is_alive()

    def test_display_server_detection_happens_outside_the_lock(self, tmp_path):
        """Detection reads env vars only; serializing it would needlessly hold
        the lock while another delivery is pasting."""
        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.delivery.detect_display_server", return_value="x11") as mock_detect,
            patch("subprocess.run"),
        ):
            delivery_mod.send_text("text")

        assert mock_detect.call_count == 1


class TestDetectDisplayServer:
    def test_wayland(self, monkeypatch):
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.delenv("DISPLAY", raising=False)
        assert delivery_mod.detect_display_server() == "wayland"

    def test_x11(self, monkeypatch):
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")
        assert delivery_mod.detect_display_server() == "x11"

    def test_wayland_takes_priority(self, monkeypatch):
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setenv("DISPLAY", ":0")
        assert delivery_mod.detect_display_server() == "wayland"

    def test_none_when_no_display(self, monkeypatch):
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        assert delivery_mod.detect_display_server() is None


class TestNormalizePastedText:
    def test_collapses_newlines_and_spaces(self):
        assert delivery_mod.normalize_pasted_text("olá\n\nmundo \t aqui") == "olá mundo aqui"

    def test_joins_whisper_wrapped_lines(self):
        sample = (
            "Eu queria fazer um teste aqui e aí por algum motivo o trans\n"
            "crevendo não sumiu.\n"
            "Então tem que ver o que está acontecendo aqui para ele não\n"
            "estar desaparecendo."
        )
        assert delivery_mod.normalize_pasted_text(sample) == (
            "Eu queria fazer um teste aqui e aí por algum motivo o trans crevendo não sumiu. "
            "Então tem que ver o que está acontecendo aqui para ele não estar desaparecendo."
        )

    def test_strips_edges(self):
        assert delivery_mod.normalize_pasted_text("  text  ") == "text"

    def test_empty(self):
        assert delivery_mod.normalize_pasted_text("") == ""


class TestDeliveryResult:
    """finish_dictation reports a typed terminal outcome instead of a bare int:
    "a function returned" alone does not say whether the take was delivered,
    rescued, empty, or failed in a way a later toggle may retry."""

    def make_config(self, tmp_path):
        config = _default_config()
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
            patch("digue.delivery.send_text"),
            patch("digue.transcribe.transcribe", return_value="hello"),
        ):
            result = dictate_mod.finish_dictation(config, rec_file)

        assert result.outcome == "delivered"
        assert result.exit_code == 0
        assert result.rescued_path is None

    def test_empty_with_exit_zero(self, tmp_path):
        rec_file = self.make_rec_file(tmp_path)
        config = self.make_config(tmp_path)

        with patch("digue.delivery.send_text"), patch("digue.transcribe.transcribe", return_value=""):
            result = dictate_mod.finish_dictation(config, rec_file)

        assert result.outcome == "empty"
        assert result.exit_code == 0

    def test_missing_recording_is_empty_with_exit_one(self, tmp_path):
        config = self.make_config(tmp_path)

        with patch("digue.notify.send_notification") as mock_notify:
            result = dictate_mod.finish_dictation(config, None)

        assert result.outcome == "empty"
        assert result.exit_code == 1
        mock_notify.assert_called_once()

    def test_transcription_failure_rescues_the_recording(self, tmp_path):
        rec_file = self.make_rec_file(tmp_path)
        config = self.make_config(tmp_path)

        with (
            patch("digue.delivery.send_text"),
            patch("digue.transcribe.transcribe", side_effect=RuntimeError("server down")),
        ):
            result = dictate_mod.finish_dictation(config, rec_file)

        assert result.outcome == "rescued"
        assert result.exit_code == 1
        assert result.rescued_path is not None
        assert result.rescued_path.read_bytes() == b"audio"
        assert not rec_file.exists()

    def test_transcription_failure_without_rescue_is_retryable(self, tmp_path):
        rec_file = self.make_rec_file(tmp_path)
        config = self.make_config(tmp_path)

        with (
            patch("digue.delivery.send_text"),
            patch("digue.transcribe.transcribe", side_effect=RuntimeError("server down")),
            patch("digue.audio.rescue_recording", return_value=None),
        ):
            result = dictate_mod.finish_dictation(config, rec_file)

        assert result.outcome == "retryable_failure"
        assert result.exit_code == 1
        assert result.rescued_path is None
        assert rec_file.exists()

    def test_paste_failure_keeps_txt_and_audio(self, tmp_path):
        rec_file = self.make_rec_file(tmp_path)
        config = self.make_config(tmp_path)

        with (
            patch("digue.delivery.send_text", side_effect=RuntimeError("no display")),
            patch("digue.transcribe.transcribe", return_value="hello"),
        ):
            result = dictate_mod.finish_dictation(config, rec_file)

        assert result.outcome == "rescued"
        assert result.exit_code == 1
        month = tmp_path / "audio" / audio_mod.month_dir_for(audio_mod.now_timestamp())
        assert (month / f"{audio_mod.now_timestamp()}.txt").exists() or list(month.glob("*.txt"))
        assert list(month.glob("*.wav"))

    def test_archive_failure_rescues_the_raw_recording(self, tmp_path):
        rec_file = self.make_rec_file(tmp_path)
        config = self.make_config(tmp_path)

        with (
            patch("digue.delivery.send_text"),
            patch("digue.transcribe.transcribe", return_value="hello"),
            patch("digue.audio.save_audio", side_effect=OSError("disk full")),
        ):
            result = dictate_mod.finish_dictation(config, rec_file)

        assert result.outcome == "rescued"
        assert result.exit_code == 1
        assert result.rescued_path is not None
        assert result.rescued_path.read_bytes() == b"audio"

    def test_archive_failure_after_paste_is_terminal(self, tmp_path):
        """Regression: once the text was pasted, no outcome may be retryable --
        a recovery retry would transcribe and paste the same text again. The
        archive failure is still reported (exit 1), but as delivered."""
        rec_file = self.make_rec_file(tmp_path)
        config = self.make_config(tmp_path)

        with (
            patch("digue.delivery.send_text"),
            patch("digue.transcribe.transcribe", return_value="hello"),
            patch("digue.audio.save_audio", side_effect=OSError("disk full")),
            patch("digue.audio.rescue_recording", return_value=None),
        ):
            result = dictate_mod.finish_dictation(config, rec_file)

        assert result.outcome == "delivered"
        assert result.outcome in dictate_mod.TERMINAL_OUTCOMES
        assert result.exit_code == 1
        assert rec_file.exists()

    def test_paste_failure_without_archive_or_rescue_is_retryable(self, tmp_path):
        """Nothing was delivered and the WAV is still in the runtime dir: the
        take state must stay so the next toggle retries."""
        rec_file = self.make_rec_file(tmp_path)
        config = self.make_config(tmp_path)

        with (
            patch("digue.delivery.send_text", side_effect=RuntimeError("no display")),
            patch("digue.transcribe.transcribe", return_value="hello"),
            patch("digue.audio.save_audio", side_effect=OSError("disk full")),
            patch("digue.audio.rescue_recording", return_value=None),
        ):
            result = dictate_mod.finish_dictation(config, rec_file)

        assert result.outcome == "retryable_failure"
        assert result.exit_code == 1
        assert rec_file.exists()

    def test_paste_failure_with_rescue_is_rescued(self, tmp_path):
        rec_file = self.make_rec_file(tmp_path)
        config = self.make_config(tmp_path)

        with (
            patch("digue.delivery.send_text", side_effect=RuntimeError("no display")),
            patch("digue.transcribe.transcribe", return_value="hello"),
            patch("digue.audio.save_audio", side_effect=OSError("disk full")),
        ):
            result = dictate_mod.finish_dictation(config, rec_file)

        assert result.outcome == "rescued"
        assert result.rescued_path is not None and result.rescued_path.exists()
        assert not rec_file.exists()

    def test_no_speech_without_archive_or_rescue_is_retryable(self, tmp_path):
        rec_file = self.make_rec_file(tmp_path)
        config = self.make_config(tmp_path)

        with (
            patch("digue.delivery.send_text"),
            patch("digue.transcribe.transcribe", return_value=""),
            patch("digue.audio.save_audio", side_effect=OSError("disk full")),
            patch("digue.audio.rescue_recording", return_value=None),
        ):
            result = dictate_mod.finish_dictation(config, rec_file)

        assert result.outcome == "retryable_failure"
        assert result.exit_code == 1
        assert rec_file.exists()

    @pytest.mark.parametrize("exit_code", [0, 1])
    def test_toggle_maps_the_result_to_the_exit_code(self, exit_code, tmp_path):
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        config["dictate"]["max_duration"] = 0
        recorder = MagicMock(pid=os.getpid(), poll=lambda: 0)
        result = dictate_mod.DeliveryResult(outcome="rescued", exit_code=exit_code)

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.container.ensure_server"),
            patch("digue.container.is_server_running", return_value=True),
            patch("subprocess.Popen", return_value=recorder),
            patch("digue.recording._wait_recorder_end_daemon", return_value="ended"),
            patch("digue.dictate.finish_dictation", return_value=result),
            patch("digue.notify.send_notification"),
            patch("digue.notify.notify_close"),
            patch("signal.signal"),
        ):
            assert dictate_mod.dictate_toggle(config) == exit_code


class TestFinishDictationBackend:
    @patch("digue.delivery.send_text")
    @patch("digue.transcribe.transcribe", return_value="hello")
    @patch("digue.audio.save_audio", return_value=("saved.flac", "2026-01-01T00:00:00"))
    def test_finish_dictation_passes_resolved_backend(self, mock_save, mock_transcribe, mock_send, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        config = _default_config()
        config["server"]["backend"] = "remote"
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")

        with patch("digue.container.detect_backend") as mock_detect:
            dictate_mod.finish_dictation(config, rec_file)

        mock_save.assert_called_once()
        assert mock_save.call_args[1]["backend"] == "remote"
        mock_detect.assert_not_called()  # only remote-or-not matters here: no nvidia-smi/lspci per delivery

    @patch("digue.delivery.send_text")
    @patch("digue.transcribe.transcribe", return_value="hello")
    @patch("digue.audio.save_audio", return_value=("saved.flac", "2026-01-01T00:00:00"))
    def test_finish_dictation_does_not_detect_hardware_for_a_local_backend(
        self, mock_save, mock_transcribe, mock_send, tmp_path
    ):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")

        with patch("digue.container.detect_backend") as mock_detect:
            dictate_mod.finish_dictation(config, rec_file)

        mock_detect.assert_not_called()
        assert mock_save.call_args[1]["backend"] != "remote"


class TestDictateArchivesAfterDelivery:
    @patch("digue.notify.send_notification")
    @patch("digue.delivery.send_text")
    @patch("digue.transcribe.transcribe", return_value="hello")
    def test_limit_warning_remains_in_transcription_progress(self, mock_transcribe, mock_send, mock_notify, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")

        assert dictate_mod.finish_dictation(config, rec_file, limit_reached=True).exit_code == 0

        first_message = mock_notify.call_args_list[0].args[0]
        assert "Limit reached" in first_message
        assert "transcribing" in first_message

    @patch("digue.delivery.send_text")
    @patch("digue.transcribe.transcribe", return_value="hello")
    @patch("digue.audio.save_audio", return_value=("saved.flac", "2026-01-01T00:00:00"))
    def test_transcribes_and_pastes_before_archiving(self, mock_save, mock_transcribe, mock_send, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")

        order = MagicMock()
        order.attach_mock(mock_transcribe, "transcribe")
        order.attach_mock(mock_send, "send_text")
        order.attach_mock(mock_save, "save_audio")

        result = dictate_mod.finish_dictation(config, rec_file)

        assert result.exit_code == 0
        assert [call_record[0] for call_record in order.mock_calls] == ["transcribe", "send_text", "save_audio"]

    @patch("digue.delivery.send_text")
    @patch("digue.transcribe.transcribe", side_effect=RuntimeError("server down"))
    def test_transcribe_failure_archives_recording(self, mock_transcribe, mock_send, tmp_path, capsys):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        audio_dir = tmp_path / "audio"
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = dictate_mod.finish_dictation(config, rec_file)

        assert result.exit_code == 1
        assert not rec_file.exists()
        saved = list(audio_dir.rglob("*.wav"))
        assert len(saved) == 1
        assert saved[0].read_bytes() == b"audio"
        assert "Transcription failed" in capsys.readouterr().err

    @patch("digue.delivery.send_text")
    @patch("digue.transcribe.transcribe", side_effect=RuntimeError("server down"))
    def test_transcribe_failure_keeps_recording_even_with_save_audio_off(
        self, mock_transcribe, mock_send, tmp_path, capsys
    ):
        """save-audio only skips the backup of a delivered take; a failed take
        is not delivered anywhere, so dropping it would lose the audio."""
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        audio_dir = tmp_path / "audio"
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)
        config["dictate"]["save_audio"] = False

        result = dictate_mod.finish_dictation(config, rec_file)

        assert result.exit_code == 1
        saved = list(audio_dir.rglob("*.wav"))
        assert len(saved) == 1
        assert saved[0].read_bytes() == b"audio"
        assert not rec_file.exists()
        assert "Recording kept at" in capsys.readouterr().err

    @patch("digue.audio.save_audio", side_effect=OSError("disk full"))
    @patch("digue.delivery.send_text")
    @patch("digue.transcribe.transcribe", return_value="hello")
    def test_save_audio_failure_after_paste_keeps_uncompressed_recording(
        self, mock_transcribe, mock_send, mock_save, tmp_path, capsys
    ):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        audio_dir = tmp_path / "audio"
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = dictate_mod.finish_dictation(config, rec_file)

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
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        with (
            patch("digue.audio.now_timestamp", side_effect=["20260904-120000", "20260904-120005"]),
            patch("digue.delivery.send_text"),
            patch("digue.transcribe.transcribe", return_value="hello"),
        ):
            result = dictate_mod.finish_dictation(config, rec_file)

        assert result.exit_code == 0
        month = tmp_path / "audio" / "2026" / "09"
        assert {path.name for path in month.iterdir()} == {"20260904-120000.txt", "20260904-120000.wav"}
