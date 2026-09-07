"""Tests for saving, compressing, and rescuing dictation audio."""

import math
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from digue import audio as audio_mod
from digue import dictate as dictate_mod
from digue.config import _default_config, load_config
from digue.container import CONTAINER_NAME


class TestTimestampFormat:
    def test_format_has_no_colon_or_dash_in_date(self):
        """Filenames must be shell-friendly: YYYYMMDD-HHMMSS (no ':' to escape)."""
        timestamp = audio_mod.now_timestamp()
        assert len(timestamp) == 15
        assert timestamp[8] == "-"
        assert ":" not in timestamp
        assert "-" not in timestamp[:8]

    def test_month_directory_uses_timestamp_not_wall_clock(self):
        """The YYYY/MM path comes from the timestamp, so audio and .txt land together."""
        assert audio_mod.month_dir_for("20260904-123456") == Path("2026") / "09"


class TestSaveAudio:
    def test_copies_with_timestamp_in_month_directory(self, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"wav data")
        audio_dir = tmp_path / "audio"
        saved, timestamp = audio_mod.save_audio(rec_file, audio_dir)
        assert saved.exists()
        # <audio_dir>/YYYY/MM/<timestamp>.wav
        assert saved.parent == audio_dir / timestamp[:4] / timestamp[4:6]
        assert timestamp in saved.name


class TestRescueRecording:
    def test_rescue_never_overwrites_an_existing_destination(self, tmp_path):
        """The temp file is exclusive, but publishing with os.replace would
        still clobber a destination that already exists; the rescue contract
        is never to overwrite another take's file."""
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"new audio")
        timestamp = "20260905-101500"
        existing = tmp_path / "audio" / audio_mod.month_dir_for(timestamp) / f"{timestamp}-0123456789abcdef.wav"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"old audio")

        rescued = audio_mod.rescue_recording(rec_file, tmp_path / "audio", timestamp, "0123456789abcdef")

        assert rescued is None
        assert existing.read_bytes() == b"old audio"
        assert rec_file.read_bytes() == b"new audio"
        assert sorted(path.name for path in existing.parent.iterdir()) == [existing.name]

    def test_moves_wav_with_take_id_and_only_then_removes_origin(self, tmp_path):
        rec_file = tmp_path / "digue-rec.wav"
        rec_file.write_bytes(b"audio")
        audio_dir = tmp_path / "audio"

        rescued = audio_mod.rescue_recording(rec_file, audio_dir, "20260904-120000", "0123456789abcdef")

        archived = audio_dir / "2026" / "09" / "20260904-120000-0123456789abcdef.wav"
        assert rescued == archived
        assert archived.read_bytes() == b"audio"
        assert not rec_file.exists()

    def test_keeps_the_suffix_of_a_native_flac_take(self, tmp_path):
        """A live take recorded natively as FLAC must not be rescued under a
        .wav name: the suffix is what save_audio and the server use to pick
        the decoder, and a .wav holding FLAC data would be re-compressed."""
        rec_file = tmp_path / "digue-rec.flac"
        rec_file.write_bytes(b"fLaC")
        audio_dir = tmp_path / "audio"

        rescued = audio_mod.rescue_recording(rec_file, audio_dir, "20260904-120000", "0123456789abcdef")

        assert rescued == audio_dir / "2026" / "09" / "20260904-120000-0123456789abcdef.flac"
        assert rescued.read_bytes() == b"fLaC"
        assert not rec_file.exists()

    def test_origin_unlink_failure_after_publish_still_reports_the_rescue(self, tmp_path, capsys):
        """Once the destination is linked, the rescue succeeded whatever
        happens to the origin: reporting None there made the caller keep the
        take state, and every retry collided with the published file."""
        rec_file = tmp_path / "digue-rec.wav"
        rec_file.write_bytes(b"audio")
        audio_dir = tmp_path / "audio"
        real_unlink = Path.unlink

        def unlink_all_but_origin(self, missing_ok=False):
            if self == rec_file:
                raise PermissionError("origin is read-only")
            real_unlink(self, missing_ok=missing_ok)

        with patch.object(Path, "unlink", unlink_all_but_origin):
            rescued = audio_mod.rescue_recording(rec_file, audio_dir, "20260904-120000", "0123456789abcdef")

        assert rescued == audio_dir / "2026" / "09" / "20260904-120000-0123456789abcdef.wav"
        assert rescued.read_bytes() == b"audio"
        assert rec_file.exists()
        assert sorted(path.name for path in rescued.parent.iterdir()) == [rescued.name]
        assert "origin is read-only" in capsys.readouterr().err

    def test_copy_failure_preserves_origin_and_returns_none(self, tmp_path, capsys):
        rec_file = tmp_path / "digue-rec.wav"  # never created: the copy must fail
        audio_dir = tmp_path / "audio"

        result = audio_mod.rescue_recording(rec_file, audio_dir, "20260904-120000", "0123456789abcdef")

        assert result is None
        assert "Failed to keep recording" in capsys.readouterr().err

    def test_publish_failure_preserves_origin_and_removes_temp(self, tmp_path, capsys):
        rec_file = tmp_path / "digue-rec.wav"
        rec_file.write_bytes(b"audio")
        audio_dir = tmp_path / "audio"

        with patch("os.link", side_effect=OSError("disk full")):
            result = audio_mod.rescue_recording(rec_file, audio_dir, "20260904-120000", "0123456789abcdef")

        month_dir = audio_dir / "2026" / "09"
        assert result is None
        assert rec_file.exists()
        assert not any(path.name.startswith(".") for path in month_dir.iterdir())
        assert "Failed to keep recording" in capsys.readouterr().err

    def test_filesystem_without_hard_links_falls_back_to_replace(self, tmp_path):
        """vfat/exFAT and some FUSE mounts refuse os.link (EPERM/EOPNOTSUPP);
        the rescue must still publish instead of stranding the take in the
        runtime dir."""
        import errno

        rec_file = tmp_path / "digue-rec.wav"
        rec_file.write_bytes(b"audio")
        audio_dir = tmp_path / "audio"

        with patch("os.link", side_effect=OSError(errno.EPERM, "Operation not permitted")):
            rescued = audio_mod.rescue_recording(rec_file, audio_dir, "20260904-120000", "0123456789abcdef")

        assert rescued == audio_dir / "2026" / "09" / "20260904-120000-0123456789abcdef.wav"
        assert rescued.read_bytes() == b"audio"
        assert not rec_file.exists()
        assert sorted(path.name for path in rescued.parent.iterdir()) == [rescued.name]

    def test_fallback_still_never_overwrites_an_existing_destination(self, tmp_path):
        import errno

        rec_file = tmp_path / "digue-rec.wav"
        rec_file.write_bytes(b"new audio")
        timestamp = "20260904-120000"
        existing = tmp_path / "audio" / "2026" / "09" / f"{timestamp}-0123456789abcdef.wav"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"old audio")

        with patch("os.link", side_effect=OSError(errno.EOPNOTSUPP, "Operation not supported")):
            rescued = audio_mod.rescue_recording(rec_file, tmp_path / "audio", timestamp, "0123456789abcdef")

        assert rescued is None
        assert existing.read_bytes() == b"old audio"
        assert rec_file.read_bytes() == b"new audio"
        assert sorted(path.name for path in existing.parent.iterdir()) == [existing.name]


class TestTakeIdInSavedNames:
    """Two takes ending in the same second used to overwrite each other silently
    (<YYYYMMDD-HHMMSS>.<ext>): the take id makes saved names unique, and every
    write is exclusive, so a collision can never drop a file."""

    def test_take_id_goes_into_audio_and_transcript_names(self, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"wav data")
        audio_dir = tmp_path / "audio"

        saved, _ = audio_mod.save_audio(rec_file, audio_dir, timestamp="20260904-120000", take_id="0123456789abcdef")
        text_path = audio_mod._write_transcript(audio_dir, "20260904-120000", "hello", take_id="0123456789abcdef")

        assert saved.name == "20260904-120000-0123456789abcdef.wav"
        assert text_path.name == "20260904-120000-0123456789abcdef.txt"

    def test_two_takes_in_the_same_second_generate_four_distinct_files(self, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"wav data")
        audio_dir = tmp_path / "audio"

        saved_a, _ = audio_mod.save_audio(rec_file, audio_dir, timestamp="20260904-120000", take_id="0" * 16)
        text_a = audio_mod._write_transcript(audio_dir, "20260904-120000", "a", take_id="0" * 16)
        saved_b, _ = audio_mod.save_audio(rec_file, audio_dir, timestamp="20260904-120000", take_id="f" * 16)
        text_b = audio_mod._write_transcript(audio_dir, "20260904-120000", "b", take_id="f" * 16)

        assert len({saved_a, saved_b, text_a, text_b}) == 4
        assert all(path.exists() for path in (saved_a, saved_b, text_a, text_b))

    def test_saving_never_overwrites_an_existing_file(self, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"new data")
        audio_dir = tmp_path / "audio"
        month_dir = audio_dir / "2026" / "09"
        month_dir.mkdir(parents=True)
        existing = month_dir / "20260904-120000.wav"
        existing.write_bytes(b"original")

        with pytest.raises(FileExistsError):
            audio_mod.save_audio(rec_file, audio_dir, timestamp="20260904-120000")

        assert existing.read_bytes() == b"original"

    def test_transcript_write_is_exclusive(self, tmp_path):
        audio_dir = tmp_path / "audio"
        month_dir = audio_dir / "2026" / "09"
        month_dir.mkdir(parents=True)
        existing = month_dir / "20260904-120000.txt"
        existing.write_text("original\n")

        with pytest.raises(FileExistsError):
            audio_mod._write_transcript(audio_dir, "20260904-120000", "hello")

        assert existing.read_text() == "original\n"

    def test_saving_with_take_id_never_overwrites_an_existing_file(self, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"new data")
        audio_dir = tmp_path / "audio"
        month_dir = audio_dir / "2026" / "09"
        month_dir.mkdir(parents=True)
        existing = month_dir / "20260904-120000-0123456789abcdef.wav"
        existing.write_bytes(b"original")

        with pytest.raises(FileExistsError):
            audio_mod.save_audio(rec_file, audio_dir, timestamp="20260904-120000", take_id="0123456789abcdef")

        assert existing.read_bytes() == b"original"

    def test_transcript_write_with_take_id_is_exclusive(self, tmp_path):
        audio_dir = tmp_path / "audio"
        month_dir = audio_dir / "2026" / "09"
        month_dir.mkdir(parents=True)
        existing = month_dir / "20260904-120000-0123456789abcdef.txt"
        existing.write_text("original\n")

        with pytest.raises(FileExistsError):
            audio_mod._write_transcript(audio_dir, "20260904-120000", "hello", take_id="0123456789abcdef")

        assert existing.read_text() == "original\n"


class TestDictationFiles:
    def test_accepts_old_and_new_layout_and_ignores_other_files(self, tmp_path):
        audio_dir = tmp_path / "audio"
        month_dir = audio_dir / "2026" / "09"
        month_dir.mkdir(parents=True)
        old_wav = month_dir / "20260904-120000.wav"
        new_flac = month_dir / "20260904-120001-0123456789abcdef.flac"
        new_txt = month_dir / "20260904-120001-0123456789abcdef.txt"
        music = month_dir / "01 - Song.wav"
        notes = month_dir / "random.txt"
        for path in (old_wav, new_flac, new_txt, music, notes):
            path.write_bytes(b"x")

        recordings = audio_mod._dictation_files(audio_dir, audio_mod.DICTATION_RECORDING_SUFFIXES)
        transcripts = audio_mod._dictation_files(audio_dir, frozenset((".txt",)))

        assert recordings == [old_wav, new_flac]
        assert transcripts == [new_txt]

    def test_symlinks_are_ignored(self, tmp_path):
        audio_dir = tmp_path / "audio"
        month_dir = audio_dir / "2026" / "09"
        month_dir.mkdir(parents=True)
        target = tmp_path / "elsewhere.wav"
        target.write_bytes(b"x")
        (month_dir / "20260904-120000.wav").symlink_to(target)

        assert audio_mod._dictation_files(audio_dir, audio_mod.DICTATION_RECORDING_SUFFIXES) == []


class TestCompressAudio:
    def test_wav_is_noop(self, tmp_path):
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"data")
        assert audio_mod._compress_audio(rec, "wav") == rec

    def test_flac_compresses_and_removes_wav(self, tmp_path):
        import shutil
        import wave

        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not available")
        rate = 16000
        samples = b"".join(
            int(8000 * math.sin(2 * math.pi * 440 * i / rate)).to_bytes(2, "little", signed=True) for i in range(rate)
        )
        rec = tmp_path / "rec.wav"
        with wave.open(str(rec), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(samples)
        wav_size = rec.stat().st_size

        result = audio_mod._compress_audio(rec, "flac")

        assert result.suffix == ".flac"
        assert result.exists()
        assert not rec.exists()
        assert result.stat().st_size < wav_size

    def test_unknown_format_raises(self, tmp_path):
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"data")
        with pytest.raises(KeyError):
            audio_mod._compress_audio(rec, "mp3")

    @patch("digue.container.container_status", return_value=None)
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_local_ffmpeg_writes_temp_in_final_directory_and_replaces(
        self, mock_run, mock_which, mock_status, tmp_path
    ):
        import subprocess

        rec = tmp_path / "20260904-120000-0123456789abcdef.wav"
        rec.write_bytes(b"wav-data")
        temp_paths = []

        def fake_run(cmd, **kwargs):
            assert "-y" not in cmd
            temp_path = Path(cmd[-1])
            temp_paths.append(temp_path)
            temp_path.write_bytes(b"flac-data")
            return subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")

        mock_run.side_effect = fake_run

        result = audio_mod._compress_audio(rec, "flac")

        assert result == tmp_path / "20260904-120000-0123456789abcdef.flac"
        assert result.read_bytes() == b"flac-data"
        assert not rec.exists()
        assert temp_paths[0].parent == tmp_path
        assert temp_paths[0].name.startswith(".") and temp_paths[0].name.endswith(".tmp")
        assert [path.name for path in tmp_path.iterdir() if path.is_file()] == [result.name]

    @patch("digue.container.container_status", return_value=None)
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_local_ffmpeg_failure_removes_temp_and_reservation_keeps_wav(
        self, mock_run, mock_which, mock_status, tmp_path
    ):
        import subprocess

        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"error")
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"wav-data")

        result = audio_mod._compress_audio(rec, "flac")

        assert result == rec
        assert rec.exists()
        assert [path.name for path in tmp_path.iterdir() if path.is_file()] == [rec.name]

    @pytest.mark.parametrize("failure", [subprocess.TimeoutExpired(cmd="ffmpeg", timeout=600), OSError("cannot fork")])
    def test_unexpected_ffmpeg_error_releases_the_reservation_and_keeps_the_wav(self, failure, tmp_path):
        """The final name is reserved up front (exclusive touch); an exception
        out of subprocess.run must not leave that empty .flac next to the WAV."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        rec = audio_dir / "rec.wav"
        rec.write_bytes(b"data")

        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("subprocess.run", side_effect=failure),
            pytest.raises(type(failure)),
        ):
            audio_mod._compress_audio(rec, "flac")

        assert rec.read_bytes() == b"data"
        assert sorted(path.name for path in audio_dir.iterdir()) == ["rec.wav"]

    def test_unexpected_container_error_releases_the_reservation_and_keeps_the_wav(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        rec = audio_dir / "rec.wav"
        rec.write_bytes(b"data")

        with (
            patch("shutil.which", return_value=None),
            patch("digue.container.container_status", return_value="running"),
            patch("subprocess.run", side_effect=OSError("docker gone")),
            pytest.raises(OSError),
        ):
            audio_mod._compress_audio(rec, "flac", backend="cpu")

        assert rec.read_bytes() == b"data"
        assert sorted(path.name for path in audio_dir.iterdir()) == ["rec.wav"]

    def test_compression_never_overwrites_an_existing_destination(self, tmp_path):
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"wav-data")
        existing = tmp_path / "rec.flac"
        existing.write_bytes(b"original")

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"), pytest.raises(FileExistsError):
            audio_mod._compress_audio(rec, "flac")

        assert existing.read_bytes() == b"original"
        assert rec.exists()

    @patch("digue.container.container_status", return_value="running")
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_container_fallback_when_host_ffmpeg_missing(self, mock_run, mock_which, mock_status, tmp_path):
        import subprocess

        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"flac-data", stderr=b"")
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"wav-data")

        result = audio_mod._compress_audio(rec, "flac", backend="amd")

        assert result == tmp_path / "rec.flac"
        assert result.read_bytes() == b"flac-data"
        assert not rec.exists()
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[:5] == ["docker", "exec", "-i", CONTAINER_NAME, "ffmpeg"]
        assert "-c:a" in cmd and "flac" in cmd
        assert mock_run.call_args[1]["input"] == b"wav-data"

    @patch("digue.container.container_status", return_value="running")
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_container_fallback_opus(self, mock_run, mock_which, mock_status, tmp_path):
        import subprocess

        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"opus-data", stderr=b"")
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"wav-data")

        result = audio_mod._compress_audio(rec, "opus")

        assert result == tmp_path / "rec.opus"
        assert result.read_bytes() == b"opus-data"
        assert not rec.exists()
        cmd = mock_run.call_args[0][0]
        assert cmd[:5] == ["docker", "exec", "-i", CONTAINER_NAME, "ffmpeg"]
        assert "-c:a" in cmd and "libopus" in cmd
        assert "-f" in cmd and "ogg" in cmd

    @patch("digue.container.container_status", return_value="running")
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_container_fallback_skipped_if_remote_backend(self, mock_run, mock_which, mock_status, tmp_path):
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"wav-data")

        result = audio_mod._compress_audio(rec, "flac", backend="remote")

        assert result == rec
        assert rec.exists()
        mock_run.assert_not_called()

    @patch("digue.container.container_status", return_value=None)
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_container_fallback_skipped_if_container_not_running(self, mock_run, mock_which, mock_status, tmp_path):
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"wav-data")

        result = audio_mod._compress_audio(rec, "flac")

        assert result == rec
        assert rec.exists()
        mock_run.assert_not_called()

    @patch("digue.container.container_status", return_value="running")
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_container_fallback_failure_keeps_wav(self, mock_run, mock_which, mock_status, tmp_path):
        import subprocess

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b"conversion error"
        )
        rec = tmp_path / "rec.wav"
        rec.write_bytes(b"wav-data")

        result = audio_mod._compress_audio(rec, "flac")

        assert result == rec
        assert rec.exists()
        assert not (tmp_path / "rec.flac").exists()


class TestArchiveRecordingAfterCopy:
    """save_audio copies the live file to <stem>.<ext> and only then
    compresses it. When the compression raises (ffmpeg timeout, a docker exec
    error), the copy is already complete and has the exact name the rescue
    would use, so rescuing collided and the live recording stayed in the
    runtime dir with no state pointing at it -- and the user was told the
    audio had failed to save although a good copy existed."""

    def make_config(self, tmp_path):
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        config["dictate"]["audio_format"] = "flac"
        return config

    def test_compression_error_after_the_copy_keeps_the_copy_and_removes_the_live_file(self, tmp_path):
        rec_file = tmp_path / "digue-rec.wav"
        rec_file.write_bytes(b"RIFF" * 100)
        config = self.make_config(tmp_path)

        def boom(*_args, **_kwargs):
            raise subprocess.TimeoutExpired("ffmpeg", 600)

        with (
            patch("digue.audio._run_compression", side_effect=boom),
            patch("digue.notify.send_notification") as mock_notify,
        ):
            archived, rescued_path = audio_mod._archive_recording(
                config, rec_file, "20260906-120000", "0123456789abcdef"
            )

        saved = tmp_path / "audio" / "2026" / "09" / "20260906-120000-0123456789abcdef.wav"
        assert archived is False
        assert rescued_path == saved
        assert saved.read_bytes() == b"RIFF" * 100
        assert not rec_file.exists()
        assert sorted(path.name for path in saved.parent.iterdir()) == [saved.name]
        message = mock_notify.call_args.args[0]
        assert "compress" in message and str(saved) in message

    def test_copy_failure_still_rescues(self, tmp_path):
        rec_file = tmp_path / "digue-rec.wav"
        rec_file.write_bytes(b"RIFF" * 100)
        config = self.make_config(tmp_path)

        with (
            patch("digue.audio._copy_file_exclusive", side_effect=OSError("disk full")),
            patch("digue.notify.send_notification"),
        ):
            archived, rescued_path = audio_mod._archive_recording(
                config, rec_file, "20260906-120000", "0123456789abcdef"
            )

        assert archived is False
        assert rescued_path == tmp_path / "audio" / "2026" / "09" / "20260906-120000-0123456789abcdef.wav"
        assert rescued_path.read_bytes() == b"RIFF" * 100
        assert not rec_file.exists()


class TestSaveAudioConfig:
    def test_default_saves_audio(self):
        config = _default_config()
        assert config["dictate"]["save_audio"] is True

    def test_loads_kebab_key(self, tmp_path):
        import textwrap

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            textwrap.dedent("""\
            [dictate]
            save-audio = false
        """)
        )
        config = load_config(config_path)
        assert config["dictate"]["save_audio"] is False

    def test_flac_source_skips_ffmpeg_when_saving_flac(self, tmp_path):
        rec_file = tmp_path / "rec.flac"
        rec_file.write_bytes(b"fLaC")
        audio_dir = tmp_path / "audio"

        with patch("digue.audio._compress_audio") as mock_compress:
            saved, _ = audio_mod.save_audio(
                rec_file,
                audio_dir,
                audio_format="flac",
                timestamp="20260904-120000",
                take_id="0123456789abcdef",
            )

        mock_compress.assert_not_called()
        assert saved.suffix == ".flac"
        assert saved.read_bytes() == b"fLaC"

    @patch("digue.audio._compress_audio")
    def test_save_audio_passes_backend(self, mock_compress, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        audio_dir = tmp_path / "audio"
        mock_compress.return_value = audio_dir / "2026/01/test.flac"

        audio_mod.save_audio(rec_file, audio_dir, audio_format="flac", timestamp="2026-01-01T00-00-00", backend="amd")

        mock_compress.assert_called_once()
        assert mock_compress.call_args[1]["backend"] == "amd"

    @patch("digue.delivery.send_text")
    @patch("digue.transcribe.transcribe", return_value="hello")
    @patch("digue.audio.save_audio")
    def test_save_audio_false_skips_wav_but_writes_txt(self, mock_save, mock_transcribe, mock_send, tmp_path):
        rec_file = tmp_path / "rec.wav"
        rec_file.write_bytes(b"audio")
        audio_dir = tmp_path / "audio"
        config = _default_config()
        config["dictate"]["save_audio"] = False
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = dictate_mod.finish_dictation(config, rec_file)

        assert result.exit_code == 0
        mock_save.assert_not_called()
        txt_files = list(audio_dir.rglob("*.txt"))
        assert len(txt_files) == 1
        assert txt_files[0].read_text().strip() == "hello"
