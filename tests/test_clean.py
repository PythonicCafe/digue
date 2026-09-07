"""Tests for the clean command."""

from unittest.mock import MagicMock

from digue import audio as audio_mod
from digue.config import _default_config


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
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)
        monkeypatch.setattr("builtins.input", lambda prompt: "n")

        result = audio_mod.cmd_clean(self._args(), config)

        assert result == 1
        err = capsys.readouterr().err
        assert "2 file(s)" in err  # recordings
        assert "1 file(s)" in err  # transcripts
        assert "Aborted" in err
        assert (audio_dir / "2026" / "09" / "20260901-100000.flac").exists()

    def test_without_a_terminal_asks_for_force_instead_of_tracebacking(self, tmp_path, capsys, monkeypatch):
        audio_dir = tmp_path / "audio"
        self._make_audio_files(audio_dir)
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        def closed_stdin(prompt):
            raise EOFError

        monkeypatch.setattr("builtins.input", closed_stdin)

        result = audio_mod.cmd_clean(self._args(), config)

        assert result == 1
        err = capsys.readouterr().err
        assert "--force" in err
        assert "Traceback" not in err
        assert (audio_dir / "2026" / "09" / "20260901-100000.flac").exists()

    def test_removes_on_confirmation(self, tmp_path, capsys, monkeypatch):
        audio_dir = tmp_path / "audio"
        self._make_audio_files(audio_dir)
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)
        monkeypatch.setattr("builtins.input", lambda prompt: "y")

        result = audio_mod.cmd_clean(self._args(), config)

        assert result == 0
        assert list(audio_dir.rglob("*.flac")) == []
        assert list(audio_dir.rglob("*.txt")) == []

    def test_force_removes_without_asking(self, tmp_path, capsys, monkeypatch):
        audio_dir = tmp_path / "audio"
        self._make_audio_files(audio_dir)
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        def fail_input(prompt):
            raise AssertionError("input() must not be called with --force")

        monkeypatch.setattr("builtins.input", fail_input)

        result = audio_mod.cmd_clean(self._args(force=True), config)

        assert result == 0
        assert list(audio_dir.rglob("*.flac")) == []

    def test_what_recordings_keeps_transcripts(self, tmp_path, capsys):
        audio_dir = tmp_path / "audio"
        self._make_audio_files(audio_dir)
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = audio_mod.cmd_clean(self._args(force=True, what="recordings"), config)

        assert result == 0
        assert list(audio_dir.rglob("*.flac")) == []
        assert list(audio_dir.rglob("*.txt"))

    def test_what_transcripts_keeps_recordings(self, tmp_path, capsys):
        audio_dir = tmp_path / "audio"
        self._make_audio_files(audio_dir)
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = audio_mod.cmd_clean(self._args(force=True, what="transcripts"), config)

        assert result == 0
        assert list(audio_dir.rglob("*.txt")) == []
        assert list(audio_dir.rglob("*.flac"))

    def test_removes_empty_month_directories(self, tmp_path, capsys):
        audio_dir = tmp_path / "audio"
        self._make_audio_files(audio_dir)
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        audio_mod.cmd_clean(self._args(force=True), config)

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
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = audio_mod.cmd_clean(self._args(force=True), config)

        assert result == 0
        assert all(path.exists() for path in foreign)
        assert not list(month.glob("2026090*"))
        assert "Removed 3 file(s)" in capsys.readouterr().err

    def test_regular_files_named_like_year_or_month_do_not_crash(self, tmp_path, capsys):
        audio_dir = tmp_path / "audio"
        self._make_audio_files(audio_dir)
        (audio_dir / "2025").write_text("a file, not a year directory")
        (audio_dir / "2026" / "08").write_text("a file, not a month directory")
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = audio_mod.cmd_clean(self._args(force=True), config)

        assert result == 0
        assert "Removed 3 file(s)" in capsys.readouterr().err
        assert (audio_dir / "2025").exists()
        assert (audio_dir / "2026" / "08").exists()

    def test_confirmation_lists_the_files_it_will_remove(self, tmp_path, capsys, monkeypatch):
        audio_dir = tmp_path / "audio"
        self._make_audio_files(audio_dir)
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)
        monkeypatch.setattr("builtins.input", lambda prompt: "n")

        audio_mod.cmd_clean(self._args(), config)

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
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = audio_mod.cmd_clean(self._args(force=True), config)

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
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = audio_mod.cmd_clean(self._args(force=True), config)

        assert result == 0
        assert "Nothing to remove" in capsys.readouterr().err
        assert json_path.exists()

    def test_json_is_never_removed_as_transcript(self, tmp_path, capsys):
        audio_dir = tmp_path / "audio"
        month = audio_dir / "2026" / "09"
        month.mkdir(parents=True)
        json_path = month / "20260904-120000-0123456789abcdef.json"
        json_path.write_text('{"state": "rescued"}')
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = audio_mod.cmd_clean(self._args(force=True, what="transcripts"), config)

        assert result == 0
        assert "Nothing to remove" in capsys.readouterr().err
        assert json_path.exists()

    def test_nothing_to_remove(self, tmp_path, capsys):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        config = _default_config()
        config["dictate"]["audio_dir"] = str(audio_dir)

        result = audio_mod.cmd_clean(self._args(), config)

        assert result == 0
        assert "Nothing to remove" in capsys.readouterr().err

    def test_missing_audio_dir(self, tmp_path, capsys):
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "nonexistent")

        result = audio_mod.cmd_clean(self._args(), config)

        assert result == 0
