"""Tests for subtitle conversion and silent-audio VTT handling."""

from unittest.mock import MagicMock, patch

import pytest

import digue
from digue.config import _default_config


class TestCmdConvert:
    def _make_vtt(self, tmp_path):
        vtt_file = tmp_path / "a.vtt"
        vtt_file.write_text(
            "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nHello\n\n2\n00:00:01.000 --> 00:00:02.000\nWorld\n"
        )
        return vtt_file

    def test_vtt_to_txt_file(self, tmp_path, capsys):
        self._make_vtt(tmp_path)
        args = MagicMock()
        args.input = str(tmp_path / "a.vtt")
        args.output = str(tmp_path / "b.txt")
        args.from_format = None
        args.to_format = None
        config = _default_config()
        result = digue.cmd_convert(args, config)
        assert result == 0
        assert "Saved:" in capsys.readouterr().err
        content = (tmp_path / "b.txt").read_text()
        assert "[00:00:00] Hello" in content
        assert "[00:00:01] World" in content

    def test_vtt_to_stdout_with_to_format(self, tmp_path, capsys):
        self._make_vtt(tmp_path)
        args = MagicMock()
        args.input = str(tmp_path / "a.vtt")
        args.output = None
        args.from_format = None
        args.to_format = "text"
        config = _default_config()
        result = digue.cmd_convert(args, config)
        assert result == 0
        assert capsys.readouterr().out.strip() == "Hello World"

    def test_stdin_requires_from_format(self, capsys):
        args = MagicMock()
        args.input = "-"
        args.output = "b.txt"
        args.from_format = None
        args.to_format = None
        config = _default_config()
        result = digue.cmd_convert(args, config)
        assert result == 1
        assert "from-format is required" in capsys.readouterr().err

    def test_directory_input_gives_clear_error(self, tmp_path, capsys):
        args = MagicMock()
        args.input = str(tmp_path)
        args.output = None
        args.from_format = "vtt"
        args.to_format = "text"
        config = _default_config()
        result = digue.cmd_convert(args, config)
        assert result == 1
        err = capsys.readouterr().err
        assert "not a file" in err
        assert "Traceback" not in err

    def test_stdin_with_from_format(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr("sys.stdin", __import__("io").StringIO("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi\n"))
        args = MagicMock()
        args.input = "-"
        args.output = str(tmp_path / "b.txt")
        args.from_format = "vtt"
        args.to_format = None
        config = _default_config()
        result = digue.cmd_convert(args, config)
        assert result == 0
        assert "[00:00:00] Hi" in (tmp_path / "b.txt").read_text()

    def test_unknown_extension_requires_from_format(self, tmp_path, capsys):
        weird = tmp_path / "subtitles.xyz"
        weird.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi\n")
        args = MagicMock()
        args.input = str(weird)
        args.output = None
        args.from_format = None
        args.to_format = "text"
        config = _default_config()
        result = digue.cmd_convert(args, config)
        assert result == 1
        assert "from-format" in capsys.readouterr().err

    def test_txt_to_stdout_without_to_format_defaults_to_text(self, tmp_path, capsys):
        # .txt input (timestamps content), no -t, stdout: defaults to text
        source = tmp_path / "a.txt"
        source.write_text("[00:00:00] Hello\n[00:00:01] World\n")
        args = MagicMock()
        args.input = str(source)
        args.output = None
        args.from_format = None
        args.to_format = None
        config = _default_config()
        result = digue.cmd_convert(args, config)
        assert result == 0
        assert capsys.readouterr().out.strip() == "Hello World"

    def test_timestamps_to_vtt(self, tmp_path, capsys):
        source = tmp_path / "a.txt"
        source.write_text("[00:00:01] Hello\n[00:00:02] World\n")
        args = MagicMock()
        args.input = str(source)
        args.output = None
        args.from_format = None
        args.to_format = "vtt"
        config = _default_config()
        result = digue.cmd_convert(args, config)
        assert result == 0
        out = capsys.readouterr().out
        assert out.startswith("WEBVTT")
        assert "00:00:01.000 --> 00:00:02.000" in out
        assert "00:00:02.000 --> 00:00:04.000" in out
        assert "Hello" in out

    def test_multiline_vtt_cues_roundtrip_through_timestamps(self):
        vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nLinha 1\nLinha 2\n\n00:00:04.000 --> 00:00:06.000\nLinha 3\n"
        timestamps = digue._convert_content(vtt, "vtt", "timestamps")
        assert "[00:00:01] Linha 1 Linha 2" in timestamps
        result_vtt = digue._convert_content(timestamps, "timestamps", "vtt")
        assert "00:00:01.000 --> 00:00:04.000" in result_vtt
        assert "Linha 1 Linha 2" in result_vtt

    def test_srt_to_vtt_preserves_times_milliseconds_and_multiline_cues(self):
        content = (
            "1\n00:00:01,234 --> 00:00:03,456\nFirst line\nSecond line\n\n"
            "2\n01:02:03,007 --> 01:02:05,089\nAnother cue\n"
        )

        result = digue._convert_content(content, "srt", "vtt")

        assert result == (
            "WEBVTT\n\n"
            "00:00:01.234 --> 00:00:03.456\nFirst line\nSecond line\n\n"
            "01:02:03.007 --> 01:02:05.089\nAnother cue\n"
        )

    def test_vtt_to_srt_preserves_times_milliseconds_and_multiline_cues(self):
        content = (
            "WEBVTT\n\n"
            "intro\n00:00:00.125 --> 00:00:02.750 align:start\nHello\nworld\n\n"
            "00:01:03.004 --> 00:01:04.999\nLast cue\n"
        )

        result = digue._convert_content(content, "vtt", "srt")

        assert result == (
            "1\n00:00:00,125 --> 00:00:02,750\nHello\nworld\n\n2\n00:01:03,004 --> 00:01:04,999\nLast cue\n"
        )

    def test_timestamp_end_is_next_start_and_last_cue_has_two_second_duration(self):
        content = "[00:00:01] First\n[00:00:03] Last\n"

        result = digue._convert_content(content, "timestamps", "srt")

        assert "00:00:01,000 --> 00:00:03,000" in result
        assert "00:00:03,000 --> 00:00:05,000" in result

    def test_text_without_timestamps_cannot_become_vtt(self, tmp_path, capsys):
        source = tmp_path / "a.txt"
        source.write_text("just plain text\nanother line\n")
        args = MagicMock()
        args.input = str(source)
        args.output = None
        args.from_format = "text"
        args.to_format = "vtt"
        config = _default_config()
        result = digue.cmd_convert(args, config)
        assert result == 1
        assert "cannot convert" in capsys.readouterr().err

    def test_srt_roundtrip(self, tmp_path, capsys):
        srt_file = tmp_path / "a.srt"
        srt_file.write_text("1\n00:00:00,320 --> 00:00:01,000\nHello\n\n")
        args = MagicMock()
        args.input = str(srt_file)
        args.output = None
        args.from_format = None
        args.to_format = "text"
        config = _default_config()
        result = digue.cmd_convert(args, config)
        assert result == 0
        assert "Hello" in capsys.readouterr().out


class TestParseSubtitleTimestamp:
    def test_parses_three_part_timestamp(self):
        assert digue._parse_subtitle_timestamp("01:02:03.456") == 3723456
        assert digue._parse_subtitle_timestamp("01:02:03,456") == 3723456

    def test_parses_two_part_vtt_timestamp_without_hours(self):
        assert digue._parse_subtitle_timestamp("02:03.456") == 123456
        assert digue._parse_subtitle_timestamp("00:05.100") == 5100

    def test_invalid_timestamps_raise_value_error(self):
        with pytest.raises(ValueError, match="invalid subtitle timestamp"):
            digue._parse_subtitle_timestamp("invalid")
        with pytest.raises(ValueError, match="invalid subtitle timestamp"):
            digue._parse_subtitle_timestamp("00:60:00.000")


# -- Batch commands -----------------------------------------------------------


class TestSilentAudioTimestamps:
    """whisper-server answers a silent file with a bare "WEBVTT" header (no
    cues); the timestamps format is built from that VTT and must not fail."""

    def test_convert_header_only_vtt_to_timestamps_gives_empty_text(self):
        assert digue._convert_content("WEBVTT\n\n", "vtt", "timestamps") == ""

    def test_convert_header_only_vtt_to_text_gives_empty_text(self):
        assert digue._convert_content("WEBVTT\n", "vtt", "text") == ""

    def test_convert_header_only_vtt_to_srt_gives_empty_srt(self):
        """Silent audio is an empty subtitle, not a conversion error: SRT must
        degrade the same way timestamps and text already do."""
        assert digue._convert_content("WEBVTT\n\n", "vtt", "srt") == "\n"

    def test_non_subtitle_content_is_still_rejected(self):
        with pytest.raises(ValueError, match="does not look like a VTT"):
            digue._convert_content("just some prose\n", "vtt", "timestamps")

    def test_header_followed_by_garbage_is_rejected(self):
        """A WEBVTT header does not make everything after it an empty file."""
        with pytest.raises(ValueError, match="does not look like a VTT"):
            digue._convert_content("WEBVTT\n\nsome text without any timing line\n", "vtt", "timestamps")

    def test_header_with_metadata_blocks_only_is_empty(self):
        content = (
            "WEBVTT\nKind: captions\nLanguage: pt\n\nNOTE\nmade by digue\nsecond line\n\nSTYLE\n::cue { color: red }\n"
        )
        assert digue._convert_content(content, "vtt", "timestamps") == ""

    @patch("digue.transcribe", return_value="WEBVTT\n")
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    def test_cmd_transcribe_timestamps_on_silent_audio_succeeds(
        self, mock_running, mock_ensure, mock_transcribe, tmp_path, capsys
    ):
        audio = tmp_path / "silence.wav"
        audio.write_bytes(b"audio")
        args = MagicMock()
        args.audio = audio
        args.output = None
        args.response_format = "timestamps"
        args.language = None
        args.prompt = None
        args.verbose = False

        result = digue.cmd_transcribe(args, _default_config())

        captured = capsys.readouterr()
        assert result == 0
        assert captured.out == "\n"
        assert "does not look like" not in captured.err

    @patch("digue.transcribe", return_value="WEBVTT\n")
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    def test_batch_transcribe_timestamps_on_silent_audio_is_not_a_failure(
        self, mock_running, mock_ensure, mock_transcribe, tmp_path
    ):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (input_dir / "silence.mp3").write_bytes(b"audio")
        args = MagicMock()
        args.input_dir = input_dir
        args.output_dir = output_dir
        args.response_format = "timestamps"
        args.language = None

        result = digue.cmd_batch_transcribe(args, _default_config())

        assert result == 0
        assert (output_dir / "silence.txt").read_text() == "\n"
