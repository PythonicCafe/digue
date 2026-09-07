"""Tests for transcription, ffmpeg fallback, VTT, language detection, and transcribe CLI."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from digue import convert as convert_mod
from digue import transcribe as transcribe_mod
from digue.config import _default_config, apply_cli_overrides

# Transcription


class TestTranscribe:
    @patch("urllib.request.urlopen")
    def test_returns_stripped_text(self, mock_urlopen, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake wav data")

        mock_response = MagicMock()
        mock_response.read.return_value = b"  hello world  \n"
        mock_urlopen.return_value = mock_response

        result = transcribe_mod.transcribe(
            "http://localhost:8178/inference",
            audio_file,
            "en",
        )
        assert result == "hello world"

    @patch("urllib.request.urlopen")
    def test_sends_language_field(self, mock_urlopen, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = b"text"
        mock_urlopen.return_value = mock_response

        transcribe_mod.transcribe("http://localhost:8178/inference", audio_file, "pt")
        request = mock_urlopen.call_args[0][0]
        assert b"language" in request.data
        assert b"pt" in request.data

    @patch("urllib.request.urlopen")
    def test_sends_language_auto_explicitly(self, mock_urlopen, tmp_path):
        """Regression: the server's default language is "en"; "auto" must be sent
        explicitly so whisper detects the language instead of forcing English."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = b"text"
        mock_urlopen.return_value = mock_response

        transcribe_mod.transcribe("http://localhost:8178/inference", audio_file, "auto")
        request = mock_urlopen.call_args[0][0]
        assert b'name="language"' in request.data
        assert b"auto" in request.data

    @patch("urllib.request.urlopen")
    def test_text_format_joins_segments_into_single_line(self, mock_urlopen, tmp_path):
        # Whisper segments start with a space and the server joins them with
        # newlines; text output must be one clean line.
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = b"First segment here.\n Second one.\n Third one.\n"
        mock_urlopen.return_value = mock_response

        result = transcribe_mod.transcribe("http://localhost:8178/inference", audio_file, "pt")
        assert result == "First segment here. Second one. Third one."

    @patch("urllib.request.urlopen")
    def test_status_messages_are_silent_by_default(self, mock_urlopen, tmp_path, capsys):
        # The 400 retry path is a normal condition: no status output unless verbose.
        import urllib.error

        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = b"text"
        error = urllib.error.HTTPError("http://x", 400, "Bad Request", {}, None)
        mock_urlopen.side_effect = [error, mock_response]

        with patch("digue.transcribe._convert_to_wav", return_value=b"wav"):
            result = transcribe_mod.transcribe("http://localhost:8178/inference", audio_file, "pt")
        assert result == "text"
        assert capsys.readouterr().err == ""

    @patch("urllib.request.urlopen")
    def test_status_messages_print_when_verbose(self, mock_urlopen, tmp_path, capsys):
        import urllib.error

        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = b"text"
        error = urllib.error.HTTPError("http://x", 400, "Bad Request", {}, None)
        mock_urlopen.side_effect = [error, mock_response]

        with patch("digue.transcribe._convert_to_wav", return_value=b"wav"):
            result = transcribe_mod.transcribe("http://localhost:8178/inference", audio_file, "pt", verbose=True)
        assert result == "text"
        err = capsys.readouterr().err
        assert "HTTP 400" in err
        assert "Trying ffmpeg conversion" in err

    @patch("urllib.request.urlopen")
    def test_vtt_cues_are_stripped_and_wrapped(self, mock_urlopen, tmp_path):
        # whisper cues start with a space; vtt output must be stripped, and
        # long cues wrapped into max_lines lines of max_line_length
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b"WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n "
            b"uma frase bem comprida que passa do limite de quarenta e dois caracteres\n"
        )
        mock_urlopen.return_value = mock_response

        result = transcribe_mod.transcribe("http://x", audio_file, "pt", response_format="vtt")
        assert "WEBVTT" in result
        for line in result.splitlines():
            if "-->" in line or line.startswith("WEBVTT"):
                continue
            assert line == line.strip()  # no leading/trailing spaces
            assert len(line) <= 42

    @patch("urllib.request.urlopen")
    def test_srt_cues_are_stripped_and_keep_index(self, mock_urlopen, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = b"1\n00:00:00,000 --> 00:00:01,000\n cue com espaco\n\n"
        mock_urlopen.return_value = mock_response

        result = transcribe_mod.transcribe("http://x", audio_file, "pt", response_format="srt")
        lines = result.splitlines()
        assert lines[0] == "1"
        assert "-->" in lines[1]
        assert lines[2] == "cue com espaco"  # stripped

    def test_srt_number_in_cue_text_is_not_treated_as_index(self):
        srt = (
            "1\n"
            "00:00:00,000 --> 00:00:01,000\n"
            "Ano de\n"
            "1984\n"
            "no Brasil\n\n"
            "2\n"
            "00:00:02,000 --> 00:00:03,000\n"
            "Segunda fala\n"
        )
        result = transcribe_mod._post_process_subtitle(srt, "srt", max_line_length=42, max_lines=2)
        lines = result.splitlines()
        assert lines[0] == "1"
        assert lines[1] == "00:00:00,000 --> 00:00:01,000"
        assert lines[2] == "Ano de 1984 no Brasil"
        assert lines[3] == ""
        assert lines[4] == "2"

    @patch("urllib.request.urlopen")
    def test_timestamps_format_from_server_vtt(self, mock_urlopen, tmp_path):
        # -f timestamps: transcribe() gets the server VTT (already stripped),
        # then cmd_transcribe converts it; distinct cues must all appear
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b"WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n Primeira frase.\n\n"
            b"00:00:02.100 --> 00:00:04.000\n Segunda frase com mais conteudo.\n"
        )
        mock_urlopen.return_value = mock_response

        with patch("digue.notify._stderr_is_tty", return_value=True):
            result = transcribe_mod.transcribe("http://x", audio_file, "pt", response_format="vtt")
        converted = convert_mod._convert_content(result, "vtt", "timestamps")
        assert "[00:00:00] Primeira frase." in converted
        assert "[00:00:02] Segunda frase com mais conteudo." in converted

    @patch("urllib.request.urlopen")
    def test_timestamps_format_joins_same_instant(self, mock_urlopen, tmp_path):
        # -f timestamps: one line per timestamp with ALL its text (no wrap);
        # transcribe requests wrapped=off so cues stay single-line and the
        # conversion yields one line per timestamp.
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = b"WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n uma frase bem comprida que passa do limite de quarenta e dois caracteres\n"
        mock_urlopen.return_value = mock_response

        # Simulate the cmd_transcribe path: vtt with wrap disabled, then convert
        with patch("digue.notify._stderr_is_tty", return_value=True):
            vtt = transcribe_mod.transcribe("http://x", audio_file, "pt", response_format="vtt", wrap_cues=False)
        result = convert_mod._convert_content(vtt, "vtt", "timestamps")
        lines = [line for line in result.splitlines() if line.strip()]
        assert len(lines) == 1  # same timestamp, single line
        assert lines[0] == "[00:00:00] uma frase bem comprida que passa do limite de quarenta e dois caracteres"

    @patch("urllib.request.urlopen")
    def test_vtt_format_keeps_lines(self, mock_urlopen, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"data")
        mock_response = MagicMock()
        mock_response.read.return_value = b"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n First segment\n"
        mock_urlopen.return_value = mock_response

        result = transcribe_mod.transcribe("http://localhost:8178/inference", audio_file, "pt", response_format="vtt")
        assert "WEBVTT" in result
        assert "\n" in result


# ffmpeg fallback


class TestTranscribeFfmpegFallback:
    @patch("digue.transcribe._send_audio")
    def test_native_format_sends_directly(self, mock_send, tmp_path):
        mock_send.return_value = "ok"
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"data")
        assert transcribe_mod.transcribe("http://x", audio) == "ok"
        mock_send.assert_called_once()

    @patch("digue.transcribe._convert_to_wav", return_value=b"wav-bytes")
    @patch("digue.transcribe._send_audio")
    def test_converts_unknown_extension_upfront(self, mock_send, mock_convert, tmp_path):
        audio = tmp_path / "file.amr"
        audio.write_bytes(b"data")
        mock_send.return_value = "text"
        result = transcribe_mod.transcribe("http://x", audio)
        assert result == "text"
        mock_convert.assert_called_once_with(audio)

    @patch("digue.transcribe._convert_to_wav", return_value=b"wav-bytes")
    @patch("digue.transcribe._send_audio")
    def test_retries_after_http_400(self, mock_send, mock_convert, tmp_path):
        import urllib.error

        audio = tmp_path / "file.ogg"
        audio.write_bytes(b"data")
        error = urllib.error.HTTPError("http://x", 400, "Bad Request", {}, None)
        mock_send.side_effect = [error, "converted text"]

        result = transcribe_mod.transcribe("http://x", audio)

        assert result == "converted text"
        assert mock_send.call_count == 2
        mock_convert.assert_called_once_with(audio)

    @patch("shutil.which", return_value=None)
    @patch("digue.transcribe._send_audio")
    def test_400_without_ffmpeg_raises(self, mock_send, mock_which, tmp_path):
        import urllib.error

        audio = tmp_path / "file.ogg"
        audio.write_bytes(b"data")
        mock_send.side_effect = urllib.error.HTTPError("http://x", 400, "Bad Request", {}, None)
        with pytest.raises(RuntimeError, match="ffmpeg is not installed"):
            transcribe_mod.transcribe("http://x", audio)

    @patch("shutil.which", return_value=None)
    @patch("digue.transcribe._send_audio")
    def test_unknown_extension_without_ffmpeg_raises(self, mock_send, mock_which, tmp_path):
        audio = tmp_path / "file.amr"
        audio.write_bytes(b"data")
        with pytest.raises(RuntimeError, match="ffmpeg is not installed"):
            transcribe_mod.transcribe("http://x", audio)

    def test_conversion_returns_bytes_not_file(self, tmp_path):
        # _convert_to_wav must work in memory: returns bytes, writes nothing
        audio = tmp_path / "tone.ogg"
        audio.write_bytes(b"x")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b"wav", stderr=b"")
            result = transcribe_mod._convert_to_wav(audio)
            argv = mock_run.call_args[0][0]
        assert result == b"wav"
        assert "pipe:1" in argv
        assert not list(tmp_path.glob("digue-*.wav"))  # no temp files on disk


class TestMultipartRequest:
    def capture_body(self, filename):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["body"] = request.data
            response = MagicMock()
            response.read.return_value = b"ok"
            return response

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            transcribe_mod._multipart_request("http://x/inference", b"AUDIO", {"language": "pt"}, 5, filename=filename)
        return captured["body"].decode("utf-8", errors="replace")

    def test_quote_in_filename_does_not_break_the_header(self):
        """The filename goes into a quoted parameter: an unescaped quote ends
        the value early and the rest lands in the header (server 400 or a
        misread name)."""
        body = self.capture_body('take "final".wav')

        header_line = next(
            line for line in body.split("\r\n") if line.startswith("Content-Disposition") and "file" in line
        )
        assert header_line == 'Content-Disposition: form-data; name="file"; filename="take %22final%22.wav"'

    def test_newline_in_filename_is_stripped(self):
        body = self.capture_body("a\r\nX-Injected: yes.wav")

        assert "X-Injected" not in body.split("\r\n\r\n")[0]
        assert 'filename="aX-Injected: yes.wav"' in body


class TestSendAudioTokenTimestamps:
    @patch("digue.transcribe._multipart_request", return_value="text")
    def test_always_sends_token_timestamps_false(self, mock_multipart, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"data")
        transcribe_mod._send_audio("http://x", audio, "en", "text", 10)
        fields = mock_multipart.call_args[0][2]
        assert fields["token_timestamps"] == "false"

    @patch("digue.transcribe._multipart_request", return_value="text")
    def test_always_sends_language_even_when_auto(self, mock_multipart, tmp_path):
        """Regression: the server's default language is "en" (server.cpp); omitting
        the field made every transcription English. "auto" must be sent as-is so
        whisper detects the language in the same pass."""
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"data")
        transcribe_mod._send_audio("http://x", audio, "auto", "text", 10)
        fields = mock_multipart.call_args[0][2]
        assert fields["language"] == "auto"

    @patch("digue.transcribe._multipart_request", return_value="text")
    def test_sends_language_when_fixed(self, mock_multipart, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"data")
        transcribe_mod._send_audio("http://x", audio, "pt", "text", 10)
        fields = mock_multipart.call_args[0][2]
        assert fields["language"] == "pt"

    @patch("digue.transcribe._multipart_request", return_value="text")
    def test_sends_original_filename(self, mock_multipart, tmp_path):
        audio = tmp_path / "tone_opus.ogg"
        audio.write_bytes(b"data")
        transcribe_mod._send_audio("http://x", audio, "auto", "text", 10, audio_data=b"converted")
        filename = mock_multipart.call_args[1]["filename"]
        assert filename == "tone_opus.ogg"
        audio_data = mock_multipart.call_args[0][1]
        assert audio_data == b"converted"


# VTT simplification


class TestWrapCueLines:
    def test_short_text_returns_single_line(self):
        assert transcribe_mod._wrap_cue_lines("hello world", 42, 2) == ["hello world"]

    def test_empty_text_returns_empty_list(self):
        assert transcribe_mod._wrap_cue_lines("", 42, 2) == []
        assert transcribe_mod._wrap_cue_lines("   ", 42, 2) == []

    def test_wraps_at_word_boundary(self):
        text = "uma frase bem comprida que passa do limite de caracteres"
        result = transcribe_mod._wrap_cue_lines(text, 35, 2)
        assert len(result) == 2
        assert " ".join(result) == text
        assert len(result[0]) <= 35

    def test_overflow_preserves_all_words_on_last_line(self):
        text = (
            "uma frase bem comprida que passa do limite de quarenta e dois caracteres "
            "e continua por mais uma linha inteira de texto"
        )
        result = transcribe_mod._wrap_cue_lines(text, 42, 2)
        assert len(result) == 2
        assert " ".join(result) == text


class TestStripVttTags:
    def test_removes_c_tags(self):
        text = "Hey<00:00:00.440><c> everyone,</c><00:00:00.960><c> I'm</c>"
        assert transcribe_mod._strip_vtt_tags(text) == "Hey everyone, I'm"

    def test_plain_text_unchanged(self):
        assert transcribe_mod._strip_vtt_tags("Hello world") == "Hello world"

    def test_empty_and_whitespace(self):
        assert transcribe_mod._strip_vtt_tags("") == ""
        assert transcribe_mod._strip_vtt_tags("   ") == ""


class TestSimplifyVtt:
    def test_standard_vtt(self):
        vtt = (
            "WEBVTT\n\n"
            "1\n00:00:00.320 --> 00:00:02.000\nHello everyone.\n\n"
            "2\n00:00:02.000 --> 00:00:05.000\nWelcome to the talk.\n\n"
        )
        result = transcribe_mod.simplify_vtt(vtt)
        assert result == "[00:00:00] Hello everyone.\n[00:00:02] Welcome to the talk."

    def test_youtube_rolling_pattern(self):
        vtt = (
            "WEBVTT\nKind: captions\nLanguage: en\n\n"
            "00:00:00.320 --> 00:00:01.990 align:start position:0%\n \n"
            "Hey<00:00:00.440><c> everyone,</c><00:00:00.960><c> I'm</c><00:00:01.120><c> Ishaan</c>\n\n"
            "00:00:01.990 --> 00:00:02.000 align:start position:0%\nHey everyone, I'm Ishaan\n \n\n"
            "00:00:02.000 --> 00:00:03.710 align:start position:0%\nHey everyone, I'm Ishaan\n"
            "and<00:00:02.480><c> today</c><00:00:03.000><c> I'm</c><00:00:03.120><c> going</c>\n\n"
        )
        result = transcribe_mod.simplify_vtt(vtt)
        lines = result.splitlines()
        assert len(lines) == 2
        assert lines[0] == "[00:00:00] Hey everyone, I'm Ishaan"
        assert lines[1] == "[00:00:02] and today I'm going"

    def test_skips_youtube_headers(self):
        vtt = "WEBVTT\nKind: captions\nLanguage: en\n\n00:00:00.000 --> 00:00:01.000\nHello\n"
        result = transcribe_mod.simplify_vtt(vtt)
        assert "Kind:" not in result
        assert result == "[00:00:00] Hello"

    def test_strips_milliseconds(self):
        vtt = "WEBVTT\n\n1\n00:01:23.456 --> 00:01:25.000\nTest line\n"
        result = transcribe_mod.simplify_vtt(vtt)
        assert result == "[00:01:23] Test line"


class TestDetectLanguage:
    def test_detect_language_uses_verbose_json_payload(self, tmp_path):
        """The server only reports the language in verbose_json (plain json returns
        {"text":""} even with detect_language=true)."""
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"data")
        payload = {
            "detected_language": "portuguese",
            "detected_language_probability": 0.999,
            "language_probabilities": {},
        }
        with (
            patch("digue.transcribe._multipart_request", return_value=json.dumps(payload)) as mock_request,
            patch("digue.transcribe.NATIVE_FORMATS", new=frozenset({".wav"})),
        ):
            result = transcribe_mod.detect_language("http://x", audio, timeout=10)
        assert result == "pt"
        fields = mock_request.call_args[0][2]
        assert fields["detect_language"] == "true"
        assert fields["response_format"] == "verbose_json"

    def test_detect_language_accepts_code_from_server(self, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"data")
        payload = {"detected_language": "pt", "detected_language_probability": 0.9, "language_probabilities": {}}
        with patch("digue.transcribe._multipart_request", return_value=json.dumps(payload)):
            assert transcribe_mod.detect_language("http://x", audio, timeout=10) == "pt"

    def test_detect_language_converts_unsupported_format_upfront(self, tmp_path):
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"m4a!")
        payload = {"detected_language": "en", "detected_language_probability": 0.9, "language_probabilities": {}}
        with (
            patch("digue.transcribe._multipart_request", return_value=json.dumps(payload)) as mock_request,
            patch("digue.transcribe._convert_to_wav", return_value=b"wav") as mock_convert,
        ):
            assert transcribe_mod.detect_language("http://x", audio, timeout=10) == "en"
        mock_convert.assert_called_once()
        assert mock_request.call_args[0][1] == b"wav"

    def test_detect_language_retries_with_ffmpeg_after_400(self, tmp_path):
        import urllib.error

        audio = tmp_path / "a.ogg"
        audio.write_bytes(b"ogg!")
        payload = {"detected_language": "en", "detected_language_probability": 0.9, "language_probabilities": {}}
        with (
            patch(
                "digue.transcribe._multipart_request",
                side_effect=[urllib.error.HTTPError("url", 400, "Bad", {}, None), json.dumps(payload)],
            ) as mock_request,
            patch("digue.transcribe._convert_to_wav", return_value=b"wav"),
        ):
            assert transcribe_mod.detect_language("http://x", audio, timeout=10) == "en"
        assert mock_request.call_count == 2

    def test_language_probabilities_via_verbose_json(self, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"data")
        payload = {
            "detected_language": "Portuguese",
            "detected_language_probability": 0.999,
            "language_probabilities": {"pt": 0.999, "en": 0.0005},
        }
        with patch("digue.transcribe._multipart_request", return_value=json.dumps(payload)):
            probs = transcribe_mod.language_probabilities("http://x", audio, timeout=10)
        assert probs["detected"] == ("pt", 0.999)
        assert probs["all"] == {"pt": 0.999, "en": 0.0005}


class TestCmdDetectLanguage:
    @patch("digue.transcribe.detect_language", return_value="pt")
    @patch("digue.container.ensure_server")
    @patch("digue.container.is_server_running", return_value=True)
    def test_prints_language_code(self, mock_running, mock_ensure, mock_detect, tmp_path, capsys):
        config = _default_config()
        args = MagicMock()
        args.audio = tmp_path / "a.wav"
        args.audio.write_bytes(b"data")
        args.json = False

        result = transcribe_mod.cmd_detect_language(args, config)

        assert result == 0
        assert capsys.readouterr().out.strip() == "pt"

    @patch("digue.transcribe.language_probabilities", return_value={"detected": ("pt", 0.999), "all": {"pt": 0.999}})
    @patch("digue.container.ensure_server")
    @patch("digue.container.is_server_running", return_value=True)
    def test_json_output_has_detected_and_all(self, mock_running, mock_ensure, mock_probs, tmp_path, capsys):
        config = _default_config()
        args = MagicMock()
        args.audio = tmp_path / "a.wav"
        args.audio.write_bytes(b"data")
        args.json = True

        result = transcribe_mod.cmd_detect_language(args, config)

        assert result == 0
        output = json.loads(capsys.readouterr().out)
        assert output["language"] == "pt"
        assert output["probability"] == 0.999
        assert output["all"] == {"pt": 0.999}

    @patch("digue.container.ensure_server")
    @patch("digue.container.is_server_running", return_value=True)
    def test_missing_file_gives_clear_error(self, mock_running, mock_ensure, tmp_path, capsys):
        config = _default_config()
        args = MagicMock()
        args.audio = tmp_path / "nope.wav"
        args.json = False

        result = transcribe_mod.cmd_detect_language(args, config)

        assert result == 1
        assert "not found" in capsys.readouterr().err
        mock_ensure.assert_not_called()
        mock_running.assert_not_called()


class TestDetectLanguageVerbose:
    @patch("digue.transcribe._multipart_request", side_effect=RuntimeError("stop"))
    def test_conversion_message_respects_verbose_false(self, mock_request, tmp_path, capsys):
        """Regression: the ffmpeg-conversion notice is progress output; without
        --verbose the stderr stays clean."""
        audio = tmp_path / "a.ogg"
        audio.write_bytes(b"ogg")
        with patch("digue.transcribe._convert_to_wav", return_value=b"wav"), pytest.raises(RuntimeError):
            transcribe_mod.detect_language("http://x", audio, timeout=10, verbose=False)
        assert "ffmpeg" not in capsys.readouterr().err

    @patch("digue.transcribe._multipart_request", side_effect=RuntimeError("stop"))
    def test_conversion_message_shown_with_verbose_true(self, mock_request, tmp_path, capsys):
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"m4a")
        with patch("digue.transcribe._convert_to_wav", return_value=b"wav"), pytest.raises(RuntimeError):
            transcribe_mod.detect_language("http://x", audio, timeout=10, verbose=True)
        assert "ffmpeg" in capsys.readouterr().err

    @patch("digue.transcribe._multipart_request", side_effect=RuntimeError("stop"))
    def test_language_probabilities_message_respects_verbose_false(self, mock_request, tmp_path, capsys):
        audio = tmp_path / "a.ogg"
        audio.write_bytes(b"ogg")
        with patch("digue.transcribe._convert_to_wav", return_value=b"wav"), pytest.raises(RuntimeError):
            transcribe_mod.language_probabilities("http://x", audio, timeout=10, verbose=False)
        assert "ffmpeg" not in capsys.readouterr().err

    def test_cmd_detect_language_passes_verbose(self, tmp_path):
        config = _default_config()
        args = MagicMock()
        args.audio = tmp_path / "a.wav"
        args.audio.write_bytes(b"data")
        args.json = False
        args.verbose = False
        with (
            patch("digue.transcribe.detect_language", return_value="pt") as mock_detect,
            patch("digue.container.ensure_server"),
            patch("digue.container.is_server_running", return_value=True),
        ):
            assert transcribe_mod.cmd_detect_language(args, config) == 0
        assert mock_detect.call_args[1]["verbose"] is False


class TestCmdTranscribeInput:
    @patch("digue.delivery.send_text")
    @patch("digue.transcribe.transcribe", return_value="text")
    @patch("digue.container.ensure_server")
    @patch("digue.container.is_server_running", return_value=True)
    def test_directory_input_gives_clear_error_not_ffmpeg(
        self, mock_running, mock_ensure, mock_transcribe, mock_send, tmp_path, capsys
    ):
        """Regression: a directory passed to transcribe must fail early with a clear error."""
        config = _default_config()
        args = MagicMock()
        args.audio = tmp_path
        args.output = None
        args.response_format = None
        args.language = None
        args.prompt = None
        args.verbose = False

        result = transcribe_mod.cmd_transcribe(args, config)

        assert result == 1
        err = capsys.readouterr().err
        assert "not a file" in err or "is a directory" in err.lower()
        mock_transcribe.assert_not_called()
        mock_ensure.assert_not_called()
        mock_running.assert_not_called()

    @patch("digue.container.ensure_server")
    @patch("digue.container.is_server_running")
    def test_missing_input_does_not_start_server(self, mock_running, mock_ensure, tmp_path, capsys):
        args = MagicMock(audio=tmp_path / "missing.wav")

        assert transcribe_mod.cmd_transcribe(args, _default_config()) == 1

        assert "not found" in capsys.readouterr().err
        mock_ensure.assert_not_called()
        mock_running.assert_not_called()

    @pytest.mark.parametrize(
        "failure",
        [RuntimeError("request failed"), OSError("disk full"), subprocess.TimeoutExpired(["ffmpeg"], 600)],
    )
    @patch("digue.container.ensure_server")
    @patch("digue.container.is_server_running", return_value=True)
    def test_operational_failure_returns_1_without_traceback(
        self, mock_running, mock_ensure, failure, tmp_path, capsys
    ):
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"audio")
        output_path = tmp_path / "transcript.txt"
        args = MagicMock(
            audio=audio_path,
            output=output_path,
            response_format="text",
            language=None,
            prompt=None,
            verbose=False,
        )
        config = _default_config()

        transcribe_result = (
            patch("digue.transcribe.transcribe", side_effect=failure)
            if isinstance(failure, (RuntimeError, subprocess.SubprocessError))
            else patch("digue.transcribe.transcribe", return_value="text")
        )
        write_result = (
            patch("pathlib.Path.write_text", side_effect=failure)
            if isinstance(failure, OSError)
            else patch("pathlib.Path.write_text")
        )
        with transcribe_result, write_result:
            result = transcribe_mod.cmd_transcribe(args, config)

        assert result == 1
        error = capsys.readouterr().err
        assert f"Error: {failure}" in error
        assert "Traceback" not in error

    @patch("digue.transcribe.transcribe", return_value="text")
    @patch("digue.container.is_server_running", return_value=True)
    @patch("digue.container.ensure_server")
    def test_uses_config_prompt_when_cli_absent(self, mock_ensure, mock_running, mock_transcribe, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"audio")
        args = MagicMock(audio=audio, language=None, response_format=None, verbose=False, prompt=None, output=None)
        config = _default_config()
        config["transcribe"]["prompt"] = "Pythonic Café"
        assert transcribe_mod.cmd_transcribe(args, config) == 0
        assert mock_transcribe.call_args.kwargs["prompt"] == "Pythonic Café"

    @patch("digue.transcribe.transcribe", return_value="text")
    @patch("digue.container.is_server_running", return_value=True)
    @patch("digue.container.ensure_server")
    def test_uses_config_timeout(self, mock_ensure, mock_running, mock_transcribe, tmp_path):
        """The server answers only after transcribing the whole file, so the
        wait is a per-file budget the user can raise for long audio on CPU."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"audio")
        args = MagicMock(audio=audio, language=None, response_format=None, verbose=False, prompt=None, output=None)
        config = _default_config()
        config["transcribe"]["timeout"] = 1800
        assert transcribe_mod.cmd_transcribe(args, config) == 0
        assert mock_transcribe.call_args.kwargs["timeout"] == 1800


class TestOutputFilesEndWithOneNewline:
    VTT = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi\n"

    @patch("digue.transcribe.transcribe", return_value=VTT)
    @patch("digue.container.ensure_server")
    @patch("digue.container.is_server_running", return_value=True)
    def test_transcribe_output_file(self, mock_running, mock_ensure, mock_transcribe, tmp_path):
        """VTT/SRT results already end with a newline (see _post_process_subtitle);
        -o appended another one, ending every subtitle file with a blank line."""
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"audio")
        args = MagicMock(audio=audio, output=str(tmp_path / "a.vtt"), response_format="vtt")
        args.language = args.prompt = None
        args.verbose = False
        config = _default_config()
        apply_cli_overrides(args, config)

        assert transcribe_mod.cmd_transcribe(args, config) == 0

        content = (tmp_path / "a.vtt").read_text()
        assert content == self.VTT

    @patch("digue.transcribe.transcribe", return_value=VTT)
    @patch("digue.container.ensure_server")
    @patch("digue.container.is_server_running", return_value=True)
    def test_batch_transcribe_output_file(self, mock_running, mock_ensure, mock_transcribe, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        (input_dir / "a.mp3").write_bytes(b"audio")
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        args = MagicMock(input_dir=input_dir, output_dir=output_dir, response_format="vtt", language=None)
        config = _default_config()
        apply_cli_overrides(args, config)

        assert transcribe_mod.cmd_batch_transcribe(args, config) == 0

        assert (output_dir / "a.vtt").read_text() == self.VTT

    @patch("digue.transcribe.transcribe", return_value="hello")
    @patch("digue.container.ensure_server")
    @patch("digue.container.is_server_running", return_value=True)
    def test_text_output_still_gets_its_newline(self, mock_running, mock_ensure, mock_transcribe, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"audio")
        args = MagicMock(audio=audio, output=str(tmp_path / "a.txt"), response_format="text")
        args.language = args.prompt = None
        args.verbose = False

        transcribe_mod.cmd_transcribe(args, _default_config())

        assert (tmp_path / "a.txt").read_text() == "hello\n"
