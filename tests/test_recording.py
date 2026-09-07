"""Tests for take state, recorder start/stop, watchdog, runtime isolation, and orphan recovery."""

import contextlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

import digue
from digue import audio as audio_mod
from digue import dictate as dictate_mod
from digue import recording as recording_mod
from digue.config import _default_config

# -- Recording ----------------------------------------------------------------


class TestTakeState:
    def make_state(self, tmp_path, **changes):
        values = {
            "version": recording_mod.TAKE_STATE_VERSION,
            "take_id": "0123456789abcdef",
            "created_at_ns": 123,
            "state": "starting",
            "rec_file": tmp_path / "digue-recording.wav",
            "daemon_pid": 42,
            "daemon_starttime": 99,
            "recorder_pid": None,
            "recorder_starttime": None,
            "recoverer_pid": None,
            "recoverer_starttime": None,
        }
        values.update(changes)
        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            return recording_mod.TakeState(**values)

    def test_round_trip_and_ordering(self, tmp_path):
        newer = self.make_state(tmp_path, take_id="ffffffffffffffff", created_at_ns=124)
        older = self.make_state(tmp_path)
        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            recording_mod._write_take_state(newer)
            recording_mod._write_take_state(older)
            assert recording_mod._read_take_state(tmp_path / "digue-take-0123456789abcdef.json") == older
            assert recording_mod._take_states() == [older, newer]

    @pytest.mark.parametrize(
        "changes",
        [
            {"version": 2},
            {"take_id": "not-hex"},
            {"created_at_ns": 0},
            {"state": "unknown"},
            {"rec_file": Path("/outside/digue-recording.wav")},
            {"rec_file": Path("PLACEHOLDER")},
            {"recorder_pid": 1},
            {"state": "recording"},
            {
                "state": "recording",
                "recorder_pid": 1,
                "recorder_starttime": 2,
                "recoverer_pid": 3,
                "recoverer_starttime": 4,
            },
            {"state": "recovering"},
        ],
    )
    def test_rejects_invalid_combinations(self, tmp_path, changes):
        if changes.get("rec_file") == Path("PLACEHOLDER"):
            changes["rec_file"] = tmp_path / "other.wav"
        with pytest.raises(ValueError):
            self.make_state(tmp_path, **changes)

    def test_accepts_each_valid_state(self, tmp_path):
        self.make_state(tmp_path)
        self.make_state(tmp_path, state="recording", recorder_pid=10, recorder_starttime=20)
        self.make_state(tmp_path, state="delivering", recorder_pid=10, recorder_starttime=20)
        self.make_state(tmp_path, state="recovering", recoverer_pid=30, recoverer_starttime=40)
        self.make_state(
            tmp_path,
            state="recovering",
            recorder_pid=10,
            recorder_starttime=20,
            recoverer_pid=30,
            recoverer_starttime=40,
        )

    def test_truncated_state_is_reported_and_never_removed(self, tmp_path, capsys):
        state_path = tmp_path / "digue-take-0123456789abcdef.json"
        state_path.write_text('{"version":')
        wav_path = tmp_path / "digue-recording.wav"
        wav_path.write_bytes(b"audio")

        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            assert recording_mod._take_states() == []

        assert state_path.exists()
        assert wav_path.exists()
        assert str(state_path) in capsys.readouterr().err


class TestStateFilesAreWrittenAtomically:
    """Path.write_text truncates before writing: a concurrent toggle reading in
    between sees an empty file. For the daemon file that reads as "no daemon"
    (the toggle starts a new take) and for a take state that reads as a corrupt
    "unreadable" state (reported, never removed). The state must appear in one
    step (temp sibling + rename), never through a truncating open."""

    @staticmethod
    def _truncating_writes(monkeypatch):
        opened = []
        real_open = Path.open

        def spy_open(self, mode="r", *args, **kwargs):
            if "w" in mode:
                opened.append(self)
            return real_open(self, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", spy_open)
        return opened

    def test_daemon_state_is_never_truncated_in_place(self, tmp_path, monkeypatch):
        opened = self._truncating_writes(monkeypatch)
        daemon_file = tmp_path / "digue-daemon.pid"
        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            dictate_mod._write_daemon_state(os.getpid(), "recording")

        assert daemon_file not in opened
        assert daemon_file.read_text().split()[:2] == [str(os.getpid()), "recording"]

    @patch("digue.recording._spawn_limit_watchdog")
    @patch("subprocess.Popen")
    def test_take_state_is_never_truncated_in_place(self, mock_popen, mock_watchdog, tmp_path, monkeypatch):
        mock_popen.return_value = MagicMock(pid=os.getpid())
        opened = self._truncating_writes(monkeypatch)
        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            processes = recording_mod.start_recording(_default_config())

        state_file = tmp_path / f"digue-take-{processes.take_id}.json"
        assert state_file not in opened
        assert json.loads(state_file.read_text())["state"] == "recording"


class TestRecordingCommand:
    @patch("shutil.which")
    def test_auto_prefers_pw_record(self, mock_which, tmp_path):
        mock_which.return_value = "/usr/bin/pw-record"
        argv = recording_mod.recording_command(tmp_path / "rec.wav", recorder="auto")
        assert argv[0] == "pw-record"

    @patch("shutil.which")
    def test_auto_falls_back_to_arecord(self, mock_which, tmp_path):
        mock_which.side_effect = lambda name: None if name == "pw-record" else "/usr/bin/arecord"
        argv = recording_mod.recording_command(tmp_path / "rec.wav", recorder="auto")
        assert argv[0] == "arecord"

    def test_pw_record_argv(self, tmp_path):
        argv = recording_mod.recording_command(tmp_path / "rec.wav", recorder="pw-record")
        assert argv[0] == "pw-record"
        assert "--rate" in argv and "16000" in argv
        assert argv[-1].endswith("rec.wav")

    def test_arecord_argv(self, tmp_path):
        argv = recording_mod.recording_command(tmp_path / "rec.wav", recorder="arecord")
        assert argv[0] == "arecord"
        assert "16000" in argv

    def test_unknown_recorder_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="Unknown recorder"):
            recording_mod.recording_command(tmp_path / "rec.wav", recorder="sox")

    def test_pw_record_omits_target_when_device_is_empty(self, tmp_path):
        argv = recording_mod.recording_command(tmp_path / "rec.wav", recorder="pw-record", device="")
        assert "--target" not in argv

    def test_pw_record_passes_target(self, tmp_path):
        argv = recording_mod.recording_command(tmp_path / "rec.wav", recorder="pw-record", device="alsa_input.usb")
        assert argv[argv.index("--target") + 1] == "alsa_input.usb"

    def test_arecord_passes_device(self, tmp_path):
        argv = recording_mod.recording_command(tmp_path / "rec.wav", recorder="arecord", device="hw:2,0")
        assert argv[argv.index("-D") + 1] == "hw:2,0"

    def test_arecord_omits_device_when_empty(self, tmp_path):
        argv = recording_mod.recording_command(tmp_path / "rec.wav", recorder="arecord", device="")
        assert "-D" not in argv

    def test_pw_record_adds_container_for_flac_suffix(self, tmp_path):
        argv = recording_mod.recording_command(tmp_path / "rec.flac", recorder="pw-record")
        assert argv[argv.index("--container") + 1] == "flac"

    def test_pw_record_wav_has_no_container_flag(self, tmp_path):
        argv = recording_mod.recording_command(tmp_path / "rec.wav", recorder="pw-record")
        assert "--container" not in argv


class TestLiveRecordingFormat:
    def test_wav_audio_format_always_records_wav(self):
        config = _default_config()
        config["dictate"]["audio_format"] = "wav"
        with patch("digue.recording._pw_record_supports_flac", return_value=True):
            assert recording_mod._live_recording_suffix(config) == ".wav"

    def test_flac_with_pw_record_support_records_flac(self):
        config = _default_config()
        config["dictate"]["audio_format"] = "flac"
        config["dictate"]["recorder"] = "pw-record"
        with patch("digue.recording._pw_record_supports_flac", return_value=True):
            assert recording_mod._live_recording_suffix(config) == ".flac"

    def test_flac_without_container_falls_back_to_wav(self):
        config = _default_config()
        config["dictate"]["audio_format"] = "flac"
        config["dictate"]["recorder"] = "pw-record"
        with patch("digue.recording._pw_record_supports_flac", return_value=False):
            assert recording_mod._live_recording_suffix(config) == ".wav"

    def test_arecord_never_records_flac(self):
        config = _default_config()
        config["dictate"]["audio_format"] = "flac"
        config["dictate"]["recorder"] = "arecord"
        with patch("digue.recording._pw_record_supports_flac", return_value=True):
            assert recording_mod._live_recording_suffix(config) == ".wav"

    def test_supports_flac_parses_list_containers(self, tmp_path):
        binary = tmp_path / "pw-record"
        binary.write_bytes(b"")
        cache = tmp_path / "cache"
        cache.mkdir()
        listed = MagicMock(stdout="    wav: WAV (Microsoft)\n    flac: FLAC (Free Lossless Audio Codec)\n")
        with (
            patch("digue.recording._cache_dir", return_value=cache),
            patch("shutil.which", return_value=str(binary)),
            patch("subprocess.run", return_value=listed),
        ):
            assert recording_mod._pw_record_supports_flac() is True
        assert "flac" in (cache / "pw-record-containers").read_text()

    def test_supports_flac_false_when_missing(self, tmp_path):
        binary = tmp_path / "pw-record"
        binary.write_bytes(b"")
        cache = tmp_path / "cache"
        cache.mkdir()
        listed = MagicMock(stdout="    wav: WAV (Microsoft)\n")
        with (
            patch("digue.recording._cache_dir", return_value=cache),
            patch("shutil.which", return_value=str(binary)),
            patch("subprocess.run", return_value=listed),
        ):
            assert recording_mod._pw_record_supports_flac() is False


class TestStartRecording:
    @patch("subprocess.Popen")
    def test_returns_owned_recorder_handle(self, mock_popen, tmp_path):
        recorder = MagicMock(pid=1234)
        mock_popen.return_value = recorder
        config = _default_config()
        config["dictate"]["max_duration"] = 0

        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            processes = recording_mod.start_recording(config)

        assert processes.recorder is recorder
        assert processes.watchdog is None
        assert processes.rec_file is not None
        assert processes.rec_file.name.startswith("digue-")
        assert processes.rec_file.suffix in {".wav", ".flac"}
        assert mock_popen.call_args[1].get("start_new_session") is True

    @patch("subprocess.Popen")
    def test_keeps_the_recorder_stderr_for_the_owner(self, mock_popen, tmp_path):
        """The stderr file outlives the startup check: a recorder that dies
        mid-take (PipeWire restarted) explains itself there, and the owner
        reports it instead of a bare "Empty or missing audio file"."""
        recorder = MagicMock(pid=1234, returncode=1)

        def popen(argv, stdout=None, stderr=None, start_new_session=False):
            stderr.write("stream disconnected\n")
            recorder.poll.return_value = None
            return recorder

        mock_popen.side_effect = popen
        config = _default_config()
        config["dictate"]["max_duration"] = 0

        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            processes = recording_mod.start_recording(config)
            assert processes.stderr_path is not None
            assert processes.stderr_path.exists()
            assert recording_mod._consume_recorder_stderr(processes) == "stream disconnected"
            assert not processes.stderr_path.exists()
            assert recording_mod._consume_recorder_stderr(processes) == "exit code 1"

    @patch("digue.recording._spawn_limit_watchdog")
    @patch("subprocess.Popen")
    def test_max_duration_returns_watchdog_handle(self, mock_popen, mock_watchdog, tmp_path):
        recorder = MagicMock(pid=os.getpid())
        watchdog = MagicMock(pid=778)
        mock_popen.return_value = recorder
        mock_watchdog.return_value = watchdog
        config = _default_config()
        config["dictate"]["max_duration"] = 300

        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            processes = recording_mod.start_recording(config)

        assert processes.recorder is recorder
        assert processes.watchdog is watchdog
        mock_watchdog.assert_called_once_with(recorder.pid, 300)

    def test_start_recording_passes_device_to_recorder(self, tmp_path):
        recorder = MagicMock(pid=os.getpid())
        config = _default_config()
        config["dictate"]["max_duration"] = 0
        config["dictate"]["recorder"] = "pw-record"
        config["dictate"]["device"] = "alsa_input.usb"
        config["dictate"]["audio_format"] = "wav"

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._live_recording_suffix", return_value=".wav"),
            patch("subprocess.Popen", return_value=recorder) as mock_popen,
        ):
            recording_mod.start_recording(config)

        argv = mock_popen.call_args[0][0]
        assert argv[argv.index("--target") + 1] == "alsa_input.usb"


class TestNativeFlacTake:
    """With audio-format = "flac" and a pw-record whose libsndfile has the flac
    container, the live take is a .flac: the take state and the /proc fd scan
    must accept it, or the default config cannot record at all on such a
    machine (the state validator raised before the recorder even started)."""

    @patch("subprocess.Popen")
    def test_start_recording_publishes_a_flac_take(self, mock_popen, tmp_path):
        mock_popen.return_value = MagicMock(pid=os.getpid())
        config = _default_config()
        config["dictate"]["max_duration"] = 0
        config["dictate"]["recorder"] = "pw-record"

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._pw_record_supports_flac", return_value=True),
        ):
            processes = recording_mod.start_recording(config)
            states = recording_mod._take_states()

        assert processes.rec_file is not None and processes.rec_file.suffix == ".flac"
        assert [take.state for take in states] == ["recording"]
        assert states[0].rec_file == processes.rec_file
        argv = mock_popen.call_args[0][0]
        assert argv[argv.index("--container") + 1] == "flac"

    def test_take_state_accepts_flac_and_rejects_other_suffixes(self, tmp_path):
        def make(name):
            return recording_mod.TakeState(
                version=recording_mod.TAKE_STATE_VERSION,
                take_id="0123456789abcdef",
                created_at_ns=1,
                state="starting",
                rec_file=tmp_path / name,
                daemon_pid=1,
                daemon_starttime=1,
            )

        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            assert make("digue-take.flac").rec_file.suffix == ".flac"
            with pytest.raises(ValueError, match="digue-\\*\\.wav or \\.flac"):
                make("digue-take.opus")

    def test_recording_file_of_finds_an_open_flac(self, tmp_path):
        flac = tmp_path / "digue-take.flac"
        with flac.open("wb") as open_flac, patch("digue.recording._runtime_dir", return_value=tmp_path):
            open_flac.write(b"x")
            found = recording_mod._recording_file_of(os.getpid())

        assert found == flac.resolve()


class TestStartRecordingPublishesTakeState:
    """The take identity is published before the recorder exists and gains the
    recorder identity before the watchdog is spawned: publishing the state is
    two syscalls (~50 us) while spawning the watchdog is fork+exec (~10 ms), so
    a recorder with identity (which recovery knows how to stop) is the state
    that is exposed the soonest."""

    @patch("digue.recording._spawn_limit_watchdog")
    @patch("subprocess.Popen")
    def test_publishes_starting_before_popen(self, mock_popen, mock_watchdog, tmp_path):
        states_at_popen = []
        recorder = MagicMock(pid=os.getpid())

        def fake_popen(*args, **kwargs):
            states_at_popen.extend(recording_mod._take_states())
            return recorder

        mock_popen.side_effect = fake_popen
        config = _default_config()
        config["dictate"]["max_duration"] = 0

        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            processes = recording_mod.start_recording(config)

        assert [take.state for take in states_at_popen] == ["starting"]
        assert processes.take_id == states_at_popen[0].take_id
        assert processes.rec_file == states_at_popen[0].rec_file

    @patch("subprocess.Popen")
    def test_publishes_recording_with_identity_before_watchdog(self, mock_popen, tmp_path):
        states_at_watchdog = []
        recorder = MagicMock(pid=os.getpid())

        def fake_watchdog(pgid, max_duration):
            states_at_watchdog.extend(recording_mod._take_states())
            return MagicMock(pid=9999)

        mock_popen.return_value = recorder
        config = _default_config()
        config["dictate"]["max_duration"] = 300

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._spawn_limit_watchdog", side_effect=fake_watchdog),
        ):
            processes = recording_mod.start_recording(config)

        assert processes.take_id is not None
        assert [take.state for take in states_at_watchdog] == ["recording"]
        take = states_at_watchdog[0]
        assert take.recorder_pid == recorder.pid
        assert take.recorder_starttime == int(recording_mod._process_starttime(os.getpid()))

    @patch("subprocess.Popen", side_effect=FileNotFoundError("pw-record"))
    def test_popen_failure_removes_state(self, mock_popen, tmp_path):
        config = _default_config()

        with patch("digue.recording._runtime_dir", return_value=tmp_path), pytest.raises(FileNotFoundError):
            recording_mod.start_recording(config)

        assert list(tmp_path.glob("digue-take-*.json")) == []

    def test_immediate_nonzero_exit_raises_with_stderr(self, tmp_path):
        config = _default_config()
        dead = MagicMock(pid=4242)
        dead.poll.return_value = 1

        def fake_popen(argv, stdout=None, stderr=None, start_new_session=False):
            if stderr is not None:
                stderr.write("no such node 'usb-mic'\n")
                stderr.flush()
            return dead

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._live_recording_suffix", return_value=".wav"),
            patch("subprocess.Popen", side_effect=fake_popen),
            pytest.raises(RuntimeError, match="pw-record failed: no such node"),
        ):
            recording_mod.start_recording(config)

        assert list(tmp_path.glob("digue-take-*.json")) == []

    def test_toggle_publishes_take_state_and_removes_it_after_delivery(self, tmp_path):
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        config["dictate"]["max_duration"] = 0
        recorder = MagicMock(pid=os.getpid(), poll=lambda: 0)
        finish_take_ids = []

        def fake_finish(_config, rec_file, limit_reached=False, take_id=None):
            finish_take_ids.append(take_id)
            # the take state follows the daemon file: "delivering" is written
            # before the recorder is stopped, so it is what the delivery sees
            # and what a recovery of a daemon killed mid-delivery reads
            assert [take.state for take in recording_mod._take_states()] == ["delivering"]
            return dictate_mod.DeliveryResult(outcome="delivered", exit_code=0)

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.container.ensure_server"),
            patch("digue.container.is_server_running", return_value=True),
            patch("subprocess.Popen", return_value=recorder),
            patch("digue.recording._wait_recorder_end_daemon", return_value="ended"),
            patch("digue.dictate.finish_dictation", side_effect=fake_finish),
            patch("digue.notify.send_notification"),
            patch("digue.notify.notify_close"),
            patch("signal.signal"),
        ):
            assert dictate_mod.dictate_toggle(config) == 0

        assert len(finish_take_ids) == 1 and finish_take_ids[0] is not None
        assert list(tmp_path.glob("digue-take-*.json")) == []


class TestSpawnLimitWatchdog:
    @patch("digue.recording._process_starttime", return_value="98765")
    @patch("subprocess.Popen")
    def test_watchdog_validates_process_identity_before_killing(self, mock_popen, mock_starttime):
        watchdog = MagicMock(pid=5000)
        mock_popen.return_value = watchdog

        result = recording_mod._spawn_limit_watchdog(4242, 300)

        argv = mock_popen.call_args[0][0]
        script = argv[2]
        assert argv[-3:] == ["4242", "98765", str(300 + recording_mod.WATCHDOG_GRACE_SECONDS)]
        assert "/proc/{pid}/stat" in script
        assert "os.killpg(pid, signal.SIGTERM)" in script
        assert result is watchdog
        assert mock_popen.call_args[1].get("start_new_session") is True

    @patch("digue.recording._process_starttime", return_value="98765")
    @patch("subprocess.Popen")
    def test_watchdog_fires_only_after_the_daemon_limit(self, mock_popen, mock_starttime):
        """The daemon polls every 200ms and drifts; a watchdog sleeping exactly
        max_duration wins the race (measured: at 20s the recorder was already
        dead when the daemon checked), so the daemon saw "died" and never
        notified the limit. The watchdog is a safety net and must fire later."""
        recording_mod._spawn_limit_watchdog(4242, 20)

        sleep_seconds = int(mock_popen.call_args[0][0][-1])
        assert sleep_seconds >= 20 + 2

    def test_watchdog_script_does_not_kill_a_pid_that_no_longer_leads_a_group(self):
        """The script promises the same identity check as _recorder_identity_valid:
        starttime AND pgrp == pid (killpg on a pid that is not a group leader
        is refused by the kernel today, but the script must not rely on that).
        Runs the real script against real processes: a session leader is
        killed, a plain child (pgrp = the test's group, not its own pid) is
        left alone."""
        import subprocess

        leader = subprocess.Popen(["sleep", "30"], start_new_session=True)
        member = subprocess.Popen(["sleep", "30"])
        try:
            with patch("subprocess.Popen") as mock_popen:
                recording_mod._spawn_limit_watchdog(leader.pid, 0)
                script = mock_popen.call_args[0][0][2]
            for process in (leader, member):
                starttime = recording_mod._process_starttime(process.pid)
                subprocess.run(
                    [sys.executable, "-c", script, str(process.pid), str(starttime), "0"], timeout=10, check=True
                )
            assert leader.wait(timeout=5) == -15
            assert member.poll() is None
        finally:
            for process in (leader, member):
                process.kill()
                process.wait(timeout=5)

    def test_process_starttime_parses_comm_with_spaces_and_parentheses(self, tmp_path):
        stat = tmp_path / "stat"
        stat.write_text("4242 (odd name) value) S " + " ".join(str(value) for value in range(4, 30)))

        assert recording_mod._process_starttime(4242, stat_path=stat) == "22"


class TestWaitRecorderEndDaemon:
    def test_polls_owned_process_and_collects_spontaneous_exit(self):
        recorder = MagicMock()
        recorder.poll.side_effect = [None, 7]

        with patch("time.monotonic", side_effect=[0.0, 0.1, 0.3]), patch("time.sleep"):
            outcome = recording_mod._wait_recorder_end_daemon(recorder, 300)

        assert outcome == "died"
        recorder.wait.assert_called_once_with(timeout=0)

    def test_sigterm_returns_manual(self):
        recorder = MagicMock()
        recorder.poll.return_value = None
        with patch("time.monotonic", side_effect=[0.0, 0.5]), patch("time.sleep"):
            dictate_mod._got_sigterm = True
            try:
                assert recording_mod._wait_recorder_end_daemon(recorder, 300) == "manual"
            finally:
                dictate_mod._got_sigterm = False

    def test_sigterm_during_the_wait_returns_manual(self):
        """Regression: the flag must be read live from `digue.dictate` on
        every poll. A `from digue.dictate import _got_sigterm` inside the
        function snapshots False at call time, and a SIGTERM (second toggle)
        or SIGINT arriving while the loop runs was ignored until the limit."""
        recorder = MagicMock()
        recorder.poll.return_value = None
        assert dictate_mod._got_sigterm is False

        def sleep_then_signal(_seconds):
            dictate_mod._on_sigterm(15, None)

        with (
            patch("time.monotonic", side_effect=[0.0, 0.1, 0.3, 0.5]),
            patch("time.sleep", side_effect=sleep_then_signal),
        ):
            try:
                assert recording_mod._wait_recorder_end_daemon(recorder, 300) == "manual"
            finally:
                dictate_mod._got_sigterm = False

    def test_sigint_during_the_wait_returns_interrupted(self):
        recorder = MagicMock()
        recorder.poll.return_value = None
        assert dictate_mod._got_sigint is False

        def sleep_then_signal(_seconds):
            dictate_mod._on_sigint(2, None)

        with (
            patch("time.monotonic", side_effect=[0.0, 0.1, 0.3, 0.5]),
            patch("time.sleep", side_effect=sleep_then_signal),
        ):
            try:
                assert recording_mod._wait_recorder_end_daemon(recorder, 300) == "interrupted"
            finally:
                dictate_mod._got_sigint = False

    def test_exit_after_the_limit_counts_as_limit(self):
        """If the watchdog killed the recorder first, the outcome is still the
        duration limit, not a spontaneous death (the notification depends on it)."""
        recorder = MagicMock()
        recorder.poll.side_effect = [None, -15]

        with patch("time.monotonic", side_effect=[0.0, 0.1, 300.2]), patch("time.sleep"):
            outcome = recording_mod._wait_recorder_end_daemon(recorder, 300)

        assert outcome == "limit"


class TestFinishOwnedRecorder:
    def test_spontaneously_exited_recorder_is_not_signaled_again(self, tmp_path):
        rec_file = tmp_path / "take.wav"
        rec_file.write_bytes(b"audio")
        recorder = MagicMock(pid=777)
        recorder.poll.return_value = 1

        with patch("digue.recording.stop_recording_pid") as mock_stop:
            result = recording_mod._finish_owned_recorder(recorder, rec_file)

        assert result == rec_file
        mock_stop.assert_not_called()


class TestCancelWatchdog:
    def test_normal_stop_terminates_and_collects_watchdog(self):
        watchdog = MagicMock()
        watchdog.poll.return_value = None

        recording_mod._cancel_watchdog(watchdog)

        watchdog.terminate.assert_called_once_with()
        watchdog.wait.assert_called_once_with(timeout=5)

    def test_no_watchdog_is_a_noop(self):
        recording_mod._cancel_watchdog(None)


class TestGlobalRecordingStateRemoved:
    """Take states superseded the global pid file: `is_recording` /
    `stop_recording` had no per-take identity (a global stop would kill
    another take's recorder), and `_recorder_pid_file` duplicated the
    recorder identity already published in the take state."""

    @pytest.mark.parametrize(
        "name",
        ["_pid_file", "is_recording", "stop_recording", "_recorder_pid_file"],
    )
    def test_superseded_global_state_functions_are_gone(self, name):
        assert not hasattr(digue, name)

    def test_stop_recording_pid_has_no_newest_wav_fallback(self, tmp_path):
        """With concurrent takes, the newest-runtime-wav fallback could grab
        another daemon's recording: without a captured rec_file and with an fd
        scan that finds nothing (recorder already dead, descriptors closed),
        there is nothing to deliver."""
        newest = tmp_path / "digue-newest.wav"
        newest.write_bytes(b"audio")

        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            assert recording_mod.stop_recording_pid(999999, expected_starttime=None) is None

        assert newest.exists()


class TestStopRecordingPidIdentity:
    """killpg assumes the pid is still the process-group leader; a recycled pid
    could belong to an unrelated process. Before each signal, both the /proc
    starttime and pgrp == pid are revalidated; on divergence the recorder is
    not signaled and the validated WAV is returned."""

    def make_rec_file(self, tmp_path):
        rec_file = tmp_path / "take.wav"
        rec_file.write_bytes(b"audio")
        return rec_file

    @patch("digue.recording._group_alive", return_value=False)
    @patch("os.killpg")
    @patch("digue.recording._process_starttime", return_value="999")
    def test_starttime_divergence_skips_killpg(self, mock_starttime, mock_killpg, mock_alive, tmp_path):
        rec_file = self.make_rec_file(tmp_path)

        result = recording_mod.stop_recording_pid(4242, rec_file, expected_starttime="111")

        mock_killpg.assert_not_called()
        assert result == rec_file

    @patch("digue.recording._group_alive", return_value=False)
    @patch("os.killpg")
    @patch("digue.recording._process_pgrp", return_value=5151)
    @patch("digue.recording._process_starttime", return_value="111")
    def test_pgrp_mismatch_skips_killpg(self, mock_starttime, mock_pgrp, mock_killpg, mock_alive, tmp_path):
        rec_file = self.make_rec_file(tmp_path)

        result = recording_mod.stop_recording_pid(4242, rec_file, expected_starttime="111")

        mock_killpg.assert_not_called()
        assert result == rec_file

    @patch("digue.recording._group_alive", return_value=False)
    @patch("os.killpg")
    @patch("digue.recording._process_pgrp", return_value=4242)
    @patch("digue.recording._process_starttime", return_value="111")
    def test_matching_identity_signals_the_group(self, mock_starttime, mock_pgrp, mock_killpg, mock_alive, tmp_path):
        rec_file = self.make_rec_file(tmp_path)

        result = recording_mod.stop_recording_pid(4242, rec_file, expected_starttime="111")

        mock_killpg.assert_called_once_with(4242, 15)
        assert result == rec_file

    @patch("digue.recording._group_alive", return_value=True)
    @patch("os.killpg")
    @patch("digue.recording._process_pgrp", return_value=4242)
    @patch("digue.recording._process_starttime", side_effect=["111", "999"])
    def test_identity_is_rechecked_before_sigkill(self, mock_starttime, mock_pgrp, mock_killpg, mock_alive, tmp_path):
        rec_file = self.make_rec_file(tmp_path)

        with patch("time.monotonic", side_effect=[0.0, 0.1, 0.6]):
            result = recording_mod.stop_recording_pid(4242, rec_file, expected_starttime="111")

        assert list(mock_killpg.call_args_list) == [call(4242, 15)]
        assert result == rec_file

    def test_stop_returns_as_soon_as_the_group_is_gone(self, tmp_path):
        """The stop slept a fixed 0.5s before checking the group, and the
        exited-but-unreaped recorder (a zombie still answers signal 0) made it
        escalate to SIGKILL and sleep again: a full second on every dictation
        (measured 1.00s). A recorder that exits on SIGTERM within
        milliseconds must be reported gone within milliseconds."""
        import time

        recorder = subprocess.Popen(["sleep", "60"], start_new_session=True)
        rec_file = tmp_path / "take.wav"
        rec_file.write_bytes(b"audio")
        try:
            start = time.perf_counter()
            result = recording_mod.stop_recording_pid(recorder.pid, rec_file)
            elapsed = time.perf_counter() - start
        finally:
            recorder.wait(timeout=5)

        assert result == rec_file
        assert elapsed < 0.3


class TestRecordingFileOf:
    def test_finds_the_open_wav_through_a_symlinked_runtime_dir(self, tmp_path):
        """readlink returns the kernel's resolved path; a symlinked
        XDG_RUNTIME_DIR must not hide the recorder's file (TakeState already
        compares resolved paths)."""

        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir)
        wav = link_dir / "digue-take.wav"
        with wav.open("wb") as open_wav, patch("digue.recording._runtime_dir", return_value=link_dir):
            open_wav.write(b"x")
            found = recording_mod._recording_file_of(os.getpid())

        assert found == real_dir / "digue-take.wav"


class TestDaemonAlive:
    def test_zombie_daemon_is_not_alive(self, tmp_path):
        """A daemon that exited but was not reaped still answers signal 0 and
        keeps its /proc starttime: without the zombie check a toggle would
        SIGTERM it and exit 0 believing it stopped a recording."""
        stat = tmp_path / "stat"
        fields = ["Z"] + [str(value) for value in range(4, 30)]
        fields[19] = "555"
        stat.write_text("4242 (digue) " + " ".join(fields))

        with (
            patch("digue.recording._pid_alive", return_value=True),
            patch("digue.recording._process_starttime", return_value="555"),
            patch("digue.recording._process_is_zombie", return_value=True),
        ):
            assert dictate_mod._daemon_alive((4242, "recording", "555")) is False
        with (
            patch("digue.recording._pid_alive", return_value=True),
            patch("digue.recording._process_starttime", return_value="555"),
            patch("digue.recording._process_is_zombie", return_value=False),
        ):
            assert dictate_mod._daemon_alive((4242, "recording", "555")) is True


class TestTakeIdentityAlive:
    """The same liveness predicate as _daemon_alive: a zombie owner (exited,
    not yet reaped) still answers signal 0 and keeps its starttime but can
    neither stop a recorder nor deliver. Counting it as alive leaves its take
    unclaimed while the daemon file already reads as dead, so the next toggle
    starts a second take on top of a recorder nobody will stop."""

    def alive_patches(self, zombie):
        return (
            patch("digue.recording._pid_alive", return_value=True),
            patch("digue.recording._process_starttime", return_value="555"),
            patch("digue.recording._process_is_zombie", return_value=zombie),
        )

    def test_zombie_owner_is_not_alive(self):
        with contextlib.ExitStack() as stack:
            for patcher in self.alive_patches(zombie=True):
                stack.enter_context(patcher)
            assert recording_mod._take_identity_alive(4242, 555) is False
        with contextlib.ExitStack() as stack:
            for patcher in self.alive_patches(zombie=False):
                stack.enter_context(patcher)
            assert recording_mod._take_identity_alive(4242, 555) is True

    def test_take_of_a_zombie_daemon_is_claimed(self, tmp_path):
        import time

        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            take = recording_mod.TakeState(
                version=recording_mod.TAKE_STATE_VERSION,
                take_id="0123456789abcdef",
                created_at_ns=time.time_ns(),
                state="recording",
                rec_file=tmp_path / "digue-recording.wav",
                daemon_pid=4242,
                daemon_starttime=555,
                recorder_pid=4243,
                recorder_starttime=556,
            )
            recording_mod._write_take_state(take)
            with contextlib.ExitStack() as stack:
                for patcher in self.alive_patches(zombie=True):
                    stack.enter_context(patcher)
                stack.enter_context(patch("digue.recording._process_starttime", side_effect=lambda pid: "555"))
                claimed = recording_mod._claim_orphan_take()

        assert claimed is not None and claimed.state == "recovering"

    def test_expire_orphan_starting_treats_a_zombie_daemon_as_dead(self, tmp_path):
        import time

        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            take = recording_mod.TakeState(
                version=recording_mod.TAKE_STATE_VERSION,
                take_id="0123456789abcdef",
                created_at_ns=time.time_ns() - int((recording_mod.ORPHAN_MIN_AGE_SECONDS + 1) * 1e9),
                state="starting",
                rec_file=tmp_path / "digue-recording.wav",
                daemon_pid=4242,
                daemon_starttime=555,
            )
            recording_mod._write_take_state(take)
            with contextlib.ExitStack() as stack:
                for patcher in self.alive_patches(zombie=True):
                    stack.enter_context(patcher)
                recording_mod._expire_orphan_starting(_default_config(), take)

        assert list(tmp_path.glob("digue-take-*.json")) == []


class TestRuntimeIsolation:
    def test_state_paths_use_isolated_runtime_dir(self):
        runtime_dir = Path(os.environ["XDG_RUNTIME_DIR"])

        assert recording_mod._runtime_dir() == runtime_dir
        assert dictate_mod._daemon_pid_file().parent == runtime_dir
        assert recording_mod._take_state_file("0123456789abcdef").parent == runtime_dir
        with dictate_mod._dictate_lock():
            assert (runtime_dir / "digue.lock").exists()

    def test_toggle_does_not_read_state_outside_isolated_runtime(self, tmp_path):
        """A stale daemon state outside the isolated runtime must not be signaled."""
        sentinel_dir = tmp_path / "sentinel"
        sentinel_dir.mkdir()
        (sentinel_dir / "digue-daemon.pid").write_text("4242 recording 555")
        config = _default_config()

        with (
            patch("digue.recording._process_starttime", return_value="555"),
            patch("digue.container.ensure_server", side_effect=RuntimeError("stop after state lookup")),
            patch("digue.notify.send_notification"),
            patch("os.kill") as mock_kill,
        ):
            assert dictate_mod.dictate_toggle(config) == 1

        mock_kill.assert_not_called()


class TestOrphanStartingTake:
    """A daemon that died between publishing 'starting' and registering the
    recorder leaves an orphaned state. Recovery is conservative (no /proc fd
    scanning): recent states are left alone, a missing/empty WAV expires with
    its state, and a non-empty WAV is rescued (never transcribed/pasted
    automatically -- the recorder may still be writing)."""

    def make_starting_take(self, tmp_path, age_seconds, rec_file=None, daemon_pid=999999, daemon_starttime=1):
        import time

        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            take = recording_mod.TakeState(
                version=recording_mod.TAKE_STATE_VERSION,
                take_id="0123456789abcdef",
                created_at_ns=time.time_ns() - int(age_seconds * 1e9),
                state="starting",
                rec_file=rec_file or tmp_path / "digue-recording.wav",
                daemon_pid=daemon_pid,
                daemon_starttime=daemon_starttime,
            )
            recording_mod._write_take_state(take)
        return take

    def test_expired_orphan_without_wav_is_removed(self, tmp_path):
        take = self.make_starting_take(tmp_path, age_seconds=recording_mod.ORPHAN_MIN_AGE_SECONDS + 1)
        config = _default_config()

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._pid_alive", return_value=False),
        ):
            rescued = recording_mod._expire_orphan_starting(config, take)

        assert rescued is None
        assert list(tmp_path.glob("digue-take-*.json")) == []

    def test_expired_orphan_with_wav_is_rescued(self, tmp_path):
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        take = self.make_starting_take(tmp_path, age_seconds=recording_mod.ORPHAN_MIN_AGE_SECONDS + 1)
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._pid_alive", return_value=False),
        ):
            rescued = recording_mod._expire_orphan_starting(config, take)

        assert rescued is not None and rescued.read_bytes() == b"audio"
        assert rescued.name.endswith("-0123456789abcdef.wav")
        assert not rec_file.exists()
        assert list(tmp_path.glob("digue-take-*.json")) == []
        # same metadata contract as the surplus rescue: the take JSON travels
        # with the audio, state "rescued"
        import json

        archived_state = json.loads(rescued.with_suffix(".json").read_text())
        assert archived_state["state"] == "rescued"
        assert archived_state["take_id"] == "0123456789abcdef"

    def test_recent_orphan_is_left_alone(self, tmp_path):
        take = self.make_starting_take(tmp_path, age_seconds=1)
        config = _default_config()

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._pid_alive", return_value=False),
        ):
            assert recording_mod._expire_orphan_starting(config, take) is None

        assert len(list(tmp_path.glob("digue-take-*.json"))) == 1

    def test_orphan_with_alive_daemon_is_left_alone(self, tmp_path):
        take = self.make_starting_take(
            tmp_path, age_seconds=recording_mod.ORPHAN_MIN_AGE_SECONDS + 1, daemon_starttime=555
        )
        config = _default_config()

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._pid_alive", return_value=True),
            patch("digue.recording._process_starttime", return_value="555"),
        ):
            assert recording_mod._expire_orphan_starting(config, take) is None

        assert len(list(tmp_path.glob("digue-take-*.json"))) == 1

    def test_toggle_rescues_expired_orphan_starting_take(self, tmp_path):
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        self.make_starting_take(tmp_path, age_seconds=recording_mod.ORPHAN_MIN_AGE_SECONDS + 1, rec_file=rec_file)

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._pid_alive", lambda pid: pid == os.getpid()),
            patch("digue.container.ensure_server"),
            patch("digue.container.is_server_running", return_value=True),
            patch("digue.recording.start_recording") as mock_start,
            patch("digue.dictate.finish_dictation") as mock_finish,
            patch("digue.notify.send_notification"),
            patch("digue.notify.notify_close"),
            patch("signal.signal"),
        ):
            assert dictate_mod.dictate_toggle(config) == 0

        rescued = list((tmp_path / "audio").rglob("*-0123456789abcdef.wav"))
        assert len(rescued) == 1
        assert list(tmp_path.glob("digue-take-*.json")) == []
        mock_finish.assert_not_called()  # never transcribed/pasted automatically
        mock_start.assert_not_called()  # the toggle returns after the rescue


class TestOrphanTakeClaim:
    """Orphan takes (daemon provably dead, or a recovering take whose recoverer
    died) are claimed for recovery under the dictate lock: exactly one claimer
    wins, the transition to "recovering" carries the claimer's identity, and an
    old orphan never blocks stopping the current recording."""

    def make_take(self, tmp_path, take_id="0123456789abcdef", created_at_ns=100, **changes):
        import time

        values = {
            "version": recording_mod.TAKE_STATE_VERSION,
            "take_id": take_id,
            "created_at_ns": created_at_ns or time.time_ns(),
            "state": "recording",
            "rec_file": tmp_path / "digue-recording.wav",
            "daemon_pid": 999999,
            "daemon_starttime": 1,
            "recorder_pid": 555,
            "recorder_starttime": 666,
        }
        values.update(changes)
        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            take = recording_mod.TakeState(**values)
            recording_mod._write_take_state(take)
        return take

    def test_claims_the_oldest_orphan_and_records_the_recoverer_identity(self, tmp_path):
        newer = self.make_take(tmp_path, take_id="ffffffffffffffff", created_at_ns=200)
        oldest = self.make_take(tmp_path)

        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            claimed = recording_mod._claim_orphan_take()

        assert claimed is not None
        assert claimed.take_id == oldest.take_id
        assert claimed.state == "recovering"
        assert claimed.recorder_pid == 555
        assert claimed.recoverer_pid == os.getpid()
        assert claimed.recoverer_starttime == int(recording_mod._process_starttime(os.getpid()))
        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            states = {take.take_id: take for take in recording_mod._take_states()}
        assert states[oldest.take_id].state == "recovering"
        assert states[newer.take_id].state == "recording"

    def test_take_with_live_daemon_is_not_claimed(self, tmp_path):
        self.make_take(
            tmp_path, daemon_pid=os.getpid(), daemon_starttime=int(recording_mod._process_starttime(os.getpid()))
        )

        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            assert recording_mod._claim_orphan_take() is None

    def test_recovering_take_with_live_recoverer_is_not_claimed(self, tmp_path):
        self.make_take(
            tmp_path,
            state="recovering",
            recorder_pid=None,
            recorder_starttime=None,
            recoverer_pid=os.getpid(),
            recoverer_starttime=int(recording_mod._process_starttime(os.getpid())),
        )

        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            assert recording_mod._claim_orphan_take() is None

    def test_recovering_take_with_dead_recoverer_is_claimed(self, tmp_path):
        self.make_take(
            tmp_path,
            state="recovering",
            recorder_pid=None,
            recorder_starttime=None,
            recoverer_pid=999999,
            recoverer_starttime=1,
        )

        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            claimed = recording_mod._claim_orphan_take()

        assert claimed is not None
        assert claimed.state == "recovering"
        assert claimed.recoverer_pid == os.getpid()

    def test_concurrent_recoverers_exactly_one_claims(self, tmp_path):
        """Two recoverers racing for the same orphan: the dictate lock makes
        the second one see a recovering take with a live recoverer (the first
        claimer, this same process) and claim nothing."""

        self.make_take(tmp_path)
        first_in_lock = threading.Event()
        release_first = threading.Event()
        claimed = []

        def recover_first():
            with dictate_mod._dictate_lock():
                first_in_lock.set()
                assert release_first.wait(2)
                claimed.append(recording_mod._claim_orphan_take())

        def recover_second():
            assert first_in_lock.wait(2)
            release_first.set()
            with dictate_mod._dictate_lock():
                claimed.append(recording_mod._claim_orphan_take())

        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            thread_first = threading.Thread(target=recover_first)
            thread_second = threading.Thread(target=recover_second)
            thread_first.start()
            assert first_in_lock.wait(2)
            thread_second.start()
            release_first.set()
            thread_first.join(timeout=2)
            thread_second.join(timeout=2)

        assert not thread_first.is_alive()
        assert not thread_second.is_alive()
        assert claimed[0] is not None
        assert claimed[1] is None

    def test_toggle_signals_current_recording_and_leaves_orphan_alone(self, tmp_path):
        """A second press must stop the current recording even with an orphan
        take waiting: an old orphan never blocks the toggle."""
        daemon_file = tmp_path / "digue-daemon.pid"
        daemon_file.write_text("4242 recording 555")
        self.make_take(tmp_path)
        config = _default_config()

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._pid_alive", return_value=True),
            patch("digue.recording._process_starttime", return_value="555"),
            patch("digue.notify.send_notification"),
            patch("os.kill") as mock_kill,
        ):
            assert dictate_mod.dictate_toggle(config) == 0

        mock_kill.assert_called_once_with(4242, 15)
        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            states = recording_mod._take_states()
        assert len(states) == 1 and states[0].state == "recording"

    def test_toggle_recovers_the_claimed_orphan_and_does_not_start_a_new_take(self, tmp_path):
        """The recovering toggle delivers the orphan and returns: recording
        here too would leave this take and a concurrent one competing for the
        same daemon state (two recorders, one stop)."""

        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        self.make_take(tmp_path)
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        finish_calls = []

        def fake_finish(_config, file, limit_reached=False, take_id=None):
            finish_calls.append((file, take_id))
            return dictate_mod.DeliveryResult(outcome="delivered", exit_code=0)

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._pid_alive", lambda pid: pid == os.getpid()),
            patch("digue.container.ensure_server"),
            patch("digue.container.is_server_running", return_value=True),
            patch("digue.recording.stop_recording_pid", return_value=rec_file) as mock_stop,
            patch("digue.dictate.finish_dictation", side_effect=fake_finish),
            patch("digue.recording.start_recording") as mock_start,
            patch("digue.notify.send_notification"),
            patch("digue.notify.notify_close"),
            patch("signal.signal"),
        ):
            assert dictate_mod.dictate_toggle(config) == 0

        assert finish_calls == [(rec_file, "0123456789abcdef")]
        mock_stop.assert_called_once()
        mock_start.assert_not_called()
        assert list(tmp_path.glob("digue-take-*.json")) == []
        assert not (tmp_path / "digue-daemon.pid").exists()

    def test_toggle_pressed_during_recovery_starts_exactly_one_new_take(self, tmp_path):
        """Regression: while a toggle recovers an orphan, a second press must
        find a single consistent picture -- no daemon state (so it starts a new
        take) and the recovering toggle never records afterwards."""

        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        self.make_take(tmp_path)
        config = _default_config()
        seen: dict[str, object] = {}

        def fake_finish(_config, file, limit_reached=False, take_id=None):
            with patch("digue.recording._runtime_dir", return_value=tmp_path):
                seen["daemon_state_during_recovery"] = dictate_mod._daemon_state()
                seen["claim_during_recovery"] = recording_mod._claim_orphan_take()
            return dictate_mod.DeliveryResult(outcome="delivered", exit_code=0)

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._pid_alive", lambda pid: pid == os.getpid()),
            patch("digue.container.ensure_server"),
            patch("digue.container.is_server_running", return_value=True),
            patch("digue.recording.stop_recording_pid", return_value=rec_file),
            patch("digue.dictate.finish_dictation", side_effect=fake_finish),
            patch("digue.recording.start_recording") as mock_start,
            patch("digue.notify.send_notification"),
            patch("digue.notify.notify_close"),
        ):
            assert dictate_mod.dictate_toggle(config) == 0

        # a second toggle at that instant sees no daemon and nothing to claim:
        # it would publish "starting" and record, as the only recorder.
        assert seen == {"daemon_state_during_recovery": None, "claim_during_recovery": None}
        mock_start.assert_not_called()

    def test_young_starting_orphan_is_not_claimed(self, tmp_path):
        """A starting orphan younger than the minimum age cannot be acted on,
        so claiming it would make the toggle return having done nothing; it is
        left alone and the toggle records normally."""
        self.make_take(
            tmp_path,
            state="starting",
            created_at_ns=None,
            recorder_pid=None,
            recorder_starttime=None,
        )

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._pid_alive", return_value=False),
        ):
            assert recording_mod._claim_orphan_take() is None
            states = recording_mod._take_states()

        assert len(states) == 1 and states[0].state == "starting"

    def test_recovery_failure_to_reach_the_server_keeps_the_claim_for_retry(self, tmp_path):
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        self.make_take(tmp_path)
        config = _default_config()

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._pid_alive", return_value=False),
            patch("digue.container.ensure_server", return_value=None),
            patch("digue.container.is_server_running", return_value=False),
            patch("digue.recording.stop_recording_pid") as mock_stop,
            patch("digue.dictate.finish_dictation") as mock_finish,
            patch("digue.recording.start_recording") as mock_start,
            patch("digue.notify.send_notification"),
        ):
            assert dictate_mod.dictate_toggle(config) == 1

        mock_stop.assert_not_called()
        mock_finish.assert_not_called()
        mock_start.assert_not_called()
        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            states = recording_mod._take_states()
        assert len(states) == 1 and states[0].state == "recovering"
        assert rec_file.exists()

    def test_toggle_during_slow_recovery_starts_a_new_take(self, tmp_path):
        """A recovering take with a live recoverer is like a delivering one:
        the toggle does not touch it and starts a new take."""

        self.make_take(
            tmp_path,
            state="recovering",
            recorder_pid=None,
            recorder_starttime=None,
            recoverer_pid=os.getpid(),
            recoverer_starttime=int(recording_mod._process_starttime(os.getpid())),
        )
        config = _default_config()

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._pid_alive", lambda pid: pid == os.getpid()),
            patch("digue.container.ensure_server", side_effect=RuntimeError("abort startup")),
            patch("digue.notify.send_notification"),
            patch("os.kill") as mock_kill,
        ):
            assert dictate_mod.dictate_toggle(config) == 1

        mock_kill.assert_not_called()
        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            states = recording_mod._take_states()
        assert len(states) == 1
        assert states[0].state == "recovering"
        assert states[0].recoverer_pid == os.getpid()


class TestRecoverClaimedTake:
    """Recovery of a claimed orphan runs outside the dictate lock: a live
    recorder with a valid identity is stopped through it, a dead one's WAV
    goes straight to delivery, and the claimed state is removed only after a
    terminal outcome (retryable failures and unexpected exceptions preserve
    state and WAV for the next toggle)."""

    def make_recovering_take(self, tmp_path, age_seconds=10.0, **changes):
        import time

        values = {
            "version": recording_mod.TAKE_STATE_VERSION,
            "take_id": "0123456789abcdef",
            "created_at_ns": time.time_ns() - int(age_seconds * 1e9),
            "state": "recovering",
            "rec_file": tmp_path / "digue-recording.wav",
            "daemon_pid": 999999,
            "daemon_starttime": 1,
            "recorder_pid": 888888,
            "recorder_starttime": 2,
            "recoverer_pid": 777777,
            "recoverer_starttime": 3,
        }
        values.update(changes)
        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            take = recording_mod.TakeState(**values)
            recording_mod._write_take_state(take)
        return take

    def make_config(self, tmp_path):
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        return config

    def patch_recovery(self, tmp_path, finish_result):
        return (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.dictate.finish_dictation", return_value=finish_result),
        )

    def test_stops_a_live_recorder_with_valid_identity_and_delivers(self, tmp_path):
        recorder = subprocess.Popen(["sleep", "60"], start_new_session=True)
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        take = self.make_recovering_take(
            tmp_path,
            recorder_pid=recorder.pid,
            recorder_starttime=int(recording_mod._process_starttime(recorder.pid)),
            rec_file=rec_file,
        )
        delivered = []

        def fake_finish(_config, file, limit_reached=False, take_id=None):
            delivered.append((file, take_id))
            return dictate_mod.DeliveryResult(outcome="delivered", exit_code=0)

        try:
            with (
                patch("digue.recording._runtime_dir", return_value=tmp_path),
                patch("digue.dictate.finish_dictation", side_effect=fake_finish),
            ):
                exit_code = recording_mod._recover_claimed_take(self.make_config(tmp_path), take)
        finally:
            recorder.wait(timeout=5)

        assert exit_code == 0
        assert delivered == [(rec_file, take.take_id)]
        assert recorder.poll() is not None
        assert list(tmp_path.glob("digue-take-*.json")) == []

    def test_dead_recorder_with_wav_delivers_without_signaling(self, tmp_path):
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        take = self.make_recovering_take(tmp_path, rec_file=rec_file)
        delivered = []

        def fake_finish(_config, file, limit_reached=False, take_id=None):
            delivered.append(file)
            return dictate_mod.DeliveryResult(outcome="delivered", exit_code=0)

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording.os.killpg") as mock_killpg,
            patch("digue.dictate.finish_dictation", side_effect=fake_finish),
        ):
            exit_code = recording_mod._recover_claimed_take(self.make_config(tmp_path), take)

        assert exit_code == 0
        mock_killpg.assert_not_called()
        assert delivered == [rec_file]
        assert list(tmp_path.glob("digue-take-*.json")) == []

    def test_take_with_a_saved_transcript_is_archived_without_pasting_again(self, tmp_path):
        """A daemon that died after pasting (during the archive) left the
        transcript behind: recovery must not paste that text a second time."""
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        take = self.make_recovering_take(tmp_path, rec_file=rec_file)
        config = self.make_config(tmp_path)
        month_dir = Path(config["dictate"]["audio_dir"]) / "2026" / "09"
        month_dir.mkdir(parents=True)
        (month_dir / f"20260905-101500-{take.take_id}.txt").write_text("already pasted\n")

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.delivery.send_text") as mock_send,
            patch("digue.transcribe.transcribe") as mock_transcribe,
            patch("digue.notify.send_notification"),
        ):
            exit_code = recording_mod._recover_claimed_take(config, take)

        assert exit_code == 0
        mock_send.assert_not_called()
        mock_transcribe.assert_not_called()
        assert not rec_file.exists()
        assert len(list(month_dir.glob(f"*-{take.take_id}.wav"))) == 1
        assert len(list(month_dir.glob(f"*-{take.take_id}.txt"))) == 1

    def test_partial_archive_leftovers_are_replaced_by_the_live_wav(self, tmp_path):
        """ "Died during the archive" usually means save_audio had already
        created the .wav copy (partial or complete) and maybe the empty
        compressed reservation. The exclusive archive must not collide with
        them: they belong to this very take and the live WAV is complete."""
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"complete audio")
        take = self.make_recovering_take(tmp_path, rec_file=rec_file)
        config = self.make_config(tmp_path)
        config["dictate"]["audio_format"] = "wav"
        month_dir = Path(config["dictate"]["audio_dir"]) / "2026" / "09"
        month_dir.mkdir(parents=True)
        stem = f"20260905-101500-{take.take_id}"
        (month_dir / f"{stem}.txt").write_text("already pasted\n")
        (month_dir / f"{stem}.wav").write_bytes(b"part")
        (month_dir / f"{stem}.flac").write_bytes(b"")
        (month_dir / f".{stem}.wav.4242.tmp").write_bytes(b"pa")

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.delivery.send_text") as mock_send,
            patch("digue.transcribe.transcribe") as mock_transcribe,
            patch("digue.notify.send_notification"),
        ):
            exit_code = recording_mod._recover_claimed_take(config, take)

        assert exit_code == 0
        mock_send.assert_not_called()
        mock_transcribe.assert_not_called()
        assert not rec_file.exists()
        assert (month_dir / f"{stem}.wav").read_bytes() == b"complete audio"
        assert (month_dir / f"{stem}.txt").read_text() == "already pasted\n"
        assert sorted(path.name for path in month_dir.iterdir()) == [f"{stem}.txt", f"{stem}.wav"]
        assert list(tmp_path.glob("digue-take-*.json")) == []

    def test_delivered_take_without_live_audio_is_just_forgotten(self, tmp_path, capsys):
        """The daemon pasted, saved the .txt, archived the audio, removed the
        live file and died before dropping its state: the take is complete.
        Reporting "Empty or missing audio file" (exit 1) here was wrong."""
        rec_file = tmp_path / "digue-recording.wav"  # already unlinked by the archive
        take = self.make_recovering_take(tmp_path, rec_file=rec_file)
        config = self.make_config(tmp_path)
        month_dir = Path(config["dictate"]["audio_dir"]) / "2026" / "09"
        month_dir.mkdir(parents=True)
        stem = f"20260905-101500-{take.take_id}"
        (month_dir / f"{stem}.txt").write_text("already pasted\n")
        (month_dir / f"{stem}.flac").write_bytes(b"fLaC")

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.delivery.send_text") as mock_send,
            patch("digue.transcribe.transcribe") as mock_transcribe,
            patch("digue.notify.send_notification") as mock_notify,
        ):
            exit_code = recording_mod._recover_claimed_take(config, take)

        assert exit_code == 0
        mock_send.assert_not_called()
        mock_transcribe.assert_not_called()
        mock_notify.assert_not_called()
        assert sorted(path.name for path in month_dir.iterdir()) == [f"{stem}.flac", f"{stem}.txt"]
        assert list(tmp_path.glob("digue-take-*.json")) == []
        assert "already delivered" in capsys.readouterr().err

    def test_partial_archive_leftovers_do_not_touch_other_takes(self, tmp_path):
        """Only the products of this take's stem are dropped: a neighbouring
        take that shares the timestamp keeps its files."""
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"complete audio")
        take = self.make_recovering_take(tmp_path, rec_file=rec_file)
        config = self.make_config(tmp_path)
        config["dictate"]["audio_format"] = "wav"
        month_dir = Path(config["dictate"]["audio_dir"]) / "2026" / "09"
        month_dir.mkdir(parents=True)
        stem = f"20260905-101500-{take.take_id}"
        other_stem = "20260905-101500-fedcba9876543210"
        (month_dir / f"{stem}.txt").write_text("already pasted\n")
        (month_dir / f"{stem}.wav").write_bytes(b"part")
        (month_dir / f"{other_stem}.wav").write_bytes(b"other audio")
        (month_dir / f"{other_stem}.txt").write_text("other text\n")

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.delivery.send_text"),
            patch("digue.transcribe.transcribe"),
            patch("digue.notify.send_notification"),
        ):
            exit_code = recording_mod._recover_claimed_take(config, take)

        assert exit_code == 0
        assert (month_dir / f"{stem}.wav").read_bytes() == b"complete audio"
        assert (month_dir / f"{other_stem}.wav").read_bytes() == b"other audio"
        assert (month_dir / f"{other_stem}.txt").read_text() == "other text\n"
        assert list(tmp_path.glob("digue-take-*.json")) == []

    def test_recycled_recorder_pid_is_not_signaled(self, tmp_path):
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        take = self.make_recovering_take(tmp_path, rec_file=rec_file, recorder_pid=os.getpid(), recorder_starttime=111)
        delivered = []

        def fake_finish(_config, file, limit_reached=False, take_id=None):
            delivered.append(file)
            return dictate_mod.DeliveryResult(outcome="delivered", exit_code=0)

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._process_starttime", return_value="999"),
            patch("digue.recording.os.killpg") as mock_killpg,
            patch("digue.dictate.finish_dictation", side_effect=fake_finish),
        ):
            exit_code = recording_mod._recover_claimed_take(self.make_config(tmp_path), take)

        assert exit_code == 0
        mock_killpg.assert_not_called()
        assert delivered == [take.rec_file]
        assert list(tmp_path.glob("digue-take-*.json")) == []

    def test_retryable_failure_preserves_state_and_wav(self, tmp_path):
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        take = self.make_recovering_take(tmp_path, rec_file=rec_file)

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch(
                "digue.dictate.finish_dictation",
                return_value=dictate_mod.DeliveryResult(outcome="retryable_failure", exit_code=1),
            ),
        ):
            exit_code = recording_mod._recover_claimed_take(self.make_config(tmp_path), take)

        assert exit_code == 1
        assert rec_file.exists()
        assert list(tmp_path.glob("digue-take-*.json")) == [tmp_path / f"digue-take-{take.take_id}.json"]

    def test_unexpected_exception_preserves_state_and_wav(self, tmp_path):
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        take = self.make_recovering_take(tmp_path, rec_file=rec_file)

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.dictate.finish_dictation", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError),
        ):
            recording_mod._recover_claimed_take(self.make_config(tmp_path), take)

        assert rec_file.exists()
        assert list(tmp_path.glob("digue-take-*.json")) == [tmp_path / f"digue-take-{take.take_id}.json"]

    def test_empty_wav_is_terminal_regardless_of_age(self, tmp_path):
        """Regression: the "empty is transient while young" guard kept the
        state of a take whose WAV was already gone (stop_recording_pid unlinks
        an empty WAV; the no-speech path archives it), so every toggle in the
        next minute reclaimed it, reported "Empty or missing audio file" and
        blocked the surplus rescue. After stop_recording_pid the recorder is
        dead, so an empty WAV is final. Uses the real finish_dictation."""
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"")
        take = self.make_recovering_take(tmp_path, age_seconds=1.0, rec_file=rec_file)

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.notify.send_notification") as mock_notify,
        ):
            exit_code = recording_mod._recover_claimed_take(self.make_config(tmp_path), take)

        assert exit_code == 1
        assert not rec_file.exists()
        assert list(tmp_path.glob("digue-take-*.json")) == []
        assert mock_notify.call_args.args[0] == "Empty or missing audio file"

    def test_no_speech_take_is_consumed_on_the_first_toggle(self, tmp_path):
        """Same regression through the no-speech path (WAV archived, text
        empty): the state must go with the audio, not survive for a minute."""
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        take = self.make_recovering_take(tmp_path, age_seconds=1.0, rec_file=rec_file)

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.transcribe.transcribe", return_value=""),
            patch("digue.notify.send_notification"),
        ):
            exit_code = recording_mod._recover_claimed_take(self.make_config(tmp_path), take)

        assert exit_code == 0
        assert not rec_file.exists()
        assert list(tmp_path.glob("digue-take-*.json")) == []


class TestSurplusOrphanRescue:
    """After the oldest orphan take was delivered, the remaining orphans are
    claimed one by one (each claim under the dictate lock) and rescued: the
    audio is kept via rescue_recording (unique name per take id) with the
    take's metadata JSON moved next to it (state "rescued"), and one
    consolidated notification is shown. Only the oldest take is ever
    transcribed and pasted."""

    def make_take(self, tmp_path, take_id, created_at_ns, **changes):
        values = {
            "version": recording_mod.TAKE_STATE_VERSION,
            "take_id": take_id,
            "created_at_ns": created_at_ns,
            "state": "recording",
            "rec_file": tmp_path / f"digue-{take_id}.wav",
            "daemon_pid": 999999,
            "daemon_starttime": 1,
            "recorder_pid": 555,
            "recorder_starttime": 666,
        }
        values.update(changes)
        with patch("digue.recording._runtime_dir", return_value=tmp_path):
            take = recording_mod.TakeState(**values)
            recording_mod._write_take_state(take)
        return take

    def test_toggle_delivers_the_oldest_and_rescues_the_rest(self, tmp_path):
        oldest_wav = tmp_path / "digue-oldest.wav"
        oldest_wav.write_bytes(b"audio oldest")
        surplus_a = tmp_path / "digue-surplus-a.wav"
        surplus_a.write_bytes(b"audio a")
        surplus_b = tmp_path / "digue-surplus-b.wav"
        surplus_b.write_bytes(b"audio b")
        oldest = self.make_take(tmp_path, "0123456789abcdef", 100, rec_file=oldest_wav)
        surplus_take_a = self.make_take(tmp_path, "aaaaaaaaaaaaaaaa", 200, rec_file=surplus_a)
        surplus_take_b = self.make_take(tmp_path, "bbbbbbbbbbbbbbbb", 300, rec_file=surplus_b)
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        config["dictate"]["max_duration"] = 0
        recorder = MagicMock(pid=os.getpid(), poll=lambda: 0)
        finish_take_ids = []

        def fake_finish(_config, rec_file, limit_reached=False, take_id=None):
            finish_take_ids.append(take_id)
            return dictate_mod.DeliveryResult(outcome="delivered", exit_code=0)

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._pid_alive", lambda pid: pid == os.getpid()),
            patch("digue.container.ensure_server"),
            patch("digue.container.is_server_running", return_value=True),
            patch(
                "digue.recording.stop_recording_pid",
                side_effect=lambda pid, rec_file, expected_starttime=None: rec_file,
            ),
            patch("digue.dictate.finish_dictation", side_effect=fake_finish),
            patch("subprocess.Popen", return_value=recorder),
            patch("digue.recording._wait_recorder_end_daemon", return_value="ended"),
            patch("digue.notify.send_notification") as mock_notify,
            patch("digue.notify.notify_close"),
            patch("signal.signal"),
        ):
            assert dictate_mod.dictate_toggle(config) == 0

        # only the oldest orphan is transcribed and pasted
        assert finish_take_ids == [oldest.take_id]
        assert surplus_take_a.take_id not in finish_take_ids
        assert surplus_take_b.take_id not in finish_take_ids
        month = tmp_path / "audio" / audio_mod.month_dir_for(audio_mod.now_timestamp())
        for take, original_bytes in (
            (surplus_take_a, b"audio a"),
            (surplus_take_b, b"audio b"),
        ):
            rescued = list(month.glob(f"*-{take.take_id}.wav"))
            assert len(rescued) == 1 and rescued[0].read_bytes() == original_bytes
            assert not take.rec_file.exists()
            metadata_path = rescued[0].with_suffix(".json")
            metadata = json.loads(metadata_path.read_text())
            assert metadata["state"] == "rescued"
            assert metadata["take_id"] == take.take_id
        assert list(tmp_path.glob("digue-take-*.json")) == []
        rescued_notifies = [
            recorded_call for recorded_call in mock_notify.call_args_list if "rescued to" in str(recorded_call.args)
        ]
        assert len(rescued_notifies) == 1
        message = rescued_notifies[0].args[0]
        assert "2 recordings rescued to" in message
        assert str(tmp_path / "audio") in message

    def test_surplus_take_with_live_recorder_is_stopped_before_rescue(self, tmp_path):
        oldest_wav = tmp_path / "digue-oldest.wav"
        oldest_wav.write_bytes(b"audio oldest")
        surplus_wav = tmp_path / "digue-surplus.wav"
        surplus_wav.write_bytes(b"audio surplus")
        self.make_take(tmp_path, "0123456789abcdef", 100, rec_file=oldest_wav)
        surplus_recorder = subprocess.Popen(["sleep", "60"], start_new_session=True)
        try:
            self.make_take(
                tmp_path,
                "aaaaaaaaaaaaaaaa",
                200,
                rec_file=surplus_wav,
                recorder_pid=surplus_recorder.pid,
                recorder_starttime=int(recording_mod._process_starttime(surplus_recorder.pid)),
            )
            config = _default_config()
            config["dictate"]["audio_dir"] = str(tmp_path / "audio")
            config["dictate"]["max_duration"] = 0
            recorder = MagicMock(pid=os.getpid(), poll=lambda: 0)

            with (
                patch("digue.recording._runtime_dir", return_value=tmp_path),
                patch("digue.recording._pid_alive", lambda pid: pid == os.getpid()),
                patch("digue.container.ensure_server"),
                patch("digue.container.is_server_running", return_value=True),
                patch(
                    "digue.dictate.finish_dictation",
                    return_value=dictate_mod.DeliveryResult(outcome="delivered", exit_code=0),
                ),
                patch("subprocess.Popen", return_value=recorder),
                patch("digue.recording._wait_recorder_end_daemon", return_value="ended"),
                patch("digue.notify.send_notification"),
                patch("digue.notify.notify_close"),
                patch("signal.signal"),
            ):
                assert dictate_mod.dictate_toggle(config) == 0

            assert surplus_recorder.poll() is not None
            month = tmp_path / "audio" / audio_mod.month_dir_for(audio_mod.now_timestamp())
            rescued = list(month.glob("*-aaaaaaaaaaaaaaaa.wav"))
            assert len(rescued) == 1 and rescued[0].read_bytes() == b"audio surplus"
            assert list(month.glob("*-aaaaaaaaaaaaaaaa.json"))
        finally:
            surplus_recorder.kill()
            surplus_recorder.wait(timeout=5)

    def test_each_surplus_claim_happens_under_the_dictate_lock(self, tmp_path):
        """_claim_orphan_take requires the lock; the surplus loop runs outside
        the toggle's lock, so it must take it around every claim, or two
        toggles could claim the same surplus take."""

        lock_depth = 0
        claims_under_lock = []

        @contextlib.contextmanager
        def recording_lock():
            nonlocal lock_depth
            lock_depth += 1
            try:
                yield
            finally:
                lock_depth -= 1

        takes = iter([MagicMock(take_id="aaaaaaaaaaaaaaaa"), None])

        def fake_claim():
            claims_under_lock.append(lock_depth > 0)
            return next(takes)

        with (
            patch("digue.dictate._dictate_lock", recording_lock),
            patch("digue.recording._claim_orphan_take", side_effect=fake_claim),
            patch("digue.recording._rescue_surplus_take", return_value=tmp_path / "rescued.wav"),
            patch("digue.recording._take_state_file", return_value=tmp_path / "gone.json"),
        ):
            rescued = recording_mod._rescue_surplus_orphans(_default_config())

        assert rescued == [tmp_path / "rescued.wav"]
        assert claims_under_lock == [True, True]

    def test_surplus_rescue_failure_preserves_state_and_wav(self, tmp_path):
        oldest_wav = tmp_path / "digue-oldest.wav"
        oldest_wav.write_bytes(b"audio oldest")
        surplus_wav = tmp_path / "digue-surplus.wav"
        surplus_wav.write_bytes(b"audio surplus")
        self.make_take(tmp_path, "0123456789abcdef", 100, rec_file=oldest_wav)
        surplus_take = self.make_take(tmp_path, "aaaaaaaaaaaaaaaa", 200, rec_file=surplus_wav)
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        config["dictate"]["max_duration"] = 0
        recorder = MagicMock(pid=os.getpid(), poll=lambda: 0)

        with (
            patch("digue.recording._runtime_dir", return_value=tmp_path),
            patch("digue.recording._pid_alive", lambda pid: pid == os.getpid()),
            patch("digue.container.ensure_server"),
            patch("digue.container.is_server_running", return_value=True),
            patch(
                "digue.dictate.finish_dictation",
                return_value=dictate_mod.DeliveryResult(outcome="delivered", exit_code=0),
            ),
            patch("digue.audio.rescue_recording", return_value=None),
            patch("subprocess.Popen", return_value=recorder),
            patch("digue.recording._wait_recorder_end_daemon", return_value="ended"),
            patch("digue.notify.send_notification"),
            patch("digue.notify.notify_close"),
            patch("signal.signal"),
        ):
            assert dictate_mod.dictate_toggle(config) == 0

        assert surplus_wav.exists()
        assert (tmp_path / f"digue-take-{surplus_take.take_id}.json").exists()
