"""Tests for take state, recorder start/stop, watchdog, and runtime isolation."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

import digue
from digue.config import _default_config

# -- Recording ----------------------------------------------------------------


class TestTakeState:
    def make_state(self, tmp_path, **changes):
        values = {
            "version": digue.TAKE_STATE_VERSION,
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
        with patch("digue._runtime_dir", return_value=tmp_path):
            return digue.TakeState(**values)

    def test_round_trip_and_ordering(self, tmp_path):
        newer = self.make_state(tmp_path, take_id="ffffffffffffffff", created_at_ns=124)
        older = self.make_state(tmp_path)
        with patch("digue._runtime_dir", return_value=tmp_path):
            digue._write_take_state(newer)
            digue._write_take_state(older)
            assert digue._read_take_state(tmp_path / "digue-take-0123456789abcdef.json") == older
            assert digue._take_states() == [older, newer]

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

        with patch("digue._runtime_dir", return_value=tmp_path):
            assert digue._take_states() == []

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
        with patch("digue._runtime_dir", return_value=tmp_path):
            digue._write_daemon_state(os.getpid(), "recording")

        assert daemon_file not in opened
        assert daemon_file.read_text().split()[:2] == [str(os.getpid()), "recording"]

    @patch("digue._spawn_limit_watchdog")
    @patch("subprocess.Popen")
    def test_take_state_is_never_truncated_in_place(self, mock_popen, mock_watchdog, tmp_path, monkeypatch):
        mock_popen.return_value = MagicMock(pid=os.getpid())
        opened = self._truncating_writes(monkeypatch)
        with patch("digue._runtime_dir", return_value=tmp_path):
            processes = digue.start_recording(_default_config())

        state_file = tmp_path / f"digue-take-{processes.take_id}.json"
        assert state_file not in opened
        assert json.loads(state_file.read_text())["state"] == "recording"


class TestRecordingCommand:
    @patch("shutil.which")
    def test_auto_prefers_pw_record(self, mock_which, tmp_path):
        mock_which.return_value = "/usr/bin/pw-record"
        argv = digue.recording_command(tmp_path / "rec.wav", recorder="auto")
        assert argv[0] == "pw-record"

    @patch("shutil.which")
    def test_auto_falls_back_to_arecord(self, mock_which, tmp_path):
        mock_which.side_effect = lambda name: None if name == "pw-record" else "/usr/bin/arecord"
        argv = digue.recording_command(tmp_path / "rec.wav", recorder="auto")
        assert argv[0] == "arecord"

    def test_pw_record_argv(self, tmp_path):
        argv = digue.recording_command(tmp_path / "rec.wav", recorder="pw-record")
        assert argv[0] == "pw-record"
        assert "--rate" in argv and "16000" in argv
        assert argv[-1].endswith("rec.wav")

    def test_arecord_argv(self, tmp_path):
        argv = digue.recording_command(tmp_path / "rec.wav", recorder="arecord")
        assert argv[0] == "arecord"
        assert "16000" in argv

    def test_unknown_recorder_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="Unknown recorder"):
            digue.recording_command(tmp_path / "rec.wav", recorder="sox")

    def test_pw_record_omits_target_when_device_is_empty(self, tmp_path):
        argv = digue.recording_command(tmp_path / "rec.wav", recorder="pw-record", device="")
        assert "--target" not in argv

    def test_pw_record_passes_target(self, tmp_path):
        argv = digue.recording_command(tmp_path / "rec.wav", recorder="pw-record", device="alsa_input.usb")
        assert argv[argv.index("--target") + 1] == "alsa_input.usb"

    def test_arecord_passes_device(self, tmp_path):
        argv = digue.recording_command(tmp_path / "rec.wav", recorder="arecord", device="hw:2,0")
        assert argv[argv.index("-D") + 1] == "hw:2,0"

    def test_arecord_omits_device_when_empty(self, tmp_path):
        argv = digue.recording_command(tmp_path / "rec.wav", recorder="arecord", device="")
        assert "-D" not in argv

    def test_pw_record_adds_container_for_flac_suffix(self, tmp_path):
        argv = digue.recording_command(tmp_path / "rec.flac", recorder="pw-record")
        assert argv[argv.index("--container") + 1] == "flac"

    def test_pw_record_wav_has_no_container_flag(self, tmp_path):
        argv = digue.recording_command(tmp_path / "rec.wav", recorder="pw-record")
        assert "--container" not in argv


class TestLiveRecordingFormat:
    def test_wav_audio_format_always_records_wav(self):
        config = _default_config()
        config["dictate"]["audio_format"] = "wav"
        with patch("digue._pw_record_supports_flac", return_value=True):
            assert digue._live_recording_suffix(config) == ".wav"

    def test_flac_with_pw_record_support_records_flac(self):
        config = _default_config()
        config["dictate"]["audio_format"] = "flac"
        config["dictate"]["recorder"] = "pw-record"
        with patch("digue._pw_record_supports_flac", return_value=True):
            assert digue._live_recording_suffix(config) == ".flac"

    def test_flac_without_container_falls_back_to_wav(self):
        config = _default_config()
        config["dictate"]["audio_format"] = "flac"
        config["dictate"]["recorder"] = "pw-record"
        with patch("digue._pw_record_supports_flac", return_value=False):
            assert digue._live_recording_suffix(config) == ".wav"

    def test_arecord_never_records_flac(self):
        config = _default_config()
        config["dictate"]["audio_format"] = "flac"
        config["dictate"]["recorder"] = "arecord"
        with patch("digue._pw_record_supports_flac", return_value=True):
            assert digue._live_recording_suffix(config) == ".wav"

    def test_supports_flac_parses_list_containers(self, tmp_path):
        binary = tmp_path / "pw-record"
        binary.write_bytes(b"")
        cache = tmp_path / "cache"
        cache.mkdir()
        listed = MagicMock(stdout="    wav: WAV (Microsoft)\n    flac: FLAC (Free Lossless Audio Codec)\n")
        with (
            patch("digue._cache_dir", return_value=cache),
            patch("shutil.which", return_value=str(binary)),
            patch("subprocess.run", return_value=listed),
        ):
            assert digue._pw_record_supports_flac() is True
        assert "flac" in (cache / "pw-record-containers").read_text()

    def test_supports_flac_false_when_missing(self, tmp_path):
        binary = tmp_path / "pw-record"
        binary.write_bytes(b"")
        cache = tmp_path / "cache"
        cache.mkdir()
        listed = MagicMock(stdout="    wav: WAV (Microsoft)\n")
        with (
            patch("digue._cache_dir", return_value=cache),
            patch("shutil.which", return_value=str(binary)),
            patch("subprocess.run", return_value=listed),
        ):
            assert digue._pw_record_supports_flac() is False


class TestStartRecording:
    @patch("subprocess.Popen")
    def test_returns_owned_recorder_handle(self, mock_popen, tmp_path):
        recorder = MagicMock(pid=1234)
        mock_popen.return_value = recorder
        config = _default_config()
        config["dictate"]["max_duration"] = 0

        with patch("digue._runtime_dir", return_value=tmp_path):
            processes = digue.start_recording(config)

        assert processes.recorder is recorder
        assert processes.watchdog is None
        assert processes.rec_file is not None
        assert processes.rec_file.name.startswith("digue-")
        assert processes.rec_file.suffix in {".wav", ".flac"}
        assert mock_popen.call_args[1].get("start_new_session") is True

    @patch("digue._spawn_limit_watchdog")
    @patch("subprocess.Popen")
    def test_max_duration_returns_watchdog_handle(self, mock_popen, mock_watchdog, tmp_path):
        recorder = MagicMock(pid=os.getpid())
        watchdog = MagicMock(pid=778)
        mock_popen.return_value = recorder
        mock_watchdog.return_value = watchdog
        config = _default_config()
        config["dictate"]["max_duration"] = 300

        with patch("digue._runtime_dir", return_value=tmp_path):
            processes = digue.start_recording(config)

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
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._live_recording_suffix", return_value=".wav"),
            patch("subprocess.Popen", return_value=recorder) as mock_popen,
        ):
            digue.start_recording(config)

        argv = mock_popen.call_args[0][0]
        assert argv[argv.index("--target") + 1] == "alsa_input.usb"


class TestStartRecordingPublishesTakeState:
    """The take identity is published before the recorder exists and gains the
    recorder identity before the watchdog is spawned: publishing the state is
    two syscalls (~50 us) while spawning the watchdog is fork+exec (~10 ms), so
    a recorder with identity (which recovery knows how to stop) is the state
    that is exposed the soonest."""

    @patch("digue._spawn_limit_watchdog")
    @patch("subprocess.Popen")
    def test_publishes_starting_before_popen(self, mock_popen, mock_watchdog, tmp_path):
        states_at_popen = []
        recorder = MagicMock(pid=os.getpid())

        def fake_popen(*args, **kwargs):
            states_at_popen.extend(digue._take_states())
            return recorder

        mock_popen.side_effect = fake_popen
        config = _default_config()
        config["dictate"]["max_duration"] = 0

        with patch("digue._runtime_dir", return_value=tmp_path):
            processes = digue.start_recording(config)

        assert [take.state for take in states_at_popen] == ["starting"]
        assert processes.take_id == states_at_popen[0].take_id
        assert processes.rec_file == states_at_popen[0].rec_file

    @patch("subprocess.Popen")
    def test_publishes_recording_with_identity_before_watchdog(self, mock_popen, tmp_path):
        states_at_watchdog = []
        recorder = MagicMock(pid=os.getpid())

        def fake_watchdog(pgid, max_duration):
            states_at_watchdog.extend(digue._take_states())
            return MagicMock(pid=9999)

        mock_popen.return_value = recorder
        config = _default_config()
        config["dictate"]["max_duration"] = 300

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._spawn_limit_watchdog", side_effect=fake_watchdog),
        ):
            processes = digue.start_recording(config)

        assert processes.take_id is not None
        assert [take.state for take in states_at_watchdog] == ["recording"]
        take = states_at_watchdog[0]
        assert take.recorder_pid == recorder.pid
        assert take.recorder_starttime == int(digue._process_starttime(os.getpid()))

    @patch("subprocess.Popen", side_effect=FileNotFoundError("pw-record"))
    def test_popen_failure_removes_state(self, mock_popen, tmp_path):
        config = _default_config()

        with patch("digue._runtime_dir", return_value=tmp_path), pytest.raises(FileNotFoundError):
            digue.start_recording(config)

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
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._live_recording_suffix", return_value=".wav"),
            patch("subprocess.Popen", side_effect=fake_popen),
            pytest.raises(RuntimeError, match="pw-record failed: no such node"),
        ):
            digue.start_recording(config)

        assert list(tmp_path.glob("digue-take-*.json")) == []

    def test_toggle_publishes_take_state_and_removes_it_after_delivery(self, tmp_path):
        config = _default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        config["dictate"]["max_duration"] = 0
        recorder = MagicMock(pid=os.getpid(), poll=lambda: 0)
        finish_take_ids = []

        def fake_finish(_config, rec_file, limit_reached=False, take_id=None):
            finish_take_ids.append(take_id)
            assert [take.state for take in digue._take_states()] == ["recording"]
            return digue.DeliveryResult(outcome="delivered", exit_code=0)

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.container.ensure_server"),
            patch("digue.container.is_server_running", return_value=True),
            patch("subprocess.Popen", return_value=recorder),
            patch("digue._wait_recorder_end_daemon", return_value="ended"),
            patch("digue.finish_dictation", side_effect=fake_finish),
            patch("digue.notify.send_notification"),
            patch("digue.notify.notify_close"),
            patch("signal.signal"),
        ):
            assert digue.dictate_toggle(config) == 0

        assert len(finish_take_ids) == 1 and finish_take_ids[0] is not None
        assert list(tmp_path.glob("digue-take-*.json")) == []


class TestSpawnLimitWatchdog:
    @patch("digue._process_starttime", return_value="98765")
    @patch("subprocess.Popen")
    def test_watchdog_validates_process_identity_before_killing(self, mock_popen, mock_starttime):
        watchdog = MagicMock(pid=5000)
        mock_popen.return_value = watchdog

        result = digue._spawn_limit_watchdog(4242, 300)

        argv = mock_popen.call_args[0][0]
        script = argv[2]
        assert argv[-3:] == ["4242", "98765", str(300 + digue.WATCHDOG_GRACE_SECONDS)]
        assert "/proc/{pid}/stat" in script
        assert "os.killpg(pid, signal.SIGTERM)" in script
        assert result is watchdog
        assert mock_popen.call_args[1].get("start_new_session") is True

    @patch("digue._process_starttime", return_value="98765")
    @patch("subprocess.Popen")
    def test_watchdog_fires_only_after_the_daemon_limit(self, mock_popen, mock_starttime):
        """The daemon polls every 200ms and drifts; a watchdog sleeping exactly
        max_duration wins the race (measured: at 20s the recorder was already
        dead when the daemon checked), so the daemon saw "died" and never
        notified the limit. The watchdog is a safety net and must fire later."""
        digue._spawn_limit_watchdog(4242, 20)

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
                digue._spawn_limit_watchdog(leader.pid, 0)
                script = mock_popen.call_args[0][0][2]
            for process in (leader, member):
                starttime = digue._process_starttime(process.pid)
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

        assert digue._process_starttime(4242, stat_path=stat) == "22"


class TestWaitRecorderEndDaemon:
    def test_polls_owned_process_and_collects_spontaneous_exit(self):
        recorder = MagicMock()
        recorder.poll.side_effect = [None, 7]

        with patch("time.monotonic", side_effect=[0.0, 0.1, 0.3]), patch("time.sleep"):
            outcome = digue._wait_recorder_end_daemon(recorder, 300)

        assert outcome == "died"
        recorder.wait.assert_called_once_with(timeout=0)

    def test_sigterm_returns_manual(self):
        recorder = MagicMock()
        recorder.poll.return_value = None
        with patch("time.monotonic", side_effect=[0.0, 0.5]), patch("time.sleep"):
            digue._got_sigterm = True
            try:
                assert digue._wait_recorder_end_daemon(recorder, 300) == "manual"
            finally:
                digue._got_sigterm = False

    def test_exit_after_the_limit_counts_as_limit(self):
        """If the watchdog killed the recorder first, the outcome is still the
        duration limit, not a spontaneous death (the notification depends on it)."""
        recorder = MagicMock()
        recorder.poll.side_effect = [None, -15]

        with patch("time.monotonic", side_effect=[0.0, 0.1, 300.2]), patch("time.sleep"):
            outcome = digue._wait_recorder_end_daemon(recorder, 300)

        assert outcome == "limit"


class TestFinishOwnedRecorder:
    def test_spontaneously_exited_recorder_is_not_signaled_again(self, tmp_path):
        rec_file = tmp_path / "take.wav"
        rec_file.write_bytes(b"audio")
        recorder = MagicMock(pid=777)
        recorder.poll.return_value = 1

        with patch("digue.stop_recording_pid") as mock_stop:
            result = digue._finish_owned_recorder(recorder, rec_file)

        assert result == rec_file
        mock_stop.assert_not_called()


class TestCancelWatchdog:
    def test_normal_stop_terminates_and_collects_watchdog(self):
        watchdog = MagicMock()
        watchdog.poll.return_value = None

        digue._cancel_watchdog(watchdog)

        watchdog.terminate.assert_called_once_with()
        watchdog.wait.assert_called_once_with(timeout=5)

    def test_no_watchdog_is_a_noop(self):
        digue._cancel_watchdog(None)


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

        with patch("digue._runtime_dir", return_value=tmp_path):
            assert digue.stop_recording_pid(999999, expected_starttime=None) is None

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

    @patch("digue._group_alive", return_value=False)
    @patch("os.killpg")
    @patch("digue._process_starttime", return_value="999")
    def test_starttime_divergence_skips_killpg(self, mock_starttime, mock_killpg, mock_alive, tmp_path):
        rec_file = self.make_rec_file(tmp_path)

        result = digue.stop_recording_pid(4242, rec_file, expected_starttime="111")

        mock_killpg.assert_not_called()
        assert result == rec_file

    @patch("digue._group_alive", return_value=False)
    @patch("os.killpg")
    @patch("digue._process_pgrp", return_value=5151)
    @patch("digue._process_starttime", return_value="111")
    def test_pgrp_mismatch_skips_killpg(self, mock_starttime, mock_pgrp, mock_killpg, mock_alive, tmp_path):
        rec_file = self.make_rec_file(tmp_path)

        result = digue.stop_recording_pid(4242, rec_file, expected_starttime="111")

        mock_killpg.assert_not_called()
        assert result == rec_file

    @patch("digue._group_alive", return_value=False)
    @patch("os.killpg")
    @patch("digue._process_pgrp", return_value=4242)
    @patch("digue._process_starttime", return_value="111")
    def test_matching_identity_signals_the_group(self, mock_starttime, mock_pgrp, mock_killpg, mock_alive, tmp_path):
        rec_file = self.make_rec_file(tmp_path)

        result = digue.stop_recording_pid(4242, rec_file, expected_starttime="111")

        mock_killpg.assert_called_once_with(4242, 15)
        assert result == rec_file

    @patch("digue._group_alive", return_value=True)
    @patch("os.killpg")
    @patch("digue._process_pgrp", return_value=4242)
    @patch("digue._process_starttime", side_effect=["111", "999"])
    def test_identity_is_rechecked_before_sigkill(self, mock_starttime, mock_pgrp, mock_killpg, mock_alive, tmp_path):
        rec_file = self.make_rec_file(tmp_path)

        with patch("time.monotonic", side_effect=[0.0, 0.1, 0.6]):
            result = digue.stop_recording_pid(4242, rec_file, expected_starttime="111")

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
            result = digue.stop_recording_pid(recorder.pid, rec_file)
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
        with wav.open("wb") as open_wav, patch("digue._runtime_dir", return_value=link_dir):
            open_wav.write(b"x")
            found = digue._recording_file_of(os.getpid())

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
            patch("digue._pid_alive", return_value=True),
            patch("digue._process_starttime", return_value="555"),
            patch("digue._process_is_zombie", return_value=True),
        ):
            assert digue._daemon_alive((4242, "recording", "555")) is False
        with (
            patch("digue._pid_alive", return_value=True),
            patch("digue._process_starttime", return_value="555"),
            patch("digue._process_is_zombie", return_value=False),
        ):
            assert digue._daemon_alive((4242, "recording", "555")) is True


class TestRuntimeIsolation:
    def test_state_paths_use_isolated_runtime_dir(self):
        runtime_dir = Path(os.environ["XDG_RUNTIME_DIR"])

        assert digue._runtime_dir() == runtime_dir
        assert digue._daemon_pid_file().parent == runtime_dir
        assert digue._take_state_file("0123456789abcdef").parent == runtime_dir
        with digue._dictate_lock():
            assert (runtime_dir / "digue.lock").exists()

    def test_toggle_does_not_read_state_outside_isolated_runtime(self, tmp_path):
        """A stale daemon state outside the isolated runtime must not be signaled."""
        sentinel_dir = tmp_path / "sentinel"
        sentinel_dir.mkdir()
        (sentinel_dir / "digue-daemon.pid").write_text("4242 recording 555")
        config = _default_config()

        with (
            patch("digue._process_starttime", return_value="555"),
            patch("digue.container.ensure_server", side_effect=RuntimeError("stop after state lookup")),
            patch("digue.notify.send_notification"),
            patch("os.kill") as mock_kill,
        ):
            assert digue.dictate_toggle(config) == 1

        mock_kill.assert_not_called()
