"""Tests for the dictate daemon, toggle, and interrupt handling."""

import contextlib
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import digue


class TestDictateDaemon:
    def test_concurrent_first_toggles_start_only_one_recorder(self, tmp_path):
        """A second toggle must observe the first toggle's startup reservation."""
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        first_in_startup = threading.Event()
        release_first = threading.Event()
        ensure_calls = 0
        results = []
        errors = []

        def ensure_server(_config):
            nonlocal ensure_calls
            ensure_calls += 1
            if ensure_calls == 1:
                first_in_startup.set()
                assert release_first.wait(timeout=2)

        def run_toggle():
            try:
                results.append(digue.dictate_toggle(config))
            except BaseException as exc:
                errors.append(exc)

        recorder = MagicMock(pid=777, poll=lambda: 0)
        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.ensure_server", side_effect=ensure_server),
            patch("digue.is_server_running", return_value=True),
            patch(
                "digue.start_recording",
                return_value=digue.RecordingProcesses(recorder=recorder, watchdog=None),
            ) as mock_start,
            patch("digue._recording_file_of", return_value=tmp_path / "take.wav"),
            patch("digue._wait_recorder_end_daemon", return_value="ended"),
            patch("digue.finish_dictation", return_value=digue.DeliveryResult(outcome="delivered", exit_code=0)),
            patch("signal.signal"),
            patch("os.kill") as mock_kill,
        ):
            first = threading.Thread(target=run_toggle)
            first.start()
            assert first_in_startup.wait(timeout=2)
            second = threading.Thread(target=run_toggle)
            second.start()
            second.join(timeout=2)
            assert not second.is_alive()
            release_first.set()
            first.join(timeout=2)
            assert not first.is_alive()

        assert errors == []
        assert results == [0, 0]
        mock_start.assert_called_once_with(config)
        assert all(call.args[1] == 0 for call in mock_kill.call_args_list)

    def test_dictate_toggle_works_when_stderr_has_no_isatty(self, tmp_path, monkeypatch):
        class NonFileStderr:
            def write(self, _message):
                pass

            def flush(self):
                pass

        monkeypatch.setattr(sys, "stderr", NonFileStderr())
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        recorder = MagicMock(pid=777, poll=lambda: 0)
        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.ensure_server"),
            patch("digue.is_server_running", return_value=True),
            patch("digue.start_recording", return_value=digue.RecordingProcesses(recorder=recorder, watchdog=None)),
            patch("digue._recording_file_of", return_value=tmp_path / "take.wav"),
            patch("digue._wait_recorder_end_daemon", return_value="ended"),
            patch("digue.finish_dictation", return_value=digue.DeliveryResult(outcome="delivered", exit_code=0)),
            patch("digue.notify"),
            patch("signal.signal"),
        ):
            assert digue.dictate_toggle(config) == 0

    def run_daemon_delivery(self, tmp_path, finish):
        """Runs the daemon path with a published take state and a mocked
        delivery; returns (exit_code, take state file)."""
        import os

        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        recorder = MagicMock(pid=777, poll=lambda: 0)
        with patch("digue._runtime_dir", return_value=tmp_path):
            take = digue.TakeState(
                version=digue.TAKE_STATE_VERSION,
                take_id="0123456789abcdef",
                created_at_ns=100,
                state="recording",
                rec_file=rec_file,
                daemon_pid=os.getpid(),
                daemon_starttime=int(digue._process_starttime(os.getpid())),
                recorder_pid=777,
                recorder_starttime=1,
            )
            digue._write_take_state(take)
        processes = digue.RecordingProcesses(recorder=recorder, watchdog=None, rec_file=rec_file, take_id=take.take_id)
        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.ensure_server"),
            patch("digue.is_server_running", return_value=True),
            patch("digue.start_recording", return_value=processes),
            patch("digue._wait_recorder_end_daemon", return_value="ended"),
            patch("digue.finish_dictation", side_effect=finish),
            patch("digue.notify"),
            patch("signal.signal"),
        ):
            exit_code = digue.dictate_toggle(config)
            state_file = digue._take_state_file(take.take_id)
        return exit_code, state_file

    def test_daemon_removes_take_state_after_terminal_outcome(self, tmp_path):
        exit_code, state_file = self.run_daemon_delivery(
            tmp_path, lambda *_args, **_kwargs: digue.DeliveryResult(outcome="delivered", exit_code=0)
        )

        assert exit_code == 0
        assert not state_file.exists()
        assert not (tmp_path / "digue-daemon.pid").exists()

    def test_daemon_keeps_take_state_on_retryable_failure(self, tmp_path):
        """Server down and the rescue failed too: the WAV is still in the
        runtime dir, and without its state no toggle would ever pick it up."""
        exit_code, state_file = self.run_daemon_delivery(
            tmp_path, lambda *_args, **_kwargs: digue.DeliveryResult(outcome="retryable_failure", exit_code=1)
        )

        assert exit_code == 1
        assert state_file.exists()
        assert not (tmp_path / "digue-daemon.pid").exists()
        with patch("digue._runtime_dir", return_value=tmp_path):
            [take] = digue._take_states()
            assert take.state == "recording"
            assert take.rec_file.exists()

    def test_daemon_keeps_take_state_on_unexpected_exception(self, tmp_path):
        def explode(*_args, **_kwargs):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            self.run_daemon_delivery(tmp_path, explode)

        with patch("digue._runtime_dir", return_value=tmp_path):
            [take] = digue._take_states()
            assert take.state == "recording"
        assert not (tmp_path / "digue-daemon.pid").exists()

    def test_startup_failure_clears_own_reservation(self, tmp_path):
        config = digue._default_config()
        daemon_file = tmp_path / "digue-daemon.pid"

        def fail_after_reservation(_config):
            import os

            assert daemon_file.read_text().split()[:2] == [str(os.getpid()), "starting"]
            raise RuntimeError("boom")

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.ensure_server"),
            patch("digue.is_server_running", return_value=True),
            patch("digue.start_recording", side_effect=fail_after_reservation),
            patch("digue.notify"),
            patch("signal.signal"),
        ):
            assert digue.dictate_toggle(config) == 1

        assert not daemon_file.exists()

    def test_startup_failure_does_not_clear_new_daemon_state(self, tmp_path):
        config = digue._default_config()
        daemon_file = tmp_path / "digue-daemon.pid"

        def replace_reservation_then_fail(_config):
            import os

            assert daemon_file.read_text().split()[:2] == [str(os.getpid()), "starting"]
            daemon_file.write_text("4242 recording 1")
            raise RuntimeError("boom")

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.ensure_server", side_effect=replace_reservation_then_fail),
            patch("digue.notify"),
        ):
            assert digue.dictate_toggle(config) == 1

        assert daemon_file.read_text() == "4242 recording 1"

    def test_second_toggle_signals_daemon_and_exits_fast(self, tmp_path, capsys):
        """The second dictate sends SIGTERM to the daemon and exits immediately;
        the daemon (not this process) runs the transcription flow."""
        daemon_pid = tmp_path / "digue-daemon.pid"
        daemon_pid.write_text("4242 recording 555")

        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        with (
            patch("digue._daemon_pid_file", return_value=daemon_pid),
            patch("digue._pid_alive", return_value=True),
            patch("digue._process_starttime", return_value="555"),
            patch("os.kill") as mock_kill,
            patch("digue.finish_dictation") as mock_finish,
        ):
            result = digue.dictate_toggle(config)

        assert result == 0
        mock_kill.assert_called_once_with(4242, 15)
        # the toggle must NOT run the transcription logic itself
        mock_finish.assert_not_called()

    def test_daemon_pid_file_removed_when_daemon_dead(self, tmp_path):
        """A stale daemon pid file (crashed daemon) must not block a new recording."""
        daemon_pid = tmp_path / "digue-daemon.pid"
        daemon_pid.write_text("4242 recording 555")
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        with (
            patch("digue._daemon_pid_file", return_value=daemon_pid),
            patch("digue._pid_alive", return_value=False),
            patch("digue.ensure_server", return_value=None),
            patch("digue.is_server_running", return_value=True),
            patch(
                "digue.start_recording",
                return_value=digue.RecordingProcesses(recorder=MagicMock(pid=777, poll=lambda: 0), watchdog=None),
            ) as mock_start,
            patch("digue.notify"),
        ):
            result = digue.dictate_toggle(config)

        assert result == 1
        assert not daemon_pid.exists()
        mock_start.assert_called_once()

    def test_remove_daemon_state_reads_and_unlinks_while_locked(self, tmp_path):
        daemon_file = tmp_path / "digue-daemon.pid"
        daemon_file.write_text("99 delivering 555")
        lock_active = False
        operations = []

        @contextlib.contextmanager
        def tracked_lock():
            nonlocal lock_active
            lock_active = True
            operations.append("enter")
            try:
                yield
            finally:
                operations.append("exit")
                lock_active = False

        original_read_text = Path.read_text
        original_unlink = Path.unlink

        def tracked_read_text(path, *args, **kwargs):
            assert lock_active
            operations.append("read")
            return original_read_text(path, *args, **kwargs)

        def tracked_unlink(path, *args, **kwargs):
            assert lock_active
            operations.append("unlink")
            return original_unlink(path, *args, **kwargs)

        with (
            patch("digue._dictate_lock", side_effect=tracked_lock),
            patch("digue._daemon_pid_file", return_value=daemon_file),
            patch.object(Path, "read_text", tracked_read_text),
            patch.object(Path, "unlink", tracked_unlink),
        ):
            assert digue._remove_daemon_state(99) is True

        assert operations == ["enter", "read", "unlink", "exit"]

    def test_remove_daemon_state_serializes_with_state_publication(self, tmp_path):
        daemon_file = tmp_path / "digue-daemon.pid"
        daemon_file.write_text("99 delivering 555")
        remover_locked = threading.Event()
        release_remover = threading.Event()
        publisher_started = threading.Event()
        errors = []

        original_read_text = Path.read_text

        def pause_after_lock(path, *args, **kwargs):
            result = original_read_text(path, *args, **kwargs)
            if path == daemon_file:
                remover_locked.set()
                assert release_remover.wait(1)
            return result

        def remove_state():
            try:
                assert digue._remove_daemon_state(99) is True
            except BaseException as exc:
                errors.append(exc)

        def publish_state():
            try:
                assert remover_locked.wait(1)
                publisher_started.set()
                with digue._dictate_lock():
                    digue._write_daemon_state(100, "recording")
            except BaseException as exc:
                errors.append(exc)

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._process_starttime", return_value="777"),
            patch.object(Path, "read_text", pause_after_lock),
        ):
            remover = threading.Thread(target=remove_state)
            publisher = threading.Thread(target=publish_state)
            remover.start()
            assert remover_locked.wait(1)
            publisher.start()
            assert publisher_started.wait(1)
            publisher.join(timeout=0.05)
            assert publisher.is_alive()
            release_remover.set()
            remover.join(timeout=1)
            publisher.join(timeout=1)
            assert not remover.is_alive()
            assert not publisher.is_alive()

        assert errors == []
        assert daemon_file.read_text() == "100 recording 777"

    def test_old_daemon_does_not_remove_new_daemon_state(self, tmp_path):
        """A delivering take may finish while a newer take is recording."""
        daemon_file = tmp_path / "digue-daemon.pid"
        daemon_file.write_text("100 recording 555")

        with patch("digue._daemon_pid_file", return_value=daemon_file):
            removed = digue._remove_daemon_state(99)

        assert removed is False
        assert daemon_file.read_text() == "100 recording 555"

    def test_remove_daemon_state_tolerates_an_empty_file(self, tmp_path):
        daemon_file = tmp_path / "digue-daemon.pid"
        daemon_file.write_text("")

        with patch("digue._daemon_pid_file", return_value=daemon_file):
            assert digue._remove_daemon_state(99) is False

        assert daemon_file.exists()

    def test_daemon_removes_only_its_own_state(self, tmp_path):
        daemon_file = tmp_path / "digue-daemon.pid"
        daemon_file.write_text("99 delivering 555")

        with patch("digue._daemon_pid_file", return_value=daemon_file):
            removed = digue._remove_daemon_state(99)

        assert removed is True
        assert not daemon_file.exists()

    def _toggle_against_state(self, tmp_path, state_text):
        """Runs a toggle against a daemon state file; startup is aborted right
        after the state check so only the signaling decision matters."""
        daemon_file = tmp_path / "digue-daemon.pid"
        daemon_file.write_text(state_text)
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.ensure_server", side_effect=RuntimeError("abort startup")),
            patch("digue.notify"),
            patch("os.kill") as mock_kill,
        ):
            digue.dictate_toggle(config)
        return [call.args for call in mock_kill.call_args_list if call.args[1] == 15]

    def test_toggle_does_not_signal_a_recycled_daemon_pid(self, tmp_path):
        """The daemon file may outlive its daemon (SIGKILL, crash before the
        try/finally). If the pid was reused by an unrelated process of the
        same user, a toggle must not SIGTERM it: liveness alone is not identity.
        The test process itself plays the unrelated process."""
        import os

        signals = self._toggle_against_state(tmp_path, f"{os.getpid()} recording")

        assert signals == []

    def test_toggle_does_not_signal_when_starttime_differs(self, tmp_path):
        import os

        signals = self._toggle_against_state(tmp_path, f"{os.getpid()} recording 1")

        assert signals == []

    def test_toggle_signals_the_daemon_whose_identity_matches(self, tmp_path):
        import os

        starttime = digue._process_starttime(os.getpid())
        signals = self._toggle_against_state(tmp_path, f"{os.getpid()} recording {starttime}")

        assert signals == [(os.getpid(), 15)]

    def test_toggle_during_startup_tells_the_user_instead_of_silently_exiting(self, tmp_path):
        """ensure_server can take minutes on first use (image pull, model
        download). A second press during that window must give feedback."""
        import os

        daemon_file = tmp_path / "digue-daemon.pid"
        daemon_file.write_text(f"{os.getpid()} starting {digue._process_starttime(os.getpid())}")
        config = digue._default_config()
        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.notify") as mock_notify,
            patch("os.kill") as mock_kill,
        ):
            result = digue.dictate_toggle(config)

        assert result == 0
        assert [call.args for call in mock_kill.call_args_list if call.args[1] != 0] == []
        assert mock_notify.call_count == 1
        assert "starting" in mock_notify.call_args.args[0].lower()
        assert mock_notify.call_args.kwargs.get("timeout_ms", 0) > 0

    def test_daemon_state_records_the_process_starttime(self, tmp_path):
        import os

        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        seen = []

        def capture_state(_config):
            seen.append((tmp_path / "digue-daemon.pid").read_text())
            raise RuntimeError("abort startup")

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.ensure_server", side_effect=capture_state),
            patch("digue.notify"),
        ):
            digue.dictate_toggle(config)

        assert seen == [f"{os.getpid()} starting {digue._process_starttime(os.getpid())}"]

    def test_recording_filename_is_unique_within_same_second(self):
        with patch("digue.now_timestamp", return_value="20260904-120000"):
            first = digue._rec_file()
            second = digue._rec_file()

        assert first != second


class TestDictateInterrupt:
    @patch("digue.stop_recording_pid", return_value=None)
    @patch("digue.ensure_server")
    @patch("digue.is_server_running", return_value=True)
    @patch("digue.start_recording")
    def test_sigint_during_daemon_wait_stops_and_delivers(
        self, mock_start, mock_running, mock_ensure, mock_stop_pid, tmp_path, capsys
    ):
        """Ctrl+c (SIGINT) in a terminal dictation must stop the recording and
        deliver the take, not discard it (the global KeyboardInterrupt handler
        must not win: the daemon installs its own SIGINT handler)."""
        import signal

        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        handlers = []
        recorder = MagicMock(pid=777)
        recorder.poll.return_value = None
        mock_start.return_value = digue.RecordingProcesses(recorder=recorder, watchdog=None)

        def fake_wait(_recorder, _limit):
            # simulate Ctrl+c arriving during the wait
            digue._on_sigint(signal.SIGINT, None)
            return "interrupted"

        with (
            patch("digue._recording_file_of", return_value=tmp_path / "take.wav"),
            patch("digue._wait_recorder_end_daemon", side_effect=fake_wait),
            patch(
                "digue.finish_dictation", return_value=digue.DeliveryResult(outcome="delivered", exit_code=0)
            ) as mock_finish,
            patch("signal.signal", side_effect=lambda sig, handler: handlers.append((sig, handler))),
        ):
            result = digue.dictate_toggle(config)

        assert result == 0
        mock_stop_pid.assert_called_once()
        mock_finish.assert_called_once()
        # the daemon must have installed its own SIGINT handler
        installed = dict(handlers)
        assert signal.SIGINT in installed
        assert installed[signal.SIGINT] is digue._on_sigint
        assert not digue._daemon_pid_file().exists()

    def test_wait_recorder_end_interrupted_outcome(self):
        recorder = MagicMock()
        recorder.poll.return_value = None
        with patch("time.monotonic", side_effect=[0.0, 0.5]), patch("time.sleep"):
            digue._on_sigint(2, None)
            try:
                assert digue._wait_recorder_end_daemon(recorder, 300) == "interrupted"
            finally:
                digue._got_sigint = False
