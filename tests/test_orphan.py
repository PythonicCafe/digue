"""Tests for orphan take claiming, recovery, and surplus rescue."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import digue


class TestOrphanStartingTake:
    """A daemon that died between publishing 'starting' and registering the
    recorder leaves an orphaned state. Recovery is conservative (no /proc fd
    scanning): recent states are left alone, a missing/empty WAV expires with
    its state, and a non-empty WAV is rescued (never transcribed/pasted
    automatically -- the recorder may still be writing)."""

    def make_starting_take(self, tmp_path, age_seconds, rec_file=None, daemon_pid=999999, daemon_starttime=1):
        import time

        with patch("digue._runtime_dir", return_value=tmp_path):
            take = digue.TakeState(
                version=digue.TAKE_STATE_VERSION,
                take_id="0123456789abcdef",
                created_at_ns=time.time_ns() - int(age_seconds * 1e9),
                state="starting",
                rec_file=rec_file or tmp_path / "digue-recording.wav",
                daemon_pid=daemon_pid,
                daemon_starttime=daemon_starttime,
            )
            digue._write_take_state(take)
        return take

    def test_expired_orphan_without_wav_is_removed(self, tmp_path):

        take = self.make_starting_take(tmp_path, age_seconds=digue.ORPHAN_MIN_AGE_SECONDS + 1)
        config = digue._default_config()

        with patch("digue._runtime_dir", return_value=tmp_path), patch("digue._pid_alive", return_value=False):
            rescued = digue._expire_orphan_starting(config, take)

        assert rescued is None
        assert list(tmp_path.glob("digue-take-*.json")) == []

    def test_expired_orphan_with_wav_is_rescued(self, tmp_path):

        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        take = self.make_starting_take(tmp_path, age_seconds=digue.ORPHAN_MIN_AGE_SECONDS + 1)
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")

        with patch("digue._runtime_dir", return_value=tmp_path), patch("digue._pid_alive", return_value=False):
            rescued = digue._expire_orphan_starting(config, take)

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
        config = digue._default_config()

        with patch("digue._runtime_dir", return_value=tmp_path), patch("digue._pid_alive", return_value=False):
            assert digue._expire_orphan_starting(config, take) is None

        assert len(list(tmp_path.glob("digue-take-*.json"))) == 1

    def test_orphan_with_alive_daemon_is_left_alone(self, tmp_path):

        take = self.make_starting_take(tmp_path, age_seconds=digue.ORPHAN_MIN_AGE_SECONDS + 1, daemon_starttime=555)
        config = digue._default_config()

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._pid_alive", return_value=True),
            patch("digue._process_starttime", return_value="555"),
        ):
            assert digue._expire_orphan_starting(config, take) is None

        assert len(list(tmp_path.glob("digue-take-*.json"))) == 1

    def test_toggle_rescues_expired_orphan_starting_take(self, tmp_path):
        import os

        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        self.make_starting_take(tmp_path, age_seconds=digue.ORPHAN_MIN_AGE_SECONDS + 1, rec_file=rec_file)

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._pid_alive", lambda pid: pid == os.getpid()),
            patch("digue.ensure_server"),
            patch("digue.is_server_running", return_value=True),
            patch("digue.start_recording") as mock_start,
            patch("digue.finish_dictation") as mock_finish,
            patch("digue.notify"),
            patch("digue.notify_close"),
            patch("signal.signal"),
        ):
            assert digue.dictate_toggle(config) == 0

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
            "version": digue.TAKE_STATE_VERSION,
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
        with patch("digue._runtime_dir", return_value=tmp_path):
            take = digue.TakeState(**values)
            digue._write_take_state(take)
        return take

    def test_claims_the_oldest_orphan_and_records_the_recoverer_identity(self, tmp_path):
        import os

        newer = self.make_take(tmp_path, take_id="ffffffffffffffff", created_at_ns=200)
        oldest = self.make_take(tmp_path)

        with patch("digue._runtime_dir", return_value=tmp_path):
            claimed = digue._claim_orphan_take()

        assert claimed is not None
        assert claimed.take_id == oldest.take_id
        assert claimed.state == "recovering"
        assert claimed.recorder_pid == 555
        assert claimed.recoverer_pid == os.getpid()
        assert claimed.recoverer_starttime == int(digue._process_starttime(os.getpid()))
        with patch("digue._runtime_dir", return_value=tmp_path):
            states = {take.take_id: take for take in digue._take_states()}
        assert states[oldest.take_id].state == "recovering"
        assert states[newer.take_id].state == "recording"

    def test_take_with_live_daemon_is_not_claimed(self, tmp_path):
        import os

        self.make_take(tmp_path, daemon_pid=os.getpid(), daemon_starttime=int(digue._process_starttime(os.getpid())))

        with patch("digue._runtime_dir", return_value=tmp_path):
            assert digue._claim_orphan_take() is None

    def test_recovering_take_with_live_recoverer_is_not_claimed(self, tmp_path):
        import os

        self.make_take(
            tmp_path,
            state="recovering",
            recorder_pid=None,
            recorder_starttime=None,
            recoverer_pid=os.getpid(),
            recoverer_starttime=int(digue._process_starttime(os.getpid())),
        )

        with patch("digue._runtime_dir", return_value=tmp_path):
            assert digue._claim_orphan_take() is None

    def test_recovering_take_with_dead_recoverer_is_claimed(self, tmp_path):
        self.make_take(
            tmp_path,
            state="recovering",
            recorder_pid=None,
            recorder_starttime=None,
            recoverer_pid=999999,
            recoverer_starttime=1,
        )

        with patch("digue._runtime_dir", return_value=tmp_path):
            claimed = digue._claim_orphan_take()

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
            with digue._dictate_lock():
                first_in_lock.set()
                assert release_first.wait(2)
                claimed.append(digue._claim_orphan_take())

        def recover_second():
            assert first_in_lock.wait(2)
            release_first.set()
            with digue._dictate_lock():
                claimed.append(digue._claim_orphan_take())

        with patch("digue._runtime_dir", return_value=tmp_path):
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
        config = digue._default_config()

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._pid_alive", return_value=True),
            patch("digue._process_starttime", return_value="555"),
            patch("digue.notify"),
            patch("os.kill") as mock_kill,
        ):
            assert digue.dictate_toggle(config) == 0

        mock_kill.assert_called_once_with(4242, 15)
        with patch("digue._runtime_dir", return_value=tmp_path):
            states = digue._take_states()
        assert len(states) == 1 and states[0].state == "recording"

    def test_toggle_recovers_the_claimed_orphan_and_does_not_start_a_new_take(self, tmp_path):
        """The recovering toggle delivers the orphan and returns: recording
        here too would leave this take and a concurrent one competing for the
        same daemon state (two recorders, one stop)."""
        import os

        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        self.make_take(tmp_path)
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        finish_calls = []

        def fake_finish(_config, file, limit_reached=False, take_id=None):
            finish_calls.append((file, take_id))
            return digue.DeliveryResult(outcome="delivered", exit_code=0)

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._pid_alive", lambda pid: pid == os.getpid()),
            patch("digue.ensure_server"),
            patch("digue.is_server_running", return_value=True),
            patch("digue.stop_recording_pid", return_value=rec_file) as mock_stop,
            patch("digue.finish_dictation", side_effect=fake_finish),
            patch("digue.start_recording") as mock_start,
            patch("digue.notify"),
            patch("digue.notify_close"),
            patch("signal.signal"),
        ):
            assert digue.dictate_toggle(config) == 0

        assert finish_calls == [(rec_file, "0123456789abcdef")]
        mock_stop.assert_called_once()
        mock_start.assert_not_called()
        assert list(tmp_path.glob("digue-take-*.json")) == []
        assert not (tmp_path / "digue-daemon.pid").exists()

    def test_toggle_pressed_during_recovery_starts_exactly_one_new_take(self, tmp_path):
        """Regression: while a toggle recovers an orphan, a second press must
        find a single consistent picture -- no daemon state (so it starts a new
        take) and the recovering toggle never records afterwards."""
        import os

        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        self.make_take(tmp_path)
        config = digue._default_config()
        seen: dict[str, object] = {}

        def fake_finish(_config, file, limit_reached=False, take_id=None):
            with patch("digue._runtime_dir", return_value=tmp_path):
                seen["daemon_state_during_recovery"] = digue._daemon_state()
                seen["claim_during_recovery"] = digue._claim_orphan_take()
            return digue.DeliveryResult(outcome="delivered", exit_code=0)

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._pid_alive", lambda pid: pid == os.getpid()),
            patch("digue.ensure_server"),
            patch("digue.is_server_running", return_value=True),
            patch("digue.stop_recording_pid", return_value=rec_file),
            patch("digue.finish_dictation", side_effect=fake_finish),
            patch("digue.start_recording") as mock_start,
            patch("digue.notify"),
            patch("digue.notify_close"),
        ):
            assert digue.dictate_toggle(config) == 0

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

        with patch("digue._runtime_dir", return_value=tmp_path), patch("digue._pid_alive", return_value=False):
            assert digue._claim_orphan_take() is None
            states = digue._take_states()

        assert len(states) == 1 and states[0].state == "starting"

    def test_recovery_failure_to_reach_the_server_keeps_the_claim_for_retry(self, tmp_path):
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        self.make_take(tmp_path)
        config = digue._default_config()

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._pid_alive", return_value=False),
            patch("digue.ensure_server", return_value=None),
            patch("digue.is_server_running", return_value=False),
            patch("digue.stop_recording_pid") as mock_stop,
            patch("digue.finish_dictation") as mock_finish,
            patch("digue.start_recording") as mock_start,
            patch("digue.notify"),
        ):
            assert digue.dictate_toggle(config) == 1

        mock_stop.assert_not_called()
        mock_finish.assert_not_called()
        mock_start.assert_not_called()
        with patch("digue._runtime_dir", return_value=tmp_path):
            states = digue._take_states()
        assert len(states) == 1 and states[0].state == "recovering"
        assert rec_file.exists()

    def test_toggle_during_slow_recovery_starts_a_new_take(self, tmp_path):
        """A recovering take with a live recoverer is like a delivering one:
        the toggle does not touch it and starts a new take."""
        import os

        self.make_take(
            tmp_path,
            state="recovering",
            recorder_pid=None,
            recorder_starttime=None,
            recoverer_pid=os.getpid(),
            recoverer_starttime=int(digue._process_starttime(os.getpid())),
        )
        config = digue._default_config()

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._pid_alive", lambda pid: pid == os.getpid()),
            patch("digue.ensure_server", side_effect=RuntimeError("abort startup")),
            patch("digue.notify"),
            patch("os.kill") as mock_kill,
        ):
            assert digue.dictate_toggle(config) == 1

        mock_kill.assert_not_called()
        with patch("digue._runtime_dir", return_value=tmp_path):
            states = digue._take_states()
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
            "version": digue.TAKE_STATE_VERSION,
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
        with patch("digue._runtime_dir", return_value=tmp_path):
            take = digue.TakeState(**values)
            digue._write_take_state(take)
        return take

    def make_config(self, tmp_path):
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        return config

    def patch_recovery(self, tmp_path, finish_result):
        return (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.finish_dictation", return_value=finish_result),
        )

    def test_stops_a_live_recorder_with_valid_identity_and_delivers(self, tmp_path):
        recorder = subprocess.Popen(["sleep", "60"], start_new_session=True)
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        take = self.make_recovering_take(
            tmp_path,
            recorder_pid=recorder.pid,
            recorder_starttime=int(digue._process_starttime(recorder.pid)),
            rec_file=rec_file,
        )
        delivered = []

        def fake_finish(_config, file, limit_reached=False, take_id=None):
            delivered.append((file, take_id))
            return digue.DeliveryResult(outcome="delivered", exit_code=0)

        try:
            with (
                patch("digue._runtime_dir", return_value=tmp_path),
                patch("digue.finish_dictation", side_effect=fake_finish),
            ):
                exit_code = digue._recover_claimed_take(self.make_config(tmp_path), take)
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
            return digue.DeliveryResult(outcome="delivered", exit_code=0)

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.os.killpg") as mock_killpg,
            patch("digue.finish_dictation", side_effect=fake_finish),
        ):
            exit_code = digue._recover_claimed_take(self.make_config(tmp_path), take)

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
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.send_text") as mock_send,
            patch("digue.transcribe") as mock_transcribe,
            patch("digue.notify"),
        ):
            exit_code = digue._recover_claimed_take(config, take)

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
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.send_text") as mock_send,
            patch("digue.transcribe") as mock_transcribe,
            patch("digue.notify"),
        ):
            exit_code = digue._recover_claimed_take(config, take)

        assert exit_code == 0
        mock_send.assert_not_called()
        mock_transcribe.assert_not_called()
        assert not rec_file.exists()
        assert (month_dir / f"{stem}.wav").read_bytes() == b"complete audio"
        assert (month_dir / f"{stem}.txt").read_text() == "already pasted\n"
        assert sorted(path.name for path in month_dir.iterdir()) == [f"{stem}.txt", f"{stem}.wav"]
        assert list(tmp_path.glob("digue-take-*.json")) == []

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
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.send_text"),
            patch("digue.transcribe"),
            patch("digue.notify"),
        ):
            exit_code = digue._recover_claimed_take(config, take)

        assert exit_code == 0
        assert (month_dir / f"{stem}.wav").read_bytes() == b"complete audio"
        assert (month_dir / f"{other_stem}.wav").read_bytes() == b"other audio"
        assert (month_dir / f"{other_stem}.txt").read_text() == "other text\n"
        assert list(tmp_path.glob("digue-take-*.json")) == []

    def test_recycled_recorder_pid_is_not_signaled(self, tmp_path):
        import os

        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        take = self.make_recovering_take(tmp_path, rec_file=rec_file, recorder_pid=os.getpid(), recorder_starttime=111)
        delivered = []

        def fake_finish(_config, file, limit_reached=False, take_id=None):
            delivered.append(file)
            return digue.DeliveryResult(outcome="delivered", exit_code=0)

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._process_starttime", return_value="999"),
            patch("digue.os.killpg") as mock_killpg,
            patch("digue.finish_dictation", side_effect=fake_finish),
        ):
            exit_code = digue._recover_claimed_take(self.make_config(tmp_path), take)

        assert exit_code == 0
        mock_killpg.assert_not_called()
        assert delivered == [take.rec_file]
        assert list(tmp_path.glob("digue-take-*.json")) == []

    def test_retryable_failure_preserves_state_and_wav(self, tmp_path):
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        take = self.make_recovering_take(tmp_path, rec_file=rec_file)

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch(
                "digue.finish_dictation",
                return_value=digue.DeliveryResult(outcome="retryable_failure", exit_code=1),
            ),
        ):
            exit_code = digue._recover_claimed_take(self.make_config(tmp_path), take)

        assert exit_code == 1
        assert rec_file.exists()
        assert list(tmp_path.glob("digue-take-*.json")) == [tmp_path / f"digue-take-{take.take_id}.json"]

    def test_unexpected_exception_preserves_state_and_wav(self, tmp_path):
        rec_file = tmp_path / "digue-recording.wav"
        rec_file.write_bytes(b"audio")
        take = self.make_recovering_take(tmp_path, rec_file=rec_file)

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.finish_dictation", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError),
        ):
            digue._recover_claimed_take(self.make_config(tmp_path), take)

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
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.notify") as mock_notify,
        ):
            exit_code = digue._recover_claimed_take(self.make_config(tmp_path), take)

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
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue.transcribe", return_value=""),
            patch("digue.notify"),
        ):
            exit_code = digue._recover_claimed_take(self.make_config(tmp_path), take)

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
            "version": digue.TAKE_STATE_VERSION,
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
        with patch("digue._runtime_dir", return_value=tmp_path):
            take = digue.TakeState(**values)
            digue._write_take_state(take)
        return take

    def test_toggle_delivers_the_oldest_and_rescues_the_rest(self, tmp_path):
        import os

        oldest_wav = tmp_path / "digue-oldest.wav"
        oldest_wav.write_bytes(b"audio oldest")
        surplus_a = tmp_path / "digue-surplus-a.wav"
        surplus_a.write_bytes(b"audio a")
        surplus_b = tmp_path / "digue-surplus-b.wav"
        surplus_b.write_bytes(b"audio b")
        oldest = self.make_take(tmp_path, "0123456789abcdef", 100, rec_file=oldest_wav)
        surplus_take_a = self.make_take(tmp_path, "aaaaaaaaaaaaaaaa", 200, rec_file=surplus_a)
        surplus_take_b = self.make_take(tmp_path, "bbbbbbbbbbbbbbbb", 300, rec_file=surplus_b)
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        config["dictate"]["max_duration"] = 0
        recorder = MagicMock(pid=os.getpid(), poll=lambda: 0)
        finish_take_ids = []

        def fake_finish(_config, rec_file, limit_reached=False, take_id=None):
            finish_take_ids.append(take_id)
            return digue.DeliveryResult(outcome="delivered", exit_code=0)

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._pid_alive", lambda pid: pid == os.getpid()),
            patch("digue.ensure_server"),
            patch("digue.is_server_running", return_value=True),
            patch("digue.stop_recording_pid", side_effect=lambda pid, rec_file, expected_starttime=None: rec_file),
            patch("digue.finish_dictation", side_effect=fake_finish),
            patch("subprocess.Popen", return_value=recorder),
            patch("digue._wait_recorder_end_daemon", return_value="ended"),
            patch("digue.notify") as mock_notify,
            patch("digue.notify_close"),
            patch("signal.signal"),
        ):
            assert digue.dictate_toggle(config) == 0

        # only the oldest orphan is transcribed and pasted
        assert finish_take_ids == [oldest.take_id]
        assert surplus_take_a.take_id not in finish_take_ids
        assert surplus_take_b.take_id not in finish_take_ids
        month = tmp_path / "audio" / digue.month_dir_for(digue.now_timestamp())
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
        rescued_notifies = [call for call in mock_notify.call_args_list if "rescued to" in str(call.args)]
        assert len(rescued_notifies) == 1
        message = rescued_notifies[0].args[0]
        assert "2 recordings rescued to" in message
        assert str(tmp_path / "audio") in message

    def test_surplus_take_with_live_recorder_is_stopped_before_rescue(self, tmp_path):
        import os

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
                recorder_starttime=int(digue._process_starttime(surplus_recorder.pid)),
            )
            config = digue._default_config()
            config["dictate"]["audio_dir"] = str(tmp_path / "audio")
            config["dictate"]["max_duration"] = 0
            recorder = MagicMock(pid=os.getpid(), poll=lambda: 0)

            with (
                patch("digue._runtime_dir", return_value=tmp_path),
                patch("digue._pid_alive", lambda pid: pid == os.getpid()),
                patch("digue.ensure_server"),
                patch("digue.is_server_running", return_value=True),
                patch("digue.finish_dictation", return_value=digue.DeliveryResult(outcome="delivered", exit_code=0)),
                patch("subprocess.Popen", return_value=recorder),
                patch("digue._wait_recorder_end_daemon", return_value="ended"),
                patch("digue.notify"),
                patch("digue.notify_close"),
                patch("signal.signal"),
            ):
                assert digue.dictate_toggle(config) == 0

            assert surplus_recorder.poll() is not None
            month = tmp_path / "audio" / digue.month_dir_for(digue.now_timestamp())
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
        import contextlib

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
            patch("digue._dictate_lock", recording_lock),
            patch("digue._claim_orphan_take", side_effect=fake_claim),
            patch("digue._rescue_surplus_take", return_value=tmp_path / "rescued.wav"),
            patch("digue._take_state_file", return_value=tmp_path / "gone.json"),
        ):
            rescued = digue._rescue_surplus_orphans(digue._default_config())

        assert rescued == [tmp_path / "rescued.wav"]
        assert claims_under_lock == [True, True]

    def test_surplus_rescue_failure_preserves_state_and_wav(self, tmp_path):
        import os

        oldest_wav = tmp_path / "digue-oldest.wav"
        oldest_wav.write_bytes(b"audio oldest")
        surplus_wav = tmp_path / "digue-surplus.wav"
        surplus_wav.write_bytes(b"audio surplus")
        self.make_take(tmp_path, "0123456789abcdef", 100, rec_file=oldest_wav)
        surplus_take = self.make_take(tmp_path, "aaaaaaaaaaaaaaaa", 200, rec_file=surplus_wav)
        config = digue._default_config()
        config["dictate"]["audio_dir"] = str(tmp_path / "audio")
        config["dictate"]["max_duration"] = 0
        recorder = MagicMock(pid=os.getpid(), poll=lambda: 0)

        with (
            patch("digue._runtime_dir", return_value=tmp_path),
            patch("digue._pid_alive", lambda pid: pid == os.getpid()),
            patch("digue.ensure_server"),
            patch("digue.is_server_running", return_value=True),
            patch("digue.finish_dictation", return_value=digue.DeliveryResult(outcome="delivered", exit_code=0)),
            patch("digue.rescue_recording", return_value=None),
            patch("subprocess.Popen", return_value=recorder),
            patch("digue._wait_recorder_end_daemon", return_value="ended"),
            patch("digue.notify"),
            patch("digue.notify_close"),
            patch("signal.signal"),
        ):
            assert digue.dictate_toggle(config) == 0

        assert surplus_wav.exists()
        assert (tmp_path / f"digue-take-{surplus_take.take_id}.json").exists()
