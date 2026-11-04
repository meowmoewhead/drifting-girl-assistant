"""Spent calibration windows yield without replaying or hiding input faults."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from drift.runtime import Clicker
from drift.tasks import meter_journal as records
from drift.tasks.meter_limits import MeterWorkflowLimit
from tests import test_meter_calibration as base


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.path = self.root/'meter_calibration_state.json'
        self.record = dict(version=1, pending=True, rank=38, stage=9, restore_running=True,
            cycles=48, deadline=100, attempts={'cancel':2}, sailing_attempts={'stop':1},
            proof=None, training={'sailing_sleep_recovery':True,'fishing_meter_limits':{'place':{'high':1604}}})
        self.path.write_text(json.dumps(self.record), encoding='utf-8')
        self.raw = self.path.read_bytes(); self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_start_retirement_archives_exact_bytes_and_preserves_restore_and_budgets(self):
        with patch('drift.tasks.meter_journal.time.time', return_value=200):
            result = records.retire(self.path, self.digest, source='startup_expired')
        self.assertEqual(Path(result['archive']).read_bytes(), self.raw)
        saved = json.loads(self.path.read_text('utf-8'))
        for key in set(self.record)-{'pending'}:
            self.assertEqual(saved[key], self.record[key], key)
        self.assertFalse(saved['pending']); self.assertFalse(saved['retirement']['released'])
        self.assertTrue(saved['retirement']['had_inputs'])
        self.assertTrue(records.status(self.root)['recovery_required'])

    def test_current_window_is_not_retired_on_start(self):
        with patch('drift.tasks.meter_journal.time.time', return_value=50):
            self.assertFalse(records.retire(self.path,self.digest,source='startup_expired')['retired'])
        self.assertEqual(self.path.read_bytes(),self.raw)

    def test_manual_clear_cannot_clear_running_worker_or_stale_record(self):
        app=object.__new__(Clicker);app.root=self.root;app.lock=threading.RLock()
        app.state={};app.event=lambda _:None;app.running=lambda:True
        with self.assertRaises(ValueError):app.clear_meter_calibration({'sha256':self.digest})
        app.running=lambda:False
        with self.assertRaises(ValueError):app.clear_meter_calibration({'sha256':'0'*64})
        self.assertEqual(self.path.read_bytes(),self.raw)
        self.assertTrue(app.clear_meter_calibration({'sha256':self.digest})['retired'])
        self.assertTrue(records.status(self.root)['recovery_required'])

    def test_archive_conflict_and_record_race_fail_closed(self):
        archive=self.root/'logs/meter-calibration-retired'/f'{self.digest}.json'
        archive.parent.mkdir(parents=True);archive.write_bytes(b'conflict')
        with self.assertRaises(RuntimeError):records.retire(self.path,self.digest,source='manual_clear')
        self.assertEqual(self.path.read_bytes(),self.raw)
        archive.write_bytes(self.raw)
        from drift.core import journal
        original=journal.archive_bytes
        def change_after_archive(path, raw, label):
            original(path,raw,label);self.path.write_bytes(b'changed')
        with patch.object(journal,'archive_bytes',side_effect=change_after_archive):
            with self.assertRaises(ValueError):records.retire(self.path,self.digest,source='manual_clear')
        self.assertEqual(self.path.read_bytes(),b'changed')

    def test_only_matching_retry_is_invalidated_and_archived(self):
        retry=self.root/'meter_calibration_retry.json'
        retry.write_text(json.dumps({'failed_sha256':'other'}));other=retry.read_bytes()
        records.retire(self.path,self.digest,source='manual_clear')
        self.assertEqual(retry.read_bytes(),other)
        self.path.write_bytes(self.raw)
        retry.write_text(json.dumps({'failed_sha256':self.digest}));raw=retry.read_bytes()
        records.retire(self.path,self.digest,source='manual_clear')
        self.assertFalse(retry.exists())
        self.assertEqual((self.root/'logs/meter-calibration-retired'/f'{self.digest}-retry.json').read_bytes(),raw)


class ControllerTests(unittest.TestCase):
    setUpClass=classmethod(base.MeterCalibrationTests.setUpClass.__func__)
    setUp=base.MeterCalibrationTests.setUp
    tearDown=base.MeterCalibrationTests.tearDown
    pending=base.MeterCalibrationTests.pending
    restart=base.MeterCalibrationTests.restart

    def spent(self):
        c=self.pending('prepare');c.journal.update(stage=9,rank=38,deadline=990,
            attempts={'cancel':2},sailing_attempts={'stop':1},cycles=48)
        c.cycles=48;c._save();return c,c.path.read_bytes()

    def test_constructor_does_not_retire_but_explicit_start_does_without_input(self):
        c,raw=self.spent();c=self.restart(c.journal)
        self.assertTrue(c.active)
        with patch('drift.tasks.meter_calibration.time.time',return_value=1000):
            self.assertTrue(c.retire_expired_on_start())
            for action in ('cancel','start','drag','reveal'):self.assertFalse(c.reserve_meter_input(action))
            for action in ('stop','open','start'):self.assertFalse(c.reserve_sailing_input(action))
        self.assertFalse(c.active);self.assertTrue(c.awaiting_recovery)
        self.assertEqual(c.journal['deadline'],990);self.assertEqual(c.journal['attempts'],{'cancel':2})
        self.assertEqual(self.inputs,[])

    def test_valid_explicit_retry_keeps_own_original_authorization(self):
        c,raw=self.spent();c.retry={'expires_at':1100}
        with patch('drift.tasks.meter_calibration.time.time',return_value=1000):
            self.assertFalse(c.retire_expired_on_start())
        self.assertEqual(c.path.read_bytes(),raw)

    def test_cancel_limit_is_skipped_but_unknown_input_failure_still_raises(self):
        c,raw=self.spent()
        with patch.object(c,'_step',side_effect=MeterWorkflowLimit('取消上限')):
            self.assertFalse(c.step(np.zeros((1280,720,3),dtype='uint8')))
        self.assertEqual(Path(c.journal['retirement']['archive']).read_bytes(),raw)
        self.assertEqual(c.cycles,48);self.assertEqual(self.inputs,[])
        c.journal['retirement']=None
        with patch.object(c,'_step',side_effect=RuntimeError('DOWN结果不确定')):
            with self.assertRaisesRegex(RuntimeError,'DOWN'):c.step(None)
        with patch.object(c,'_step',side_effect=InterruptedError('stop')):
            with self.assertRaises(InterruptedError):c.step(None)

    def test_elapsed_interval_and_restart_never_reset_unresolved_attempts(self):
        c,raw=self.spent();c._retire_window('取消上限')
        saved=deepcopy(c.journal);c=self.restart(saved)
        with patch('drift.tasks.meter_calibration.time.time',return_value=9999999999):
            for _ in range(8):self.assertFalse(c.step(np.zeros((1280,720,3),dtype='uint8')))
        self.assertEqual(c.journal,saved);self.assertEqual(self.inputs,[])

    def test_real_new_frames_restore_readonly_and_then_wait_interval(self):
        c,raw=self.spent();c._retire_window('取消上限')
        image=np.asarray(Image.open(base.ROOT/'tests/fixtures/meter-max-drag/restored-boss-1.png').convert('RGB'))
        # Real game pixels: original resource route has reached a BOSS at R37.
        # Read-only release may observe it; meter inputs still reject BOSS.
        with patch('drift.tasks.meter_calibration.time.time',return_value=2000):
            for now in (100,101.5):
                with patch('drift.tasks.meter_calibration.time.monotonic',return_value=now):
                    self.assertFalse(c.step(image.copy()))
                self.assertTrue(c.awaiting_recovery)
            with patch('drift.tasks.meter_calibration.time.monotonic',return_value=103):
                c.pre_battle_step(image.copy())
            self.assertFalse(c.awaiting_recovery)
            self.assertEqual(c.journal['next_check'],2060)
            self.assertFalse(c.step(image.copy()))
        self.assertFalse(c.journal['retirement']['calibration_completed'])
        self.assertEqual(c.cycles,48);self.assertEqual(c.journal['attempts'],{'cancel':2})
        self.assertEqual(self.inputs,[])

    def test_single_reused_frame_unknown_frame_and_off_route_cannot_release(self):
        c,raw=self.spent();c._retire_window('取消上限')
        image=np.asarray(Image.open(base.ROOT/'tests/fixtures/meter-max-drag/restored-boss-1.png').convert('RGB'))
        for now in (100,101.5,103,104.5):
            with patch('drift.tasks.meter_calibration.time.monotonic',return_value=now):c.step(image)
        self.assertTrue(c.awaiting_recovery);self.assertEqual(c.seen,1)
        c.step(np.zeros_like(image));self.assertEqual(c.seen,0)
        with patch.object(c.sailing,'running',return_value=False):
            self.assertIsNone(c.screen(image.copy(),observe_retired=True))
        with patch.object(c.world,'_rank_value',return_value=51):
            self.assertIsNone(c.screen(image.copy(),observe_retired=True))
        self.assertEqual(self.inputs,[])

    def test_released_legacy_failure_uses_backoff_without_rewriting_saved_evidence(self):
        c=self.controller;c.interval=5400
        c.journal.update(pending=False,finished_at=1000,next_check=15400,
            deadline=9900,attempts={'cancel':1},cycles=59,
            retirement=dict(released=True,verified_at=10000,calibration_completed=False))
        c._save();raw=c.path.read_bytes();saved=deepcopy(c.journal)
        self.assertEqual(c._next_check(),10300)
        self.assertEqual(c.status()['wait_kind'],'failure_backoff')
        self.assertEqual(c.journal,saved)
        self.assertEqual(c.path.read_bytes(),raw)
        self.assertEqual(self.inputs,[])

    def test_failure_backoff_never_bypasses_unreleased_record_or_success_cooldown(self):
        c=self.controller;c.interval=5400
        c.journal.update(pending=False,finished_at=9000,next_check=15400,
            retirement=dict(released=True,verified_at=10000,calibration_completed=False))
        self.assertEqual(c._next_check(),14400)
        c.journal['retirement']['released']=False
        self.assertTrue(c.awaiting_recovery)
        self.assertEqual(c._next_check(),15400)
        c.journal['retirement']['released']=True
        c.journal.update(finished_at=11000,next_check=16400)
        self.assertEqual(c._next_check(),16400)
        self.assertEqual(c.status()['wait_kind'],'success_cooldown')

    def test_failure_backoff_respects_shorter_user_interval_and_readonly_resolution(self):
        c=self.controller;c.interval=60
        c.journal.update(pending=False,next_check=15400,
            resolution=dict(verified_at=10000,calibration_completed=False))
        self.assertEqual(c._next_check(),10060)
        c.journal['resolution']['calibration_completed']=True
        self.assertEqual(c._next_check(),15400)

    def test_failed_calibration_wait_is_not_reported_as_success(self):
        c=self.controller;c.interval=5400
        c.journal.update(pending=False,next_check=15400,
            retirement=dict(released=True,verified_at=10000,calibration_completed=False))
        with patch('drift.tasks.meter_calibration.time.time',return_value=10001), \
             patch.object(c,'screen',side_effect=AssertionError('backoff is read only')):
            self.assertFalse(c.step(None))
        self.assertIn('未完成',c.status()['hold'])
        self.assertNotIn('刚完成',c.status()['hold'])
        self.assertEqual(self.inputs,[])

    def test_real_release_after_failed_window_waits_five_minutes_not_ninety(self):
        c,raw=self.spent();c.interval=5400;c._retire_window('取消上限')
        image=np.asarray(Image.open(base.ROOT/'tests/fixtures/meter-max-drag/restored-boss-1.png').convert('RGB'))
        with patch('drift.tasks.meter_calibration.time.time',return_value=10000):
            for now in (100,101.5,103):
                with patch('drift.tasks.meter_calibration.time.monotonic',return_value=now):
                    c.step(image.copy())
        self.assertFalse(c.awaiting_recovery)
        self.assertEqual(c.journal['next_check'],10300)
        self.assertEqual(c._next_check(),10300)
        self.assertFalse(c.journal['retirement']['calibration_completed'])
        self.assertEqual(c.cycles,48)
        self.assertEqual(c.journal['attempts'],{'cancel':2})
        self.assertEqual(Path(c.journal['retirement']['archive']).read_bytes(),raw)
        self.assertEqual(self.inputs,[])
