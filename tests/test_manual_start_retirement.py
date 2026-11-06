from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from drift.core.session import LoopState
from drift.tasks import handoff_journal as records
from drift.tasks.stone_farm import StoneFarm
from drift.runtime import Clicker
from tests import test_stone_farm as base


class RetirementTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path=self.root/'stone_farm_state.json'
        self.record=dict(version=1,pending=True,phase='first_close',deadline=100,phase_deadline=80,
            first_deadline=90,return_deadline=50,started_at=1,reason='cycle',from_stage=8,to_stage=9,
            rank=40,entry_rank=0,attempts={'first-cast':1},issued_inputs=[{'kind':'first-cast','input_id':'original'}],
            cycles=5,handoffs=6)
        self.path.write_text(json.dumps(self.record),encoding='utf-8')
        self.raw=self.path.read_bytes();self.digest=hashlib.sha256(self.raw).hexdigest()
        self.clock=patch('drift.tasks.handoff_journal.time.time',return_value=110);self.clock.start();self.addCleanup(self.clock.stop)

    def test_expired_start_archives_exact_bytes_and_keeps_every_budget(self):
        result=records.retire(self.root,self.digest,source='startup_expired')
        self.assertEqual(Path(result['archive']).read_bytes(),self.raw)
        saved=json.loads(self.path.read_text('utf-8'))
        self.assertFalse(saved['pending']);self.assertEqual(saved['phase'],'idle')
        for key in set(self.record)-{'pending','phase'}:self.assertEqual(saved[key],self.record[key],key)
        # entry_rank=0 is the cycle journal's "not observed" marker, never a
        # legal RANK, so only the real from-side context may be blocked.
        self.assertEqual(saved['retirement']['blocked_contexts'],[[8,40]])
        self.assertFalse(records.status(self.root)['pending'])

    def test_live_record_is_not_auto_retired_but_manual_clear_can_archive_it(self):
        with patch('drift.tasks.handoff_journal.time.time',return_value=10):
            self.assertFalse(records.retire(self.root,self.digest,source='startup_expired')['retired'])
            self.assertEqual(self.path.read_bytes(),self.raw)
            self.assertTrue(records.retire(self.root,self.digest,source='manual_clear')['retired'])

    def test_stale_hash_and_archive_conflict_never_clear_original(self):
        with self.assertRaises(ValueError):records.retire(self.root,'0'*64,source='manual_clear')
        archive=self.root/'logs/stone-farm-retired'/f'{self.digest}.json'
        archive.parent.mkdir(parents=True);archive.write_bytes(b'conflict')
        with self.assertRaises(RuntimeError):records.retire(self.root,self.digest,source='startup_expired')
        self.assertEqual(self.path.read_bytes(),self.raw)

    def test_retry_receipt_is_archived_and_invalidated_only_for_exact_old_hash(self):
        retry=self.path.with_name('stone_farm_retry.json');raw=json.dumps(dict(failed_sha256=self.digest,expires_at=80)).encode()
        retry.write_bytes(raw);records.retire(self.root,self.digest,source='manual_clear')
        self.assertFalse(retry.exists())
        self.assertEqual((self.root/'logs/stone-farm-retired'/f'{self.digest}-retry.json').read_bytes(),raw)

    def test_bad_deadline_keeps_record_and_reports_in_panel(self):
        self.record['first_deadline']=float('nan');self.path.write_text(json.dumps(self.record))
        self.assertIn('error',records.status(self.root))
        with self.assertRaises(ValueError):records.expiry(self.record,110)

    def test_panel_clear_requires_stopped_worker_and_current_hash(self):
        app=object.__new__(Clicker);app.root=self.root;app.lock=threading.RLock();app.state={};app.event=lambda _:None
        app.running=lambda:True
        with self.assertRaises(ValueError):app.clear_stone_farm(dict(sha256=self.digest))
        self.assertEqual(self.path.read_bytes(),self.raw)
        app.running=lambda:False
        with self.assertRaises(ValueError):app.clear_stone_farm(dict(sha256='x'))
        self.assertTrue(app.clear_stone_farm(dict(sha256=self.digest))['retired'])


class StartupControllerTests(unittest.TestCase):
    setUpClass=classmethod(base.StoneFarmTests.setUpClass.__func__)
    setUp=base.StoneFarmTests.setUp
    tearDown=base.StoneFarmTests.tearDown
    act=base.StoneFarmTests.act
    begin=base.StoneFarmTests.begin
    tick=base.StoneFarmTests.tick

    def expired(self):
        self.begin();self.w.journal['attempts']={'stop':3};self.w._save()
        self.now+=601
        return self.path.read_bytes()

    def test_constructor_is_read_only_and_startup_retirement_prevents_same_handoff_replay(self):
        raw=self.expired()
        self.w=StoneFarm(base.ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertTrue(self.w.retire_expired_on_start())
        for _ in range(8):self.assertFalse(self.tick(self.main.copy()))
        self.assertEqual(self.actions,[]);self.assertEqual(self.w.journal['attempts'],{'stop':3})
        self.assertFalse(self.w.active)

    def test_fresh_confirmed_progress_releases_block_but_unknown_or_single_frame_cannot(self):
        self.expired();self.w.retire_expired_on_start()
        progress=dict(stage=9,rank=38,menu=False,completion=False,route=True)
        with patch.object(self.w,'screen',return_value=progress):
            self.tick();self.assertFalse(self.w.journal['retirement']['released'])
            self.tick();self.tick()
        self.assertTrue(self.w.journal['retirement']['released']);self.assertEqual(self.actions,[])
        # A new legitimate 8/R40 cycle can subsequently acquire ownership.
        with patch.object(self.w,'screen',return_value=dict(progress,stage=8,rank=40)),patch.object(self.w,'_expected',return_value=None):
            for _ in range(3):self.tick()
        self.assertTrue(self.w.active);self.assertEqual(self.w.journal['reason'],'cycle')

    def test_valid_existing_explicit_retry_wins_over_startup_retirement(self):
        raw=self.expired();self.w.retry={'expires_at':self.now+60}
        self.assertFalse(self.w.retire_expired_on_start());self.assertEqual(self.path.read_bytes(),raw)

    def test_first_close_timeout_describes_map_and_preserves_deadlines(self):
        self.begin();self.w.journal['first_deadline']=self.now+60;self.w._set('first_close')
        old=deepcopy(self.w.journal)
        self.w.report_first_close_wait('骷髅红下颌低于阈值')
        self.assertEqual(self.w.journal,old)
        self.now+=61
        with self.assertRaisesRegex(RuntimeError,'关闭首任务地图.*红下颌'):self.w.check_deadline()


class RecoveryClockTests(unittest.TestCase):
    def state(self):
        return LoopState(stage='achievements_tab',stage_trigger='crown',stage_started_at=0,
                         stage_ready_at=2,stage_deadline=30,run_deadline=200)

    def test_wait_time_and_gap_are_counted_once_without_extending_formal_run(self):
        st=self.state()
        st.account_recovery_wait('wait',10,12)
        st.account_recovery_wait('wait',20,23)
        st.account_recovery_wait(None,30,32)
        self.assertEqual((st.stage_started_at,st.stage_ready_at,st.stage_deadline),(22,24,52))
        self.assertEqual(st.run_deadline,200)
        st.account_recovery_wait(None,33,40)
        self.assertEqual(st.stage_deadline,52)

    def test_expired_before_recovery_is_never_revived(self):
        st=self.state();st.account_recovery_wait('wait',31,100)
        self.assertEqual(st.stage_deadline,30)

    def test_cleanup_deadline_uses_same_recovery_budget(self):
        st=self.state();st.daily_cleanup_deadline=35
        st.account_recovery_wait('wait',10,12);st.account_recovery_wait(None,20,22)
        self.assertEqual((st.stage_deadline,st.daily_cleanup_deadline),(42,47))
        self.assertEqual(st.run_deadline,200)

    def test_unknown_waits_cannot_pause_one_stage_without_bound(self):
        st=self.state();st.account_recovery_wait('wait',10,100)
        st.account_recovery_wait('wait',100,500)
        self.assertEqual(st.stage_deadline,150);self.assertEqual(st.recovery_pause_used,120)
        st.account_recovery_wait(None,501,600);self.assertEqual(st.stage_deadline,150)

    def test_reset_and_phase_change_drop_previous_pause_tracking(self):
        st=self.state();st.account_recovery_wait('wait',10,12);st.reset_after_return()
        self.assertIsNone(st.recovery_pause_at);self.assertEqual(st.recovery_pause_used,0)
        st.stage='mailbox';st.stage_deadline=60
        st.account_recovery_wait('wait',20,22);self.assertEqual(st.stage_deadline,62)
