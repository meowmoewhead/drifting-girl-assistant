"""Observed sidebar shift and bounded return repair must retain old budgets."""
from copy import deepcopy
import hashlib
import json
import threading
import unittest
from unittest.mock import patch

import numpy as np

from drift.runtime import Clicker
from drift.tasks.stone_farm import StoneFarm
from tests import test_stone_farm as base


class ReturnEntryTests(unittest.TestCase):
    setUpClass=classmethod(base.StoneFarmTests.setUpClass.__func__)
    setUp=base.StoneFarmTests.setUp
    tearDown=base.StoneFarmTests.tearDown
    begin=base.StoneFarmTests.begin
    tick=base.StoneFarmTests.tick
    act=base.StoneFarmTests.act
    action=base.StoneFarmTests.action

    def shifted(self):
        return base.image('tests/fixtures/stone-farm/return-shifted-stopped-20261018.png')

    def prepare_return(self):
        self.begin()
        self.w._set('return',180)
        self.w.off_verified=self.w.fishing_stopped=True

    def test_real_shifted_icon_and_old_layouts_keep_exact_threshold(self):
        for name,point in [('return-shifted-stopped-20261018.png',(677,758)),
                           ('stage9-rank39-0.png',(677,848)),
                           ('stage8-rank40-return-badge.png',(677,938))]:
            frame=base.image('tests/fixtures/stone-farm/'+name)
            self.assertEqual(self.w._return_button(frame),point,name)
            self.assertGreaterEqual(self.w.return_probe['score'],.96)
            self.assertEqual(self.w.return_probe['candidates'],1)

    def test_current_frame_target_is_required_after_shift(self):
        frame=self.shifted();self.prepare_return()
        self.assertEqual(self.w.screen(frame),dict(stage=9,rank=39,menu=False,completion=False,route=False))
        self.assertTrue(self.w.validate_portal(self.action('return-request',(677,758),frame),frame))
        self.assertFalse(self.w.validate_portal(self.action('return-request',(677,848),frame),frame))
        changed=frame.copy()
        changed[727:790,643:711]=0
        self.assertFalse(self.w.validate_portal(self.action('return-request',(677,758),frame),changed))

    def test_duplicate_missing_and_invalid_size_icons_are_rejected(self):
        frame=self.shifted()
        duplicate=frame.copy()
        duplicate[817:880,643:711]=frame[727:790,643:711]
        self.assertIsNone(self.w._return_button(duplicate))
        self.assertEqual(self.w.return_probe['candidates'],2)
        self.assertIsNone(self.w._return_button(np.zeros_like(frame)))
        self.assertIsNone(self.w._return_button(frame[:1000]))

    def test_return_still_requires_stop_idle_world_rank_and_deadline(self):
        frame=self.shifted();self.prepare_return()
        action=self.action('return-request',(677,758),frame)
        self.w.off_verified=False
        self.assertFalse(self.w.validate_portal(action,frame))
        self.w.off_verified=True;self.w.fishing_stopped=False
        self.assertFalse(self.w.validate_portal(action,frame))
        self.w.fishing_stopped=True
        for owner,name,value in [(self.w.sailing,'running',True),
                                 (self.w,'_idle_fishing',False),
                                 (self.w,'_parallel',False)]:
            with patch.object(owner,name,return_value=value):
                self.assertFalse(self.w.validate_portal(action,frame),name)
        dim=frame//2
        self.assertFalse(self.w.validate_portal(self.action('return-request',(677,758),dim),dim))
        self.w.journal['return_deadline']=self.now-1
        self.assertFalse(self.w.validate_portal(action,frame))

    def test_missing_icon_logs_reason_without_changing_window_or_input_count(self):
        frame=self.shifted();self.prepare_return()
        frame[727:790,643:711]=0
        before=deepcopy(self.w.journal);events=[];self.w.event=events.append
        for _ in range(12):self.tick(frame.copy())
        self.assertEqual(self.actions,[])
        self.assertEqual(self.w.journal,before)
        self.assertEqual(len(events),2)
        self.assertIn('唯一返回入口',self.w.hold)
        self.now=before['return_deadline']+1
        with self.assertRaisesRegex(RuntimeError,'(到期|超过180秒).*唯一返回入口'):
            self.tick(frame)

    def failed_return(self):
        self.prepare_return()
        self.w.journal.update(attempts={'stop':1,'handoff-cancel':1},
            issued_inputs=[dict(kind='stop',input_id='old-stop'),
                           dict(kind='handoff-cancel',input_id='old-cancel')],
            off_verified_at=self.now,fishing_stopped_at=self.now)
        self.now+=181;self.w._save()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[
            dict(id='driftinggirl_stone_farm',enabled=True)])),encoding='utf-8')
        raw=self.path.read_bytes()
        return raw,hashlib.sha256(raw).hexdigest()

    def test_explicit_repair_archives_exact_bytes_and_retains_used_inputs(self):
        raw,digest=self.failed_return();old=json.loads(raw)
        approval=StoneFarm.request_retry(self.path.parent,digest,allow_return_entry_restore=True)
        self.assertEqual(approval['mode'],'return_entry_restore')
        self.assertEqual(self.path.read_bytes(),raw)
        self.w=StoneFarm(base.ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        actual=dict(stage=9,rank=39,menu=False,completion=False,route=True)
        with patch.object(self.w,'_expected',return_value=actual):
            self.tick();self.tick()
            self.assertEqual(self.path.read_bytes(),raw)
            self.tick()
        self.assertEqual(self.actions,[])
        self.assertEqual(self.w.phase,'stop')
        self.assertFalse(self.w.off_verified)
        self.assertFalse(self.w.fishing_stopped)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        for key in ('started_at','attempts','issued_inputs','cycles','handoffs'):
            self.assertEqual(self.w.journal[key],old[key],key)
        self.assertEqual(self.w.journal['return_entry_previous_deadlines'],
                         {k:old[k] for k in ('deadline','phase_deadline','return_deadline')})
        self.assertEqual(self.w.journal['return_deadline']-self.w.journal['return_entry_repair_started_at'],180)
        self.assertEqual(self.w.journal['return_entry_repair_runs'],1)
        # Even this separately approved repair cannot replenish a used cancel.
        self.w.phase='stop_fishing';self.w.journal['phase']='stop_fishing'
        action=self.action('handoff-cancel',(690,1090))
        self.w.reserve_input(action)
        with self.assertRaises(RuntimeError):self.w.reserve_input(action)

    def test_repair_rejects_unknown_scene_and_cannot_repeat_or_add_boss_permission(self):
        raw,digest=self.failed_return()
        with self.assertRaises(ValueError):
            StoneFarm.request_retry(self.path.parent,digest,allow_return_entry_restore=True,allow_boss_abandon=True)
        StoneFarm.request_retry(self.path.parent,digest,allow_return_entry_restore=True)
        self.w=StoneFarm(base.ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        with patch.object(self.w,'_expected',return_value=None):
            for _ in range(4):self.tick()
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual(self.actions,[])
        self.w.journal['return_entry_repair_runs']=1;self.w._save()
        with self.assertRaises(ValueError):
            StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest(),
                                    allow_return_entry_restore=True)

    def test_repair_does_not_apply_to_return_already_clicked_or_without_stop_proof(self):
        raw,_=self.failed_return();old=json.loads(raw)
        for changed in (dict(attempts={'return-request':1}),dict(off_verified_at=None),
                        dict(fishing_stopped_at=None),dict(phase='stop_fishing')):
            self.path.write_text(json.dumps(dict(old,**changed)),encoding='utf-8')
            with self.assertRaises(ValueError):
                StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest(),
                                        allow_return_entry_restore=True)

    def test_endpoint_requires_stopped_worker_hash_and_explicit_repair_flag(self):
        raw,digest=self.failed_return()
        app=object.__new__(Clicker);app.root=self.path.parent
        app.lock=threading.RLock();app.event=lambda _:None;app.running=lambda:True
        payload=dict(failed_sha256=digest,allow_ordinary_cancel=True,allow_return_entry_restore=True)
        with self.assertRaises(ValueError):app.retry_stone_farm(payload)
        app.running=lambda:False
        with self.assertRaises(ValueError):app.retry_stone_farm(dict(payload,allow_return_entry_restore=False))
        result=app.retry_stone_farm(payload)
        self.assertEqual(result['mode'],'return_entry_restore')
        self.assertEqual(self.path.read_bytes(),raw)


if __name__=='__main__':unittest.main()
