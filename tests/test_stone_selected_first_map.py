"""Replay the 20:54 selected Seoul skull failure without repeating navigation."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from drift.tasks.stone_farm import StoneFarm
from tests import test_stone_farm as base


class SelectedFirstMapTests(unittest.TestCase):
    setUpClass=classmethod(base.StoneFarmTests.setUpClass.__func__)
    setUp=base.StoneFarmTests.setUp
    tearDown=base.StoneFarmTests.tearDown
    act=base.StoneFarmTests.act
    begin=base.StoneFarmTests.begin
    tick=base.StoneFarmTests.tick
    action=base.StoneFarmTests.action

    def selected(self):
        frame=base.image('tests/fixtures/stone-first-budget/selected-rank0-map.png')
        self.begin(cycle=True)
        kinds=('enter-confirm','first-map','first-select')
        self.w.journal.update(verified_gate_stage=9,entered_verified_at=self.now,
            first_unselected=True,first_recovery=True,fishing_recovery=True,
            first_deadline=self.now+60,first_selected_rank=0,first_selected_point=[416,444],
            first_signature=self.w.signature(frame),attempts={k:1 for k in kinds},
            issued_inputs=[{'kind':k} for k in kinds])
        self.w.off_verified=True
        self.w._set('first_close',45)
        return frame

    def failed(self):
        frame=self.selected()
        self.now+=61
        self.w._save()
        raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(
            rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        return frame,raw,digest

    def test_current_real_selected_map_requires_new_sample_at_unchanged_threshold(self):
        frame=self.selected()
        old=deepcopy(self.w.journal)
        with patch.object(self.w,'selected_first_map_core',self.w.first_map_core):
            self.assertFalse(self.w._first_selected_map(frame))
        with patch('drift.tasks.parallel_world.ParallelWorld._map_task_target',side_effect=AssertionError('global scan')):
            self.assertTrue(self.w._first_selected_map(frame))
        self.assertEqual(self.w.journal,old);self.assertEqual(self.actions,[])

    def test_original_variant_and_new_variant_both_require_two_complete_parts(self):
        self.selected()
        for path,point in (('tests/fixtures/stone-farm/stage9-rank0-selected-map.png',[377,444]),
                           ('tests/fixtures/stone-first-budget/selected-rank0-map.png',[416,444])):
            frame=base.image(path);self.w.journal['first_selected_point']=point
            self.assertTrue(self.w._first_selected_map(frame))
            x,y=point[0]-15,point[1]-16
            for r,c in ((slice(10,27),slice(23,31)),(slice(27,33),slice(4,29))):
                bad=frame.copy();bad[y+r.start:y+r.stop,x+c.start:x+c.stop]=0
                self.assertFalse(self.w._first_selected_map(bad))
        core=self.w.selected_first_map_core
        mask=np.zeros((33,31),bool);mask[10:27,23:31]=True;mask[27:33,4:29]=True
        self.assertTrue((core[~mask]==0).all())

    def test_three_current_map_frames_close_only_the_receipted_selection(self):
        frame=self.selected();old=deepcopy(self.w.journal)
        for _ in range(2):self.tick(frame.copy())
        self.assertEqual(self.actions,[])
        self.tick(frame.copy())
        self.assertEqual([a['kind'] for a in self.actions],['first-close'])
        self.assertTrue(self.w.validate_portal(self.actions[-1],frame.copy()))
        self.assertEqual(self.w.journal['first_deadline'],old['first_deadline'])
        self.assertEqual(self.w.journal['phase_deadline'],old['phase_deadline'])
        self.assertEqual(self.w.journal['attempts']['first-select'],1)

    def test_unknown_or_changed_map_context_never_closes(self):
        frame=self.selected()
        bad=frame.copy();bad[1136:1241,9:405]=0
        for image in (bad,frame//2,frame[:900],np.zeros_like(frame)):
            self.assertFalse(self.w._first_selected_map(image))
        for owner,name,value in ((self.w,'_parallel',False),(self.w,'_rank_value',1),
                (self.w,'_stage',8),(self.w,'_off_icon',False),(self.w.training,'boss_mode',False)):
            with self.subTest(name=name),patch.object(owner,name,return_value=value):
                self.assertFalse(self.w._first_selected_map(frame))
        for point in ([417,444],[400,444],[416],['416',444],[416.,444],[-1,444]):
            self.w.journal['first_selected_point']=point
            self.assertFalse(self.w._first_selected_map(frame))
        self.assertEqual(self.actions,[])

    def test_expired_map_does_not_resume_without_a_new_hash_bound_request(self):
        frame,raw,digest=self.failed()
        with self.assertRaises(RuntimeError):self.tick(frame)
        self.assertEqual(self.path.read_bytes(),raw);self.assertEqual(self.actions,[])
        with self.assertRaises(ValueError):StoneFarm.request_retry(self.path.parent,'0'*64)
        request=StoneFarm.request_retry(self.path.parent,digest)
        self.assertEqual(request['mode'],'first_close_restore')
        self.assertEqual(self.path.read_bytes(),raw)

    def test_separate_restore_retains_failed_window_attempts_and_selection(self):
        frame,raw,digest=self.failed();old=json.loads(raw)
        StoneFarm.request_retry(self.path.parent,digest)
        self.w=StoneFarm(base.ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        for _ in range(3):self.tick(frame.copy())
        self.assertIsNone(self.w.retry)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertEqual(self.w.journal['first_close_repair_runs'],1)
        self.assertEqual(self.w.journal['deadline']-self.now,180)
        self.assertEqual(self.w.journal['first_deadline']-self.now,60)
        for field in ('started_at','return_deadline','attempts','issued_inputs','first_selected_point'):
            self.assertEqual(self.w.journal[field],old[field])
        for field in ('deadline','phase_deadline','first_deadline','return_deadline'):
            self.assertEqual(self.w.journal['first_close_previous_deadlines'][field],old[field])
        self.assertEqual(self.actions,[])
        for _ in range(3):self.tick(frame.copy())
        self.assertEqual([a['kind'] for a in self.actions],['first-close'])
        self.now+=181;self.w._save()
        with self.assertRaises(ValueError):StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_restore_does_not_accept_used_close_cast_meter_or_a_second_window(self):
        _,_,_=self.failed();old=deepcopy(self.w.journal)
        changes=[{'first_close_repair_runs':1},{'first_selected_rank':1},{'first_selected_point':['416',444]},
                 {'verified_gate_stage':8},{'meter_attempts':{'start':1}},{'issued_inputs':[]},
                 {'issued_inputs':[*old['issued_inputs'],{'kind':'first-close'}]},
                 *[{'attempts':{**old['attempts'],kind:1}} for kind in
                   ('first-close','first-wake','first-cast','restore-route-open','restore-route-start')]]
        for change in changes:
            self.w.journal={**deepcopy(old),**change};self.w._save()
            with self.subTest(change=change),self.assertRaises(ValueError):
                StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(self.actions,[])

    def test_new_restore_waits_without_mutation_if_live_selection_is_not_confirmed(self):
        frame,raw,digest=self.failed();StoneFarm.request_retry(self.path.parent,digest)
        self.w=StoneFarm(base.ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        with patch.object(self.w,'_first_selected_map',return_value=False):
            for _ in range(4):self.tick(frame.copy())
        self.assertEqual(self.path.read_bytes(),raw);self.assertEqual(self.actions,[])
        self.now=self.w.retry['expires_at']+1
        with self.assertRaises(RuntimeError):self.tick(frame)
        self.assertEqual(self.path.read_bytes(),raw)

    def native_fishing(self):
        self.selected()
        frame=base.image('tests/fixtures/stone-first-result/rank1-20261017.png')
        self.w.journal['issued_inputs'].append({'kind':'first-close'})
        self.w.journal['attempts']['first-close']=1
        self.w.journal['first_signature']=self.w.signature(frame)
        return frame

    def test_user_closed_current_task_skips_map_and_uses_only_unused_cast(self):
        _,raw,digest=self.failed()
        frame=base.image('tests/fixtures/stone-first-budget/stopped-rank0-hook.png')
        StoneFarm.request_retry(self.path.parent,digest)
        self.w=StoneFarm(base.ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        for _ in range(3):self.tick(frame.copy())
        self.assertEqual(self.w.phase,'first_idle')
        self.assertTrue(self.w.journal['first_map_closed_before_recovery'])
        self.assertEqual(self.actions,[])
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        for _ in range(3):self.tick(frame.copy())
        self.assertEqual([a['kind'] for a in self.actions],['first-cast'])
        self.assertNotIn('first-close',self.w.journal['attempts'])
        self.assertEqual(self.w.journal['attempts']['first-select'],1)

    def test_closed_map_recovery_rejects_running_fishing_unknown_and_new_task(self):
        _,raw,digest=self.failed();StoneFarm.request_retry(self.path.parent,digest)
        self.w=StoneFarm(base.ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        frame=base.image('tests/fixtures/stone-first-budget/stopped-rank0-hook.png')
        for owner,name,value in ((self.w,'_rank_value',1),(self.w,'_stage',8),
                (self.w,'_off_icon',False),(self.w.sailing,'running',True),
                (self.w.fishing,'blue_fill',True),(self.w,'_battle',True)):
            with self.subTest(name=name),patch.object(owner,name,return_value=value):
                for _ in range(3):self.tick(frame.copy())
            self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual(self.actions,[])

    def test_new_dormant_world_starts_first_task_before_opening_patrol(self):
        self.begin(cycle=True);self.w.journal['verified_gate_stage']=9
        self.w._set('arrive',90)
        frame=base.image('tests/fixtures/stone-farm/stage9-rank0-unselected-stopped.png')
        for _ in range(3):self.tick(frame.copy())
        self.assertEqual(self.w.phase,'first_map')
        self.assertEqual(self.actions,[])
        self.assertNotIn('route-open',self.w.journal['attempts'])

    def test_existing_real_fishing_never_gets_a_redundant_first_cast(self):
        frame=self.native_fishing();old=deepcopy(self.w.journal)
        self.assertIsNotNone(self.w._selected_first_fishing(frame))
        for _ in range(3):self.tick(frame.copy())
        self.assertEqual(self.w.phase,'route_open')
        self.assertEqual(self.w.journal['selected_fishing_verified_rank'],1)
        self.assertNotIn('first_cast_verified_at',self.w.journal)
        self.assertEqual(self.w.journal['attempts'],old['attempts'])
        self.assertEqual(self.w.journal['issued_inputs'],old['issued_inputs'])
        self.assertEqual(self.w.journal['first_deadline'],old['first_deadline'])
        self.assertTrue(self.w.first_route_resume_requested)
        self.assertIsNotNone(self.w._expected(frame,entry=True))
        self.assertEqual(self.actions,[])

    def test_native_fishing_requires_real_fishing_and_fresh_stopped_same_place(self):
        frame=self.native_fishing()
        for owner,name,value in ((self.w,'_parallel',False),(self.w,'_off_icon',False),
                (self.w,'_moving',True),(self.w.sailing,'running',True),
                (self.w.fishing,'at_port',True),(self.w,'signature',['changed']),
                (self.w,'screen',{'stage':8,'rank':1,'menu':False,'route':False})):
            with self.subTest(name=name),patch.object(owner,name,return_value=value):
                self.assertIsNone(self.w._selected_first_fishing(frame))
        with patch.object(self.w.fishing,'blue_fill',return_value=False),patch.object(self.w.defense,'_battle_footer',return_value=False):
            self.assertIsNone(self.w._selected_first_fishing(frame))
        for bad in (frame//2,frame[:900],base.image('tests/fixtures/stone-first-result/popup-20261017.png')):
            self.assertIsNone(self.w._selected_first_fishing(bad))
        self.w.journal['issued_inputs']=[{'kind':'first-select'}]
        self.assertIsNone(self.w._selected_first_fishing(frame))
        self.assertEqual(self.actions,[])

    def test_same_native_frame_and_expired_window_cannot_commit_fishing(self):
        frame=self.native_fishing()
        for _ in range(3):self.tick(frame)
        self.assertEqual(self.w.phase,'first_close');self.assertEqual(self.w.seen,1)
        self.w._save();raw=self.path.read_bytes()
        self.now=self.w.journal['phase_deadline']+1
        with self.assertRaises(RuntimeError):self.tick(frame.copy())
        self.assertEqual(self.path.read_bytes(),raw);self.assertEqual(self.actions,[])


if __name__=='__main__':unittest.main()
