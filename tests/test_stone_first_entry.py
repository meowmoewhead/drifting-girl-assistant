"""A moving first task is not dormant; an unselected gate cannot cast."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

from drift.tasks.stone_farm import StoneFarm
from tests import test_stone_farm as base

ROOT, image = base.ROOT, base.image


class FirstEntryTests(unittest.TestCase):
    setUpClass = classmethod(base.StoneFarmTests.setUpClass.__func__)
    setUp = base.StoneFarmTests.setUp
    tearDown = base.StoneFarmTests.tearDown
    act = base.StoneFarmTests.act
    tick = base.StoneFarmTests.tick
    begin = base.StoneFarmTests.begin
    action = base.StoneFarmTests.action

    def entry(self, phase='route_running'):
        self.begin(cycle=True)
        self.w.journal.update(verified_gate_stage=9,entered_verified_at=self.now,
                              first_recovery=True,fishing_recovery=True,
                              first_deadline=self.now+60)
        self.w._set(phase,180)
        return image('tests/fixtures/stone-farm/stage9-rank0-unselected-stopped.png')

    def test_real_moving_hook_waits_for_arrival_without_stopping_or_renewing_deadline(self):
        self.entry()
        frame=image('tests/fixtures/stone-farm/stage9-rank0-moving.png')
        self.assertTrue(self.w._moving(frame))
        self.assertTrue(self.w.training.boss_mode(frame))
        self.assertEqual(self.w.fishing.idle_icon(frame),'hook')
        self.assertIsNone(self.w._first_idle(frame,allow_running=True))
        deadline=self.w.journal['phase_deadline']
        for _ in range(8):self.tick(frame)
        self.assertEqual(self.w.phase,'route_running')
        self.assertEqual(self.w.journal['phase_deadline'],deadline)
        self.assertEqual(self.actions,[])
        # The numeral and countdown punctuation never identify movement.
        changed=frame.copy();changed[38:79,366:480]=0
        self.assertTrue(self.w._moving(changed))

    def test_real_unselected_gate_can_only_open_map_after_stopped_proof(self):
        frame=self.entry('first_map')
        self.w.journal['first_signature']=self.w.signature(frame)
        self.assertIsNotNone(self.w._first_unselected_idle(frame))
        self.assertIsNone(self.w._first_idle(frame))
        action=self.action('first-map',(350,30),frame)
        self.assertFalse(self.w.validate_portal(action,frame))
        self.w.off_verified=True
        self.assertTrue(self.w.validate_portal(action,frame))
        self.w.phase=self.w.journal['phase']='first_idle'
        self.w.journal.update(first_selected_rank=0,issued_inputs=[dict(kind='first-select')])
        self.w.first_idle_key=self.w._first_unselected_idle(frame)
        self.assertFalse(self.w.validate_portal(self.action('first-cast',(360,1010),frame),frame))
        self.w.phase=self.w.journal['phase']='fishing_prepare'
        self.assertFalse(self.w.meter_frame_allowed(frame))
        self.assertFalse(self.w.fishing.ordinary_preparation(frame,allow_fill=True))
        self.w.phase=self.w.journal['phase']='stop_fishing'
        self.assertFalse(self.w.validate_portal(self.action('handoff-cancel',(690,1090),frame),frame))

    def test_unselected_map_origin_rejects_unknown_world_rank_route_or_fishing(self):
        frame=self.entry('first_map')
        for owner,name,value in ((self.w,'_stage',8),(self.w,'_rank_value',1),
                                 (self.w,'_parallel',False),(self.w,'_off_icon',False),
                                 (self.w,'_moving',True),(self.w,'_battle',True),
                                 (self.w.sailing,'running',True),(self.w.sailing,'running',None),
                                 (self.w.fishing,'blue_fill',True),(self.w.fishing,'busy',True),
                                 (self.w.fishing,'at_port',True),(self.w.training,'boss_mode',True)):
            with self.subTest(name=name,value=value),patch.object(owner,name,return_value=value):
                self.assertIsNone(self.w._first_unselected_idle(frame))
        for bad in (frame//2,frame[:900]):
            self.assertIsNone(self.w._first_unselected_idle(bad))
        self.assertEqual(self.actions,[])

    def test_stopped_unselected_gate_enters_map_workflow_after_three_seconds(self):
        frame=self.entry('fishing_stop')
        for _ in range(2):self.tick(frame)
        self.assertEqual(self.w.phase,'fishing_stop')
        self.tick(frame)
        self.assertEqual(self.w.phase,'first_map')
        self.assertTrue(self.w.off_verified)
        self.assertTrue(self.w.journal['first_unselected'])
        self.assertEqual(self.actions,[])
        self.tick(frame)
        self.assertEqual([a['kind'] for a in self.actions],['first-map'])
        self.assertTrue(self.w.validate_portal(self.actions[-1],frame))

    def ordinary_entry(self, phase='fishing_stop'):
        self.entry(phase)
        return image('tests/fixtures/stone-farm/stage9-rank0-ordinary-stopped.png')

    def test_real_rank0_ordinary_origin_selects_task_instead_of_unreadable_meter(self):
        frame=self.ordinary_entry('first_map')
        self.w.off_verified=True
        self.w.journal['first_signature']=self.w.signature(frame)
        self.assertFalse(self.w._no_destination(frame))
        self.assertIsNone(self.w._first_unselected_idle(frame))
        self.assertIsNotNone(self.w._first_map_origin(frame))
        self.w.training.state=dict(mode='farming',rank=0)
        self.assertFalse(self.w.fishing.world_matches(frame))
        self.assertTrue(self.w.validate_portal(self.action('first-map',(350,30),frame),frame))
        self.assertFalse(self.w.validate_portal(self.action('first-cast',(360,1010),frame),frame))
        for owner,name,value in ((self.w,'_stage',8),(self.w,'_rank_value',1),
                                 (self.w,'_parallel',False),(self.w,'_off_icon',False),
                                 (self.w,'_moving',True),(self.w,'_battle',True),
                                 (self.w.sailing,'running',None),(self.w.sailing,'running',True),
                                 (self.w.fishing,'blue_fill',True),(self.w.fishing,'busy',True),
                                 (self.w.fishing,'at_port',True),(self.w.training,'boss_mode',True)):
            with self.subTest(name=name,value=value),patch.object(owner,name,return_value=value):
                self.assertIsNone(self.w._first_unselected_idle(frame,ordinary=True))
        self.assertIsNone(self.w._first_unselected_idle(frame//2,ordinary=True))
        self.assertEqual(self.actions,[])

    def test_real_ordinary_stop_reuses_task_selection_without_meter_input(self):
        frame=self.ordinary_entry()
        self.w.journal.pop('first_recovery')
        self.w.journal['attempts']={'route-start':1,'restore-stop':1}
        for _ in range(3):self.tick(frame)
        self.assertEqual(self.w.phase,'first_map')
        self.assertTrue(self.w.journal['first_unselected'])
        self.assertEqual(self.w.journal['attempts'],{'route-start':1,'restore-stop':1})
        self.assertNotIn('meter_attempts',self.w.journal)
        self.assertEqual(self.actions,[])
        self.tick(frame)
        self.assertEqual([x['kind'] for x in self.actions],['first-map'])

    def test_running_ordinary_gray_hook_is_handed_to_bounded_first_task(self):
        frame=self.ordinary_entry('route_running')
        self.w.journal.pop('first_recovery')
        deadline=self.w.journal['deadline']
        with patch.object(self.w.sailing,'running',return_value=True):
            for _ in range(3):self.tick(frame)
        self.assertEqual(self.w.phase,'fishing_stop')
        self.assertTrue(self.w.journal['first_recovery'])
        self.assertEqual(self.w.journal['deadline'],deadline)
        self.assertEqual(self.actions,[])

    def test_untouched_ordinary_prepare_rebinds_task_without_renewing_old_window(self):
        frame=self.ordinary_entry('fishing_prepare')
        self.w.off_verified=True
        self.w.journal['phase_deadline']=self.now+40
        old=self.w.journal['phase_deadline']
        for _ in range(3):self.tick(frame)
        self.assertEqual(self.w.phase,'first_map')
        self.assertEqual(self.w.journal['first_deadline'],old)
        self.assertLessEqual(self.w.journal['phase_deadline'],old)
        self.assertEqual(self.actions,[])
        self.now=old+1
        with self.assertRaises(RuntimeError):self.tick(frame)
        self.assertEqual(self.actions,[])

    def test_actual_startup_retry_admits_stopped_ordinary_origin_then_selects_task(self):
        frame=self.ordinary_entry('fishing_prepare')
        self.w.journal.pop('first_deadline')
        self.w.journal.update(phase_deadline=self.now-1,
            attempts={'route-start':1,'restore-stop':1,'enter-confirm':1},
            issued_inputs=[dict(kind='route-start'),dict(kind='restore-stop'),dict(kind='enter-confirm')])
        self.w._save()
        raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        self.assertEqual(StoneFarm.request_retry(self.path.parent,digest)['mode'],'startup_restore')
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        with patch.object(self.w,'_moving',return_value=True):
            self.tick(frame)
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual(self.actions,[])
        for _ in range(3):self.tick(frame)
        self.assertIsNone(self.w.retry)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertEqual(self.w.phase,'first_map')
        self.assertEqual(self.w.journal['attempts'],json.loads(raw)['attempts'])
        self.assertNotIn('meter_attempts',self.w.journal)
        self.assertEqual([x['kind'] for x in self.actions],['first-map'])

    def failed_entry(self):
        frame=self.entry('fishing_stop')
        self.w.journal.update(phase_deadline=self.now-1,
                              attempts={'route-start':1,'restore-stop':1},
                              issued_inputs=[dict(kind='route-start'),dict(kind='restore-stop')])
        self.w._save()
        raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        return frame,raw,digest

    def test_entry_repair_archives_once_retains_all_attempts_and_original_return_deadline(self):
        frame,raw,digest=self.failed_entry()
        old=deepcopy(self.w.journal)
        with self.assertRaises(ValueError):StoneFarm.request_retry(self.path.parent,'0'*64)
        self.assertEqual(StoneFarm.request_retry(self.path.parent,digest)['mode'],'first_entry_restore')
        self.assertEqual(self.path.read_bytes(),raw)
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        for _ in range(3):self.tick(frame)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertEqual(self.w.journal['attempts'],old['attempts'])
        self.assertEqual(self.w.journal['return_deadline'],old['return_deadline'])
        self.assertEqual(self.w.journal['first_entry_repair_runs'],1)
        self.assertAlmostEqual(self.w.journal['deadline']-self.w.journal['started_at'],180)
        self.assertAlmostEqual(self.w.journal['first_deadline']-self.w.journal['started_at'],60)
        self.assertTrue(all(a['kind']=='first-map' for a in self.actions))
        self.now+=181;self.w._save()
        with self.assertRaises(ValueError):StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_entry_repair_cannot_reset_prior_task_or_meter_input(self):
        _,_,_=self.failed_entry()
        saved=deepcopy(self.w.journal)
        for change in ({'issued_inputs':[dict(kind='route-start')]},
                       {'attempts':{'first-map':1}}, {'meter_attempts':{'start':1}},
                       {'first_entry_repair_runs':1}, {'verified_gate_stage':8}):
            self.w.journal={**deepcopy(saved),**change};self.w._save()
            with self.assertRaises(ValueError):StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_actual_selected_map_rebinds_location_only_after_matching_issued_skull(self):
        origin=self.entry('first_close');self.w.off_verified=True
        frame=image('tests/fixtures/stone-farm/stage9-rank0-selected-map.png')
        self.w.journal.update(first_unselected=True,first_signature=self.w.signature(origin),
                              first_selected_rank=0,first_selected_point=[377,444],
                              attempts={'first-select':1},issued_inputs=[dict(kind='first-select')])
        self.assertNotEqual(self.w.signature(frame),self.w.signature(origin))
        self.assertTrue(self.w._first_selected_map(frame))
        for _ in range(2):self.tick(frame)
        self.assertEqual(self.actions,[])
        self.tick(frame)
        self.assertEqual([a['kind'] for a in self.actions],['first-close'])
        self.assertTrue(self.w.validate_portal(self.actions[-1],frame))
        self.w.journal['first_selected_point']=[378,444]
        self.assertFalse(self.w.validate_portal(self.actions[-1],frame))
        self.w.journal['first_selected_point']=[377,444]
        self.w.journal['issued_inputs']=[]
        self.assertFalse(self.w.validate_portal(self.actions[-1],frame))

    def test_unselected_first_map_can_use_exposed_skull_without_authorising_a_cast(self):
        origin=self.entry('first_select');self.w.off_verified=True
        # Composite the two actual headers/map samples to isolate the missing
        # selection marker. This is a replay case, not a new live screenshot.
        frame=image('tests/fixtures/stone-farm/stage9-rank0-selected-map.png')
        frame[:80]=origin[:80]
        self.w.journal.update(first_unselected=True,first_signature=self.w.signature(frame))
        self.assertFalse(self.w.training.boss_mode(frame))
        self.assertTrue(self.w._first_map_ready(frame))
        self.assertEqual(self.w._map_task_target(frame)[:2],(377,444))
        self.w.journal['first_unselected']=False
        self.assertIsNone(self.w._map_task_target(frame))
        self.w.journal['first_unselected']=True
        for rows,cols in ((slice(438,455),slice(385,393)),(slice(455,461),slice(366,391))):
            damaged=frame.copy();damaged[rows,cols]=0
            self.assertIsNone(self.w._map_task_target(damaged))
        self.assertEqual(self.actions,[])

    def restored_route(self):
        self.entry('route_running')
        kinds=('return-request','return-confirm','activate','enter','enter-confirm',
               'first-cast','restore-route-start')
        self.w.journal.update(gate_closed=True,resumed_after_unverified_close=True,
                              restoration_only=True,
                              first_cast_verified_rank=1,first_cast_verified_at=self.now,
                              issued_inputs=[dict(kind=k) for k in kinds],
                              attempts={k:1 for k in kinds})
        return image('tests/fixtures/stone-farm/stage9-rank1-moving-filled.png')

    def test_real_fishing_fill_during_native_travel_can_finish_restoration(self):
        frame=self.restored_route()
        self.assertTrue(self.w._moving(frame))
        self.assertTrue(self.w.fishing.blue_fill(frame,main_known=True))
        for _ in range(2):self.tick(frame)
        self.assertTrue(self.w.active)
        self.tick(frame)
        self.assertFalse(self.w.active)
        self.assertEqual(self.w.journal['cycles'],0)
        self.assertEqual(self.w.journal['recoveries'],1)
        self.assertEqual(self.w.journal['restored_rank'],1)
        self.assertNotIn('route-start',self.w.journal['attempts'])
        self.assertEqual(self.actions,[])

    def test_route_restoration_requires_issued_restore_input_not_reserved_attempt(self):
        frame=self.restored_route()
        self.w.journal['issued_inputs']=[x for x in self.w.journal['issued_inputs']
                                         if x['kind']!='restore-route-start']
        self.w.journal['issued_inputs'].append(dict(kind='route-start'))
        self.assertEqual(self.w.journal['attempts']['restore-route-start'],1)
        with self.assertRaises(RuntimeError):
            for _ in range(3):self.tick(frame)
        self.assertTrue(self.w.active)
        self.assertEqual(self.w.journal['cycles'],0)
        self.assertEqual(self.actions,[])

    def test_normal_cycle_does_not_substitute_a_restore_route_receipt(self):
        frame=self.restored_route()
        self.w.journal['fishing_recovery']=False
        with self.assertRaises(RuntimeError):
            for _ in range(3):self.tick(frame)
        self.assertTrue(self.w.active)
        self.assertEqual(self.w.journal['cycles'],0)
        self.assertEqual(self.actions,[])

    def test_approved_route_restoration_is_not_a_complete_unattended_cycle(self):
        frame=self.restored_route()
        self.w.journal.pop('resumed_after_unverified_close')
        self.w.journal['issued_inputs'].append(dict(kind='close-gate'))
        old=deepcopy(self.w.journal)
        for _ in range(3):self.tick(frame)
        self.assertFalse(self.w.active)
        self.assertEqual(self.w.journal['cycles'],0)
        self.assertEqual(self.w.journal['recoveries'],1)
        self.assertEqual(self.w.journal['resolution'],'route_restored_after_approved_recovery')
        self.assertEqual(self.w.journal['attempts'],old['attempts'])
        self.assertEqual(self.w.journal['issued_inputs'],old['issued_inputs'])
        self.assertEqual(self.actions,[])

    def test_expired_route_result_only_reconciles_real_fishing_without_input_or_timer_reset(self):
        frame=self.restored_route()
        self.w.journal.update(phase_deadline=self.now-1,deadline=self.now-1)
        self.w._save();raw=self.path.read_bytes();old=deepcopy(self.w.journal)
        digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        self.assertEqual(StoneFarm.request_retry(self.path.parent,digest)['mode'],'route_result_restore')
        self.assertEqual(self.path.read_bytes(),raw)
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        with patch.object(self.w.fishing,'blue_fill',return_value=False),patch.object(self.w,'_battle',return_value=False):
            for _ in range(3):self.tick(frame)
        self.assertEqual(self.path.read_bytes(),raw)
        for _ in range(3):self.tick(frame)
        self.assertFalse(self.w.active)
        self.assertEqual(self.w.journal['attempts'],old['attempts'])
        for field in ('deadline','phase_deadline','return_deadline','started_at'):
            self.assertEqual(self.w.journal[field],old[field])
        self.assertEqual(self.w.journal['route_result_repair_runs'],1)
        self.assertEqual(self.w.journal['cycles'],0)
        self.assertEqual(self.w.journal['recoveries'],1)
        self.assertEqual(self.actions,[])
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        with self.assertRaises(ValueError):StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_route_result_reconciliation_rejects_wrong_stage_rank_menu_and_route(self):
        frame=self.restored_route()
        self.w.journal.update(phase_deadline=self.now-1,deadline=self.now-1)
        self.w._save();raw=self.path.read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        StoneFarm.request_retry(self.path.parent,digest)
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        actual=self.w.screen(frame)
        for change in ({'stage':8},{'rank':0},{'rank':39},{'rank':40},
                       {'menu':True},{'route':False},{'route':None}):
            with self.subTest(change=change),patch.object(self.w,'screen',return_value={**actual,**change}):
                for _ in range(3):self.tick(frame)
            self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual(self.actions,[])


if __name__=='__main__':unittest.main()
