"""A lowered gate can arrive unselected with continuous fishing stopped."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

from tests import test_stone_farm as base


class LowerEntryTests(unittest.TestCase):
    setUpClass = classmethod(base.StoneFarmTests.setUpClass.__func__)
    setUp = base.StoneFarmTests.setUp
    tearDown = base.StoneFarmTests.tearDown
    act = base.StoneFarmTests.act
    tick = base.StoneFarmTests.tick
    begin = base.StoneFarmTests.begin
    action = base.StoneFarmTests.action

    def entry(self, *, entered=True):
        self.begin()
        self.w.journal.update(verified_gate_stage=8,
            attempts={'stop':1,'handoff-cancel':1,'enter-confirm':1},
            issued_inputs=[dict(kind=k) for k in ('stop','handoff-cancel','enter-confirm')])
        if entered:
            self.w.journal['entered_verified_at']=self.now
        self.w._set('arrive',90)
        return base.image('tests/fixtures/stone-lower-entry/unselected-stopped.png')

    def test_real_lowered_unselected_scene_only_authorizes_map_origin(self):
        frame=self.entry()
        self.assertEqual(self.w.screen(frame),dict(stage=8,rank=39,menu=False,completion=False,route=False))
        self.assertTrue(self.w._no_destination(frame))
        self.assertFalse(self.w.training.boss_mode(frame))
        self.assertEqual(self.w.fishing.idle_icon(frame),'hook')
        self.assertIsNotNone(self.w._first_unselected_idle(frame))
        self.assertIsNone(self.w._first_unselected_idle(frame,ordinary=True))
        self.assertIsNone(self.w._first_idle(frame))
        self.assertEqual(self.actions,[])

    def test_arrival_selects_real_task_before_any_route_or_fishing_input(self):
        frame=self.entry(entered=False)
        old=deepcopy(self.w.journal)
        self.assertIsNone(self.w._first_unselected_idle(frame))
        self.tick(frame);self.tick(frame)
        self.assertEqual(self.w.phase,'arrive')
        self.tick(frame)
        self.assertEqual(self.w.phase,'first_map')
        self.assertTrue(self.w.off_verified)
        self.assertTrue(self.w.journal['first_unselected'])
        self.assertEqual(self.actions,[])
        self.assertEqual(self.w.journal['deadline'],old['deadline'])
        self.assertEqual(self.w.journal['attempts'],old['attempts'])
        self.assertEqual(self.w.journal['issued_inputs'],old['issued_inputs'])
        self.assertNotIn('meter_attempts',self.w.journal)
        self.assertLessEqual(self.w.journal['first_deadline'],old['phase_deadline'])
        self.assertLessEqual(self.w.journal['first_deadline'],self.now+60)
        self.tick(frame)
        self.assertEqual([x['kind'] for x in self.actions],['first-map'])
        self.assertTrue(self.w.validate_portal(self.actions[-1],frame))

    def test_unselected_scene_never_authorizes_cast_meter_hold_or_cancel(self):
        frame=self.entry()
        self.w.off_verified=True
        self.w.journal.update(first_signature=self.w.signature(frame),
            first_selected_rank=39,first_deadline=self.now+60)
        self.w.journal['issued_inputs'].append(dict(kind='first-select'))
        self.w.phase=self.w.journal['phase']='first_idle'
        self.w.first_idle_key=self.w._first_unselected_idle(frame)
        self.assertFalse(self.w.validate_portal(self.action('first-cast',(360,1010),frame),frame))
        self.w.phase=self.w.journal['phase']='fishing_prepare'
        self.assertFalse(self.w.meter_frame_allowed(frame))
        self.w.phase=self.w.journal['phase']='stop_fishing'
        self.assertFalse(self.w.validate_portal(self.action('handoff-cancel',(690,1090),frame),frame))
        self.assertEqual(self.actions,[])

    def test_every_new_scene_guard_is_still_required(self):
        frame=self.entry()
        for owner,name,value in ((self.w,'_stage',9),(self.w,'_rank_value',38),
                (self.w,'_parallel',False),(self.w,'_off_icon',False),
                (self.w,'_moving',True),(self.w,'_battle',True),
                (self.w,'_no_destination',False),(self.w.sailing,'running',True),
                (self.w.sailing,'running',None),(self.w.fishing,'blue_fill',True),
                (self.w.fishing,'busy',True),(self.w.fishing,'at_port',True),
                (self.w.training,'boss_mode',True),(self.w.fishing,'selected',None)):
            with self.subTest(name=name,value=value),patch.object(owner,name,return_value=value):
                self.assertIsNone(self.w._first_unselected_idle(frame))
        for bad in (frame//2,frame[:900]):
            self.assertIsNone(self.w._first_unselected_idle(bad))
        self.assertEqual(self.actions,[])

    def test_first_task_deadline_stays_inside_remaining_arrival_window(self):
        frame=self.entry(entered=False)
        self.w.journal['phase_deadline']=self.now+18
        old=self.w.journal['phase_deadline']
        for _ in range(3):self.tick(frame)
        self.assertEqual(self.w.phase,'first_map')
        self.assertEqual(self.w.journal['first_deadline'],old)
        self.assertEqual(self.w.journal['phase_deadline'],old)
        self.now=old+1
        raw=self.path.read_bytes()
        with self.assertRaises(RuntimeError):self.tick(frame)
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual(self.actions,[])

    def test_new_travel_bounds_are_fixed_inside_total_and_old_records_keep_sixty_seconds(self):
        frame=self.entry()
        old_total=self.w.journal['deadline']
        self.w._prepare_first_task(frame)
        travel=self.w.journal['first_travel_deadline'];cast=self.w.journal['first_cast_deadline']
        self.assertLessEqual(travel,old_total);self.assertLessEqual(cast,old_total)
        self.assertGreater(travel,self.w.journal['first_deadline'])
        self.now+=120
        self.assertEqual(self.w._first_phase_limit('first_close'),travel)
        self.assertEqual(self.w._first_phase_limit('first_idle'),cast)
        self.assertEqual(self.w.journal['deadline'],old_total)
        self.w.journal.pop('first_travel_deadline');self.w.journal.pop('first_cast_deadline')
        self.assertEqual(self.w._first_phase_limit('first_close'),self.w.journal['first_deadline'])
        self.assertEqual(self.w._first_phase_limit('first_idle'),self.w.journal['first_deadline'])

    def test_expired_arrival_is_not_reopened_by_new_branch(self):
        frame=self.entry()
        self.w.journal['phase_deadline']=self.now-1
        self.w._save()
        raw=self.path.read_bytes()
        with self.assertRaises(RuntimeError):self.tick(frame)
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual(self.actions,[])

    def test_existing_fishing_skips_new_first_cast_recovery(self):
        frame=self.entry(entered=False)
        with patch.object(self.w.fishing,'blue_fill',return_value=True):
            for _ in range(3):self.tick(frame)
        self.assertEqual(self.w.phase,'route_open')
        self.assertNotIn('first_deadline',self.w.journal)
        self.assertNotIn('meter_attempts',self.w.journal)
        self.assertEqual(self.actions,[])

    def unselected_map(self):
        origin = self.entry()
        frame = base.image('tests/fixtures/stone-farm/stage8-rank39-selected-map.png')
        # Compose the two real rasters to isolate the unselected header from
        # the exposed skull. This is a regression fixture, not live evidence.
        frame[:79] = origin[:79]
        self.w.journal.update(first_unselected=True,first_signature=self.w.signature(frame),
                              first_deadline=self.now+60)
        self.w._set('first_select')
        return frame

    def test_lowered_unselected_map_can_find_exposed_skull(self):
        frame = self.unselected_map()
        self.assertFalse(self.w.training.boss_mode(frame))
        self.assertTrue(self.w._first_map_ready(frame))
        self.assertEqual(self.w._map_task_target(frame)[:2],(197,485))
        self.assertEqual(self.actions,[])

    def test_unselected_skull_still_requires_verified_lowered_context(self):
        frame = self.unselected_map()
        for key,value in [('first_unselected',False),('verified_gate_stage',9),
                          ('entered_verified_at',None),('entry_rank',38)]:
            old = self.w.journal.get(key)
            self.w.journal[key] = value
            with self.subTest(key=key):
                self.assertIsNone(self.w._map_task_target(frame))
            self.w.journal[key] = old
        for method,value in [('_no_destination',False),('_off_icon',False),('_stage',9)]:
            with self.subTest(method=method),patch.object(self.w,method,return_value=value):
                self.assertIsNone(self.w._map_task_target(frame))

    def test_select_wait_searches_map_without_extending_deadlines(self):
        frame = self.unselected_map()
        before = deepcopy(self.w.journal)
        events = []
        self.w.event = events.append
        with patch.object(self.w,'_map_task_target',return_value=None):
            for _ in range(3):
                self.tick(frame)
        self.assertEqual(self.w.journal,before)
        self.assertEqual([a['kind'] for a in self.actions],['first-pan'])
        self.assertEqual(self.actions[0]['swipe'],[610,600,400])
        self.assertTrue(any('骨面和红色下颌' in line for line in events))
        self.assertIn('选择当前任务',self.w._first_close_diagnostic())

    def test_map_search_requires_fresh_stopped_map_and_missing_target(self):
        frame=self.unselected_map();self.w.off_verified=True
        action=self.action('first-pan',(120,600),frame);action['swipe']=[610,600,400]
        with patch.object(self.w,'_map_task_target',return_value=None):
            self.assertTrue(self.w.validate_portal(action,frame))
            for owner,name,value in ((self.w,'_parallel',False),(self.w,'_off_icon',False),
                                     (self.w,'_rank_value',38),(self.w,'_stage',9)):
                with self.subTest(name=name),patch.object(owner,name,return_value=value):
                    self.assertFalse(self.w.validate_portal(action,frame))
            bad={**action,'swipe':[120,600,1000]}
            self.assertFalse(self.w.validate_portal(bad,frame))
        self.assertFalse(self.w.validate_portal(action,frame))

    def test_map_search_limit_never_resets_used_attempts_or_deadlines(self):
        frame=self.unselected_map()
        self.w.journal['attempts']['first-pan']=16
        old=deepcopy(self.w.journal)
        with patch.object(self.w,'_map_task_target',return_value=None):
            for _ in range(4):self.tick(frame)
        self.assertEqual(self.actions,[])
        self.assertEqual(self.w.journal,old)

    def test_bounded_map_recovery_accepts_same_unselected_origin_and_preserves_usage(self):
        frame = self.entry()
        self.w.journal.update(phase='first_select',phase_deadline=self.now-1,
            first_deadline=self.now-1,first_unselected=True,
            first_signature=self.w.signature(frame))
        self.w.phase = 'first_select'
        self.w.journal['attempts']['first-map'] = 1
        self.w.journal['issued_inputs'].append(dict(kind='first-map'))
        self.w._save()
        raw = self.path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[
            dict(id='driftinggirl_stone_farm',enabled=True)])))
        base.StoneFarm.request_retry(self.path.parent,digest)
        self.w = base.StoneFarm(base.ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        for _ in range(3):
            self.tick(frame)
        self.assertEqual(self.w.phase,'first_map')
        self.assertEqual(self.w.journal['attempts']['first-map'],1)
        self.assertEqual(self.w.journal['first_map_repair_runs'],1)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertEqual([a['kind'] for a in self.actions],['first-map'])
        self.assertTrue(self.w.validate_portal(self.actions[-1],frame))

    def real_covered_map(self):
        self.unselected_map()
        frame = base.image('tests/fixtures/stone-lower-entry/unselected-map-eye-covered.png')
        self.w.journal['first_signature'] = self.w.signature(frame)
        return frame

    def test_real_eye_covered_map_matches_bone_and_jaw_without_avatar(self):
        frame = self.real_covered_map()
        self.assertTrue(self.w._first_map_ready(frame))
        self.assertEqual(self.w._map_task_target(frame)[:2],(197,485))
        changed = frame.copy()
        changed[450:491,160:214] = 0
        self.assertEqual(self.w._map_task_target(changed)[:2],(197,485))
        self.assertEqual(self.actions,[])

    def test_real_bone_path_requires_both_shapes_and_current_context(self):
        frame = self.real_covered_map()
        for rows in [slice(491,496),slice(496,502)]:
            bad = frame.copy()
            bad[rows,186:211] = 0
            self.assertIsNone(self.w._map_task_target(bad))
        for method,value in [('_stage',9),('_rank_value',38),('_off_icon',False)]:
            with self.subTest(method=method),patch.object(self.w,method,return_value=value):
                self.assertIsNone(self.w._map_task_target(frame))
        self.assertIsNone(self.w._map_task_target(frame//2))

    def test_target_repair_archives_once_and_cannot_repeat_selection(self):
        frame = self.real_covered_map()
        self.w.journal.update(phase_deadline=self.now-1,first_deadline=self.now-1,
            first_map_repair_runs=1,attempts={'first-map':2},
            issued_inputs=[dict(kind='first-map')])
        self.w._save()
        raw = self.path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[
            dict(id='driftinggirl_stone_farm',enabled=True)])))
        base.StoneFarm.request_retry(self.path.parent,digest,allow_map_target_repair=True)
        self.w = base.StoneFarm(base.ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        for _ in range(3):
            self.tick(frame)
        self.assertEqual(self.w.journal['attempts'],{'first-map':2})
        self.assertEqual(self.w.journal['first_map_target_repair_runs'],1)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.now += 61
        with self.assertRaises(ValueError):
            base.StoneFarm.request_retry(self.path.parent,
                hashlib.sha256(self.path.read_bytes()).hexdigest(),allow_map_target_repair=True)


if __name__=='__main__':
    unittest.main()
