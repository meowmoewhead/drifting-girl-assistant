"""Stage-10 native replays plus journal-bound 10 -> 9 -> 10 transitions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from drift.tasks.cruise_watchdog import CruiseView
from drift.tasks.parallel_world import ParallelWorld
from drift.tasks.stone_farm import StoneFarm
from tests import test_stone_farm as base


class AdaptiveStageTests(unittest.TestCase):
    setUpClass = classmethod(base.StoneFarmTests.setUpClass.__func__)
    setUp = base.StoneFarmTests.setUp
    tearDown = base.StoneFarmTests.tearDown
    act = base.StoneFarmTests.act
    tick = base.StoneFarmTests.tick
    action = base.StoneFarmTests.action

    def native(self, n=0):
        return base.image(f'tests/fixtures/stone-stage10/main-{n}.png')

    def adopt(self):
        for n in range(3):
            f = self.native(n); self.now += 1.6
            self.w.observe_stage_policy(f, self.w.screen(f))

    def begin10(self, *, cycle=False):
        self.w._begin(dict(stage=9 if cycle else 10, rank=40 if cycle else 39,
                           menu=False, completion=False, route=True))

    def test_three_actual_stage10_frames_adopt_without_touching_old_config_or_journal(self):
        self.assertEqual(self.w.START_STAGE, 9)
        for n in range(3):
            frame = self.native(n); screen = self.w.screen(frame)
            self.assertEqual((screen['stage'], screen['rank']), (10,34))
            self.now += 1.6
            self.assertEqual(self.w.observe_stage_policy(frame, screen), n == 2)
        self.assertEqual((self.w.START_STAGE,self.w.FINAL_STAGE), (10,9))
        self.assertEqual(self.w.status()['start_stage'],10)
        self.assertFalse(self.path.exists())
        self.assertEqual(self.actions, [])

    def test_duplicate_frame_resource_battle_and_dark_unknown_do_not_adopt(self):
        frame = self.native()
        for _ in range(4):
            self.now += 1.6
            self.w.observe_stage_policy(frame,self.w.screen(frame))
        self.assertEqual(self.w.START_STAGE,9)
        for bad in (frame//2,base.image('tests/fixtures/stone-stage10/resource-r51.png')):
            self.assertIsNone(self.w.screen(bad))
            self.w.observe_stage_policy(bad,None)
        self.assertEqual(self.w.START_STAGE,9)
        self.assertEqual(self.actions, [])

    def test_active_legacy_trip_is_not_reinterpreted_by_new_stage10_frames(self):
        self.w._begin(dict(stage=9,rank=39,menu=False,completion=False,route=True))
        raw=self.path.read_bytes();old=deepcopy(self.w.journal)
        for n in range(3):
            f=self.native(n);self.now+=1.6
            self.assertFalse(self.w.observe_stage_policy(f,self.w.screen(f)))
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual(self.w.journal,old)
        self.assertEqual((self.w.START_STAGE,self.w.FINAL_STAGE),(9,8))

    def test_rank39_stage10_trigger_uses_new_pair_and_stops_once(self):
        self.adopt()
        with patch.object(self.w,'screen',return_value=dict(stage=10,rank=39,menu=False,
                                                          completion=False,route=True)):
            for _ in range(3):self.tick(self.native().copy())
        self.assertEqual((self.w.journal['from_stage'],self.w.journal['to_stage']), (10,9))
        self.assertEqual([a['kind'] for a in self.actions],['stop'])
        self.assertTrue(self.w.ordinary_cancel_allowed)
        self.assertFalse(self.w.boss_abandon_handoff)

    def test_handoff_preempts_cruise_even_before_stage10_adoption(self):
        current=dict(stage=10,rank=39,menu=False,completion=False,route=False)
        self.assertTrue(self.w.handoff_due(current))
        with patch.object(self.w,'screen',return_value=current):
            self.assertIsNone(CruiseView(self.w).observe(self.native()))
            self.assertIsNone(CruiseView(self.w).observe_setup(self.native()))

    def test_new_stage10_rank38_does_not_trigger_a_portal_handoff(self):
        self.adopt()
        with patch.object(self.w,'screen',return_value=dict(stage=10,rank=38,menu=False,
                                                          completion=False,route=True)):
            for _ in range(3):self.tick(self.native().copy())
        self.assertFalse(self.w.active)
        self.assertEqual(self.actions,[])

    def test_reload_preserves_10_to_9_inputs_and_original_deadlines(self):
        self.begin10();self.w.journal['attempts']['stop']=1;self.w._save()
        raw=self.path.read_bytes()
        restored=StoneFarm(base.ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        self.assertEqual((restored.START_STAGE,restored.FINAL_STAGE),(10,9))
        self.assertEqual(restored.journal,self.w.journal)
        self.assertEqual(self.path.read_bytes(),raw)

    def test_nonadjacent_boolean_or_reversed_saved_pairs_are_rejected_unchanged(self):
        self.begin10();old=deepcopy(self.w.journal)
        for change in (dict(to_stage=8),dict(from_stage=True),dict(to_stage=11)):
            record={**old,**change};self.path.write_text(json.dumps(record),encoding='utf-8')
            raw=self.path.read_bytes()
            with self.assertRaisesRegex(RuntimeError,'记录无效'):
                StoneFarm(base.ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
            self.assertEqual(self.path.read_bytes(),raw)

    def test_completed_lower_pair_stays_10_to_9_at_rank39_without_double_lowering(self):
        self.begin10();self.w.journal.update(pending=False,phase='idle',verified_gate_stage=9,
                                           entered_verified_at=self.now)
        self.w.phase='idle';self.w._save()
        frame=self.main
        for _ in range(4):
            self.now+=1.6
            self.w.observe_stage_policy(frame.copy(),dict(stage=9,rank=39,menu=False,completion=False))
        self.assertEqual((self.w.START_STAGE,self.w.FINAL_STAGE),(10,9))
        self.assertFalse(self.w.handoff_due(dict(stage=9,rank=39)))
        self.assertTrue(self.w.handoff_due(dict(stage=9,rank=40)))

    def test_rank40_cycle_returns_to_10_and_has_no_boss_abandon_authorization(self):
        self.begin10(cycle=True)
        self.assertTrue(self.w.completed_rank_40)
        self.assertEqual((self.w.journal['from_stage'],self.w.journal['to_stage']),(9,10))
        self.assertTrue(self.w.ordinary_cancel_allowed)
        self.assertFalse(self.w.boss_abandon_handoff)
        self.assertEqual((self.w.START_STAGE,self.w.FINAL_STAGE),(10,9))

    def test_gate_and_confirmation_inputs_are_bound_to_10_and_9_not_old_constants(self):
        self.begin10();self.w._set('manager')
        frame=base.image('assets/recovery/parallel-manager.png')
        with patch.object(self.w,'_gate_stage',return_value=10):
            self.assertTrue(self.w.validate_portal(self.action('lower',(357,873),frame),frame))
        self.assertFalse(self.w.validate_portal(self.action('lower',(357,873),frame),frame))
        self.w._set('lower_confirm')
        question=base.image('assets/recovery/parallel-lower-question.png')
        with patch.object(self.w,'_lower_prompt',return_value=9):
            self.assertTrue(self.w.validate_portal(self.action('lower-confirm',(208,788),question),question))
        self.assertFalse(self.w.validate_portal(self.action('lower-confirm',(208,788),question),question))

    def test_actual_stage9_manager_is_only_closeable_in_completed_stage9_cycle(self):
        frame=base.image('assets/recovery/parallel-manager.png')
        self.begin10();self.w._set('manager')
        self.assertFalse(self.w.validate_portal(self.action('close-gate',(354,739),frame),frame))
        self.begin10(cycle=True);self.w._set('manager')
        self.assertTrue(self.w.validate_portal(self.action('close-gate',(354,739),frame),frame))

    def test_new_gate_waits_for_exact_10_not_9_or_unknown(self):
        self.begin10(cycle=True);self.w._set('new_gate')
        frame=base.image('assets/recovery/parallel-manager.png')
        for _ in range(3):self.tick(frame.copy())
        self.assertEqual(self.w.phase,'new_gate')
        with patch.object(self.w,'_gate_stage',return_value=10):
            for _ in range(3):self.tick(frame.copy())
        self.assertEqual(self.w.journal['verified_gate_stage'],10)
        self.assertEqual(self.w.phase,'panel_close')

    def test_two_digit_gate_fallback_needs_four_independent_strict_votes(self):
        frame=base.image('assets/recovery/parallel-manager.png')
        good=[SimpleNamespace(value=10,confidence=.99) for _ in range(4)]
        with patch.object(ParallelWorld,'_gate_stage',return_value=1), \
             patch('drift.tasks.stone_farm.read_numeric_views',return_value=good) as read:
            self.assertEqual(self.w._gate_stage(frame),10)
            self.assertEqual(len(read.call_args.args[1]),4)
        for readings in (good[:3],good[:3]+[SimpleNamespace(value=11,confidence=.99)],
                         good[:3]+[SimpleNamespace(value=10,confidence=.94)]):
            with patch.object(ParallelWorld,'_gate_stage',return_value=1), \
                 patch('drift.tasks.stone_farm.read_numeric_views',return_value=readings):
                self.assertIsNone(self.w._gate_stage(frame))
        self.assertEqual(self.actions,[])

    def test_old_single_digit_gate_uses_unchanged_reader(self):
        frame=base.image('assets/recovery/parallel-manager.png')
        with patch('drift.tasks.stone_farm.read_numeric_views') as fallback:
            self.assertEqual(self.w._gate_stage(frame),9)
            fallback.assert_not_called()

    def test_actual_stage10_managers_read_full_digits_in_four_views(self):
        from drift.tasks.parallel_world import ParallelWorld
        for n in range(4):
            frame=base.image(f'tests/fixtures/stone-stage10/manager-{n}.png')
            with self.subTest(frame=n):
                self.assertIsNone(ParallelWorld._gate_stage(self.w,frame))
                self.assertEqual(self.w._gate_stage(frame),10,self.w.gate_probe)
                self.assertEqual(len(self.w.gate_probe['views']),4)
                self.assertTrue(all(r['value']==10 and r['confidence']>=.95
                                    for r in self.w.gate_probe['views']))
        self.assertEqual(self.actions,[])

    def test_actual_manager_dark_missing_digit_or_other_screen_stays_unknown(self):
        frame=base.image('tests/fixtures/stone-stage10/manager-0.png')
        hidden=frame.copy();hidden[374:478,286:440]=0
        for candidate in (frame//2,hidden,self.native(),self.native()//2):
            with self.subTest(kind=int(candidate.sum())):
                self.assertIsNone(self.w._gate_stage(candidate))

    def test_unknown_manager_reports_number_votes_without_input_or_renewal(self):
        self.begin10();self.w._set('manager');deadline=self.w.journal['phase_deadline']
        frame=base.image('tests/fixtures/stone-stage10/manager-0.png');frame[374:478,286:440]=0
        self.tick(frame)
        self.assertEqual(self.w.journal['phase_deadline'],deadline)
        self.assertIn('复核',self.w._first_close_diagnostic())
        self.assertEqual(self.actions,[])

    def test_old_once_only_stage9_boss_abandon_is_not_granted_to_stage10(self):
        self.begin10();self.w.phase='stop_fishing'
        self.w.journal.update(phase='stop_fishing',return_deadline=self.now-1,
                              phase_deadline=self.now-1)
        self.w._save();raw=self.path.read_bytes()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(
            id='driftinggirl_stone_farm',enabled=True)])),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'放弃BOSS仅限'):
            StoneFarm.request_retry(self.path.parent,hashlib.sha256(raw).hexdigest(),allow_boss_abandon=True)
        self.assertEqual(self.path.read_bytes(),raw)


if __name__ == '__main__':
    unittest.main()
