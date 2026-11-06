"""Cancelled fishing must be restored with the original remaining input budget."""
from copy import deepcopy
from contextlib import ExitStack
from unittest.mock import patch
import unittest
import numpy as np
from PIL import Image

from tests import test_meter_calibration as base
from drift.tasks.meter_limits import MeterWorkflowLimit


class ContinuationTests(unittest.TestCase):
    setUpClass = classmethod(base.MeterCalibrationTests.setUpClass.__func__)
    setUp = base.MeterCalibrationTests.setUp
    tearDown = base.MeterCalibrationTests.tearDown
    pending = base.MeterCalibrationTests.pending
    restart = base.MeterCalibrationTests.restart

    def current(self):
        c = self.pending()
        c.journal.update(route_policy='preserve', calibration_context='old',
                         attempts={'cancel': 1})
        c.route_verified = True
        c._save()
        return c

    def views(self, c, *, fill=False, task=True):
        stack = ExitStack()
        for target, name, value in (
                (c.world, '_parallel', True), (c.world, '_rank_value', 38),
                (c, '_stage', 8), (c.world.training, 'boss_mode', task),
                (c.sailing, 'running', True), (c.sailing, 'menu', False),
                (c.fishing, '_main_visible', True), (c.fishing, 'at_port', False),
                (c.fishing, '_context', 'new-task'), (c.fishing, 'blue_fill', fill),
                (c.fishing, 'real_battle', False)):
            stack.enter_context(patch.object(target, name, return_value=value))
        return stack

    def test_task_arrival_after_cancel_switches_without_any_drag_or_hold(self):
        c = self.current()
        before = deepcopy(c.journal)
        with self.views(c), patch('meter_calibration.time.time', return_value=1020), \
             patch.object(c.fishing, 'resume_cast_data', return_value=None), \
             patch.object(c.fishing, 'prepare', side_effect=AssertionError('no adjustment')):
            c.step(base.frame('sleep').copy())
        self.assertEqual(c.phase, 'restore_fishing')
        self.assertEqual(c.journal['deadline'], before['deadline'])
        self.assertEqual(c.journal['attempts'], before['attempts'])
        self.assertEqual(self.inputs, [])

    def test_last_thirty_seconds_are_reserved_for_restoring_not_more_adjustment(self):
        c = self.current()
        with patch('meter_calibration.time.time', return_value=1060):
            for action in ('scan_cancel', 'cancel', 'drag', 'reveal'):
                self.assertFalse(c.reserve_meter_input(action))
        self.assertEqual(c.journal['attempts'], {'cancel': 1})
        with self.views(c, task=False), patch('meter_calibration.time.time', return_value=1060), \
             patch.object(c.fishing, 'resume_cast_data', return_value=None):
            c.step(base.frame('sleep').copy())
        self.assertEqual(c.phase, 'restore_fishing')
        self.assertEqual(c.journal['deadline'], 1090)

    def test_gray_hook_three_seconds_one_short_cast_shares_start_budget(self):
        c = self.current()
        data = dict(context='new-task', digit_strokes='digits', world_strokes='stage',
                    box=(210,940,270,980), value=1200)
        with patch('meter_calibration.time.time', return_value=1020):
            c._begin_fishing_restore('任务地点变化')
        def issue(action, expected):
            self.assertEqual(action, 'resume_cast')
            self.assertTrue(c.reserve_meter_input(action))
            self.inputs.append((action, expected))
            return True
        with self.views(c), patch('meter_calibration.time.time', return_value=1020), \
             patch.object(c.fishing, 'resume_cast_data', return_value=data), \
             patch.object(c.fishing, 'action', side_effect=issue):
            for now in (100,101.5,103,104,105.5,107):
                with patch('meter_calibration.time.monotonic', return_value=now):
                    c.step(base.frame('sleep').copy())
        self.assertEqual(len(self.inputs), 1)
        self.assertEqual(c.journal['attempts'], {'cancel': 1, 'start': 1})
        self.assertTrue(c.active)
        self.assertEqual(c.cycles, 0)
        restarted = self.restart(c.journal, now=1025)
        self.assertEqual(restarted.phase, 'restore_fishing')
        with patch('meter_calibration.time.time', return_value=1025):
            self.assertFalse(restarted.reserve_meter_input('resume_cast'))
        self.assertEqual(restarted.journal['deadline'], 1090)

    def test_fresh_route_rank_stage_context_or_port_change_rejects_recovery_input(self):
        c = self.current()
        with patch('meter_calibration.time.time', return_value=1020):
            c._begin_fishing_restore('任务地点变化')
        c.restore_cast_context = 'new-task'
        with self.views(c), patch('meter_calibration.time.time', return_value=1020), \
             patch.object(c.fishing, 'resume_cast_data', return_value={'context':'new-task'}):
            self.assertTrue(c.resume_frame_allowed(base.frame('sleep')))
            for target,name,value in (
                (c.sailing,'running',False), (c.sailing,'running',None),
                (c.world,'_rank_value',39), (c,'_stage',9),
                (c.fishing,'_context','elsewhere'), (c.fishing,'at_port',True),
                (c.fishing,'_main_visible',False), (c.world,'_parallel',False)):
                with self.subTest(name=name), patch.object(target,name,return_value=value):
                    self.assertFalse(c.resume_frame_allowed(base.frame('sleep')))

    def test_used_start_or_last_eight_seconds_never_replenishes_budget(self):
        for attempts,now in (({'cancel':1,'start':1},1020), ({'cancel':1},1083)):
            c = self.current()
            c.journal['attempts'] = deepcopy(attempts)
            with patch('meter_calibration.time.time',return_value=now):
                c._begin_fishing_restore('恢复')
                self.assertFalse(c.reserve_meter_input('resume_cast'))
            self.assertEqual(c.journal['attempts'], attempts)

    def test_three_real_fill_frames_release_as_restored_not_calibrated(self):
        c = self.current()
        with patch('meter_calibration.time.time',return_value=1020):
            c._begin_fishing_restore('任务地点变化')
        with self.views(c,fill=True), patch('meter_calibration.time.time',return_value=1020):
            for now in (100,101.5,103):
                with patch('meter_calibration.time.monotonic',return_value=now):
                    c.step(base.frame('sleep').copy())
        self.assertFalse(c.active)
        self.assertTrue(c.journal['retirement']['released'])
        self.assertFalse(c.journal['retirement']['calibration_completed'])
        self.assertEqual(c.cycles, 0)
        self.assertEqual(c.journal['attempts'], {'cancel':1})
        self.assertEqual(c.journal['deadline'], 1090)
        self.assertEqual(self.inputs, [])

    def test_duplicate_proof_frame_cannot_finish_restore(self):
        c = self.current()
        with patch('meter_calibration.time.time',return_value=1020):
            c._begin_fishing_restore('任务地点变化')
        same = base.frame('sleep')
        with self.views(c,fill=True), patch('meter_calibration.time.time',return_value=1020):
            for now in (100,101.5,103):
                with patch('meter_calibration.time.monotonic',return_value=now):
                    c.step(same)
        self.assertTrue(c.active)
        self.assertEqual(c.seen, 1)

    def test_expired_restore_archives_and_never_inputs(self):
        c = self.current()
        with patch('meter_calibration.time.time',return_value=1020):
            c._begin_fishing_restore('任务地点变化')
        with patch('meter_calibration.time.time',return_value=1091), \
             patch.object(c,'_resolve_expired',return_value=False):
            c.step(base.frame('sleep'))
        self.assertTrue(c.awaiting_recovery)
        self.assertEqual(c.journal['attempts'], {'cancel':1})
        self.assertEqual(c.journal['deadline'], 1090)
        self.assertEqual(self.inputs, [])

    def test_spent_adjustment_limit_can_only_use_remaining_start_and_time(self):
        c = self.current()
        c.journal['attempts']['drag'] = 4
        with patch('meter_calibration.time.time',return_value=1020), \
             patch.object(c,'_step',side_effect=MeterWorkflowLimit('调整次数已用尽')):
            self.assertTrue(c.step(base.frame('sleep')))
        self.assertEqual(c.phase, 'restore_fishing')
        self.assertEqual(c.journal['attempts'], {'cancel':1,'drag':4})
        self.assertEqual(c.journal['deadline'], 1090)
        self.assertEqual(self.inputs, [])

    def test_pre_input_candidate_wait_does_not_arm_in_battle_or_result(self):
        c = self.controller
        for battle,result in ((True,False),(False,True)):
            with patch.object(c.fishing,'real_battle',return_value=battle), \
                 patch.object(c,'_catch_result',return_value=result), \
                 patch.object(c.fishing,'ordinary_preparation',return_value=True):
                self.assertFalse(c._candidate_ready(base.frame('sleep')))
        with patch.object(c.fishing,'real_battle',return_value=False), \
             patch.object(c,'_catch_result',return_value=False), \
             patch.object(c.fishing,'idle_icon',return_value=None), \
             patch.object(c.fishing,'ordinary_preparation',return_value=True):
            self.assertTrue(c._candidate_ready(base.frame('sleep')))

    def test_spent_start_keeps_readonly_start_proof_inside_last_thirty_seconds(self):
        c=self.current()
        c.context='new-task'
        c.journal['attempts']={'cancel':1,'drag':1,'start':1}
        c.fishing.start_fill_verified=True
        c.fishing.start_proof='post_release_fishing'
        with self.views(c,fill=True,task=False), \
             patch('meter_calibration.time.time',return_value=1075), \
             patch.object(c,'meter_frame_allowed',return_value=True), \
             patch.object(c.fishing,'post_start_evidence',return_value=True), \
             patch.object(c,'_begin_fishing_restore',side_effect=AssertionError('start proof must finish first')):
            c.step(base.frame('sleep').copy())
        self.assertFalse(c.active)
        self.assertEqual(c.journal['proof'],'post_release_fishing')
        self.assertEqual(c.journal['deadline'],1090)
        self.assertEqual(c.journal['attempts'],{'cancel':1,'drag':1,'start':1})
        self.assertEqual(c.cycles,1)
        self.assertEqual(self.inputs,[])

    def test_missing_start_proof_is_not_promoted_or_repeated_near_deadline(self):
        c=self.current()
        c.context='new-task'
        c.journal['attempts']={'cancel':1,'drag':1,'start':1}
        c.fishing.phase='starting'
        with self.views(c,fill=False,task=False), \
             patch('meter_calibration.time.time',return_value=1075), \
             patch.object(c,'meter_frame_allowed',return_value=True), \
             patch.object(c.fishing,'confirm_start',return_value=False), \
             patch.object(c.fishing,'prepare',return_value=True):
            c.step(base.frame('sleep').copy())
        self.assertTrue(c.active)
        self.assertEqual(c.phase,'prepare')
        self.assertFalse(c.journal.get('fishing_restore'))
        self.assertEqual(c.cycles,0)
        self.assertEqual(c.journal['attempts'],{'cancel':1,'drag':1,'start':1})
        self.assertEqual(self.inputs,[])

    def test_candidate_abandons_after_three_hundred_seconds_without_input(self):
        c = self.controller
        c.candidate_at = 1000
        original = deepcopy(c.journal)
        with patch('meter_calibration.time.time', return_value=1300):
            c.step(base.frame('sleep'))
        self.assertFalse(c.active)
        self.assertEqual(c.journal.get('deadline'), original.get('deadline'))
        self.assertEqual(c.journal['candidate_result']['inputs'], 0)
        self.assertEqual(self.inputs, [])

    def test_restore_cast_rejects_filled_busy_combat_port_sleep_and_changed_strokes(self):
        m = self.controller.fishing
        data = dict(context='place',box=(210,940,270,980),
                    digit_strokes='digits',world_strokes='stage')
        with ExitStack() as stack:
            for name,value in (('_main_visible',True),('world_matches',True),('at_port',False),
                               ('idle_icon','hook'),('busy',False),('real_battle',False),
                               ('cancel_button',False),('cast_data',data)):
                stack.enter_context(patch.object(m,name,return_value=value))
            # Marker alone is not a battle. Only this short restoration action
            # permits a task point; long starts remain prohibited.
            stack.enter_context(patch.object(m.training,'boss_mode',return_value=True))
            self.assertTrue(m.validate('resume_cast',base.frame('sleep'),data))
            self.assertFalse(m.validate('start',base.frame('sleep'),data))
            for name,value in (('busy',True),('real_battle',True),('at_port',True),
                               ('idle_icon','sleep'),('cancel_button',True)):
                with self.subTest(name=name), patch.object(m,name,return_value=value):
                    self.assertFalse(m.validate('resume_cast',base.frame('sleep'),data))
            self.assertFalse(m.validate('resume_cast',base.frame('sleep'),dict(data,digit_strokes='different')))

    def test_real_task_gray_hook_is_restore_only_and_running_fill_is_rejected(self):
        c = self.controller
        native = np.asarray(Image.open(base.ROOT/'tests/fixtures/meter-continuation/task-idle-after-cancel.png').convert('RGB'))
        c.world.training.state.update(mode='farming',rank=39)
        c.fishing.current = 1530
        self.assertTrue(c.world.training.boss_mode(native))
        self.assertEqual(c.fishing.idle_icon(native), 'hook')
        data = c.fishing.resume_cast_data(native)
        self.assertIsNotNone(data)
        self.assertTrue(c.fishing.validate('resume_cast',native,data))
        self.assertFalse(c.fishing.validate('start',native,data))
        self.assertFalse(c.fishing.validate('cancel',native,data))
        fill = np.asarray(Image.open(base.ROOT/'tests/fixtures/meter-preserve-route/running-fill.png').convert('RGB'))
        self.assertIsNone(c.fishing.resume_cast_data(fill))
        self.assertEqual(self.inputs, [])


if __name__ == '__main__':
    unittest.main()
