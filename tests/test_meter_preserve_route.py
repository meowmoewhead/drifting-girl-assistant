"""Calibration may stop fishing without toggling the native sailing route."""
from copy import deepcopy
from unittest.mock import patch
import unittest
import numpy as np
from PIL import Image

from tests import test_meter_calibration as base, test_meter_cancel_timing as cancel
frame, image = base.frame, cancel.image
from drift.tasks import handoff_journal


class PreserveRouteTests(unittest.TestCase):
    setUpClass = classmethod(base.MeterCalibrationTests.setUpClass.__func__)
    setUp = base.MeterCalibrationTests.setUp
    tearDown = base.MeterCalibrationTests.tearDown
    pending = base.MeterCalibrationTests.pending
    restart = base.MeterCalibrationTests.restart

    def current(self):
        c = self.pending()
        c.journal.update(route_policy='preserve', calibration_context='place')
        c.route_verified, c.off_verified = True, False
        c.world.training.state['sailing_enabled'] = True
        c.fishing.start_route_expected = True
        return c

    def test_new_policy_cannot_toggle_route_even_with_remaining_input_budget(self):
        c = self.current()
        with patch('meter_calibration.time.time', return_value=1000):
            for action in ('stop','open','start','auto_sale'):
                self.assertFalse(c.reserve_sailing_input(action))
                self.assertFalse(c.validate_sailing(action, frame('sleep')))
        self.assertNotIn('sailing_attempts', c.journal)
        self.assertEqual(self.inputs, [])

    def test_each_input_requires_known_unchanged_route_world_and_location(self):
        c = self.current()
        with patch('meter_calibration.time.time', return_value=1000), \
             patch.object(c.world, '_parallel', return_value=True), \
             patch.object(c.world, '_stage_strokes', return_value=b'stage'), \
             patch.object(c.world.training, 'boss_mode', return_value=False), \
             patch.object(c.sailing, 'running', return_value=True), \
             patch.object(c.fishing, 'at_port', return_value=False), \
             patch.object(c.fishing, '_context', return_value='place'):
            self.assertTrue(c.meter_frame_allowed(frame('sleep')))
            for target, name, value in ((c.sailing,'running',False),(c.sailing,'running',None),
                    (c.fishing,'_context','new-place'),(c.fishing,'at_port',True),
                    (c.world.training,'boss_mode',True),(c.world,'_parallel',False)):
                with self.subTest(name=name, value=value), patch.object(target,name,return_value=value):
                    self.assertFalse(c.meter_frame_allowed(frame('sleep')))
        self.assertEqual(self.inputs, [])

    def test_new_place_reanchors_three_times_without_resetting_deadline_or_attempts(self):
        c = self.current(); c.journal['attempts'] = {'cancel':1,'drag':1}
        original = deepcopy(c.journal)
        data = dict(value=1200,player=1300,high=1400,low=1100,context='new-place',adaptive=True)
        with patch.object(c,'screen',return_value=dict(rank=38,stage=8,menu=False,route=True)), \
             patch.object(c,'meter_frame_allowed',return_value=False), \
             patch.object(c.fishing,'_context',return_value='new-place'), \
             patch.object(c.fishing,'measure',return_value=data), \
             patch.object(c.fishing,'prepare',side_effect=AssertionError('no input during re-anchor')), \
             patch('meter_calibration.time.time',return_value=1000):
            for now in (100,101.5,103):
                with patch('meter_calibration.time.monotonic',return_value=now):
                    c.step(frame('sleep').copy())
        self.assertEqual(c.phase,'prepare')
        self.assertEqual(c.context,'new-place')
        self.assertTrue(c.route_verified)
        self.assertFalse(c.off_verified)
        self.assertEqual(c.journal['deadline'],original['deadline'])
        self.assertEqual(c.journal['attempts'],original['attempts'])
        self.assertTrue(c.world.training.state['sailing_enabled'])
        self.assertEqual(self.inputs, [])

    def test_current_fill_completes_without_menu_and_starts_persisted_cooldown(self):
        c = self.current(); c.interval = 5400
        data = dict(player=1348,value=1343,high=1421,low=1195,adaptive=True,y=985)
        with patch.object(c,'screen',return_value=dict(rank=38,stage=8,menu=False,route=True)), \
             patch.object(c,'meter_frame_allowed',return_value=True), \
             patch.object(c.fishing,'_context',return_value='place'), \
             patch.object(c.fishing,'measure',return_value=data), \
             patch.object(c.fishing,'blue_fill',return_value=True), \
             patch.object(c.fishing,'prepare',side_effect=AssertionError('existing fill cannot be cancelled')), \
             patch('meter_calibration.time.time',return_value=1000):
            for now in (100,101.5,103):
                with patch('meter_calibration.time.monotonic',return_value=now): c.step(frame('sleep'))
        self.assertEqual(c.phase,'idle')
        self.assertEqual(c.cycles,1)
        self.assertEqual(c.journal['next_check'],6400)
        self.assertEqual(self.inputs,[])
        restored = self.restart(c.journal,now=1200)
        restored.interval = 5400
        with patch('meter_calibration.time.time',return_value=6399), \
             patch.object(restored,'screen',side_effect=AssertionError('cooldown must skip probing')):
            self.assertFalse(restored.step(frame('sleep')))
        self.assertEqual(restored.status()['next_check'],6400)

    def test_restart_rechecks_context_without_migrating_old_records(self):
        c=self.current(); c.journal['training']=deepcopy(c.world.training.state)
        restarted=self.restart(c.journal)
        self.assertEqual(restarted.phase,'route_context')
        self.assertFalse(restarted.route_verified)
        self.assertFalse(restarted.off_verified)
        self.assertTrue(restarted.fishing.start_route_expected)
        old=deepcopy(c.journal);old.pop('route_policy')
        legacy=self.restart(old)
        self.assertEqual(legacy.phase,'stop')
        self.assertFalse(legacy.expected_route())

    def test_identical_reason_does_not_repeat_every_minute_but_status_updates(self):
        c=self.controller
        for now in (100,160,220,400):
            with patch('meter_calibration.time.monotonic',return_value=now):
                c._report_wait('差距不足',dict(player=int(now),value=100,high=900))
        self.assertEqual(len(self.events),1)
        self.assertEqual(c.player,400)
        with patch('meter_calibration.time.monotonic',return_value=500): c._report_wait('等待普通鱼')
        self.assertEqual(len(self.events),2)

    def test_late_first_cast_observation_not_retired_by_old_input_deadline(self):
        record=dict(pending=True,phase='first_verify',deadline=100,phase_deadline=90,
                    first_deadline=60,first_observation_deadline=90)
        self.assertEqual(handoff_journal.expiry(record,70),[])
        self.assertEqual(handoff_journal.expiry(record,90),['phase_deadline','first_observation_deadline'])
        record.pop('first_observation_deadline')
        self.assertEqual(handoff_journal.expiry(record,70),['first_deadline'])

    def test_running_scan_is_explicit_and_changed_or_unknown_route_never_inputs(self):
        meter=cancel.MeterCancelTimingTests.make_meter(self,self.temp.name)
        native=image('cancel-rank35'); expected=meter.cast_data(native)
        self.assertIsNone(meter.scan_cancel_frame(lambda:native.copy(),lambda _:True,expected,lambda:False))
        accepted=meter.scan_cancel_frame(lambda:native.copy(),lambda _:True,expected,lambda:False,
                                         expected_route=True)
        self.assertIsNotNone(accepted)
        self.assertIsNone(meter.scan_cancel_frame(lambda:native.copy(),lambda _:False,expected,lambda:False,
                                                 expected_route=True))
        self.assertIsNone(meter.scan_cancel_frame(lambda:native.copy(),lambda _:None,expected,lambda:False,
                                                 expected_route=True))

    def test_real_running_fill_needs_explicit_route_policy_and_three_fresh_votes(self):
        m=self.controller.fishing
        native=np.asarray(Image.open(base.ROOT/'tests/fixtures/meter-preserve-route/running-fill.png').convert('RGB'))
        self.assertFalse(m.post_start_evidence(native))
        m.start_route_expected=True
        self.assertTrue(m.post_start_evidence(native))
        m.arm_start_observation(native)
        for n,now in enumerate((100,101.5,103)):
            with patch('fishing_meter.time.monotonic',return_value=now):
                self.assertEqual(m.confirm_start(native.copy()), n==2)
        self.assertEqual(m.start_proof,'post_release_fishing')
        self.assertEqual(m.confirmed_starts,1)
        self.assertEqual(self.inputs,[])


if __name__ == '__main__': unittest.main()
