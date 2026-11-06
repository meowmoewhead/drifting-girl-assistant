"""Native AUTO resource selection and safe handoff back to boss navigation."""
from pathlib import Path
from copy import deepcopy
import json
import shutil
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from boss_training import BossTraining, training_options
from digit_ocr import DigitReader, NumberReading
from parallel_world import ParallelWorld
from training_sailing import TrainingSailing
from fishing_meter import FishingMeter
from app import Clicker

ROOT = Path(__file__).resolve().parents[1]


def frame(name):
    return np.asarray(Image.open(ROOT/'tests/fixtures/training-sailing'/(name+'.png')).convert('RGB'))


class Reader:
    def read(self,_):
        return NumberReading('38',38,.999)


class SailingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'training.json'
        self.events, self.actions = [],[]
        self.training = BossTraining(ROOT,Reader(),self.events.append,state_path=self.path)
        self.training.begin(38,1535,1604)
        self.sailing = TrainingSailing(ROOT,self.training,self.click,self.events.append)

    def tearDown(self):
        self.temp.cleanup()

    def click(self,action,x,y,label):
        self.sailing.before_click(action)
        self.actions.append((action,x,y,label))
        return True

    def step(self,name,wanted=True):
        self.sailing.next_at = 0
        return self.sailing.step(frame(name),wanted)

    def world(self):
        w = ParallelWorld(ROOT,lambda *a:self.actions.append(('map',*a)),lambda *a:None,
                          self.events.append,Reader(),training_state_path=self.path,
                          sailing_action=self.click)
        self.sailing = w.sailing
        w.phase, w.rank = 'play',38
        return w

    def test_native_selected_tiles_and_main_toggle_states(self):
        s = self.sailing
        self.assertTrue(s.menu(frame('menu-one')))
        self.assertFalse(s.running(frame('inactive')))
        self.assertFalse(s.running(frame('stopped')))
        self.assertTrue(s.running(frame('active')))
        self.assertFalse(self.training.main(frame('active-flying-cake')))
        self.assertTrue(s.running(frame('active-flying-cake')))
        self.assertIsNone(s.running(frame('menu-three')))
        for name,value in (('pearl',True),('amber',False),('gem',False),('material',False),('boss',False)):
            self.assertIs(s.selected(frame('menu-one'),name),value)
        self.assertIs(s.selected(frame('menu-no-pearl'),'pearl'),False)
        self.assertTrue(s.configured(frame('menu-three')))
        self.assertFalse(s.configured(frame('menu-one')))

    def test_native_arrow_and_checkbox_glow_are_identified_but_dimmed_frames_are_rejected(self):
        for i in range(3):
            image = frame(f'active-glow-{i}')
            self.assertTrue(self.sailing.running(image))
            self.assertTrue(self.sailing.validate('stop',image))
            self.assertFalse(self.sailing.validate('stop',(image*.7).astype('uint8')))

    def test_sailing_guard_allows_level_reading_when_flying_fishcake_hides_center_footer(self):
        image = frame('active-glow-2')
        self.assertFalse(self.training.main(image))
        self.assertIsNone(self.training.levels(image))
        self.training.reader = DigitReader(ROOT)
        known = self.sailing.running(image) is True
        self.assertTrue(known)
        self.assertIsNone(self.training.levels(image,main_known=known))
        self.assertEqual(self.training.levels(image,main_known=known),(1541,None))
        self.assertEqual(self.training.state['boss_level'],1604)

    def test_configure_start_and_confirm_preserves_the_original_training_target(self):
        self.assertTrue(self.step('inactive'))
        self.assertEqual(self.actions[-1][0],'open')
        self.step('menu-one')
        self.assertEqual(self.actions[-1][:3],('target:amber',417,458))
        self.step('menu-two')
        self.assertEqual(self.actions[-1][:3],('target:gem',490,458))
        self.step('menu-three')
        self.step('menu-three')
        self.assertEqual(self.actions[-1][0],'start')
        self.assertTrue(self.training.state['sailing_enabled'])
        self.assertFalse(self.training.state['sailing_verified'])
        self.step('active')
        self.assertFalse(self.step('active'))
        self.assertTrue(self.training.state['sailing_verified'])
        self.assertEqual(self.training.state['start_level'],1535)
        self.assertEqual(self.training.state['boss_level'],1604)
        self.assertEqual(self.sailing.cycles,1)

    def test_unknown_checked_extra_or_dimmed_menu_cannot_start(self):
        for image in ((frame('menu-three')*.7).astype('uint8'),frame('menu-three').copy()):
            if image.max() == 255:
                image[636:658,566:586] = (25,25,25)
            self.assertFalse(self.sailing.validate('start',image))
        image = frame('menu-three').copy()
        image[428:436,391:399] = (225,180,100)
        self.assertIsNone(self.sailing.selected(image,'amber'))
        self.assertFalse(self.sailing.validate('start',image))

    def test_fresh_frame_validation_rejects_a_toggle_after_state_changed(self):
        self.assertTrue(self.sailing.validate('open',frame('inactive')))
        self.assertFalse(self.sailing.validate('open',frame('active')))
        self.assertTrue(self.sailing.validate('stop',frame('active')))
        self.assertFalse(self.sailing.validate('stop',frame('stopped')))
        self.assertTrue(self.sailing.validate('target:amber',frame('menu-one')))
        self.assertFalse(self.sailing.validate('target:amber',frame('menu-three')))
        self.assertFalse(self.sailing.validate('start',frame('menu-no-pearl')))
        self.assertFalse(self.sailing.validate('stop',(frame('active')*.7).astype('uint8')))
        self.assertFalse(self.sailing.validate('stop',(frame('active-flying-cake')*.7).astype('uint8')))
        for path in ('assets/equipment/panel.png','tests/fixtures/parallel/android-home.png'):
            if (ROOT/path).is_file():
                other = np.asarray(Image.open(ROOT/path).convert('RGB'))
                self.assertIsNone(self.sailing.running(other))

    def test_stop_requires_confirmed_inactive_auto_and_keeps_training_progress(self):
        t = self.training
        t.set_sailing(True,verified=True)
        self.step('active',False)
        self.assertEqual(self.actions[-1][0],'stop')
        with self.assertRaisesRegex(RuntimeError,'自动航行停止'):
            t.clear()
        self.assertTrue(self.step('stopped',False))
        self.assertTrue(t.state['sailing_enabled'])
        self.assertFalse(self.step('stopped',False))
        self.assertFalse(t.state['sailing_enabled'])
        self.assertEqual(t.state['mode'],'farming')
        self.assertEqual(t.state['boss_level'],1604)

    def test_verified_route_survives_restart_without_repeated_start_or_stop(self):
        self.training.set_sailing(True,verified=True)
        w = self.world()
        w.training.player_level = 1537
        with patch.object(w,'_rank'),patch.object(w.training,'levels',return_value=(1537,None)):
            self.assertFalse(w.step(frame('active')))
        self.assertEqual(self.actions,[])
        self.assertTrue(w.training.state['sailing_enabled'])

    def test_training_does_not_return_before_gap_and_stops_route_before_boss_map(self):
        self.training.set_sailing(True,verified=True)
        w = self.world()
        w.training.state['started_at'] -= 3600
        w.training.player_level = 1583
        with patch.object(w,'_rank'),patch.object(w.training,'levels',return_value=(1583,None)):
            self.assertFalse(w.step(frame('active')))
        self.assertEqual(self.actions,[])
        w.training.player_level = 1584
        with patch.object(w,'_rank'),patch.object(w.training,'levels',return_value=(1584,None)):
            w.step(frame('active'))
            self.assertEqual(self.actions[-1][0],'stop')
            w.sailing.next_at = 0
            w.step(frame('stopped'))
            self.assertEqual(len(self.actions),1)
            w.sailing.next_at = 0
            w.step(frame('stopped'))
        self.assertEqual(w.phase,'find_boss')
        self.assertEqual(self.actions[-1][:3],('map',350,30))
        self.assertEqual(w.training.state['mode'],'trial')
        self.assertFalse(w.training.state['sailing_enabled'])

    def test_rank40_stops_auto_before_clearing_training_or_reaching_portal(self):
        self.training.set_sailing(True,verified=True)
        w = self.world()
        w.rank,w.completed_rank_40 = 40,True
        with patch.object(w,'_rank'):
            w.step(frame('active'))
            self.assertEqual(w.training.state['mode'],'farming')
            self.assertEqual(w.phase,'play')
            self.assertEqual(self.actions[-1][0],'stop')
            for _ in range(2):
                w.sailing.next_at = 0
                w.step(frame('stopped'))
        self.assertEqual(w.phase,'return_start')
        self.assertEqual(w.training.state['mode'],'idle')
        self.assertFalse(any('次元门' in a[-1] for a in self.actions))

    def test_final_boss_keeps_sailing_past_normal_gap_then_stops_before_retry(self):
        self.training.begin(39,1622,1810)
        self.training.set_sailing(True,verified=True)
        w = self.world()
        w.rank = 39
        w.training.player_level = 1790
        with patch.object(w,'_rank'),patch.object(w.training,'levels',return_value=(1790,None)):
            self.assertTrue(w._sailing_wanted())
            self.assertFalse(w.step(frame('active')))
            self.assertFalse(self.step('active',True))
        self.assertEqual(self.actions,[])
        w.training.player_level = 1810
        with patch.object(w,'_rank'),patch.object(w.training,'levels',return_value=(1810,None)):
            self.assertFalse(w._sailing_wanted())
            w.step(frame('active'))
            self.assertEqual(self.actions[-1][0],'stop')
            for _ in range(2):
                w.sailing.next_at = 0
                w.step(frame('stopped'))
        self.assertEqual(w.phase,'find_boss')
        self.assertEqual(self.actions[-1][:3],('map',350,30))
        self.assertEqual(w.training.state['mode'],'trial')
        self.assertFalse(w.training.state['sailing_enabled'])

    def test_rank40_during_menu_closes_it_without_starting_sailing(self):
        w = self.world()
        w.rank,w.completed_rank_40 = 40,True
        w.reader = Mock(read=Mock(return_value=NumberReading('40',40,.999)))
        w.sailing.phase = 'configure'
        with patch.object(w,'_rank'):
            w.step(frame('menu-three'))
        self.assertEqual(self.actions[-1][0],'close')
        self.assertFalse(any(a[0]=='start' for a in self.actions))

    def test_unverified_saved_start_is_checked_before_resuming_and_timeout_keeps_record(self):
        self.training.set_sailing(True,verified=False)
        w = self.world()
        w.training.player_level = 1537
        with patch.object(w,'_rank'),patch.object(w.training,'levels',return_value=(1537,None)):
            w.step(frame('active'))
        self.assertEqual(self.actions[-1][0],'stop')
        saved = self.path.read_bytes()
        w.sailing.next_at = 0
        w.sailing.since -= 41
        with self.assertRaisesRegex(RuntimeError,'保留练级记录'):
            w.sailing.step(frame('active'),True)
        self.assertEqual(self.path.read_bytes(),saved)

    def test_auto_sailing_option_is_strict_and_false_stops_the_owned_route(self):
        self.assertTrue(training_options()['auto_sailing'])
        for value in ('yes',1,None):
            with self.assertRaises(ValueError):
                training_options({'auto_sailing':value})
        self.training.set_sailing(True,verified=True)
        w = self.world()
        w.training.options['auto_sailing'] = False
        with patch.object(w,'_rank'):
            w.step(frame('active'))
        self.assertEqual(self.actions[-1][0],'stop')

    def test_opening_and_start_retries_cannot_reset_the_workflow_deadline(self):
        self.step('inactive')
        original = self.sailing.since
        self.step('menu-three')
        self.step('menu-three')
        self.assertEqual(self.sailing.since,original)
        self.sailing.since -= 41
        count = len(self.actions)
        with self.assertRaisesRegex(RuntimeError,'保留练级记录'):
            self.step('menu-three')
        self.assertEqual(len(self.actions),count)

    def test_repeated_fresh_frame_rejections_are_bounded_without_clearing_progress(self):
        self.sailing.action = lambda *args:False
        self.step('active')
        self.assertTrue(self.sailing.active)
        self.sailing.since -= 41
        saved = self.path.read_bytes()
        with self.assertRaisesRegex(RuntimeError,'保留练级记录'):
            self.step('active')
        self.assertEqual(self.path.read_bytes(),saved)
        self.assertEqual(self.actions,[])

    def sleep_frame(self, *, off=False, filled=False, awake=False):
        image = np.asarray(Image.open(ROOT/'tests/fixtures/fishing-meter/sailing-sleep-bottom.png')
                           .convert('RGB')).copy()
        if off:
            image[4:65,186:254] = frame('stopped')[4:65,186:254]
        if filled or awake:
            name = 'fill-hook' if filled else 'ready'
            dial = np.asarray(Image.open(ROOT/'tests/fixtures/fishing-meter'/(name+'.png'))
                              .convert('RGB'))
            image[940:1090,280:442] = dial[940:1090,280:442]
        return image

    def sleep_step(self, image, at, wanted=True):
        self.sailing.next_at = 0
        with patch('training_sailing.time.monotonic',return_value=at):
            return self.sailing.step(image,wanted)

    def prepare_sleep_test(self):
        self.training.set_sailing(True,verified=True)
        self.meter = FishingMeter(ROOT,self.training,None,self.events.append)
        self.meter.step = Mock(return_value=True)
        self.sailing.fishing = self.meter

    def test_low_meter_stops_route_then_waits_for_calibration_proof(self):
        self.prepare_sleep_test()
        data=dict(value=1400,player=1535,low=1359,high=1604,goal=1530,context='same')
        self.meter.measure=Mock(return_value=data)
        self.meter.prepare=Mock(return_value=True)
        active=self.sleep_frame(awake=True)
        stopped=self.sleep_frame(off=True,awake=True)
        for at in (100,102):self.sleep_step(active,at)
        self.assertEqual(self.actions,[])
        self.sleep_step(active,104)
        self.assertEqual([a[0] for a in self.actions],['stop'])
        self.assertTrue(self.training.state['sailing_meter_adjustment'])
        for at in (108,112):self.sleep_step(stopped,at)
        self.meter.prepare.assert_called_once()
        self.sleep_step(self.sleep_frame(off=True,filled=True),116)
        self.assertTrue(self.training.state['sailing_meter_adjustment'])
        self.assertFalse(any(a[0] in ('open','start') for a in self.actions))
        self.meter.confirmed_starts += 1
        self.sleep_step(stopped,120)
        self.assertNotIn('sailing_meter_adjustment',self.training.state)
        self.assertEqual(self.training.state['boss_level'],1604)

    def test_disabled_meter_calibration_does_not_stop_a_healthy_route(self):
        self.prepare_sleep_test()
        self.training.options['meter_auto_adjust']=False
        self.meter.measure=Mock(return_value=dict(value=1400,player=1535,goal=1530,context='same'))
        for at in (100,104,108):self.sleep_step(self.sleep_frame(awake=True),at)
        self.assertEqual(self.actions,[])
        self.meter.measure.assert_not_called()

    def test_game_adaptation_near_player_does_not_repeat_maximum_calibration(self):
        self.prepare_sleep_test()
        data=dict(value=1525,player=1535,low=1359,high=1604,goal=1604,
                  recalibration_goal=1530,adaptive=True,context='same')
        self.meter.measure=Mock(return_value=data)
        active=self.sleep_frame(awake=True)
        for at in (100,104,108,140):
            self.sleep_step(active,at)
        self.assertEqual(self.actions,[])
        self.assertNotIn('sailing_sleep_recovery',self.training.state)
        self.assertTrue(self.training.state['sailing_verified'])
        data['value']=1510
        for at in (172,174,176):
            self.sleep_step(active,at)
        self.assertEqual([a[0] for a in self.actions],['stop'])
        self.assertEqual(self.meter.goal,1604)
        self.assertTrue(self.training.state['sailing_meter_adjustment'])

    def test_native_moving_countdown_does_not_reset_low_meter_confirmation(self):
        self.prepare_sleep_test()
        self.training.reader = DigitReader(ROOT)
        self.training.state.update(rank=39,boss_level=1810)
        self.training.player_level = 1734
        first = np.asarray(Image.open(ROOT/'tests/fixtures/fishing-meter/moving-context-1.png').convert('RGB'))
        second = np.asarray(Image.open(ROOT/'tests/fixtures/fishing-meter/moving-context-2.png').convert('RGB'))
        self.sleep_step(first,100)
        self.assertEqual(self.actions,[])
        self.sleep_step(second,104)
        self.assertEqual([a[0] for a in self.actions],['stop'])
        self.assertTrue(self.training.state['sailing_meter_adjustment'])

    def test_saved_meter_calibration_at_port_resumes_route_before_fishing(self):
        self.prepare_sleep_test()
        self.training.reader = DigitReader(ROOT)
        self.training.state.update(rank=39,boss_level=1810,sailing_enabled=False,
                                   sailing_sleep_recovery=True,sailing_meter_adjustment=True)
        self.training.player_level = 1736
        native = np.asarray(Image.open(ROOT/'tests/fixtures/fishing-meter/port-after-sale.png').convert('RGB'))
        self.assertFalse(self.sailing.running(native))
        self.sleep_step(native,100)
        self.assertEqual([a[0] for a in self.actions],['open'])
        self.assertNotIn('sailing_meter_adjustment',self.training.state)
        self.assertNotIn('sailing_sleep_recovery',self.training.state)
        self.assertEqual(self.training.state['boss_level'],1810)
        self.assertEqual(self.training.state['start_level'],1535)
        self.meter.step.assert_not_called()
        for at in (104,108):self.sleep_step(frame('menu-three'),at)
        self.assertEqual([a[0] for a in self.actions],['open','start'])

    def test_stopped_waiting_cast_is_serviced_before_slow_level_ocr(self):
        self.prepare_sleep_test()
        self.training.reader = DigitReader(ROOT)
        self.training.state.update(rank=39,boss_level=1810,sailing_enabled=False,
                                   sailing_sleep_recovery=True,sailing_meter_adjustment=True)
        self.training.player_level = 1712
        native = np.asarray(Image.open(ROOT/'tests/fixtures/fishing-meter/low-meter-waiting.png').convert('RGB')).copy()
        port = np.asarray(Image.open(ROOT/'tests/fixtures/fishing-meter/port-after-sale.png').convert('RGB'))
        native[4:65,186:254] = port[4:65,186:254]
        self.assertFalse(self.sailing.running(native))
        self.assertTrue(self.meter.waiting_cast(native))
        self.sailing.wake_off_seen = 2
        self.meter.prepare = Mock(return_value=True)
        self.training.levels = Mock(side_effect=AssertionError('slow OCR lost the waiting-cast window'))
        self.sleep_step(native,100)
        self.meter.prepare.assert_called_once()
        self.training.levels.assert_not_called()

    def test_restarted_low_filled_meter_is_read_before_cancelling_the_old_cast(self):
        self.prepare_sleep_test()
        self.training.reader = DigitReader(ROOT)
        self.training.state.update(rank=39,boss_level=1810,sailing_enabled=False,
                                   sailing_sleep_recovery=True,sailing_meter_adjustment=True)
        self.training.player_level = 1752
        native = np.asarray(Image.open(ROOT/'tests/fixtures/fishing-meter/filled-prebattle.png').convert('RGB'))
        self.sailing.phase = 'wake_prepare'
        self.sailing.since = 100
        self.sailing.wake_off_seen = 2
        self.meter.action = Mock(return_value=True)
        self.assertIsNone(self.meter.current)
        self.sleep_step(native,104)
        self.assertEqual(self.meter.current,1709)
        self.assertEqual(self.meter.goal,1752)
        self.assertEqual(self.meter.action.call_args.args[0],'scan_cancel')
        self.assertEqual(self.meter.cancels,1)

    def test_confirmed_new_start_is_adopted_without_fast_cancelling_it(self):
        self.prepare_sleep_test()
        self.training.reader = DigitReader(ROOT)
        self.training.state.update(rank=39,boss_level=1810,sailing_enabled=False,
                                   sailing_sleep_recovery=True,sailing_meter_adjustment=True)
        self.training.player_level = 1752
        native = np.asarray(Image.open(ROOT/'tests/fixtures/fishing-meter/filled-prebattle.png').convert('RGB'))
        self.meter.current,self.meter.goal = 1709,1747
        self.meter.confirmed_starts = 1
        self.meter.start_fill_verified = True
        self.meter.prepare = Mock(side_effect=AssertionError('new fishing must not be cancelled'))
        self.sailing.phase = 'wake_prepare'
        self.sailing.since = 100
        self.sailing.wake_off_seen = 2
        self.sailing.wake_confirmed_before = 0
        self.sleep_step(native,104)
        self.meter.prepare.assert_not_called()
        self.assertNotIn('sailing_meter_adjustment',self.training.state)
        self.assertEqual(self.sailing.phase,'idle')

    def expired_verified_start(self):
        self.prepare_sleep_test()
        self.training.reader = DigitReader(ROOT)
        self.training.state.update(rank=39,boss_level=1810,sailing_enabled=False,
                                   sailing_sleep_recovery=True,sailing_meter_adjustment=True)
        self.training.player_level = 1752
        self.meter.current,self.meter.goal = 1709,1747
        self.meter.confirmed_starts = 1
        self.meter.start_fill_verified = True
        self.meter.prepare = Mock(side_effect=AssertionError('do not cancel or hold again'))
        self.sailing.phase,self.sailing.since = 'wake_prepare',100
        self.sailing.wake_off_seen,self.sailing.wake_confirmed_before = 2,0
        world = ParallelWorld(ROOT,lambda *args:False,lambda *args:False,self.events.append,
                              self.training.reader,training_state_path=self.path)
        world.rank = 39
        world.phase = 'play'
        world.training,world.fishing = self.training,self.meter
        world.sailing = self.sailing
        self.sailing.context_check = world.sailing_context
        self.expired_world = world
        return np.asarray(Image.open(ROOT/'tests/fixtures/fishing-meter/filled-prebattle.png').convert('RGB')).copy()

    def test_verified_hold_result_survives_next_deadline_check_without_any_input(self):
        native = self.expired_verified_start()
        original = {key:self.training.state[key] for key in ('rank','boss_level','start_level','started_at')}
        self.sleep_step(native,191)
        self.assertEqual(self.sailing.phase,'idle')
        self.assertEqual(self.actions,[])
        self.meter.prepare.assert_not_called()
        self.assertNotIn('sailing_sleep_recovery',self.training.state)
        self.assertNotIn('sailing_meter_adjustment',self.training.state)
        self.assertEqual({key:self.training.state[key] for key in original},original)
        saved = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertFalse(saved['sailing_enabled'])
        self.assertNotIn('sailing_sleep_recovery',saved)

    def test_expired_old_fill_does_not_gain_a_new_deadline(self):
        native = self.expired_verified_start()
        self.meter.confirmed_starts = self.sailing.wake_confirmed_before
        before = self.path.read_bytes()
        with self.assertRaises(RuntimeError):self.sleep_step(native,191)
        self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(self.actions,[])

    def test_expired_new_start_cannot_override_world_boss_running_or_unknown_guards(self):
        for case in ('rank','world','boss','running','unknown','fill','off-proof','disabled','meter-disabled','unpaired'):
            with self.subTest(case=case):
                native = self.expired_verified_start()
                wanted = True
                if case == 'rank':self.training.state['rank'] = 38
                elif case == 'world':native[43:76,9:43] = 0
                elif case == 'boss':self.training.boss_mode = Mock(return_value=True)
                elif case == 'running':native[4:65,186:254] = frame('active')[4:65,186:254]
                elif case == 'unknown':native[4:65,186:254] = 0
                elif case == 'fill':self.meter.blue_fill = Mock(return_value=False)
                elif case == 'off-proof':self.sailing.wake_off_seen = 1
                elif case == 'disabled':wanted = False
                elif case == 'meter-disabled':self.training.options['meter_auto_adjust'] = False
                elif case == 'unpaired':self.meter.start_fill_verified = False
                before = self.path.read_bytes()
                with self.assertRaises(RuntimeError):self.sleep_step(native,191,wanted=wanted)
                self.assertEqual(self.path.read_bytes(),before)
                self.assertEqual(self.actions,[])
                # The next subcase uses the genuine recognizer again.
                self.training.boss_mode = BossTraining.boss_mode.__get__(self.training)

    def test_parallel_entry_commits_completed_hold_before_its_outer_timeout_or_wait(self):
        native = self.expired_verified_start()
        world = self.expired_world
        world.next_action_at = float('inf')
        original = {key:self.training.state[key] for key in ('rank','boss_level','start_level','started_at')}
        with patch('training_sailing.time.monotonic',return_value=191):
            self.assertTrue(world.step(native))
        self.assertEqual(self.sailing.phase,'idle')
        self.assertNotIn('sailing_sleep_recovery',self.training.state)
        self.assertEqual({key:self.training.state[key] for key in original},original)
        self.assertEqual(self.actions,[])
        self.meter.prepare.assert_not_called()

    def test_parallel_entry_cannot_commit_old_fill_or_a_changed_world_after_deadline(self):
        for case in ('old-fill','world'):
            with self.subTest(case=case):
                native = self.expired_verified_start()
                if case == 'old-fill':self.meter.confirmed_starts = self.sailing.wake_confirmed_before
                else:native[43:76,9:43] = 0
                saved = self.path.read_bytes()
                with patch('training_sailing.time.monotonic',return_value=191):
                    with self.assertRaises(RuntimeError):self.expired_world.step(native)
                self.assertEqual(self.path.read_bytes(),saved)
                self.assertEqual(self.actions,[])

    def test_wake_commits_both_fill_captures_before_the_next_loop_can_time_out(self):
        native = self.expired_verified_start()
        self.meter.confirmed_starts = 0
        self.meter.start_fill_verified = False
        self.meter.blue_fill = Mock(return_value=False)
        self.meter.cancellable_cast = Mock(return_value=False)
        self.meter.measure = Mock(return_value=None)
        def completed(_):
            self.meter.start_fill_verified = True
            self.meter.confirmed_starts += 1
            return True
        self.meter.prepare = Mock(side_effect=completed)
        self.sleep_step(native,189)
        self.assertEqual(self.sailing.phase,'idle')
        self.assertNotIn('sailing_sleep_recovery',self.training.state)
        self.assertEqual(self.actions,[])
        with patch('training_sailing.time.monotonic',return_value=191):
            self.sailing._timeout()

    def test_obscured_frame_blocks_input_but_keeps_an_already_verified_stop(self):
        self.prepare_sleep_test()
        for at,native in ((100,self.sleep_frame()),(104,self.sleep_frame()),
                          (108,self.sleep_frame(off=True)),(112,self.sleep_frame(off=True))):
            self.sleep_step(native,at)
        self.assertEqual(self.sailing.wake_off_seen,2)
        count=len(self.actions)
        hidden=self.sleep_frame(off=True)
        hidden[1137:1192,208:269]=0
        self.assertIsNone(self.sailing.running(hidden))
        self.sleep_step(hidden,116)
        self.assertEqual(self.sailing.wake_off_seen,2)
        self.assertEqual(len(self.actions),count)
        self.sleep_step(self.sleep_frame(awake=True),120)
        self.assertEqual(self.sailing.wake_off_seen,0)
        self.assertEqual(self.actions[-1][0],'stop')

    def test_sleeping_route_stops_once_and_requires_two_off_frames_before_meter_input(self):
        self.prepare_sleep_test()
        active, stopped = self.sleep_frame(),self.sleep_frame(off=True)
        self.assertTrue(self.sailing.running(active))
        self.assertFalse(self.sailing.running(stopped))
        self.sleep_step(active,100)
        self.sleep_step(active,102)
        self.assertEqual(self.actions,[])
        self.sleep_step(active,104)
        self.sleep_step(active,106)
        self.assertEqual([a[0] for a in self.actions],['stop'])
        self.assertTrue(self.training.state['sailing_sleep_recovery'])
        self.meter.step.assert_not_called()
        self.sleep_step(stopped,108)
        self.meter.step.assert_not_called()
        self.assertTrue(self.training.state['sailing_enabled'])
        self.sleep_step(stopped,110)
        self.meter.step.assert_called_once()
        self.assertFalse(self.training.state['sailing_enabled'])
        self.assertEqual(self.training.state['boss_level'],1604)
        self.assertEqual(self.training.state['start_level'],1535)
        self.assertFalse(any(a[0] in ('open','start') for a in self.actions))

    def test_sleep_recovery_resumes_resource_route_only_after_blue_fill_proof(self):
        self.prepare_sleep_test()
        active, stopped = self.sleep_frame(),self.sleep_frame(off=True)
        for at, image in ((100,active),(104,active),(108,stopped),(112,stopped),(116,stopped)):
            self.sleep_step(image,at)
        self.assertTrue(self.training.state['sailing_sleep_recovery'])
        self.assertEqual([a[0] for a in self.actions],['stop'])
        filled = self.sleep_frame(off=True,filled=True)
        self.assertTrue(self.meter.blue_fill(filled))
        self.sleep_step(filled,120)
        self.assertNotIn('sailing_sleep_recovery',self.training.state)
        self.assertEqual(self.sailing.phase,'idle')
        self.sleep_step(self.sleep_frame(off=True,awake=True),124)
        self.assertEqual(self.actions[-1][0],'open')
        self.sleep_step(frame('menu-three'),128)
        self.sleep_step(frame('menu-three'),132)
        self.assertEqual(self.actions[-1][0],'start')
        for at in (136,140):
            self.sleep_step(self.sleep_frame(awake=True),at)
        self.assertTrue(self.training.state['sailing_verified'])
        self.assertEqual(self.sailing.cycles,1)
        self.assertEqual(self.training.state['mode'],'farming')

    def test_pending_sleep_recovery_survives_restart_without_starting_the_route(self):
        self.prepare_sleep_test()
        active = self.sleep_frame()
        self.sleep_step(active,100)
        self.sleep_step(active,104)
        restored = BossTraining(ROOT,Reader(),self.events.append,state_path=self.path)
        self.assertTrue(restored.state['sailing_sleep_recovery'])
        self.training = restored
        self.sailing = TrainingSailing(ROOT,restored,self.click,self.events.append,self.meter)
        self.actions.clear()
        for at in (200,204):
            self.sleep_step(self.sleep_frame(off=True),at)
        self.assertEqual(self.actions,[])
        self.assertTrue(restored.state['sailing_sleep_recovery'])
        self.assertEqual(restored.state['boss_level'],1604)
        self.assertEqual(restored.state['start_level'],1535)

    def test_active_fill_and_unknown_dial_do_not_interrupt_verified_sailing(self):
        self.prepare_sleep_test()
        filled = self.sleep_frame(filled=True)
        unknown = self.sleep_frame()
        unknown[947:1063,308:405] = 0
        for image in (filled,unknown,self.sleep_frame(awake=True)):
            self.assertTrue(self.sailing.running(image))
            for at in (100,104,108):
                self.sleep_step(image,at)
        self.assertEqual(self.actions,[])
        self.meter.step.assert_not_called()
        self.assertNotIn('sailing_sleep_recovery',self.training.state)

    def test_rank40_or_disabled_training_cancels_pending_wake_without_starting_fishing(self):
        self.prepare_sleep_test()
        self.sleep_step(self.sleep_frame(),100)
        self.sleep_step(self.sleep_frame(),104)
        for at in (108,112):
            self.sleep_step(self.sleep_frame(off=True),at,wanted=False)
        self.meter.step.assert_not_called()
        self.assertNotIn('sailing_sleep_recovery',self.training.state)
        self.assertFalse(self.training.state['sailing_enabled'])
        self.assertEqual([a[0] for a in self.actions],['stop'])

    def test_unconfirmed_wake_times_out_without_clearing_progress_or_starting_sailing(self):
        self.prepare_sleep_test()
        self.sleep_step(self.sleep_frame(),100)
        self.sleep_step(self.sleep_frame(),104)
        saved = self.path.read_bytes()
        with self.assertRaisesRegex(RuntimeError,'保留练级记录'):
            self.sleep_step(self.sleep_frame(),196)
        self.assertEqual(self.path.read_bytes(),saved)
        self.assertEqual([a[0] for a in self.actions],['stop'])

    def test_reaching_saved_boss_level_target_during_wake_skips_meter_preparation(self):
        self.prepare_sleep_test()
        self.sleep_step(self.sleep_frame(),100)
        self.sleep_step(self.sleep_frame(),104)
        self.sleep_step(self.sleep_frame(off=True),108)
        self.training.player_level = 1584
        with patch.object(self.training,'levels',return_value=(1584,None)):
            self.sleep_step(self.sleep_frame(off=True),112)
            self.sleep_step(self.sleep_frame(off=True),116)
        self.meter.step.assert_not_called()
        self.assertNotIn('sailing_sleep_recovery',self.training.state)
        self.assertFalse(self.training.state['sailing_enabled'])


class SailingWorkerTests(unittest.TestCase):
    def run_worker(self,disabled=False,disable_live=False,fresh_changed=False,stop_fresh=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT/'assets',root/'assets')
            config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            rule = deepcopy(next(r for r in config['rules'] if r['id']=='driftinggirl_parallel_world'))
            rule['enabled'] = not disabled
            passive = deepcopy(rule)
            passive.update(id='passive-test',stage='never',enabled=True)
            passive.pop('training',None)
            config.update(rules=[rule,passive],poll_interval=.1)
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            t = BossTraining(ROOT,Reader(),lambda _:None,state_path=root/'training_state.json')
            t.begin(38,1535,1604)
            t.set_sailing(True,verified=True)
            taps, observations = [],[]
            app = None

            class Device:
                def __init__(self,_,stop):
                    self.stop,self.io_lock = stop,threading.RLock()
                    self.image,self.index,self.screen = None,0,'active'

                def screenshot(self):
                    self.index += 1
                    if fresh_changed and self.index >= 2:
                        self.screen = 'stopped'
                    self.image = frame(self.screen).copy()
                    if disable_live and self.index == 1:
                        app.change_feature({'id':'parallel_world','enabled':False})
                    if stop_fresh and self.index == 2 or self.index >= 10:
                        self.stop.set()
                    record = json.loads((root/'training_state.json').read_text())
                    observations.append((app.state['applied_revision'],record['sailing_enabled']))
                    return self.image

                def click(self,button):
                    taps.append((button.button[0]+5,button.button[1]+5))
                    self.screen = 'stopped'

                def stuck_record_add(self,_):
                    pass

            def create_world(*args,**kwargs):
                w = ParallelWorld(*args,**kwargs)
                w.phase,w.rank,w.training.player_level = 'play',38,1537
                w._rank = lambda _:38
                w.training.levels = lambda _,**kwargs:(1537,None)
                step = w.sailing.step
                def immediate(image,wanted):
                    w.sailing.next_at = 0
                    return step(image,False if fresh_changed or stop_fresh else wanted)
                w.sailing.step = immediate
                return w

            app = Clicker(root,Device)
            app.digit_reader = Reader()
            try:
                with patch('app.ParallelWorld',side_effect=create_world):
                    app.start(preview=False)
                    app.thread.join(7)
                self.assertFalse(app.running())
                self.assertIsNone(app.state['last_error'])
                return deepcopy(app.state),taps,observations,json.loads((root/'training_state.json').read_text())
            finally:
                app.stop()
                if app.thread:
                    app.thread.join(3)
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_live_disable_stops_route_before_applying_feature_switch(self):
        state,taps,observations,saved = self.run_worker(disable_live=True)
        self.assertEqual(taps,[(233,45)])
        self.assertEqual(state['applied_revision'],1)
        self.assertIsNone(state['parallel'])
        self.assertTrue(all(not running for revision,running in observations if revision==1))
        self.assertFalse(saved['sailing_enabled'])
        self.assertEqual((saved['mode'],saved['boss_level']),('farming',1604))

    def test_disabled_while_paused_drains_saved_route_on_next_start(self):
        state,taps,_,saved = self.run_worker(disabled=True)
        self.assertEqual(taps,[(233,45)])
        self.assertIsNone(state['parallel'])
        self.assertFalse(saved['sailing_enabled'])
        self.assertEqual(saved['start_level'],1535)

    def test_worker_rechecks_fresh_auto_state_and_never_opens_menu_to_stop(self):
        _,taps,_,saved = self.run_worker(fresh_changed=True)
        self.assertEqual(taps,[])
        self.assertFalse(saved['sailing_enabled'])

    def test_user_stop_during_fresh_validation_preserves_owned_route_record(self):
        _,taps,_,saved = self.run_worker(stop_fresh=True)
        self.assertEqual(taps,[])
        self.assertTrue(saved['sailing_enabled'])


if __name__ == '__main__':
    unittest.main()
