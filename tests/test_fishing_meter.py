"""Replay native idle/casting states and the guarded long-press worker path."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import Clicker
from boss_training import BossTraining
from digit_ocr import DigitReader
from fishing_meter import FishingMeter
from parallel_world import ParallelWorld
from tools.configure_parallel_world import RULE

ROOT = Path(__file__).resolve().parents[1]


def frame(name):
    if name.endswith('-max'):
        # Derived frames simulate a box at the upper stop for control-flow
        # tests. They are not evidence of a maximum in the native recording.
        image = frame(name[:-4]).copy()
        x,y,X,Y = FishingMeter.box(FishingMeter.__new__(FishingMeter),image)
        crop = image[y:Y,x:X].copy()
        image[y:Y,x:X] = image[y-1:y,x:X]
        top = FishingMeter.TOP-(Y-y)//2
        image[top:top+Y-y,x:X] = crop
        return image
    return np.asarray(Image.open(ROOT/'tests/fixtures/fishing-meter'/(name+'.png')).convert('RGB'))


def same_place_fill(name, source):
    image,before = frame(name).copy(),frame(source)
    image[9:59,71:136] = before[9:59,71:136]
    image[12:34,286:434] = before[12:34,286:434]
    return image


class FishingMeterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.actions, self.events = [], []
        self.training = BossTraining(ROOT, self.reader, self.events.append,
                                     state_path=Path(self.temp.name)/'training_state.json')
        self.meter = FishingMeter(ROOT, self.training,
                                  lambda *a: self.actions.append(a) or True, self.events.append)
        self.clock = 100.

    def tearDown(self):
        self.temp.cleanup()

    def maximum_frame(self, name):
        """Supply an independently established limit for derived control-flow frames.

        These synthetic frames move a gold box to the top. Its position alone
        must never establish MAX in production; actual endpoint recognition is
        covered by the unmodified near-upper and night-meter recordings.
        """
        image = frame(name)
        selected = self.meter.selected(image,self.meter.box(image))
        self.meter.range_cache = dict(version=1,context=self.meter._context(image),
            high=selected,high_at=time.time(),low=1195,low_at=time.time())
        return image

    def step(self, name):
        self.meter.next_action_at = 0
        self.clock += 4
        with patch('fishing_meter.time.monotonic',return_value=self.clock):
            return self.meter.step(self.maximum_frame(name) if name.endswith('-max') else frame(name))

    def test_native_sleep_ready_levels_and_casting_are_distinct(self):
        for name, value, player, icon in (('sleep',1195,1348,'sleep'), ('ready',1343,1349,'hook'),
                                         ('ready-shifted',1348,1353,'hook')):
            a = frame(name)
            self.assertEqual(self.meter.idle_icon(a), icon)
            self.assertFalse(self.meter.busy(a))
            data = self.meter.read(a)
            self.assertEqual((data['value'], data['player'], data['low'], data['high']),
                             (value,player,1195,1421))
            self.assertEqual(data['goal'],data['player'])
            self.assertIs(data['adaptive'],True)
        for name in ('casting','active'):
            a = frame(name)
            self.assertTrue(self.meter.busy(a), name)
            self.assertIsNone(self.meter.read(a), name)
            self.assertFalse(self.meter.validate('start', a, {}), name)

    def test_native_low_active_meter_is_read_only_and_only_waiting_cast_is_cancellable(self):
        waiting,bite = frame('low-meter-waiting'),frame('low-meter-bite')
        for native,player in ((waiting,1712),(bite,1716)):
            data = self.meter.measure(native)
            self.assertEqual((data['value'],data['player'],data['high']),(1580,player,1810))
            self.assertIsNone(data['low'])
            self.assertFalse(self.meter.validate('drag',native,data))
            self.assertFalse(self.meter.validate('start',native,data))
        data = self.meter.cast_data(waiting)
        self.assertTrue(self.meter.waiting_cast(waiting))
        self.assertTrue(self.meter.validate('cancel',waiting,data))
        self.assertFalse(self.meter.validate('cancel',bite,data))
        self.assertFalse(self.meter.waiting_cast(bite))
        self.assertFalse(self.meter.cancellable_cast(bite))
        short,warning = frame('short-wait'),frame('prebattle-stop')
        self.assertTrue(self.meter.cancellable_cast(warning))
        self.assertTrue(self.meter.validate('cancel',warning,self.meter.cast_data(short)))
        self.assertFalse(self.meter.validate('cancel',waiting,{**data,'value':1600}))
        self.assertFalse(self.meter.validate('cancel',waiting,{**data,'digit_strokes':'changed'}))
        for name in ('fill-hook','fill-catch'):
            self.assertFalse(self.meter.waiting_cast(frame(name)))
            self.assertFalse(self.meter.cancellable_cast(frame(name)))

    def test_checked_adaptive_meter_accepts_player_level_within_ten(self):
        native=frame('ready-shifted')
        data=self.meter.read(native)
        self.assertEqual((data['value'],data['player'],data['goal']),(1348,1353,1353))
        self.assertTrue(self.meter.close_enough(data))
        self.assertTrue(self.meter.validate('start',native,data))
        self.step('ready-shifted');self.step('ready-shifted')
        self.assertEqual([a[0] for a in self.actions],['start'])

    def test_adaptive_checkbox_must_be_visible_again_in_the_input_frame(self):
        before=frame('ready')
        plan=self.meter.plan(self.meter.read(before))
        selected=self.maximum_frame('ready-max')
        data=self.meter.read(selected)
        self.assertTrue(self.meter.validate('start',selected,data))
        for original in (before,selected):
            hidden=original.copy()
            hidden[881:913,276:310]=0
            # Even a known main screen or green sailing AUTO cannot replace
            # the distinct dial checkbox proof.
            hidden[4:65,186:254]=frame('sailing-sleep-bottom')[4:65,186:254]
            self.assertIsNone(self.meter.measure(hidden,main_known=True))
            self.assertFalse(self.meter.validate('drag',hidden,plan))
            self.assertFalse(self.meter.validate('start',hidden,data))
        self.assertFalse(self.meter.validate('start',selected,{**data,'adaptive':False}))
        self.assertFalse(self.meter.validate('start',selected,{**data,'context':'changed'}))

    def test_near_upper_stop_cannot_turn_a_covered_label_into_a_false_maximum(self):
        near=frame('near-upper-stop')
        data=self.meter.read(near)
        self.assertEqual((data['value'],data['y']),(1802,939))
        self.assertIsNone(data['high'])
        self.assertEqual(self.meter.plan(data)['kind'],'reveal')
        self.assertFalse(self.meter.validate('start',near,data))
        self.meter.read(frame('near-upper-range'))
        data=self.meter.read(near)
        self.assertEqual((data['high'],data['goal']),(1810,1810))
        self.assertEqual(self.meter.plan(data)['to_y'],
                         self.meter.TOP-self.meter.MAX_DRAG_OVERSHOOT)
        self.assertTrue(self.meter.close_enough(data))

    def test_player_target_requires_fresh_range_but_not_maximum(self):
        data=dict(value=105,player=105,low=100,high=200,x=245,y=1060,goal=105,adaptive=True)
        self.assertTrue(self.meter.close_enough(data))
        self.assertFalse(self.meter.close_enough({**data,'value':200}))
        self.assertFalse(self.meter.close_enough({**data,'high':None}))
        self.assertTrue(self.meter.close_enough({**data,'value':200,'player':220,'goal':200}))
        self.assertTrue(self.meter.close_enough({**data,'low':200,'value':200,'player':220,'y':1085}))
        self.assertGreater(self.meter.plan({**data,'value':150,'y':980})['to_y'],980)

    def test_old_filled_cast_can_only_be_stopped_for_verified_low_meter_calibration(self):
        self.training.begin(39,1623,1810)
        native = frame('filled-prebattle')
        self.assertTrue(self.meter.blue_fill(native))
        self.assertFalse(self.meter.cancellable_cast(native))
        data = self.meter.measure(native)
        self.assertEqual((data['value'],data['player'],data['goal']),(1709,1752,1752))
        self.training.state.update(sailing_sleep_recovery=True,sailing_meter_adjustment=True)
        self.meter.current,self.meter.player,self.meter.goal = data['value'],data['player'],data['goal']
        self.assertTrue(self.meter.cancellable_cast(native))
        self.assertTrue(self.meter.validate('cancel',native,self.meter.cast_data(native)))
        self.assertFalse(self.meter.validate('drag',native,data))
        self.assertFalse(self.meter.validate('start',native,data))
        self.assertFalse(self.meter.cancellable_cast(frame('filled-combat')))
        self.meter.current = self.meter.goal-5
        self.assertFalse(self.meter.cancellable_cast(native))
        self.meter.current = data['value']
        self.training.options['meter_auto_adjust'] = False
        self.assertFalse(self.meter.cancellable_cast(native))

    def test_fast_cancel_scan_waits_for_a_fresh_window_and_rejects_route_changes(self):
        self.training.begin(39,1623,1810)
        self.training.state.update(sailing_sleep_recovery=True,sailing_meter_adjustment=True)
        before,native = frame('scan-combat'),frame('filled-prebattle')
        data = self.meter.measure(native)
        self.meter.current,self.meter.player,self.meter.goal = data['value'],data['player'],data['goal']
        expected = self.meter.cast_data(native)
        self.assertFalse(self.meter.cancellable_cast(before))
        captures = iter((before,native))
        found = self.meter.scan_cancel_frame(lambda:next(captures),lambda _:False,expected,lambda:False)
        self.assertIs(found,native)
        self.assertIsNone(self.meter.scan_cancel_frame(lambda:native,lambda _:True,expected,lambda:False))
        self.assertIsNone(self.meter.scan_cancel_frame(lambda:native,lambda _:False,expected,lambda:True))
        changed = {**expected,'digit_strokes':'changed'}
        self.assertIsNone(self.meter.scan_cancel_frame(lambda:native,lambda _:False,changed,lambda:False))
        count = [0]
        def capture_busy():
            count[0] += 1
            return before
        with patch('fishing_meter.time.monotonic',return_value=100):
            self.assertIsNone(self.meter.scan_cancel_frame(capture_busy,lambda _:False,expected,lambda:False))
        self.assertEqual(count[0],40)
        self.assertEqual(self.actions,[])

    def test_native_moving_background_and_countdown_keep_the_meter_context(self):
        first,second = frame('moving-context-1'),frame('moving-context-2')
        self.assertEqual(self.meter._context(first),self.meter._context(second))
        self.assertNotEqual(self.meter._context(first),self.meter._context(frame('moving-other-place')))
        self.assertNotEqual(self.meter._context(first),self.meter._context(frame('wrong-world')))
        self.training.begin(39,1623,1810)
        for native in (first,second):
            self.assertTrue(self.meter.world_matches(native))
            self.assertEqual(self.meter.measure(native)['value'],1580)
        wrong = frame('wrong-world')
        self.assertFalse(self.meter.world_matches(wrong))
        self.assertIsNone(self.meter.measure(wrong))
        self.meter.phase = 'starting'
        self.assertFalse(self.meter.confirm_start(wrong))
        self.assertFalse(self.meter.step(wrong))
        self.assertEqual(self.actions,[])

    def test_same_place_antialias_change_does_not_abort_the_short_cancel_scan(self):
        first,covered,clear = (frame('same-place-rank39-'+str(i)) for i in (1,2,3))
        self.training.begin(39,1724,1810)
        self.training.state.update(sailing_sleep_recovery=True,sailing_meter_adjustment=True)
        data = self.meter.measure(first)
        self.assertEqual((data['value'],data['player'],data['low'],data['high']),
                         (1674,1724,1524,1810))
        self.meter.current,self.meter.player,self.meter.goal = data['value'],data['player'],data['goal']
        expected = self.meter.cast_data(first)
        self.assertEqual(expected['context'],self.meter.cast_data(covered)['context'])
        self.assertEqual(expected['context'],self.meter.cast_data(clear)['context'])
        self.assertFalse(self.meter.cancellable_cast(covered))
        self.assertTrue(self.meter.cancellable_cast(clear))
        captures = iter((covered,clear))
        found = self.meter.scan_cancel_frame(lambda:next(captures),lambda _:False,
                                           expected,lambda:False)
        self.assertIs(found,clear)
        for key in ('digit_strokes','world_strokes','context'):
            self.assertIsNone(self.meter.scan_cancel_frame(lambda:clear,lambda _:False,
                              {**expected,key:'changed'},lambda:False),key)
        self.assertIsNone(self.meter.scan_cancel_frame(lambda:clear,lambda _:True,
                                                      expected,lambda:False))
        self.assertEqual(self.actions,[])

    def test_port_anchor_is_not_a_fishing_state_after_selling(self):
        port = frame('port-after-sale')
        self.assertTrue(self.meter.at_port(port))
        self.assertEqual(self.meter.observe(port),'docked')
        self.assertIsNone(self.meter.measure(port))
        self.assertIsNone(self.meter.read(port))
        self.assertFalse(self.meter.step(port))
        self.assertFalse(self.meter.validate('start',port,{}))
        self.assertFalse(self.meter.at_port(frame('moving-context-1')))
        self.assertFalse(self.meter.at_port(frame('ready')))
        self.assertEqual(self.actions,[])

    def test_only_identical_verified_rank_strokes_reuse_the_rank_reading(self):
        self.training.begin(39,1623,1810)
        with patch.object(self.training,'_number',wraps=self.training._number) as read:
            self.assertTrue(self.meter.world_matches(frame('moving-context-1')))
            self.assertTrue(self.meter.world_matches(frame('moving-context-2')))
            self.assertEqual(read.call_count,1)
            self.assertFalse(self.meter.world_matches(frame('wrong-world')))
            self.assertEqual(read.call_count,2)
        p = ParallelWorld(ROOT,lambda *a:None,lambda *a:None,self.events.append,self.reader,
                          training_state_path=self.training.path,fishing_action=lambda *a:True,
                          sailing_action=lambda *a:True)
        p.phase,p.rank = 'play',39
        p.training.state['sailing_sleep_recovery'] = True
        p.sailing.step = unittest.mock.Mock(return_value=True)
        with patch.object(p,'_rank',side_effect=AssertionError('full OCR before waiting-cast handling')):
            self.assertTrue(p.step(frame('moving-context-1')))
        p.sailing.step.assert_called_once()

    def test_wrong_world_cannot_raise_the_training_level_or_retry_the_boss(self):
        self.training.begin(39,1623,1810)
        p = ParallelWorld(ROOT,lambda *a:self.actions.append(a),lambda *a:None,self.events.append,self.reader,
                          training_state_path=self.training.path,fishing_action=lambda *a:True,
                          sailing_action=lambda *a:True)
        p.phase,p.rank = 'play',39
        p.training.player_level = 1736
        with patch.object(p,'_rank',return_value=None),patch.object(p.training,'evaluate',side_effect=AssertionError('wrong world advanced training')):
            self.assertTrue(p.step(frame('wrong-world')))
        self.assertEqual(p.training.player_level,1736)
        self.assertEqual(p.training.state['boss_level'],1810)
        self.assertEqual(self.actions,[])

    def test_sailing_sleep_at_bottom_retains_the_leading_digit_of_the_endpoint(self):
        a = frame('sailing-sleep-bottom')
        self.assertEqual(self.meter.idle_icon(a), 'sleep')
        data = self.meter.read(a)
        self.assertEqual((data['value'],data['low'],data['high'],data['player']),
                         (1359,1359,1604,1560))
        self.assertEqual(data['goal'],1560)
        self.assertFalse(self.meter.close_enough(data))
        self.assertEqual(self.meter.observe(a),'sleeping')
        self.assertFalse(self.meter.status()['blue_fill'])
        self.assertEqual(self.meter.confirmed_starts,0)

    def test_real_middle_connector_cannot_hide_the_idle_hook(self):
        native = frame('hook-middle-connector')
        self.assertEqual(self.meter.idle_icon(native),'hook')
        self.assertFalse(self.meter.busy(native))
        data = self.meter.read(native)
        self.assertEqual((data['value'],data['player'],data['low'],data['high']),(1666,1747,1524,1810))
        plan = self.meter.plan(data)
        self.assertEqual(plan['goal'],1747)
        self.assertGreater(plan['to_y'],self.meter.TOP)
        self.assertTrue(self.meter.validate('drag',native,plan))
        self.assertEqual(self.meter.observe(frame('fill-hook')),'filled')
        self.assertEqual(self.meter.confirmed_starts,0)
        self.assertEqual(self.meter.observe(np.zeros((1280,720,3),dtype='uint8')),'unknown')

    def test_real_range_label_ignores_animated_ring_background(self):
        self.training.begin(39,1623,1810)
        native = frame('range-background')
        self.assertTrue(self.meter.world_matches(native))
        data = self.meter.read(native)
        self.assertEqual((data['value'],data['player'],data['low'],data['high']),
                         (1666,1747,1524,1810))
        self.assertTrue(self.meter.validate('drag',native,self.meter.plan(data)))
        missing = native.copy()
        missing[1070:1100,234:284] = 0
        self.assertIsNone(self.meter.read(missing))

    def test_gold_box_overlapping_upper_label_reuses_only_recent_same_location_limit(self):
        before, overlapping = frame('sailing-sleep-bottom'),frame('sleep-top-overlap')
        self.meter.read(before)
        data = self.meter.read(overlapping)
        self.assertEqual((data['value'],data['player'],data['low'],data['high']),(1563,1562,1350,1604))
        plan = self.meter.plan(data)
        self.assertEqual((plan['kind'],plan['goal'],plan['to_y']),('drag',1562,960))
        self.assertTrue(self.meter.close_enough(data))
        self.assertFalse(self.meter.validate('drag',overlapping,plan))
        changed = overlapping.copy()
        changed[12:62,302:430] = 0
        self.assertIsNone(self.meter.read(changed)['high'])
        self.assertFalse(self.meter.validate('drag',changed,plan))
        self.meter.read(before)
        self.meter.range_cache['high_at'] -= 181
        self.assertIsNone(self.meter.read(overlapping)['high'])
        self.assertFalse(self.meter.validate('start',overlapping,{}))

    def test_restart_with_hidden_limit_reveals_labels_before_selecting_or_starting(self):
        overlapping = frame('sleep-top-overlap')
        data = self.meter.read(overlapping)
        self.assertEqual(data['value'],1563)
        self.assertIsNone(data['high'])
        plan = self.meter.plan(data)
        self.assertEqual((plan['kind'],plan['to_y']),('reveal',1010))
        self.assertTrue(self.meter.validate('reveal',overlapping,plan))
        self.assertFalse(self.meter.validate('drag',overlapping,plan))
        self.assertFalse(self.meter.validate('start',overlapping,{}))
        with patch.object(self.training,'_number',side_effect=[1562,1350,None]):
            self.assertIsNone(self.meter.read(frame('ready')))

    def test_pending_wake_restores_verified_range_without_resetting_training_identity(self):
        self.training.begin(38,1535,1604)
        self.training.state['sailing_sleep_recovery'] = True
        original = {k:self.training.state[k] for k in ('rank','boss_level','start_level','started_at')}
        self.meter.read(frame('sailing-sleep-bottom'))
        restored = BossTraining(ROOT,self.reader,self.events.append,state_path=self.training.path)
        meter = FishingMeter(ROOT,restored,None,self.events.append)
        self.assertEqual(meter.read(frame('sleep-top-overlap'))['high'],1604)
        self.assertTrue(restored.state['sailing_sleep_recovery'])
        self.assertTrue(all(restored.state[k]==v for k,v in original.items()))
        self.assertEqual(meter.confirmed_starts,0)

    def test_drag_then_start_and_confirm_without_repeating_during_cast(self):
        screen = ['sleep']
        def action(kind, data):
            self.assertTrue(self.meter.validate(kind, frame(screen[0]), data))
            self.actions.append((kind,data))
            screen[0] = 'ready' if kind == 'drag' else 'fill-hook'
            return True if kind == 'drag' else [same_place_fill('fill-hook','ready')]*2
        self.meter.action = action
        for _ in range(8):
            with patch.object(self.meter,'post_start_evidence',return_value=True):
                if screen[0] == 'fill-hook':
                    self.clock += 4
                    with patch('fishing_meter.time.monotonic',return_value=self.clock):
                        self.meter.step(same_place_fill('fill-hook','ready'))
                else:
                    self.step(screen[0])
        self.assertEqual([a[0] for a in self.actions], ['drag','start'])
        drag = self.actions[0][1]
        self.assertEqual((drag['x'],drag['y'],drag['to_y']), (245,1085,984))
        self.assertEqual(self.meter.phase, 'active')
        self.assertEqual(self.meter.confirmed_starts, 1)

    def test_restart_with_selected_level_still_sleeping_needs_long_press(self):
        self.meter.action = lambda *a:self.actions.append(a) or [same_place_fill('fill-hook','sleep-selected')]*2
        a = self.maximum_frame('sleep-selected-max')
        self.assertEqual(self.meter.idle_icon(a), 'sleep')
        data = self.meter.read(a)
        self.assertEqual((data['value'],data['player']), (1343,1353))
        self.assertTrue(self.meter.close_enough(data))
        self.assertTrue(self.meter.validate('start',a,data))
        self.step('sleep-selected-max')
        self.step('sleep-selected-max')
        self.assertEqual([action[0] for action in self.actions], ['start'])
        for now in (self.clock+1,self.clock+2.5,self.clock+4):
            with patch('fishing_meter.time.monotonic',return_value=now), patch.object(self.meter,'post_start_evidence',return_value=True):
                self.meter.confirm_start(same_place_fill('fill-hook','sleep-selected'))
        self.assertEqual(self.meter.confirmed_starts, 1)

    def test_post_release_fill_requires_three_new_frames_and_three_seconds(self):
        self.meter.action = lambda *_: [same_place_fill('fill-hook','ready-shifted'),same_place_fill('fill-catch','ready-shifted')]
        self.step('ready-shifted-max');self.step('ready-shifted-max')
        self.assertEqual(self.meter.confirmed_starts,0)
        for now in (self.clock+1,self.clock+2.5,self.clock+4):
            with patch('fishing_meter.time.monotonic',return_value=now), patch.object(self.meter,'post_start_evidence',return_value=True):
                self.meter.confirm_start(same_place_fill('fill-hook','ready-shifted'))
        self.assertEqual(self.meter.confirmed_starts,1)
        self.assertEqual(self.meter.phase,'active')

    def test_during_fill_followed_by_a_gray_hook_does_not_confirm_start(self):
        self.meter.action = lambda *_: [frame('fill-hook'),frame('ready-shifted')]
        self.step('ready-shifted-max')
        self.step('ready-shifted-max')
        self.assertEqual(self.meter.confirmed_starts, 0)
        self.assertEqual(self.meter.phase, 'starting')
        self.assertFalse(self.meter.confirm_start(frame('fill-hook')))

    def test_after_fill_without_during_fill_or_changed_rank_does_not_confirm_start(self):
        wrong = frame('fill-hook').copy()
        wrong[9:59,71:136] = frame('filled-prebattle')[9:59,71:136]
        self.training.begin(37,1300,1421)
        for captures in ([frame('ready-shifted'),frame('fill-hook')],
                         [frame('fill-hook')], [frame('fill-hook'),wrong]):
            self.actions.clear()
            self.meter.action = lambda kind,data,captures=captures:self.actions.append(kind) or captures
            self.meter.starts,self.meter.phase,self.meter.idle_since = 0,'detect',None
            self.step('ready-shifted-max')
            self.step('ready-shifted-max')
            self.assertEqual(self.meter.confirmed_starts,0)
            self.assertFalse(self.meter.start_fill_verified)
            self.assertEqual(self.actions,['start'])

    def test_fishing_result_without_fill_does_not_confirm_long_press(self):
        result = np.asarray(Image.open(ROOT/'tests/fixtures/training/success.png').convert('RGB'))
        self.assertFalse(self.training.boss_mode(result))
        self.meter.action = lambda *_: result
        self.step('ready-max')
        self.step('ready-max')
        self.assertEqual(self.meter.confirmed_starts,0)
        self.assertEqual(self.meter.phase,'starting')
        self.assertFalse(self.meter.confirm_start(frame('fill-result')))
        self.assertEqual(self.meter.confirmed_starts,0)

    def test_native_fill_distinguishes_wood_zzz_hook_and_partial_fills(self):
        for name in ('sleep','ready','ready-shifted','sleep-selected','casting','active'):
            self.assertFalse(self.meter.blue_fill(frame(name)),name)
        for name in ('fill-hook','fill-catch','fill-empty','fill-result'):
            a = frame(name)
            self.assertTrue(self.meter.blue_fill(a),name)
            self.assertTrue(self.meter.busy(a),name)
            self.assertIsNone(self.meter.read(a),name)
            self.assertFalse(self.meter.validate('start',a,{}),name)
            self.assertFalse(self.meter.validate('drag',a,{}),name)
        self.assertFalse(self.meter.blue_fill((frame('fill-empty')*.4).astype(np.uint8)))
        for path in ('tests/fixtures/equipment/top.png','tests/fixtures/parallel/parallel-map-rank2.png',
                     'tests/fixtures/parallel-skills/parallel-skill-panel.png'):
            other = np.asarray(Image.open(ROOT/path).convert('RGB'))
            self.assertFalse(self.meter.blue_fill(other),path)

    def test_fill_prevents_input_even_when_auto_is_bright_between_catches(self):
        a = frame('fill-hook').copy()
        a[885:907,212:274] = frame('ready-shifted')[885:907,212:274]
        self.assertGreater(float((a[885:907,212:274].max(axis=2)>=210).mean()),.08)
        self.assertTrue(self.meter.busy(a))
        for _ in range(3):
            self.assertFalse(self.meter.step(a))
        self.assertTrue(self.meter.status()['blue_fill'])
        self.assertEqual(self.actions,[])
        # A cyan shape on one side, or an effect outside the rim, is not a fill.
        other = frame('ready-shifted').copy()
        other[982:1068,301:321] = a[982:1068,301:321]
        self.assertFalse(self.meter.blue_fill(other))

    def test_idle_must_remain_clear_before_long_press(self):
        with patch('fishing_meter.time.monotonic',return_value=100.):
            self.assertTrue(self.meter.step(self.maximum_frame('ready-shifted-max')))
        with patch('fishing_meter.time.monotonic',return_value=102.):
            self.assertTrue(self.meter.step(self.maximum_frame('ready-shifted-max')))
        self.assertEqual(self.actions,[])
        self.assertFalse(self.meter.step(frame('fill-hook')))
        with patch('fishing_meter.time.monotonic',return_value=110.):
            self.assertTrue(self.meter.step(self.maximum_frame('ready-shifted-max')))
        with patch('fishing_meter.time.monotonic',return_value=112.):
            self.assertTrue(self.meter.step(self.maximum_frame('ready-shifted-max')))
        self.assertEqual(self.actions,[])
        with patch('fishing_meter.time.monotonic',return_value=114.):
            self.assertTrue(self.meter.step(self.maximum_frame('ready-shifted-max')))
        self.assertEqual([a[0] for a in self.actions],['start'])

    def test_dimmed_auto_without_fill_cannot_confirm_long_press(self):
        self.meter.phase,self.meter.since = 'starting',time.monotonic()
        self.assertTrue(self.meter.busy(frame('casting')))
        self.assertFalse(self.meter.confirm_start(frame('casting')))
        self.assertTrue(self.meter.step(frame('casting')))
        self.assertEqual(self.meter.phase,'starting')
        self.assertEqual(self.meter.confirmed_starts,0)
        self.meter.since -= 31
        with self.assertRaisesRegex(RuntimeError,'未连续确认真实填充或战斗'):
            self.meter.step(frame('casting'))
        self.assertEqual(self.actions,[])

    def test_unknown_dial_never_claims_a_fishing_start(self):
        a = frame('ready').copy()
        a[947:1063,308:405] = 0
        self.assertTrue(self.training.main(a))
        self.assertFalse(self.meter.step(a))
        self.assertEqual(self.meter.phase, 'unknown')
        self.assertEqual(self.meter.confirmed_starts, 0)
        self.meter.phase, self.meter.since = 'starting',time.monotonic()
        self.assertTrue(self.meter.step(a))
        self.assertEqual(self.meter.phase, 'starting')
        self.meter.since -= 31
        with self.assertRaisesRegex(RuntimeError,'未连续确认真实填充或战斗'):
            self.meter.step(a)
        self.assertEqual(self.actions, [])

    def test_fresh_frame_changed_value_player_or_endpoint_prevents_input(self):
        a = frame('sleep')
        data = self.meter.read(a)
        plan = self.meter.plan(data)
        self.assertTrue(self.meter.validate('drag', a, plan))
        self.assertFalse(self.meter.validate('drag', frame('ready'), plan))
        for key, value in (('to_y',990), ('x',360), ('player',1349), ('goal',1343),('adaptive',False)):
            self.assertFalse(self.meter.validate('drag', a, {**plan,key:value}), key)
        self.assertFalse(self.meter.validate('start', a, data))
        ready = self.maximum_frame('ready-max')
        self.assertTrue(self.meter.validate('start', ready, {}))
        with patch.object(self.training, 'boss_mode', return_value=True):
            self.assertFalse(self.meter.validate('start', ready, {}))
            self.assertFalse(self.meter.step(ready))

    def test_menus_dimmed_and_incomplete_ocr_are_rejected(self):
        for path in ('tests/fixtures/equipment/top.png', 'tests/fixtures/parallel/parallel-map-rank2.png',
                     'tests/fixtures/parallel-skills/parallel-skill-panel.png'):
            other = np.asarray(Image.open(ROOT/path).convert('RGB'))
            self.assertIsNone(self.meter.read(other), path)
            self.assertFalse(self.meter.step(other), path)
        self.assertIsNone(self.meter.read((frame('ready')*.4).astype(np.uint8)))
        self.assertFalse(self.meter.step(np.zeros((50,50,3), dtype=np.uint8)))
        with patch.object(self.training, '_number', return_value=None):
            self.assertFalse(self.step('sleep'))
            self.assertFalse(self.step('sleep'))
        self.assertEqual(self.actions, [])
        self.assertEqual(len(self.events), 1)

    def test_level_target_clamps_to_available_range_and_never_sleeps(self):
        data = dict(value=100, player=105, low=100, high=200, x=245, y=1085, goal=101)
        self.assertFalse(self.meter.close_enough(data))
        self.assertTrue(self.meter.close_enough({**data,'value':101}))
        self.assertFalse(self.meter.close_enough({**data,'value':106}))
        self.assertTrue(self.meter.close_enough({**data,'value':200,'player':220}))
        self.assertFalse(self.meter.close_enough({**data,'value':199,'player':220}))
        self.assertTrue(self.meter.close_enough({**data,'value':101,'player':90}))
        self.assertEqual(self.meter.plan({**data,'goal':200})['to_y'], self.meter.TOP)

    def test_unresponsive_drag_or_start_stops_after_bounded_attempts(self):
        for name, kind, limit, text in (('sleep','drag',4,'连续调整'), ('ready-max','start',1,'未连续确认真实填充或战斗')):
            self.meter = FishingMeter(ROOT, self.training,
                                      lambda *a: self.actions.append(a) or True, self.events.append)
            self.actions.clear()
            with self.assertRaisesRegex(RuntimeError, text):
                for _ in range(12):
                    self.meter.since = -100
                    self.step(name)
            self.assertEqual([a[0] for a in self.actions], [kind]*limit)

    def test_training_navigation_and_rank40_take_priority_over_meter(self):
        p = ParallelWorld(ROOT, lambda *a:self.actions.append(a), lambda *a:self.actions.append(a),
                          self.events.append, self.reader, training_state_path=self.training.path,
                          fishing_action=lambda *a:self.actions.append(a) or True)
        p.phase, p.rank = 'play',37
        p.fishing.IDLE_SECONDS = 0
        p.training.state = dict(mode='farming', rank=37, start_level=1331, boss_level=1421,
                                started_at=time.time()-10000, version=1)
        with patch.object(p, '_rank', return_value=37):
            p.step(frame('sleep'))
            p.step(frame('sleep'))
        self.assertEqual([a[0] for a in self.actions], ['drag'])
        self.actions.clear()
        p.rank, p.completed_rank_40 = 40,True
        with patch.object(p, '_rank', return_value=40):
            self.assertTrue(p.step(frame('sleep')))
        self.assertEqual(p.phase, 'return_start')
        self.assertEqual(self.actions, [])
        self.assertIn('fishing', p.status())

    def test_worker_adjusts_and_holds_but_does_not_confirm_during_only_fill(self):
        root = Path(self.temp.name)
        shutil.copytree(ROOT/'assets', root/'assets')
        config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        config.update(rules=[deepcopy(RULE)], poll_interval=.1)
        config['rules'][0]['training']['auto_sailing'] = False
        (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
        state = dict(version=1, mode='farming', rank=37, start_level=1331,
                     boss_level=1421, started_at=time.time()-1000)
        path = root/'training_state.json'
        path.write_text(json.dumps(state), encoding='utf-8')
        actions = []
        images = {n:frame(n if n in ('sleep','fill-hook') else n+'-max')
                  for n in ('sleep','ready','fill-hook','ready-shifted')}
        class Device:
            def __init__(self, config, stop):
                self.stop, self.io_lock = stop,threading.RLock()
                self.screen, self.image, self.casts = 'sleep',None,0
            def screenshot(self):
                if self.screen == 'ready-shifted':
                    self.casts += 1
                    if self.casts >= 2:
                        self.stop.set()
                self.image = images[self.screen].copy()
                return self.image
            def command(self, *args):
                actions.append(args)
                self.screen = 'ready'
                return b''
            def hold_feedback(self,x,y,milliseconds):
                actions.append(('hold',x,y,milliseconds))
                self.screen = 'ready-shifted'
                # Only the in-gesture screenshot still shows the cast. The
                # post-gesture screenshot is already back at the idle hook.
                return images['fill-hook'].copy()
            def click(self, _):
                raise AssertionError('Fishing must use long presses')
            def stuck_record_add(self, _):
                pass
        app = Clicker(root, Device)
        app.digit_reader = self.reader
        try:
            # The derived top-box frame has a separately arranged upper limit;
            # moving that box alone is no longer accepted as numeric MAX.
            with patch.object(FishingMeter,'_cached_limit',
                              side_effect=lambda context,name:1343 if name=='high' else 1195):
                app.start(preview=False, timeout=45)
                app.thread.join(30)
            self.assertFalse(app.running())
            self.assertIsNone(app.state['last_error'])
            self.assertEqual(len(actions), 2)
            self.assertEqual(actions[0][0:3], ('shell','input','swipe'))
            self.assertEqual(actions[0][-1], '900')
            self.assertEqual(actions[1], ('hold',360,1010,1500))
            self.assertEqual(app.state['parallel']['fishing']['confirmed_starts'], 0)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), state)
        finally:
            app.stop()
            if app.thread:
                app.thread.join(3)
            for handler in list(app.log.handlers):
                handler.close()
                app.log.removeHandler(handler)

    def test_worker_rejects_meter_input_when_the_fresh_frame_has_sailing_on(self):
        root = Path(self.temp.name)
        shutil.copytree(ROOT/'assets',root/'assets')
        config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        config.update(rules=[deepcopy(RULE)],poll_interval=.1)
        config['rules'][0]['training']['auto_sailing'] = False
        (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
        record = dict(version=1,mode='farming',rank=37,start_level=1331,
                      boss_level=1421,started_at=time.time()-1000)
        (root/'training_state.json').write_text(json.dumps(record),encoding='utf-8')
        off = frame('ready')
        on = off.copy()
        on[4:65,186:254] = frame('sailing-sleep-bottom')[4:65,186:254]
        inputs = []

        class Device:
            def __init__(self,_,stop):
                self.stop,self.io_lock = stop,threading.RLock()
                self.image,self.index = None,0
            def screenshot(self):
                self.index += 1
                self.image = (off if self.index < 3 else on).copy()
                if self.index >= 8:
                    self.stop.set()
                return self.image
            def command(self,*args):
                inputs.append(args)
                return b''
            def hold_feedback(self,*args):
                inputs.append(('hold',*args))
                return frame('fill-hook')
            def click(self,button):
                inputs.append(('click',button))
            def stuck_record_add(self,_):
                pass

        app = Clicker(root,Device)
        app.digit_reader = self.reader
        try:
            with patch.object(FishingMeter,'IDLE_SECONDS',0):
                app.start(preview=False,timeout=45)
                app.thread.join(30)
            self.assertFalse(app.running())
            self.assertIsNone(app.state['last_error'])
            self.assertEqual(inputs,[])
            self.assertIn('航巡状态或校准上下文变化',json.dumps(app.status()['events'],ensure_ascii=False))
        finally:
            app.stop()
            if app.thread:
                app.thread.join(3)
            for handler in list(app.log.handlers):
                handler.close()
                app.log.removeHandler(handler)

    def test_pending_calibration_preempts_idle_backpack_ocr(self):
        root=Path(self.temp.name)
        shutil.copytree(ROOT/'assets',root/'assets')
        config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        keys={'driftinggirl_parallel_world','driftinggirl_backpack_sell'}
        config.update(rules=[deepcopy(r) for r in config['rules'] if r['id'] in keys],poll_interval=.1)
        for rule in config['rules']:rule['enabled']=True
        (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
        native=frame('low-meter-waiting');inputs=[]
        class Device:
            def __init__(self,_,stop):
                self.stop,self.io_lock=stop,threading.RLock();self.index=0;self.image=None
            def screenshot(self):
                self.index+=1;self.image=native.copy()
                if self.index>=4:self.stop.set()
                return self.image
            def command(self,*args):inputs.append(args)
            def click(self,button):inputs.append(button)
            def stuck_record_add(self,_):pass
        def world(*args,**kwargs):
            p=ParallelWorld(*args,**kwargs);p.phase,p.rank='play',39
            p.training.begin(39,1623,1810)
            p.training.state['sailing_sleep_recovery']=True
            p.step=unittest.mock.Mock(return_value=True)
            return p
        app=Clicker(root,Device);app.digit_reader=self.reader
        try:
            with patch('app.ParallelWorld',side_effect=world),patch('app.BackpackSale.step',side_effect=AssertionError('bag OCR delayed short cast')):
                app.start(preview=False);app.thread.join(5)
            self.assertFalse(app.running())
            self.assertIsNone(app.state['last_error'])
            self.assertEqual(inputs,[])
        finally:
            app.stop()
            if app.thread:app.thread.join(2)
            for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)


if __name__ == '__main__':
    unittest.main()
