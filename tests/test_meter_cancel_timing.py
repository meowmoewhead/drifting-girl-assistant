"""Real short cancellation windows, transport age and stopped-route guards."""
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
import cv2
from PIL import Image

from app import Clicker
from boss_training import BossTraining
from digit_ocr import DigitReader
from fishing_meter import FishingMeter
from parallel_world import ParallelWorld
from training_sailing import TrainingSailing
from meter_calibration import MeterCalibration

ROOT = Path(__file__).resolve().parents[1]


def image(name):
    return np.asarray(Image.open(ROOT/'tests/fixtures/fishing-meter'/(name+'.png')).convert('RGB'))


class MeterCancelTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)

    def make_meter(self, directory):
        training = BossTraining(ROOT,self.reader,lambda _:None,state_path=Path(directory)/'training.json')
        training.begin(35,814,1112)
        training.state.update(sailing_enabled=False,sailing_verified=False,
                              sailing_sleep_recovery=True,sailing_meter_adjustment=True)
        meter = FishingMeter(ROOT,training,lambda *_:True,lambda _:None)
        meter.current,meter.player,meter.goal = 934,982,977
        return meter

    def test_real_cancel_window_is_not_combat_or_sleep_and_requires_current_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            meter = self.make_meter(directory)
            before,closed = image('cancel-rank35'),image('cancel-window-closed-rank35')
            self.assertTrue(meter.cancellable_cast(before))
            self.assertFalse(meter.cancellable_cast(closed))
            self.assertFalse(meter.blue_fill(before))
            self.assertIsNone(meter.idle_icon(before))
            data = meter.cast_data(before)
            self.assertTrue(meter.validate('cancel',before,data))
            self.assertFalse(meter.validate('start',before,data))
            meter.training.state['rank'] = 34
            self.assertIsNone(meter.cast_data(before))

    def test_real_filled_dots_require_saved_low_meter_and_all_three_gold_dots(self):
        with tempfile.TemporaryDirectory() as directory:
            meter = self.make_meter(directory)
            meter.current,meter.player,meter.goal = 1038,1079,1074
            native = image('filled-dots-rank35')
            score = cv2.minMaxLoc(cv2.matchTemplate(native[997:1024,385:411],
                meter.assets['waiting-dot'],cv2.TM_CCOEFF_NORMED))[1]
            self.assertLess(score,.94)  # Real old RGB match missed the right dot.
            self.assertTrue(meter.blue_fill(native))
            self.assertFalse(meter.waiting_cast(native))
            self.assertTrue(meter.cancellable_cast(native))
            self.assertTrue(meter.validate('cancel',native,meter.cast_data(native)))
            self.assertFalse(meter.validate('start',native,{}))
            self.assertFalse(meter.validate('drag',native,{}))
            for name in ('sailing_sleep_recovery','sailing_meter_adjustment'):
                meter.training.state[name] = False
                self.assertFalse(meter.cancellable_cast(native),name)
                meter.training.state[name] = True
            meter.goal = meter.current+19
            self.assertFalse(meter.cancellable_cast(native))
            meter.goal = 1074
            meter.training.options['meter_auto_adjust'] = False
            self.assertFalse(meter.cancellable_cast(native))
            meter.training.options['meter_auto_adjust'] = True
            for x in (322,360,398):
                missing = native.copy()
                missing[1000:1023,x-11:x+11] = 0
                self.assertFalse(meter.cancellable_cast(missing),x)
            gray = native.copy()
            dot = gray[1002:1020,389:407]
            dot[:,:,:] = cv2.cvtColor(cv2.cvtColor(dot,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
            self.assertFalse(meter.cancellable_cast(gray))

    def test_real_warning_without_cancel_button_and_combat_remain_forbidden(self):
        with tempfile.TemporaryDirectory() as directory:
            meter = self.make_meter(directory)
            meter.current,meter.player,meter.goal = 1038,1079,1074
            for name in ('filled-line-warning-rank35','filled-combat','fill-result','port-after-sale'):
                native = image(name)
                self.assertFalse(meter.cancellable_cast(native),name)
            native = image('filled-dots-rank35')
            # This older ! frame really has a cancel button, but another RANK
            # may not borrow the current calibration's authorization.
            self.assertFalse(meter.validate('cancel',image('fill-catch'),meter.cast_data(native)))
            no_x = native.copy(); no_x[1066:1110,634:682] = 0
            self.assertFalse(meter.cancellable_cast(no_x))
            with patch.object(meter.training,'boss_mode',return_value=True):
                self.assertFalse(meter.cancellable_cast(native))

    def test_filled_dot_scan_uses_fresh_stopped_frame_without_modifying_range_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            meter = self.make_meter(directory)
            meter.current,meter.player,meter.goal = 1038,1079,1074
            native = image('filled-dots-rank35')
            data,cache = meter.cast_data(native),deepcopy(meter.range_cache)
            before = image('filled-line-warning-rank35')
            # Digit/world/place strokes are identical; lack of X keeps waiting.
            self.assertEqual(meter.cast_data(before),data)
            captures = iter((before,native))
            self.assertIs(meter.scan_cancel_frame(lambda:next(captures),lambda _:False,
                          data,lambda:False),native)
            self.assertEqual(meter.range_cache,cache)

    def test_stage_strokes_change_abandons_window_without_touching_range_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            meter = self.make_meter(directory)
            before = image('cancel-rank35')
            data = meter.cast_data(before)
            changed = before.copy()
            changed[48:66,43:53] = 0
            cache = deepcopy(meter.range_cache)
            self.assertFalse(meter.validate('cancel',changed,data))
            self.assertIsNone(meter.scan_cancel_frame(lambda:changed,lambda _:False,data,lambda:False))
            self.assertEqual(meter.range_cache,cache)

    def test_scan_discards_stale_frame_and_uses_new_window(self):
        with tempfile.TemporaryDirectory() as directory:
            meter = self.make_meter(directory)
            first,second = image('cancel-rank35'),image('cancel-rank35')
            frames = iter((first,second))
            checks = iter((False,True))
            result = meter.scan_cancel_frame(lambda:next(frames),lambda _:False,
                meter.cast_data(first),lambda:False,lambda _:next(checks))
            self.assertIs(result,second)

    def test_scan_deadline_and_stop_are_rechecked_after_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            meter = self.make_meter(directory)
            native = image('cancel-rank35'); data = meter.cast_data(native)
            clock = [100.]
            def late_capture():
                clock[0] += 6.1
                return native
            with patch('fishing_meter.time.monotonic',side_effect=lambda:clock[0]):
                self.assertIsNone(meter.scan_cancel_frame(late_capture,lambda _:False,data,lambda:False))
            stopped = [False]
            def stop_capture():
                stopped[0] = True
                return native
            self.assertIsNone(meter.scan_cancel_frame(stop_capture,lambda _:False,data,lambda:stopped[0]))

    def test_two_attempt_cap_is_preserved_and_is_not_cancellation_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            meter = self.make_meter(directory)
            native = image('cancel-rank35')
            meter.prepare(native)
            meter.next_action_at = 0
            meter.prepare(native)
            meter.next_action_at = 0
            self.assertEqual(meter.cancels,2)
            self.assertEqual(meter.confirmed_starts,0)
            with self.assertRaisesRegex(RuntimeError,'仍未休眠'):
                meter.prepare(native)

    def run_worker(self, *, changed=None, slow_first=False, stop_capture=False, down_error=False,
                   native_name='cancel-rank35', independent=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT/'assets',root/'assets')
            config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            rule_id='driftinggirl_meter_calibration' if independent else 'driftinggirl_parallel_world'
            config['rules'] = [deepcopy(r) for r in config['rules'] if r['id']==rule_id]
            config['rules'][0]['enabled'] = True
            config['poll_interval'] = .1
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            native,closed = image(native_name),image('cancel-window-closed-rank35')
            if independent:
                native,closed=native.copy(),closed.copy()
                active=image('sailing-sleep-bottom')
                for fixture in (native,closed): fixture[4:65,186:254]=active[4:65,186:254]
            contacts,commands,fast = [],[],[]
            class Device:
                def __init__(self,_,stop):
                    self.stop,self.io_lock = stop,threading.RLock()
                    self.image,self.screenshot_transport = None,'test'
                def screenshot(self):
                    self.image = native.copy()
                    return self.image
                def screenshot_fast(self):
                    fast.append(1)
                    self.screenshot_transport = 'mumu-native'
                    # The rebuilt outer loop also uses native capture. Delay
                    # the first input-guard frame, not that outer observation.
                    if slow_first and len(fast)==2:
                        time.sleep(.13)
                    self.image = (closed if contacts else native).copy()
                    if changed and len(fast)>=2:
                        self.image = changed(self.image)
                    if stop_capture and len(fast)>=2 or len(fast)>4 and not contacts:
                        self.stop.set()
                    return self.image
                def touch_fast(self,x,y,captured_at,stopped):
                    contacts.append((x,y,time.monotonic()-captured_at,stopped()))
                    if down_error:
                        raise RuntimeError('MuMu 原生触摸按下失败；未重复或回退输入')
                    return dict(input_transport='mumu-native',native_xy=[x,y],hold_ms=40)
                def command(self,*args):
                    commands.append(args)
                def stuck_record_add(self,_):
                    pass
            def make_world(*args,**kwargs):
                world = ParallelWorld(*args,**kwargs)
                world.phase,world.rank = 'play',35
                world.training.begin(35,814,1112)
                world.training.state.update(sailing_enabled=False,sailing_verified=False,
                    sailing_sleep_recovery=True,sailing_meter_adjustment=True)
                world.fishing.current,world.fishing.player,world.fishing.goal = (
                    (1038,1079,1074) if native_name=='filled-dots-rank35' else (934,982,977))
                world.fishing.world_matches(native)
                world.step = world.fishing.prepare
                return world
            def make_calibrator(*args,**kwargs):
                c=MeterCalibration(*args,**kwargs)
                c.journal.update(pending=True,rank=35,stage=c._stage(native),restore_running=True,
                    deadline=time.time()+90,route_policy='preserve',stage_strokes=c.world._stage_strokes(native).hex(),
                    attempts={},sailing_attempts={},value=934,trigger_player=982,high=1112)
                c.phase,c.route_verified,c.off_verified='prepare',True,False
                c.context=c.fishing._context(native)
                c.world.rank=35
                c.world.training.state.update(mode='farming',rank=35,sailing_enabled=True,
                    sailing_sleep_recovery=True,sailing_meter_adjustment=True)
                c.fishing.current,c.fishing.goal=934,977
                c.fishing.saved_player_gap=True
                c.fishing.world_matches(native)
                c.journal['selected_signature']=c._selection_signature(c.fishing.cast_data(native))
                c._save()
                return c
            app = Clicker(root,Device); app.digit_reader = self.reader
            try:
                with patch('app.ParallelWorld',side_effect=make_world), \
                     patch('app.MeterCalibration',side_effect=make_calibrator):
                    app.start(preview=False,max_clicks=1)
                    app.thread.join(12)
                self.assertFalse(app.running(),list(app.events))
                records = [json.loads(p.read_text(encoding='utf-8'))
                           for p in (root/'logs/fishing-input').glob('input-*.json')]
                if records:
                    record = records[0]
                    before = np.asarray(Image.open(root/'logs/fishing-input'/record['before_file']).convert('RGB'))
                    after = np.asarray(Image.open(root/'logs/fishing-input'/record['after_file']).convert('RGB'))
                    np.testing.assert_array_equal(before,native)
                    np.testing.assert_array_equal(after,closed)
                return contacts,commands,fast,deepcopy(app.state),records
            finally:
                app.stop()
                if app.thread:app.thread.join(2)
                for handler in list(app.log.handlers):
                    handler.close();app.log.removeHandler(handler)

    def test_worker_uses_fresh_native_window_and_keeps_unique_before_after_evidence(self):
        contacts,commands,fast,state,records = self.run_worker()
        self.assertEqual(len(contacts),1)
        self.assertEqual(contacts[0][:2],(690,1090))
        self.assertLessEqual(contacts[0][2],.12)
        self.assertFalse(contacts[0][3])
        self.assertEqual(commands,[])
        self.assertEqual(len(fast),3)  # Outer observation, fresh guard, after-input evidence.
        self.assertEqual(state['clicks'],1)
        self.assertEqual(state['stop_reason'],'click_limit')
        self.assertIsNone(state['last_error'])
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]['result'],'issued_only')
        self.assertNotEqual(records[0]['frame_id'],records[0]['after_frame_id'])

    def test_worker_refreshes_slow_capture_before_input(self):
        contacts,commands,fast,state,records = self.run_worker(slow_first=True)
        self.assertEqual(len(contacts),1)
        self.assertGreaterEqual(len(fast),4)
        self.assertLessEqual(contacts[0][2],.12)
        self.assertEqual(commands,[])
        self.assertEqual(len(records),1)

    def test_worker_filled_dots_cancel_only_once_and_never_drag_or_restart_fishing(self):
        contacts,commands,fast,state,records = self.run_worker(native_name='filled-dots-rank35')
        self.assertEqual(len(contacts),1)
        self.assertEqual(contacts[0][:2],(690,1090))
        self.assertLessEqual(contacts[0][2],.12)
        self.assertEqual(commands,[])
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['clicks'],1)
        self.assertEqual(records[0]['result'],'issued_only')

    def test_worker_rejects_green_arrow_world_change_and_stop_before_input(self):
        def green(frame):
            active = image('sailing-sleep-bottom')
            frame[4:65,186:254] = active[4:65,186:254]
            return frame
        def other_world(frame):
            frame[40:78,6:70] = 0
            return frame
        for options in (dict(changed=green),dict(changed=other_world),dict(stop_capture=True)):
            with self.subTest(options=options):
                contacts,commands,_,state,records = self.run_worker(**options)
                self.assertEqual(contacts,[])
                self.assertEqual(commands,[])
                self.assertEqual(records,[])
                self.assertIsNone(state['last_error'])

    def test_uncertain_native_down_stops_without_adb_fallback_or_second_contact(self):
        contacts,commands,_,state,records = self.run_worker(down_error=True)
        self.assertEqual(len(contacts),1)
        self.assertEqual(commands,[])
        self.assertIn('未重复或回退',state['last_error'])
        self.assertEqual(records,[])

    def test_independent_calibration_cancels_on_running_route_without_any_auto_input(self):
        contacts,commands,_,state,records=self.run_worker(independent=True)
        self.assertEqual(len(contacts),1)
        self.assertEqual(contacts[0][:2],(690,1090))
        self.assertLessEqual(contacts[0][2],.12)
        self.assertEqual(commands,[])
        self.assertIsNone(state['last_error'])
        self.assertEqual(len(records),1)

    def test_independent_calibration_rejects_route_change_before_native_cancel(self):
        def off(frame):
            original=image('cancel-rank35')
            frame[4:65,186:254]=original[4:65,186:254]
            return frame
        contacts,commands,_,state,records=self.run_worker(independent=True,changed=off)
        self.assertEqual((contacts,commands,records),([],[],[]))
        self.assertIsNone(state['last_error'])


if __name__=='__main__':
    unittest.main()
