"""Offline replay of the observed wrong-world dialog; never connect a device."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from drift.runtime import Clicker
from drift.tasks.frame_runtime import FrameRuntime
from digit_ocr import parse_counter
from features import FEATURE_BY_ID

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'driftinggirl_potion_ad_open'
CANCEL = 'potion_ad_wrong_world_cancel'


def image(path):
    with Image.open(ROOT/path) as source:
        return np.asarray(source.convert('RGB'))


class PotionReturnHomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'config.json').read_text('utf-8-sig'))
        cls.entry = image('tests/fixtures/ad-position/potion-up-paused.png')
        cls.dialog = image('assets/potion-ad/return-world-confirm-source.png')
        cls.normal = image('assets/potion-ad-confirm-source.png')
        cls.reward = image('tests/fixtures/legacy-captures/potion-ad-after-continue.png')

    def replay(self, scenario, *, stop_on_main=True, dialog=None):
        config = deepcopy(self.config)
        config['poll_interval'] = .1
        ids = FEATURE_BY_ID['potion_ad']['rules']
        config['rules'] = [r for r in config['rules'] if r['id'] in ids]
        for r in config['rules']:
            r.update(enabled=True, interval=.2, next_delay=0)
            if r['id'] == ENTRY:
                r['stable_frames'] = 1
        clock = [0.]
        entry, wrong, normal, reward = self.entry, self.dialog if dialog is None else dialog, self.normal, self.reward
        class Device:
            def __init__(self, config, stop):
                self.stop, self.image, self.shots = stop, None, 0
                self.taps = []
                self.io_lock = threading.RLock()
            def screenshot(self):
                self.shots += 1
                clock[0] += 1
                if self.shots > 260:
                    raise InterruptedError('bounded offline replay')
                if scenario.startswith('vanish'):
                    cutoff = 2 if scenario == 'vanish_at_recheck' else 3
                    frame = entry if self.shots < cutoff else np.zeros_like(entry)
                elif not self.taps:
                    frame = entry
                elif self.taps[-1][0] == ENTRY:
                    frame = normal if scenario == 'normal' else wrong
                elif self.taps[-1][0] == CANCEL:
                    frame = wrong if scenario == 'cancel_stuck' else entry
                else:
                    frame = reward
                self.image = frame.copy()
                return self.image
            def click(self, button):
                if self.stop.is_set():
                    raise InterruptedError()
                self.taps.append((button.name, tuple(button.button)))
            def stuck_record_add(self, button):
                pass
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            assets = {r['asset'] for r in config['rules']} | {'assets/potion-ad/return-world-confirm-source.png'}
            for name in assets:
                (root/name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/name, root/name)
            (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
            app = Clicker(root, Device)
            app.digit_reader = SimpleNamespace(read_counter=lambda _: parse_counter('0/40', .99))
            try:
                with patch('app.time', SimpleNamespace(monotonic=lambda: clock[0])), \
                     patch.object(FrameRuntime, 'start_engine'), \
                     patch.object(FrameRuntime, 'wait', return_value=False), \
                     patch.object(app.task_schedule, 'finish', wraps=app.task_schedule.finish) as finish:
                    app.run([app.load_rule(r) for r in config['rules']], False,
                            stop_on_main=stop_on_main, timeout=240)
                return (deepcopy(app.state), list(app.device.taps), list(app.events),
                        finish.call_args_list, app.daily_schedule.status())
            finally:
                for h in list(app.log.handlers):
                    h.close()
                    app.log.removeHandler(h)

    def test_missing_current_icon_prevents_both_recheck_and_final_tap(self):
        for scenario in ('vanish_at_recheck', 'vanish_at_tap'):
            with self.subTest(scenario=scenario):
                state, taps, _, _, _ = self.replay(scenario)
                self.assertEqual(taps, [])
                self.assertIsNone(state['last_error'])

    def test_wrong_world_cancels_once_and_returns_without_success(self):
        state, taps, events, finishes, daily = self.replay('wrong_world')
        self.assertIsNone(state['last_error'])
        self.assertEqual([t[0] for t in taps], [ENTRY, CANCEL])
        x1, y1, x2, y2 = taps[1][1]
        self.assertTrue(365 < (x1+x2)//2 < 655 and 742 < (y1+y2)//2 < 838)
        self.assertEqual(state['stage'], 'main')
        self.assertEqual(state['stop_reason'], 'observation_complete')
        self.assertEqual(state['potion_ad']['attempts'], 1)
        self.assertFalse(state['potion_ad']['pending'])
        self.assertFalse(state['potion_ad']['success'])
        self.assertEqual(len(finishes), 1)
        self.assertFalse(finishes[0].kwargs['success'])
        self.assertFalse(any(d.get('completed') for d in daily.values()))
        self.assertTrue(any('取消' in e['message'] and '不记成功' in e['message'] for e in events))

    def test_repeated_wrong_world_is_capped_at_three_attempts(self):
        state, taps, _, _, _ = self.replay('wrong_world', stop_on_main=False)
        self.assertIsNone(state['last_error'])
        self.assertEqual([t[0] for t in taps], [ENTRY, CANCEL]*3)
        self.assertEqual(state['potion_ad']['attempts'], 3)
        self.assertFalse(state['potion_ad']['pending'])

    def test_cancel_still_visible_is_not_clicked_twice_or_called_success(self):
        state, taps, _, _, _ = self.replay('cancel_stuck')
        self.assertEqual([t[0] for t in taps], [ENTRY, CANCEL])
        self.assertIsNotNone(state['last_error'])
        self.assertFalse(state['potion_ad']['success'])

    def test_partial_title_or_cancel_never_authorizes_a_click(self):
        for box in ((222,556,493,595), (195,597,524,626), (457,764,559,816)):
            changed = self.dialog.copy()
            x1,y1,x2,y2 = box
            changed[y1:y2,x1:x2] = 0
            with self.subTest(box=box):
                state, taps, _, _, _ = self.replay('wrong_world', dialog=changed)
                self.assertEqual([t[0] for t in taps], [ENTRY])
                self.assertIsNotNone(state['last_error'])

    def test_normal_ad_keeps_existing_continue_and_return(self):
        state, taps, _, _, _ = self.replay('normal')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')
        self.assertEqual([t[0] for t in taps], [ENTRY, 'driftinggirl_potion_ad_confirm'])


if __name__ == '__main__':
    unittest.main()
