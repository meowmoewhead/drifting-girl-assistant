"""Actual Sea God frames must reach combat even with legacy navigation off."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest

import numpy as np
from PIL import Image

from app import Clicker
from digit_ocr import NumberReading

ROOT = Path(__file__).resolve().parents[1]


class Reader:
    def read(self, _):
        return NumberReading('39', 39, .999)


def frame(name):
    return np.asarray(Image.open(ROOT/'tests/fixtures/boss-defense'/f'{name}.png')
                      .convert('RGB')).copy()


class StoneBossDefenseTests(unittest.TestCase):
    def exercise(self, images, *, stop_before_hit=False, slow=False, expected_error=None, on_touch=None,
                 capture_limit=8):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT/'assets', root/'assets')
            config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            keys = {'driftinggirl_stone_farm', 'driftinggirl_parallel_world',
                    'driftinggirl_equipment_upgrade'}
            config['rules'] = [deepcopy(r) for r in config['rules'] if r['id'] in keys]
            for rule in config['rules']:
                rule['enabled'] = rule['id'] != 'driftinggirl_parallel_world'
            config['poll_interval'] = .1
            (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
            (root/'training_state.json').write_text('{"untouched": true}', encoding='utf-8')
            before = hashlib.sha256((root/'training_state.json').read_bytes()).hexdigest()
            # inputs: native contacts issued by the combat path, which is the
            # only one this suite authorises.
            inputs, menus = [], []

            class Device:
                def __init__(self, _, stop):
                    self.stop, self.io_lock = stop, threading.RLock()
                    self.image, self.index = None, 0

                def screenshot(self):
                    self.index += 1
                    if slow and self.index > 1:
                        time.sleep(.13)
                    self.image = images[min(self.index-1, len(images)-1)].copy()
                    if self.index >= capture_limit or stop_before_hit and self.index >= 2:
                        self.stop.set()
                    return self.image

                screenshot_fast = screenshot

                def touch_fast(self, x, y, started, stopped):
                    if stopped():
                        return None
                    # Combat is the only input this suite authorises; the cruise
                    # setup reuses the same transport for its own taps and keeps
                    # its own evidence, so it must not pollute this counter.
                    if sys._getframe(1).f_code.co_name == 'hit_defense_target':
                        inputs.append((x, y))
                    if on_touch is not None:on_touch()
                    return dict(input_transport='test-native', hold_ms=40)

                def click(self, button):
                    menus.append(button.button)

                def stuck_record_add(self, _):
                    pass

            app = Clicker(root, Device)
            app.digit_reader = Reader()
            try:
                app.start(preview=False, max_clicks=3, timeout=4)
                app.thread.join(6)
                self.assertFalse(app.running())
                if expected_error is None:
                    self.assertIsNone(app.state['last_error'])
                else:
                    self.assertIn(expected_error,app.state['last_error'])
                self.assertIsNone(app.state['parallel'])
                self.assertEqual(menus, [], 'combat must preempt equipment/menu inputs')
                self.assertEqual(hashlib.sha256((root/'training_state.json').read_bytes()).hexdigest(), before)
                self.assertEqual(json.loads((root/'config.json').read_text()), config)
                # The evidence writer is an async daemon: wait for the queued
                # records instead of racing the PNG encoder.
                for writer in (app.boss_evidence,app.portal_evidence,app.sailing_evidence):
                    if writer is not None:
                        writer.flush()
                evidence = [json.loads(p.read_text())
                            for p in (root/'logs/boss-input').glob('input-*.json')]
                # Every combat contact must be evidenced. Taps that other
                # subsystems issue through the same transport (cruise setup
                # rejects its own contact and keeps no record) are out of scope.
                self.assertEqual(len(evidence), len(inputs))
                self.assertEqual(len({r['frame_id'] for r in evidence}), len(evidence))
                return inputs, evidence
            finally:
                app.stop()
                app.thread.join(2)
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_native_stone_route_counters_with_legacy_world_disabled(self):
        inputs, evidence = self.exercise([frame('native-right-charge-low')])
        self.assertEqual(len(inputs), 3)
        self.assertTrue(all(455 <= x <= 635 and 435 <= y <= 660 for x, y in inputs))
        self.assertTrue(all(r['controller']=='stone_farm' and r['rank']==39
                            and r['kind']=='defense' and r['hold_ms']==40
                            and r['result']=='issued_only' for r in evidence))

    def test_fresh_target_change_does_not_reuse_initial_charge(self):
        inputs, _ = self.exercise([frame('native-right-charge-low'), frame('task-ordinary-fish')])
        self.assertEqual(inputs, [])

    def test_native_right_touch_effect_does_not_degrade_into_neutral_damage(self):
        inputs,evidence=self.exercise([frame('native-right-touch-arc-joined')])
        self.assertEqual(len(inputs),3)
        self.assertTrue(all(490<=x<=550 and 550<=y<=620 for x,y in inputs))
        self.assertTrue(all(r['kind']=='defense' and r['hold_ms']==40 for r in evidence))

    def test_stop_and_aged_frames_cannot_authorize_counter(self):
        for kwargs in ({'stop_before_hit': True}, {'slow': True}):
            with self.subTest(**kwargs):
                inputs, _ = self.exercise([frame('native-right-charge-low')], **kwargs)
                self.assertEqual(inputs, [])

    def test_dimmed_and_rank40_images_are_not_a_final_boss(self):
        source = frame('native-right-charge-low')
        rank40 = source.copy()
        rank40[9:63, 65:139] = 0
        for picture in (source//2, rank40):
            with self.subTest(dimmed=np.array_equal(picture, source//2)):
                inputs, _ = self.exercise([picture])
                self.assertEqual(inputs, [])

    def test_guardian_taps_work_without_legacy_navigation_or_rank39_override(self):
        inputs,evidence=self.exercise([frame('guardian-battle-0')])
        self.assertGreater(len(inputs),0)
        self.assertTrue(all(point==(360,330) for point in inputs))
        self.assertTrue(all(r['controller']=='stone_farm' and r['rank']==38
                            and r['controller_rank']==38 and r['kind']=='damage'
                            and r['hold_ms']==40 and r['result']=='issued_only' for r in evidence))

    def test_guardian_new_frame_change_stop_or_age_never_issues_input(self):
        guardian=frame('guardian-battle-0')
        ordinary=guardian.copy()
        ordinary[168:199,187:330]=frame('task-ordinary-fish')[168:199,187:330]
        for pictures,kwargs in (([guardian,ordinary],{}),
                                # End before a later outer loop may legitimately
                                # take over the newly identified Sea God fight.
                                ([guardian,frame('native-right-charge-low')],{'capture_limit':3}),
                                ([guardian],{'stop_before_hit':True}),
                                ([guardian],{'slow':True})):
            with self.subTest(options=kwargs):
                inputs,_=self.exercise(pictures,**kwargs)
                self.assertEqual(inputs,[])


if __name__ == '__main__':
    unittest.main()
