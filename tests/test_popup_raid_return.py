"""Replay the 08:00 stop without treating the RAID queue as the main screen."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import Clicker
from drift.core.session import LoopState
from drift.ui.popup_raid import PopupRaidReturn

ROOT = Path(__file__).resolve().parents[1]


def image(path):
    with Image.open(ROOT / path) as source:
        return np.asarray(source.convert('RGB'))


class PopupRaidReturnTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        self.rules = {r['id']: deepcopy(r) for r in self.config['rules']}
        self.waiting = image('tests/fixtures/popup-raid/waiting-after-close.png')
        self.notice = image('assets/raid/task-popup.png').copy()
        self.notice[:80] = self.waiting[:80]
        self.context = SimpleNamespace(root=ROOT, config=self.config)
        self.st = LoopState(stage='raid_wait', stage_started_at=100,
                            stage_deadline=280, stage_ready_at=100.8,
                            stage_trigger=self.rules[PopupRaidReturn.PREPARE]['name'],
                            previous=PopupRaidReturn.PREPARE, consecutive=1)
        for key in (PopupRaidReturn.CLOSE, PopupRaidReturn.PREPARE,
                    PopupRaidReturn.START, PopupRaidReturn.RETURN):
            self.rules[key]['enabled'] = True
            self.st.matchers[key] = Clicker.load_rule(self.context, self.rules[key])
        self.guard = PopupRaidReturn(ROOT)

    def arm(self):
        self.assertTrue(self.guard.arm(self.st, self.notice, now=130))
        self.st.handling_interrupt = True
        self.st.stage = 'task_popup_return'
        self.st.stage_started_at, self.st.stage_ready_at, self.st.stage_deadline = 130, 130.8, 145
        self.st.stage_trigger = self.rules[PopupRaidReturn.CLOSE]['name']

    def observe(self, at, frame=None, **kwargs):
        return self.guard.observe(self.st, self.waiting if frame is None else frame,
                                  now=at, **kwargs)

    def test_real_waiting_and_ready_are_recognized_with_original_high_threshold(self):
        for path in ('tests/fixtures/popup-raid/waiting-after-close.png',
                     'assets/raid/waiting.png', 'assets/raid/ready.png'):
            with self.subTest(path=path):
                self.assertTrue(self.guard._waiting(image(path)))
        self.assertGreater(self.guard.buttons[1].last_similarity, .99)

    def test_three_seconds_restore_the_exact_original_step_not_new_deadline(self):
        before = {key: getattr(self.st, key) for key in self.guard.FIELDS}
        self.st.daily_period = '2026-10-15'
        self.arm()
        self.st.daily_period = '2026-10-16'  # Reset does not replace the saved RAID deadline.
        self.assertFalse(self.observe(131))
        self.assertFalse(self.observe(132))
        self.assertTrue(self.observe(134))
        self.assertTrue(self.guard.restore(self.st, now=134))
        self.assertEqual({key: getattr(self.st, key) for key in self.guard.FIELDS}, before)
        self.assertEqual(self.st.daily_period, '2026-10-16')
        self.assertFalse(self.st.handling_interrupt)
        self.assertFalse(self.guard.restore(self.st, now=135))

    def test_fast_frames_and_unknown_scenes_cannot_complete_proof(self):
        self.arm()
        for at in (131, 131.1, 131.2):
            self.assertFalse(self.observe(at))
        for path in ('assets/raid/task-popup.png', 'assets/raid/prepare.png',
                     'assets/raid/battle.png', 'assets/raid/result.png',
                     'assets/guild-shop/potion-confirm.png', 'assets/daily-reward/premium-ready.png'):
            with self.subTest(path=path):
                self.assertFalse(self.observe(134, image(path)))
                self.assertEqual(self.guard.frames, 0)
        self.assertFalse(self.observe(135))
        self.assertFalse(self.observe(136))
        self.assertTrue(self.observe(138))

    def test_every_fixed_label_and_brightness_are_required(self):
        for x,y,X,Y in ((310,254,410,280), (420,584,466,611),
                        (325,637,396,655), (185,240,226,279)):
            frame = self.waiting.copy()
            frame[y:Y,x:X] = 0
            self.assertFalse(self.guard._waiting(frame))
        self.assertFalse(self.guard._waiting(self.waiting//2))

    def test_changed_rank_world_or_obscured_frame_resets_consecutive_proof(self):
        changes = []
        for area in ((72,10,126,62), (35,46,56,66), (310,254,410,280)):
            frame = self.waiting.copy()
            x,y,X,Y = area
            frame[y:Y,x:X] = 0
            changes.append(frame)
        changes.extend((self.waiting[:900], self.waiting//2, self.notice))
        for frame in changes:
            self.setUp(); self.arm()
            self.assertFalse(self.observe(131))
            self.assertFalse(self.observe(132, frame))
            self.assertFalse(self.observe(134))
            self.assertEqual(self.guard.frames, 1)

    def test_no_owner_wrong_stage_disabled_or_expired_original_cannot_arm(self):
        self.assertFalse(self.observe(131))
        for mode in ('stage', 'trigger', 'disabled', 'return_disabled', 'expired', 'stop'):
            self.setUp()
            if mode == 'stage': self.st.stage = 'daily_tab'
            if mode == 'trigger': self.st.stage_trigger = 'unrelated'
            if mode == 'disabled': self.st.matchers[PopupRaidReturn.START][0]['enabled'] = False
            if mode == 'return_disabled': self.st.matchers[PopupRaidReturn.RETURN][0]['enabled'] = False
            if mode == 'expired': self.st.stage_deadline = 130
            self.assertFalse(self.guard.arm(self.st, self.notice, now=130, stopped=mode=='stop'))

    def test_either_deadline_and_stop_still_win_after_confirmation(self):
        for mode in ('popup', 'original', 'stop', 'commit_expired', 'commit_stop'):
            self.setUp(); self.arm()
            if mode == 'original': self.guard.saved['stage_deadline'] = 134
            self.assertFalse(self.observe(131)); self.assertFalse(self.observe(132))
            if mode in ('commit_expired', 'commit_stop'):
                self.assertTrue(self.observe(134))
                self.assertFalse(self.guard.restore(self.st, now=145 if mode=='commit_expired' else 134,
                                                     stopped=mode=='commit_stop'))
            else:
                self.assertFalse(self.observe(145 if mode=='popup' else 134, stopped=mode=='stop'))
            self.assertEqual(self.st.stage, 'task_popup_return')

    def test_new_interrupt_cannot_reuse_previous_votes(self):
        self.arm()
        self.assertFalse(self.observe(131)); self.assertFalse(self.observe(132))
        self.st.stage_started_at = 133
        self.assertFalse(self.observe(134))
        self.assertEqual(self.guard.frames, 1)

    def test_runtime_popup_resumes_wait_then_original_ready_and_result_guards(self):
        config = deepcopy(self.config)
        rules = [deepcopy(r) for r in config['rules']
                 if r['id'].startswith('driftinggirl_raid_') or r['id'] in
                 (PopupRaidReturn.CLOSE, 'driftinggirl_task_popup_return')]
        for rule in rules:
            rule.update(enabled=True, interval=.2, next_delay=0)
        config['rules'] = rules
        config['poll_interval'] = .1
        frames = {name: image('assets/raid/'+name+'.png').copy() for name in
                  ('main','prepare','waiting','ready','battle','result','return')}
        frames['waiting'] = self.waiting.copy()
        frames['notice'] = self.notice.copy()
        # Synthetic sequence uses real pages with a fixed top context.
        for frame in frames.values(): frame[:80] = self.waiting[:80]
        expected = ['driftinggirl_raid_open', PopupRaidReturn.PREPARE,
                    PopupRaidReturn.CLOSE, PopupRaidReturn.START, 'driftinggirl_raid_confirm']
        clock = [0.]
        class Device:
            def __init__(self, cfg, stop):
                self.stop, self.image, self.taps, self.phase = stop, None, [], 0
            def screenshot(self):
                clock[0] += 1
                self.phase += 1
                count = len(self.taps)
                name = ('main' if count==0 else 'prepare' if count==1 else 'notice' if count==2
                        else ('waiting' if self.phase<=7 else 'ready') if count==3
                        else ('battle' if self.phase<=4 else 'result') if count==4 else 'return')
                self.image = frames[name].copy()
                return self.image
            def click(self, button):
                if button.name != expected[len(self.taps)]:
                    raise AssertionError('Unexpected click '+button.name)
                if button.name == PopupRaidReturn.START:
                    assert self.phase >= 10, 'Start requires fresh ready frames after return'
                self.taps.append(button.name); self.phase = 0
            def stuck_record_add(self, button): pass
        with tempfile.TemporaryDirectory(prefix='popup-raid-') as directory:
            root = Path(directory)
            paths = {p for r in rules for p in [r['asset'], *r.get('asset_variants', [])]}
            paths.add('assets/raid/waiting.png')
            for path in paths:
                (root/path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/path, root/path)
            (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
            app = Clicker(root, Device)
            try:
                with patch('app.time', SimpleNamespace(monotonic=lambda:clock[0])):
                    app.start(preview=False, stop_on_main=True, max_clicks=6, timeout=180)
                    app.thread.join(20)
                    self.assertFalse(app.running())
                    self.assertIsNone(app.state['last_error'])
                    self.assertEqual(app.state['stop_reason'], 'cycle_complete')
                    self.assertEqual(app.device.taps, expected)
                    self.assertEqual(app.state['clicks'], 5)
            finally:
                app.stop()
                if app.thread: app.thread.join(3)
                for handler in list(app.log.handlers):
                    handler.close(); app.log.removeHandler(handler)


if __name__ == '__main__': unittest.main()
