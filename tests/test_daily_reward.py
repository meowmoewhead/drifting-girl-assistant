"""Daily free rewards must not match paid products or unrelated shop pages."""
import json
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import Clicker


class DailyRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8-sig'))
        cls.context = SimpleNamespace(root=ROOT, config=config)
        cls.rules = {r['id']: r for r in config['rules'] if r['id'].startswith('driftinggirl_daily_')}
        cls.frames = {name: np.asarray(Image.open(ROOT / 'assets' / 'daily-reward' / name).convert('RGB'))
                      for name in ('main.png', 'shop-open.png', 'premium-ready.png', 'premium-claimed.png', 'bonus-ready.png', 'bonus-claimed.png', 'return.png')}

    def match(self, suffix, frame, guards=True):
        rule, button, offset = Clicker.load_rule(self.context, self.rules['driftinggirl_daily_' + suffix])
        matched = button.match(frame, offset=offset, similarity=rule['threshold'])
        if matched and guards and rule.get('absent_rule'):
            other = self.rules[rule['absent_rule']]
            _, guard, guard_offset = Clicker.load_rule(self.context, other)
            matched = not guard.match(frame, offset=guard_offset, similarity=other['threshold'])
        return matched, tuple(button.button)

    def test_shop_entry_needs_bright_cart_but_does_not_wait_for_green_badge(self):
        self.assertTrue(self.match('open', self.frames['main.png'])[0])
        no_badge = self.frames['main.png'].copy()
        no_badge[416:440, 682:718] = 0
        self.assertTrue(self.match('open', no_badge)[0])
        self.assertFalse(self.match('open', self.frames['shop-open.png'])[0])
        self.assertFalse(self.match('open', (self.frames['main.png'] * .45).astype(np.uint8))[0])

    def test_premium_tab_matches_both_selection_states(self):
        for name in ('shop-open.png', 'premium-ready.png'):
            with self.subTest(frame=name):
                matched, box = self.match('premium', self.frames[name])
                self.assertTrue(matched)
                self.assertTrue(154 <= box[0] < box[2] <= 283)
                self.assertTrue(215 <= box[1] < box[3] <= 306)

    def test_free_reward_requires_title_and_free_card_and_taps_only_free_card(self):
        frame = self.frames['premium-ready.png']
        matched, box = self.match('claim', frame)
        self.assertTrue(matched)
        self.assertTrue(183 < box[0] < box[2] < 397)
        self.assertTrue(380 < box[1] < box[3] < 526)
        self.assertFalse(self.match('claim', self.frames['shop-open.png'])[0])
        paid_only = frame.copy()
        paid_only[378:528, 180:400] = 0
        self.assertFalse(self.match('claim', paid_only)[0])
        wrong_page = frame.copy()
        wrong_page[331:371, 170:316] = 0
        self.assertFalse(self.match('claim', wrong_page)[0])

    def test_close_and_return_wait_for_their_screen_conditions(self):
        ready = self.frames['premium-ready.png']
        self.assertFalse(self.match('unavailable', ready)[0])
        self.assertFalse(self.match('close', ready)[0])
        self.assertFalse(self.match('return', ready)[0])
        self.assertTrue(self.match('return', self.frames['main.png'])[0])
        claimed = self.frames['premium-claimed.png']
        self.assertFalse(self.match('claim', claimed)[0])
        self.assertTrue(self.match('unavailable', claimed)[0])
        self.assertTrue(self.match('close', claimed)[0])
        uncertain = claimed.copy()
        uncertain[433:481, 254:331] = 0
        self.assertFalse(self.match('unavailable', uncertain)[0])

    def replay(self, names, expected_taps, already_complete=False):
        frames = [self.frames[name] for name in names]
        class ReplayDevice:
            def __init__(self, config, stop):
                self.stop, self.image, self.taps = stop, None, []
            def screenshot(self):
                self.image = frames[min(len(self.taps), len(frames)-1)].copy()
                return self.image
            def click(self, button):
                if self.stop.is_set():
                    raise InterruptedError()
                box = tuple(button.button)
                center = ((box[0]+box[2])//2, (box[1]+box[3])//2)
                index = len(self.taps)
                if index >= len(expected_taps):
                    raise AssertionError('Unexpected extra click')
                left, top, right, bottom = expected_taps[index]
                if not (left <= center[0] <= right and top <= center[1] <= bottom):
                    raise AssertionError('Click outside expected free workflow target: ' + str(center))
                self.taps.append(center)
            def stuck_record_add(self, button):
                pass
        with tempfile.TemporaryDirectory(prefix='daily-replay-') as directory:
            root = Path(directory)
            shutil.copytree(ROOT/'assets'/'daily-reward', root/'assets'/'daily-reward')
            rules = deepcopy(list(self.rules.values()))
            for rule in rules:
                rule.update(enabled=True, interval=0.2, next_delay=0, stage_timeout=5)
            config = dict(adb_path='fake', serial='test', resolution=[720,1280],
                          poll_interval=0.1, max_consecutive_clicks=2, rules=rules)
            (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
            app = Clicker(root, ReplayDevice)
            old_free = None
            if already_complete:
                app.daily_schedule.require_bonus=False
                app.daily_schedule.mark_complete('daily_reward',app.daily_schedule.period())
                old_free=deepcopy(app.daily_schedule.completed)
                app.task_schedule.finish('daily_reward',3600,app.daily_schedule.status()['daily_reward'])
                app.daily_schedule.require_bonus=True
            try:
                app.start(preview=False, stop_on_main=True, max_clicks=6, timeout=10)
                app.thread.join(12)
                self.assertFalse(app.running())
                self.assertIsNone(app.state['last_error'])
                self.assertEqual(app.state['stop_reason'], 'cycle_complete')
                self.assertEqual(app.state['clicks'], len(expected_taps))
                self.assertTrue(app.daily_schedule.is_complete('daily_reward'))
                self.assertEqual(app.daily_schedule.bonus_checked['period'],app.daily_schedule.period())
                if old_free is not None:self.assertEqual(app.daily_schedule.completed,old_free)
            finally:
                app.stop()
                if app.thread:
                    app.thread.join(3)
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_full_claim_cycle_with_real_screens(self):
        # The right card is gray in the old screenshots: no bonus input.
        self.replay(['main.png','shop-open.png','premium-ready.png','premium-claimed.png',
                     'return.png'],
                    [(648,367,709,423),(168,228,270,297),(210,433,377,510),
                     (634,247,695,296)])

    def test_already_claimed_cycle_never_taps_free_card(self):
        self.replay(['main.png','shop-open.png','premium-claimed.png','return.png'],
                    [(648,367,709,423),(168,228,270,297),(634,247,695,296)])

    def test_bonus_ready_after_free_claimed_is_claimed_once_then_verified_gray(self):
        self.replay(['main.png','shop-open.png','bonus-ready.png','premium-claimed.png','return.png'],
                    [(648,367,709,423),(168,228,270,297),(410,430,605,500),(634,247,695,296)],
                    already_complete=True)

    def test_restart_on_bonus_ready_page_claims_instead_of_orphan_close(self):
        self.replay(['bonus-ready.png','premium-claimed.png','return.png'],
                    [(410,430,605,500),(634,247,695,296)],already_complete=True)

    def test_restart_on_gray_page_skips_bonus_without_reopening_or_reclaiming_free(self):
        self.replay(['premium-claimed.png','return.png'],
                    [(634,247,695,296)],already_complete=True)

    def test_actual_bonus_claim_followup_closes_without_a_second_claim(self):
        self.replay(['bonus-ready.png','bonus-claimed.png','return.png'],
                    [(410,430,605,500),(634,247,695,296)],already_complete=True)

    def test_restart_on_actual_claimed_card_only_closes_and_preserves_free_record(self):
        self.replay(['bonus-claimed.png','return.png'],
                    [(634,247,695,296)],already_complete=True)


if __name__ == '__main__':
    unittest.main()
