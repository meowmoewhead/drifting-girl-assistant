"""Replay the full fourth counter that actually timed out on 2026-11-05."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from drift.perception.digit_ocr import DigitReader
from drift.perception.restaurant import RestaurantVisual
from tests import test_restaurant_upgrade as existing

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'diagnostics/2026-11-05/restaurant-early-finish/inputs'


class RestaurantCounterShadowFrames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.visual = RestaurantVisual(ROOT, DigitReader(ROOT))
        cls.frames = []
        for name in ('05c187dcd1e242ae84ed28e4faa1baf2-before.png',
                     '05c187dcd1e242ae84ed28e4faa1baf2-after.png',
                     'e803a6a4117f42979f30faa842f34ec9-before.png'):
            frame = cv2.imread(str(EVIDENCE/name))
            if frame is None:
                raise unittest.SkipTest('Native evidence fixture unavailable')
            cls.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def test_completed_counter_is_read_without_lowering_confidence(self):
        values = [self.visual.detail_values(f.copy()) for f in self.frames]
        self.assertEqual(len({v['identity'] for v in values}), 1)
        self.assertEqual([v['level'] for v in values], [8, 8, 8])
        self.assertEqual(values[0]['counters'], [(16, 16)]*3+[(12, 16)])
        self.assertEqual(values[1]['counters'], [(16, 16)]*4)
        self.assertEqual(values[2]['counters'], [(16, 16)]*4)
        self.assertTrue(self.visual.upgrade_ready(self.frames[1].copy(), values[1]))

    def test_issued_hold_needs_three_fresh_frames_before_upgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            c = existing.RestaurantJournalTests().controller(directory, 'feed_wait')
            before = self.visual.detail_values(self.frames[0].copy())
            attempts = {'feed:'+before['identity']+':8:3:11': 1}
            c.journal.update(awaiting=before, ingredient=3, attempts=attempts.copy(),
                             card=dict(level=8, identity=before['identity']))
            c.visual = self.visual
            c.same_world = lambda _: True
            c.view._off_icon.return_value = True
            c.action = Mock(return_value=True)
            old_deadline = c.journal['deadline']
            for now in (100, 101, 103):
                c.clock = lambda now=now: now
                c.step(self.frames[1].copy(), due=False)
                self.assertEqual(c.journal['submitted'], 4 if now == 103 else 0)
            self.assertEqual(c.phase, 'detail')
            self.assertEqual(c.journal['attempts'], attempts)
            self.assertEqual(c.journal['deadline'], old_deadline)
            c.action.assert_not_called()
            for now in (104, 105, 107):
                c.clock = lambda now=now: now
                c.step(self.frames[2].copy(), due=False)
            self.assertEqual(c.phase, 'upgrade_wait')
            self.assertEqual(c.action.call_count, 1)
            self.assertTrue(c.action.call_args.args[0]['kind'].startswith('upgrade:'))
            self.assertNotIn('hold_ms', c.action.call_args.args[0])

    def test_cached_full_frame_does_not_confirm_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            c = existing.RestaurantJournalTests().controller(directory, 'feed_wait')
            c.journal.update(awaiting=self.visual.detail_values(self.frames[0].copy()), ingredient=3)
            c.visual = self.visual
            c.same_world = lambda _: True
            frame = self.frames[1].copy()
            for now in (100, 101, 104):
                c.clock = lambda now=now: now
                c.step(frame, due=False)
            self.assertEqual(c.journal['submitted'], 0)
            self.assertEqual(c.phase, 'feed_wait')


class RestaurantCounterShadowGuards(unittest.TestCase):
    @staticmethod
    def reading(text, confidence=.99):
        return SimpleNamespace(text=text, confidence=confidence)

    def visual(self, final, extra):
        reader = Mock()
        first = [SimpleNamespace(value=8, confidence=.99)]*2
        first += [self.reading('016/016')]*9 + final
        reader.read_many.side_effect = [first, extra]
        visual = RestaurantVisual.__new__(RestaurantVisual)
        visual.reader, visual.cache = reader, {}
        visual.detail, visual.max_detail = lambda _: True, lambda _: False
        visual.identity = lambda _: 'dish'
        return visual

    def test_unread_counter_recovers_only_with_two_confident_decodes(self):
        for confidence, expected in ((.99, True), (.94, False)):
            with self.subTest(confidence=confidence):
                visual = self.visual([self.reading('?', .2)]*3,
                    [self.reading('016/016'), self.reading('016/016', confidence)])
                value = visual.detail_values(np.zeros((1280, 720, 3), dtype=np.uint8))
                self.assertEqual(value is not None, expected)

    def test_fallback_cannot_vote_away_conflicting_original_or_new_values(self):
        for final, extra in (([self.reading('015/016')]+[self.reading('?', .2)]*2,
                              [self.reading('016/016')]*2),
                             ([self.reading('?', .2)]*3,
                              [self.reading('015/016'), self.reading('016/016')]),
                             ([self.reading('?', .2)]*3,
                              [self.reading('012/012')]*2)):
            with self.subTest(extra=extra):
                visual = self.visual(final, extra)
                self.assertIsNone(visual.detail_values(np.zeros((1280, 720, 3), dtype=np.uint8)))

    def test_existing_conflict_is_rejected_before_fallback(self):
        visual = self.visual([self.reading('015/016'), self.reading('016/016'),
                              self.reading('?', .2)], [])
        self.assertIsNone(visual.detail_values(np.zeros((1280, 720, 3), dtype=np.uint8)))
        self.assertEqual(visual.reader.read_many.call_count, 1)

    def test_success_and_failure_logs_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            c = existing.RestaurantJournalTests().controller(directory)
            c.journal['failure'] = dict(reason='feed_wait 阶段确认超时')
            old = deepcopy(c.journal['failure'])
            c._finish(True)
            self.assertFalse(c.done)
            self.assertEqual(c.journal['failure'], old)
            self.assertIn('本轮未完成', c.event.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
