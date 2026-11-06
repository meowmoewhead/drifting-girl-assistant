from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2

from drift.perception.digit_ocr import DigitReader
from drift.perception.restaurant import RestaurantVisual

ROOT = Path(__file__).resolve().parents[1]


def reading(value, confidence=.999):
    return SimpleNamespace(value=value, confidence=confidence)


class RestaurantCardConsensusTests(unittest.TestCase):
    def frame(self):
        return cv2.cvtColor(cv2.imread(str(ROOT /
            'tests/fixtures/restaurant-list/level11-low-confidence.png')), cv2.COLOR_BGR2RGB)

    def test_real_level11_is_retained_in_left_to_right_order(self):
        visual = RestaurantVisual(ROOT, DigitReader(ROOT))
        cards = visual.cards(self.frame(), strict_order=True)
        self.assertEqual([c['level'] for c in cards], [20, 11, 3, 3])

    def visual(self, first, extra):
        reader = Mock()
        reader.read_many.side_effect = [first, extra]
        return RestaurantVisual(ROOT, reader)

    def first(self, pair):
        return [reading(20), reading(20), *pair,
                reading(3), reading(3), reading(3), reading(3)]

    def test_low_confidence_read_needs_two_independent_confident_values(self):
        visual = self.visual(self.first([reading(11, .978), reading(11, .938)]),
                             [reading(11, .99), reading(11, .92), reading(11, .90)])
        self.assertEqual([c['level'] for c in visual.cards(self.frame(), strict_order=True)],
                         [20, 11, 3, 3])

    def test_conflicting_confident_value_is_never_voted_away(self):
        visual = self.visual(self.first([reading(11), reading(1)]), [reading(11)] * 3)
        self.assertEqual([c['level'] for c in visual.cards(self.frame(), strict_order=True)],
                         [20, None, 3, 3])

    def test_insufficient_or_invalid_values_keep_the_order_barrier(self):
        for extra in ([reading(11, .94)] * 3,
                      [reading(21)] * 3,
                      [reading(None)] * 3):
            visual = self.visual(self.first([reading(11, .978), reading(11, .938)]), extra)
            self.assertEqual([c['level'] for c in visual.cards(self.frame(), strict_order=True)],
                             [20, None, 3, 3])


if __name__ == '__main__':
    unittest.main()
