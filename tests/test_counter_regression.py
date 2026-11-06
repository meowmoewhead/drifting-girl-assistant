"""Real counter crops that reproduced the skipped potion-ad reward."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import Clicker
from digit_ocr import DigitReader, prepare_counter_crop


class PotionCounterRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)

    def evaluate(self, name, mode):
        frame = np.asarray(Image.open(ROOT / 'tests' / 'fixtures' / 'potion-counter' / name).convert('RGB'))
        rule = {'id': 'potion', 'name': 'potion counter', 'counter_condition': {'source_area': [0, 0, frame.shape[1], frame.shape[0]],
                 'label': 'potion counter', 'expected_total': 40, 'mode': 'used', 'confidence': 0.93,
                 'preprocess': mode}}
        context = SimpleNamespace(root=ROOT, digit_reader=self.reader, lock=threading.RLock(),
                                  state={'frames': 1, 'numbers': {}})
        button = SimpleNamespace(source_to_screen=lambda area: area)
        passed, reading = Clicker.counter_condition(context, rule, button, frame)
        return passed, reading, context.state['numbers']['potion']

    def test_real_skipped_counter_is_read_without_lowering_confidence_threshold(self):
        for index in range(3):
            with self.subTest(frame=index):
                name = f'counter-2-{index}.png'
                old_passed, old_reading, old_info = self.evaluate(name, 'raw')
                self.assertFalse(old_passed)
                self.assertEqual(old_reading, (2, 40))
                self.assertIn('置信度', old_info['skip_reason'])
                passed, reading, info = self.evaluate(name, 'white_text')
                self.assertTrue(passed)
                self.assertEqual(reading, (2, 40))
                self.assertGreaterEqual(info['confidence'], 0.93)
                self.assertEqual(info['remaining'], 38)
                self.assertIsNone(info['skip_reason'])

    def test_previously_working_one_of_forty_still_passes(self):
        passed, reading, info = self.evaluate('counter-1.png', 'white_text')
        self.assertTrue(passed)
        self.assertEqual(reading, (1, 40))
        self.assertEqual(info['remaining'], 39)
        self.assertGreaterEqual(info['confidence'], 0.93)

    def test_blank_white_text_counter_is_not_invented_as_zero(self):
        for background in (0, 128, 255):
            crop = np.full((30, 68, 3), background, dtype=np.uint8)
            reading = self.reader.read_counter(prepare_counter_crop(crop, 'white_text'))
            self.assertIsNone(reading.current)
            self.assertIsNone(reading.total)


if __name__ == '__main__':
    unittest.main()
