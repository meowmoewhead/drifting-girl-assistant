"""Real potion counters plus conservative agreement and daily-quota boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import Clicker
from digit_ocr import DigitReader, parse_counter, prepare_counter_crop, read_daily_counter
from tools.configure_potion_counter_consensus import RULE_ID, update

FIXTURES = ROOT/'tests/fixtures/potion-counter-consensus'
MODE = 'white_text_consensus'


class PotionCounterConsensusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)
        cls.samples = json.loads((FIXTURES/'manifest.json').read_text(encoding='utf-8'))
        cls.config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.rule = next(r for r in cls.config['rules'] if r['id'] == RULE_ID)

    def crop(self, name):
        with Image.open(FIXTURES/name) as source:
            return np.asarray(source.convert('RGB'))

    def evaluate(self, crop, reader=None, mode=MODE):
        rule = deepcopy(self.rule)
        rule['counter_condition'].update(preprocess=mode,
            source_area=[0, 0, crop.shape[1], crop.shape[0]])
        context = SimpleNamespace(root=ROOT, digit_reader=reader or self.reader,
            lock=threading.RLock(), state={'frames': 1, 'numbers': {}})
        button = SimpleNamespace(source_to_screen=lambda area: area)
        passed, value = Clicker.counter_condition(context, rule, button, crop)
        return passed, value, context.state['numbers'][RULE_ID]

    def test_all_visually_verified_historical_counters(self):
        self.assertEqual(len(self.samples), 40)
        self.assertEqual({s['current'] for s in self.samples}, {0, 1, 2, 3, 4, 6, 21, 28})
        for sample in self.samples:
            with self.subTest(file=sample['file']):
                passed, value, info = self.evaluate(self.crop(sample['file']))
                self.assertTrue(passed, info)
                self.assertEqual(value, (sample['current'], sample['total']))
                self.assertGreaterEqual(info['confidence'], .93)
                self.assertEqual(info['remaining'], 40-sample['current'])
                self.assertIsNone(info['skip_reason'])

    def test_three_actual_28_frames_reproduce_and_fix_the_skip(self):
        samples = [s for s in self.samples if 'visible-' in s['source']]
        self.assertEqual(len(samples), 3)
        for sample in samples:
            with self.subTest(file=sample['file']):
                crop = self.crop(sample['file'])
                passed, value, info = self.evaluate(crop, mode='white_text')
                self.assertFalse(passed)
                self.assertEqual(value, (28, 40))
                self.assertLess(info['confidence'], .80)
                self.assertIn('置信度', info['skip_reason'])
                passed, value, info = self.evaluate(crop)
                self.assertTrue(passed, info)
                self.assertEqual(value, (28, 40))
                self.assertGreaterEqual(info['confidence'], .93)

    def test_every_quota_value_and_full_daily_limit(self):
        # The reading is mocked here: this checks quota policy, not unseen glyphs.
        crop = self.crop(self.samples[-1]['file'])
        for current in range(41):
            with self.subTest(current=current):
                reader = Mock()
                reader.read_counter.return_value = parse_counter(f'{current}/40', .99)
                passed, value, info = self.evaluate(crop, reader)
                self.assertEqual(passed, current < 40)
                self.assertEqual(value, (current, 40))
                self.assertEqual(info['remaining'], 40-current)
                if current == 40:
                    self.assertIn('已用完', info['skip_reason'])

    def test_incomplete_invalid_and_wrong_denominator_stay_rejected(self):
        crop = self.crop(self.samples[-1]['file'])
        for text in ('', '28', '/40', '28/', '2B/40', '28.0/40', '41/40',
                     '-1/40', '0/0', '28/48', '8/10'):
            with self.subTest(text=text):
                reader = Mock()
                reader.read_counter.return_value = parse_counter(text, .999)
                passed, _, info = self.evaluate(crop, reader)
                self.assertFalse(passed, info)
                self.assertIsNotNone(info['skip_reason'])

    def synthetic_crop(self):
        # Three distinct views: original, full binary, and tightly bounded ink.
        crop = np.zeros((20, 68, 3), dtype=np.uint8)
        crop[4:15, 20:40] = 250
        return crop

    def test_conflicting_high_confidence_reading_rejects_even_a_majority(self):
        for conflicting in ('23/40', '28/48', '40/40'):
            with self.subTest(conflicting=conflicting):
                reader = Mock()
                reader.read_counter.side_effect = [parse_counter('28/40', .99),
                    parse_counter('28/40', .98), parse_counter(conflicting, .95)]
                passed, value, info = self.evaluate(self.synthetic_crop(), reader)
                self.assertFalse(passed)
                self.assertEqual(value, (None, None))
                self.assertIn('不一致', info['skip_reason'])

    def test_two_reliable_views_agree_and_report_the_lower_confidence(self):
        reader = Mock()
        reader.read_counter.side_effect = [parse_counter('28/40', .99),
            parse_counter('28/40', .95), parse_counter('23/40', .80)]
        result = read_daily_counter(reader, self.synthetic_crop(), MODE)
        self.assertEqual((result.current, result.total), (28, 40))
        self.assertEqual(result.confidence, .95)

    def test_one_reliable_view_is_not_enough(self):
        reader = Mock()
        reader.read_counter.side_effect = [parse_counter('28/40', .99),
            parse_counter('28/40', .92), parse_counter('28/40', .80)]
        result = read_daily_counter(reader, self.synthetic_crop(), MODE)
        self.assertIsNone(result.current)
        self.assertIn('两路', result.reason)

    def test_custom_confidence_is_not_lowered(self):
        reader = Mock()
        reader.read_counter.side_effect = [parse_counter('28/40', .99),
            parse_counter('28/40', .96), parse_counter('28/40', .95)]
        result = read_daily_counter(reader, self.synthetic_crop(), MODE, .98)
        self.assertIsNone(result.current)

    def test_identical_threshold_images_cannot_supply_two_votes(self):
        reader = Mock()
        reader.read_counter.side_effect = [parse_counter('', 0),
            parse_counter('28/40', .99), parse_counter('', 0)]
        result = read_daily_counter(reader, self.synthetic_crop(), MODE)
        self.assertEqual(reader.read_counter.call_count, 3)
        self.assertIsNone(result.current)

    def test_blank_frames_do_not_become_zero(self):
        for background in (0, 128, 255):
            with self.subTest(background=background):
                crop = np.full((20, 68, 3), background, dtype=np.uint8)
                result = read_daily_counter(self.reader, crop, MODE)
                self.assertIsNone(result.current)
                self.assertIsNone(result.total)

    def test_invalid_crop_does_not_run_ocr(self):
        for shape in ((0, 68, 3), (2, 68, 3), (20, 68)):
            reader = Mock()
            result = read_daily_counter(reader, np.zeros(shape, dtype=np.uint8), MODE)
            reader.read_counter.assert_not_called()
            self.assertIsNone(result.current)

    def test_legacy_modes_keep_their_single_read_behavior(self):
        crop = self.synthetic_crop()
        for mode in ('raw', 'white_text'):
            reader = Mock()
            expected = parse_counter('28/40', .99)
            reader.read_counter.return_value = expected
            self.assertIs(read_daily_counter(reader, crop, mode), expected)
            self.assertEqual(reader.read_counter.call_count, 1)
            np.testing.assert_array_equal(reader.read_counter.call_args.args[0],
                                           prepare_counter_crop(crop, mode))

    def test_migration_only_changes_potion_preprocessing_and_note(self):
        before = deepcopy(self.config)
        entry = next(r for r in before['rules'] if r['id'] == RULE_ID)
        entry.update(enabled=False, interval=357)
        entry['counter_condition'].update(preprocess='white_text', confidence=.97)
        after = update(before)
        self.assertEqual(update(after), after)
        changed = next(r for r in after['rules'] if r['id'] == RULE_ID)
        self.assertEqual(changed['counter_condition']['preprocess'], MODE)
        changed['counter_condition']['preprocess'] = entry['counter_condition']['preprocess']
        changed['note'] = entry['note']
        self.assertEqual(after, before)

    def test_migration_requires_exactly_one_entry(self):
        config = deepcopy(self.config)
        config['rules'] = [r for r in config['rules'] if r['id'] != RULE_ID]
        with self.assertRaises(ValueError):
            update(config)
        config['rules'].extend([deepcopy(self.rule), deepcopy(self.rule)])
        with self.assertRaises(ValueError):
            update(config)


if __name__ == '__main__':
    unittest.main()
