"""Blue-star entry: independent native frames and unchanged purchase guards."""
from copy import deepcopy
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from app import Clicker
from digit_ocr import DigitReader, NumberReading
from dev_tools.configure_event_star import ASSET, ENTRY, update
from dev_tools.verify_event_potion import inspect_rule, replay

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/event-star'


class EventStarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = json.loads((ROOT / 'config.json').read_text(encoding='utf-8-sig'))
        cls.config = update(cls.before)
        cls.rule = next(r for r in cls.config['rules'] if r['id'] == ENTRY)
        cls.reader = DigitReader(ROOT)

    def frame(self, name):
        with Image.open(FIXTURES / (name + '.png')) as im:
            return np.asarray(im.convert('RGB'))

    def context(self, reader=None):
        return SimpleNamespace(root=ROOT, config=self.config,
            digit_reader=reader or self.reader, lock=threading.RLock(),
            state={'numbers': {}, 'frames': 1})

    def test_independent_main_frame_identifies_star_and_reads_points(self):
        context = self.context()
        result = inspect_rule(context, self.rule, self.frame('main-2200'))
        self.assertTrue(result['matched'], result)
        self.assertEqual(result['asset'], ASSET)
        self.assertGreater(result['template'], .98)
        self.assertEqual(result['number']['value'], 2200)

    def test_real_low_balance_is_seen_but_does_not_purchase(self):
        result = inspect_rule(self.context(), self.rule, self.frame('main-262'))
        self.assertTrue(result['visual'], result)
        self.assertFalse(result['matched'], result)
        self.assertEqual(result['number']['value'], 262)

    def test_dim_battle_and_globally_dimmed_main_are_rejected(self):
        frame = self.frame('main-2200')
        for dim in [self.frame('battle-255'), (frame.astype(float) * .7).astype(np.uint8)]:
            result = inspect_rule(self.context(), self.rule, dim)
            self.assertFalse(result['visual'], result)

    def test_missing_icon_or_bright_core_is_rejected(self):
        for area in [[645, 273, 707, 323], [665, 293, 685, 311]]:
            frame = self.frame('main-2200').copy()
            x1,y1,x2,y2 = area
            frame[y1:y2,x1:x2] = 0
            self.assertFalse(inspect_rule(self.context(), self.rule, frame)['visual'])

    def test_500_boundary_and_uncertain_digit_guard_are_unchanged(self):
        frame = self.frame('main-2200')
        _,button,offset = Clicker.load_rule(self.context(), self.rule)
        self.assertTrue(button.match(frame, offset=offset, similarity=self.rule['threshold']))
        for value,confidence,expected in [(500,.999,False),(501,.999,True),
                (2200,.92,False),(None,.999,False)]:
            reader = SimpleNamespace(read=lambda crop,v=value,c=confidence:
                                      NumberReading(str(v),v,c))
            self.assertEqual(Clicker.number_condition(self.context(reader),self.rule,button,frame)[0],expected)

    def test_three_old_seasons_keep_their_page_and_purchase_guards(self):
        for result in replay(self.config):
            with self.subTest(frame=result['frame'],rule=result['rule']):
                self.assertEqual(result['matched'],result['expected'],result)

    def test_migration_is_idempotent_and_preserves_switches_and_records(self):
        before = deepcopy(self.before)
        rule = next(r for r in before['rules'] if r['id'] == ENTRY)
        rule.update(enabled=False, interval=91)
        before['custom_user_value'] = {'keep': True}
        after = update(before)
        self.assertEqual(update(after),after)
        self.assertEqual({k:v for k,v in before.items() if k!='rules'},
                         {k:v for k,v in after.items() if k!='rules'})
        for old,new in zip(before['rules'],after['rules'],strict=True):
            if old['id'] != ENTRY:
                self.assertEqual(old,new)
            else:
                old=deepcopy(old);new=deepcopy(new)
                old.pop('asset_variants',None);new.pop('asset_variants',None)
                old['color_condition'].pop('source_area');new['color_condition'].pop('source_area')
                self.assertEqual(old,new)

    def test_unknown_user_layout_is_not_rewritten(self):
        config=deepcopy(self.config)
        next(r for r in config['rules'] if r['id']==ENTRY)['source_crop']=[640,270,701,320]
        with self.assertRaises(ValueError):
            update(config)


if __name__ == '__main__':
    unittest.main()
