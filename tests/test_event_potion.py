"""Event currency buys only the first 500-cost potion, once per completed cycle."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import Clicker
from digit_ocr import DigitReader, NumberReading


class EventPotionTests(unittest.TestCase):
    season = ''
    main_balance = 1305

    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.rules = {r['id']: r for r in cls.config['rules'] if r['id'].startswith('driftinggirl_event_potion_')}
        cls.context = SimpleNamespace(root=ROOT, config=cls.config)
        cls.frames = {name: np.asarray(Image.open(ROOT/'assets'/'event-potion'/(name+cls.season+'.png')).convert('RGB'))
                      for name in ('main', 'shop', 'confirm')}
        cls.reader = DigitReader(ROOT)

    def match(self, name, frame):
        rule, button, offset = Clicker.load_rule(self.context, self.rules['driftinggirl_event_potion_'+name])
        matched = button.match(frame, offset=offset, similarity=rule['threshold'])
        if matched and rule.get('absent_rule'):
            other = self.rules[rule['absent_rule']]
            _, guard, guard_offset = Clicker.load_rule(self.context, other)
            matched = not guard.match(frame, offset=guard_offset, similarity=other['threshold'])
        return matched, button

    def test_material_boundary_and_uncertain_numbers(self):
        rule = self.rules['driftinggirl_event_potion_open']
        matched, button = self.match('open', self.frames['main'])
        self.assertTrue(matched)
        context = SimpleNamespace(digit_reader=None, root=ROOT, lock=threading.RLock(),
                                  state={'numbers': {}, 'frames': 1})
        for value, confidence, expected in [(499, .999, False), (500, .999, False),
                (501, .999, True), (1305, .999, True), (1305, .92, False), (None, .999, False)]:
            context.digit_reader = SimpleNamespace(read=lambda crop, v=value, c=confidence:
                                                  NumberReading(str(v), v, c))
            with self.subTest(value=value, confidence=confidence):
                self.assertEqual(Clicker.number_condition(context, rule, button, self.frames['main'])[0], expected)
        context.digit_reader = self.reader
        self.assertEqual(Clicker.number_condition(context, rule, button, self.frames['main']), (True, self.main_balance))
        unknown = self.frames['main'].copy()
        unknown[324:347, 645:708] = 240
        self.assertFalse(Clicker.number_condition(context, rule, button, unknown)[0])

    def test_entry_does_not_match_dimmed_icon_behind_either_window(self):
        self.assertTrue(self.match('open', self.frames['main'])[0])
        for name in ('shop', 'confirm'):
            with self.subTest(frame=name):
                self.assertFalse(self.match('open', self.frames[name])[0])
        changed = self.frames['main'].copy()
        changed[324:347, 645:708] = 0
        self.assertTrue(self.match('open', changed)[0])  # Count is not part of the icon template.

    def test_select_requires_first_potion_500_price_and_event_shop(self):
        matched, button = self.match('select', self.frames['shop'])
        self.assertTrue(matched)
        self.assertTrue(120 < button.button[0] < button.button[2] < 226)
        self.assertTrue(598 < button.button[1] < button.button[3] < 708)
        for area in ((140,715,214,741), (143,621,206,694), (557,521,597,564)):
            frame = self.frames['shop'].copy()
            x1,y1,x2,y2 = area
            frame[y1:y2,x1:x2] = 0
            with self.subTest(missing=area):
                self.assertFalse(self.match('select', frame)[0])
        self.assertFalse(self.match('select', self.frames['main'])[0])
        self.assertFalse(self.match('select', self.frames['confirm'])[0])
        changed = self.frames['shop'].copy()
        changed[522:560, 160:308] = 0  # Balance and collection cap may change.
        changed[814:842, 197:529] = 0  # Event countdown may change.
        self.assertTrue(self.match('select', changed)[0])

    def test_purchase_requires_exact_price_and_active_buy_button(self):
        # 活动币图案每期都会换（月亮币/糖果/幽灵），所以 confirm 的语义身份是可变的
        # 文案之外的固定部分：蓝色 500 价格与绿色购买按钮。币图区域不再参与判定，
        # 见 2026-10-27 的幽灵活动修复（币图从 identity 中移除，只保留价格与按钮）。
        self.assertTrue(self.match('confirm', self.frames['confirm'])[0])
        for area in ((265,577,315,603), (165,769,256,816)):
            frame = self.frames['confirm'].copy()
            x1,y1,x2,y2 = area
            frame[y1:y2,x1:x2] = 0
            with self.subTest(missing=area):
                self.assertFalse(self.match('confirm', frame)[0])
        disabled = self.frames['confirm'].copy()
        disabled[772:817, 90:143] = 100
        self.assertFalse(self.match('confirm', disabled)[0])
        for name in ('main', 'shop'):
            self.assertFalse(self.match('confirm', self.frames[name])[0])
        for path in ('assets/potion-ad-confirm-source.png', 'assets/mana-confirm-native.png'):
            if (ROOT/path).is_file():
                unrelated = np.asarray(Image.open(ROOT/path).convert('RGB'))
                self.assertFalse(self.match('confirm', unrelated)[0])

    def test_close_waits_for_shop_and_return_requires_main_screen(self):
        self.assertTrue(self.match('close', self.frames['shop'])[0])
        self.assertFalse(self.match('close', self.frames['confirm'])[0])
        self.assertFalse(self.match('close', self.frames['main'])[0])
        self.assertTrue(self.match('return', self.frames['main'])[0])
        self.assertFalse(self.match('return', self.frames['shop'])[0])
        self.assertFalse(self.match('return', self.frames['confirm'])[0])

    def test_full_cycle_buys_once_then_closes_even_when_materials_remain(self):
        frames = self.frames
        expected = ['open', 'select', 'confirm', 'close']
        class ReplayDevice:
            def __init__(self, config, stop):
                self.stop, self.image, self.taps = stop, None, []
            def screenshot(self):
                name = ['main', 'shop', 'confirm', 'shop', 'main'][min(len(self.taps), 4)]
                self.image = frames[name].copy()
                return self.image
            def click(self, button):
                if self.stop.is_set():
                    raise InterruptedError()
                index = len(self.taps)
                if index >= 4 or button.name != 'driftinggirl_event_potion_'+expected[index]:
                    raise AssertionError('Unexpected purchase flow action: '+button.name)
                self.taps.append(button.name)
            def stuck_record_add(self, button):
                pass
        with tempfile.TemporaryDirectory(prefix='event-potion-replay-') as directory:
            root = Path(directory)
            shutil.copytree(ROOT/'assets'/'event-potion', root/'assets'/'event-potion')
            config = deepcopy(self.config)
            config['poll_interval'] = .1
            config['rules'] = deepcopy(list(self.rules.values()))
            for rule in config['rules']:
                rule.update(enabled=True, interval=.2, next_delay=0, stage_timeout=5)
            (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
            app = Clicker(root, ReplayDevice)
            app.digit_reader = self.reader
            try:
                app.start(preview=False, stop_on_main=True, max_clicks=5, timeout=12)
                app.thread.join(15)
                self.assertFalse(app.running())
                self.assertIsNone(app.state['last_error'])
                self.assertEqual(app.state['stop_reason'], 'cycle_complete')
                self.assertEqual(app.state['clicks'], 4)
                self.assertEqual(app.device.taps, ['driftinggirl_event_potion_'+name for name in expected])
            finally:
                app.stop()
                if app.thread:
                    app.thread.join(3)
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)


class CandyEventPotionTests(EventPotionTests):
    """Replay the same safeguards and complete workflow on the October event."""
    season = '-candy'
    main_balance = 2200

    def test_wrong_product_price_is_rejected_and_currency_art_is_ignored(self):
        expensive = self.frames['shop'].copy()
        # Replace the first price with the observed 2000-candy product price.
        expensive[715:741, 168:214] = self.frames['shop'][715:741, 256:302]
        self.assertFalse(self.match('select', expensive)[0])
        # 活动币图案每期都换，confirm 不再用币图判身份：涂掉币图不应改变判定。
        unknown_currency = self.frames['confirm'].copy()
        unknown_currency[575:606, 228:259] = (30, 140, 240)
        self.assertTrue(self.match('confirm', unknown_currency)[0])

    def test_return_ignores_animated_skill_and_central_spiral_but_rejects_dim_main(self):
        changed = self.frames['main'].copy()
        changed[1140:1200, 0:82] = 0
        changed[1125:1200, 300:425] = 0
        changed[324:347, 645:708] = 0
        self.assertTrue(self.match('return', changed)[0])
        dim = (self.frames['main'].astype(float) * .7).astype(np.uint8)
        self.assertFalse(self.match('open', dim)[0])
        self.assertFalse(self.match('return', dim)[0])
        for area in ((211,1140,266,1189), (542,1140,597,1183)):
            missing = self.frames['main'].copy()
            x1,y1,x2,y2 = area
            missing[y1:y2,x1:x2] = 0
            self.assertFalse(self.match('return', missing)[0])

    def test_migration_preserves_user_choices_and_all_other_workflows(self):
        from tools.configure_event_candy import update, IDS
        before = deepcopy(self.config)
        for rule in before['rules']:
            if rule['id'] in IDS:
                rule['enabled'] = False
                rule['interval'] = 90 if rule['id'] == IDS[0] else 3
        after = update(before)
        self.assertEqual(update(after), after)
        self.assertEqual(len(after['rules']), len(before['rules']))
        self.assertEqual({k:v for k,v in before.items() if k != 'rules'},
                         {k:v for k,v in after.items() if k != 'rules'})
        for old,new in zip(before['rules'], after['rules']):
            if old['id'] not in IDS:
                self.assertEqual(old, new)
            else:
                self.assertEqual(old['enabled'], new['enabled'])
                self.assertEqual(old['interval'], new['interval'])


if __name__ == '__main__':
    unittest.main()
