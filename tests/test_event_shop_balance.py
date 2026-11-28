"""The event shop balance changes width as material is earned."""
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from drift.perception.digit_ocr import DigitReader
from drift.ui.event_shop import EventShopVisual


class EventShopBalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        reader = DigitReader(root)
        cls.shop = EventShopVisual(root,lambda:reader)
        with Image.open(root/'tests/fixtures/toy-shop/balance-1045.png') as image:
            cls.frame = np.array(image.convert('RGB'))

    def test_four_digit_balance_is_read_and_skin_stays_first(self):
        self.assertTrue(self.shop.toy_shop(self.frame))
        self.assertEqual(self.shop.balance(self.frame),1045)
        decision,_ = self.shop.first_toy_offer(self.frame)
        self.assertEqual(decision.kind,'skip')
        self.assertEqual(decision.balance,1045)
        self.assertIn('等待 2000',decision.reason)

    def test_missing_cap_or_amount_is_never_read_as_material(self):
        for area in (np.s_[441:461,247:312],np.s_[433:462,160:243]):
            changed = self.frame.copy()
            changed[area] = 200
            self.assertIsNone(self.shop.balance(changed))
