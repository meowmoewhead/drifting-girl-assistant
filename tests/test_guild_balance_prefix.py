"""Real 780-point guild balance with its shifted blue coin inside the crop."""
from pathlib import Path
import unittest
from unittest.mock import Mock
import cv2
import numpy as np
from digit_ocr import DigitReader, NumberReading
from purchase_checks import currency_floor, read_anchored_guild_balance, read_guild_balance

ROOT=Path(__file__).resolve().parents[1]

class GuildBalancePrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader=DigitReader(ROOT)
        cls.crop=cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/guild-balance-780.png')),cv2.COLOR_BGR2RGB)

    def test_real_balance_keeps_value_and_meets_original_confidence(self):
        raw=self.reader.read(self.crop)
        self.assertEqual(raw.value,780);self.assertLess(raw.confidence,.93)
        result=read_guild_balance(self.reader,self.crop)
        self.assertEqual(result.value,780);self.assertGreaterEqual(result.confidence,.93)

    def test_disagreement_or_low_confidence_is_not_promoted(self):
        raw=NumberReading('780',780,.78)
        for other in (NumberReading('780',780,.8),NumberReading('7800',7800,.99)):
            reader=Mock();reader.read.side_effect=[raw,NumberReading('780',780,.99),other]
            self.assertEqual(read_guild_balance(reader,self.crop),raw)

    def test_missing_blue_currency_cannot_rescue_low_confidence(self):
        reader=Mock();raw=NumberReading('780',780,.78);reader.read.return_value=raw
        self.assertEqual(read_guild_balance(reader,np.full_like(self.crop,200)),raw)
        self.assertEqual(reader.read.call_count,1)

    def test_clear_and_unknown_text_keep_original_reading(self):
        for raw in (NumberReading('4.22k',None,.99),NumberReading('?',None,.6)):
            reader=Mock();reader.read.return_value=raw
            self.assertEqual(read_guild_balance(reader,self.crop),raw)
            self.assertEqual(reader.read.call_count,1)

    def test_blue_coin_anchors_both_abbreviated_and_shifted_integer_amounts(self):
        def frame(path):
            return cv2.cvtColor(cv2.imread(str(ROOT/path)),cv2.COLOR_BGR2RGB)
        original=frame('assets/guild-shop/guild.png')
        coin=original[348:377,525:554]
        shifted=frame('tests/fixtures/guild-selection/close-balance-980-20261019.png')
        for screenshot,balance in ((original,4210),(shifted,980)):
            lane=screenshot[342:381,460:643]
            self.assertEqual(currency_floor(read_anchored_guild_balance(self.reader,lane,coin),.93),balance)
        unknown=original[342:381,460:643].copy()
        unknown[:,65:96]=220
        wrong=Mock()
        self.assertIsNone(read_anchored_guild_balance(wrong,unknown,coin))
        wrong.read.assert_not_called()
        wrong.read.side_effect=[NumberReading('4.22k',None,.99),NumberReading('4.72k',None,.99)]
        self.assertIsNone(read_anchored_guild_balance(wrong,original[342:381,460:643],coin))

if __name__=='__main__':unittest.main()
