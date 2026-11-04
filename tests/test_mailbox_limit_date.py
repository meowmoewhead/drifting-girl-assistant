"""Real mailbox limit frames: dates and item counts cannot block safe closing."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from app import Clicker

ROOT = Path(__file__).resolve().parents[1]

class MailboxLimitDateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.rule = next(r for r in cls.config['rules'] if r['id']=='driftinggirl_mailbox_limit_close')
        cls.context = SimpleNamespace(root=ROOT, config=cls.config)
        cls.current = cls.image('tests/fixtures/mailbox-limit/current-date-two-items.png')

    @staticmethod
    def image(name):
        with Image.open(ROOT/name) as image:
            return np.array(image.convert('RGB'))

    def match(self, image, rule=None):
        loaded, button, offset = Clicker.load_rule(self.context, self.rule if rule is None else rule)
        return button.match(image, offset=offset, similarity=loaded['threshold']), button, loaded

    def test_real_current_and_original_notice_close_same_button(self):
        for frame in (self.current, self.image('assets/mailbox-limit-native.png')):
            passed, button, _ = self.match(frame)
            self.assertTrue(passed)
            self.assertEqual(button.button, (612,170,690,230))

    def test_changed_date_item_count_and_expiration_are_excluded(self):
        frame = self.current.copy()
        for x1,y1,x2,y2 in ((215,356,390,387),(132,350,190,390),(506,348,577,382),(214,466,393,495)):
            frame[y1:y2,x1:x2] = 0
        self.assertTrue(self.match(frame)[0])

    def test_notice_header_claim_button_and_close_cross_are_required(self):
        for x1,y1,x2,y2 in ((216,319,390,350),(144,228,197,253),(488,987,582,1019),(615,175,685,225)):
            with self.subTest(box=(x1,y1,x2,y2)):
                frame = self.current.copy()
                frame[y1:y2,x1:x2] = 0
                self.assertFalse(self.match(frame)[0])

    def test_dark_modal_and_other_screens_rejected(self):
        for frame in (np.uint8(self.current.astype(float)*.4),
                      self.image('assets/mailbox-empty-native.png'),
                      self.image('assets/raid/waiting.png'),
                      self.image('assets/main-ui-native.png')):
            self.assertFalse(self.match(frame)[0])

    def test_saved_rule_and_workflow_parameters_are_unchanged(self):
        before = deepcopy(self.rule)
        _, _, loaded = self.match(self.current)
        self.assertEqual(self.rule, before)
        for key in ('enabled','threshold','stable_frames','interval','stage','next_stage','stage_timeout','next_delay'):
            self.assertEqual(loaded[key], before[key])

    def test_custom_geometry_is_not_overridden(self):
        custom = {**deepcopy(self.rule), 'source_crop':[217,343,453,369]}
        _, _, loaded = self.match(self.current, custom)
        self.assertEqual(loaded, custom)

    def test_existing_limit_page_closes_once_without_repeating_claim(self):
        from tests.test_fault_recovery import FaultRecoveryTests
        clock = [0]
        limited = self.current
        returned = self.image('tests/fixtures/legacy-captures/mailbox-return.png')
        class Device:
            def __init__(self,config,stop):
                self.stop,self.image,self.taps,self.phase_frames = stop,None,[],0
            def screenshot(self):
                self.phase_frames += 1
                clock[0] += 1
                self.image = (returned if self.taps else limited).copy()
                return self.image
            def click(self,button):
                assert button.name == 'driftinggirl_mailbox_limit_close'
                assert not self.taps and clock[0]>=3
                self.taps.append(button.name)
                self.phase_frames = 0
            def stuck_record_add(self,button):pass
        state,taps,_ = FaultRecoveryTests.replay(self,'mailbox',Device,clock)
        self.assertEqual(taps,['driftinggirl_mailbox_limit_close'])
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'],'cycle_complete')

if __name__ == '__main__':
    unittest.main()
