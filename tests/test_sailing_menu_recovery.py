"""Regress the native RANK35 AUTO-panel deadlock and reject changed contexts."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from digit_ocr import DigitReader, NumberReading
from parallel_world import ParallelWorld

ROOT = Path(__file__).resolve().parents[1]


class SailingMenuRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)
        cls.image = np.asarray(Image.open(ROOT/'tests/fixtures/training-sailing/menu-rank35-boss.png').convert('RGB'))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'training.json'
        self.actions = []
        self.w = ParallelWorld(ROOT,lambda *a:self.fail('Unexpected map input'),lambda *a:None,
            lambda _:None,self.reader,training_state_path=self.path,
            sailing_action=self.click,fishing_action=lambda *a:False)
        self.w.training.begin(35,814,1112)
        self.w.phase,self.w.rank,self.w.target_rank = 'play',35,35

    def tearDown(self):
        self.temp.cleanup()

    def click(self,action,x,y,label):
        self.w.sailing.before_click(action)
        self.actions.append((action,x,y))
        return True

    def test_native_dimmed_background_does_not_starve_menu_configuration(self):
        w = self.w
        w.sailing.phase = 'opening'
        self.assertTrue(w._parallel(self.image))
        self.assertTrue(w.sailing.menu(self.image))
        self.assertFalse(w.fishing.world_matches(self.image))
        self.assertEqual(w._sailing_menu_rank(self.image),35)
        self.assertTrue(w.step(self.image))
        self.assertEqual(self.actions,[('target:amber',417,458)])
        self.assertEqual((w.training.state['rank'],w.training.state['boss_level'],w.training.state['start_level']),(35,1112,814))
        self.assertEqual(w.training.retry_level(),1092)

    def test_saved_farming_menu_can_be_adopted_after_restart(self):
        w = self.w
        w.phase,w.rank,w.target_rank = 'detect',None,None
        for _ in range(3):
            w.sailing.next_at = 0
            w.step(self.image)
        self.assertEqual(w.rank,35)
        self.assertEqual(self.actions,[('target:amber',417,458)])

    def test_current_changed_or_unknown_rank_cannot_toggle_or_start(self):
        w = self.w
        for rank in (None,34,36,40):
            w.rank = rank
            for action in ('target:amber','start','close'):
                self.assertFalse(w.sailing.validate(action,self.image),(rank,action))
        w.rank = 35
        for values in ((35,35,35,36),(35,35,None,35)):
            results = [NumberReading(str(v),v,.999) for v in values]
            w.reader = Mock(read=Mock(side_effect=results))
            self.assertIsNone(w._sailing_menu_rank(self.image))
        w.reader = self.reader
        w.rank = 36
        w.sailing.phase = 'opening'
        with patch.object(w,'_rank'):
            self.assertTrue(w.step(self.image))
        self.assertEqual(self.actions,[])

    def test_world_badge_or_bright_panel_missing_rejects_inputs(self):
        w = self.w
        missing = self.image.copy()
        missing[39:79,6:46] = 0
        for image in (missing,(self.image*.7).astype('uint8')):
            self.assertIsNone(w._sailing_menu_rank(image))
            for action in ('target:amber','start','close'):
                self.assertFalse(w.sailing.validate(action,image))

    def test_menu_with_running_or_unknown_top_auto_cannot_change_targets(self):
        w = self.w
        active = np.asarray(Image.open(ROOT/'tests/fixtures/training-sailing/active.png').convert('RGB'))
        running = self.image.copy()
        running[4:65,186:254] = (active[4:65,186:254]*149/255).astype('uint8')
        unknown = self.image.copy()
        unknown[4:65,186:254] = 0
        for image in (running,unknown):
            self.assertTrue(w.sailing.menu(image))
            for action in ('target:amber','start','close'):
                self.assertFalse(w.sailing.validate(action,image))
            self.assertTrue(w.step(image))
        self.assertEqual(self.actions,[])

    def test_context_rejections_and_outer_wait_cannot_extend_sailing_timeout(self):
        w = self.w
        w.sailing.phase = 'opening'
        w.sailing.since -= 41
        w.next_action_at = float('inf')
        saved = self.path.read_bytes()
        with self.assertRaisesRegex(RuntimeError,'保留练级记录'):
            w.step(self.image)
        self.assertEqual(self.actions,[])
        self.assertEqual(self.path.read_bytes(),saved)

    def test_start_requires_current_farming_settings_and_exact_three_resources(self):
        w = self.w
        self.assertTrue(w.sailing.validate('target:amber',self.image))
        self.assertFalse(w.sailing.validate('start',self.image))
        configured = self.image.copy()
        reference = np.asarray(Image.open(ROOT/'tests/fixtures/training-sailing/menu-three.png').convert('RGB'))
        configured[421:491,239:597] = reference[421:491,239:597]
        self.assertTrue(w.sailing.configured(configured))
        self.assertTrue(w.sailing.validate('start',configured))
        w.training.options['auto_sailing'] = False
        self.assertFalse(w.sailing.validate('start',configured))
        self.assertFalse(w.sailing.validate('target:amber',self.image))


if __name__ == '__main__':
    unittest.main()
