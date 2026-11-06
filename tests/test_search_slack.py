"""搜索框保底余量:以前 search_area 写得和模板一样大,位移几像素就失配。"""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import Clicker
from drift.runtime import SEARCH_SLACK_X, SEARCH_SLACK_Y, search_with_slack


class SearchWithSlackTests(unittest.TestCase):
    def test_grows_a_zero_slack_box(self):
        search = (100, 100, 130, 130)
        template = (100, 100, 130, 130)
        self.assertEqual(search_with_slack(search, template, 720, 1280),
                         (100 - SEARCH_SLACK_X, 100 - SEARCH_SLACK_Y,
                          130 + SEARCH_SLACK_X, 130 + SEARCH_SLACK_Y))

    def test_never_shrinks_an_existing_wider_box(self):
        search = (10, 20, 700, 1200)
        grown = search_with_slack(search, (300, 400, 340, 440), 720, 1280)
        self.assertEqual(grown, search)

    def test_clamped_into_the_frame(self):
        grown = search_with_slack((0, 0, 720, 1280), (2, 2, 30, 30), 720, 1280)
        self.assertEqual(grown, (0, 0, 720, 1280))
        self.assertGreaterEqual(grown[0], 0)
        self.assertGreaterEqual(grown[1], 0)
        self.assertLessEqual(grown[2], 720)
        self.assertLessEqual(grown[3], 1280)


class LoadedRuleSlackTests(unittest.TestCase):
    """所有裁图规则加载后都应带足余量(除非显式退出)。"""

    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8-sig'))
        cls.clicker = Clicker(ROOT, lambda cfg, stop: SimpleNamespace(image=None))
        cls.clicker.config = cls.config
        cls.rows = []
        for rule in cls.config['rules']:
            if not rule.get('source_crop') or rule.get('search_slack') is False:
                continue
            try:
                _, button, _ = cls.clicker.load_rule(dict(rule))
            except Exception:
                continue
            # green_reward_list 是列表型按钮,没有 variants;它的范围另有约束。
            for variant, offset, _companions in getattr(button, 'variants', ()):
                width, height = variant.image.shape[1], variant.image.shape[0]
                vertical = (offset[3] + height) - offset[1] - height
                horizontal = (offset[2] + width) - offset[0] - width
                cls.rows.append((rule['id'], vertical, horizontal))

    def test_every_cropped_rule_keeps_vertical_slack(self):
        tight = [row for row in self.rows if row[1] < SEARCH_SLACK_Y]
        self.assertEqual(tight, [], f'纵向余量不足: {tight[:5]}')

    def test_every_cropped_rule_keeps_horizontal_slack(self):
        tight = [row for row in self.rows if row[2] < SEARCH_SLACK_X]
        self.assertEqual(tight, [], f'横向余量不足: {tight[:5]}')

    def test_known_zero_slack_rules_are_fixed(self):
        # 这两条原来是 0px / 3px,是最容易"UI 一动就不认"的
        for rule_id, minimum in (('driftinggirl_pass_daily', SEARCH_SLACK_Y),
                                 ('driftinggirl_task_popup_return', SEARCH_SLACK_Y)):
            rows = [row for row in self.rows if row[0] == rule_id]
            self.assertTrue(rows, f'{rule_id} 没有被加载')
            self.assertTrue(all(row[1] >= minimum for row in rows),
                            f'{rule_id} 余量仍不足: {rows}')


if __name__ == '__main__':
    unittest.main()
