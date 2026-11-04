"""公会入口「粉色 G 徽标」的动画相位兜底。

2026-10-27 20:41 事故:船已停在港口,`travel` 阶段只等 `g-icon` 连续三帧命中,
但那枚 G 徽标有脉冲动画(实测 34px ↔ 38px 两个相位),固定尺度模板在胀大相位只有
0.60~0.92,低于 0.96 阈值,于是 600 秒都凑不齐三帧 → 往返中止。这里把兜底判据钉住:
模板失配时用「粉色像素量 + 外接框尺寸」识别,且不能被活动商店/地图等画面误命中。
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drift.tasks.guild_daily import GuildVisual  # noqa: E402


def frame_with_badge(x, y, size, rgb):
    """在 1280x720 的空白帧上画一块方形的「徽标」。"""
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    frame[y:y + size, x:x + size] = rgb
    return frame


class GuildBadgePulseTests(unittest.TestCase):
    def setUp(self):
        self.visual = GuildVisual.__new__(GuildVisual)
        self.visual.spec = {'g-icon': {'file': 'g-icon.png', 'area': [610, 500, 720, 1130],
                                       'scales': [0.95, 1, 1.05], 'threshold': .96}}
        self.visual.images = {}
        self.visual._frame = None
        self.visual._cache = {}
        # find() 恒不命中，模拟脉冲胀大相位
        self.visual.find = Mock(return_value=None)

    def test_badge_is_found_when_the_template_misses(self):
        frame = frame_with_badge(668, 999, 27, (235, 160, 225))
        self.assertEqual(self.visual.point(frame, 'g-icon'), (681, 1012))

    def test_badge_is_cached_after_the_fallback(self):
        frame = frame_with_badge(668, 999, 27, (235, 160, 225))
        self.visual.point(frame, 'g-icon')
        self.assertEqual(self.visual._cache['g-icon'][0], (681, 1012))

    def test_template_hit_wins_and_fallback_is_not_consulted(self):
        self.visual.find = Mock(return_value=((679, 1013), (662, 996, 34, 34), .99))
        frame = frame_with_badge(668, 999, 27, (235, 160, 225))
        self.assertEqual(self.visual.point(frame, 'g-icon'), (679, 1013))

    def test_other_names_never_use_the_fallback(self):
        frame = frame_with_badge(668, 999, 27, (235, 160, 225))
        self.assertIsNone(self.visual.point(frame, 'guild-location'))

    def test_empty_sidebar_is_not_a_badge(self):
        frame = np.zeros((1280, 720, 3), dtype=np.uint8)
        self.assertIsNone(self.visual.point(frame, 'g-icon'))

    def test_ordinary_scene_colours_are_not_a_badge(self):
        # 活动商店/地图那种青绿、土黄背景不该触发
        for rgb in ((40, 120, 110), (190, 160, 110), (30, 30, 30)):
            frame = frame_with_badge(668, 999, 27, rgb)
            self.assertIsNone(self.visual.point(frame, 'g-icon'), msg=str(rgb))

    def test_wrong_size_blob_is_rejected(self):
        # 太小的粉点（< 18px）与太大的一片（> 40px）都不算徽标
        self.assertIsNone(self.visual.point(frame_with_badge(668, 999, 12, (235, 160, 225)), 'g-icon'))
        self.assertIsNone(self.visual.point(frame_with_badge(650, 980, 60, (235, 160, 225)), 'g-icon'))

    def test_badge_outside_the_search_area_is_ignored(self):
        frame = frame_with_badge(200, 300, 27, (235, 160, 225))
        self.assertIsNone(self.visual.point(frame, 'g-icon'))


if __name__ == '__main__':
    unittest.main()
