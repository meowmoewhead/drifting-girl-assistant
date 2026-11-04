"""端到端验证:return 步骤超时后,页面图把"停机"换成"退回主画面"。

构造方式贴近真实故障:皇冠页上的按钮突然识别不到(游戏改版 / 模板漂移),
但"关闭"仍然可见。此前这种情况会抛 "未识别到后续界面" 停掉整轮;
标了 ``ui_nav`` 之后,页面图会算出"这一步该点关闭",退回主画面继续跑。

做法:把该阶段两条规则的阈值改成 1.01(不可能命中),模拟"预期按钮消失";
关闭按钮的规则保持原样,供页面图识别当前页并选择退回边。
"""
import json
import shutil
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import Clicker  # noqa: E402

HIDDEN = ('driftinggirl_achievements_crown', 'driftinggirl_achievements_ready')
FLAGGED = 'driftinggirl_achievements_crown'


class NavRetreatReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8-sig'))
        cls.frames = {
            'main': cls.image('assets/navigation/main.png'),
            # 皇冠页:关闭可见,列表按钮识别不到
            'crown': cls.image('tests/fixtures/legacy-captures/achievements-cycle-result.png'),
        }

    @staticmethod
    def image(relative):
        from PIL import Image
        return np.asarray(Image.open(ROOT / relative).convert('RGB'))

    def replay(self, *, ui_nav, claimable=False, skip_on_timeout=True):
        config = deepcopy(self.config)
        config['poll_interval'] = .1
        config['rules'] = [rule for rule in config['rules']
                           if rule['id'].startswith('driftinggirl_achievements_')]
        for rule in config['rules']:
            rule.update(enabled=True, interval=.2, next_delay=0)
            if rule['id'] in HIDDEN:
                rule['threshold'] = 1.0       # 模拟"预期按钮识别不到"
            if ui_nav and rule['id'] == FLAGGED:
                rule['ui_nav'] = True
            if rule['id']=='driftinggirl_achievements_open':
                rule['skip_on_timeout']=skip_on_timeout
        frames, clock = dict(self.frames), [0]
        if claimable:
            frames['crown']=self.image('assets/achievements-list-native.png')

        class Device:
            def __init__(self, cfg, stop):
                self.stop, self.image, self.taps = stop, None, []

            def screenshot(self):
                clock[0] += 1
                # 0 次点击:主画面;1 次(已打开皇冠页):按钮失效页;之后:主画面
                count = len(self.taps)
                self.image = frames['main' if count in (0, 2) else 'crown'].copy()
                return self.image

            def click(self, button):
                if self.stop.is_set():
                    raise InterruptedError()
                self.taps.append(button.name.removeprefix('driftinggirl_achievements_'))

            def stuck_record_add(self, _button):
                pass

        with tempfile.TemporaryDirectory(prefix='nav-retreat-') as folder:
            root = Path(folder)
            for path in {rule['asset'] for rule in config['rules']}:
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / path, root / path)
            (root / 'config.json').write_text(json.dumps(config), encoding='utf-8')
            app = Clicker(root, Device)
            try:
                with patch('app.time', SimpleNamespace(monotonic=lambda: clock[0])):
                    app.start(preview=False, stop_on_main=True, max_clicks=8, timeout=180)
                    app.thread.join(60)
                    self.assertFalse(app.running(), '回放没有按时结束')
                return deepcopy(app.state), list(app.device.taps), list(app.events)
            finally:
                app.stop()
                if app.thread:
                    app.thread.join(3)
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_enabled_return_step_walks_back_to_main(self):
        state, taps, events = self.replay(ui_nav=True)
        self.assertIsNone(state['last_error'], state['last_error'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')
        # 一次打开 + 一次由页面图发出的关闭;没有多余的盲目点击。
        self.assertEqual(taps, ['open', 'close_cleanup'])
        messages = [event['message'] for event in events]
        self.assertTrue(any('页面图退回建议' in message for message in messages), messages[-6:])
        self.assertTrue(any('连续三帧三秒' in message for message in messages), messages[-6:])
        self.assertEqual(sum(state['optional_cleanup']['attempts'].values()),1)

    def test_no_graph_still_uses_authorized_optional_cleanup(self):
        state, taps, _ = self.replay(ui_nav=False)
        self.assertIsNone(state['last_error'])
        self.assertEqual(taps, ['open','close_cleanup'])

    def test_claimable_rewards_are_abandoned_only_by_explicit_cleanup(self):
        state,taps,_=self.replay(ui_nav=True,claimable=True)
        self.assertIsNone(state['last_error'])
        self.assertEqual(taps,['open','close_cleanup'])

    def test_saved_no_skip_is_not_overridden_by_navigation(self):
        state,taps,_=self.replay(ui_nav=True,skip_on_timeout=False)
        self.assertIn('未识别到后续界面',state['last_error'])
        self.assertEqual(taps, ['open'])


if __name__ == '__main__':
    unittest.main()
