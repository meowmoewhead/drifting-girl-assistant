"""每日奖励"领不上就跳过":中途超时不得把整轮挂机停掉。

2026-10-11 真机现场:点击「每日奖励：本周期未完成时打开商店」后,游戏停留在一个脚本
不认识的画面(当天是公会商店盖在上面),阶段超时后整轮运行停止:
    RuntimeError: 执行「每日奖励：本周期未完成时打开商店」后未识别到后续界面，已停止

用户要求"如果领不上就跳过"。这里守住:
- 规则上标了 skip_on_timeout 的步骤,超时后回到主界面继续挂机,且本轮**不记为完成**
  (下一次按入口自己的间隔重试,每日奖励是 3600 秒);
- 没标这个标记的步骤必须保持原样**停机**(其它流程仍要人看画面);
- skip_on_timeout 必须显式声明为布尔,避免配置写错被静默当成关闭。
"""
from types import SimpleNamespace
import unittest

from drift.core.session import LoopState


class SkipOnTimeoutStateTests(unittest.TestCase):
    def test_state_defaults_to_stopping(self):
        state = LoopState()
        self.assertFalse(state.skip_on_timeout, "默认必须维持原有停机行为")

    def test_return_to_main_clears_the_skip_flag(self):
        state = LoopState()
        state.skip_on_timeout = True
        state.stage, state.stage_trigger = 'daily_tab', '每日奖励'
        state.reset_after_return()
        self.assertFalse(state.skip_on_timeout)
        self.assertEqual(state.stage, 'main')


class SkipOnTimeoutDeclarationTests(unittest.TestCase):
    def test_daily_steps_declare_skip_on_timeout(self):
        from pathlib import Path
        import json

        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / 'config.json').read_text(encoding='utf-8-sig'))
        rules = {rule['id']: rule for rule in config['rules']}
        for identifier in ('driftinggirl_daily_premium', 'driftinggirl_daily_claim',
                           'driftinggirl_daily_unavailable', 'driftinggirl_daily_close'):
            self.assertTrue(rules[identifier].get('skip_on_timeout'), identifier)
        # 入口规则本身在 main 阶段,不该也不需要这个标记
        self.assertNotIn('skip_on_timeout', rules['driftinggirl_daily_open'])

    def test_validation_rejects_non_boolean(self):
        import json
        import shutil
        import tempfile
        from pathlib import Path

        from drift.runtime import Clicker

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'assets').mkdir()
            shutil.copy(root / 'assets/daily-reward/main.png', folder / 'assets/main.png')
            config = {'adb_path': 'fake', 'serial': 'test', 'resolution': [720, 1280],
                      'poll_interval': .1, 'rules': []}
            (folder / 'config.json').write_text(json.dumps(config), encoding='utf-8')
            instance = object.__new__(Clicker)
            instance.root = folder
            instance.config = dict(config)
            base = {'id': 'x', 'name': 'x', 'asset': 'assets/main.png',
                    'source_crop': [0, 0, 10, 10], 'search_area': [0, 0, 20, 20],
                    'click_source_area': [0, 0, 10, 10], 'threshold': .9,
                    'interval': 1, 'stable_frames': 1}
            instance.load_rule({**base, 'skip_on_timeout': True})
            for bad in (1, 'yes', None):
                with self.assertRaises(ValueError):
                    instance.load_rule({**base, 'skip_on_timeout': bad})


if __name__ == '__main__':
    unittest.main()
