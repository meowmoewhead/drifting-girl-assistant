"""面板重启后旧页面令牌失效,前端必须自己刷新页面重新取令牌。

这条守卫很便宜:断言前端确实有 403 分支和 reload,且令牌是由服务端注入页面的
(否则 reload 也拿不到新令牌)。
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class PanelTokenReloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_js = (ROOT / 'web' / 'app.js').read_text(encoding='utf-8')
        cls.index = (ROOT / 'web' / 'index.html').read_text(encoding='utf-8')
        cls.server = (ROOT / 'drift' / 'web' / 'server.py').read_text(encoding='utf-8')

    def test_token_is_injected_per_page_load(self):
        self.assertIn('__TOKEN__', self.index)
        self.assertIn("html.replace(\"__TOKEN__\", token)", self.server)

    def test_token_is_regenerated_per_panel_session(self):
        # 每个面板进程一份新令牌:这正是旧页面 POST 全部 403 的原因。
        self.assertIn('token = secrets.token_hex(24)', self.server)

    def test_frontend_reloads_on_stale_token(self):
        self.assertIn('response.status === 403', self.app_js)
        self.assertIn('location.reload()', self.app_js)
        # 刷新前要有可见提示,不能静默什么都不做
        self.assertIn('面板已重启', self.app_js)


if __name__ == '__main__':
    unittest.main()
