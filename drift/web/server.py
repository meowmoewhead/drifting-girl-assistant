"""控制面板 HTTP 层:``make_handler(app)``。

实现逐字取自原 ``app.py`` 的 ``make_handler``,只把静态资源根从 ``app.BUNDLE``
换成 :data:`drift.core.paths.BUNDLE`(打包 ``--add-data web`` 后指向 ``sys._MEIPASS``,
源码运行指向项目根),因此 ``web/index.html``、``web/style.css``、``web/app.js``
两种形态下都能找到。

``app`` 按鸭子类型使用,只调用 ``status()`` / ``capture()`` / ``frame_png()`` /
``add_rule()`` / ``change_rule()`` / ``change_feature()`` / ``change_task_time()`` /
``change_settings()`` / ``start()`` / ``stop()``。这里**不** import ``Clicker``,
避免 ``drift.web`` 与 ``drift.runtime`` 循环依赖,也便于测试传入假对象。

由 ``app.py`` 转发:``app.make_handler`` 即本模块的 :func:`make_handler`。
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
import json
import secrets
from urllib.parse import urlsplit

from drift.core.paths import BUNDLE, UPSTREAM, VERSION
from drift.tasks.features import TASK_GROUPS

__all__ = ["make_handler"]

CONFIG_MISSING_MESSAGE = "未找到 config.json，请把发行包里的 config.json 放回程序目录"


class MissingConfigApp:
    """配置误删时仅提供面板状态,不初始化运行期、设备或替代配置。"""

    config_missing = True
    thread = None

    def __init__(self, root):
        self.root = root

    def status(self):
        return {
            "mode": "idle", "running": False, "config_missing": True,
            "frames": 0, "matches": 0, "clicks": 0, "recoveries": 0,
            "last_match": None, "last_error": CONFIG_MISSING_MESSAGE,
            "frame_id": None, "last_stop_request": None, "stage": "main",
            "numbers": {}, "parallel": None, "equipment": None,
            "backpack": None, "skills": None, "meter": None, "stone": None,
            "defense": None, "pass_daily": None, "applied_revision": 0,
            "scoped": False, "rules": [], "features": [],
            "settings": {"adb_path": "", "poll_interval": "", "max_consecutive_clicks": ""},
            "settings_pending": False, "serial": "", "resolution": [],
            "events": [{"time": "", "message": CONFIG_MISSING_MESSAGE}],
            "performance": None, "stone_record": None, "meter_record": None,
        }

    def stop(self, **kwargs):
        pass


def make_handler(app):
    """为一次面板会话生成请求处理器类;令牌每次调用独立。"""
    token = secrets.token_hex(24)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def respond(self, payload, content_type="application/json", status=200):
            if not isinstance(payload, bytes):
                payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    html = (BUNDLE / "web" / "index.html").read_text(encoding="utf-8")
                    return self.respond(html.replace("__TOKEN__", token).encode("utf-8"), "text/html; charset=utf-8")
                if path == "/api/status":
                    return self.respond({**app.status(), "version": VERSION, "upstream": UPSTREAM,
                                         "task_groups": TASK_GROUPS})
                if path in ("/static/style.css", "/static/app.js"):
                    content_type = "text/css" if path.endswith(".css") else "text/javascript"
                    return self.respond((BUNDLE / "web" / path.rsplit("/", 1)[-1]).read_bytes(),
                                        content_type + "; charset=utf-8")
                if path == "/api/capture":
                    if getattr(app, "config_missing", False):
                        raise ValueError(CONFIG_MISSING_MESSAGE)
                    return self.respond(app.capture())
                if path.startswith("/api/frame/") and path.endswith(".png"):
                    if getattr(app, "config_missing", False):
                        raise ValueError(CONFIG_MISSING_MESSAGE)
                    return self.respond(app.frame_png(path.rsplit("/", 1)[-1][:-4]), "image/png")
                self.respond({"error": "不存在"}, status=404)
            except Exception as exc:
                self.respond({"error": str(exc)}, status=400)

        def do_POST(self):
            if not secrets.compare_digest(self.headers.get("X-Session-Token", ""), token):
                return self.respond({"error": "请刷新控制面板"}, status=403)
            try:
                if getattr(app, "config_missing", False):
                    raise ValueError(CONFIG_MISSING_MESSAGE)
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 65536:
                    raise ValueError("请求体大小无效")
                data = json.loads(self.rfile.read(length))
                path = urlsplit(self.path).path
                if path == "/api/rules":
                    result = app.add_rule(data)
                elif path == "/api/rules/change":
                    result = app.change_rule(data)
                elif path == "/api/features/change":
                    result = app.change_feature(data)
                elif path == "/api/portal-mode":
                    result = app.change_portal_mode(data)
                elif path == "/api/tasks/schedule":
                    result = app.change_task_time(data)
                elif path == "/api/settings":
                    result = app.change_settings(data)
                elif path == "/api/devices/detect":
                    result = app.list_devices(data)
                elif path == "/api/devices/select":
                    result = app.select_device(data)
                elif path == "/api/start":
                    result = app.start(preview=data.get("preview", True) is not False,
                                       rule_ids=data.get("rule_ids"), stop_on_main=data.get("stop_on_main", False),
                                       max_clicks=data.get("max_clicks"), timeout=data.get("timeout"))
                elif path == "/api/stop":
                    result = app.stop(source='http_api', peer=self.client_address[0])
                elif path == "/api/stone-retry":
                    result = app.retry_stone_farm(data)
                elif path == "/api/stone-clear":
                    result = app.clear_stone_farm(data)
                elif path == "/api/meter-retry":
                    result = app.retry_meter_calibration(data)
                elif path == "/api/meter-clear":
                    result = app.clear_meter_calibration(data)
                else:
                    return self.respond({"error": "不存在"}, status=404)
                self.respond({"ok": True, "result": result})
            except Exception as exc:
                self.respond({"error": str(exc)}, status=400)

    return Handler


# Native desktop adapter; retain the deployed HTTP/task implementation above.
_desktop_base_factory = make_handler
def make_handler(app):
    from drift.desktop.service import extend_handler
    return extend_handler(_desktop_base_factory, app)


