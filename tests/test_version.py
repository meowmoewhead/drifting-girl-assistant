"""Version propagation and a device-free missing-config panel."""
from contextlib import contextmanager, redirect_stdout
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import runpy
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from drift.core import paths
from drift.web import cli, server


@contextmanager
def panel(app):
    service = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
    thread = threading.Thread(target=service.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", service.server_port, timeout=5)
    try:
        yield connection
    finally:
        connection.close()
        service.shutdown()
        service.server_close()
        thread.join(timeout=5)


def request(connection, method, path, body=None, headers=None):
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    return response.status, response.read()


class VersionTests(unittest.TestCase):
    def test_version_comes_from_root_file(self):
        self.assertEqual(paths.VERSION, (paths.ROOT / "VERSION").read_text(encoding="utf-8").strip())
        self.assertEqual(cli.VERSION, paths.VERSION)
        self.assertEqual(server.VERSION, paths.VERSION)

    def test_missing_version_falls_back_to_dev(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(paths.read_version(Path(directory)), "0.0.0+dev")

    def test_unreadable_empty_or_invalid_encoding_version_falls_back(self):
        for error in (PermissionError("unreadable"), UnicodeError("invalid UTF-8")):
            with self.subTest(error=error), patch.object(Path, "read_text", side_effect=error):
                self.assertEqual(paths.read_version(), "0.0.0+dev")
        with patch.object(Path, "read_text", return_value="\n "):
            self.assertEqual(paths.read_version(), "0.0.0+dev")

    def test_frozen_version_uses_executable_directory_not_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "VERSION").write_text("2.3.4\n", encoding="utf-8")
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "VERSION").write_text("9.9.9\n", encoding="utf-8")
            with patch.object(sys, "frozen", True, create=True), \
                    patch.object(sys, "executable", str(root / "mumu-panel.exe")), \
                    patch.object(sys, "_MEIPASS", str(bundle), create=True):
                values = runpy.run_path(str(Path(paths.__file__).resolve()))
            self.assertEqual(values["ROOT"], root)
            self.assertEqual(values["BUNDLE"], bundle)
            self.assertEqual(values["VERSION"], "2.3.4")

    def test_cli_version_exits_before_runtime_initialization(self):
        with patch.object(sys, "argv", ["mumu-panel.exe", "--version"]), \
                patch.object(cli, "prefer_system_msvc_runtime") as initialize, \
                patch.dict(sys.modules, {"drift.runtime": None}), \
                redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as exit_status:
                cli.main()
        self.assertEqual(exit_status.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), f"mumu-clicker {paths.VERSION} (upstream {paths.UPSTREAM[:12]})")
        initialize.assert_not_called()

    def test_api_status_adds_version_and_preserves_existing_fields(self):
        original = {"mode": "idle", "running": False, "rules": [{"id": "fixture"}],
                    "features": [], "last_error": "existing message", "custom_field": {"keep": 7}}
        app = SimpleNamespace(status=lambda: original)
        with panel(app) as connection:
            code, data = request(connection, "GET", "/api/status")
        self.assertEqual(code, 200)
        payload = json.loads(data)
        self.assertEqual(payload, {**original, "version": paths.VERSION, "upstream": paths.UPSTREAM,
                                   "task_groups": json.loads(json.dumps(server.TASK_GROUPS))})
        self.assertNotIn("version", original)
        self.assertNotIn("task_groups", original)


class MissingConfigTests(unittest.TestCase):
    def test_cli_serves_idle_with_warning_and_version_without_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            service = Mock(server_address=("127.0.0.1", 18765))
            service.serve_forever.side_effect = KeyboardInterrupt
            handler = Mock(return_value=object())
            with patch.object(sys, "argv", ["mumu-panel.exe", "serve", "--root", directory, "--port", "18765"]), \
                    patch.object(cli, "prefer_system_msvc_runtime"), \
                    patch.object(cli, "ThreadingHTTPServer", return_value=service) as constructor, \
                    patch.object(cli, "make_handler", handler), \
                    patch.dict(sys.modules, {"drift.runtime": None}), \
                    redirect_stdout(io.StringIO()) as output:
                cli.main()
            app = handler.call_args.args[0]
            self.assertIsInstance(app, server.MissingConfigApp)
            self.assertEqual(app.status()["mode"], "idle")
            self.assertTrue(app.status()["config_missing"])
            constructor.assert_called_once_with(("127.0.0.1", 18765), handler.return_value)
            service.server_close.assert_called_once()
            self.assertIn(server.CONFIG_MISSING_MESSAGE, output.getvalue())
            self.assertIn(f"mumu-clicker {paths.VERSION}", output.getvalue())
            self.assertFalse((Path(directory) / "config.json").exists())

    def test_missing_config_panel_loads_and_reports_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with panel(server.MissingConfigApp(Path(directory))) as connection:
                code, data = request(connection, "GET", "/")
                self.assertEqual(code, 200)
                html = data.decode("utf-8")
                self.assertIn('id="app-version"', html)
                self.assertNotIn("__TOKEN__", html)
                code, data = request(connection, "GET", "/api/status")
                payload = json.loads(data)
                self.assertEqual(code, 200)
                self.assertEqual(payload["mode"], "idle")
                self.assertFalse(payload["running"])
                self.assertTrue(payload["config_missing"])
                self.assertEqual(payload["version"], paths.VERSION)
                self.assertEqual(payload["last_error"], server.CONFIG_MISSING_MESSAGE)
                self.assertEqual(payload["rules"], [])
                self.assertEqual(payload["features"], [])
                code, script = request(connection, "GET", "/static/app.js")
                self.assertEqual(code, 200)
                self.assertIn(b"notice(state.last_error)", script)
                self.assertIn(b"state.version", script)
                self.assertIn(b"state.upstream", script)
            self.assertFalse((Path(directory) / "config.json").exists())

    def test_missing_config_rejects_capture_and_actions_without_device(self):
        with tempfile.TemporaryDirectory() as directory:
            with panel(server.MissingConfigApp(Path(directory))) as connection:
                _, html = request(connection, "GET", "/")
                token = re.search(r"const TOKEN='([^']+)'", html.decode("utf-8")).group(1)
                for path in ("/api/capture", "/api/frame/example.png"):
                    code, data = request(connection, "GET", path)
                    self.assertEqual(code, 400)
                    self.assertEqual(json.loads(data)["error"], server.CONFIG_MISSING_MESSAGE)
                for path in ("/api/start", "/api/settings", "/api/rules", "/api/stop"):
                    code, data = request(connection, "POST", path, "{}", {"X-Session-Token": token})
                    self.assertEqual(code, 400)
                    self.assertEqual(json.loads(data)["error"], server.CONFIG_MISSING_MESSAGE)
                code, _ = request(connection, "POST", "/api/start", "{}")
                self.assertEqual(code, 403)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
