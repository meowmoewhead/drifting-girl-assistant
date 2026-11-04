"""The watchdog must ship inside the panel exe without a standalone Python install."""
import importlib
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class WatchdogEntryTests(unittest.TestCase):
    def test_tools_module_reexports_the_packaged_implementation(self):
        # tests keep importing the tools path; the exe imports the drift path
        tools = importlib.import_module('tools.run_watchdog')
        packaged = importlib.import_module('drift.web.watchdog')
        for name in ('main', 'should_restart', 'read_status', 'start_run', 'read_history',
                     'save_history', 'history_lock'):
            self.assertIs(getattr(tools, name), getattr(packaged, name))
        self.assertIs(tools.BASE_URL, packaged.BASE_URL)
        self.assertEqual(tools.NORMAL_STOPS, packaged.NORMAL_STOPS)

    def test_default_log_directory_is_the_program_directory(self):
        # A frozen build must write history next to the exe, not into _MEIPASS.
        packaged = importlib.import_module('drift.web.watchdog')
        self.assertEqual(packaged.default_log_dir(), packaged.ROOT / 'logs')
        self.assertTrue(str(packaged.default_log_dir()).endswith('logs'))

    def test_cli_dispatches_watchdog_before_parsing_panel_arguments(self):
        cli = importlib.import_module('drift.web.cli')
        packaged = importlib.import_module('drift.web.watchdog')
        seen = []

        def fake_main(argv=None):
            seen.append(argv)
            return 0

        with mock.patch.object(packaged, 'main', fake_main), \
                mock.patch.object(sys, 'argv', ['mumu-panel.exe', 'watchdog', '--dry-run']), \
                mock.patch.object(cli, 'prefer_system_msvc_runtime'):
            with self.assertRaises(SystemExit) as raised:
                cli.main()
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(seen, [['--dry-run']])

    def test_new_run_clears_the_previous_stop_request(self):
        # A stale last_stop_request would make the watchdog treat every hard stop
        # as a user stop and never restart the run.
        runtime = importlib.import_module('drift.runtime')
        state = runtime.fresh_run_state(mode='clicking', active_rule_ids=['a'],
                                        revision=7, scoped=False)
        self.assertIsNone(state['last_stop_request'])
        self.assertIsNone(state['last_error'])
        self.assertIsNone(state['stop_reason'])
        self.assertEqual((state['mode'], state['frames'], state['clicks']),
                         ('clicking', 0, 0))
        self.assertEqual((state['active_rule_ids'], state['applied_revision'], state['scoped']),
                         (['a'], 7, False))

    def test_dry_run_never_posts_and_never_consumes_budget(self):
        packaged = importlib.import_module('drift.web.watchdog')
        posted = []
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            code = packaged.main(
                ['--dry-run'], log_dir=folder,
                fetcher=lambda url: ('token', {'running': False, 'last_error': 'boom'}),
                poster=lambda url, token: posted.append(url), clock=lambda: 1000.)
            history = Path(folder) / 'run-watchdog-history.json'
            self.assertEqual(code, 0)
            self.assertEqual(posted, [])
            self.assertFalse(history.exists())
            line = (Path(folder) / 'run-watchdog.log').read_text(encoding='utf-8')
            self.assertIn('dry_run=true', line)
            self.assertIn('决策=重启', line)


if __name__ == '__main__':
    unittest.main()
