"""The watchdog is tested with fake HTTP only; never contact a live panel."""
from importlib import import_module
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError


class RunWatchdogTests(unittest.TestCase):
    def setUp(self):
        # Target the implementation module: patching urlopen must hit the object
        # the code actually calls, which a re-export shim cannot forward.
        self.watchdog = import_module('drift.web.watchdog')
        self.status = dict(running=False, last_error='RuntimeError: stage expired',
                           last_stop_request=None, stop_reason=None)
        self.now = 10000.

    def test_hard_stop_restarts(self):
        self.assertEqual(self.watchdog.should_restart(self.status, self.now, []),
                         (True, 'hard_stop:RuntimeError: stage expired'))

    def test_user_stop_never_restarts(self):
        self.status['last_stop_request'] = {'source': 'user', 'time': 9999}
        self.assertEqual(self.watchdog.should_restart(self.status, self.now, []), (False, 'user_stop'))

    def test_all_intentional_stop_reasons_are_excluded(self):
        for reason in ('cycle_complete', 'observation_complete', 'click_limit', 'game_stuck'):
            self.status['stop_reason'] = reason
            self.assertEqual(self.watchdog.should_restart(self.status, self.now, []),
                             (False, 'stop_reason:' + reason))

    def test_running_missing_running_and_missing_error_never_restart(self):
        for running in (True, None, 0, 'false'):
            self.assertFalse(self.watchdog.should_restart(dict(self.status, running=running), self.now, [])[0])
        for error in ('', None):
            self.assertEqual(self.watchdog.should_restart(dict(self.status, last_error=error), self.now, []),
                             (False, 'no_error'))

    def test_cooldown_and_exact_boundary(self):
        self.assertEqual(self.watchdog.should_restart(self.status, self.now, [self.now - 179.9]),
                         (False, 'cooldown'))
        self.assertTrue(self.watchdog.should_restart(self.status, self.now, [self.now - 180])[0])
        self.assertEqual(self.watchdog.COOLDOWN, 180)

    def test_hourly_budget_and_window_boundary_without_mutating_history(self):
        history = [self.now - age for age in (200, 400, 600, 800, 3599)]
        original = history.copy()
        self.assertEqual(self.watchdog.should_restart(self.status, self.now, history), (False, 'budget_exhausted'))
        self.assertEqual(history, original)
        history[-1] = self.now - 3600
        self.assertTrue(self.watchdog.should_restart(self.status, self.now, history)[0])
        self.assertEqual(self.watchdog.BUDGET, 5)

    def test_unsorted_history_and_clock_rollback_preserve_cooldown(self):
        for history in ([9990, 9000], [self.now + 5]):
            self.assertEqual(self.watchdog.should_restart(self.status, self.now, history), (False, 'cooldown'))

    def invoke(self, folder, *, argv=(), fetcher=None, poster=None, now=None):
        with patch('sys.stdout', new_callable=io.StringIO) as output:
            result = self.watchdog.main(list(argv), log_dir=Path(folder),
                fetcher=fetcher or Mock(return_value=('abc123', self.status)),
                poster=poster if poster is not None else Mock(), clock=lambda: self.now if now is None else now)
        return result, output.getvalue()

    def test_dry_run_prints_and_logs_decision_without_post_or_history(self):
        with tempfile.TemporaryDirectory() as folder:
            poster = Mock()
            result, output = self.invoke(folder, argv=['--dry-run'], poster=poster)
            self.assertEqual(result, 0)
            poster.assert_not_called()
            self.assertIn('决策=重启', output)
            self.assertIn('dry_run', output)
            log = (Path(folder) / 'run-watchdog.log').read_text('utf-8')
            self.assertEqual(len(log.splitlines()), 1)
            self.assertIn('hard_stop:RuntimeError: stage expired', log)
            self.assertFalse((Path(folder) / 'run-watchdog-history.json').exists())

    def test_restart_history_persists_across_invocations(self):
        with tempfile.TemporaryDirectory() as folder:
            poster = Mock()
            self.assertEqual(self.invoke(folder, poster=poster)[0], 0)
            poster.assert_called_once_with('http://127.0.0.1:8765', 'abc123')
            result, output = self.invoke(folder, poster=poster, now=self.now + 60)
            self.assertEqual(result, 0)
            self.assertIn('原因=cooldown', output)
            self.assertEqual(poster.call_count, 1)
            for delta in (300, 600, 900, 1200):
                self.invoke(folder, poster=poster, now=self.now + delta)
            self.assertEqual(poster.call_count, 5)
            _, output = self.invoke(folder, poster=poster, now=self.now + 1500)
            self.assertIn('原因=budget_exhausted', output)
            self.assertEqual(poster.call_count, 5)
            self.assertEqual(len((Path(folder) / 'run-watchdog.log').read_text('utf-8').splitlines()), 7)

    def test_base_url_is_forwarded_and_history_is_per_panel(self):
        with tempfile.TemporaryDirectory() as folder:
            fetcher = Mock(return_value=('abc123', self.status))
            poster = Mock()
            self.invoke(folder, poster=poster)
            self.invoke(folder, argv=['--base-url', 'http://localhost:9876/'], fetcher=fetcher, poster=poster)
            fetcher.assert_called_once_with('http://localhost:9876')
            poster.assert_called_with('http://localhost:9876', 'abc123')
            self.assertEqual(poster.call_count, 2)

    def test_unreachable_panel_returns_two_and_logs_once_without_post(self):
        with tempfile.TemporaryDirectory() as folder:
            poster = Mock()
            result, output = self.invoke(folder, fetcher=Mock(side_effect=URLError('offline')), poster=poster)
            self.assertEqual(result, 2)
            poster.assert_not_called()
            self.assertIn('panel_unavailable', output)
            self.assertEqual(len((Path(folder) / 'run-watchdog.log').read_text('utf-8').splitlines()), 1)

    def test_ambiguous_post_failure_still_consumes_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            poster = Mock(side_effect=URLError('lost response'))
            self.assertEqual(self.invoke(folder, poster=poster)[0], 2)
            result, output = self.invoke(folder, poster=poster, now=self.now + 1)
            self.assertEqual(result, 0)
            self.assertIn('cooldown', output)
            poster.assert_called_once()

    def test_corrupt_history_does_not_silently_reset_the_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'run-watchdog-history.json').write_text('broken', encoding='utf-8')
            poster = Mock()
            result, output = self.invoke(folder, poster=poster)
            self.assertEqual(result, 0)
            self.assertIn('history_unavailable', output)
            poster.assert_not_called()

    def test_overlapping_invocations_do_not_share_a_restart_slot(self):
        with tempfile.TemporaryDirectory() as folder:
            fetcher, poster = Mock(), Mock()
            with self.watchdog.history_lock(Path(folder) / 'run-watchdog.lock'):
                result, output = self.invoke(folder, fetcher=fetcher, poster=poster)
            self.assertEqual(result, 0)
            self.assertIn('history_unavailable', output)
            fetcher.assert_not_called()
            poster.assert_not_called()

    def test_missing_token_and_invalid_status_return_two_without_post(self):
        for responses in ([io.BytesIO(b'<html>unrelated service</html>')],
                          [io.BytesIO(b"TOKEN='abc123'"), io.BytesIO(b'[]')]):
            with tempfile.TemporaryDirectory() as folder, \
                 patch.object(self.watchdog, 'urlopen', side_effect=responses), \
                 patch('sys.stdout', new_callable=io.StringIO):
                poster = Mock()
                result = self.watchdog.main([], log_dir=Path(folder), poster=poster)
                self.assertEqual(result, 2)
                poster.assert_not_called()

    def test_error_newlines_cannot_forge_extra_log_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            self.status['last_error'] = 'RuntimeError:\r\nfirst\nsecond'
            self.invoke(folder, argv=['--dry-run'])
            log = (Path(folder) / 'run-watchdog.log').read_text('utf-8')
            self.assertEqual(len(log.splitlines()), 1)

    def test_urllib_protocol_uses_token_header_and_exact_start_body(self):
        responses = [io.BytesIO(b"<script>const TOKEN='abc123';</script>"),
                     io.BytesIO(json.dumps(self.status).encode('utf-8')),
                     io.BytesIO(b'{"ok":true}')]
        with patch.object(self.watchdog, 'urlopen', side_effect=responses) as opener:
            token, status = self.watchdog.read_status('http://127.0.0.1:8765')
            self.assertEqual((token, status), ('abc123', self.status))
            self.watchdog.start_run('http://127.0.0.1:8765', token)
        requests = [call.args[0] for call in opener.call_args_list]
        self.assertEqual([r.full_url for r in requests],
                         ['http://127.0.0.1:8765/', 'http://127.0.0.1:8765/api/status',
                          'http://127.0.0.1:8765/api/start'])
        self.assertEqual([r.get_method() for r in requests], ['GET', 'GET', 'POST'])
        for request in requests[1:]:
            self.assertEqual(dict((k.lower(), v) for k, v in request.header_items())['x-session-token'], token)
        self.assertEqual(json.loads(requests[-1].data), {'preview': False})


if __name__ == '__main__':
    unittest.main()
