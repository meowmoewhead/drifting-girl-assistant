"""Exercise parameter edits through native Tk variables and asynchronous replies."""
from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from drift.desktop.panel import Panel


class DesktopAutosaveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.window = tk.Tk()
        self.window.withdraw()
        self.client = Mock(address='http://127.0.0.1:8765')
        self.client.request.return_value = {'ok': True}
        with patch.object(Panel, 'poll'):
            self.panel = Panel(self.window, Path(self.directory.name), self.client)
        self.feature = {
            'id': 'restaurant_upgrade', 'name': '餐品强化', 'enabled': True,
            'interval': 86400, 'targets': ['干货'], 'steps': [],
            'execution': {'mode': 'interval', 'next_run': '2026-11-19T01:04:19+08:00'},
        }
        self.panel.state = {
            'running': True, 'features': [self.feature],
            'settings': {'poll_interval': .6, 'max_consecutive_clicks': 20},
        }
        self.panel.build_nav()
        self.panel.page = 'settings'
        self.panel.render_page()

    def tearDown(self):
        self.panel.flush_parameters()
        self.wait(lambda: not self.panel.parameter_active and not self.panel.parameter_queue)
        self.panel.parameter_error = None
        self.panel.close()
        self.directory.cleanup()

    def wait(self, predicate, limit=3):
        end = time.monotonic() + limit
        while time.monotonic() < end:
            self.window.update()
            if predicate():
                return
            time.sleep(.01)
        self.fail('The automatic save did not settle')

    def settled(self):
        return (not self.panel.parameter_active and not self.panel.parameter_queue
                and not self.panel.parameter_timers)

    def test_number_edits_debounce_and_only_save_the_latest_value(self):
        variable = self.panel.fields['poll_interval'][0]
        for value in ('1', '1.2', '1.25'):
            variable.set(value)
        self.client.request.assert_not_called()
        self.wait(self.settled)
        self.client.request.assert_called_once_with(
            '/api/settings', {'poll_interval': 1.25, 'max_consecutive_clicks': 20})
        self.assertFalse(self.panel.dirty())

    def test_checkbox_saves_immediately_even_while_an_action_is_busy(self):
        self.panel.page = 'task/restaurant_upgrade'
        self.panel.render_page()
        self.panel.busy = True
        self.panel.fields['category.汤'][0].set(True)
        self.wait(self.settled, limit=.5)
        path, payload = self.client.request.call_args.args
        self.assertEqual(path, '/api/features/change')
        self.assertEqual(payload['targets'], ['干货', '汤'])
        self.assertEqual(payload['interval'], 86400)

    def test_old_reply_cannot_mark_a_newer_edit_saved(self):
        first = threading.Event()
        second = threading.Event()
        release_first = threading.Event()
        release_second = threading.Event()
        self.addCleanup(release_first.set)
        self.addCleanup(release_second.set)
        calls = []

        def request(path, data):
            calls.append(deepcopy(data))
            waiting, release = ((first, release_first) if len(calls) == 1
                                else (second, release_second))
            waiting.set()
            release.wait(2)
            return {'ok': True}

        self.client.request.side_effect = request
        variable = self.panel.fields['poll_interval'][0]
        variable.set('1.1')
        self.panel.flush_parameters()
        self.wait(first.is_set)
        variable.set('1.2')
        self.panel.flush_parameters()
        release_first.set()
        self.wait(second.is_set)
        self.assertTrue(self.panel.dirty())
        self.assertEqual(self.panel.fields['poll_interval'][1], '1.1')
        release_second.set()
        self.wait(self.settled)
        self.assertEqual([data['poll_interval'] for data in calls], [1.1, 1.2])
        self.assertFalse(self.panel.dirty())

    def test_invalid_number_is_kept_until_a_valid_edit_is_saved(self):
        variable = self.panel.fields['max_consecutive_clicks'][0]
        variable.set('')
        self.assertFalse(self.panel.flush_parameters())
        self.client.request.assert_not_called()
        self.assertTrue(self.panel.dirty())
        variable.set('25')
        self.panel.flush_parameters()
        self.wait(self.settled)
        self.assertEqual(self.client.request.call_args.args[1]['max_consecutive_clicks'], 25)
        self.assertFalse(self.panel.dirty())

    def test_navigation_flushes_the_old_form_to_its_original_task(self):
        self.panel.page = 'task/restaurant_upgrade'
        self.panel.render_page()
        self.panel.fields['interval'][0].set('12')
        self.panel.nav.selection_set('settings')
        self.panel.navigate()
        self.assertEqual(self.panel.page, 'settings')
        self.panel.fields['poll_interval'][0].set('1.5')
        self.panel.flush_parameters()
        self.wait(self.settled)
        calls = [call.args for call in self.client.request.call_args_list]
        self.assertEqual(calls[0], ('/api/features/change', {
            'id': 'restaurant_upgrade', 'interval': 43200.0, 'targets': ['干货']}))
        self.assertEqual(calls[1][0], '/api/settings')
        self.assertEqual(calls[1][1]['poll_interval'], 1.5)

    def test_failed_save_keeps_the_edited_value_and_can_be_corrected(self):
        self.client.request.side_effect = ValueError('参数超出范围')
        variable = self.panel.fields['poll_interval'][0]
        variable.set('100')
        self.panel.flush_parameters()
        self.wait(self.settled)
        self.assertIn('自动保存失败', self.panel.notice.get())
        self.assertEqual(variable.get(), '100')
        self.assertTrue(self.panel.dirty())
        self.client.request.side_effect = None
        variable.set('1.5')
        self.panel.flush_parameters()
        self.wait(self.settled)
        self.assertFalse(self.panel.dirty())


if __name__ == '__main__':
    unittest.main()
