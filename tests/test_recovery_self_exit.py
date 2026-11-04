"""Any-foreground recovery retains bounded network and login workflows."""
import unittest
from unittest.mock import patch

from drift.tasks.game_recovery import COMPONENT, GameRecovery, restartable
from tests.test_game_recovery import Device, ROOT, dead_state, image

STORE = 'com.android.vending/.AssetBrowserActivity'
SETTINGS = 'com.android.settings/.Settings'

class RecoverySelfExitTests(unittest.TestCase):
    def setUp(self):
        for name, value in (('time', 1_793_100_000.), ('monotonic', 1000.)):
            patcher = patch('drift.tasks.game_recovery.time.' + name, return_value=value)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)
        self.d = Device()
        self.events = []
        self.r = GameRecovery(ROOT, self.d, self.events.append)
        self.network = image('assets/recovery/network-error.png')

    def wait_for_network_exit(self):
        self.d.state.update(foreground=COMPONENT, running=True)
        for _ in range(3):
            self.assertEqual(self.r.step(self.network, True, False), 'wait')
        self.assertEqual(self.r.phase, 'loading')
        self.assertEqual(self.d.actions, [])
        self.assertIn('仍沿用120秒总期限', self.events[-1])

    def missing_frames(self, count=3):
        for _ in range(count):
            self.assertEqual(self.r.step(self.d.images['clash'], False, False), 'wait')

    def exited_to_store(self, record=None):
        self.d.state = dict(dead_state(), foreground=STORE, exit=record)
        self.d.screen = 'clash'

    def complete(self):
        for _ in range(3):
            self.r.step(self.d.images['login'], False, False)
        for _ in range(3):
            result = self.r.step(self.d.images['main'], True, True)
        self.assertEqual(result, 'ready')
        self.assertEqual(self.r.phase, 'idle')

    def test_transient_process_read_recovers_before_launch(self):
        self.wait_for_network_exit()
        self.d.state['running'] = False
        self.d.on_inspect = lambda: self.d.state.update(running=True) if self.d.inspections == 2 else None
        self.missing_frames()
        self.assertEqual(self.d.inspections, 2)
        self.assertEqual(self.r.missing, 0)
        self.assertEqual(self.d.actions, [])
        self.d.screen = 'login'
        self.complete()
        self.assertEqual(self.d.actions, [('login', (360, 914))])

    def test_network_exit_to_store_relaunches_with_no_exit_record(self):
        self.wait_for_network_exit()
        began = self.r.started_at
        self.exited_to_store()
        self.assertTrue(restartable(self.d.state))
        self.monotonic.return_value = began + 44
        self.missing_frames()
        self.assertEqual(self.d.actions, [('launch', COMPONENT)])
        self.assertEqual(self.r.started_at, began)
        self.complete()
        self.assertEqual(self.d.actions, [('launch', COMPONENT), ('login', (360, 914))])
        self.assertFalse(self.r.awaiting_self_exit)

    def test_idle_and_loading_launch_do_not_require_network_prompt(self):
        self.exited_to_store()
        self.missing_frames()
        began = self.r.started_at
        self.exited_to_store()
        self.monotonic.return_value += 20
        self.missing_frames()
        self.assertEqual(self.d.actions, [('launch', COMPONENT)] * 2)
        self.assertEqual(self.r.started_at, began)

    def test_stale_manual_or_unknown_exit_records_do_not_block(self):
        records = [None, {'timestamp': 0, 'reason': 2, 'status': 9},
                   {'timestamp': self.time.return_value + 1, 'reason': 10, 'status': 0},
                   {'timestamp': self.time.return_value, 'reason': 0, 'status': 0}]
        for record in records:
            with self.subTest(record=record):
                self.d = Device()
                self.r = GameRecovery(ROOT, self.d, self.events.append)
                self.exited_to_store(record)
                self.missing_frames()
                self.assertEqual(self.d.actions, [('launch', COMPONENT)])

    def test_background_game_is_brought_forward_without_force_stop(self):
        self.d.state.update(foreground=SETTINGS, running=True, exit=None)
        self.missing_frames()
        self.assertEqual(self.d.actions, [('launch', COMPONENT)])
        self.assertIn('切回后台游戏', self.events[-1])
        self.complete()

    def test_closed_game_with_unknown_foreground_can_launch(self):
        self.d.state.update(foreground=None, running=False, exit=None)
        self.missing_frames()
        self.assertEqual(self.d.actions, [('launch', COMPONENT)])

    def test_live_game_with_unknown_foreground_only_waits_until_deadline(self):
        self.wait_for_network_exit()
        began = self.r.started_at
        self.d.state.update(foreground=None, running=True)
        self.missing_frames()
        self.assertEqual(self.d.actions, [])
        self.assertEqual(self.r.started_at, began)
        self.monotonic.return_value = began + 120.1
        with self.assertRaisesRegex(RuntimeError, '120 秒'):
            self.r.step(self.d.images['clash'], False, False)
        self.assertEqual(self.d.actions, [])

    def test_context_change_at_final_launch_read_defers_input(self):
        self.wait_for_network_exit()
        self.exited_to_store()
        began = self.r.started_at
        self.d.on_inspect = lambda: self.d.state.update(foreground=SETTINGS) if self.d.inspections == 2 else None
        self.missing_frames()
        self.assertEqual(self.d.inspections, 2)
        self.assertEqual(self.d.actions, [])
        self.assertEqual(self.r.missing, 0)
        self.assertEqual(self.r.started_at, began)
        self.d.on_inspect = lambda: None
        self.missing_frames()
        self.assertEqual(self.d.actions, [('launch', COMPONENT)])

    def test_stop_during_either_state_read_prevents_launch(self):
        for inspection in (1, 2):
            with self.subTest(inspection=inspection):
                self.d = Device()
                self.r = GameRecovery(ROOT, self.d, self.events.append)
                self.exited_to_store()
                self.d.on_inspect = lambda: self.d.stop.set() if self.d.inspections == inspection else None
                self.missing_frames(2)
                with self.assertRaises(InterruptedError):
                    self.r.step(self.d.images['clash'], False, False)
                self.assertEqual(self.d.actions, [])

    def test_adb_failure_cannot_be_treated_as_an_absent_game(self):
        self.exited_to_store()
        self.d.on_inspect = lambda: (_ for _ in ()).throw(RuntimeError('device offline'))
        self.missing_frames(2)
        with self.assertRaisesRegex(RuntimeError, 'device offline'):
            self.r.step(self.d.images['clash'], False, False)
        self.assertEqual(self.d.actions, [])

    def test_slow_final_read_cannot_extend_recovery_deadline(self):
        self.wait_for_network_exit()
        began = self.r.started_at
        self.exited_to_store()
        self.d.on_inspect = lambda: setattr(self.monotonic, 'return_value', began + 120.1) if self.d.inspections == 2 else None
        self.missing_frames(2)
        with self.assertRaisesRegex(RuntimeError, '120 秒'):
            self.r.step(self.d.images['clash'], False, False)
        self.assertEqual(self.r.started_at, began)
        self.assertEqual(self.d.actions, [])

    def test_any_foreground_keeps_two_launch_limit_and_original_deadline(self):
        self.wait_for_network_exit()
        began = self.r.started_at
        for elapsed in (44, 60):
            self.monotonic.return_value = began + elapsed
            self.exited_to_store()
            self.missing_frames()
            self.assertEqual(self.r.started_at, began)
        self.exited_to_store()
        self.missing_frames(2)
        with self.assertRaisesRegex(RuntimeError, '30 分钟内已恢复两次'):
            self.r.step(self.d.images['clash'], False, False)
        self.assertEqual(self.d.actions, [('launch', COMPONENT)] * 2)

if __name__ == '__main__':
    unittest.main()
