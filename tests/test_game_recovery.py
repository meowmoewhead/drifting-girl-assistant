"""Replay the real exit, guarded login, and the production worker restart path."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import Clicker
from digit_ocr import NumberReading
from game_recovery import GameRecovery, PACKAGE, COMPONENT, game_state, restartable
from parallel_world import ParallelWorld

ROOT = Path(__file__).resolve().parents[1]


def image(path):
    return np.asarray(Image.open(ROOT/path).convert('RGB'))


def dead_state():
    return {'foreground':'com.github.metacubex.clash.meta/com.github.kr328.clash.ProxyActivity',
            'running':False, 'exit':{'timestamp':time.time()-10, 'reason':2, 'status':9}}


class Device:
    def __init__(self):
        self.stop, self.io_lock = threading.Event(), threading.RLock()
        self.state = dead_state()
        self.actions, self.inspections = [], 0
        self.screen = 'clash'
        self.images = {'clash':image('tests/fixtures/recovery/clash.png'),
                       'login':image('assets/recovery/login.png'),
                       'main':image('tests/fixtures/parallel/parallel-rank3-check.png')}
        self.on_inspect = lambda: None

    def game_state(self):
        self.inspections += 1
        self.on_inspect()
        return deepcopy(self.state)

    def command(self, *args):
        assert args == ('shell', 'am', 'start', '-n', COMPONENT), args
        self.actions.append(('launch', COMPONENT))
        self.screen = 'login'
        self.state.update(foreground=COMPONENT, running=True)
        return b'Starting: Intent { cmp=' + COMPONENT.encode() + b' }'

    def screenshot(self):
        return self.images[self.screen].copy()

    def click(self, button):
        box = button.button
        point = ((box[0]+box[2])//2, (box[1]+box[3])//2)
        assert self.screen == 'login' and point == (360, 914), (self.screen, point)
        self.actions.append(('login', point))
        self.screen = 'main'


class GameRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.d = Device()
        self.events = []
        self.r = GameRecovery(ROOT, self.d, self.events.append)

    def launch(self):
        for _ in range(3):
            self.assertEqual(self.r.step(self.d.images['clash'], False, False), 'wait')

    def complete(self):
        for _ in range(3):
            self.r.step(self.d.images['login'], False, False)
        for _ in range(3):
            result = self.r.step(self.d.images['main'], True, True)
        self.assertEqual(result, 'ready')

    def test_real_android_exit_record_and_live_process_detection(self):
        native = json.loads((ROOT/'tests/fixtures/recovery/exit-record.json').read_text(encoding='utf-8-sig'))
        s = game_state('\n'.join(native['activity']), 'NAME\ninit\nclash', '\n'.join(native['exit_info']))
        self.assertEqual(s['foreground'], self.d.state['foreground'])
        self.assertEqual((s['exit']['reason'], s['exit']['status']), (2, 9))
        with patch('game_recovery.time.time', return_value=s['exit']['timestamp']+30):
            self.assertTrue(restartable(s))
        live = game_state('\n'.join(native['activity']), 'NAME\n'+PACKAGE, '\n'.join(native['exit_info']))
        self.assertTrue(live['running'])
        self.assertTrue(restartable(live))
        self.assertFalse(restartable(dict(live,foreground=COMPONENT)))

    def test_activity_formats_use_explicit_resumed_fields_and_prefer_top(self):
        for label in ('topResumedActivity=', 'mResumedActivity: ', 'ResumedActivity: '):
            state=game_state(label+'ActivityRecord{abc u0 '+COMPONENT+' t1}',PACKAGE,'')
            self.assertEqual(state['foreground'],COMPONENT)
        state=game_state('ResumedActivity: ActivityRecord{abc u0 com.android.settings/.Settings t1}\n'
            'topResumedActivity=ActivityRecord{def u0 '+COMPONENT+' t2}',PACKAGE,'')
        self.assertEqual(state['foreground'],COMPONENT)

    def test_device_recovery_reads_do_not_require_android_exit_history(self):
        from drift.device.adb import AdbDevice
        device=AdbDevice({})
        replies={('shell','dumpsys','activity','activities'):
                 b'topResumedActivity=ActivityRecord{abc u0 com.android.vending/.AssetBrowserActivity t1}',
                 ('shell','ps','-A','-o','NAME'):b'NAME\ninit\ncom.android.vending'}
        with patch.object(device,'command',side_effect=lambda *args:replies[args]) as command:
            state=device.game_state()
        self.assertEqual(command.call_count,2)
        self.assertEqual(state,dict(foreground='com.android.vending/.AssetBrowserActivity',running=False,exit=None))
        self.assertTrue(restartable(state))

    def test_real_network_exit_message_is_observed_without_pressing_exit(self):
        network=image('assets/recovery/network-error.png')
        self.assertTrue(self.r._network_error(network))
        self.assertFalse(self.r._network_error(network//2))
        self.assertFalse(self.r._network_error(self.d.images['main']))
        damaged=network.copy();damaged[541:574,194:547]=0
        self.assertFalse(self.r._network_error(damaged))
        self.d.state.update(foreground=COMPONENT,running=True)
        for _ in range(3):self.assertEqual(self.r.step(network,True,False),'wait')
        began=self.r.started_at
        self.assertEqual(self.d.actions,[])
        self.assertEqual(self.r.phase,'loading')
        self.d.state.update(foreground='app.lawnchair/.LawnchairLauncher',running=False)
        for _ in range(3):self.r.step(self.d.images['clash'],False,False)
        self.assertEqual(self.d.actions,[('launch',COMPONENT)])
        self.assertEqual(self.r.started_at,began)
        self.complete()
        self.assertEqual(self.r.recoveries,1)

    def test_unknown_foreground_transition_waits_but_cannot_renew_the_deadline(self):
        self.d.state.update(foreground=None,running=True)
        for _ in range(3):self.r.step(self.d.images['clash'],False,False)
        began=self.r.started_at
        self.assertIsNotNone(began)
        self.assertEqual(self.d.actions,[])
        for _ in range(3):self.r.step(self.d.images['clash'],False,False)
        self.assertEqual(self.r.started_at,began)
        with patch('game_recovery.time.monotonic',return_value=began+120.1):
            with self.assertRaisesRegex(RuntimeError,'120 秒'):
                self.r.step(self.d.images['clash'],False,False)
        self.assertEqual(self.d.actions,[])

    def test_latest_exit_is_kept_but_no_longer_gates_relaunch(self):
        native = json.loads((ROOT/'tests/fixtures/recovery/exit-record.json').read_text(encoding='utf-8-sig'))
        raw = '\n'.join(native['exit_info']).replace('reason=2 (SIGNALED)', 'reason=10 (USER REQUESTED)', 1)
        s = game_state('\n'.join(native['activity']), 'NAME\ninit', raw)
        self.assertEqual(s['exit']['reason'], 10)
        self.assertTrue(restartable(s))

    def test_any_foreground_and_exit_record_allow_relaunch(self):
        cases = [dict(self.d.state, running=True), dict(self.d.state, foreground='com.android.settings/.Settings'),
                 dict(self.d.state, foreground=None), dict(self.d.state, exit=None),
                 dict(self.d.state, foreground='com.android.vending/.AssetBrowserActivity')]
        for record in ({'timestamp':time.time()-1900, 'reason':2, 'status':9},
                       {'timestamp':time.time()+60, 'reason':2, 'status':9},
                       {'timestamp':time.time(), 'reason':10, 'status':0},
                       {'timestamp':time.time(), 'reason':0, 'status':9}):
            cases.append(dict(self.d.state, exit=record))
        for state in cases:
            with self.subTest(state=state):
                self.assertTrue(restartable(state))
        for state in (dict(self.d.state,running=True,foreground=COMPONENT),
                      dict(self.d.state,running=True,foreground=None),
                      dict(self.d.state,running=None), {}, {'running':0}):
            with self.subTest(state=state):
                self.assertFalse(restartable(state))

    def test_login_requires_real_title_text_and_normal_brightness(self):
        self.assertTrue(self.r._login(self.d.images['login']))
        for name in ('clash', 'main'):
            self.assertFalse(self.r._login(self.d.images[name]))
        self.assertFalse(self.r._login((self.d.images['login']*.4).astype(np.uint8)))
        self.assertFalse(self.r._login(np.zeros((800, 720, 3), dtype=np.uint8)))
        missing = self.d.images['login'].copy()
        missing[903:926, 270:451] = 0
        self.assertFalse(self.r._login(missing))

    def test_launcher_relaunches_closed_game_without_requiring_crash_record(self):
        for record in (None, {'timestamp':time.time()-5000,'reason':10,'status':0}):
            state=dict(self.d.state,foreground='app.lawnchair/.LawnchairLauncher',exit=record)
            self.assertTrue(restartable(state))
            self.assertTrue(restartable(dict(state,running=True)))
            self.assertTrue(restartable(dict(state,foreground='com.android.settings/.Settings')))
        self.d.state=state
        self.launch()
        self.complete()
        self.assertEqual(self.d.actions,[('launch',COMPONENT),('login',(360,914))])

    def test_existing_start_page_is_adopted_and_enters_main_without_relaunch(self):
        self.d.screen='login'
        self.d.state.update(foreground=COMPONENT,running=True)
        for _ in range(3):
            self.r.step(self.d.images['login'],False,False)
        self.assertEqual(self.r.phase,'loading')
        self.assertEqual(self.d.actions,[])
        self.complete()
        self.assertEqual(self.d.actions,[('login',(360,914))])
        self.assertEqual(self.r.recoveries,1)

    def test_injected_capture_enters_main_and_rechecks_changed_title(self):
        captures = []
        def capture():
            captures.append(self.d.screen)
            return self.d.images[self.d.screen].copy()
        self.r = GameRecovery(ROOT,self.d,self.events.append,capture=capture)
        self.d.state.update(foreground=COMPONENT,running=True)
        self.d.screen='main'
        for _ in range(3):self.r.step(self.d.images['login'],False,False)
        self.assertEqual(self.r.phase,'idle')
        self.assertEqual(self.d.actions,[])
        self.d.screen='login'
        self.r.step(self.d.images['login'],False,False)
        self.complete()
        self.assertEqual(self.d.actions,[('login',(360,914))])
        self.assertEqual(self.r.recoveries,1)
        self.assertEqual(captures,['main','login','login'])

    def test_existing_start_page_fresh_change_or_other_foreground_is_not_adopted(self):
        self.d.state.update(foreground=COMPONENT,running=True)
        self.d.screen='main'
        for _ in range(3):
            self.r.step(self.d.images['login'],False,False)
        self.assertEqual(self.r.phase,'idle')
        self.assertEqual(self.d.actions,[])
        self.d.screen='login'
        self.d.state['foreground']='com.android.settings/.Settings'
        self.r.step(self.d.images['login'],False,False)
        self.assertEqual(self.r.phase,'idle')
        self.assertEqual(self.d.actions,[])

    def test_exit_during_loading_retries_without_extending_deadline(self):
        self.launch()
        original=self.r.started_at
        self.d.state=dict(dead_state(),foreground='app.lawnchair/.LawnchairLauncher',exit=None)
        self.d.screen='clash'
        for _ in range(3):
            self.r.step(self.d.images['clash'],False,False)
        self.assertEqual(self.r.started_at,original)
        self.assertEqual(len(self.d.actions),2)
        self.complete()
        self.d.state=dict(dead_state(),foreground='app.lawnchair/.LawnchairLauncher',exit=None)
        for _ in range(2):
            self.r.step(self.d.images['clash'],False,False)
        with self.assertRaisesRegex(RuntimeError,'30 分钟内已恢复两次'):
            self.r.step(self.d.images['clash'],False,False)

    def test_loading_reopens_from_another_app_but_never_after_stop(self):
        self.launch()
        self.d.state.update(foreground='com.android.settings/.Settings',running=False)
        for _ in range(2):
            self.r.step(self.d.images['clash'],False,False)
        self.r.step(self.d.images['clash'],False,False)
        self.assertEqual(self.d.actions,[('launch',COMPONENT)]*2)
        self.d.stop.set()
        with self.assertRaises(InterruptedError):
            self.r.step(self.d.images['login'],False,False)

    def test_start_page_title_confirmation_resets_on_main(self):
        self.d.screen='login'
        self.d.state.update(foreground=COMPONENT,running=True)
        for _ in range(2):
            self.r.step(self.d.images['login'],False,False)
        self.r.step(self.d.images['main'],True,True)
        self.r.step(self.d.images['login'],False,False)
        self.assertEqual(self.r.phase,'idle')
        self.assertEqual(self.d.actions,[])

    def test_restart_waits_for_three_frames_and_rechecks_before_launch(self):
        for _ in range(2):
            self.r.step(self.d.images['clash'], False, False)
        self.assertEqual(self.d.actions, [])
        self.r.step(self.d.images['clash'], False, False)
        self.assertEqual(self.d.inspections, 2)
        self.assertEqual(self.d.actions, [('launch', COMPONENT)])
        self.assertEqual(self.r.phase, 'loading')

    def test_changed_foreground_cancels_launch(self):
        def change():
            if self.d.inspections == 2:
                self.d.state['foreground'] = 'com.android.settings/.Settings'
        self.d.on_inspect = change
        self.launch()
        self.assertEqual(self.d.actions, [])
        self.assertEqual(self.r.phase, 'idle')

    def test_manual_stop_before_or_during_inspection_sends_no_command(self):
        self.d.stop.set()
        with self.assertRaises(InterruptedError):
            self.r.step(self.d.images['clash'], False, False)
        self.assertEqual(self.d.actions, [])
        self.d.stop.clear()
        self.d.on_inspect = self.d.stop.set
        for _ in range(2):
            self.r.step(self.d.images['clash'], False, False)
        with self.assertRaises(InterruptedError):
            self.r.step(self.d.images['clash'], False, False)
        self.assertEqual(self.d.actions, [])

    def test_unknown_process_state_preserves_the_foreground_app(self):
        self.d.state['running'] = None
        for _ in range(2):
            self.r.step(self.d.images['clash'], False, False)
        with self.assertRaisesRegex(RuntimeError, '保留当前应用'):
            self.r.step(self.d.images['clash'], False, False)
        self.assertEqual(self.d.actions, [])

    def test_login_is_clicked_once_then_three_main_frames_complete_recovery(self):
        self.launch()
        self.complete()
        self.assertEqual(self.d.actions, [('launch', COMPONENT), ('login', (360, 914))])
        self.assertEqual(self.r.recoveries, 1)
        self.assertEqual(self.r.phase, 'idle')
        self.assertIsNone(self.r.step(self.d.images['main'], True, True))

    def test_foreground_or_fresh_login_change_prevents_touch(self):
        self.launch()
        self.d.screen = 'main'
        for _ in range(3):
            self.r.step(self.d.images['login'], False, False)
        self.assertFalse(self.r.login_clicked)
        self.assertEqual(self.d.actions, [('launch', COMPONENT)])
        self.d.screen = 'login'
        self.d.state['foreground'] = 'com.android.settings/.Settings'
        for _ in range(3):
            self.r.step(self.d.images['login'], False, False)
        self.assertEqual(self.d.actions, [('launch', COMPONENT)])

    def test_unknown_loading_times_out_without_repeat_launch_or_touch(self):
        self.launch()
        with patch('game_recovery.time.monotonic', return_value=self.r.started_at+120.1):
            with self.assertRaisesRegex(RuntimeError, '120 秒'):
                self.r.step(np.zeros_like(self.d.images['clash']), False, False)
        self.assertEqual(self.d.actions, [('launch', COMPONENT)])

    def test_two_restarts_limit_persists_after_successful_recovery(self):
        for _ in range(2):
            self.d.state, self.d.screen = dead_state(), 'clash'
            self.launch()
            self.complete()
        self.d.state, self.d.screen = dead_state(), 'clash'
        for _ in range(2):
            self.r.step(self.d.images['clash'], False, False)
        with self.assertRaisesRegex(RuntimeError, '30 分钟内已恢复两次'):
            self.r.step(self.d.images['clash'], False, False)
        self.assertEqual(len([a for a in self.d.actions if a[0]=='launch']), 2)

    def worker(self, preview=False, max_clicks=None, stone_only=False, expired_stone=False, equipment_only=False,
               locked_capture=False, existing_start=False, store_start=False):
        outer, worlds = self, []
        class ReplayDevice(Device):
            def __init__(self, config, stop):
                super().__init__()
                self.stop = stop
                self.shots = self.main_shots = 0
                self.image = None
                if store_start:
                    self.state.update(foreground='com.android.vending/.AssetBrowserActivity',exit=None)
                if existing_start:
                    self.screen = 'login'
                    self.state.update(foreground=COMPONENT, running=True)
            def screenshot(self):
                # Real AdbDevice captures acquire this same RLock. A bounded
                # acquire exposes cross-thread dispatch without hanging tests.
                acquired = not locked_capture or self.io_lock.acquire(timeout=.5)
                if not acquired:
                    raise RuntimeError('Recovery capture waited for another thread holding device I/O')
                try:
                    self.shots += 1
                    if self.screen == 'main':
                        self.main_shots += 1
                    if self.main_shots >= 4 or self.shots >= 20:
                        self.stop.set()
                    self.image = super().screenshot()
                    return self.image
                finally:
                    if locked_capture:self.io_lock.release()
            def stuck_record_add(self, _):
                pass
        class Reader:
            def read(self, _):
                return NumberReading('37', 37, .999)
        def world(*args, **kwargs):
            p = ParallelWorld(*args, **kwargs)
            worlds.append(p)
            return p
        with tempfile.TemporaryDirectory(prefix='game-recovery-replay-') as folder:
            root = Path(folder)
            shutil.copytree(ROOT/'assets', root/'assets')
            config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            entry = ('driftinggirl_equipment_upgrade' if equipment_only else
                     'driftinggirl_stone_farm' if stone_only else 'driftinggirl_parallel_world')
            config['rules'] = [r for r in config['rules'] if r['id'] == entry]
            for rule in config['rules']:
                rule['enabled'] = True
            config['poll_interval'] = .1
            (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
            state = {'version':1, 'mode':'farming', 'rank':37, 'start_level':1323,
                     'boss_level':1421, 'started_at':time.time()-10}
            (root/'training_state.json').write_text(json.dumps(state), encoding='utf-8')
            failed = None
            if expired_stone:
                failed = dict(version=1,pending=True,phase='route_running',reason='lower',
                    rank=39,from_stage=9,to_stage=8,entry_rank=39,
                    deadline=time.time()-1,phase_deadline=time.time()-1,
                    return_deadline=time.time()-1,attempts={'route-start':1})
                (root/'stone_farm_state.json').write_text(json.dumps(failed),encoding='utf-8')
            app = Clicker(root, ReplayDevice)
            app.digit_reader = Reader()
            try:
                with patch('app.ParallelWorld', side_effect=world):
                    app.start(preview=preview, max_clicks=max_clicks)
                    app.thread.join(8)
                outer.assertFalse(app.running())
                if expired_stone:
                    outer.assertIsNone(app.state['last_error'])
                    current=json.loads((root/'stone_farm_state.json').read_text())
                    outer.assertFalse(current['pending'])
                    outer.assertEqual(current['attempts'],failed['attempts'])
                    for key in ('deadline','phase_deadline','return_deadline'):
                        outer.assertEqual(current[key],failed[key])
                    archived=root/current['retirement']['archive']
                    outer.assertEqual(archived.read_text(),json.dumps(failed))
                else:
                    outer.assertIsNone(app.state['last_error'])
                outer.assertEqual(json.loads((root/'training_state.json').read_text()), state)
                return deepcopy(app.state), list(app.device.actions), worlds, app.device.inspections
            finally:
                app.stop()
                if app.thread:
                    app.thread.join(3)
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_worker_preempts_external_screen_and_rebuilds_controllers_with_saved_training(self):
        state, actions, worlds, _ = self.worker()
        self.assertEqual(actions, [('launch', COMPONENT), ('login', (360, 914))])
        self.assertEqual(state['recoveries'], 1)
        self.assertEqual(state['clicks'], 1)
        self.assertEqual(len(worlds), 2)
        self.assertEqual(worlds[-1].training.state['start_level'], 1323)

    def test_worker_relaunches_from_play_store_without_exit_record(self):
        state, actions, worlds, _ = self.worker(store_start=True, locked_capture=True)
        self.assertEqual(actions, [('launch', COMPONENT), ('login', (360, 914))])
        self.assertEqual(state['recoveries'], 1)
        self.assertEqual(state['clicks'], 1)
        self.assertEqual(worlds[-1].training.state['start_level'], 1323)

    def test_real_maa_login_guard_keeps_capture_on_device_lock_owner_thread(self):
        state, actions, worlds, _ = self.worker(locked_capture=True)
        self.assertEqual(actions, [('launch', COMPONENT), ('login', (360, 914))])
        self.assertEqual(state['recoveries'], 1)
        self.assertEqual(state['clicks'], 1)
        self.assertEqual(len(worlds), 2)

    def test_real_maa_adopts_existing_start_page_without_relaunch_or_lock_wait(self):
        state, actions, worlds, _ = self.worker(locked_capture=True, existing_start=True)
        self.assertEqual(actions, [('login', (360, 914))])
        self.assertEqual(state['recoveries'], 1)
        self.assertEqual(state['clicks'], 1)
        self.assertEqual(len(worlds), 2)

    def test_preview_never_queries_processes_or_launches_or_touches_login(self):
        state, actions, worlds, inspections = self.worker(preview=True)
        self.assertEqual((actions, worlds, inspections), ([], [], 0))
        self.assertEqual(state['recoveries'], 0)

    def test_worker_counts_login_against_the_scoped_click_limit(self):
        state, actions, _, _ = self.worker(max_clicks=1)
        self.assertEqual(state['stop_reason'], 'click_limit')
        self.assertEqual(actions, [('launch', COMPONENT), ('login', (360, 914))])

    def test_native_stone_worker_recovers_with_legacy_parallel_task_disabled(self):
        state, actions, worlds, inspections = self.worker(stone_only=True)
        self.assertEqual(actions, [('launch', COMPONENT), ('login', (360, 914))])
        self.assertEqual(state['recoveries'], 1)
        self.assertEqual(state['clicks'], 1)
        self.assertGreater(inspections, 0)
        self.assertEqual(worlds, [])
        self.assertIsNone(state['parallel'])

    def test_native_stone_preview_cannot_launch_or_inspect_game_processes(self):
        state, actions, worlds, inspections = self.worker(preview=True, stone_only=True)
        self.assertEqual((actions, worlds, inspections), ([], [], 0))
        self.assertEqual(state['recoveries'], 0)

    def test_equipment_only_worker_still_relaunches_and_resumes_saved_work(self):
        state,actions,worlds,inspections=self.worker(equipment_only=True)
        self.assertEqual(actions,[('launch',COMPONENT),('login',(360,914))])
        self.assertEqual(state['recoveries'],1)
        self.assertGreater(inspections,0)
        self.assertEqual(worlds,[])

    def test_game_recovery_cannot_renew_expired_stone_workflow_or_reset_its_attempts(self):
        state, actions, worlds, inspections = self.worker(stone_only=True,expired_stone=True)
        self.assertEqual(actions, [('launch', COMPONENT), ('login', (360, 914))])
        self.assertEqual(state['recoveries'], 1)
        self.assertEqual(worlds, [])


if __name__ == '__main__':
    unittest.main()
