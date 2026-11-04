"""Task notices preempt background clicks, then resume the requested workflow."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import Clicker
from features import RECOVERY_RULE_IDS
from tools.configure_task_popup import update
from parallel_world import ParallelWorld
from digit_ocr import NumberReading


class TaskPopupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.rules = {r['id']: r for r in cls.config['rules']}
        cls.context = SimpleNamespace(root=ROOT, config=cls.config)
        cls.notice = cls.image('task-popup/notice')
        cls.main = cls.image('task-popup/main')

    @staticmethod
    def image(name):
        with Image.open(ROOT/'assets'/(name+'.png')) as source:
            return np.asarray(source.convert('RGB'))

    def matches(self, index, frame):
        rule, button, offset = Clicker.load_rule(self.context, self.rules[RECOVERY_RULE_IDS[index]])
        return button.match(frame, offset=offset, similarity=rule['threshold'])

    def test_both_notices_match_without_depending_on_npc_or_progress(self):
        self.assertTrue(self.matches(0, self.notice))
        self.assertTrue(self.matches(0, self.image('raid/task-popup')))
        changed = self.notice.copy()
        changed[475:672,90:268] = 0  # Portrait.
        changed[490:636,337:600] = 0  # Title and description inside the speech bubble.
        changed[699:730,113:610] = 0  # Variable task text and progress.
        changed[780:820,176:320] = 0  # Reward type and amount.
        self.assertTrue(self.matches(0, changed))
        for area in ([618,406,674,453],[315,465,350,492],[106,785,162,810]):
            missing = self.notice.copy()
            x1,y1,x2,y2 = area
            missing[y1:y2,x1:x2] = 0
            self.assertFalse(self.matches(0, missing))

    def test_other_windows_and_battle_never_trigger_close_or_return(self):
        for name in ('raid/battle','raid/result','raid/waiting','raid/prepare',
                     'mailbox-native','mailbox-loading-native','mana-confirm-native',
                     'ad-confirm-source','guild-shop/potion-confirm','guild-shop/material-confirm',
                     'event-potion/confirm','achievements-list-native','daily-reward/premium-ready'):
            with self.subTest(name=name):
                frame = self.image(name)
                self.assertFalse(self.matches(0, frame))
                self.assertFalse(self.matches(1, frame))
        self.assertFalse(self.matches(0, self.main))
        self.assertFalse(self.matches(1, self.notice))
        self.assertTrue(self.matches(1, self.main))
        self.assertTrue(self.matches(1, self.image('raid/main')))
        dark = (self.main.astype(float)*.6).astype(np.uint8)
        self.assertFalse(self.matches(1, dark))

    def test_interrupt_configuration_requires_verified_close_and_return(self):
        for values in ({'interrupt':'yes'}, {'interrupt':False}, {'stage':'main'},
                       {'action':'observe'}, {'next_stage':''}, {'next_stage':'main'},
                       {'next_stage':'*'}, {'required_source_areas':[]}):
            rule = {**self.rules[RECOVERY_RULE_IDS[0]], **values}
            with self.subTest(values=values), self.assertRaises(ValueError):
                Clicker.load_rule(self.context, rule)

    def test_migration_preserves_existing_tasks_and_toggle(self):
        before = deepcopy(self.config)
        before['poll_interval'] = .9
        for rule in before['rules']:
            if rule['id'] in RECOVERY_RULE_IDS:
                rule['enabled'] = False
                rule['interval'] = 4
        after = update(before)
        self.assertEqual(after, update(after))
        self.assertEqual(after['poll_interval'], .9)
        old = {r['id']:r for r in before['rules']}
        for rule in after['rules']:
            if rule['id'] not in RECOVERY_RULE_IDS:
                self.assertEqual(rule, old[rule['id']])
            else:
                self.assertFalse(rule['enabled'])
        self.assertEqual(after['rules'][0]['interval'], 4)
        returned = next(r for r in after['rules'] if r['id'] == RECOVERY_RULE_IDS[1])
        self.assertEqual(returned['interval'], 4)

    def test_native_farming_return_ignores_changed_mana_badge_but_requires_all_guards(self):
        native = np.asarray(Image.open(ROOT/'tests/fixtures/task-popup/parallel-farming.png').convert('RGB'))
        notice = np.asarray(Image.open(ROOT/'tests/fixtures/task-popup/parallel-notice.png').convert('RGB'))
        self.assertTrue(self.matches(1, native))
        self.assertTrue(self.matches(0, notice))
        self.assertFalse(self.matches(1, notice))
        changed = native.copy()
        changed[1145:1198,15:79] = 0
        self.assertTrue(self.matches(1, changed))
        for area in ([279,884,307,910],[211,1140,266,1189],[542,1140,597,1183]):
            missing = native.copy()
            x1,y1,x2,y2 = area
            missing[y1:y2,x1:x2] = 0
            self.assertFalse(self.matches(1, missing))
        self.assertFalse(self.matches(1, (native*.6).astype(np.uint8)))

    def replay(self, scenario, preview=False, enabled=True, native=False):
        clock = [0]
        config = deepcopy(self.config)
        config['poll_interval'] = .1
        entry = deepcopy(self.rules['driftinggirl_achievements_open'])
        entry.pop('absent_rule', None)  # This synthetic workflow has no crown-menu step.
        entry.update(id='normal-entry', name='normal entry', enabled=True, stable_frames=1, interval=.2,
                     next_stage='normal_finish', stage_timeout=10, next_delay=0)
        finish = deepcopy(entry)
        finish.update(id='normal-finish', name='normal finish', stage='normal_finish',
                      next_stage='main', action='observe')
        popup = [deepcopy(self.rules[key]) for key in RECOVERY_RULE_IDS]
        for rule in popup:
            rule.update(enabled=enabled, interval=.2, next_delay=0)
        # Put normal rules first: priority must not depend on configuration order.
        config['rules'] = [entry, finish] + popup
        notice, main = self.notice, self.main
        if native:
            notice = np.asarray(Image.open(ROOT/'tests/fixtures/task-popup/parallel-notice.png').convert('RGB'))
            main = np.asarray(Image.open(ROOT/'tests/fixtures/task-popup/parallel-farming.png').convert('RGB'))
        class Device:
            def __init__(self, cfg, stop):
                self.stop, self.image, self.taps, self.reads, self.phase = stop, None, [], 0, 0
            def screenshot(self):
                self.reads += 1
                self.phase += 1
                clock[0] += 2
                count = len(self.taps)
                if scenario == 'expired' and count == 1 and self.phase == 1:
                    clock[0] += 12
                visible = (count == 0 if scenario == 'initial' else
                           count == 1 if scenario == 'expired' else
                           True if scenario in ('stuck', 'stop', 'preview') else
                           (count == 0 or self.phase > 3) if scenario == 'repeat' else False)
                if scenario == 'stop' and self.reads == 2:
                    self.stop.set()
                self.image = (notice if visible else main).copy()
                return self.image
            def click(self, button):
                if self.stop.is_set():
                    raise InterruptedError()
                if button.name == RECOVERY_RULE_IDS[0]:
                    assert self.phase >= 3, 'Closed before three stable frames'
                    box = [int(v) for v in button.button]
                    assert 625 <= box[0] < box[2] <= 668 and 410 <= box[1] < box[3] <= 452, box
                elif button.name != 'normal-entry':
                    raise AssertionError('Unexpected click '+button.name)
                self.taps.append(button.name)
                self.phase = 0
            def stuck_record_add(self, button):
                pass
        with tempfile.TemporaryDirectory(prefix='task-popup-replay-') as directory:
            root = Path(directory)
            for path in {p for r in config['rules'] for p in [r['asset'],*r.get('asset_variants',[])]}:
                (root/path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/path, root/path)
            (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
            app = Clicker(root, Device)
            try:
                with patch('app.time', SimpleNamespace(monotonic=lambda:clock[0])):
                    app.start(preview=preview, rule_ids=['normal-entry','normal-finish'],
                              stop_on_main=scenario != 'repeat', max_clicks=8,
                              timeout=12 if preview else 180)
                    app.thread.join(12)
                    self.assertFalse(app.running(), 'Replay did not finish')
                    return deepcopy(app.state), list(app.device.taps)
            finally:
                app.stop()
                if app.thread:
                    app.thread.join(3)
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_popup_blocks_background_clicks_then_finishes_original_scoped_task(self):
        state, taps = self.replay('initial')
        self.assertIsNone(state['last_error'])
        self.assertEqual(taps, [RECOVERY_RULE_IDS[0], 'normal-entry'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')
        self.assertEqual(state['recoveries'], 1)
        self.assertTrue(set(RECOVERY_RULE_IDS) <= set(state['active_rule_ids']))

    def test_popup_recovers_expired_workflow_then_retries_original_entry(self):
        state, taps = self.replay('expired')
        self.assertIsNone(state['last_error'])
        self.assertEqual(taps, ['normal-entry', RECOVERY_RULE_IDS[0], 'normal-entry'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')

    def test_real_parallel_farming_popup_returns_then_resumes_selected_workflow(self):
        state, taps = self.replay('initial', native=True)
        self.assertIsNone(state['last_error'])
        self.assertEqual(taps, [RECOVERY_RULE_IDS[0], 'normal-entry'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')

    def test_failed_close_reclicks_up_to_three_times_then_stops(self):
        # 契约在 2026-10-26 改过:红叉只点一次时,若那一下被弹窗滑入动画吃掉,
        # stage 过滤会让通配的中断规则彻底失效,只能撞 15 秒期限停机
        # (现场:弹窗卡 6 小时、08:00 三个日常全部错过)。现在重按要有
        # stable_frames 证据,三次仍关不掉才停,并且停机文案点名原因。
        state, taps = self.replay('stuck')
        self.assertIn('红叉连续三次未关掉', state['last_error'])
        # 1 次首次关闭 + 3 次重按,每次重按前都要重新凑够 stable_frames。
        self.assertEqual(taps, [RECOVERY_RULE_IDS[0]]*4)

    def test_repeated_popups_have_a_bounded_recovery_limit(self):
        state, taps = self.replay('repeat')
        self.assertIn('五分钟', state['last_error'])
        self.assertEqual(taps, [RECOVERY_RULE_IDS[0]]*3)

    def test_stop_and_preview_never_send_popup_taps(self):
        for scenario, preview in (('stop',False),('preview',True)):
            with self.subTest(scenario=scenario):
                state,taps = self.replay(scenario,preview=preview)
                self.assertIsNone(state['last_error'])
                self.assertEqual(taps, [])

    def test_disabled_popup_feature_is_not_injected_into_scoped_tests(self):
        state, taps = self.replay('clear',enabled=False)
        self.assertIsNone(state['last_error'])
        self.assertEqual(taps, ['normal-entry'])
        self.assertFalse(set(RECOVERY_RULE_IDS) & set(state['active_rule_ids']))


class TaskPopupMapReturnTests(unittest.TestCase):
    @staticmethod
    def world(root):
        reader=SimpleNamespace(read=lambda _:NumberReading('39',39,.999))
        w=ParallelWorld(root,lambda *_:None,lambda *_:None,lambda _:None,reader)
        w.rank=39;w.phase='find_boss';w.map_swipes=15
        return w

    def test_native_map_requires_legend_world_rank_and_bright_close(self):
        w=self.world(ROOT)
        native=np.asarray(Image.open(ROOT/'tests/fixtures/task-popup/parallel-map.png').convert('RGB'))
        self.assertTrue(w.popup_map_return(native))
        for area in ([13,46,39,72],[632,1151,705,1225],[16,1143,398,1234]):
            with self.subTest(area=area):
                changed=native.copy();x,y,X,Y=area;changed[y:Y,x:X]=0
                self.assertFalse(w.popup_map_return(changed))
        self.assertFalse(w.popup_map_return((native*.6).astype(np.uint8)))
        for name in ('parallel-notice','parallel-farming'):
            frame=np.asarray(Image.open(ROOT/f'tests/fixtures/task-popup/{name}.png').convert('RGB'))
            self.assertFalse(w.popup_map_return(frame))
        w.reader=SimpleNamespace(read=lambda _:NumberReading('38',38,.999))
        self.assertFalse(w.popup_map_return(native))
        w.reader=SimpleNamespace(read=lambda _:NumberReading('40',40,.999))
        for _ in range(3):self.assertFalse(w.popup_map_return(native))
        self.assertEqual(w.rank,39)
        self.assertFalse(w.completed_rank_40)
        self.assertEqual(w.rounds,0)
        w.rank=None
        self.assertFalse(w.popup_map_return(native))

    def test_closed_popup_resumes_map_only_after_consecutive_native_confirmations(self):
        for obscured in (False,True):
            with self.subTest(obscured=obscured), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);shutil.copytree(ROOT/'assets',root/'assets')
                config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
                config['rules']=[deepcopy(r) for r in config['rules'] if r['id'] in
                    (*RECOVERY_RULE_IDS,'driftinggirl_parallel_world')]
                for rule in config['rules']:rule.update(enabled=True,next_delay=0)
                config['poll_interval']=.1
                (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
                notice=np.asarray(Image.open(ROOT/'tests/fixtures/task-popup/parallel-notice.png').convert('RGB'))
                native=np.asarray(Image.open(ROOT/'tests/fixtures/task-popup/parallel-map.png').convert('RGB'))
                taps=[];resumed=[];worlds=[]
                class Device:
                    def __init__(self,_,stop):self.stop=stop;self.image=None;self.map_reads=0
                    def screenshot(self):
                        if taps:self.map_reads+=1
                        self.image=(native if taps else notice).copy()
                        if obscured and taps:self.image=(self.image*.6).astype(np.uint8)
                        if self.map_reads>=6:self.stop.set()
                        return self.image
                    def click(self,button):taps.append(button.name)
                    def stuck_record_add(self,_):pass
                def make_world(*_,**__):
                    w=self.world(root)
                    def step(frame):
                        resumed.append((w.phase,w.map_swipes,app.device.map_reads))
                        app.stop_event.set();return True
                    w.step=step;worlds.append(w);return w
                app=Clicker(root)
                app.device=Device(config,app.stop_event)
                app.digit_reader=SimpleNamespace(read=lambda _:NumberReading('39',39,.999))
                try:
                    with patch('app.ParallelWorld',side_effect=make_world):
                        app.start(preview=False);app.thread.join(6)
                    self.assertFalse(app.running())
                    self.assertIsNone(app.state['last_error'])
                    self.assertEqual(taps,[RECOVERY_RULE_IDS[0]])
                    if obscured:self.assertEqual(resumed,[])
                    else:
                        self.assertEqual(resumed,[('find_boss',15,4)])
                        self.assertFalse(worlds[0].completed_rank_40)
                        self.assertEqual(worlds[0].rounds,0)
                finally:
                    app.stop()
                    if app.thread:app.thread.join(2)
                    for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)


if __name__ == '__main__':
    unittest.main()
