"""Replay the four October 5 faults, including deadline-boundary screenshots."""
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
from digit_ocr import DigitReader, NumberReading
from features import FEATURE_BY_ID
from tools.configure_recent_stops import update, NEW_ID, CHANGED_FIELDS


class RecentStopsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.rules = {r['id']: r for r in cls.config['rules']}
        cls.before = json.loads((ROOT/'tests/fixtures/recent-stops/before-rules.json').read_text(encoding='utf-8'))
        cls.context = SimpleNamespace(root=ROOT, config=cls.config)
        cls.reader = DigitReader(ROOT)
        cls.frames = {name: cls.image(path) for name, path in {
            'mana-226': 'tests/fixtures/recent-stops/mana-226.png',
            'mana-220': 'tests/fixtures/recent-stops/mana-220.png',
            'raid-missed': 'tests/fixtures/recent-stops/raid-missed.png',
            'crown-late': 'tests/fixtures/recent-stops/crown-late.png',
            'main': 'assets/navigation/main.png',
            'menu': 'assets/achievements-menu-native.png',
            'list': 'tests/fixtures/legacy-captures/achievements-cycle-result.png',
            'low-mana': 'tests/fixtures/legacy-captures/mana-cycle-003-main.png',
            'mana-confirm': 'assets/mana-confirm-native.png',
            'mana-close': 'tests/fixtures/legacy-captures/mana-cycle-008-mana_close.png',
            **{name: 'assets/raid/'+name+'.png' for name in ('prepare', 'ready', 'battle', 'result', 'return')},
        }.items()}
        moved = cls.frames['raid-missed'].copy()
        moved[710:789, 638:718] = moved[810:889, 638:718]
        moved[810:889, 638:718] = 0
        cls.frames['raid-moved'] = moved

    @staticmethod
    def image(path):
        with Image.open(ROOT/path) as source:
            return np.asarray(source.convert('RGB'))

    def matches(self, rule, frame):
        _, button, offset = Clicker.load_rule(self.context, rule)
        return button.match(frame, offset=offset, similarity=rule['threshold'])

    def test_both_real_stock_failures_pass_without_lowering_confidence(self):
        rule = self.rules['driftinggirl_mana_potion']
        old = next(r for r in self.before if r['id'] == rule['id'])
        self.assertEqual(rule['number_condition']['confidence'], old['number_condition']['confidence'])
        for name, expected in (('mana-226', 226), ('mana-220', 220)):
            frame = self.frames[name]
            self.assertTrue(self.matches(rule, frame))
            x1,y1,x2,y2 = old['number_condition']['source_area']
            self.assertLess(self.reader.read(frame[y1:y2,x1:x2]).confidence, .93)
            x1,y1,x2,y2 = rule['number_condition']['source_area']
            reading = self.reader.read(frame[y1:y2,x1:x2])
            self.assertEqual(reading.value, expected)
            self.assertGreater(reading.confidence, .99)
        for file, expected in (('assets/mana-panel-native.png',160),
                               ('tests/fixtures/legacy-captures/mana-cycle-008-mana_close.png',159)):
            reading = self.reader.read(self.image(file)[802:826,635:690])
            self.assertEqual(reading.value, expected)
            self.assertGreater(reading.confidence, .99)

    def test_raid_recovery_requires_unobstructed_game_and_never_matches_menus(self):
        rule = self.rules[NEW_ID]
        self.assertTrue(self.matches(rule, self.frames['raid-missed']))
        for name in ('prepare','ready','battle','result','mana-220','mana-226','list','mana-confirm'):
            self.assertFalse(self.matches(rule, self.frames[name]), name)
        for path in ('assets/mailbox-native.png','assets/ad-confirm-source.png',
                     'assets/task-popup/notice.png','assets/event-potion/confirm.png'):
            self.assertFalse(self.matches(rule, self.image(path)), path)
        frame = self.frames['raid-missed']
        for box in [rule['source_crop'], *rule['required_source_areas']]:
            blocked = frame.copy();x1,y1,x2,y2 = box;blocked[y1:y2,x1:x2] = 0
            self.assertFalse(self.matches(rule, blocked))
        self.assertFalse(self.matches(rule, (frame*.6).astype(np.uint8)))

    def test_migration_is_idempotent_and_preserves_existing_settings(self):
        before = deepcopy(self.config)
        before['rules'] = [r for r in before['rules'] if r['id'] != NEW_ID]
        for rule in before['rules']:
            rule.update(enabled=False, interval=123)
            if rule['id'] == 'driftinggirl_mana_potion':rule['number_condition']['confidence'] = .97
        after = update(before)
        self.assertEqual(update(after), after)
        indexed = {r['id']:r for r in after['rules']}
        self.assertFalse(indexed[NEW_ID]['enabled'])
        for old in before['rules']:
            new = deepcopy(indexed[old['id']]);old = deepcopy(old)
            if old['id'] == 'driftinggirl_mana_potion':
                self.assertEqual(new['number_condition']['confidence'], .97)
                new['number_condition']['source_area'] = old['number_condition']['source_area']
            for field in CHANGED_FIELDS.get(old['id'], ()):
                if field != 'number_condition':new.pop(field,None);old.pop(field,None)
            self.assertEqual(old, new)

    def test_grace_cannot_enable_clicks_retries_or_unguarded_returns(self):
        rule = self.rules['driftinggirl_achievements_return']
        for change in ({'action':'click'},{'stage':'main'},{'next_stage':'another'},
                       {'stable_frames':1},{'required_source_areas':[]},{'color_condition':None},
                       {'timeout_grace':11},{'recovery_entry':'driftinggirl_achievements_open'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                Clicker.load_rule(self.context, {**rule, **change})

    def replay(self, feature, scenario, legacy=False):
        config = deepcopy(self.config);config['poll_interval'] = .1
        ids = FEATURE_BY_ID[feature]['rules']
        config['rules'] = [deepcopy(r) for r in (self.before if legacy else config['rules']) if r['id'] in ids]
        for rule in config['rules']:rule.update(enabled=True,interval=.2,next_delay=0)
        frames, clock, reader = self.frames, [0], self.reader
        class Device:
            def __init__(self, cfg, stop):
                self.stop,self.image,self.taps,self.phase,self.shots,self.final_frames = stop,None,[],0,0,0
            def screenshot(self):
                self.phase += 1;self.shots += 1;clock[0] += 1
                count = len(self.taps)
                if feature == 'mana':
                    name = ['low-mana',scenario if scenario in ('mana-220','mana-226') else 'mana-220',
                            'mana-confirm','mana-close','main'][min(count,4)]
                elif feature == 'raid':
                    if count == 0:
                        name = 'raid-moved' if scenario == 'move' and self.shots <= 3 else 'raid-missed'
                    elif scenario == 'repeat' or (scenario in ('missed','unknown') and count == 1):
                        name = 'raid-missed'
                    else:
                        step = count-int(scenario == 'missed')
                        name = ['raid-missed','prepare','ready','result','return'][min(step,4)]
                    if scenario == 'unknown' and count == 1:
                        self.image = np.zeros_like(frames['main']);return self.image
                else:
                    name = ['main','menu','list','menu','crown-late'][min(count,4)]
                    if count >= 4:
                        self.final_frames += 1
                        # The first clear screenshot arrives exactly at the old timeout.
                        if self.final_frames == 1:clock[0] += 14
                        if scenario == 'late-slow':clock[0] += 4
                        if scenario == 'late-flicker' and self.final_frames == 2 or scenario == 'unknown':
                            self.image = np.zeros_like(frames['main']);return self.image
                self.image = frames[name].copy();return self.image
            def click(self, button):
                if self.stop.is_set():raise InterruptedError()
                self.taps.append((button.name,tuple(int(v) for v in button.button)))
                self.phase = 0
            def stuck_record_add(self, button):pass
        with tempfile.TemporaryDirectory(prefix='recent-stops-') as directory:
            root = Path(directory)
            paths = {r['asset'] for r in config['rules']} | {p for r in config['rules'] for p in r.get('asset_variants', [])}
            for path in paths:
                (root/path).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,root/path)
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            app = Clicker(root,Device)
            def read_number(crop):
                if feature == 'mana' and len(app.device.taps) == 1:
                    if scenario == 'zero':return NumberReading('0',0,.999)
                    if scenario == 'unclear':return NumberReading('220',220,.5)
                return reader.read(crop)
            app.digit_reader = SimpleNamespace(read=read_number)
            try:
                with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])):
                    app.start(preview=False,stop_on_main=True,max_clicks=10,timeout=240)
                    app.thread.join(18)
                    self.assertFalse(app.running(),'Replay did not complete')
                failures = [json.loads(p.read_text(encoding='utf-8')) for p in (root/'logs').glob('error-*.json')]
                return {**deepcopy(app.state),'events':list(app.events)},app.device.taps,failures
            finally:
                app.stop()
                if app.thread:app.thread.join(3)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

    def test_actual_stock_failures_now_finish_with_exactly_one_potion(self):
        old,taps,diagnostics = self.replay('mana','mana-220',legacy=True)
        self.assertIn('未识别到后续界面',old['last_error']);self.assertEqual(len(taps),1)
        self.assertFalse(diagnostics[0]['numbers']['driftinggirl_mana_potion']['passed'])
        for name in ('mana-220','mana-226'):
            state,taps,_ = self.replay('mana',name)
            self.assertIsNone(state['last_error']);self.assertEqual(state['stop_reason'],'cycle_complete')
            self.assertEqual([t[0] for t in taps],['driftinggirl_mana_'+s for s in ('open','potion','confirm','close')])

    def test_zero_or_unclear_inventory_never_consumes_a_potion(self):
        for scenario in ('zero','unclear'):
            state,taps,_ = self.replay('mana',scenario)
            self.assertIn('未识别到后续界面',state['last_error']);self.assertEqual(len(taps),1)

    def test_missed_raid_entry_recovers_and_finishes_the_actual_raid(self):
        state,taps,_ = self.replay('raid','missed')
        self.assertIsNone(state['last_error']);self.assertEqual(state['stop_reason'],'cycle_complete')
        self.assertEqual(state['recoveries'],1)
        self.assertEqual([t[0] for t in taps],['driftinggirl_raid_'+s for s in ('open','open','prepare','start','confirm')])

    def test_moving_raid_entry_cancels_stale_tap_without_needing_a_counter(self):
        state,taps,_ = self.replay('raid','move')
        self.assertIsNone(state['last_error']);self.assertEqual(len(taps),4)
        self.assertTrue(any('取消本次点击' in e['message'] for e in state['events']))
        x1,y1,x2,y2 = taps[0][1];self.assertEqual(((x1+x2)//2,(y1+y2)//2),(678,844))

    def test_failed_raid_retries_are_bounded_and_unknown_screens_still_stop(self):
        state,taps,_ = self.replay('raid','repeat')
        self.assertIn('五分钟内已恢复三次',state['last_error']);self.assertEqual(len(taps),4)
        state,taps,_ = self.replay('raid','unknown')
        self.assertIn('未识别到后续界面',state['last_error']);self.assertEqual(len(taps),1)
        self.assertEqual(state['recoveries'],0)

    def test_crown_last_frame_gets_bounded_observation_time_and_no_more_clicks(self):
        old,taps,_ = self.replay('achievements','late',legacy=True)
        self.assertIn('未识别到后续界面',old['last_error'])
        state,fixed,_ = self.replay('achievements','late')
        self.assertIsNone(state['last_error']);self.assertEqual(state['stop_reason'],'cycle_complete')
        self.assertEqual(taps,fixed);self.assertEqual(len(fixed),4)

    def test_crown_grace_never_extends_itself_or_accepts_flicker_and_unknown(self):
        for scenario in ('late-slow','late-flicker','unknown'):
            state,taps,_ = self.replay('achievements',scenario)
            self.assertIn('未识别到后续界面',state['last_error']);self.assertEqual(len(taps),4)
            self.assertNotEqual(state['stop_reason'],'cycle_complete')


if __name__ == '__main__':unittest.main()
