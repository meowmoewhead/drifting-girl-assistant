"""A moving/missed advertisement must not strand the automation or double-claim."""
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
sys.path.insert(0,str(ROOT))
from app import Clicker
from digit_ocr import parse_counter
from features import FEATURE_BY_ID
from dev_tools.configure_ad_recovery import ENTRIES, NEW_IDS, update


class AdvertisementRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.rules={r['id']:r for r in cls.config['rules']}
        cls.context=SimpleNamespace(root=ROOT,config=cls.config)
        cls.frames={key:cls.image(path) for key,path in {
            'growth':'tests/fixtures/ad-recovery/growth-missed.png',
            'potion':'tests/fixtures/ad-position/potion-up-paused.png',
            'battle':'tests/fixtures/ad-recovery/whale-battle.png',
            'growth-confirm':'assets/ad-confirm-source.png',
            'potion-confirm':'assets/potion-ad-confirm-source.png',
            'growth-return':'assets/ad-reward-active-source.png',
            'potion-return':'tests/fixtures/legacy-captures/potion-ad-after-continue.png',
        }.items()}
        moved=cls.frames['growth'].copy()
        moved[650:1115,600:720]=0
        moved[700:810,630:720]=cls.frames['growth'][790:900,630:720]
        cls.frames['growth-before-move']=moved

    @staticmethod
    def image(path):
        with Image.open(ROOT/path) as source:
            return np.asarray(source.convert('RGB'))

    def matches(self,key,frame):
        r,b,o=Clicker.load_rule(self.context,self.rules[key])
        return b.match(frame,offset=o,similarity=r['threshold'])

    def test_observed_failure_and_battle_are_recoverable_game_screens(self):
        for key in NEW_IDS:
            for name in ('growth','potion','battle'):
                self.assertTrue(self.matches(key,self.frames[name]),(key,name))
            for box in ([211,1140,266,1189],[542,1140,597,1183],[437,1144,486,1193]):
                incomplete=self.frames['growth'].copy();x1,y1,x2,y2=box;incomplete[y1:y2,x1:x2]=0
                self.assertFalse(self.matches(key,incomplete))

    def test_dialogs_and_unknown_screens_do_not_trigger_recovery(self):
        names=('ad-confirm-source','potion-ad-confirm-source','mailbox-native','mana-confirm-native',
               'task-popup/notice','guild-shop/potion-confirm','event-potion/confirm',
               'achievements-list-native','raid/battle','raid/result','raid/waiting',
               'raid/prepare','daily-reward/premium-ready')
        frames=[self.image('assets/'+n+'.png') for n in names]
        frames.extend([np.zeros_like(self.frames['growth']),(self.frames['growth']*.6).astype(np.uint8)])
        for key in NEW_IDS:
            for frame in frames:
                self.assertFalse(self.matches(key,frame),key)

    def test_migration_preserves_original_rules_limits_and_switches(self):
        before=deepcopy(self.config);before['rules']=[r for r in before['rules'] if r['id'] not in NEW_IDS]
        for r in before['rules']:
            if r['id'] in ENTRIES:r.update(enabled=False,interval=321)
        after=update(before);self.assertEqual(update(after),after)
        by_id={r['id']:r for r in after['rules']}
        for old in before['rules']:
            new=by_id[old['id']]
            self.assertEqual({k:v for k,v in old.items() if k!='recheck_before_click'},
                             {k:v for k,v in new.items() if k!='recheck_before_click'})
        for key in NEW_IDS:self.assertFalse(by_id[key]['enabled'])
        for entry in ENTRIES:self.assertTrue(by_id[entry]['recheck_before_click'])

    def test_recheck_and_recovery_cannot_be_attached_to_claim_steps(self):
        original=self.rules[ENTRIES[0]]
        for changes in ({'recheck_before_click':'yes'},{'stage':'ad_reward'},
                        {'action':'observe'},{'purchase_condition':{'mode':'available'}},
                        {'absent_rule':'driftinggirl_ad_confirm'}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                Clicker.load_rule(self.context,{**original,**changes})
        with self.assertRaises(ValueError):
            Clicker.load_rule(self.context,{**self.rules[NEW_IDS[0]],'stage':'ad_reward'})

    def replay(self,feature,scenario):
        kind='growth' if feature=='growth_ad' else 'potion'
        entry=ENTRIES[0 if kind=='growth' else 1]
        total=10 if kind=='growth' else 40
        config=deepcopy(self.config);config['poll_interval']=.1
        ids=FEATURE_BY_ID[feature]['rules'];config['rules']=[r for r in config['rules'] if r['id'] in ids]
        for rule in config['rules']:rule.update(enabled=True,interval=.2,next_delay=0)
        frames,clock=self.frames,[0]
        class Device:
            def __init__(self,cfg,stop):
                self.stop,self.image,self.taps,self.phase,self.shots=stop,None,[],0,0
            def screenshot(self):
                self.shots+=1;self.phase+=1;clock[0]+=1;count=len(self.taps)
                if count==0:
                    name='growth-before-move' if scenario=='move' and self.shots<4 else kind
                    if scenario=='stop' and self.shots==4:self.stop.set()
                elif scenario=='repeat':name=kind
                elif count==1 and scenario in ('missed','battle','unknown'):
                    name='battle' if scenario=='battle' and self.phase<=14 else kind
                    if scenario=='unknown':
                        self.image=np.zeros_like(frames[kind]);return self.image
                else:
                    step=count-int(scenario in ('missed','battle'))
                    name=kind+'-confirm' if step==1 or scenario=='confirm-stuck' else kind+'-return'
                self.image=frames[name].copy();return self.image
            def click(self,button):
                if self.stop.is_set():raise InterruptedError()
                assert not np.array_equal(self.image,frames['battle']), 'Unexpected battle tap'
                self.taps.append((button.name,tuple(button.button)));self.phase=0
            def stuck_record_add(self,button):pass
        with tempfile.TemporaryDirectory(prefix='ad-replay-') as folder:
            root=Path(folder)
            for path in {r['asset'] for r in config['rules']}:
                (root/path).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,root/path)
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            app=Clicker(root,Device)
            app.digit_reader=SimpleNamespace(read_counter=lambda _:parse_counter(
                f'{total if scenario=="full" and app.device.shots>=4 else 0}/{total}',.99))
            try:
                with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])):
                    app.start(preview=scenario=='preview',stop_on_main=True,max_clicks=8,
                              timeout=18 if scenario in ('preview','full') else 180)
                    app.thread.join(18)
                    self.assertFalse(app.running(),'Replay timed out')
                return {**deepcopy(app.state),'events':list(app.events)},list(app.device.taps)
            finally:
                app.stop()
                if app.thread:app.thread.join(3)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

    def test_entry_move_between_scan_and_tap_cancels_stale_coordinate(self):
        state,taps=self.replay('growth_ad','move')
        self.assertIsNone(state['last_error']);self.assertEqual(state['stop_reason'],'cycle_complete')
        self.assertEqual(len(taps),2)
        x1,y1,x2,y2=taps[0][1];self.assertEqual(((x1+x2)//2,(y1+y2)//2),(649,843))
        self.assertTrue(any('取消本次点击' in e['message'] for e in state['events']))

    def test_full_counter_at_last_recheck_prevents_click(self):
        for feature in ('growth_ad','potion_ad'):
            state,taps=self.replay(feature,'full')
            self.assertEqual(taps,[]);self.assertIsNone(state['last_error'])
            entry=ENTRIES[0 if feature=='growth_ad' else 1]
            self.assertFalse(state['numbers'][entry]['passed'])

    def test_missed_open_and_battle_recover_then_complete_both_ad_workflows(self):
        for feature in ('growth_ad','potion_ad'):
            for scenario in ('missed','battle'):
                with self.subTest(feature=feature,scenario=scenario):
                    state,taps=self.replay(feature,scenario)
                    self.assertIsNone(state['last_error']);self.assertEqual(state['recoveries'],1)
                    self.assertEqual(state['stop_reason'],'cycle_complete');self.assertEqual(len(taps),3)
                    self.assertEqual(taps[0][0],taps[1][0]);self.assertNotEqual(taps[1][0],taps[2][0])

    def test_repeated_failure_stops_after_three_recoveries(self):
        state,taps=self.replay('growth_ad','repeat')
        self.assertIn('五分钟内已恢复三次',state['last_error']);self.assertEqual(state['recoveries'],3)
        self.assertEqual([tap[0] for tap in taps],[ENTRIES[0]]*4)

    def test_unknown_screen_still_stops_without_blind_retry(self):
        state,taps=self.replay('growth_ad','unknown')
        self.assertIn('未识别到后续界面',state['last_error']);self.assertEqual(len(taps),1)
        self.assertEqual(state['recoveries'],0)

    def test_continue_is_not_retried_when_reward_does_not_return(self):
        state,taps=self.replay('growth_ad','confirm-stuck')
        self.assertIn('未识别到后续界面',state['last_error']);self.assertEqual(len(taps),2)
        self.assertEqual(state['recoveries'],0)

    def test_stop_during_recheck_and_preview_send_no_taps(self):
        for scenario in ('stop','preview'):
            state,taps=self.replay('growth_ad',scenario)
            self.assertEqual(taps,[]);self.assertIsNone(state['last_error'])


if __name__=='__main__':unittest.main()
