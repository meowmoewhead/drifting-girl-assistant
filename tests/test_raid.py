"""RAID must wait for readiness, avoid fighting controls, and confirm settlement."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import Clicker

class RaidTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.rules={r['id']:r for r in cls.config['rules'] if r['id'].startswith('driftinggirl_raid_')}
        cls.context=SimpleNamespace(root=ROOT,config=cls.config)
        cls.frames={name:np.asarray(Image.open(ROOT/'assets'/'raid'/(name+'.png')).convert('RGB'))
                    for name in ('main','prepare','waiting','ready','battle','result','return')}

    def match(self,key,frame):
        rule,button,offset=Clicker.load_rule(self.context,self.rules['driftinggirl_raid_'+key])
        matched=button.match(frame,similarity=rule['threshold'])
        if matched and rule.get('absent_rule'):
            other=self.rules[rule['absent_rule']]
            _,guard,_=Clicker.load_rule(self.context,other)
            matched=not guard.match(frame,similarity=other['threshold'])
        return matched,button

    def test_start_requires_both_ready_text_and_active_color(self):
        ready=self.frames['ready'];waiting=self.frames['waiting']
        self.assertTrue(self.match('start',ready)[0])
        self.assertFalse(self.match('start',waiting)[0])
        text_only=ready.copy();text_only[580:610,215:239]=waiting[580:610,215:239]
        matched,button=self.match('start',text_only)
        self.assertFalse(matched)
        self.assertEqual(button.last_similarity,-1.0)
        color_only=waiting.copy();color_only[566:625,203:352]=ready[566:625,203:352]
        self.assertFalse(self.match('start',color_only)[0])

    def test_names_and_countdown_do_not_affect_start(self):
        frame=self.frames['ready'].copy()
        frame[300:553,192:528]=0
        frame[660:685,312:410]=0
        self.assertTrue(self.match('start',frame)[0])

    def test_prepare_ignores_raid_level_remaining_time_and_ranking(self):
        frame=self.frames['prepare'].copy()
        frame[430:465,329:390]=0
        frame[749:844,418:548]=0
        self.assertTrue(self.match('prepare',frame)[0])

    def test_settlement_ignores_changing_numbers_and_never_matches_battle(self):
        frame=self.frames['result'].copy()
        frame[377:419,319:444]=0
        frame[824:855,445:537]=0
        self.assertTrue(self.match('confirm',frame)[0])
        self.assertFalse(self.match('confirm',self.frames['battle'])[0])
        self.assertFalse(self.match('confirm',self.frames['main'])[0])
        self.assertFalse(self.match('return',self.frames['result'])[0])
        self.assertTrue(self.match('return',self.frames['return'])[0])

    def test_invalid_color_conditions_are_rejected(self):
        for condition in ({'source_area':[0,0,8,8],'hsv':[[0,-1,0],[120,255,255]],'minimum_fraction':.9},
                          {'source_area':[0,0,8,8],'hsv':[[150,0,0],[120,255,255]],'minimum_fraction':.9},
                          {'source_area':[0,0,8,8],'hsv':[[90,0,0],[120,255,255]],'minimum_fraction':1.1},
                          {'source_area':[0,0,8,8],'hsv':[[90,0,0],[120,255]],'minimum_fraction':.9}):
            rule=deepcopy(self.rules['driftinggirl_raid_start']);rule['color_condition']=condition
            with self.subTest(condition=condition),self.assertRaises(ValueError):Clicker.load_rule(self.context,rule)

    def test_full_replay_waits_without_clicking_other_main_tasks(self):
        frames=self.frames
        expected=['open','prepare','start','confirm']
        class ReplayDevice:
            def __init__(self,config,stop):
                self.stop,self.image,self.taps,self.phase_frames=stop,None,[],0
            def screenshot(self):
                self.phase_frames+=1
                count=len(self.taps)
                name=('main' if count==0 else 'prepare' if count==1 else
                      ('waiting' if self.phase_frames<=5 else 'ready') if count==2 else
                      ('battle' if self.phase_frames<=8 else 'result') if count==3 else 'return')
                self.image=frames[name].copy();return self.image
            def click(self,button):
                if self.stop.is_set():raise InterruptedError()
                index=len(self.taps)
                if index>=4 or button.name!='driftinggirl_raid_'+expected[index]:raise AssertionError('Unexpected click: '+button.name)
                if index==2 and self.phase_frames<8:raise AssertionError('Started before stable ready state')
                if index==3 and self.phase_frames<11:raise AssertionError('Clicked during battle')
                self.taps.append(button.name);self.phase_frames=0
            def stuck_record_add(self,button):pass
        with tempfile.TemporaryDirectory(prefix='raid-replay-') as directory:
            root=Path(directory)
            for path in {r['asset'] for r in self.rules.values()} | {
                    p for r in self.rules.values() for p in r.get('asset_variants', [])}:
                (root/path).parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/path,root/path)
            config=deepcopy(self.config);config['poll_interval']=.1
            rules=deepcopy(list(self.rules.values()))
            for rule in rules:rule.update(enabled=True,interval=.2,next_delay=0,stage_timeout=5)
            distraction=deepcopy(rules[2]);distraction.update(id='other-main-task',stage='main',action='click')
            for key in ('next_stage','color_condition','required_source_areas'):distraction.pop(key,None)
            config['rules']=[distraction]+rules
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            app=Clicker(root,ReplayDevice)
            try:
                app.start(preview=False,stop_on_main=True,max_clicks=5,timeout=15)
                app.thread.join(18)
                self.assertFalse(app.running());self.assertIsNone(app.state['last_error'])
                self.assertEqual(app.state['stop_reason'],'cycle_complete')
                self.assertEqual(app.state['clicks'],4)
            finally:
                app.stop()
                if app.thread:app.thread.join(3)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

if __name__=='__main__':unittest.main()
