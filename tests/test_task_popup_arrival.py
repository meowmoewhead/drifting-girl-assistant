"""Real completed-task layout and the input guard at the moment of closure."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from app import Clicker

ROOT=Path(__file__).resolve().parents[1]
CLOSE='driftinggirl_task_popup_close'
RETURN='driftinggirl_task_popup_return'

def image(path):
    with Image.open(ROOT/path) as source:
        return np.asarray(source.convert('RGB')).copy()

class TaskPopupArrivalTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        self.rule=deepcopy(next(r for r in self.config['rules'] if r['id']==CLOSE))
        self.context=SimpleNamespace(root=ROOT,config=self.config)
        self.frame=image('tests/fixtures/task-popup-arrival/arrival-moving.png')

    def match(self,frame,rule=None):
        r,b,o=Clicker.load_rule(self.context,rule or self.rule)
        return b.match(frame,offset=o,similarity=r['threshold']),b

    def test_two_independent_new_frames_and_original_layouts(self):
        before=deepcopy(self.config)
        for path,box in [('tests/fixtures/task-popup-arrival/arrival-moving.png',(627,387,665,424)),
                         ('tests/fixtures/task-popup-arrival/arrival-battle.png',(627,387,665,424)),
                         ('assets/task-popup/notice.png',(627,412,665,449)),
                         ('assets/task-popup/kraken-stage3.png',(627,412,665,449))]:
            passed,button=self.match(image(path))
            self.assertTrue(passed,path);self.assertEqual(tuple(button.button),box)
            self.assertGreater(button.last_similarity,.93)
        self.assertEqual(self.config,before)

    def test_each_fixed_fragment_is_required(self):
        for x,y,X,Y in ((618,381,674,428),(315,440,350,467),(271,523,302,572),
                        (106,810,162,835),(131,724,190,747)):
            damaged=self.frame.copy();damaged[y:Y,x:X]=0
            self.assertFalse(self.match(damaged)[0],(x,y,X,Y))

    def test_unknown_modal_battle_and_dark_frames_are_not_close_targets(self):
        for path in ('assets/guild-shop/potion-confirm.png','assets/raid/battle.png',
                     'assets/daily-reward/premium-ready.png','assets/task-popup/main.png'):
            self.assertFalse(self.match(image(path))[0],path)
        self.assertFalse(self.match(self.frame//2)[0])

    def test_custom_geometry_does_not_inherit_added_layout(self):
        changed=deepcopy(self.rule);changed['click_source_area']=[626,412,665,449]
        self.assertFalse(self.match(self.frame,changed)[0])

    def runtime(self,mode='normal'):
        config=deepcopy(self.config)
        config['rules']=[r for r in config['rules'] if r['id'] in (CLOSE,RETURN)]
        for r in config['rules']:r.update(enabled=True,interval=.2,next_delay=0)
        config['poll_interval']=.1
        notice=self.frame.copy();main=image('assets/task-popup/main.png')
        old=image('assets/task-popup/notice.png');clock=[0.]
        class Device:
            def __init__(self,cfg,stop):
                self.stop=stop;self.image=None;self.n=0;self.taps=[]
            def screenshot(self):
                self.n+=1;clock[0]+=1
                frame=(main if self.taps else notice).copy()
                if self.n==4 and mode!='normal':
                    if mode=='stop':self.stop.set()
                    if mode=='expired':clock[0]+=20
                    if mode=='run_deadline':clock[0]+=8
                    if mode=='gone':frame=main.copy()
                    if mode=='layout':frame=old.copy()
                    if mode=='rank':frame[10:62,72:126]=0
                    if mode=='world':frame[46:66,35:56]=0
                if self.n>=5 and mode not in ('normal','expired','run_deadline'):self.stop.set()
                self.image=frame;return frame
            def click(self,button):
                if self.n!=4 or button.name!=CLOSE:raise AssertionError('Unexpected input')
                self.taps.append((button.name,tuple(map(int,button.button))))
            def stuck_record_add(self,button):pass
        with tempfile.TemporaryDirectory(prefix='popup-arrival-') as directory:
            root=Path(directory)
            paths={p for r in config['rules'] for p in [r['asset'],*r.get('asset_variants',[])]}
            paths.update(('assets/task-popup/kraken-stage3.png','assets/task-popup/kraken-arrival.png'))
            for path in paths:
                (root/path).parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/path,root/path)
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            app=Clicker(root,Device)
            try:
                with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])):
                    app.start(preview=False,rule_ids=[CLOSE,RETURN],stop_on_main=True,max_clicks=2,
                              timeout=10 if mode=='run_deadline' else 40)
                    app.thread.join(15)
                    self.assertFalse(app.running())
                receipts=[json.loads(p.read_text(encoding='utf-8')) for p in (root/'logs/popup-input').glob('input-*.json')]
                if mode=='normal':
                    self.assertEqual(app.device.taps,[(CLOSE,(627,387,665,424))])
                    self.assertIsNone(app.state['last_error'])
                    self.assertEqual(app.state['stop_reason'],'cycle_complete',
                                     {key:app.state[key] for key in ('stage','last_match','recoveries','clicks')})
                    self.assertEqual(len(receipts),1)
                    self.assertNotEqual(receipts[0]['before_frame_id'],receipts[0]['after_frame_id'])
                else:
                    self.assertEqual(app.device.taps,[])
                    self.assertEqual(receipts,[])
                    if mode=='expired':self.assertIn('任务弹窗持续遮挡',app.state['last_error'])
                    if mode=='run_deadline':self.assertEqual(app.state['stop_reason'],'timeout')
            finally:
                app.stop()
                if app.thread:app.thread.join(3)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

    def test_runtime_closes_once_with_new_frame_and_records_return(self):self.runtime()
    def test_fresh_frame_stop_prevents_close(self):self.runtime('stop')
    def test_old_interrupt_deadline_is_not_extended(self):self.runtime('expired')
    def test_run_deadline_during_fresh_capture_prevents_close(self):self.runtime('run_deadline')
    def test_notice_disappearing_before_input_is_not_clicked(self):self.runtime('gone')
    def test_layout_changing_before_input_is_not_clicked(self):self.runtime('layout')
    def test_rank_changing_before_input_is_not_clicked(self):self.runtime('rank')
    def test_world_changing_before_input_is_not_clicked(self):self.runtime('world')

if __name__=='__main__':unittest.main()
