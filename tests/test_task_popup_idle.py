"""2026-10-21: notice closed over the unselected lowered-world main screen."""
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

class PopupIdleTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        self.context=SimpleNamespace(root=ROOT,config=self.config)
        self.rule=deepcopy(next(r for r in self.config['rules'] if r['id']==RETURN))
        self.main=image('tests/fixtures/task-popup-idle/returned.png')
        self.notice=image('tests/fixtures/task-popup-idle/notice.png')

    def matches(self,frame,rule=None):
        r,b,o=Clicker.load_rule(self.context,rule or self.rule)
        return b.match(frame,offset=o,similarity=r['threshold'])

    def test_independent_error_frame_matches_without_changing_saved_rule(self):
        before=deepcopy(self.config)
        self.assertTrue(self.matches(self.main))
        self.assertTrue(self.matches(image('assets/task-popup/main-idle.png')))
        self.assertEqual(self.config,before)
        self.assertEqual(self.rule['threshold'],.93)

    def test_all_three_fragments_and_brightness_remain_required(self):
        for x,y,X,Y in ((279,884,307,910),(211,1140,266,1189),(542,1140,597,1183)):
            damaged=self.main.copy();damaged[y:Y,x:X]=0
            self.assertFalse(self.matches(damaged),(x,y,X,Y))
        for frame in (self.notice,self.main//2,np.zeros_like(self.main)):
            self.assertFalse(self.matches(frame))
        for path in ('assets/raid/waiting.png','assets/raid/battle.png',
                     'assets/guild-shop/potion-confirm.png','assets/daily-reward/premium-ready.png'):
            self.assertFalse(self.matches(image(path)),path)

    def test_custom_geometry_does_not_inherit_the_new_layout(self):
        rule=deepcopy(self.rule);rule['search_area']=[276,880,310,913]
        self.assertFalse(self.matches(self.main,rule))

    def test_real_notice_closes_once_and_resumes_after_consecutive_frames(self):
        config=deepcopy(self.config)
        config['rules']=[r for r in config['rules'] if r['id'] in (CLOSE,RETURN)]
        for r in config['rules']:r.update(enabled=True,interval=.2,next_delay=0)
        config['poll_interval']=.1
        notice,main=self.notice,self.main;clock=[0.]
        class Device:
            def __init__(self,cfg,stop):
                self.stop=stop;self.image=None;self.taps=[];self.main_reads=0
            def screenshot(self):
                clock[0]+=1
                if self.taps:self.main_reads+=1
                self.image=(main if self.taps else notice).copy()
                return self.image
            def click(self,button):
                if button.name!=CLOSE:raise AssertionError('Unexpected input')
                self.taps.append((button.name,tuple(map(int,button.button))))
            def stuck_record_add(self,button):pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            shutil.copytree(ROOT/'assets/task-popup',root/'assets/task-popup')
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            app=Clicker(root,Device)
            try:
                with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])):
                    app.start(preview=False,rule_ids=[CLOSE,RETURN],stop_on_main=True,
                              timeout=35,max_clicks=2)
                    app.thread.join(15)
                self.assertFalse(app.running())
                self.assertIsNone(app.state['last_error'])
                self.assertEqual(app.state['stop_reason'],'cycle_complete')
                self.assertEqual(app.device.taps,[(CLOSE,(627,412,665,449))])
                self.assertGreaterEqual(app.device.main_reads,3)
                self.assertEqual(json.loads((root/'config.json').read_text()),config)
            finally:
                app.stop()
                if app.thread:app.thread.join(2)
                for h in list(app.log.handlers):h.close();app.log.removeHandler(h)

if __name__=='__main__':unittest.main()
