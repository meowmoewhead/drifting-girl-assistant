"""Replay the real completion notice that interrupted the production upgrade."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import Clicker
from digit_ocr import DigitReader

ROOT=Path(__file__).resolve().parents[1]


class CompletionHandoffTests(unittest.TestCase):
    def replay(self, changed=False):
        modal=np.asarray(Image.open(ROOT/'tests/fixtures/parallel/complete-tier8-equipment.png').convert('RGB'))
        top=np.asarray(Image.open(ROOT/'tests/fixtures/equipment/top.png').convert('RGB'))
        clock=[0.]
        actions=[]
        class Device:
            def __init__(self,config,stop):
                self.stop=stop;self.io_lock=threading.RLock()
                self.screen='modal';self.count=0
            def screenshot(self):
                clock[0]+=.5;self.count+=1
                if changed and self.count>=4:return np.zeros_like(modal)
                return (modal if self.screen=='modal' else top).copy()
            def click(self,button):
                xy=((button.button[0]+button.button[2])//2,(button.button[1]+button.button[3])//2)
                actions.append(xy)
                self.screen='top'
            def command(self,*args):
                raise AssertionError('No scroll or long hold is allowed during completion handoff')
            def stuck_record_add(self,_):pass
        with tempfile.TemporaryDirectory(prefix='completion-handoff-') as directory:
            root=Path(directory)
            shutil.copytree(ROOT/'assets',root/'assets')
            config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            config['rules']=[deepcopy(r) for r in config['rules']
                             if r['id'] in ('driftinggirl_parallel_world','driftinggirl_equipment_upgrade')]
            for r in config['rules']:r['enabled']=True
            config['poll_interval']=.1
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            app=Clicker(root,Device)
            app.digit_reader=DigitReader(ROOT)
            try:
                with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])), \
                     patch('parallel_world.time',SimpleNamespace(monotonic=lambda:clock[0])), \
                     patch('equipment_upgrade.time',SimpleNamespace(monotonic=lambda:clock[0])):
                    app.start(preview=False,max_clicks=2,timeout=100)
                    app.thread.join(15)
                self.assertFalse(app.running())
                if changed:
                    self.assertEqual(actions,[])
                    self.assertIn('通关确认窗口未安全消失',app.state['last_error'])
                else:
                    self.assertIsNone(app.state['last_error'])
                    self.assertEqual(actions,[(360,840),(360,1166)])
                    self.assertEqual(app.state['stop_reason'],'click_limit')
                    self.assertEqual(app.state['equipment']['cycles'],0)
                    self.assertFalse(app.state['parallel']['completed_rank_40'])
                    self.assertEqual(app.state['parallel']['rounds'],0)
            finally:
                app.stop()
                if app.thread:app.thread.join(3)
                for handler in list(app.log.handlers):
                    handler.close();app.log.removeHandler(handler)

    def test_completion_preempts_equipment_and_folds_without_a_hold(self):self.replay()

    def test_changed_fresh_frame_prevents_confirmation_input(self):self.replay(changed=True)


if __name__=='__main__':unittest.main()
