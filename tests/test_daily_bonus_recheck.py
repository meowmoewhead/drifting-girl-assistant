"""A claim must survive an independent new frame, and never retry an unchanged card."""
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


class BonusRecheckTests(unittest.TestCase):
    def replay(self,mutation):
        ready=np.asarray(Image.open(ROOT/'assets/daily-reward/bonus-ready.png').convert('RGB'))
        gray=np.asarray(Image.open(ROOT/'assets/daily-reward/premium-claimed.png').convert('RGB'))
        returned=np.asarray(Image.open(ROOT/'assets/daily-reward/return.png').convert('RGB'))
        clock=[0.];devices=[]
        class Device:
            def __init__(self,config,stop):
                self.stop=stop;self.image=None;self.reads=0;self.taps=[];devices.append(self)
            def screenshot(self):
                clock[0]+=.6;self.reads+=1
                frame=ready.copy()
                if self.taps and self.taps[-1][0]>630:frame=returned.copy()
                elif self.reads>=7:
                    if mutation=='gray':frame=gray.copy()
                    elif mutation=='dark':frame=ready//2
                    elif mutation=='stop':self.stop.set()
                    elif mutation=='rank':frame[10:62,72:126]=255 if self.reads%2 else 0
                self.image=frame
                return frame
            def click(self,button):
                if self.stop.is_set():raise AssertionError('Input after stop')
                a,b,c,d=button.button
                self.taps.append(((a+c)//2,(b+d)//2))
            def stuck_record_add(self,_):pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            shutil.copytree(ROOT/'assets/daily-reward',root/'assets/daily-reward')
            config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            config['poll_interval']=.1
            config['rules']=[deepcopy(r) for r in config['rules'] if r['id'].startswith('driftinggirl_daily_')]
            for r in config['rules']:r.update(enabled=True,interval=.2,next_delay=0,stage_timeout=5)
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            app=Clicker(root,Device)
            try:
                with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])):
                    app.start(preview=False,stop_on_main=True,max_clicks=4,timeout=35)
                    app.thread.join(12)
                self.assertFalse(app.running())
                taps=devices[0].taps
                if mutation=='gray':
                    self.assertEqual(len(taps),1)
                    self.assertGreater(taps[0][0],630) # close, never a bonus input
                    self.assertTrue(app.daily_schedule.is_complete('daily_reward'))
                elif mutation=='unchanged':
                    self.assertEqual(len(taps),2) # one claim attempt, then bounded cleanup
                    self.assertEqual(taps[0],(507,465))
                    self.assertGreater(taps[1][0],630)
                    self.assertFalse(app.daily_schedule.is_complete('daily_reward'))
                else:
                    self.assertEqual(taps,[])
                    self.assertFalse(app.daily_schedule.is_complete('daily_reward'))
            finally:
                app.stop()
                if app.thread:app.thread.join(2)
                for h in list(app.log.handlers):h.close();app.log.removeHandler(h)

    def test_ready_turns_gray_before_input_so_only_close_is_clicked(self):self.replay('gray')
    def test_dimming_before_input_issues_no_claim(self):self.replay('dark')
    def test_stop_before_input_issues_no_claim(self):self.replay('stop')
    def test_rank_changes_before_input_issues_no_claim(self):self.replay('rank')
    def test_unchanged_ready_card_is_not_repeatedly_clicked_or_counted_complete(self):self.replay('unchanged')


if __name__=='__main__':unittest.main()
