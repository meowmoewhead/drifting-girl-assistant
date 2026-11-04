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
from drift.ui.resource_battle import ResourceBattleRecoveryButton, adapt_resource_battle

ROOT=Path(__file__).resolve().parents[1]


class ShiftedTimerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.rule=next(r for r in cls.config['rules'] if r['id']=='driftinggirl_achievements_battle_recover')
        cls.frames={n:np.asarray(Image.open(ROOT/'tests/fixtures/crown-shifted-timer'/(n+'.png')).convert('RGB'))
                    for n in ('battle','menu','missing-timer')}

    def button(self,rule=None):
        return Clicker.load_rule(SimpleNamespace(root=ROOT,config=self.config),deepcopy(rule or self.rule))[1]

    def test_real_moved_timer_keeps_fixed_bottom_guards(self):
        b=self.button();f=self.frames['battle']
        self.assertFalse(b.original.match(f,similarity=self.rule['threshold']))
        self.assertTrue(b.match(f,similarity=self.rule['threshold']))
        self.assertGreater(b.last_similarity,.96)

    def test_missing_each_guard_dark_corrupt_and_menu_reject(self):
        cases=[self.frames['menu'],self.frames['missing-timer'],self.frames['battle']//2,
               np.zeros_like(self.frames['battle']),self.frames['battle'][:900]]
        for x,y,X,Y in ((188,125,610,165),(211,1140,266,1189),(542,1140,597,1183),(558,1147,585,1168),(113,153,187,208)):
            f=self.frames['battle'].copy();f[y:Y,x:X]=0;cases.append(f)
        for frame in cases:self.assertFalse(self.button().match(frame,similarity=.93))

    def test_adaptation_never_changes_saved_rule_or_clicks_or_custom_geometry(self):
        original=deepcopy(self.rule)
        self.assertIsInstance(self.button(),ResourceBattleRecoveryButton)
        self.assertEqual(self.rule,original)
        for change in ({'action':'click'},{'source_crop':[335,133,356,157]},
                       {'search_area':[320,123,365,167]},{'scales':[.9]},
                       {'stage':'main'},{'next_stage':'achievements_claim'}):
            frame=np.asarray(Image.open(ROOT/self.rule['asset']).convert('RGB'))
            self.assertNotIsInstance(adapt_resource_battle({**self.rule,**change},self.button().original,frame),ResourceBattleRecoveryButton)

    def test_actual_menu_interrupted_by_whale_recovers_without_click(self):
        clock=[0.];frames=self.frames;devices=[]
        config=deepcopy(self.config);config['poll_interval']=.1
        config['rules']=[r for r in config['rules'] if r['id'].startswith('driftinggirl_achievements_')]
        for r in config['rules']:r.update(enabled=True,next_delay=0)
        class Device:
            def __init__(self,_config,stop):self.stop=stop;self.n=0;self.taps=[];devices.append(self)
            def screenshot(self):
                clock[0]+=1;self.n+=1
                if self.n>22:raise InterruptedError('End bounded replay')
                return frames['menu' if self.n<=2 else 'battle'].copy()
            def click(self,button):self.taps.append(button.name);raise AssertionError('Recovery must be read only')
            def stuck_record_add(self,_):pass
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            assets={r['asset'] for r in config['rules']}
            assets.update(p for r in config['rules'] for p in r.get('asset_variants',[]))
            for name in assets:
                (root/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,root/name)
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            clicker=Clicker(root,Device)
            try:
                with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])):
                    clicker.start(preview=False,timeout=60);clicker.thread.join(15)
                self.assertFalse(clicker.running())
                self.assertIsNone(clicker.state['last_error'])
                self.assertEqual(clicker.state['recoveries'],1)
                self.assertEqual(clicker.state['clicks'],0)
                self.assertEqual(devices[0].taps,[])
                self.assertTrue(any('巨鱼战斗打断' in e['message'] for e in clicker.events))
            finally:
                clicker.stop()
                if clicker.thread:clicker.thread.join(2)
                for h in list(clicker.log.handlers):h.close();clicker.log.removeHandler(h)


if __name__=='__main__':unittest.main()
