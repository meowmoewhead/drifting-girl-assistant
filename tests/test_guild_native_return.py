"""Actual dim-sidebar battle after a guild close; observing never purchases."""
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
from digit_ocr import DigitReader
from drift.core.session import LoopState
from drift.tasks.stone_farm import StoneFarm
from drift.ui.guild_selection import GuildReturnGuard

ROOT=Path(__file__).resolve().parents[1]
def image(path):return np.asarray(Image.open(ROOT/path).convert('RGB'))

class GuildNativeReturnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.reader=DigitReader(ROOT)
        cls.real_battle=image('tests/fixtures/guild-selection/return-battle-20261016.png')
        # The fixed cart-core rule now recognizes this historical frame.
        # Obscure only the cart to keep exercising the independent native
        # fallback; the real fish, world, RANK and footer remain unchanged.
        cls.battle=cls.real_battle.copy()
        cls.battle[250:450,610:720]=0
        cls.shop=image('tests/fixtures/guild-selection/material-missed-20261016.png')
        ctx=SimpleNamespace(root=ROOT,config=cls.config)
        cls.matchers={r['id']:Clicker.load_rule(ctx,{**deepcopy(r),'enabled':True})
                      for r in cls.config['rules'] if r['id'].startswith('driftinggirl_guild_')}

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='guild-return-');self.addCleanup(self.tmp.cleanup)
        self.inputs=[];self.path=Path(self.tmp.name)/'stone_farm_state.json'
        self.world=StoneFarm(ROOT,self.reader,lambda *a:self.inputs.append(a),lambda *_:None,state_path=self.path)
        self.guard=GuildReturnGuard()
        self.st=LoopState(stage='guild_return',stage_ready_at=0,stage_deadline=20,
                          stage_trigger=self.matchers['driftinggirl_guild_close'][0]['name'],matchers=self.matchers)

    def observe(self,now,frame=None,**kwargs):
        return self.guard.observe(self.battle if frame is None else frame,self.st,lambda:self.world,now=now,**kwargs)

    def test_actual_battle_proves_return_without_input_or_record_change(self):
        before=deepcopy(self.world.journal),deepcopy(self.world.training.state)
        self.assertFalse(self.observe(1));self.assertFalse(self.observe(2));self.assertTrue(self.observe(4))
        self.assertEqual(before,(self.world.journal,self.world.training.state))
        self.assertEqual(self.inputs,[]);self.assertFalse(self.path.exists())
        self.assertEqual(self.st.stage_deadline,20)

    def test_real_historical_frame_now_uses_normal_return_rule(self):
        r,b,o=self.matchers['driftinggirl_guild_return']
        self.assertTrue(b.match(self.real_battle,offset=o,similarity=r['threshold']))
        self.assertEqual(r['action'],'observe')
        self.assertFalse(b.match(self.battle,offset=o,similarity=r['threshold']))

    def test_fast_frames_cannot_replace_three_seconds(self):
        for now in (1,1.1,1.2,1.5):self.assertFalse(self.observe(now))
        self.assertTrue(self.observe(4))

    def test_shop_modal_dark_unknown_and_raid_do_not_prove_return(self):
        for f in (self.shop,image('assets/guild-shop/material-min.png'),
                  image('assets/raid/waiting.png'),self.battle//2,np.zeros_like(self.battle)):
            for now in (1,2,4):self.assertFalse(self.observe(now,f))
            self.assertEqual(self.guard.frames,0)

    def test_stopped_interrupted_expired_or_other_stage_is_rejected(self):
        for changes in ({'stopped':True},{'interrupted':True}):
            for now in (1,2,4):self.assertFalse(self.observe(now,**changes))
        for attr,value in (('stage','guild_material_quantity'),('stage_trigger','other'),
                           ('stage_deadline',0),('handling_interrupt',True)):
            old=getattr(self.st,attr);setattr(self.st,attr,value)
            for now in (1,2,4):self.assertFalse(self.observe(now))
            setattr(self.st,attr,old)
        self.assertFalse(self.observe(20))

    def test_changed_actual_rank_restarts_consecutive_proof(self):
        other=image('tests/fixtures/crown-native-return/rank22-battle.png')
        self.assertFalse(self.observe(1));self.assertFalse(self.observe(2))
        self.assertFalse(self.observe(4,other));self.assertFalse(self.observe(5))
        self.assertFalse(self.observe(6));self.assertTrue(self.observe(8))

    def test_occlusion_resets_independent_votes_but_general_vote_clear_does_not(self):
        self.assertFalse(self.observe(1))
        self.assertFalse(self.observe(2,self.shop))
        self.assertFalse(self.observe(3))
        self.st.stable.clear()
        self.assertFalse(self.observe(4));self.assertTrue(self.observe(6))

    def test_runtime_closes_once_and_accepts_real_battle_before_original_deadline(self):
        shop,battle=self.shop,self.battle
        clock=[0.];devices=[]
        class Device:
            def __init__(self,config,stop):self.stop,self.image,self.taps=stop,None,[];devices.append(self)
            def screenshot(self):clock[0]+=.8;self.image=(battle if self.taps else shop).copy();return self.image
            def click(self,button):
                if self.stop.is_set() or self.taps or button.name!='driftinggirl_guild_close':
                    raise AssertionError('Unexpected guild input')
                self.taps.append(button.name)
            def stuck_record_add(self,_):pass
        root=Path(self.tmp.name)
        shutil.copytree(ROOT/'assets',root/'assets')
        cfg=deepcopy(self.config);cfg['poll_interval']=.1
        cfg['rules']=[deepcopy(self.matchers[k][0]) for k in ('driftinggirl_guild_close','driftinggirl_guild_return')]
        for r in cfg['rules']:r.update(interval=.2,next_delay=0)
        (root/'config.json').write_text(json.dumps(cfg),encoding='utf-8')
        app=Clicker(root,Device);app.digit_reader=self.reader
        try:
            with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])):
                app.start(preview=False,stop_on_main=True,max_clicks=2,timeout=35);app.thread.join(20)
            self.assertFalse(app.running());self.assertIsNone(app.state['last_error'])
            self.assertEqual(app.state['stop_reason'],'cycle_complete');self.assertEqual(app.state['clicks'],1)
            self.assertLess(clock[0],24)
            self.assertFalse(app.daily_schedule.is_complete('guild_shop'))
            self.assertFalse((root/'stone_farm_state.json').exists())
        finally:
            app.stop()
            if app.thread:app.thread.join(3)
            for h in list(app.log.handlers):h.close();app.log.removeHandler(h)

if __name__=='__main__':unittest.main()
