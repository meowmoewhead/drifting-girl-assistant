"""Replay the real RANK39 resource battle which interrupted the crown list."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import Clicker
from digit_ocr import DigitReader
from parallel_world import ParallelWorld

ROOT=Path(__file__).resolve().parents[1]

def image(path):
    return np.asarray(Image.open(ROOT/path).convert('RGB'))

class ResourceCrownRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.battle=image('tests/fixtures/resource-crown/battle.png')
        cls.menu=image('tests/fixtures/resource-crown/menu.png')
        cls.reader=DigitReader(ROOT)
        cls.config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))

    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(prefix='resource-crown-')
        self.addCleanup(self.directory.cleanup)
        self.actions=[]
        self.world=ParallelWorld(ROOT,lambda *a:self.actions.append(a),
            lambda *a:self.actions.append(a),lambda *a:None,self.reader,
            training_state_path=Path(self.directory.name)/'training_state.json',
            fishing_action=lambda *a:self.actions.append(a),
            sailing_action=lambda *a:self.actions.append(a),
            defense_action=lambda *a:self.actions.append(a))
        w=self.world
        w.rank,w.target_rank,w.phase=39,39,'play'
        w.training.options.update(enabled=True,auto_sailing=True,battle_tap=False,boss_interrupt=False)
        w.training.state=dict(mode='farming',rank=39,boss_level=1810,start_level=1633,
            started_at=1,sailing_enabled=True,sailing_verified=True)

    def test_real_resource_battle_is_read_only_and_works_with_battle_tapping_disabled(self):
        self.assertIsNotNone(self.world.resource_battle_recovery_context(self.battle))
        self.assertEqual(self.actions,[])
        self.assertFalse((Path(self.directory.name)/'training_state.json').exists())

    def test_saved_same_rank_verified_route_and_enabled_training_are_required(self):
        self.assertIsNotNone(self.world.resource_battle_recovery_context(self.battle))
        state=deepcopy(self.world.training.state)
        for update in (dict(rank=38),dict(mode='idle'),dict(sailing_enabled=False),
                       dict(sailing_verified=False),dict(boss_level=0)):
            with self.subTest(update=update):
                self.world.training.state={**state,**update}
                self.assertIsNone(self.world.resource_battle_recovery_context(self.battle))
        self.world.training.state=state
        for key in ('enabled','auto_sailing'):
            self.world.training.options[key]=False
            self.assertIsNone(self.world.resource_battle_recovery_context(self.battle))
            self.world.training.options[key]=True
        self.world.sailing_shutdown=True
        self.assertIsNone(self.world.resource_battle_recovery_context(self.battle))

    def test_dark_modal_wrong_world_rank_boss_port_and_missing_combat_guards_are_rejected(self):
        w=self.world
        self.assertIsNotNone(w.resource_battle_recovery_context(self.battle))
        for frame in (self.menu,(self.battle*.6).astype(np.uint8),np.zeros_like(self.battle),
            image('tests/fixtures/training/sea-god-meter-high.png'),
            image('assets/mailbox-native.png'),image('assets/navigation/main.png')):
            self.assertIsNone(w.resource_battle_recovery_context(frame))
        for box in ((8,40,46,80),(65,9,139,63),(194,4,246,42),
                    (153,205,192,245),(188,125,610,165),(307,962,415,1057),
                    (208,1137,269,1192),(539,1137,600,1186),(276,881,310,913)):
            altered=self.battle.copy();x,y,X,Y=box;altered[y:Y,x:X]=0
            with self.subTest(box=box):
                self.assertIsNone(w.resource_battle_recovery_context(altered))
        w.rank=38
        self.assertIsNone(w.resource_battle_recovery_context(self.battle))
        w.rank=40
        self.assertIsNone(w.resource_battle_recovery_context(self.battle))

    def test_only_fixed_location_stage_and_rank_strokes_form_context(self):
        w=self.world; before=w.resource_battle_recovery_context(self.battle)
        self.assertIsNotNone(before)
        altered=self.battle.copy()
        altered[42:68,295:422]=0 # moving travel countdown
        altered[180:194,390:510]=0 # numeric health amount
        altered[128:158,460:530]=0 # moving battle countdown, not its hourglass
        self.assertEqual(w.resource_battle_recovery_context(altered),before)
        moved=self.battle.copy();moved[12:34,286:434]=255
        self.assertIsNotNone(w.resource_battle_recovery_context(moved))
        self.assertNotEqual(w.resource_battle_recovery_context(moved),before)

    def replay(self,scenario):
        config=deepcopy(self.config)
        config['poll_interval']=.1
        config['rules']=[r for r in config['rules'] if '_achievements_' in r['id']
                         or r['id']=='driftinggirl_parallel_world']
        for rule in config['rules']:
            # Exercise resource-recovery limits independently of optional skip;
            # the new default skip/cleanup path has its own worker replay.
            rule.update(enabled=True,interval=.2,next_delay=0,skip_on_timeout=False)
        if scenario=='disabled':
            config['rules']=[r for r in config['rules']
                if r['id']!='driftinggirl_achievements_parallel_battle_recover']
        frames=dict(menu=self.menu,battle=self.battle)
        clock=[0.]
        world=self.world
        device_reference=[]

        class Device:
            def __init__(self,_config,stop):
                self.stop,self.image=stop,None
                self.taps=[];self.after_frames=0;self.total_frames=0
                device_reference.append(self)
            def screenshot(self):
                clock[0]+=1;self.total_frames+=1
                if not self.taps:
                    self.image=frames['menu'].copy()
                else:
                    self.after_frames+=1
                    if scenario=='expired' and self.after_frames==1:
                        clock[0]+=31
                    if scenario=='unknown':
                        self.image=np.zeros_like(frames['battle'])
                    elif scenario=='repeat' and self.after_frames>=8:
                        self.image=frames['menu'].copy()
                    elif scenario=='flicker' and self.after_frames==3:
                        self.image=(frames['battle']*.6).astype(np.uint8)
                    else:
                        self.image=frames['battle'].copy()
                    if self.after_frames>=(44 if scenario in ('disabled','unknown','repeat','expired') else 14):
                        raise InterruptedError('End of bounded replay')
                return self.image
            def click(self,button):
                assert np.array_equal(self.image,frames['menu']), 'No battle clicks allowed'
                assert button.name=='driftinggirl_achievements_crown'
                self.taps.append(button.name)
                self.after_frames=0
            def stuck_record_add(self,_button):
                pass

        class Defense:
            def status(self):return {}
            def owns(self,_frame,_rank):return False
            def step(self,frame,_rank):
                return (scenario in ('preempt','flicker')
                        and world.resource_battle_recovery_context(frame) is not None)
        class Parallel:
            only_sailing_shutdown=False
            def __init__(self,*_args,**_kwargs):
                self.training,self.sailing,self.rank=world.training,world.sailing,world.rank
                self.defense=Defense();self.phase='play'
            def observe(self,_frame):pass
            def observe_boss_sailing_rank(self,_frame,**_kwargs):return False
            def boss_sailing_stop_requested(self,*_args):return False
            def completion_step(self,_frame,**_kwargs):return False
            def resource_battle_recovery_context(self,frame):
                return world.resource_battle_recovery_context(frame)
            def suspend(self,_seconds):pass
            def status(self):return world.status()
            def step(self,_frame):return scenario!='repeat' and bool(device_reference[0].taps)

        root=Path(self.directory.name)/'runner';root.mkdir(exist_ok=True)
        assets={r['asset'] for r in config['rules']}
        assets.update(p for r in config['rules'] for p in r.get('asset_variants',[]))
        for path in assets:
            (root/path).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/path,root/path)
        (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
        clicker=Clicker(root,Device)
        clicker.digit_reader=self.reader
        try:
            with patch('app.ParallelWorld',Parallel),patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])):
                clicker.start(preview=False,timeout=160)
                clicker.thread.join(65)
                self.assertFalse(clicker.running(),f'Bounded replay did not finish: {clicker.state}')
            return deepcopy(clicker.state),clicker.device.taps,list(clicker.events)
        finally:
            clicker.stop()
            if clicker.thread:clicker.thread.join(2)
            for handler in list(clicker.log.handlers):
                handler.close();clicker.log.removeHandler(handler)

    def test_defense_preemption_cannot_starve_independent_read_only_recovery(self):
        state,taps,events=self.replay('preempt')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['recoveries'],1)
        self.assertEqual(taps,['driftinggirl_achievements_crown'])
        self.assertTrue(any('本次不记领奖完成' in e['message'] for e in events))

    def test_flicker_resets_recovery_confirmation_without_battle_input(self):
        state,taps,_=self.replay('flicker')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['recoveries'],1)
        self.assertEqual(len(taps),1)

    def test_disabled_recovery_and_unknown_frame_cannot_abandon_pending_crown(self):
        for scenario in ('disabled','unknown'):
            state,taps,events=self.replay(scenario)
            self.assertEqual(state['recoveries'],0)
            self.assertIn('后续界面',state['last_error'])
            self.assertFalse(any('普通资源战斗打断' in e['message'] for e in events))
            self.assertEqual(len(taps),1)

    def test_expired_pending_stage_is_not_renewed_by_resource_recovery(self):
        state,taps,events=self.replay('expired')
        self.assertIn('后续界面',state['last_error'])
        self.assertEqual(state['recoveries'],0)
        self.assertEqual(len(taps),1)
        self.assertFalse(any('普通资源战斗打断' in e['message'] for e in events))

    def test_repeated_resource_interruptions_share_original_three_per_five_minute_limit(self):
        state,taps,_=self.replay('repeat')
        self.assertIn('五分钟内已恢复三次',state['last_error'])
        self.assertEqual(state['recoveries'],3)
        self.assertEqual(len(taps),4)

if __name__=='__main__':unittest.main()
