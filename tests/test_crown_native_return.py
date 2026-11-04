"""A read-only crown return must survive combat owning the main loop."""
from copy import deepcopy
import hashlib
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
from drift.core.matcher import observe_native_crown_return
from drift.core.session import LoopState
from drift.tasks.features import FEATURE_BY_ID
from drift.tasks.stone_farm import StoneFarm

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT/'tests/fixtures/crown-native-return'

def image(path):
    return np.asarray(Image.open(ROOT/path).convert('RGB'))


class CrownNativeReturnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.reader=DigitReader(ROOT)
        cls.frames={name:image('tests/fixtures/crown-native-return/'+file) for name,file in (
            ('battle','rank22-battle.png'),('filled','rank22-filled-preparation.png'),
            ('success','rank22-success.png'),('occluded','rank5-occluded-result.png'))}
        cls.frames.update(main=image('assets/navigation/main.png'),
            menu=image('assets/achievements-menu-native.png'),
            listing=image('tests/fixtures/legacy-captures/achievements-cycle-result.png'),
            completion=image('assets/recovery/parallel-complete.png'))

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='crown-native-return-')
        self.addCleanup(self.tmp.cleanup)
        self.inputs=[]
        self.journal=Path(self.tmp.name)/'stone_farm_state.json'
        self.world=StoneFarm(ROOT,self.reader,lambda *a:self.inputs.append(a),lambda *a:None,
                             state_path=self.journal)
        self.clock=[0.]
        self.clock_object=SimpleNamespace(monotonic=lambda:self.clock[0])
        self.stop=threading.Event()
        self.st=LoopState(stage='achievements_return',stage_started_at=0,
                          stage_ready_at=0,stage_deadline=15)
        cfg=SimpleNamespace(root=ROOT,config=self.config)
        for r in self.config['rules']:
            if r['id'].startswith('driftinggirl_achievements_'):
                self.st.matchers[r['id']]=Clicker.load_rule(cfg,{**deepcopy(r),'enabled':True})
        self.st.stage_trigger=self.st.matchers['driftinggirl_achievements_fold'][0]['name']
        self.frame=self.frames['battle']
        self.module=SimpleNamespace(appear=lambda button,offset,similarity:
            button.match(self.frame,offset=offset,similarity=similarity))

    def observe(self,second,frame=None,**changes):
        self.clock[0]=second
        self.frame=self.frames['battle'] if frame is None else frame
        return observe_native_crown_return(self.frame,self.st,self.module,self.world,
            preview=changes.get('preview',False),interruption=changes.get('interruption'),
            stop=self.stop,clock=lambda:self.clock_object)

    def test_three_fresh_frames_and_three_seconds_confirm_without_inputs_or_records(self):
        before=deepcopy(self.world.journal),deepcopy(self.world.training.state)
        self.assertFalse(self.observe(1))
        self.assertFalse(self.observe(2))
        self.assertTrue(self.observe(4))
        self.assertEqual(self.st.stage_deadline,15)
        self.assertEqual((self.world.journal,self.world.training.state),before)
        self.assertFalse(self.journal.exists())
        self.assertEqual(self.inputs,[])

    def test_fast_frames_do_not_substitute_for_three_seconds(self):
        for second in (1,1.1,1.2,1.3,1.4):self.assertFalse(self.observe(second))
        self.assertTrue(self.observe(4))

    def test_combat_stage_pause_and_general_vote_reset_do_not_erase_return_proof(self):
        for second in (1,2):
            self.assertFalse(self.observe(second))
            self.st.stable={key:0 for key in self.st.matchers}
            self.st.stage_started_at+=.5
            self.st.stage_deadline+=.5
        self.assertTrue(self.observe(4))

    def test_real_flying_rewards_and_menu_occlusion_still_reject(self):
        for frame in (self.frames['occluded'],self.frames['menu'],self.frames['listing'],
                      self.frames['completion'],self.frames['battle']//2,
                      np.zeros_like(self.frames['battle'])):
            with self.subTest(shape=frame.shape):
                for second in (1,2,4):self.assertFalse(self.observe(second,frame))
                self.assertEqual(self.st.crown_return_frames,0)

    def test_occlusion_restarts_proof_and_unobscured_activity_variants_can_confirm(self):
        self.assertFalse(self.observe(1))
        self.assertFalse(self.observe(2,self.frames['occluded']))
        self.assertFalse(self.observe(3,self.frames['filled']))
        self.assertFalse(self.observe(4,self.frames['success']))
        self.assertTrue(self.observe(6,self.frames['battle']))

    def test_changed_location_restarts_proof(self):
        changed=self.frames['battle'].copy();changed[12:34,286:434]=255
        self.assertFalse(self.observe(1))
        self.assertFalse(self.observe(2))
        self.assertFalse(self.observe(4,changed))
        self.assertFalse(self.observe(5))
        self.assertFalse(self.observe(6))
        self.assertTrue(self.observe(8))

    def test_original_deadline_stop_preview_and_interrupt_cannot_finish(self):
        for mode in ('expired','stop','preview','interrupt','handling','unrelated'):
            with self.subTest(mode=mode):
                self.setUp()
                self.assertFalse(self.observe(1));self.assertFalse(self.observe(2))
                kwargs={}
                if mode=='stop':self.stop.set()
                if mode=='preview':kwargs['preview']=True
                if mode=='interrupt':kwargs['interruption']='task_popup'
                if mode=='handling':self.st.handling_interrupt=True
                if mode=='unrelated':self.st.stage_trigger='Unrelated operation'
                self.assertFalse(self.observe(15 if mode=='expired' else 4,**kwargs))

    def test_expiry_or_stop_during_current_context_read_cannot_finish(self):
        for mode in ('expired','stop'):
            with self.subTest(mode=mode):
                self.setUp();self.assertFalse(self.observe(1));self.assertFalse(self.observe(2))
                original=self.world.native_entry_recovery_context
                def context(frame):
                    result=original(frame)
                    if mode=='expired':self.clock[0]=15
                    else:self.stop.set()
                    return result
                with patch.object(self.world,'native_entry_recovery_context',side_effect=context):
                    self.assertFalse(self.observe(4))

    def test_disabled_stone_or_active_portal_never_completes_crown_return(self):
        for mode in ('disabled','pending'):
            with self.subTest(mode=mode):
                self.setUp()
                if mode=='disabled':self.world.enabled=lambda:False
                else:self.world.journal['pending']=True
                for second in (1,2,4):self.assertFalse(self.observe(second))

    def test_actual_fixture_bytes_preserved(self):
        manifest=json.loads((FIXTURES/'manifest.json').read_text(encoding='utf-8'))
        for name,meta in manifest.items():
            self.assertEqual(hashlib.sha256((FIXTURES/name).read_bytes()).hexdigest(),meta['sha256'])

    def test_actual_final_boss_retains_priority_without_numeric_reads(self):
        with patch.object(self.world.defense,'encounter',return_value=True), \
             patch.object(self.world,'native_entry_recovery_context',side_effect=AssertionError('No OCR before final boss defense')):
            for second in (1,2,4):self.assertFalse(self.observe(second))

    def test_disabled_return_rule_never_confirms(self):
        self.st.matchers['driftinggirl_achievements_return'][0]['enabled']=False
        for second in (1,2,4):self.assertFalse(self.observe(second))

    def replay(self,legacy=False):
        config=deepcopy(self.config);config['poll_interval']=.1
        ids=set(FEATURE_BY_ID['achievements']['rules'])|{'driftinggirl_stone_farm'}
        config['rules']=[r for r in config['rules'] if r['id'] in ids]
        for rule in config['rules']:rule.update(enabled=True,interval=.2,next_delay=0)
        clock,devices,world,frames=[0.],[],self.world,self.frames
        class Device:
            def __init__(self,_cfg,stop):
                self.stop,self.image,self.taps,self.after=stop,None,[],0;devices.append(self)
            def screenshot(self):
                clock[0]+=1
                count=len(self.taps)
                name=('main','menu','listing','menu','battle')[min(count,4)]
                if count>=4:
                    self.after+=1
                    if self.after>12:raise InterruptedError('End of bounded battle replay')
                self.image=frames[name].copy();return self.image
            def click(self,button):
                self.taps.append(button.name)
            def stuck_record_add(self,_button):pass
        class Defense:
            def encounter(self,*_a):return False
            def encounter_rank(self,frame):return world.defense.encounter_rank(frame)
            def owns(self,*_a):return False
            def step(self,*_a):return len(devices[0].taps)>=4
            def status(self):return {}
        class Native:
            phase,active='idle',False
            def __init__(self,*_a,**_kw):self.defense=Defense()
            def retire_expired_on_start(self):return False
            def enabled(self):return True
            def step(self,_frame):return False
            def status(self):return dict(phase='idle',pending=False)
            def native_entry_recovery_context(self,frame):return world.native_entry_recovery_context(frame)
        root=Path(self.tmp.name)/('baseline' if legacy else 'fixed');root.mkdir()
        assets={r['asset'] for r in config['rules']}
        assets.update(a for r in config['rules'] for a in r.get('asset_variants',[]))
        for path in assets:
            (root/path).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,root/path)
        (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
        clicker=Clicker(root,Device);clicker.digit_reader=self.reader
        try:
            real_observer=observe_native_crown_return
            with patch('app.StoneFarm',Native),patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])), \
                 patch('drift.runtime.observe_native_crown_return',
                       side_effect=(lambda *a,**kw:False) if legacy else real_observer):
                clicker.start(preview=False,stop_on_main=True,max_clicks=8,timeout=120)
                clicker.thread.join(40)
                self.assertFalse(clicker.running(),'Replay failed to stop')
            return deepcopy(clicker.state),devices[0].taps,list(clicker.events)
        finally:
            clicker.stop()
            if clicker.thread:clicker.thread.join(2)
            for handler in list(clicker.log.handlers):handler.close();clicker.log.removeHandler(handler)

    def test_runtime_finishes_read_only_return_even_when_combat_owns_every_frame(self):
        old,old_taps,_=self.replay(legacy=True)
        new,new_taps,events=self.replay()
        self.assertNotEqual(old['stop_reason'],'cycle_complete')
        self.assertEqual(new['stop_reason'],'cycle_complete')
        self.assertIsNone(new['last_error'])
        self.assertEqual(old_taps,new_taps)
        self.assertEqual(len(new_taps),4)
        self.assertTrue(any('战斗接管前连续三帧三秒' in e['message'] for e in events))


if __name__=='__main__':unittest.main()
