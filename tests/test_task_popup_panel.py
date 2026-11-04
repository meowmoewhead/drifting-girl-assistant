from copy import deepcopy
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest

import cv2
from app import Clicker
from drift.core.matcher import observe_popup_equipment_return
from drift.core.session import LoopState
from drift.tasks.equipment_upgrade import EquipmentUpgrade

ROOT=Path(__file__).resolve().parents[1]
def image(name):
    return cv2.cvtColor(cv2.imread(str(ROOT/name)),cv2.COLOR_BGR2RGB)

class PopupPanelTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        self.rule=next(r for r in self.config['rules'] if r['id']=='driftinggirl_task_popup_close')
        self.context=SimpleNamespace(root=ROOT,config=self.config)
        self.st=LoopState(stage='task_popup_return',stage_ready_at=0,stage_deadline=15,
                          stage_trigger=self.rule['name'],handling_interrupt=True)
        self.st.matchers[self.rule['id']]=(self.rule,None,None)
        self.inputs=[]
        self.equipment=EquipmentUpgrade(ROOT,lambda *a:self.inputs.append(a),
            lambda *a:self.inputs.append(a),lambda *a:self.inputs.append(a),lambda _:None)
        self.equipment.phase='upgrade'
        self.stop=threading.Event();self.now=0
        self.clock=lambda:SimpleNamespace(monotonic=lambda:self.now)
        self.panel=image('tests/fixtures/equipment/top.png')

    def matches(self,frame):
        _,button,offset=Clicker.load_rule(self.context,self.rule)
        return button.match(frame,offset=offset,similarity=self.rule['threshold'])

    def observe(self,at,frame=None,**kwargs):
        self.now=at
        return observe_popup_equipment_return(self.panel if frame is None else frame,self.st,self.equipment,
            notice_gone=kwargs.get('notice_gone',True),stop=self.stop,clock=self.clock)

    def test_real_kraken_notice_uses_full_existing_guards_without_config_write(self):
        before=deepcopy(self.config)
        frame=image('assets/task-popup/kraken-stage3.png')
        self.assertTrue(self.matches(frame))
        self.assertEqual(self.config,before)
        for x,y,X,Y in ((618,406,674,453),(315,465,350,492),(106,785,162,810)):
            damaged=frame.copy();damaged[y:Y,x:X]=0
            self.assertFalse(self.matches(damaged))
        self.assertFalse(self.matches(self.panel))
        self.assertFalse(self.matches(image('tests/fixtures/stone-first-combat/lowered-battle.png')))

    def test_original_notice_still_matches(self):
        self.assertTrue(self.matches(image('assets/task-popup/notice.png')))

    def test_three_frames_three_seconds_resume_only_the_existing_panel(self):
        before=self.equipment.status().copy()
        self.assertFalse(self.observe(1))
        self.assertFalse(self.observe(2))
        self.assertTrue(self.observe(4))
        self.assertEqual(self.inputs,[])
        self.assertEqual(self.st.stage_deadline,15)
        self.assertEqual(self.equipment.status(),before)

    def test_fast_frames_and_visible_notice_cannot_resume(self):
        for at in (1,1.1,1.2,1.3):self.assertFalse(self.observe(at))
        self.assertFalse(self.observe(4,notice_gone=False))
        self.assertFalse(self.observe(5))
        self.assertFalse(self.observe(6))
        self.assertTrue(self.observe(8))

    def test_a_new_popup_cannot_reuse_the_previous_return_votes(self):
        self.assertFalse(self.observe(1));self.assertFalse(self.observe(2))
        self.assertTrue(self.observe(4))
        self.st.stage_started_at=5
        self.assertFalse(self.observe(6))
        self.assertFalse(self.observe(7))
        self.assertTrue(self.observe(9))

    def test_real_popup_or_unknown_and_dim_frames_break_panel_proof(self):
        for frame in (image('assets/task-popup/kraken-stage3.png'),self.panel//2,
                      image('assets/task-popup/main.png')):
            self.assertFalse(self.observe(1))
            self.assertFalse(self.observe(2,frame))
            self.assertFalse(self.observe(4))
            self.st.popup_panel_context=None
        self.assertEqual(self.inputs,[])

    def test_verified_existing_panel_can_be_adopted_after_worker_restart(self):
        self.equipment.phase='idle'
        self.assertFalse(self.observe(1));self.assertFalse(self.observe(2))
        self.assertTrue(self.observe(4))
        self.assertEqual(self.equipment.phase,'idle')
        self.assertEqual(self.inputs,[])

    def test_expiry_stop_unrelated_flow_and_absent_equipment_are_rejected(self):
        for mode in ('expiry','stop','unrelated','absent','not_interrupted'):
            self.setUp()
            self.assertFalse(self.observe(1));self.assertFalse(self.observe(2))
            if mode=='stop':self.stop.set()
            if mode=='unrelated':self.st.stage_trigger='Unrelated'
            if mode=='absent':self.equipment=None
            if mode=='not_interrupted':self.st.handling_interrupt=False
            self.assertFalse(self.observe(15 if mode=='expiry' else 4))
            self.assertEqual(self.inputs,[])
