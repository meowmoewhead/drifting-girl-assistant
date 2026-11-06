from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from drift.tasks.cruise_options import normalize
from drift.tasks.cruise_settings import CruiseSettings, SettingsView
from drift.tasks.stone_farm import StoneFarm
from drift.tasks.parallel_skills import ParallelSkills
from tests.test_features import FeatureTests
from digit_ocr import DigitReader

ROOT=Path(__file__).resolve().parents[1]


class OptionsTests(unittest.TestCase):
    def test_independent_options_and_invalid_spending_input(self):
        value=normalize()
        value['skills']['movement']={'auto':True,'enhance':False}
        value['skills']['time']={'auto':False,'enhance':True}
        value['cruise']['auto_sale']=False
        self.assertEqual(normalize(value),value)
        for bad in (True,-1,1.5,10000,'20'):
            with self.assertRaises(ValueError):normalize({'cruise':{'ticket_min':bad}})
        with self.assertRaises(ValueError):normalize({'skills':{'diamonds':True}})
        with self.assertRaises(ValueError):normalize({'cruise':{'targets':dict.fromkeys(value['cruise']['targets'],False)}})


class HotApiTests(FeatureTests):
    def test_hot_save_preserves_other_config(self):
        self.feature('parallel_skills')
        other=self.rule();before=deepcopy(other)
        with patch.object(self.app,'running',return_value=True):
            result=self.app.change_feature({'id':'parallel_skills','cruise_settings':normalize(),'interval':100})
        self.assertEqual(result['cruise_settings'],normalize())
        self.assertEqual(self.app.config['rules'][-1],before)
        self.assertEqual(self.app.config_revision,1)
        saved=json.loads((self.root/'config.json').read_text(encoding='utf-8'))
        self.assertEqual(saved,self.app.config)


class Harness:
    def __init__(self,root,route=True):
        self.now=100.;self.route=route;self.menu=False;self.scene=(8,30)
        self.options=normalize()['cruise']
        self.options['targets']['material']=True
        self.options['auto_sale']=True
        self.options['auto_upgrade']=False
        self.values={'targets':dict.fromkeys(self.options['targets'],False),'auto_sale':False,'auto_upgrade':True}
        self.count=1;self.known=True;self.receipts=[]
        self.view=SimpleNamespace(owner=SimpleNamespace(sailing=SimpleNamespace(running=lambda f:None if self.menu else self.route)),
            observe_setup=lambda f:dict(scene=self.scene,menu=self.menu) if self.known else None,
            safe_main=lambda f:True,values=lambda f:deepcopy(self.values),
            tickets=lambda f:self.count,ticket_ink=lambda f:b'current',wanted=SettingsView.wanted)
        self.controller=CruiseSettings(root=root,view=self.view,options=self.options,click=self.click,
                                       event=lambda s:None,interval=300,clock=lambda:self.now)
    def click(self,kind,x,y,ms):
        if not self.controller.reserve(kind):return False
        self.receipts.append((kind,ms))
        if kind=='stop':self.route=False
        elif kind=='open':self.menu=True
        elif kind=='start':self.route=True;self.menu=False
        elif kind=='close':self.menu=False
        elif kind=='tickets':self.count+=5
        elif kind in self.values:self.values[kind]=not self.values[kind]
        else:self.values['targets'][kind]=not self.values['targets'][kind]
        return True
    def tick(self,n=1):
        for _ in range(n):
            self.now+=1
            self.controller.step(object())


class WorkflowTests(unittest.TestCase):
    def test_running_route_reconfigured_and_ticket_hold_confirmed(self):
        with TemporaryDirectory() as root:
            h=Harness(root);h.tick(110)
            self.assertTrue(h.route)
            self.assertEqual(h.values,SettingsView.wanted(h.options))
            self.assertEqual([ms for k,ms in h.receipts if k=='tickets'],[2000,2000])
            self.assertEqual(h.count,11)
            self.assertTrue(h.controller.journal['last_result']['success'])
            before=list(h.receipts);h.tick(50)
            self.assertEqual(before,h.receipts)
            restarted=CruiseSettings(root=root,view=h.view,options=h.options,click=h.click,
                                     event=lambda s:None,clock=lambda:h.now,interval=300)
            restarted.step(object());self.assertEqual(before,h.receipts)

    def test_no_sale_no_purchase_and_can_turn_upgrade_off(self):
        with TemporaryDirectory() as root:
            h=Harness(root,False);h.options['auto_sale']=False;h.tick(100)
            self.assertNotIn('tickets',[k for k,_ in h.receipts])
            self.assertFalse(h.values['auto_upgrade'])
            self.assertTrue(h.controller.journal['last_result']['success'])

    def test_unknown_scene_deadline_and_spent_counts_preserved(self):
        with TemporaryDirectory() as root:
            h=Harness(root);h.tick(6);before=list(h.receipts)
            h.known=False;h.tick(180)
            self.assertEqual(before,h.receipts)
            self.assertFalse(h.controller.journal['last_result']['success'])
            self.assertEqual(h.controller.journal['last_result']['attempts']['stop'],1)

    def test_world_change_never_gets_old_input(self):
        with TemporaryDirectory() as root:
            h=Harness(root);h.tick(6);before=list(h.receipts)
            h.scene=(9,31);h.tick()
            self.assertEqual(before,h.receipts)
            self.assertFalse(h.controller.busy)


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader=DigitReader(ROOT)
    def setUp(self):
        self.owner=StoneFarm(ROOT,self.reader,lambda a:False,lambda s:None,state_path=ROOT/'tests/nonexistent-cruise-state.json')
        self.owner.enabled=lambda:True
        self.view=SettingsView(self.owner,ROOT,self.reader)

    def image(self,name):return np.asarray(Image.open(ROOT/name).convert('RGB'))

    def test_ticket_real_21_and_new_guard(self):
        frame=self.image('tests/fixtures/native-auto-sale/sale-on.png')
        self.assertEqual(self.view.tickets(frame),21)
        options=normalize()['cruise'];options['ticket_min']=25
        context=self.view.observe_setup(frame)
        self.assertIsNotNone(context)
        context['ticket_ink']=self.view.ticket_ink(frame)
        context['ticket_count']=21
        self.assertTrue(self.view.validate_action(frame,context,'tickets',options))
        broken=frame.copy();broken[690:710,485:515]=0
        self.assertFalse(self.view.validate_action(broken,context,'tickets',options))
        self.assertIsNone(self.view.tickets(frame//2))

    def test_ticket_label_moves_with_single_digit_count(self):
        frame=self.image('tests/fixtures/cruise-settings/original-tickets-8.png')
        self.assertEqual(self.view.tickets(frame),8)
        # Bright menus alone do not authorize spending if the plus is hidden.
        broken=frame.copy();broken[676:719,556:597]=0
        self.assertIsNone(self.view.tickets(broken))

    def test_task_skull_allows_navigation_settings_but_actual_battle_does_not(self):
        frame=self.image('tests/fixtures/cruise-settings/original-boss-preparation.png')
        self.assertTrue(self.owner.training.boss_mode(frame))
        self.assertTrue(self.view.safe_main(frame))
        context=self.view.observe_setup(frame)
        self.assertIsNotNone(context)
        self.assertTrue(self.view.validate_action(frame,context,'stop',normalize()['cruise']))
        battle=self.image('tests/fixtures/cruise-settings/original-boss-battle.png')
        self.assertFalse(self.view.safe_main(battle))
        self.assertFalse(self.view.validate_action(battle,context,'stop',normalize()['cruise']))

    def test_original_menu_context_survives_animation_and_restart(self):
        main=self.image('tests/fixtures/cruise-settings/original-boss-preparation.png')
        context=self.view.observe_setup(main)
        self.assertIsNotNone(context)
        battle=self.image('tests/fixtures/cruise-settings/original-boss-battle.png')
        self.assertIsNone(self.view.observe_setup(battle))
        self.assertTrue(self.view.original_seen)
        menu=self.image('tests/fixtures/cruise-settings/original-menu-tickets-13.png')
        self.assertIsNotNone(self.view.observe_setup(menu))
        cold=SettingsView(self.owner,ROOT,self.reader)
        self.assertIsNone(cold.observe_setup(menu))
        cold.restore_context(context['scene'])
        self.assertIsNotNone(cold.observe_setup(menu))
        wrong=menu.copy();wrong[9:59,71:136]=0
        self.assertIsNone(cold.observe_setup(wrong))
        parallel=self.image('tests/fixtures/native-auto-sale/sale-on.png')
        cold.observe_setup(parallel)
        self.assertIsNone(cold.resume_scene)

    def test_six_skill_switches_and_purchase_record_preserved(self):
        frame=self.image('tests/fixtures/parallel-skills/parallel-skill-panel.png')
        with TemporaryDirectory() as root:
            unit=ParallelSkills(ROOT,lambda *a:False,lambda s:None,self.reader,options=normalize()['skills'])
            unit.path=Path(root)/'skill.json';unit.journal={'growth_until':9999999999,'pending':False}
            for row,name in enumerate(('fishing','movement','time'),1):
                for enhance,field in ((True,'enhance'),(False,'auto')):
                    actual=unit.toggle(frame,enhance,row)
                    self.assertIsNotNone(actual)
                    unit.options[name][field]=not actual
                    action=('enhance' if row==1 else 'enhance'+str(row)) if enhance else 'auto'+str(row)
                    self.assertTrue(unit.validate(action,frame,*unit.ACTIONS[action]))
                    unit.options[name][field]=actual
                    self.assertFalse(unit.validate(action,frame,*unit.ACTIONS[action]))
            self.assertFalse(unit._purchase_allowed())
            self.assertEqual(unit.journal,{'growth_until':9999999999,'pending':False})

    def test_stone_custom_choices_do_not_force_sale_on(self):
        owner=self.owner
        frame=self.image('tests/fixtures/native-auto-sale/sale-on.png')
        owner.cruise_policy=normalize()['cruise']
        owner.cruise_policy['auto_sale']=False
        owner.phase='route_menu';owner.off_verified=True
        owner.journal.update(cruise_priority=owner.cruise_priority(frame),pending=True,
                             deadline=9999999999,phase_deadline=9999999999)
        action=dict(kind='cruise-auto-sale',x=576,y=647,signature=owner.signature(frame),phase='route_menu')
        self.assertTrue(owner.validate_portal(action,frame))
        owner.cruise_policy=None;owner.phase='idle'


if __name__=='__main__':unittest.main()
