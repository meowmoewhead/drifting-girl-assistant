"""Dormant native cruise must start actual fishing once, preserving route options."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from digit_ocr import DigitReader, parse_counter, read_daily_counter
from drift.tasks.cruise_settings import CruiseSettings, SettingsView
from drift.tasks.stone_farm import StoneFarm
from tests.test_cruise_preferences import Harness

ROOT=Path(__file__).resolve().parents[1]
FIXTURES=ROOT/'tests/fixtures/cruise-start'


class FishingHarness(Harness):
    def __init__(self,root,route=True):
        super().__init__(root,route)
        self.scope='original-r51-place';self.asleep=True;self.active=False;self.start_works=True
        self.valid_range=True;self.events=[]
        self.view.fishing_scope=lambda f,c:self.scope
        self.view.fishing_idle=lambda f,c:(self.scope,'sleep') if self.asleep and not self.menu else None
        self.view.fishing_evidence=lambda f,c:self.active and not self.menu
        self.view.fishing_guard=lambda f:{'fresh':'digits'}
        self.view.owner.fishing=SimpleNamespace(measure=lambda f,remember=False:
            dict(value=5031,low=5031,high=5915,player=5826) if self.valid_range else None)
        self.controller.event=self.events.append
        self.options['ticket_min']=0
        self.values=SettingsView.wanted(self.options)

    def click(self,kind,x,y,ms):
        if kind!='fish-start':return super().click(kind,x,y,ms)
        if not self.controller.reserve(kind):return False
        self.receipts.append((kind,ms))
        if self.start_works:self.asleep=False;self.active=True
        return True


class StartupWorkflowTests(unittest.TestCase):
    def test_unchanged_selection_can_start_with_covered_limits(self):
        with TemporaryDirectory() as root:
            h=FishingHarness(root)
            h.view.start_reading=lambda f:dict(value=10,player=15,low=None,high=None,unchanged_selection=True)
            h.tick(65)
            self.assertTrue(h.route)
            self.assertEqual(sum(k=='fish-start' for k,_ in h.receipts),1)
            self.assertTrue(h.controller.journal['last_result']['success'])

    def test_known_conflicting_limit_still_rejects_start(self):
        with TemporaryDirectory() as root:
            h=FishingHarness(root)
            h.view.start_reading=lambda f:dict(value=10,player=15,low=11,high=None)
            h.tick(200)
            self.assertNotIn('fish-start',[k for k,_ in h.receipts])

    def test_legacy_block_with_no_start_input_is_archived_not_treated_as_spent_hold(self):
        import hashlib
        with TemporaryDirectory() as root:
            h=FishingHarness(root);h.valid_range=False
            while not h.controller.journal.get('last_result'):h.tick()
            old=json.loads(json.dumps(h.controller.journal['last_result']))
            h.controller.journal['fishing_blocked_scope']=h.scope;h.controller._save()
            raw=h.controller.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            h.controller=CruiseSettings(root=root,view=h.view,options=h.options,click=h.click,
                event=h.events.append,interval=300,clock=lambda:h.now)
            self.assertNotIn('fishing_blocked_scope',h.controller.journal)
            self.assertEqual(h.controller.journal['last_result'],old)
            self.assertEqual(Path(root,'logs/cruise-settings-retired',digest+'.json').read_bytes(),raw)
            h.valid_range=True;h.tick(65)
            self.assertEqual(sum(k=='fish-start' for k,_ in h.receipts),1)

    def test_earlier_spent_hold_or_truncated_history_never_unlocks(self):
        for scenario in ('spent','oversized'):
            with self.subTest(scenario=scenario),TemporaryDirectory() as root:
                h=FishingHarness(root);h.valid_range=False
                while not h.controller.journal.get('last_result'):h.tick()
                h.controller.journal['fishing_blocked_scope']=h.scope;h.controller._save()
                history=h.controller.path.with_suffix('.history.jsonl')
                with history.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(fishing_start_scope=h.scope,attempts={'fish-start':1}))+'\n'
                                 if scenario=='spent' else ' '*1048577)
                h.controller=CruiseSettings(root=root,view=h.view,options=h.options,click=h.click,
                    event=h.events.append,interval=300,clock=lambda:h.now)
                self.assertEqual(h.controller.journal['fishing_blocked_scope'],h.scope)
                self.assertNotIn('unspent_start_release',h.controller.journal)

    def test_running_sleep_stops_once_starts_once_and_restores_saved_route(self):
        with TemporaryDirectory() as root:
            h=FishingHarness(root);h.tick(65)
            self.assertEqual(h.receipts,[('stop',40),('fish-start',1500),('open',40),('start',40)])
            self.assertTrue(h.route)
            self.assertTrue(h.controller.journal['last_result']['success'])
            self.assertTrue(h.controller.journal['last_result']['fishing_start_verified'])
            self.assertEqual(h.controller.journal['fishing_start_reading'],
                dict(player=5826,value=5031,low=5031,high=5915))
            self.assertTrue(any('未调整标尺' in e for e in h.events))
            h.tick(100)
            self.assertEqual(sum(k=='fish-start' for k,_ in h.receipts),1)

    def test_existing_fishing_never_held(self):
        with TemporaryDirectory() as root:
            h=FishingHarness(root);h.asleep=False;h.active=True;h.tick(65)
            self.assertNotIn('fish-start',[k for k,_ in h.receipts])

    def test_unknown_range_never_held_and_does_not_extend_deadline(self):
        with TemporaryDirectory() as root:
            h=FishingHarness(root);h.valid_range=False;h.tick(200)
            self.assertNotIn('fish-start',[k for k,_ in h.receipts])
            result=h.controller.journal['last_result']
            self.assertFalse(result['success'])
            self.assertEqual(result['deadline']-result['started_at'],180)

    def test_duplicate_frames_and_under_three_seconds_do_not_prove_sleep(self):
        with TemporaryDirectory() as root:
            h=FishingHarness(root);frame=object()
            for _ in range(20):
                h.controller.step(frame);h.now+=.1
            self.assertFalse(h.controller.journal.get('fishing_start_requested'))
            for _ in range(3):h.controller.step(object());h.now+=.1
            self.assertFalse(h.controller.journal.get('fishing_start_requested'))

    def test_unknown_frame_breaks_consecutive_proof(self):
        with TemporaryDirectory() as root:
            h=FishingHarness(root);h.tick(2);h.known=False;h.tick()
            h.known=True;h.tick(2)
            self.assertFalse(h.controller.journal.get('fishing_start_requested'))

    def test_failed_hold_is_not_repeated_by_periodic_setting_audits(self):
        with TemporaryDirectory() as root:
            h=FishingHarness(root);h.start_works=False;h.tick(500)
            self.assertEqual(sum(k=='fish-start' for k,_ in h.receipts),1)
            self.assertEqual(h.controller.journal['fishing_blocked_scope'],h.scope)
            history=Path(root,'cruise_settings_state.history.jsonl').read_text(encoding='utf-8')
            self.assertIn('不重复长按',history)

    def test_unknown_after_hold_keeps_thirty_second_verification_limit(self):
        with TemporaryDirectory() as root:
            h=FishingHarness(root);h.start_works=False
            for _ in range(30):
                h.tick()
                if any(k=='fish-start' for k,_ in h.receipts):break
            h.known=False;h.tick(31)
            self.assertFalse(h.controller.busy)
            self.assertIn('未确认实际钓鱼',h.controller.journal['last_result']['reason'])

    def test_late_actual_fishing_restores_route_without_replaying_spent_hold(self):
        with TemporaryDirectory() as root:
            h=FishingHarness(root);h.start_works=False;h.tick(55)
            old=dict(h.controller.journal['last_result'])
            self.assertFalse(old['success'])
            h.asleep=False;h.active=True;h.tick(35)
            self.assertTrue(h.route)
            self.assertEqual(sum(k=='fish-start' for k,_ in h.receipts),1)
            self.assertIn('fishing_late_observed_at',h.controller.journal)
            rows=[json.loads(line) for line in Path(root,'cruise_settings_state.history.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(rows[0],json.loads(json.dumps(old)))
            self.assertEqual(rows[0]['attempts']['fish-start'],1)

    def test_late_battle_after_normal_audit_still_gets_read_only_confirmation(self):
        with TemporaryDirectory() as root:
            h=FishingHarness(root);h.start_works=False;h.tick(500)
            self.assertFalse(h.controller.journal['fishing_start_requested'])
            self.assertTrue(h.controller.journal['fishing_blocked_scope'])
            spent=list(h.receipts);h.asleep=False;h.active=True
            h.view.observe_setup=lambda f:None
            h.view.observe_fishing=lambda f,scene:dict(scene=scene,menu=False)
            h.view.safe_main=lambda f:False
            h.tick(5)
            self.assertNotIn('fishing_blocked_scope',h.controller.journal)
            self.assertEqual(spent,h.receipts)

    def test_place_change_during_owned_start_prevents_input(self):
        with TemporaryDirectory() as root:
            h=FishingHarness(root)
            while h.controller.state!='fish_idle':h.tick()
            h.scope='different-place';h.tick()
            self.assertFalse(h.controller.busy)
            self.assertNotIn('fish-start',[k for k,_ in h.receipts])

    def test_restart_archives_spent_start_and_blocks_same_sleep_episode(self):
        with TemporaryDirectory() as root:
            h=FishingHarness(root);h.start_works=False
            while h.controller.state!='fish_verify':h.tick()
            old=dict(h.controller.journal);old_deadline=old['deadline']
            h.controller=CruiseSettings(root=root,view=h.view,options=h.options,click=h.click,
                event=h.events.append,interval=300,clock=lambda:h.now)
            self.assertEqual(h.controller.journal['last_result']['deadline'],old_deadline)
            self.assertEqual(h.controller.journal['last_result']['attempts']['fish-start'],1)
            h.tick(500)
            self.assertEqual(sum(k=='fish-start' for k,_ in h.receipts),1)


class ActualFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.reader=DigitReader(ROOT)

    def setUp(self):
        self.owner=StoneFarm(ROOT,self.reader,lambda a:False,lambda s:None,
            state_path=ROOT/'tests/nonexistent-cruise-start.json')
        self.owner.enabled=lambda:True
        self.view=SettingsView(self.owner,ROOT,self.reader)
        self.frame=np.array(Image.open(FIXTURES/'original-sleep-r51.png').convert('RGB'))

    def test_actual_sleep_and_three_counter_frames(self):
        context=self.view.observe_setup(self.frame)
        self.assertEqual(context['scene'][0],'original')
        self.assertIsNotNone(self.view.fishing_idle(self.frame,context))
        self.assertFalse(self.view.fishing_evidence(self.frame,context))
        self.assertTrue(self.owner.sailing.running(self.frame))
        data=self.owner.fishing.measure(self.frame,remember=False)
        self.assertEqual([data[k] for k in ('player','value','low','high')],[5826,5031,5031,5915])
        for path in sorted(FIXTURES.glob('potion-3-*.png')):
            crop=np.array(Image.open(path).convert('RGB'))
            result=read_daily_counter(self.reader,crop,'white_text_consensus')
            self.assertEqual((result.current,result.total),(3,40),result)
            self.assertGreaterEqual(result.confidence,.93)

    def test_no_destination_restores_route_before_attempting_fishing(self):
        frame=np.array(Image.open(FIXTURES/'parallel-no-destination-r5.png').convert('RGB'))
        context=self.view.observe_setup(frame)
        self.assertEqual(context['scene'],(10,5))
        self.assertTrue(self.owner._no_destination(frame))
        self.assertIsNone(self.view.fishing_idle(frame,context))
        self.assertFalse(self.view.fishing_evidence(frame,context))
        self.assertFalse(self.view.validate_action(frame,context,'fish-start',{}))

    def test_actual_r5_hook_starts_existing_value_without_guessing_obscured_range(self):
        frame=np.array(Image.open(FIXTURES/'parallel-r5-hidden-range.png').convert('RGB'))
        context=self.view.observe_setup(frame)
        self.assertEqual(context['scene'],(10,5))
        self.assertIsNone(self.owner.fishing.measure(frame,remember=False))
        data=self.view.start_reading(frame)
        self.assertEqual(data,dict(value=10,player=15,low=None,high=None,unchanged_selection=True))
        context['fishing_guard']=self.view.fishing_guard(frame)
        self.assertTrue(self.view.validate_action(frame,context,'fish-start',{}))
        with patch.object(self.owner.fishing,'selected',return_value=None):
            self.assertIsNone(self.view.start_reading(frame))
        with patch.object(self.owner.training,'boss_mode',return_value=True):
            self.assertIsNone(self.view.start_reading(frame))

    def test_fresh_hook_guard_uses_new_strokes_without_slow_ocr(self):
        frame=np.array(Image.open(FIXTURES/'parallel-r5-hidden-range.png').convert('RGB'))
        context=self.view.observe_setup(frame)
        context['fishing_guard']=self.view.fishing_guard(frame)
        with patch.object(self.view,'observe_setup',side_effect=AssertionError('OCR in fast guard')), \
             patch.object(self.reader,'read',side_effect=AssertionError('OCR in fast guard')):
            self.assertTrue(self.view.validate_action(frame.copy(),context,'fish-start',{}))
            for box in ((71,9,136,59),(40,45,68,71),(286,12,434,34),(240,915,281,956),(76,1248,137,1278)):
                changed=frame.copy();x,y,X,Y=box;changed[y:Y,x:X]=255
                with self.subTest(box=box):
                    self.assertFalse(self.view.validate_action(changed,context,'fish-start',{}))

    def test_fresh_start_rejects_running_route_or_numeric_change(self):
        context=self.view.observe_setup(self.frame)
        context['fishing_guard']=self.view.fishing_guard(self.frame)
        options={'targets':{}}
        self.assertFalse(self.view.validate_action(self.frame,context,'fish-start',options))
        with patch.object(self.owner,'_off_icon',return_value=True), \
             patch.object(self.owner.sailing,'running',return_value=False):
            self.assertTrue(self.view.validate_action(self.frame,context,'fish-start',options))
            changed=self.frame.copy();changed[1070:1100,234:284]=255
            self.assertFalse(self.view.validate_action(changed,context,'fish-start',options))

    def test_padded_counter_cannot_override_conflict_or_lower_confidence(self):
        crop=np.array(Image.open(FIXTURES/'potion-3-0.png').convert('RGB'))
        reader=Mock()
        reader.read_counter.side_effect=lambda im:parse_counter(
            '4/40' if im.shape[0]>20 else '3/40',
            .99 if im.shape[0]>20 or (im.shape==crop.shape and np.array_equal(im,crop)) else .80)
        result=read_daily_counter(reader,crop,'white_text_consensus')
        self.assertIsNone(result.current)
        self.assertIn('不一致',result.reason)
        result=read_daily_counter(self.reader,crop,'white_text_consensus',.995)
        self.assertIsNone(result.current)

    def test_task_port_motion_and_unknown_do_not_authorize_hold(self):
        context=self.view.observe_setup(self.frame)
        for owner,name in ((self.owner.training,'boss_mode'),(self.owner.fishing,'at_port'),
                           (self.owner,'_moving'),(self.owner.fishing,'busy')):
            with self.subTest(name=name),patch.object(owner,name,return_value=True):
                self.assertIsNone(self.view.fishing_idle(self.frame,context))
        self.assertIsNone(self.view.observe_setup(self.frame//2))

    def test_owed_lowering_never_starts_an_original_world_farming_route(self):
        self.owner.journal.update(reason='lower',retirement=dict(released=False))
        self.assertIsNone(self.view.observe_setup(self.frame))
        self.owner.journal['retirement']['released']=True
        self.assertIsNotNone(self.view.observe_setup(self.frame))

    def test_battle_read_only_observation_keeps_menu_input_forbidden(self):
        context=self.view.observe_setup(self.frame)
        with patch.object(self.owner,'_off_icon',return_value=True), \
             patch.object(self.owner,'_battle',return_value=True):
            self.assertIsNone(self.view.observe_setup(self.frame))
            self.assertIsNotNone(self.view.observe_fishing(self.frame,context['scene']))
            self.assertFalse(self.view.validate_action(self.frame,context,'open',{}))

    def test_actual_whale_combat_has_no_hp_sword_but_all_other_guards(self):
        context=self.view.observe_setup(self.frame)
        battle=np.array(Image.open(FIXTURES/'original-whale-battle.png').convert('RGB'))
        self.assertFalse(self.owner.fishing.real_battle(battle))
        observed=self.view.observe_fishing(battle,context['scene'])
        self.assertIsNotNone(observed)
        self.assertTrue(self.view.fishing_evidence(battle,observed))
        for x,y,X,Y in ((113,153,187,208),(188,125,610,165),(307,962,415,1057),
                        (208,1137,269,1192),(539,1137,600,1186)):
            changed=battle.copy();changed[y:Y,x:X]=0
            with self.subTest(box=(x,y,X,Y)):
                self.assertFalse(self.view.fishing_evidence(changed,observed))
        self.assertFalse(self.view.fishing_evidence(battle,dict(scene=(10,39),menu=False)))
        self.assertFalse(self.view.validate_action(battle,context,'open',{}))


if __name__=='__main__':unittest.main()
