"""Stage handoff must not discard RANK39 or credit attempted inputs as proof."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from digit_ocr import DigitReader
from drift.tasks import handoff_journal as records
from drift.tasks.stone_farm import StoneFarm
from tools.configure_stone_farm import add
from tests import test_clicker as helpers

ROOT = Path(__file__).resolve().parents[1]


def image(path):
    return cv2.cvtColor(cv2.imread(str(ROOT/path)), cv2.COLOR_BGR2RGB)


class StoneFarmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)
        cls.main = image('tests/fixtures/stone-farm/stage9-rank39-0.png')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'stone_farm_state.json'
        self.actions = []
        self.w = StoneFarm(ROOT, self.reader, self.act, lambda _:None, state_path=self.path)
        self.now = 1000.
        self.clock = patch('drift.tasks.stone_farm.time.monotonic', side_effect=lambda:self.now)
        self.wall = patch('drift.tasks.stone_farm.time.time', side_effect=lambda:self.now)
        self.clock.start()
        self.wall.start()

    def tearDown(self):
        self.wall.stop()
        self.clock.stop()
        self.temp.cleanup()

    def act(self, action):
        self.actions.append(deepcopy(action))
        return True

    def tick(self, frame=None):
        self.now += 1.6
        self.w.next_at = 0
        return self.w.step(self.main if frame is None else frame)

    def begin(self, *, cycle=False, completion=False):
        self.w._begin(dict(stage=8 if cycle else 9, rank=40 if cycle else 39,
                           completion=completion, menu=False, route=True))

    def action(self, kind, point, frame=None):
        frame = self.main if frame is None else frame
        return dict(kind=kind,x=point[0],y=point[1],phase=self.w.phase,
                    signature=self.w.signature(frame))

    def test_real_stage9_rank39_and_native_route(self):
        self.assertEqual(self.w.screen(self.main),dict(stage=9,rank=39,menu=False,completion=False,route=True))
        self.assertEqual(self.w._return_button(self.main),(677,848))

    def test_actual_rank40_notice_can_establish_stage8_without_dark_main_proof(self):
        notice=image('assets/recovery/parallel-complete.png')
        self.assertEqual(self.w._rank_value(notice),40)
        self.assertEqual(self.w._stage(notice),8)
        self.assertTrue(self.w.screen(notice)['completion'])
        self.assertIsNone(self.w.screen(notice//2))

    def test_rank40_notice_requests_existing_consensus_before_completion_input(self):
        notice=image('assets/recovery/parallel-complete.png')
        for _ in range(3):
            self.now+=1.6
            self.assertTrue(self.w.completion_step(notice))
        self.assertTrue(self.w.completed_rank_40)
        self.assertEqual(self.w.journal['reason'],'cycle')
        self.assertEqual([a['kind'] for a in self.actions],['completion'])

    def test_rank39_and_dark_notice_do_not_seize_completion_priority(self):
        self.assertFalse(self.w.completion_step(self.main))
        self.assertFalse(self.w.completion_step(image('assets/recovery/parallel-complete.png')//2))
        self.assertFalse(self.w.active)
        self.assertEqual(self.actions,[])

    def test_pending_lower_handoff_cannot_be_reclassified_by_completion_priority(self):
        self.begin()
        before=deepcopy(self.w.journal)
        self.assertFalse(self.w.completion_step(image('assets/recovery/parallel-complete.png')))
        self.assertEqual(self.w.journal,before)
        self.assertEqual(self.actions,[])

    def test_old_return_coordinate_is_not_current_button(self):
        self.begin()
        self.w._set('return')
        self.w.off_verified = True
        self.w.fishing_stopped = True
        with patch.object(self.w.sailing,'running',return_value=False), patch.object(self.w,'_off_icon',return_value=True), patch.object(self.w,'_idle_fishing',return_value=True):
            self.assertFalse(self.w.validate_portal(self.action('return-request',(673,936)),self.main))
            self.assertTrue(self.w.validate_portal(self.action('return-request',(677,848)),self.main))

    def test_real_rank40_notification_badge_preserves_return_guards(self):
        frame = image('tests/fixtures/stone-farm/stage8-rank40-return-badge.png')
        self.assertEqual(self.w.screen(frame),
                         dict(stage=8,rank=40,menu=False,completion=False,route=False))
        self.assertEqual(self.w._return_button(frame),(677,938))
        self.begin(cycle=True)
        self.w._set('return')
        self.w.off_verified = self.w.fishing_stopped = True
        action = self.action('return-request',(677,938),frame)
        self.assertTrue(self.w.validate_portal(action,frame))
        self.assertFalse(self.w.validate_portal(self.action('return-request',(677,848),frame),frame))
        for attribute in ('off_verified','fishing_stopped'):
            setattr(self.w,attribute,False)
            self.assertFalse(self.w.validate_portal(action,frame),attribute)
            setattr(self.w,attribute,True)
        for probe in ('_battle','_map'):
            with patch.object(self.w,probe,return_value=True):
                self.assertFalse(self.w.validate_portal(action,frame),probe)
        with patch.object(self.w.sailing,'running',return_value=True):
            self.assertFalse(self.w.validate_portal(action,frame))
        for box in ((90,20,100,30),(320,16,330,25),(40,45,68,71)):
            altered = frame.copy()
            x,y,X,Y = box
            altered[y:Y,x:X] = 0
            self.assertFalse(self.w.validate_portal(action,altered),box)
        self.now = self.w.journal['return_deadline']+1
        self.assertFalse(self.w.validate_portal(action,frame))

    def test_return_badge_and_animated_center_cannot_replace_metal_frame(self):
        frame = image('tests/fixtures/stone-farm/stage8-rank40-return-badge.png')
        # Retain the actual notification and swirl while removing the top and
        # left metal frame. Their colours alone must not locate a portal.
        damaged = frame.copy()
        damaged[907:918,643:711] = 0
        damaged[907:970,643:655] = 0
        self.assertIsNone(self.w._return_button(damaged))
        self.assertIsNone(self.w._return_button(np.zeros_like(frame)))

    def test_dark_unknown_or_corrupt_frames_never_trigger(self):
        for frame in (self.main//2, np.zeros_like(self.main), self.main[:900]):
            for _ in range(4):
                self.assertFalse(self.tick(frame))
        self.assertFalse(self.w.active)
        self.assertEqual(self.actions,[])

    def test_only_three_confirmed_seconds_trigger_rank39_handoff(self):
        self.assertTrue(self.tick())
        self.assertTrue(self.tick())
        self.assertFalse(self.w.active)
        self.assertEqual(self.actions,[])
        self.tick()
        self.assertTrue(self.w.active)
        self.assertEqual(self.w.journal['reason'],'lower')
        self.assertEqual(self.w.journal['entry_rank'],39)
        self.assertFalse(self.w.completed_rank_40)

    def test_normal_early_ranks_leave_native_game_alone(self):
        for stage,rank in ((9,0),(9,38),(8,39),(8,12),(10,39),(9,40)):
            with patch.object(self.w,'screen',return_value=dict(stage=stage,rank=rank,menu=False,completion=False,route=True)):
                for _ in range(4):self.assertFalse(self.tick())
        self.assertEqual(self.actions,[])

    def test_rank39_can_never_close_or_activate_gate(self):
        self.begin()
        for phase,kind,point in (('manager','close-gate',(354,739)),('closed','activate',(355,803)),
                                  ('close_confirm','close-confirm',(208,788))):
            self.w._set(phase)
            self.assertFalse(self.w.validate_portal(self.action(kind,point),self.main))

    def test_lowering_is_exactly_9_to_8_on_actual_original_menu(self):
        self.begin()
        self.w._set('manager')
        active9 = image('assets/recovery/parallel-manager.png')
        active8 = image('tests/fixtures/parallel/gate-active-tier8.png')
        self.assertTrue(self.w.validate_portal(self.action('lower',(357,873),active9),active9))
        self.assertFalse(self.w.validate_portal(self.action('lower',(357,873),active8),active8))
        self.assertFalse(self.w.validate_portal(self.action('lower',(357,873),active9),active9//2))

    def test_lower_confirm_requires_actual_target8_prompt(self):
        self.begin()
        self.w._set('lower_confirm')
        question=image('assets/recovery/parallel-lower-question.png')
        self.assertTrue(self.w.validate_portal(self.action('lower-confirm',(208,788),question),question))
        self.assertFalse(self.w.validate_portal(self.action('lower-confirm',(208,788)),self.main))

    def test_lowering_keeps_rank39_and_waits_for_three_actual8_frames(self):
        self.begin()
        self.w._set('lower_verify')
        active8=image('tests/fixtures/parallel/gate-active-tier8.png')
        self.tick(active8)
        self.tick(active8)
        self.assertEqual(self.w.phase,'lower_verify')
        self.tick(active8)
        self.assertEqual(self.w.phase,'panel_close')
        self.assertEqual(self.w.journal['entry_rank'],39)
        self.assertEqual(self.w.journal['verified_gate_stage'],8)
        self.assertEqual(self.w.journal['cycles'],0)

    def test_fresh_changed_rank_or_location_cannot_reuse_action(self):
        self.begin()
        action=self.action('stop',(233,45))
        changed=self.main.copy()
        changed[20:30,90:100]=0
        self.assertFalse(self.w.validate_portal(action,changed))
        changed=self.main.copy()
        changed[16:25,320:330]=0
        self.assertFalse(self.w.validate_portal(action,changed))

    def test_return_rejects_active_route_battle_and_unknown_button(self):
        self.begin()
        self.w._set('return')
        self.w.off_verified=True
        action=self.action('return-request',(677,848))
        self.assertFalse(self.w.validate_portal(action,self.main))
        with patch.object(self.w.sailing,'running',return_value=False), patch.object(self.w,'_off_icon',return_value=True):
            with patch.object(self.w,'_battle',return_value=True):
                self.assertFalse(self.w.validate_portal(action,self.main))
            with patch.object(self.w,'_return_button',return_value=None):
                self.assertFalse(self.w.validate_portal(action,self.main))

    def test_no_return_until_three_fresh_off_frames_after_restart(self):
        self.begin()
        self.w._set('return')
        deadline=self.w.journal['deadline']
        resumed=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        self.assertFalse(resumed.off_verified)
        self.assertEqual(resumed.journal['deadline'],deadline)
        self.assertFalse(resumed.validate_portal(self.action('return-request',(677,848)),self.main))

    def test_success_preparation_and_fill_are_not_idle_return_proof(self):
        self.begin()
        self.w._set('return')
        self.w.off_verified = self.w.fishing_stopped = True
        for name in ('fill-result.png','filled-prebattle.png','night-filled-prebattle.png','fill-hook.png','scan-combat.png'):
            path = ROOT/'tests/fixtures/fishing-meter'/name
            frame=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
            self.assertFalse(self.w._idle_fishing(frame),name)
            with patch.object(self.w,'_parallel',return_value=True), patch.object(self.w,'_off_icon',return_value=True), patch.object(self.w.sailing,'running',return_value=False), patch.object(self.w,'_return_button',return_value=(677,848)):
                self.assertFalse(self.w.validate_portal(self.action('return-request',(677,848),frame),frame),name)

    def test_handoff_probe_does_not_change_saved_calibration_permission(self):
        frame=image('tests/fixtures/fishing-meter/night-filled-prebattle.png')
        self.assertTrue(self.w.fishing.ordinary_preparation(frame,allow_fill=True))
        self.assertFalse(self.w.fishing.cancellable_cast(frame))
        self.assertEqual(self.w.training.state,{})

    def test_cancel_needs_consent_fresh_context_and_never_battle_or_boss(self):
        self.begin()
        self.w._set('stop_fishing',180)
        self.w.off_verified=True
        frame=image('tests/fixtures/fishing-meter/night-filled-prebattle.png')
        context=self.w.cancel_context(frame)
        self.assertIsNotNone(context)
        action=self.action('handoff-cancel',(690,1090),frame)
        action['cancel_context']=context
        self.assertTrue(self.w.validate_portal(action,frame))
        for target in ('_battle',):
            with patch.object(self.w,target,return_value=True):
                self.assertFalse(self.w.validate_portal(action,frame))
        with patch.object(self.w.training,'boss_mode',return_value=True):
            self.assertFalse(self.w.validate_portal(action,frame))
        with patch.object(self.w.sailing,'running',return_value=True):
            self.assertFalse(self.w.validate_portal(action,frame))
        changed=frame.copy();changed[925:945,215:250]=0
        self.assertFalse(self.w.validate_portal(action,changed))
        self.assertFalse(self.w.validate_portal(action,frame[:900]))
        self.w.journal['allow_ordinary_cancel']=False
        self.assertFalse(self.w.validate_portal(action,frame))

    def test_actual_idle_needs_three_seconds_and_never_scans_while_confirming(self):
        self.begin()
        self.w._set('stop_fishing',180)
        self.w.off_verified=True
        screen=dict(stage=9,rank=39,menu=False,completion=False,route=False)
        with patch.object(self.w,'_expected',return_value=screen), patch.object(self.w,'_idle_fishing',return_value=True), patch.object(self.w,'cancel_context') as cancel:
            self.tick();self.tick()
            self.assertFalse(self.w.fishing_stopped)
            self.tick()
            self.assertTrue(self.w.fishing_stopped)
            self.assertEqual(self.w.phase,'return')
            cancel.assert_not_called()
        self.assertEqual(self.actions,[])

    def test_rank40_cancel_uses_same_fresh_ordinary_guard_and_completion_scope(self):
        self.begin(cycle=True)
        self.w._set('stop_fishing',180)
        self.w.off_verified=True
        frame=image('tests/fixtures/fishing-meter/night-filled-prebattle.png')
        # The existing context guard is exercised above; the completed-world
        # scope must independently reject a stale or unverified cycle journal.
        action=self.action('handoff-cancel',(690,1090),frame)
        with patch.object(self.w,'same_cancel_context',return_value=True):
            self.assertTrue(self.w.validate_portal(action,frame))
            with patch.object(self.w.fishing,'ordinary_preparation',return_value=False):
                self.assertFalse(self.w.validate_portal(action,frame))
            for field,value in (('rank40_verified',False),('rank',39),('from_stage',9),('allow_ordinary_cancel',False)):
                original=self.w.journal[field];self.w.journal[field]=value
                self.assertFalse(self.w.validate_portal(action,frame),field)
                self.w.journal[field]=original
            self.w.off_verified=False
            self.assertFalse(self.w.validate_portal(action,frame))

    def test_rank40_expired_stop_retry_archives_original_and_reproves_current_rank(self):
        self.begin(cycle=True);self.w._set('stop_fishing',180)
        self.w.journal['allow_ordinary_cancel']=False  # the previous installed version
        self.w.journal['return_deadline']=self.now-1;self.w._save()
        raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        approval=StoneFarm.request_retry(self.path.parent,digest)
        self.assertEqual(approval['mode'],'return_retry')
        self.assertEqual(self.path.read_bytes(),raw)
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        wrong=dict(stage=8,rank=39,menu=False,completion=False,route=False)
        with patch.object(self.w,'_expected',return_value=None):
            for _ in range(3):self.tick()
        self.assertEqual(self.path.read_bytes(),raw)
        actual=dict(wrong,rank=40)
        with patch.object(self.w,'_expected',return_value=actual):
            self.tick();self.tick()
            self.assertEqual(self.path.read_bytes(),raw)
            self.tick()
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertEqual(self.w.journal['previous_failed_sha256'],digest)
        self.assertTrue(self.w.ordinary_cancel_allowed)
        self.assertFalse(self.w.off_verified)
        self.assertAlmostEqual(self.w.journal['return_deadline']-self.w.journal['started_at'],180)
        self.assertEqual(self.actions,[])

    def test_rank40_retry_rejects_unverified_or_wrong_stage_cycle(self):
        self.begin(cycle=True);self.w._set('stop_fishing',180)
        self.w.journal['return_deadline']=self.now-1
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        for field,value in (('rank40_verified',False),('rank',39),('from_stage',9),('to_stage',8),('entry_rank',39)):
            original=self.w.journal[field];self.w.journal[field]=value;self.w._save()
            with self.assertRaises(ValueError):
                StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest())
            self.w.journal[field]=original

    def test_expired_completed_cycle_observatory_resume_retains_attempts_without_return_credit(self):
        self.begin(cycle=True);self.w._set('stop_fishing',180)
        self.w.journal['return_deadline']=self.now-1
        self.w.journal['attempts']={'stop':2};self.w._save()
        raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        StoneFarm.request_retry(self.path.parent,digest)
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        with patch.object(self.w,'_expected',return_value=None), patch.object(self.w,'_original',return_value=True):
            self.tick();self.tick()
            self.assertEqual(self.path.read_bytes(),raw)
            self.tick()
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertTrue(self.w.journal['resumed_after_unverified_return'])
        self.assertEqual(self.w.journal['attempts'],{'stop':2})
        self.assertEqual(self.w.journal['return_deadline'],json.loads(raw)['return_deadline'])
        self.assertEqual(self.w.phase,'original')
        self.assertEqual(self.w.journal['cycles'],0)
        self.assertEqual(self.actions,[])

    def test_observatory_recovery_finish_cannot_count_an_unattended_cycle(self):
        self.begin(cycle=True)
        self.w.journal.update(resumed_after_unverified_return=True,gate_closed=True,
            issued_inputs=[dict(kind=k) for k in ('close-gate','activate','enter','enter-confirm','route-start')])
        self.w._finish(self.main)
        self.assertEqual(self.w.journal['cycles'],0)
        self.assertEqual(self.w.journal['recoveries'],1)
        self.assertEqual(self.w.journal['resolution'],'reopened_after_unverified_return_or_close')
        self.assertEqual(self.w.journal['missing_return_input_proof'],['return-confirm','return-request'])

    def test_actual_already_closed_gate_recovery_never_invents_close_input(self):
        self.begin(cycle=True);self.w._set('manager')
        self.w.journal.update(resumed_after_unverified_return=True,returned_at=self.now-30)
        self.w.journal['phase_deadline']=self.now-1;self.w._save()
        raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        request=StoneFarm.request_retry(self.path.parent,digest)
        self.assertEqual(request['mode'],'gate_restore')
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        frame=image('tests/fixtures/stone-farm/already-closed-stage9.png')
        self.assertIs(self.w._gate_manager(frame),False)
        self.assertEqual(self.w._gate_stage(frame),9)
        self.tick(frame);self.tick(frame)
        self.assertEqual(self.path.read_bytes(),raw)
        self.tick(frame)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertTrue(self.w.journal['gate_closed'])
        self.assertTrue(self.w.journal['resumed_after_unverified_close'])
        self.assertNotIn('close-gate',self.w.journal['attempts'])
        self.assertEqual(self.actions,[])
        self.w.journal['issued_inputs']=[dict(kind=k) for k in ('activate','enter','enter-confirm','route-start')]
        self.w._finish(frame)
        self.assertEqual(self.w.journal['cycles'],0)
        self.assertEqual(self.w.journal['recoveries'],1)
        self.assertEqual(self.w.journal['missing_cycle_input_proof'],['close-gate','return-confirm','return-request'])

    def test_actual_new_gate_without_destination_opens_route_before_any_meter_input(self):
        self.begin(cycle=True);self.w._set('arrive',90)
        frame=image('tests/fixtures/stone-farm/stage9-rank0-no-destination.png')
        self.assertTrue(self.w._no_destination(frame))
        self.assertFalse(self.w._no_destination(self.main))
        for _ in range(3):self.tick(frame)
        self.assertEqual(self.w.phase,'route_open')
        self.assertEqual(self.actions,[])
        self.assertNotIn('meter_attempts',self.w.journal)
        self.w._set('fishing_prepare',90)
        self.w.journal['restore_signature']=self.w.signature(frame)
        self.w.meter_context=self.w.fishing._context(frame)
        self.assertFalse(self.w.meter_frame_allowed(frame))
        self.tick(frame)
        self.assertEqual(self.w.phase,'route_open')

    def test_no_destination_retry_retains_all_gate_inputs_and_never_reopens_gate(self):
        self.begin(cycle=True);self.w._set('fishing_prepare',90)
        self.w.journal.update(verified_gate_stage=9,entered_verified_at=self.now-10,
            phase_deadline=self.now-1,attempts={'activate':1,'enter':1,'enter-confirm':1},
            issued_inputs=[dict(kind='enter-confirm')])
        self.w._save();raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        request=StoneFarm.request_retry(self.path.parent,digest)
        self.assertEqual(request['mode'],'startup_restore')
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        frame=image('tests/fixtures/stone-farm/stage9-rank0-no-destination.png')
        self.tick(frame);self.tick(frame);self.tick(frame)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertEqual(self.w.journal['attempts'],json.loads(raw)['attempts'])
        self.assertEqual(self.w.phase,'route_menu')
        self.assertEqual([a['kind'] for a in self.actions],['route-open'])

    def test_actual_new_gate_dim_menu_reads_zero_and_starts_four_existing_targets(self):
        self.begin(cycle=True);self.w._set('route_menu')
        frame=image('tests/fixtures/stone-farm/stage9-rank0-menu.png')
        self.assertEqual(self.w.screen(frame),dict(stage=9,rank=0,menu=True,completion=False,route=None))
        settings=self.w.settings_key(frame)
        self.assertEqual(settings[0],self.w.TARGET_VALUES)
        for _ in range(3):self.tick(frame)
        self.assertEqual(self.w.phase,'route_running')
        self.assertEqual([a['kind'] for a in self.actions],['route-start'])
        self.assertEqual(self.w.journal['native_extras'],settings[1])
        self.w.phase='route_menu';self.w.journal['phase']='route_menu'
        self.assertTrue(self.w.validate_portal(self.action('route-start',(360,873),frame),frame))
        changed=frame.copy();changed[16:25,320:330]=0
        self.assertFalse(self.w.validate_portal(self.action('route-start',(360,873),frame),changed))

    def test_zero_menu_digit_rejects_dark_corrupt_extra_and_damaged_strokes(self):
        frame=image('tests/fixtures/stone-farm/stage9-rank0-menu.png')
        self.assertIsNone(self.w._sailing_menu_rank(frame//2))
        self.assertIsNone(self.w._sailing_menu_rank(frame[:900]))
        damaged=frame.copy();damaged[20:29,94:103]=0
        self.assertIsNone(self.w._sailing_menu_rank(damaged))
        extra=frame.copy();extra[15:52,74:81]=149
        self.assertIsNone(self.w._sailing_menu_rank(extra))
        # A real nonzero number must retain its own reading, never become zero.
        other=image('tests/fixtures/stone-farm/stage8-menu-0.png')
        self.assertEqual(self.w._sailing_menu_rank(other),39)

    def test_zero_template_cannot_override_a_conflicting_numeric_vote(self):
        frame=image('tests/fixtures/stone-farm/stage9-rank0-menu.png')
        readings=[SimpleNamespace(value=v,confidence=.99) for v in (0,0,8,0)]
        with patch('drift.tasks.stone_farm.read_numeric_views',return_value=readings):
            self.assertIsNone(self.w._sailing_menu_rank(frame))

    def test_actual_rank1_menu_restores_four_targets_after_verified_first_result(self):
        self.begin(cycle=True);self.w._set('route_menu')
        self.w.journal.update(first_cast_verified_at=self.now,fishing_recovery=True)
        frame=image('tests/fixtures/stone-farm/stage9-rank1-menu.png')
        self.assertEqual(self.w.screen(frame),dict(stage=9,rank=1,menu=True,completion=False,route=None))
        self.assertEqual(self.w.settings_key(frame)[0],self.w.TARGET_VALUES)
        for _ in range(3):self.tick(frame)
        self.assertEqual(self.w.phase,'route_running')
        self.assertEqual([a['kind'] for a in self.actions],['restore-route-start'])

    def test_one_digit_fallback_needs_full_strokes_and_rejects_conflicting_vote(self):
        frame=image('tests/fixtures/stone-farm/stage9-rank1-menu.png')
        self.assertIsNone(self.w._sailing_menu_rank(frame//2))
        self.assertIsNone(self.w._sailing_menu_rank(frame[:900]))
        readings=[SimpleNamespace(value=v,confidence=.99 if v is not None else .5) for v in (None,None,1,1)]
        with patch('drift.tasks.stone_farm.read_numeric_views',return_value=readings):
            self.assertEqual(self.w._sailing_menu_rank(frame),1)
            bad=frame.copy();bad[26:36,99:108]=0
            self.assertIsNone(self.w._sailing_menu_rank(bad))
            bad=frame.copy();bad[15:52,74:81]=149
            self.assertIsNone(self.w._sailing_menu_rank(bad))
        readings=[SimpleNamespace(value=v,confidence=.99) for v in (1,1,8,1)]
        with patch('drift.tasks.stone_farm.read_numeric_views',return_value=readings):
            self.assertIsNone(self.w._sailing_menu_rank(frame))

    def test_cycle_route_restore_keeps_failed_record_and_requires_actual_entry_zero(self):
        self.begin(cycle=True);self.w._set('route_menu')
        self.w.journal.update(verified_gate_stage=9,entered_verified_at=self.now-10,
            phase_deadline=self.now-1,attempts={'activate':1,'enter':1,'enter-confirm':1,'route-open':1},
            issued_inputs=[dict(kind='enter-confirm')],resumed_after_unverified_return=True,
            resumed_after_unverified_close=True)
        self.w._save();raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        request=StoneFarm.request_retry(self.path.parent,digest)
        self.assertEqual(request['mode'],'route_restore')
        self.assertEqual(self.path.read_bytes(),raw)
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        wrong=image('tests/fixtures/stone-farm/stage8-menu-0.png')
        for _ in range(3):self.tick(wrong)
        self.assertEqual(self.path.read_bytes(),raw)
        frame=image('tests/fixtures/stone-farm/stage9-rank0-menu.png')
        self.tick(frame);self.tick(frame)
        self.assertEqual(self.path.read_bytes(),raw)
        self.tick(frame)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertEqual(self.w.journal['attempts'],json.loads(raw)['attempts'])
        self.assertEqual(self.w.phase,'route_menu')
        self.assertEqual(self.actions,[])
        self.assertAlmostEqual(self.w.journal['deadline']-self.w.journal['started_at'],90)
        self.assertTrue(self.w.journal['resumed_after_unverified_return'])
        self.assertTrue(self.w.journal['resumed_after_unverified_close'])

    def first_task(self, phase='route_running'):
        self.begin(cycle=True)
        self.w.journal.update(verified_gate_stage=9,entered_verified_at=self.now,
                              first_deadline=self.now+60)
        self.w._set(phase,90)
        return image('tests/fixtures/stone-farm/stage9-rank0-gray-boss.png')

    def test_actual_rank0_gray_task_stops_route_before_first_cast_recovery(self):
        frame=self.first_task()
        self.assertIsNotNone(self.w._first_idle(frame,allow_running=True))
        self.assertIsNone(self.w._first_idle(frame))
        for _ in range(3):self.tick(frame)
        self.assertEqual(self.w.phase,'fishing_stop')
        self.assertTrue(self.w.journal['first_recovery'])
        self.assertEqual(self.actions,[])
        self.assertNotIn('meter_attempts',self.w.journal)
        for bad in (frame//2,frame[:900]):
            self.assertIsNone(self.w._first_idle(bad,allow_running=True))

    def test_expired_first_cast_restoration_archives_without_resetting_route_budget(self):
        frame=self.first_task()
        self.w.journal.update(phase_deadline=self.now-1,attempts={'activate':1,'route-start':1},
                              issued_inputs=[dict(kind='route-start')])
        self.w._save();raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        self.assertEqual(StoneFarm.request_retry(self.path.parent,digest)['mode'],'first_cast_restore')
        self.assertEqual(self.path.read_bytes(),raw)
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        for _ in range(3):self.tick(frame)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertEqual(self.w.journal['attempts'],json.loads(raw)['attempts'])
        self.assertEqual(self.w.phase,'fishing_stop')
        self.assertEqual([a['kind'] for a in self.actions],['restore-stop'])
        self.assertAlmostEqual(self.w.journal['deadline']-self.w.journal['started_at'],180)

    def test_first_cast_needs_actual_map_selection_three_seconds_and_fresh_stopped_frame(self):
        frame=self.first_task('first_idle');self.w.off_verified=True
        self.w.journal['first_signature']=self.w.signature(frame)
        with patch.object(self.w.sailing,'running',return_value=False),patch.object(self.w,'_off_icon',return_value=True):
            key=self.w._first_idle(frame);self.assertIsNotNone(key)
            self.w.first_idle_key=key
            action=self.action('first-cast',(360,1010),frame)
            self.assertFalse(self.w.validate_portal(action,frame))
            self.w.journal.update(first_selected_rank=0,issued_inputs=[dict(kind='first-select')])
            self.assertFalse(self.w.validate_portal(action,frame))
            for _ in range(3):
                self.now+=1.6;self.w._confirmed(('first-idle',key))
            self.assertTrue(self.w.validate_portal(action,frame))
            with patch.object(self.w.fishing,'blue_fill',return_value=True):
                self.assertFalse(self.w.validate_portal(action,frame))
            with patch.object(self.w,'_battle',return_value=True):
                self.assertFalse(self.w.validate_portal(action,frame))
            self.w.journal['first_selected_rank']=None
            self.assertFalse(self.w.validate_portal(action,frame))
        self.assertFalse(self.w.validate_portal(action,frame))

    def test_first_map_selection_requires_new_target_and_off_legend_guard(self):
        frame=self.first_task('first_select');self.w.off_verified=True
        frame=image('tests/fixtures/parallel/quest-map-covered-rank3.png')
        self.w.journal['first_signature']=self.w.signature(frame)
        with patch.object(self.w,'_rank_value',return_value=0),patch.object(self.w,'_stage',return_value=9):
            self.assertTrue(self.w._first_map_ready(frame))
            point=self.w._map_task_target(frame)[:2]
            self.w.first_map_target=point;self.w.first_map_verified=True
            action=self.action('first-select',point,frame)
            self.assertTrue(self.w.validate_portal(action,frame))
            bad=frame.copy();bad[613:630,385:393]=0
            self.assertFalse(self.w.validate_portal(action,bad))
            with patch.object(self.w,'_off_icon',return_value=False):
                self.assertFalse(self.w.validate_portal(action,frame))
            self.w.first_map_verified=False
            self.assertFalse(self.w.validate_portal(action,frame))

    def test_first_cast_count_and_sixty_second_deadline_cannot_repeat(self):
        frame=self.first_task('first_idle')
        action=self.action('first-cast',(360,1010),frame)
        self.assertTrue(self.w.reserve_input(action))
        resumed=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        with self.assertRaises(RuntimeError):resumed.reserve_input(action)
        self.now+=61
        with self.assertRaisesRegex(RuntimeError,'60秒'):self.w._first_step(frame)
        self.assertEqual(self.w.journal['attempts']['first-cast'],1)

    def test_current_first_map_skull_core_excludes_avatar_and_needs_both_parts(self):
        self.first_task('first_select')
        frame=image('tests/fixtures/stone-farm/stage9-rank0-selected-map.png')
        self.w.journal['first_signature']=self.w.signature(frame)
        self.assertTrue(self.w._first_map_ready(frame))
        self.assertEqual(self.w._map_task_target(frame)[:2],(377,444))
        # Only the exposed eye/outline and lower red jaw are read.
        changed=frame.copy();changed[410:455,337:385]=0
        self.assertEqual(self.w._map_task_target(changed)[:2],(377,444))
        for region in ((slice(438,455),slice(385,393)),(slice(455,461),slice(366,391))):
            bad=frame.copy();bad[region]=0
            self.assertIsNone(self.w._map_task_target(bad))
        for bad in (frame//2,frame[:900]):
            self.assertIsNone(self.w._map_task_target(bad))
        mask=np.zeros(self.w.first_map_core.shape[:2],bool)
        mask[10:27,23:31]=True;mask[27:33,4:29]=True
        self.assertFalse(self.w.first_map_core[~mask].any())

    def test_new_first_map_sample_cannot_override_wrong_rank_or_running_route(self):
        self.first_task('first_select')
        frame=image('tests/fixtures/stone-farm/stage9-rank0-selected-map.png')
        self.w.journal['first_signature']=self.w.signature(frame)
        for name,value in (('_rank_value',1),('_stage',8),('_off_icon',False)):
            with patch.object(self.w,name,return_value=value):
                self.assertIsNone(self.w._map_task_target(frame))

    def test_first_map_repair_is_one_archived_run_without_resetting_any_input_count(self):
        self.first_task('first_select')
        frame=image('tests/fixtures/stone-farm/stage9-rank0-selected-map.png')
        self.w.journal.update(first_signature=self.w.signature(frame),phase_deadline=self.now-1,
                              attempts={'activate':1,'route-start':1,'first-map':1},
                              issued_inputs=[dict(kind='first-map')])
        self.w._save();raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        self.assertEqual(StoneFarm.request_retry(self.path.parent,digest)['mode'],'first_map_restore')
        self.assertEqual(self.path.read_bytes(),raw)
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        for _ in range(3):self.tick(frame)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertEqual(self.w.journal['attempts'],json.loads(raw)['attempts'])
        self.assertEqual(self.w.journal['first_map_repair_runs'],1)
        self.assertTrue(self.w.off_verified)
        self.assertEqual(self.w.phase,'first_select')
        self.assertAlmostEqual(self.w.journal['deadline']-self.w.journal['started_at'],60)
        self.assertEqual(self.actions,[])
        self.w.journal['phase_deadline']=self.now-1;self.w._save()
        with self.assertRaises(ValueError):StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_first_map_repair_refuses_missing_map_receipt_or_previous_selection(self):
        self.first_task('first_select')
        self.w.journal.update(phase_deadline=self.now-1,issued_inputs=[])
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        for update in ({},{'issued_inputs':[dict(kind='first-map')],'attempts':{'first-select':1}},
                       {'attempts':{'first-cast':1}}):
            self.w.journal.update(update);self.w._save()
            with self.assertRaises(ValueError):StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_late_first_result_restoration_needs_new_rank1_and_never_repeats_cast(self):
        frame=self.first_task('first_verify')
        self.w.journal.update(phase_deadline=self.now-1,first_selected_rank=0,
                              attempts={'route-start':1,'first-select':1,'first-cast':1},
                              issued_inputs=[dict(kind='first-select'),dict(kind='first-cast')])
        self.w._save();raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        self.assertEqual(StoneFarm.request_retry(self.path.parent,digest)['mode'],'first_result_restore')
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        screen=dict(stage=9,rank=0,route=False,menu=False,completion=False)
        with patch.object(self.w,'screen',return_value=screen),patch.object(self.w,'_off_icon',return_value=True),patch.object(self.w.fishing,'_main_visible',return_value=True):
            for _ in range(3):self.tick(frame)
            self.assertEqual(self.path.read_bytes(),raw)
            for changes in ({'rank':1,'stage':8},{'stage':9,'route':True},{'route':False,'rank':40}):
                screen.update(changes)
                for _ in range(3):self.tick(frame)
                self.assertEqual(self.path.read_bytes(),raw)
            screen.update(rank=1)
            for _ in range(3):self.tick(frame)
        self.assertEqual(self.w.phase,'route_menu')
        self.assertEqual(self.w.journal['first_cast_verified_rank'],1)
        self.assertEqual(self.w.journal['attempts'],json.loads(raw)['attempts'])
        self.assertEqual([a['kind'] for a in self.actions],['restore-route-open'])
        self.assertAlmostEqual(self.w.journal['deadline']-self.w.journal['started_at'],90)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)

    def test_late_first_result_requires_real_selection_and_cast_receipts(self):
        self.first_task('first_verify')
        self.w.journal.update(phase_deadline=self.now-1,first_selected_rank=0,attempts={'first-cast':1})
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        for receipts in ([],[dict(kind='first-cast')],[dict(kind='first-select')]):
            self.w.journal['issued_inputs']=receipts;self.w._save()
            with self.assertRaises(ValueError):StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_first_deadline_is_rechecked_at_reservation_and_bounds_later_phases(self):
        frame=self.first_task('fishing_stop')
        self.w.journal['first_deadline']=self.now+4
        self.w._set('first_select')
        self.assertEqual(self.w.journal['phase_deadline'],self.now+4)
        # Even an inconsistent later phase deadline cannot renew the total.
        self.w.journal['phase_deadline']=self.now+90
        self.now+=4.1
        self.assertFalse(self.w.input_allowed())
        self.assertFalse(self.w.reserve_input(self.action('first-select',(377,619),frame)))
        self.assertNotIn('first-select',self.w.journal['attempts'])

    def test_first_cast_fill_alone_is_not_success_but_actual_rank_advance_is(self):
        frame=self.first_task('first_verify')
        self.w.journal['first_signature']=self.w.signature(frame)
        screen=dict(stage=9,rank=0,route=False,menu=False,completion=False)
        with patch.object(self.w,'screen',return_value=screen),patch.object(self.w.fishing,'blue_fill',return_value=True):
            for _ in range(3):self.tick(frame.copy())
        self.assertEqual(self.w.phase,'first_verify')
        screen['rank']=1
        with patch.object(self.w,'screen',return_value=screen),patch.object(self.w,'_off_icon',return_value=True):
            for _ in range(3):self.tick(frame.copy())
            self.assertEqual(self.w.phase,'first_verify')
            self.w.journal['issued_inputs']=[dict(kind='first-cast')]
            for _ in range(3):self.tick(frame.copy())
        self.assertEqual(self.w.phase,'route_open')
        self.assertEqual(self.w.journal['first_cast_verified_rank'],1)
        self.assertEqual(self.w.journal['cycles'],0)
        self.assertEqual(self.actions,[])

    def test_dim_entry_context_can_only_be_reproved_before_any_meter_input(self):
        self.begin();self.w._set('fishing_prepare',90);self.w.off_verified=True
        frame=image('tests/fixtures/stone-farm/stage8-gray-hook.png')
        self.w.journal.update(verified_gate_stage=8,restore_signature=self.w.signature(frame))
        self.w.meter_context='old-dim-entry-context'
        until=self.w.journal['phase_deadline']
        with patch.object(self.w,'_expected',return_value=dict(stage=8,rank=39,menu=False,route=False)):
            self.tick(frame);self.tick(frame)
            self.assertEqual(self.w.meter_context,'old-dim-entry-context')
            self.tick(frame)
        self.assertEqual(self.w.meter_context,self.w.fishing._context(frame))
        self.assertEqual(self.w.journal['phase_deadline'],until)
        self.assertEqual(self.actions,[])

    def test_new_running_arrow_requires_stopping_again(self):
        self.begin();self.w._set('stop_fishing',180)
        self.w.off_verified=True
        with patch.object(self.w,'_expected',return_value=dict(stage=9,rank=39,menu=False,route=True)):
            self.tick()
        self.assertEqual(self.w.phase,'stop')
        self.assertFalse(self.w.off_verified)
        self.assertEqual(self.actions,[])

    def test_retry_keeps_original_bytes_deadlines_and_counts_until_new_proof(self):
        self.begin();self.w._set('return')
        self.w.journal['attempts']['return-request']=8
        self.w._save()
        raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        approval=StoneFarm.request_retry(self.path.parent,digest)
        self.assertEqual(self.path.read_bytes(),raw)
        resumed=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        self.w=resumed
        screen=dict(stage=9,rank=39,menu=False,completion=False,route=False)
        with patch.object(resumed,'_expected',return_value=screen):
            self.tick();self.tick()
            self.assertEqual(self.path.read_bytes(),raw)
            self.tick()
        archived=self.path.parent/'logs/stone-farm-failed'/f'{digest}.json'
        self.assertEqual(archived.read_bytes(),raw)
        self.assertEqual(resumed.journal['previous_failed_sha256'],digest)
        self.assertEqual(resumed.journal['attempts'],{})
        self.assertGreater(resumed.journal['started_at'],json.loads(raw)['started_at'])
        self.assertEqual(resumed.journal['retry_approved_at'],approval['approved_at'])

    def test_retry_rejects_changed_hash_unexhausted_run_and_disabled_feature(self):
        self.begin();self.w._set('return')
        config=self.path.parent/'config.json'
        config.write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        digest=hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):StoneFarm.request_retry(self.path.parent,digest)
        self.w.journal['attempts']['return-request']=8;self.w._save()
        with self.assertRaises(ValueError):StoneFarm.request_retry(self.path.parent,digest)
        config.write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=False)])))
        with self.assertRaises(ValueError):StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_two_cancel_inputs_cannot_be_renewed_by_restart(self):
        self.begin();self.w._set('stop_fishing',180)
        action=self.action('handoff-cancel',(690,1090))
        self.assertTrue(self.w.reserve_input(action));self.assertTrue(self.w.reserve_input(action))
        resumed=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        with self.assertRaises(RuntimeError):resumed.reserve_input(action)

    def test_actual_dim_menu_keeps_fixed_text_but_rechecks_native_extras(self):
        self.begin();self.w._set('route_menu');self.w.off_verified=True
        frames=[image(f'tests/fixtures/stone-farm/stage8-menu-{i}.png') for i in range(3)]
        settings=self.w.settings_key(frames[0])
        self.w.journal['native_extras']=settings[1]
        self.w.menu_key=settings
        action=self.action('target:boss',(562,458),frames[0])
        for frame in frames:
            self.assertEqual(self.w.screen(frame)['stage'],8)
            self.assertEqual(self.w.signature(frame),action['signature'])
            # These old frames vary in the native extra/priority fingerprint.
            # Fixed place text alone cannot authorise changing a target after
            # the saved menu settings changed; preserve the current guard.
            same_extras=self.w.settings_key(frame)[1] == settings[1]
            self.assertEqual(self.w.validate_portal(action,frame),same_extras)
        changed=frames[1].copy();changed[16:25,320:330]=0
        self.assertFalse(self.w.validate_portal(action,changed))

    def test_route_restoration_archives_timeout_and_keeps_all_portal_counts(self):
        self.begin();self.w._set('route_menu')
        self.w.journal.update(verified_gate_stage=8,entered_verified_at=self.now,
                              issued_inputs=[dict(kind='enter-confirm')],attempts={'lower':1,'return-request':1,'route-open':1})
        self.w.journal['phase_deadline']=self.now-1;self.w._save()
        raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        request=StoneFarm.request_retry(self.path.parent,digest)
        self.assertEqual(request['mode'],'route_restore')
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        frame=image('tests/fixtures/stone-farm/stage8-menu-0.png')
        self.tick(frame);self.tick(frame)
        self.assertEqual(self.path.read_bytes(),raw)
        self.tick(frame)
        self.assertTrue(self.w.journal['restoration_only'])
        self.assertEqual(self.w.journal['attempts'],json.loads(raw)['attempts'])
        self.assertEqual(self.w.phase,'route_menu')
        self.assertAlmostEqual(self.w.journal['deadline']-self.w.journal['started_at'],90)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertEqual(self.actions,[])

    def test_deadlines_and_input_counts_survive_restart(self):
        self.begin()
        action=self.action('stop',(233,45))
        self.w.reserve_input(action)
        before=deepcopy(self.w.journal)
        resumed=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        self.assertEqual(resumed.journal,before)
        self.now=before['deadline']+1
        self.assertFalse(resumed.reserve_input(action))
        self.assertEqual(json.loads(self.path.read_text()),before)

    def test_gray_hook_recovery_archives_expired_route_without_resetting_portal_attempts(self):
        self.begin()
        self.w.journal.update(phase='route_running',phase_deadline=self.now-1,
            verified_gate_stage=8,entered_verified_at=self.now-20,
            attempts={'route-start':1,'target:boss':1},
            issued_inputs=[dict(kind='route-start',input_id='actual-start')])
        self.w._save()
        raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        request=StoneFarm.request_retry(self.path.parent,digest)
        self.assertEqual(request['mode'],'fishing_restore')
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        frame=image('tests/fixtures/stone-farm/stage8-gray-hook.png')
        self.tick(frame);self.tick(frame)
        self.assertEqual(self.path.read_bytes(),raw)
        self.tick(frame)
        self.assertTrue(self.w.journal['fishing_recovery'])
        self.assertEqual(self.w.journal['attempts']['route-start'],1)
        self.assertEqual(self.w.journal['attempts']['target:boss'],1)
        self.assertAlmostEqual(self.w.journal['deadline']-self.w.journal['started_at'],180)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)

    def test_recovery_needs_stopped_ordinary_idle_and_exact_world_place_strokes(self):
        self.begin()
        frame=image('tests/fixtures/stone-farm/stage8-gray-hook.png')
        self.w.off_verified=True
        self.w._prepare_fishing(frame)
        self.assertFalse(self.w.meter_frame_allowed(frame))  # Actual green arrow.
        with patch.object(self.w,'_off_icon',return_value=True), patch.object(self.w.sailing,'running',return_value=False):
            self.assertTrue(self.w.meter_frame_allowed(frame))
            for owner,name,value in ((self.w.training,'boss_mode',True),
                                     (self.w.fishing,'blue_fill',True),
                                     (self.w,'_battle',True),
                                     (self.w.fishing,'at_port',True)):
                with patch.object(owner,name,return_value=value):
                    self.assertFalse(self.w.meter_frame_allowed(frame))
            changed=frame.copy();changed[12:50,72:130]=255
            self.assertFalse(self.w.meter_frame_allowed(changed))
            changed=frame.copy();changed[12:34,286:434]=255
            self.assertFalse(self.w.meter_frame_allowed(changed))
            self.now=self.w.journal['phase_deadline']+1
            self.assertFalse(self.w.meter_frame_allowed(frame))

    def test_recovery_reserves_only_one_hold_and_never_cancels_existing_fill(self):
        self.begin();self.w._prepare_fishing(self.main)
        self.assertFalse(self.w.reserve_meter_input('cancel'))
        self.assertTrue(self.w.reserve_meter_input('start'))
        with self.assertRaises(RuntimeError):self.w.reserve_meter_input('start')
        resumed=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        self.assertFalse(resumed.off_verified)
        self.assertEqual(resumed.journal['meter_attempts'],{'start':1})
        self.assertEqual(resumed.journal['phase_deadline'],self.w.journal['phase_deadline'])

    def test_end_only_fill_does_not_credit_a_complete_restart_or_repeat_hold(self):
        self.begin();self.w.off_verified=True;self.w._prepare_fishing(self.main)
        self.w.journal['meter_attempts']={'start':1}
        screen=dict(stage=8,rank=39,menu=False,completion=False,route=False)
        with patch.object(self.w,'_expected',return_value=screen), patch.object(self.w.fishing,'blue_fill',return_value=True):
            for _ in range(4):self.tick()
        self.assertEqual(self.w.phase,'fishing_prepare')
        self.assertEqual(self.actions,[])
        self.assertEqual(self.w.journal.get('handoffs'),0)

    def test_paired_restore_reopens_four_targets_without_repeating_original_start_budget(self):
        self.begin();self.w.off_verified=True;self.w._prepare_fishing(self.main)
        self.w.journal.update(fishing_recovery=True,meter_attempts={'start':1},attempts={'route-start':1})
        self.w.fishing.start_fill_verified=True
        screen=dict(stage=8,rank=39,menu=False,completion=False,route=False)
        with patch.object(self.w,'_expected',return_value=screen), patch.object(self.w.fishing,'blue_fill',return_value=True):
            for _ in range(3):self.tick()
        self.assertEqual(self.w.phase,'route_open')
        self.assertEqual(self.w.journal['restored_fishing_proof'],'paired_hold')
        self.assertEqual(self.w._route_action('route-start'),'restore-route-start')
        self.assertTrue(self.w.reserve_input(self.action('restore-route-start',(360,873))))
        self.assertEqual(self.w.journal['attempts'],{'route-start':1,'restore-route-start':1})

    def test_return_retries_do_not_extend_original180_seconds(self):
        self.begin()
        until=self.w.journal['return_deadline']
        for phase in ('return','return_confirm','return','return_confirm'):
            self.now+=10
            self.w._set(phase,180)
        self.assertEqual(self.w.journal['return_deadline'],until)
        self.now=until+1
        self.assertFalse(self.w.validate_portal(self.action('return-confirm',(208,788)),self.main))

    def test_attempt_is_never_an_issued_close_or_full_cycle(self):
        self.begin(cycle=True)
        self.w.journal['gate_closed']=True
        self.w.reserve_input(self.action('close-gate',(354,739)))
        with self.assertRaises(RuntimeError):self.w._finish(self.main)
        self.assertTrue(self.w.active)
        self.assertEqual(self.w.journal['cycles'],0)

    def test_limits_reserve_before_input_and_do_not_reset(self):
        self.begin()
        action=self.action('lower',(357,873))
        self.assertTrue(self.w.reserve_input(action))
        with self.assertRaises(RuntimeError):self.w.reserve_input(action)
        self.assertEqual(self.w.journal['attempts']['lower'],1)

    def test_new_gate_must_be_actual9_before_entry(self):
        self.begin(cycle=True)
        self.w._set('new_gate')
        active8=image('tests/fixtures/parallel/gate-active-tier8.png')
        for _ in range(4):self.tick(active8)
        self.assertEqual(self.w.phase,'new_gate')
        active9=image('assets/recovery/parallel-manager.png')
        for _ in range(3):self.tick(active9)
        self.assertEqual(self.w.phase,'panel_close')
        self.assertEqual(self.w.journal['verified_gate_stage'],9)

    def test_native_targets_have_four_entries_and_no_talent(self):
        self.assertEqual(dict(zip(TARGET_NAMES,self.w.TARGET_VALUES)),
                         dict(pearl=True,material=False,amber=True,gem=True,boss=True))

    def test_unknown_continue_cannot_confirm_gate_close(self):
        self.begin(cycle=True)
        self.w._set('close_confirm')
        lower=image('assets/recovery/parallel-lower-question.png')
        self.w.journal['attempts']['close-gate']=1
        self.assertFalse(self.w.validate_portal(self.action('close-confirm',(208,788),lower),lower))

    def test_corrupt_saved_cross_stage_or_rank_is_rejected(self):
        self.begin()
        original=deepcopy(self.w.journal)
        for field,value in (('rank',40),('to_stage',9),('entry_rank',0),('deadline',float('inf'))):
            bad={**original,field:value}
            self.path.write_text(json.dumps(bad),encoding='utf-8')
            with self.assertRaises(RuntimeError):StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)

    def test_native_route_must_show_fishing_before_handoff_is_finished(self):
        self.begin()
        self.w._set('route_running',90)
        frame=image('tests/fixtures/stone-farm/stage8-rank39-restored-battle.png')
        self.assertFalse(self.w._moving(frame))
        screen=dict(stage=8,rank=39,menu=False,completion=False,route=True)
        with patch.object(self.w,'screen',return_value=screen), patch.object(self.w.fishing,'blue_fill',return_value=False), patch.object(self.w,'_battle',return_value=False):
            for _ in range(4):self.tick(frame)
        self.assertTrue(self.w.active)
        self.assertEqual(self.w.journal['handoffs'],0)
        with patch.object(self.w,'screen',return_value=screen), patch.object(self.w.fishing,'blue_fill',return_value=True):
            for _ in range(3):self.tick(frame)
        self.assertFalse(self.w.active)
        self.assertEqual(self.w.journal['handoffs'],1)
        self.assertEqual(self.w.journal['cycles'],0)

    def test_disabled_task_sends_no_input_or_journal_rewrite(self):
        self.w.enabled=lambda:False
        for _ in range(4):self.assertFalse(self.tick())
        self.assertFalse(self.path.exists())
        self.assertEqual(self.actions,[])

    def test_fast_final_boss_notice_finishes_only_lower_handoff(self):
        self.begin()
        self.w._set('route_running',90)
        notice=dict(stage=8,rank=40,menu=False,completion=True,route=None)
        with patch.object(self.w,'screen',return_value=notice):
            for _ in range(3):self.tick()
        self.assertFalse(self.w.active)
        self.assertEqual(self.w.journal['handoffs'],1)
        self.assertEqual(self.w.journal['cycles'],0)
        self.assertFalse(self.w.completed_rank_40)

    def test_config_migration_is_idempotent_and_preserves_other_switches(self):
        config=dict(rules=[dict(id='driftinggirl_parallel_world',enabled=False,training={'parallel_stage_limit':8}),
                           dict(id='driftinggirl_equipment_upgrade',enabled=True,interval=900)])
        before=deepcopy(config)
        updated=add(config,enabled=True)
        self.assertEqual(config,before)
        self.assertEqual(updated['rules'][:-1],before['rules'])
        self.assertTrue(updated['rules'][-1]['enabled'])
        self.assertEqual(add(updated),updated)


TARGET_NAMES=('pearl','material','amber','gem','boss')


class StoneRuntimeTests(unittest.TestCase):
    setUp = helpers.FrameworkTests.setUp
    tearDown = helpers.FrameworkTests.tearDown
    rule = helpers.FrameworkTests.rule
    wait_for = helpers.FrameworkTests.wait_for

    def run_owner(self, allowed, kind='route-open', context=True, slow=False, boss_handoff=False,
                  continuous_handoff=False,ready_after=0,reserve_delay=0,change_after_reserve=False):
        entry=self.rule()
        entry.update(id='driftinggirl_stone_farm',stage='stone_owned',action='observe')
        self.app.persist()
        self.app.digit_reader=object()
        self.app.device.io_lock=threading.RLock()
        def touch(x,y,started,stopped):
            if stopped():return None
            self.app.device.taps.append((x,y))
            return dict(input_transport='mumu-native',hold_ms=40)
        self.app.device.touch_fast=touch
        self.app.device.commands=[]
        self.app.device.command=lambda *args:self.app.device.commands.append(args)
        self.app.device.holds=[]
        def hold_feedback(x,y,milliseconds):
            self.app.device.holds.append((x,y,milliseconds))
            return helpers.FakeDevice.source.copy()
        self.app.device.hold_feedback=hold_feedback
        instances=[]
        class Controller:
            def retire_expired_on_start(self):return False
            boss_abandon_handoff=boss_handoff
            handoff_stop_requested=continuous_handoff
            active=False
            phase='idle'
            rank=39
            completed_rank_40=False
            def __init__(self,root,reader,action,event,interval):
                self.action=action
                self.defense=SimpleNamespace(action=None,scan=False,status=lambda:{},
                    encounter_rank=lambda frame:None,
                    owns=lambda *args:boss_handoff or continuous_handoff,
                    step=lambda *args:boss_handoff or continuous_handoff)
                self.asked=False
                self.validations=self.reservations=self.issued=self.context_checks=0
                self.world=SimpleNamespace(
                    fishing=SimpleNamespace(validate=lambda *args: allowed,HOLD_MS=1500),
                    sailing=SimpleNamespace(running=lambda frame:False),
                    training=SimpleNamespace(options={'auto_sailing':True},state={'sailing_enabled':False,'rank':39}))
                if kind == 'meter-start':
                    # Exercise the real adapter that production passes to the
                    # shared fishing input wrapper, with isolated state files.
                    adapter=StoneFarm(ROOT,reader,None,lambda _:None,
                                      state_path=Path(root)/'adapter-state.json')
                    self.world=adapter.world
                    self.world.fishing.validate=lambda *args:allowed
                    self.world.sailing.running=lambda frame:False
                    self.world.training.state={'sailing_enabled':False,'rank':39}
                instances.append(self)
            def input_allowed(self):return allowed
            def meter_frame_allowed(self,frame):return allowed
            def reserve_meter_input(self,action):
                self.reservations+=1
                return allowed
            def validate_portal(self,action,frame):
                self.validations+=1
                if slow:
                    helpers.time.sleep(.13)
                return allowed and self.validations > ready_after
            def same_cancel_context(self,action,frame):
                self.context_checks+=1
                return context and not (change_after_reserve and self.reservations)
            def reserve_input(self,action):
                self.reservations+=1
                if reserve_delay:
                    helpers.time.sleep(reserve_delay)
                return True
            def input_issued(self,action,metadata):
                self.issued+=1
                self.metadata=metadata
            def status(self):return dict(phase=self.phase,pending=False,rank=self.rank)
            def step(self,frame):
                if self.asked:return False
                self.asked=True
                if kind == 'meter-start':
                    return bool(self.meter_action('start',dict(value=1810,goal=1810)))
                return self.action(dict(kind=kind,x=23,y=45,label='test native route'))
        with patch('app.StoneFarm',Controller):
            self.app.start(preview=False,timeout=1,max_clicks=1)
            self.wait_for(lambda:not self.app.running())
        self.assertIsNone(self.app.status()['last_error'])
        return instances[0]

    def test_runtime_validates_and_reserves_the_stone_owner_on_a_fresh_frame(self):
        owner=self.run_owner(True)
        self.assertEqual((owner.validations,owner.reservations,owner.issued),(1,1,1))
        self.assertEqual(len(self.app.device.taps),1)
        self.assertEqual(owner.metadata['controller'],'stone_farm')
        self.assertIsNone(self.app.status()['parallel'])

    def test_rejected_fresh_stone_frame_consumes_no_input_or_reservation(self):
        owner=self.run_owner(False)
        self.assertEqual((owner.validations,owner.reservations,owner.issued),(1,0,0))
        self.assertEqual(self.app.device.taps,[])

    def test_cancel_scan_stops_at40_fresh_frames_without_reserving(self):
        owner=self.run_owner(False,kind='handoff-cancel')
        self.assertEqual((owner.context_checks,owner.validations),(40,40))
        self.assertEqual((owner.reservations,owner.issued),(0,0))
        self.assertEqual(self.app.device.taps,[])

    def test_cancel_changed_context_abandons_scan_immediately(self):
        owner=self.run_owner(True,kind='handoff-cancel',context=False)
        self.assertEqual((owner.context_checks,owner.validations,owner.reservations),(1,0,0))
        self.assertEqual(self.app.device.taps,[])

    def test_cancel_uses_fresh_native40ms_input_and_records_actual_issue(self):
        owner=self.run_owner(True,kind='handoff-cancel')
        self.assertEqual((owner.reservations,owner.issued),(1,1))
        self.assertEqual(owner.metadata['input_transport'],'mumu-native')
        self.assertEqual(owner.metadata['hold_ms'],40)

    def test_continuous_stop_observes_new_frames_until_cast_and_yields_battle(self):
        owner=self.run_owner(True,kind='handoff-cancel',continuous_handoff=True,ready_after=3)
        self.assertEqual((owner.context_checks,owner.validations),(5,5))
        self.assertEqual((owner.reservations,owner.issued),(1,1))
        self.assertEqual(owner.metadata['hold_ms'],40)
        self.assertEqual(self.app.device.commands,[])

    def test_cancel_recaptures_after_slow_journal_write_without_second_reservation(self):
        owner=self.run_owner(True,kind='handoff-cancel',reserve_delay=.15)
        self.assertEqual((owner.reservations,owner.issued),(1,1))
        self.assertEqual((owner.context_checks,owner.validations),(2,2))
        self.assertLessEqual(owner.metadata['capture_age_ms'],120)
        self.assertEqual(len(self.app.device.taps),1)

    def test_cancel_changed_context_after_reservation_retains_attempt_without_input(self):
        owner=self.run_owner(True,kind='handoff-cancel',change_after_reserve=True)
        self.assertEqual((owner.reservations,owner.issued),(1,0))
        self.assertEqual(self.app.device.taps,[])

    def test_cancel_over120ms_frame_never_reserves_or_inputs(self):
        owner=self.run_owner(True,kind='handoff-cancel',slow=True)
        self.assertEqual((owner.reservations,owner.issued),(0,0))
        self.assertEqual(self.app.device.taps,[])

    def test_approved_boss_abandon_uses_native_contact_and_yields_battle_priority(self):
        owner=self.run_owner(True,kind='lower-boss-abandon',boss_handoff=True)
        self.assertEqual((owner.reservations,owner.issued),(1,1))
        self.assertEqual(owner.metadata['input_transport'],'mumu-native')
        self.assertEqual(owner.metadata['hold_ms'],40)

    def test_boss_abandon_stale_frame_consumes_no_input_or_reservation(self):
        owner=self.run_owner(True,kind='lower-boss-abandon',slow=True,boss_handoff=True)
        self.assertEqual((owner.reservations,owner.issued),(0,0))
        self.assertEqual(self.app.device.taps,[])

    def test_separately_approved_menu_contact_uses_one_stationary80ms_input(self):
        owner=self.run_owner(True,kind='lower-boss-abandon-menu',boss_handoff=True)
        self.assertEqual((owner.reservations,owner.issued),(1,1))
        self.assertEqual(self.app.device.commands,[('shell','input','swipe','23','45','23','45','80')])
        self.assertEqual(self.app.device.taps,[])
        self.assertEqual(owner.metadata['input_transport'],'adb-stationary')
        self.assertEqual(owner.metadata['hold_ms'],80)

    def test_menu_contact_stale_frame_or_failed_guard_does_not_input(self):
        owner=self.run_owner(True,kind='lower-boss-abandon-menu',slow=True,boss_handoff=True)
        self.assertEqual((owner.reservations,owner.issued),(0,0))
        self.assertEqual(self.app.device.commands,[])
        self.assertEqual(self.app.device.taps,[])

    def test_retry_endpoint_requires_idle_worker_and_explicit_cancel_permission(self):
        with self.assertRaises(ValueError):self.app.retry_stone_farm(dict(failed_sha256='x',allow_ordinary_cancel=False))
        with patch.object(self.app,'running',return_value=True):
            with self.assertRaises(ValueError):self.app.retry_stone_farm(dict(failed_sha256='x',allow_ordinary_cancel=True))
        with patch('app.StoneFarm.request_retry',return_value={'approved':True}) as approve:
            self.assertEqual(self.app.retry_stone_farm(dict(failed_sha256='x',allow_ordinary_cancel=True)),dict(approved=True))
            approve.assert_called_once_with(self.root,'x')

    def test_retry_endpoint_boss_permission_is_explicit_and_optional(self):
        with patch('app.StoneFarm.request_retry',return_value={'approved':True}) as approve:
            self.app.retry_stone_farm(dict(failed_sha256='x',allow_ordinary_cancel=True,allow_boss_abandon=True))
            approve.assert_called_once_with(self.root,'x',allow_boss_abandon=True)
        with self.assertRaises(ValueError):
            self.app.retry_stone_farm(dict(failed_sha256='x',allow_ordinary_cancel=True,allow_boss_abandon=False))

    def test_menu_retry_endpoint_passes_only_the_new_explicit_permission(self):
        with patch('app.StoneFarm.request_retry',return_value={'approved':True}) as approve:
            self.app.retry_stone_farm(dict(failed_sha256='x',allow_ordinary_cancel=True,allow_boss_abandon_menu=True))
            approve.assert_called_once_with(self.root,'x',allow_boss_abandon_menu=True)
        with self.assertRaises(ValueError):
            self.app.retry_stone_farm(dict(failed_sha256='x',allow_ordinary_cancel=True,allow_boss_abandon_menu=False))

    def test_continuous_stop_retry_endpoint_requires_explicit_flag(self):
        with patch('app.StoneFarm.request_retry',return_value={'approved':True}) as approve:
            self.app.retry_stone_farm(dict(failed_sha256='x',allow_ordinary_cancel=True,allow_continuous_cancel=True))
            approve.assert_called_once_with(self.root,'x',allow_continuous_cancel=True)
        with self.assertRaises(ValueError):
            self.app.retry_stone_farm(dict(failed_sha256='x',allow_ordinary_cancel=True,allow_continuous_cancel=False))

    def test_stone_meter_hold_uses_its_own_budget_with_legacy_calibrator_absent(self):
        owner=self.run_owner(True,kind='meter-start')
        self.assertEqual(owner.reservations,1)
        self.assertEqual(self.app.device.holds,[(360,1010,1500)])
        self.assertEqual(self.app.device.taps,[])


class RetiredSelfUnlockTests(unittest.TestCase):
    """The retired handoff's own working position must block only within backoff.

    The dimension-stone lowering acts at [8,40]; a startup retirement built
    from a cycle record blocks exactly that context, so an unreleased block
    used to be permanent: RANK could only advance through the very input the
    block withheld. Release must therefore be bounded by the retry backoff,
    while the no-blind-replay gate keeps holding inside the window.
    """
    setUpClass=classmethod(StoneFarmTests.setUpClass.__func__)
    setUp=StoneFarmTests.setUp
    tearDown=StoneFarmTests.tearDown
    act=StoneFarmTests.act
    tick=StoneFarmTests.tick

    def retire(self,age,blocked):
        # Field record 10-28 13:13: the startup retirement blocked [[8,40],[9,0]].
        self.w.journal['retirement']=dict(at=self.now-age,source='startup_expired',
            previous_phase='stop',expired_fields=['deadline'],released=False,
            blocked_contexts=blocked)
        self.w._save()

    def screen8(self):
        return dict(stage=8,rank=40,menu=False,completion=False,route=True)

    def test_blocked_working_context_releases_after_backoff_window(self):
        self.retire(1801,[[8,40],[9,0]])
        with patch.object(self.w,'screen',return_value=self.screen8()):
            self.assertFalse(self.tick())
            self.assertFalse(self.tick())
            self.assertFalse(self.w.journal['retirement']['released'])
            self.assertTrue(self.tick())  # third confirmation releases and re-enters trigger
        released=self.w.journal['retirement']
        self.assertTrue(released['released'])
        self.assertEqual(released['released_context'],[8,40])
        self.assertAlmostEqual(released['released_at'],self.now)
        saved=json.loads(self.path.read_text('utf-8'))
        self.assertTrue(saved['retirement']['released'])
        self.assertEqual(saved['retirement']['released_context'],[8,40])
        self.assertEqual(self.actions,[])

    def test_blocked_working_context_within_backoff_stays_conservative(self):
        self.retire(1,[[8,40],[9,0]])
        with patch.object(self.w,'screen',return_value=self.screen8()):
            for _ in range(6):
                self.assertFalse(self.tick())
        self.assertFalse(self.w.journal['retirement']['released'])
        self.assertEqual(self.actions,[])  # no input is replayed at the archived context
        self.assertIn('不重复输入',self.w.hold)
        self.assertIn('退避',self.w.hold)
        self.assertFalse(json.loads(self.path.read_text('utf-8'))['retirement']['released'])

    def test_real_progress_releases_without_waiting_for_backoff(self):
        self.retire(1,[[8,40],[9,0]])
        progress=dict(stage=9,rank=39,menu=False,completion=False,route=True)
        with patch.object(self.w,'screen',return_value=progress):
            self.assertFalse(self.tick())
            self.assertFalse(self.tick())
            self.assertFalse(self.w.journal['retirement']['released'])
            self.tick()
        self.assertTrue(self.w.journal['retirement']['released'])
        self.assertEqual(self.w.journal['retirement']['released_context'],[9,39])
        self.assertEqual(self.actions,[])


class RetiredBlockedContextTests(unittest.TestCase):
    """Both retirement constructors must block real contexts, never sentinels.

    entry_rank=0 is the cycle journal's "rank not observed" marker; rank 0 is
    not a legal in-game RANK (1..40), so [to_stage,0] must never be blocked.
    """
    setUpClass=classmethod(StoneFarmTests.setUpClass.__func__)
    setUp=StoneFarmTests.setUp
    tearDown=StoneFarmTests.tearDown
    act=StoneFarmTests.act

    def write(self,**overrides):
        record=dict(version=1,pending=True,phase='stop',deadline=500.,phase_deadline=400.,
                    started_at=1,reason='cycle',from_stage=8,to_stage=9,rank=40,entry_rank=0,
                    attempts={},issued_inputs=[],cycles=0,handoffs=0)
        record.update(overrides)
        self.path.write_text(json.dumps(record),encoding='utf-8')
        return record

    def test_startup_retirement_never_blocks_the_unobserved_entry_rank(self):
        self.write()
        digest=hashlib.sha256(self.path.read_bytes()).hexdigest()
        with patch('drift.tasks.handoff_journal.time.time',return_value=600.):
            result=records.retire(self.path.parent,digest,source='startup_expired')
        self.assertTrue(result['retired'])
        saved=json.loads(self.path.read_text('utf-8'))
        self.assertEqual(saved['retirement']['blocked_contexts'],[[8,40]])
        self.assertNotIn([9,0],saved['retirement']['blocked_contexts'])

    def test_startup_retirement_still_blocks_the_observed_lower_entry(self):
        self.write(reason='lower',rank=39,from_stage=9,to_stage=8,entry_rank=39)
        digest=hashlib.sha256(self.path.read_bytes()).hexdigest()
        with patch('drift.tasks.handoff_journal.time.time',return_value=600.):
            result=records.retire(self.path.parent,digest,source='startup_expired')
        self.assertTrue(result['retired'])
        saved=json.loads(self.path.read_text('utf-8'))
        self.assertEqual(saved['retirement']['blocked_contexts'],[[9,39],[8,39]])

    def test_retryable_retirement_matches_the_same_context_rule(self):
        self.write(phase='route_running')
        controller=object.__new__(StoneFarm)
        controller.path=self.path
        controller.journal=json.loads(self.path.read_text('utf-8'))
        controller.phase,controller.retry='route_running',None
        controller.screen=Mock(return_value=dict(stage=9,rank=40,menu=False,completion=False,route=True))
        controller.fishing=SimpleNamespace(_main_visible=Mock(return_value=True))
        with patch('drift.tasks.stone_farm.time.time',return_value=600.):
            self.assertTrue(controller.retire_retryable(object()))
        self.assertEqual(controller.journal['retirement']['blocked_contexts'],[[8,40]])
        self.assertNotIn([9,0],controller.journal['retirement']['blocked_contexts'])


if __name__ == '__main__':unittest.main()
