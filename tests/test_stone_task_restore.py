"""Replay the actual lowered RANK39 dormant task without loosening meter guards."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
import unittest

import numpy as np

from drift.tasks.stone_farm import StoneFarm
from tests import test_stone_farm as base

ROOT, image = base.ROOT, base.image


class LoweredTaskRestoreTests(unittest.TestCase):
    setUpClass = classmethod(base.StoneFarmTests.setUpClass.__func__)
    setUp = base.StoneFarmTests.setUp
    tearDown = base.StoneFarmTests.tearDown
    act = base.StoneFarmTests.act
    tick = base.StoneFarmTests.tick
    begin = base.StoneFarmTests.begin
    action = base.StoneFarmTests.action

    def lowered(self, phase='route_running'):
        self.begin()
        self.w.journal.update(verified_gate_stage=8,entered_verified_at=self.now,
                              first_deadline=self.now+60)
        self.w._set(phase,90)
        return image('tests/fixtures/stone-farm/stage8-rank39-gray-boss.png')

    def test_actual_error_has_task_boss_gray_hook_and_no_fishing(self):
        frame = self.lowered()
        self.assertEqual(self.w.screen(frame),dict(stage=8,rank=39,route=True,menu=False,completion=False))
        self.assertTrue(self.w.training.boss_mode(frame))
        self.assertFalse(self.w.fishing.blue_fill(frame))
        self.assertFalse(self.w._battle(frame))
        self.assertEqual(self.w.fishing.idle_icon(frame),'hook')
        self.assertEqual(self.w._first_idle(frame,allow_running=True)[2],1569)
        self.assertIsNone(self.w._first_idle(frame))

    def test_three_seconds_dormant_task_stops_route_before_any_task_input(self):
        frame = self.lowered()
        for _ in range(2):self.tick(frame)
        self.assertEqual(self.w.phase,'route_running')
        self.tick(frame)
        self.assertEqual(self.w.phase,'fishing_stop')
        self.assertTrue(self.w.journal['first_recovery'])
        self.assertEqual(self.actions,[])
        self.assertNotIn('meter_attempts',self.w.journal)
        self.tick(frame)
        self.assertEqual([a['kind'] for a in self.actions],['restore-stop'])

    def test_task_restore_rejects_wrong_world_rank_battle_fill_port_or_unknown_auto(self):
        frame = self.lowered()
        for owner,name,value in ((self.w,'_stage',9),(self.w,'_rank_value',38),
                                 (self.w.training,'boss_mode',False),
                                 (self.w.fishing,'blue_fill',True),(self.w,'_battle',True),
                                 (self.w.fishing,'at_port',True),
                                 (self.w.sailing,'running',None)):
            with self.subTest(name=name),patch.object(owner,name,return_value=value):
                self.assertIsNone(self.w._first_idle(frame,allow_running=True))
        for bad in (frame//2,frame[:900]):
            self.assertIsNone(self.w._first_idle(bad,allow_running=True))

    def test_expired_route_needs_explicit_retry_and_preserves_failed_bytes_and_all_counts(self):
        frame = self.lowered()
        self.w.journal.update(phase_deadline=self.now-1,attempts={'lower':1,'route-start':1},
                              issued_inputs=[dict(kind='route-start')])
        self.w._save();raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        with self.assertRaises(RuntimeError):self.tick(frame)
        self.assertEqual(self.path.read_bytes(),raw)
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        self.assertEqual(StoneFarm.request_retry(self.path.parent,digest)['mode'],'fishing_restore')
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        for _ in range(3):self.tick(frame)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertTrue(self.w.journal['first_recovery'])
        self.assertEqual(self.w.phase,'fishing_stop')
        self.assertEqual(self.w.journal['attempts']['lower'],1)
        self.assertEqual(self.w.journal['attempts']['route-start'],1)
        self.assertAlmostEqual(self.w.journal['deadline']-self.w.journal['started_at'],180)

    def test_lowered_task_requires_real_selection_receipt_and_three_seconds_off_before_cast(self):
        frame = self.lowered('first_idle')
        self.w.off_verified=True
        self.w.journal.update(first_signature=self.w.signature(frame),first_deadline=self.now+60)
        with patch.object(self.w.sailing,'running',return_value=False),patch.object(self.w,'_off_icon',return_value=True):
            key=self.w._first_idle(frame);self.assertIsNotNone(key)
            self.w.first_idle_key=key
            action=self.action('first-cast',(360,1010),frame)
            self.assertFalse(self.w.validate_portal(action,frame))
            self.w.journal.update(first_selected_rank=39,issued_inputs=[dict(kind='first-select')])
            self.assertFalse(self.w.validate_portal(action,frame))
            for _ in range(3):
                self.now+=1.6;self.w._confirmed(('first-idle',key))
            self.assertTrue(self.w.validate_portal(action,frame))
            for owner,name,value in ((self.w.fishing,'blue_fill',True),(self.w,'_battle',True),
                                     (self.w.sailing,'running',True),(self.w,'_rank_value',40)):
                with patch.object(owner,name,return_value=value):
                    self.assertFalse(self.w.validate_portal(action,frame))
            self.w.journal['first_selected_rank']=0
            self.assertFalse(self.w.validate_portal(action,frame))

    def test_lowered_task_still_cannot_drag_hold_or_cancel_boss(self):
        frame=self.lowered('fishing_prepare');self.w.off_verified=True
        self.w._prepare_fishing(frame)
        with patch.object(self.w.sailing,'running',return_value=False),patch.object(self.w,'_off_icon',return_value=True):
            self.assertFalse(self.w.meter_frame_allowed(frame))
            self.assertIsNone(self.w.cancel_context(frame))
        self.assertEqual(self.actions,[])

    def test_single_cast_limit_and_sixty_second_deadline_survive_restart(self):
        frame=self.lowered('first_idle')
        self.w.journal.update(first_deadline=self.now+60,first_signature=self.w.signature(frame))
        action=self.action('first-cast',(360,1010),frame)
        self.assertTrue(self.w.reserve_input(action))
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        with self.assertRaises(RuntimeError):self.w.reserve_input(action)
        self.now+=61
        self.assertFalse(self.w.input_allowed())
        with self.assertRaisesRegex(RuntimeError,'60秒'):self.w._first_step(frame)
        self.assertEqual(self.w.journal['attempts']['first-cast'],1)

    def test_fill_or_issued_tap_is_not_task_start_proof(self):
        frame=self.lowered('first_verify')
        self.w.journal.update(first_deadline=self.now+60,first_signature=self.w.signature(frame),
                              issued_inputs=[dict(kind='first-cast')])
        screen=dict(stage=8,rank=39,route=False,menu=False,completion=False)
        with patch.object(self.w,'screen',return_value=screen),patch.object(self.w,'_off_icon',return_value=True),patch.object(self.w.fishing,'blue_fill',return_value=True):
            for _ in range(3):self.tick(frame)
        self.assertEqual(self.w.phase,'first_verify')
        self.assertNotIn('first_cast_verified_at',self.w.journal)
        self.assertEqual(self.actions,[])

    def lowered_map(self):
        self.lowered('first_select')
        frame=image('tests/fixtures/stone-farm/stage8-rank39-selected-map.png')
        self.w.journal['first_signature']=self.w.signature(frame)
        return frame

    def test_real_lowered_map_uses_exposed_skull_parts_and_keeps_all_context_guards(self):
        frame=self.lowered_map()
        self.assertTrue(self.w._first_map_ready(frame))
        target=self.w._map_task_target(frame)
        self.assertEqual(target[:2],(197,485))
        self.assertGreaterEqual(target[2],.97)
        template=self.w.lower_map_core
        mask=np.zeros(template.shape[:2],bool)
        mask[8:27,19:31]=True
        mask[27:33,4:29]=True
        self.assertFalse(template[~mask].any())
        self.assertGreaterEqual(int(((template.min(axis=2)>=170)&mask).sum()),40)
        for owner,name,value in ((self.w,'_rank_value',38),(self.w,'_stage',9),
                                 (self.w,'_off_icon',False),(self.w,'_parallel',False),
                                 (self.w.training,'boss_mode',False)):
            with self.subTest(name=name),patch.object(owner,name,return_value=value):
                self.assertIsNone(self.w._map_task_target(frame))
        for bad in (frame//2,frame[:900]):
            self.assertIsNone(self.w._map_task_target(bad))
        self.assertEqual(self.actions,[])

    def test_missing_skull_eye_or_red_jaw_never_uses_avatar_as_task(self):
        frame=self.lowered_map()
        for rows,columns in ((slice(477,496),slice(201,213)),
                             (slice(496,502),slice(186,211))):
            changed=frame.copy()
            changed[rows,columns]=0
            self.assertIsNone(self.w._map_task_target(changed))
        # Altering the obscuring portrait outside the two skull parts must
        # not locate a different task or change its match score.
        changed=frame.copy()
        changed[469:496,182:201]=0
        target=self.w._map_task_target(changed)
        self.assertEqual(target[:2],(197,485))
        self.assertEqual(self.actions,[])

    def test_expired_lowered_map_is_archived_once_without_repeating_map_or_cast(self):
        frame=self.lowered_map()
        self.w.journal.update(phase_deadline=self.now-1,first_deadline=self.now-1,
                              attempts={'lower':1,'route-start':1,'first-map':1},
                              issued_inputs=[dict(kind='first-map')])
        self.w._save();raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        with self.assertRaises(RuntimeError):self.tick(frame)
        self.assertEqual(self.path.read_bytes(),raw)
        request=StoneFarm.request_retry(self.path.parent,digest)
        self.assertEqual(request['mode'],'first_map_restore')
        self.assertEqual(self.path.read_bytes(),raw)
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        for _ in range(3):self.tick(frame)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertEqual(self.w.journal['attempts'],{'lower':1,'route-start':1,'first-map':1})
        self.assertEqual(self.w.journal['first_map_repair_runs'],1)
        self.assertAlmostEqual(self.w.journal['deadline']-self.w.journal['started_at'],60)
        self.assertEqual(self.actions,[])
        self.now+=61
        self.assertFalse(self.w.input_allowed())
        with self.assertRaises(RuntimeError):self.tick(frame)
        with self.assertRaises(ValueError):StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_user_closed_map_reopens_only_from_same_dormant_task_with_original_attempt_count(self):
        frame=image('tests/fixtures/stone-farm/stage8-rank39-stopped-sleep.png')
        self.lowered('first_select')
        self.w.journal.update(first_signature=self.w.signature(frame),
                              phase_deadline=self.now-1,first_deadline=self.now-1,
                              attempts={'first-map':1,'route-start':1},
                              issued_inputs=[dict(kind='first-map')])
        self.w._save();raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        StoneFarm.request_retry(self.path.parent,digest)
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        for _ in range(2):self.tick(frame)
        self.assertEqual(self.actions,[])
        self.tick(frame)
        self.assertEqual(self.w.phase,'first_map')
        self.assertEqual(self.w.journal['attempts']['first-map'],1)
        self.assertEqual([a['kind'] for a in self.actions],['first-map'])
        self.assertTrue(self.w.validate_portal(self.actions[-1],frame))
        self.assertTrue(self.w.reserve_input(self.actions[-1]))
        self.assertEqual(self.w.journal['attempts']['first-map'],2)
        self.assertTrue(self.w.off_verified)
        self.assertAlmostEqual(self.w.journal['deadline']-self.w.journal['started_at'],60)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)

    def woken_task(self):
        self.lowered('first_idle')
        self.w.off_verified=True
        frame=image('tests/fixtures/stone-farm/stage8-rank39-woken-hook.png')
        self.w.journal.update(first_signature=self.w.signature(frame),first_selected_rank=39,
                              attempts={'first-select':1,'first-wake':1},
                              issued_inputs=[dict(kind='first-select'),dict(kind='first-wake')])
        return frame

    def test_popup_return_requires_three_seconds_of_same_stopped_woken_task(self):
        frame=self.woken_task()
        for _ in range(2):
            self.now+=1.6
            self.assertFalse(self.w.observe_task_popup_return(frame,notice_gone=True))
        self.now+=1.6
        self.assertTrue(self.w.observe_task_popup_return(frame,notice_gone=True))
        self.assertTrue(self.w.validate_portal(self.action('first-cast',(360,1010),frame),frame))
        self.assertEqual(self.actions,[])
        self.assertNotIn('first_cast_verified_at',self.w.journal)
        self.assertFalse(self.w.observe_task_popup_return(frame,notice_gone=False))
        self.now+=1.6
        self.assertFalse(self.w.observe_task_popup_return(frame,notice_gone=True))

    def test_popup_return_rejects_fill_world_rank_missing_receipt_and_expired_limits(self):
        frame=self.woken_task()
        for owner,name,value in ((self.w,'_rank_value',38),(self.w,'_stage',9),
                                 (self.w,'_off_icon',False),(self.w.fishing,'blue_fill',True),
                                 (self.w,'_battle',True),(self.w.fishing,'idle_icon','sleep')):
            with self.subTest(name=name),patch.object(owner,name,return_value=value):
                self.now+=1.6
                self.assertFalse(self.w.observe_task_popup_return(frame,notice_gone=True))
        saved=deepcopy(self.w.journal)
        for field,value in (('first_selected_rank',38),('issued_inputs',[]),
                            ('attempts',{'first-wake':1,'first-cast':1}),
                            ('phase_deadline',self.now-1)):
            self.w.journal=deepcopy(saved);self.w.journal[field]=value
            self.assertFalse(self.w.observe_task_popup_return(frame,notice_gone=True))
        self.assertEqual(self.actions,[])

    def test_wake_verification_starts_at_issued_wake_without_extending_total_deadline(self):
        frame=image('tests/fixtures/stone-farm/stage8-rank39-stopped-sleep.png')
        self.lowered('first_idle')
        self.w.journal.update(first_signature=self.w.signature(frame),first_selected_rank=39,
                              issued_inputs=[dict(kind='first-select')])
        original_deadline=self.w.journal['deadline']
        for _ in range(3):self.tick(frame)
        self.assertEqual([a['kind'] for a in self.actions],['first-wake'])
        self.assertAlmostEqual(self.w.journal['phase_deadline'],self.now+15)
        self.assertEqual(self.w.journal['deadline'],original_deadline)

    def test_interrupted_wake_retry_archives_once_and_never_repeats_selection_or_wake(self):
        frame=self.woken_task()
        self.w.journal.update(phase_deadline=self.now-1,first_deadline=self.now-1)
        self.w._save();raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        self.assertEqual(StoneFarm.request_retry(self.path.parent,digest)['mode'],'first_idle_restore')
        self.assertEqual(self.path.read_bytes(),raw)
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        for _ in range(3):self.tick(frame)
        self.assertEqual(self.actions,[])
        self.assertEqual(self.w.journal['attempts'],{'first-select':1,'first-wake':1})
        self.assertEqual(self.w.journal['first_idle_repair_runs'],1)
        self.assertAlmostEqual(self.w.journal['deadline']-self.w.journal['started_at'],60)
        self.assertAlmostEqual(self.w.journal['phase_deadline']-self.w.journal['started_at'],15)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.now+=61
        self.w._save()
        with self.assertRaises(ValueError):StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_worker_popup_return_resumes_native_task_without_legacy_controller(self):
        from app import Clicker
        from features import RECOVERY_RULE_IDS
        from types import SimpleNamespace
        frame=self.woken_task()
        notice=image('tests/fixtures/task-popup/parallel-notice.png')
        clock=[self.now]
        taps=[];resumed=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shutil.copytree(ROOT/'assets',root/'assets')
            config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            config['rules']=[deepcopy(r) for r in config['rules'] if r['id'] in
                             (*RECOVERY_RULE_IDS,'driftinggirl_stone_farm')]
            for rule in config['rules']:rule.update(enabled=True,next_delay=0)
            config['poll_interval']=.1
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            worker=self.w
            worker.journal.update(first_deadline=self.now+60,phase_deadline=self.now+15)
            class Device:
                def __init__(self,_,stop):self.stop=stop;self.image=None;self.reads=0
                def screenshot(self):
                    clock[0]+=.8;self.reads+=1
                    self.image=(frame if taps else notice).copy()
                    if self.reads>35:self.stop.set()
                    return self.image
                def click(self,button):taps.append(button.name)
                def stuck_record_add(self,_):pass
            app=Clicker(root,Device)
            app.digit_reader=self.reader
            def step(current):
                resumed.append(worker.validate_portal(self.action('first-cast',(360,1010),current),current))
                app.stop_event.set();return True
            worker.step=step
            try:
                with patch('app.StoneFarm',return_value=worker),patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])),patch('drift.tasks.stone_farm.time.time',side_effect=lambda:clock[0]),patch('drift.tasks.stone_farm.time.monotonic',side_effect=lambda:clock[0]):
                    app.start(preview=False);app.thread.join(12)
                self.assertFalse(app.running())
                self.assertIsNone(app.state['last_error'])
                self.assertEqual(taps,[RECOVERY_RULE_IDS[0]])
                self.assertEqual(resumed,[True])
                self.assertEqual(worker.journal['attempts'],{'first-select':1,'first-wake':1})
            finally:
                app.stop()
                if app.thread:app.thread.join(3)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

    def test_generic_return_rule_cannot_finish_stage_while_native_proof_is_pending(self):
        """Only the native observer may finish ``task_popup_return``.

        ``load_rule`` adds ``assets/task-popup/main-idle.png`` to the generic
        ``driftinggirl_task_popup_return`` variants, so once the notice is gone
        the ordinary scan matches that rule on the idle screen and declares the
        recovery finished while the native task proof (three seconds on the same
        stopped, woken task) is still pending.  The popup branch must let the
        native observer finish first; if someone moves that hold below the
        generic evidence, this case turns red.
        """
        from app import Clicker
        from features import RECOVERY_RULE_IDS
        from types import SimpleNamespace
        frame=self.woken_task()
        notice=image('tests/fixtures/task-popup/parallel-notice.png')
        clock=[self.now];taps=[];pending_stages=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shutil.copytree(ROOT/'assets',root/'assets')
            config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            config['rules']=[deepcopy(r) for r in config['rules'] if r['id'] in
                             (*RECOVERY_RULE_IDS,'driftinggirl_stone_farm')]
            for rule in config['rules']:rule.update(enabled=True,next_delay=0)
            config['poll_interval']=.1
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            worker=self.w
            worker.journal.update(first_deadline=self.now+60,phase_deadline=self.now+15)
            pending=[True,True,True]
            class Device:
                def __init__(self,_,stop):self.stop=stop;self.image=None;self.reads=0
                def screenshot(self):
                    clock[0]+=.8;self.reads+=1
                    self.image=(frame if taps else notice).copy()
                    if self.reads>18:self.stop.set()
                    return self.image
                def click(self,button):taps.append(button.name)
                def stuck_record_add(self,_):pass
            app=Clicker(root,Device)
            app.digit_reader=self.reader
            def popup_return_pending(current,*,notice_gone):
                if not pending:return False
                pending_stages.append(app.state['stage'])
                return pending.pop(0)
            def observe_task_popup_return(current,*,notice_gone):
                return not pending
            worker.popup_return_pending=popup_return_pending
            worker.observe_task_popup_return=observe_task_popup_return
            worker.step=lambda current:True
            try:
                with patch('app.StoneFarm',return_value=worker),patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])),patch('drift.tasks.stone_farm.time.time',side_effect=lambda:clock[0]),patch('drift.tasks.stone_farm.time.monotonic',side_effect=lambda:clock[0]):
                    app.start(preview=False)
                    app.thread.join(20)
                events='\n'.join(item['message'] for item in app.events)
                self.assertFalse(app.running())
                self.assertIsNone(app.state['last_error'])
                self.assertEqual(taps,[RECOVERY_RULE_IDS[0]])
                # Every frame that still owes the native proof stayed in the stage.
                self.assertEqual(pending_stages,['task_popup_return']*3)
                # The generic rule really does match the same idle screen.
                generic=app.load_rule(deepcopy(next(r for r in app.config['rules']
                    if r['id']==RECOVERY_RULE_IDS[1])))
                self.assertIn('assets/task-popup/main-idle.png',generic[0]['asset_variants'])
                # Only the native confirmation may report the recovery.
                self.assertIn('保留唤醒次数继续单次抛竿',events)
                for generic_event in ('主界面已恢复','平行世界地图','鱼饼强化面板','原餐品图鉴'):
                    self.assertNotIn(generic_event,events)
            finally:
                app.stop()
                if app.thread:app.thread.join(3)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

    def test_verified_lowered_rank39_cast_keeps_rank39_valid_for_route_reopen(self):
        frame=self.lowered('route_open')
        self.w.journal.update(first_cast_verified_at=self.now,first_cast_verified_rank=39)
        expected=dict(stage=8,rank=39,route=False,menu=False,completion=False)
        with patch.object(self.w,'screen',return_value=expected):
            self.assertEqual(self.w._expected(frame,entry=True),expected)
        for rank in (0,38,40):
            with patch.object(self.w,'screen',return_value={**expected,'rank':rank}):
                self.assertIsNone(self.w._expected(frame,entry=True))

    def test_verified_cast_route_retry_preserves_cast_and_never_returns_to_first_input(self):
        frame=image('tests/fixtures/stone-farm/stage8-rank39-restored-battle.png')
        self.lowered('route_open')
        self.w.journal.update(first_signature=self.w.signature(frame),
                              first_cast_verified_at=self.now-5,first_cast_verified_rank=39,
                              phase_deadline=self.now-1,attempts={'first-cast':1,'route-start':1},
                              issued_inputs=[dict(kind='first-cast')])
        self.w._save();raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        self.assertEqual(StoneFarm.request_retry(self.path.parent,digest)['mode'],'route_open_restore')
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        for _ in range(3):self.tick(frame)
        self.assertEqual(self.w.phase,'route_menu')
        self.assertEqual(self.w.journal['route_open_repair_runs'],1)
        self.assertEqual(self.w.journal['attempts'],{'first-cast':1,'route-start':1})
        self.assertAlmostEqual(self.w.journal['deadline']-self.w.journal['started_at'],90)
        self.assertTrue(all(a['kind']=='restore-route-open' for a in self.actions))
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.now+=91;self.w._save()
        with self.assertRaises(ValueError):StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest())


if __name__=='__main__':unittest.main()
