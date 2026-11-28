"""The fish/X stops continuous casting; the blood-bar X cannot replace it."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from digit_ocr import DigitReader
from drift.tasks.stone_farm import StoneFarm
from tests import test_stone_farm as helpers

ROOT, image = helpers.ROOT, helpers.image


class ContinuousStopTests(unittest.TestCase):
    setUp = helpers.StoneFarmTests.setUp
    tearDown = helpers.StoneFarmTests.tearDown
    tick = helpers.StoneFarmTests.tick
    begin = helpers.StoneFarmTests.begin
    act = helpers.StoneFarmTests.act
    action = helpers.StoneFarmTests.action

    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)
        cls.main = image('tests/fixtures/stone-farm/lower-seagod-battle.png')
        cls.dots = image('tests/fixtures/stone-farm/task-cast-dots.png')
        cls.bite = image('tests/fixtures/stone-farm/task-cast-bite.png')

    def prepare(self, frame=None):
        frame = self.dots if frame is None else frame
        self.begin()
        self.w._set('stop_fishing',180)
        self.w.off_verified = True
        context = self.w.cancel_context(frame,continuous=True)
        action = self.action('handoff-cancel',(690,1090),frame)
        action.update(continuous_stop=True,cancel_context=context)
        return action

    def test_direct_restart_refuses_task_preparation_and_battle(self):
        for frame in (self.dots,self.bite,self.main):
            with self.subTest(frame=self.w.fishing.idle_icon(frame)):
                action = self.prepare(frame)
                self.assertIsNone(action['cancel_context'])
                self.assertFalse(self.w.validate_portal(action,frame))
                self.tick(frame)
                self.assertEqual(self.actions,[])

    def test_task_preparation_does_not_relax_meter_or_ordinary_cancel_guards(self):
        action = self.prepare()
        for frame in (self.dots,self.bite,self.main):
            self.assertFalse(self.w.fishing.ordinary_preparation(frame,allow_fill=True))
            self.assertFalse(self.w.fishing.cancellable_cast(frame))
            self.w.phase='fishing_prepare'
            self.assertFalse(self.w.meter_frame_allowed(frame))
            self.w.phase='stop_fishing'
        action.pop('continuous_stop')
        self.assertFalse(self.w.validate_portal(action,self.dots))

    def test_completed_rank40_scope_and_new_handoffs_have_continuous_stop(self):
        self.begin(cycle=True)
        self.assertTrue(self.w.continuous_cancel_allowed)
        self.assertNotIn('allow_boss_abandon',self.w.journal)
        self.w.journal['rank40_verified']=False
        self.assertFalse(self.w.continuous_cancel_allowed)
        self.begin()
        self.assertTrue(self.w.continuous_cancel_allowed)

    def test_real_pale_rank40_digits_keep_exact_context_and_all_cancel_guards(self):
        names=('e4b02e249c194f89b4c815d47b7678f2','ec964e945ee344c89c2c226809f4ec4c',
               '69f52dce23b0412e917ddcd5dace57f9')
        frames=[image('tests/fixtures/stone-pale-stop/'+n+'.png') for n in names]
        self.begin(cycle=True)
        self.w._set('stop_fishing',180)
        self.w.off_verified=True
        context=self.w.cancel_context(frames[0],continuous=True)
        self.assertIsNotNone(context)
        action=self.action('handoff-cancel',(690,1090),frames[0])
        action.update(continuous_stop=True,cancel_context=context)
        for frame in frames:
            self.assertTrue(self.w.continuous_preparation(frame))
            self.assertTrue(self.w.same_cancel_context(action,frame))
            self.assertTrue(self.w.validate_portal(action,frame))
        changed=frames[0].copy()
        x,y,X,Y=context['box']
        hsv=cv2.cvtColor(changed[y+5:Y-3,x+4:X-7],cv2.COLOR_RGB2HSV)
        ink=((hsv[:,:,0]>=15)&(hsv[:,:,0]<=35)&(hsv[:,:,1]>60)&(hsv[:,:,2]>170))
        ys,xs=np.where(ink)
        changed[y+5+ys[0],x+4+xs[0]]=0
        self.assertFalse(self.w.same_cancel_context(action,changed))
        for box in ((194,4,246,42),(8,40,46,80),(65,9,139,63),
                    (40,45,68,71),(634,1052,720,1124),(300,950,420,1062)):
            changed=frames[0].copy();a,b,A,B=box;changed[b:B,a:A]=0
            self.assertFalse(self.w.validate_portal(action,changed))
        self.assertEqual(self.w.journal['attempts'],{})

    def test_pale_digit_probe_does_not_renew_expired_or_used_return_budget(self):
        frame=image('tests/fixtures/stone-pale-stop/e4b02e249c194f89b4c815d47b7678f2.png')
        self.begin(cycle=True)
        self.w._set('stop_fishing',180);self.w.off_verified=True
        context=self.w.cancel_context(frame,continuous=True)
        action=self.action('handoff-cancel',(690,1090),frame)
        action.update(continuous_stop=True,cancel_context=context)
        original=self.w.journal['deadline']
        self.w.journal['return_deadline']=self.now-1
        self.assertFalse(self.w.validate_portal(action,frame))
        self.assertEqual(self.w.journal['deadline'],original)
        self.assertEqual(self.w.journal['attempts'],{})

    def test_unknown_overlay_pauses_but_new_green_arrow_requires_fresh_stop(self):
        self.prepare()
        self.tick(self.dots//2)
        self.assertTrue(self.w.off_verified)
        self.assertEqual(self.actions,[])
        with patch.object(self.w.sailing,'running',return_value=True):
            self.tick()
        self.assertFalse(self.w.off_verified)
        self.assertEqual(self.w.phase,'stop')
        self.assertIsNone(self.w.cancel_anchor)

    def test_input_receipt_does_not_prove_stopped_and_two_attempts_survive_restart(self):
        action=self.prepare()
        self.w.reserve_input(action)
        self.w.input_issued(action,dict(input_id='a'*32,frame_id='b'*32,
            input_begin_time=self.now,input_transport='mumu-native'))
        self.assertFalse(self.w.fishing_stopped)
        self.tick(self.bite)
        self.assertFalse(self.w.fishing_stopped)
        self.w.reserve_input(action)
        self.actions.clear()
        self.tick(self.dots)
        self.assertEqual(self.actions,[])
        self.assertEqual(self.w.phase,'stop_fishing')
        resumed=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        self.assertEqual(resumed.journal['attempts']['handoff-cancel'],2)
        self.assertFalse(resumed.off_verified)
        self.assertIsNone(resumed.cancel_anchor)

    def test_only_three_seconds_actual_idle_allows_return(self):
        self.prepare()
        idle=image('tests/fixtures/stone-farm/stage9-rank39-0.png')
        with patch.object(self.w,'_expected',return_value=dict(stage=9,rank=39,menu=False,route=False)), \
                patch.object(self.w,'_idle_fishing',return_value=True):
            self.tick(idle);self.tick(idle)
            self.assertFalse(self.w.fishing_stopped)
            self.tick(idle)
        self.assertTrue(self.w.fishing_stopped)
        self.assertEqual(self.w.phase,'return')
        self.assertEqual(self.actions,[])

    def expired(self):
        self.prepare()
        self.w.journal.update(attempts={'stop':1,'handoff-cancel':1},
            issued_inputs=[dict(kind='stop',input_id='a'*32),
                           dict(kind='handoff-cancel',input_id='b'*32)])
        self.w.journal['return_deadline']=self.now-1
        self.w.journal['stop_fishing_deadline']=self.now-1
        self.w._save()
        (self.path.parent/'config.json').write_text(json.dumps(
            dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])),encoding='utf-8')
        raw=self.path.read_bytes()
        return raw,hashlib.sha256(raw).hexdigest()

    def test_explicit_repair_keeps_prior_contacts_and_archives_original_bytes(self):
        raw,digest=self.expired();prior=deepcopy(self.w.journal)
        request=StoneFarm.request_retry(self.path.parent,digest,allow_continuous_cancel=True)
        self.assertEqual(request['mode'],'continuous_cast_stop_retry')
        self.assertEqual(self.path.read_bytes(),raw)
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        self.tick();self.tick();self.tick()
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        for field in ('attempts','inputs','issued_inputs','started_at','cycles','handoffs'):
            self.assertEqual(self.w.journal[field],prior[field],field)
        self.assertTrue(self.w.continuous_cancel_allowed)
        self.assertFalse(self.w.off_verified)
        self.assertEqual(self.actions,[])
        self.now=self.w.journal['return_deadline']+1;self.w._save()
        with self.assertRaises(ValueError):
            StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest(),
                                    allow_continuous_cancel=True)

    def test_bad_hash_scope_combined_flags_exhausted_budget_and_repeat_are_rejected(self):
        raw,digest=self.expired();original=deepcopy(self.w.journal)
        with self.assertRaises(ValueError):
            StoneFarm.request_retry(self.path.parent,digest,allow_continuous_cancel=True,
                                    allow_return_entry_restore=True)
        for updates in (dict(rank=34),dict(from_stage=10),dict(phase='return'),
                        dict(continuous_stop_repair_runs=1),
                        dict(attempts={**original['attempts'],'handoff-cancel':2})):
            with self.subTest(updates=updates):
                self.w.journal={**original,**updates};self.w._save()
                with self.assertRaises(ValueError):
                    StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest(),
                                            allow_continuous_cancel=True)
        self.w.journal=original;self.w._save()
        with self.assertRaises(ValueError):
            StoneFarm.request_retry(self.path.parent,'0'*64,allow_continuous_cancel=True)
        self.assertEqual(self.path.read_bytes(),raw)


if __name__ == '__main__':unittest.main()
