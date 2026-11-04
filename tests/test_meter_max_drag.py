"""Actual MAX-1 drag failure and explicit bounded repair retry guards."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from boss_training import BossTraining
from digit_ocr import DigitReader
from fishing_meter import FishingMeter
from meter_calibration import MeterCalibration
from drift.runtime import Clicker

ROOT = Path(__file__).resolve().parents[1]

def frame(name):
    return np.asarray(Image.open(ROOT/'tests/fixtures/meter-max-drag'/name).convert('RGB'))

class MeterMaxDragTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.inputs = []
        self.training = BossTraining(ROOT,self.reader,lambda _:None,
                                    state_path=Path(self.temp.name)/'training.json')
        self.training.state.update(mode='farming',rank=37,sailing_sleep_recovery=True)
        self.meter = FishingMeter(ROOT,self.training,lambda *a:self.inputs.append(a),lambda _:None)
        clock = patch('fishing_meter.time.time',return_value=1791724410)
        clock.start();self.addCleanup(clock.stop)
        self.range = self.meter.read(frame('range-before.png'))

    def test_actual_last_two_inputs_were_not_full_maximum(self):
        self.assertEqual((self.range['low'],self.range['high']),(1195,1421))
        for name in ('stuck-before.png','stuck-after.png'):
            with self.subTest(name=name):
                image = frame(name)
                data = self.meter.read(image)
                self.assertEqual((data['value'],data['y'],data['high']),(1420,935,1421))
                self.assertEqual(data['y'],self.meter.TOP-1) # old input had zero travel
                self.assertFalse(self.meter.close_enough(data))
                self.assertFalse(self.meter.validate('start',image,data))
        self.assertEqual(self.inputs,[])

    def test_actual_stuck_handle_gets_sufficient_travel_and_keeps_numeric_guard(self):
        image = frame('stuck-before.png');data = self.meter.read(image)
        plan = self.meter.plan(data)
        self.assertEqual(plan['goal'],data['player'])
        self.assertGreater(plan['to_y'],data['y'])
        self.assertTrue(self.meter.validate('drag',image,plan))
        self.assertFalse(self.meter.validate('start',image,data))
        self.assertFalse(self.meter.validate('drag',image,{**plan,'to_y':919}))
        self.assertEqual(self.inputs,[])

    def test_actual_near_maximum_also_avoids_a_short_repeat(self):
        image = frame('near-before.png');data = self.meter.read(image)
        self.assertEqual(data['value'],1412)
        self.assertLessEqual(data['y'],self.meter.TOP+self.meter.ENDPOINT_SLOP)
        plan = self.meter.plan(data)
        self.assertGreater(plan['to_y'],data['y'])
        self.assertTrue(self.meter.validate('drag',image,plan))

    def test_hidden_or_expired_upper_limit_is_revealed_before_any_max_drag(self):
        self.meter.range_cache = {}
        image = frame('stuck-before.png');data = self.meter.read(image)
        self.assertIsNone(data['high'])
        self.assertEqual(self.meter.plan(data)['kind'],'reveal')
        self.assertFalse(self.meter.validate('start',image,data))

    def test_changed_rank_checkbox_or_filled_dial_rejects_the_real_plan(self):
        image = frame('stuck-before.png');plan = self.meter.plan(self.meter.read(image))
        for box in ((72,12,130,50),(276,881,310,913)):
            changed=image.copy();x,y,X,Y=box;changed[y:Y,x:X]=0
            self.assertFalse(self.meter.validate('drag',changed,plan))
        filled=np.asarray(Image.open(ROOT/'tests/fixtures/fishing-meter/night-hold-after.png').convert('RGB'))
        self.assertFalse(self.meter.validate('drag',filled,plan))
        self.assertEqual(self.inputs,[])

class MeterRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        # 这些用例的差距取自旧阈值 50 时代的 fixture；生产阈值（GAP）另调，
        # 这里只验证"未输入/拖动耗尽可申请一次修复"的判定链。
        gap=patch.object(MeterCalibration,'GAP',50);gap.start();self.addCleanup(gap.stop)
        self.root=Path(self.temp.name)
        self.path=self.root/'meter_calibration_state.json'
        self.journal=dict(version=1,pending=True,rank=37,stage=9,restore_running=True,
            deadline=900,attempts={'drag':4,'cancel':1},sailing_attempts={'stop':1},
            stopped_context='place',stage_strokes=b'stage'.hex(),cycles=5,
            resume_requested=False,training={})
        self.config=dict(rules=[dict(id='driftinggirl_meter_calibration',enabled=True)])
        self.write()

    def write(self):
        self.path.write_text(json.dumps(self.journal),encoding='utf-8')
        (self.root/'config.json').write_text(json.dumps(self.config),encoding='utf-8')
        self.raw=self.path.read_bytes();self.digest=hashlib.sha256(self.raw).hexdigest()

    def request(self):
        with patch('meter_calibration.time.time',return_value=1000):
            return MeterCalibration.request_retry(self.root,self.digest)

    def controller(self):
        c=object.__new__(MeterCalibration)
        c.path=self.path;c.retry_path=self.root/'meter_calibration_retry.json'
        c.retry=self.request();c.journal=deepcopy(self.journal)
        c.enabled=lambda:True;c.key=c.seen_at=None;c.seen=0;c.last_seen=0
        c.cycles=5;c.context='place';c.off_verified=False;c.events=[];c.event=c.events.append
        c.world=SimpleNamespace(training=SimpleNamespace(state={}),
                                _stage_strokes=lambda f:b'stage')
        c.data=dict(context='place',value=1420,player=1318,low=1195,high=None)
        c.fishing=SimpleNamespace(measure=lambda f,remember:c.data,idle_icon=lambda f:'hook',
            busy=lambda f:False,cast_data=lambda f:None,range_cache={})
        c.screen=lambda f:dict(rank=37,stage=9,menu=False,route=False)
        c.phase='stop';c.resumed_pending=True;c.interval=60
        return c

    def apply(self,c,now=1000,monotonic=100):
        with patch('meter_calibration.time.time',return_value=now), \
             patch('meter_calibration.time.monotonic',return_value=monotonic):
            return c._apply_retry(np.zeros((1280,720,3),dtype=np.uint8))

    def test_request_does_not_touch_failed_attempts_or_deadline(self):
        request=self.request()
        self.assertEqual(self.path.read_bytes(),self.raw)
        self.assertEqual(request['failed_sha256'],self.digest)
        with self.assertRaisesRegex(ValueError,'已有'):
            self.request()

    def test_unexpired_incomplete_started_disabled_and_conflicting_records_reject(self):
        cases=[{'deadline':1001},{'attempts':{'drag':3}},
               {'attempts':{'drag':4,'start':1}},{'resume_requested':True},
               {'pending':False},{'stopped_context':None}]
        for change in cases:
            original=deepcopy(self.journal);self.journal.update(change);self.write()
            with self.assertRaises(ValueError):self.request()
            self.assertEqual(self.path.read_bytes(),self.raw)
            self.journal=original
        self.config['rules'][0]['enabled']=False;self.write()
        with self.assertRaises(ValueError):self.request()
        self.config['rules'][0]['enabled']=True;self.write()
        with patch('meter_calibration.time.time',return_value=1000),self.assertRaises(ValueError):
            MeterCalibration.request_retry(self.root,'0'*64)

    def test_three_fresh_seconds_archive_exact_failure_and_start_a_separate_budget(self):
        c=self.controller()
        self.assertFalse(self.apply(c,monotonic=100))
        self.assertFalse(self.apply(c,monotonic=102))
        self.assertEqual(self.path.read_bytes(),self.raw)
        self.assertTrue(self.apply(c,now=1004,monotonic=104))
        archive=self.root/'logs/meter-calibration'/('failed-'+self.digest+'.json')
        self.assertEqual(archive.read_bytes(),self.raw)
        self.assertEqual(json.loads(archive.read_bytes())['attempts']['drag'],4)
        self.assertEqual(c.journal['deadline'],1094)
        self.assertEqual(c.journal['attempts'],{})
        self.assertIsNone(c.journal['high']) # stale MAX cannot replace the hidden label
        self.assertEqual(c.phase,'prepare');self.assertTrue(c.off_verified)
        self.assertFalse(c.retry_path.exists())
        with patch('meter_calibration.time.time',return_value=1010):c._set('prepare')
        self.assertEqual(c.journal['deadline'],1094)

    def test_bad_or_occluded_frame_cannot_inherit_earlier_idle_proof(self):
        c=self.controller();self.assertFalse(self.apply(c,monotonic=100))
        c.fishing.busy=lambda f:True
        self.assertFalse(self.apply(c,monotonic=102));self.assertEqual(c.seen,0)
        c.fishing.busy=lambda f:False
        self.assertFalse(self.apply(c,monotonic=104))
        self.assertFalse(self.apply(c,monotonic=105))
        self.assertEqual(self.path.read_bytes(),self.raw)

    def test_running_menu_unknown_stage_and_location_cannot_retry(self):
        c=self.controller()
        for screen in (None,dict(rank=37,stage=9,menu=True,route=False),
                       dict(rank=37,stage=9,menu=False,route=True)):
            c.screen=lambda f,value=screen:value
            self.assertFalse(self.apply(c));self.assertEqual(self.path.read_bytes(),self.raw)
        c.screen=lambda f:dict(rank=37,stage=9,menu=False,route=False)
        c.data['context']='other';self.assertFalse(self.apply(c))
        c.data['context']='place';c.world._stage_strokes=lambda f:None
        self.assertFalse(self.apply(c));self.assertEqual(self.path.read_bytes(),self.raw)

    def test_expired_request_record_changes_and_archive_conflict_are_fail_closed(self):
        c=self.controller()
        with self.assertRaises(RuntimeError):self.apply(c,now=1300)
        self.path.write_bytes(self.raw+b' ')
        with self.assertRaises(RuntimeError):self.apply(c)
        self.path.write_bytes(self.raw)
        archive=self.root/'logs/meter-calibration'/('failed-'+self.digest+'.json')
        archive.parent.mkdir(parents=True);archive.write_bytes(b'conflict')
        self.assertFalse(self.apply(c,monotonic=100))
        self.assertFalse(self.apply(c,monotonic=102))
        with self.assertRaises(RuntimeError):self.apply(c,monotonic=104)
        self.assertEqual(self.path.read_bytes(),self.raw)

    def test_retry_endpoint_requires_stopped_worker_and_exact_payload(self):
        app=object.__new__(Clicker);app.lock=threading.RLock();app.root=self.root
        app.event=lambda _:None;app.running=lambda:True
        with self.assertRaisesRegex(ValueError,'停止'):app.retry_meter_calibration({'failed_sha256':self.digest})
        app.running=lambda:False
        with self.assertRaises(ValueError):app.retry_meter_calibration({'failed_sha256':self.digest,'reset':True})
        with patch('meter_calibration.time.time',return_value=1000):
            self.assertEqual(app.retry_meter_calibration({'failed_sha256':self.digest})['failed_sha256'],self.digest)
        self.assertEqual(self.path.read_bytes(),self.raw)

    def pre_input(self):
        self.journal.update(attempts={},value=1409,trigger_player=1472,proof=None,
            selected_signature=dict(box=[210,1030,281,1071],digit_strokes='digits',world_strokes='world'))
        self.write()

    def test_failed_before_first_meter_input_can_register_without_changing_original(self):
        self.pre_input()
        self.assertEqual(self.request()['failed_sha256'],self.digest)
        self.assertEqual(self.path.read_bytes(),self.raw)
        with self.assertRaisesRegex(ValueError,'已有'):
            self.request()

    def test_pre_input_retry_requires_saved_gap_stop_signature_and_no_completed_start(self):
        self.pre_input()
        original=deepcopy(self.journal)
        for change in ({'trigger_player':1459},{'trigger_player':None},
                       {'selected_signature':None},{'sailing_attempts':{}},
                       {'proof':'paired_hold'},{'attempts':{'cancel':1}},
                       {'attempts':{'start':1}},{'resume_requested':True}):
            self.journal=deepcopy(original);self.journal.update(change);self.write()
            with self.assertRaises(ValueError):self.request()
            self.assertEqual(self.path.read_bytes(),self.raw)

    def pre_input_controller(self):
        self.pre_input()
        c=self.controller()
        c.cast=dict(context='place',value=1409,box=(210,1030,281,1071),
                    digit_strokes='digits',world_strokes='world')
        c.fishing.cast_data=lambda f:c.cast
        c.fishing.measure=lambda *a,**k:self.fail('Range must wait for idle proof')
        c.fishing.idle_icon=lambda f:None
        c.fishing.busy=lambda f:True
        return c

    def test_pre_input_retry_archives_exact_failure_before_new_bounded_cancel_workflow(self):
        c=self.pre_input_controller()
        self.assertFalse(self.apply(c,monotonic=100))
        self.assertFalse(self.apply(c,monotonic=102))
        self.assertEqual(self.path.read_bytes(),self.raw)
        self.assertTrue(self.apply(c,now=1004,monotonic=104))
        archive=self.root/'logs/meter-calibration'/('failed-'+self.digest+'.json')
        self.assertEqual(archive.read_bytes(),self.raw)
        self.assertEqual(c.journal['deadline'],1094)
        self.assertEqual(c.journal['attempts'],{})
        self.assertIsNone(c.fishing.goal)
        self.assertTrue(c.fishing.saved_player_gap)
        self.assertFalse(c.fishing.start_fill_verified)
        self.assertIsNone(c.fishing.idle_since)
        self.assertEqual(c.fishing.starts,0)
        self.assertEqual(c.fishing.cancels,0)
        self.assertFalse(c.retry_path.exists())

    def test_pre_input_retry_cannot_use_changed_digits_place_or_unknown_running_state(self):
        c=self.pre_input_controller()
        original=deepcopy(c.cast)
        for key,value in (('digit_strokes','other'),('world_strokes','other'),
                          ('box',(211,1030,281,1071)),('context','other')):
            c.cast=deepcopy(original);c.cast[key]=value
            self.assertFalse(self.apply(c))
            self.assertEqual(c.seen,0)
            self.assertEqual(self.path.read_bytes(),self.raw)
        c.cast=original
        for route in (True,None):
            c.screen=lambda f,r=route:dict(rank=37,stage=9,menu=False,route=r)
            self.assertFalse(self.apply(c))
            self.assertEqual(self.path.read_bytes(),self.raw)

    def restore_only(self):
        self.journal.update(attempts={'cancel':1,'drag':2,'start':1},proof='paired_hold',
            sailing_attempts={'open':1},resume_requested=False,restore_retry=None)
        self.write()

    def restore_controller(self):
        self.restore_only()
        c=self.controller()
        c.screen=lambda f:dict(rank=37,stage=9,menu=True,route=None)
        c.settings_key=lambda f:((True,False,True,True,True),'choices')
        return c

    def test_restore_only_archives_without_resetting_meter_deadline_or_any_attempt(self):
        c=self.restore_controller()
        self.assertFalse(self.apply(c,monotonic=100))
        self.assertFalse(self.apply(c,monotonic=102))
        self.assertTrue(self.apply(c,now=1004,monotonic=104))
        archive=self.root/'logs/meter-calibration'/('failed-'+self.digest+'.json')
        self.assertEqual(archive.read_bytes(),self.raw)
        self.assertEqual(c.journal['deadline'],900)
        self.assertEqual(c.journal['attempts'],{'cancel':1,'drag':2,'start':1})
        self.assertEqual(c.journal['sailing_attempts'],{'open':1})
        self.assertEqual(c.journal['restore_retry']['deadline'],1044)
        self.assertEqual(c.phase,'menu')
        with patch('meter_calibration.time.time',return_value=1010):
            for action in ('cancel','drag','reveal','start'):
                self.assertFalse(c.reserve_meter_input(action))
            for action in ('open','stop'):
                self.assertFalse(c.reserve_sailing_input(action))
            self.assertTrue(c.reserve_sailing_input('start'))
            with self.assertRaises(RuntimeError):c.reserve_sailing_input('start')
        self.assertEqual(c.journal['deadline'],900)
        self.assertEqual(c.journal['attempts'],{'cancel':1,'drag':2,'start':1})
        with self.assertRaises(ValueError):self.request()

    def test_restore_only_request_rejects_missing_proof_prior_start_and_repeat(self):
        self.restore_only();original=deepcopy(self.journal)
        for change in ({'proof':None},{'restore_running':False},
                       {'sailing_attempts':{'open':1,'start':1}},
                       {'resume_requested':True},{'restore_retry':{'deadline':1040}}):
            self.journal=deepcopy(original);self.journal.update(change);self.write()
            with self.assertRaises(ValueError):self.request()
            self.assertEqual(self.path.read_bytes(),self.raw)

    def test_restore_only_requires_new_full_menu_and_stable_current_choices(self):
        c=self.restore_controller()
        self.assertFalse(self.apply(c,monotonic=100))
        c.settings_key=lambda f:((True,False,False,True,True),'changed')
        self.assertFalse(self.apply(c,monotonic=102))
        self.assertEqual(c.seen,1)
        c.screen=lambda f:None
        self.assertFalse(self.apply(c,monotonic=104))
        self.assertEqual(c.seen,0)
        self.assertEqual(self.path.read_bytes(),self.raw)

    def test_restore_only_timeout_keeps_failed_meter_state_and_cannot_open_new_budget(self):
        c=self.restore_controller()
        for now,mono in ((1000,100),(1002,102),(1004,104)):
            self.apply(c,now=now,monotonic=mono)
        c.expired_observe_until=115
        c._resolve_expired=lambda f:False
        before=self.path.read_bytes()
        with patch('meter_calibration.time.time',return_value=1044):
            self.assertFalse(c.step(np.zeros((1280,720,3),dtype=np.uint8)))
        self.assertTrue(c.awaiting_recovery)
        self.assertEqual(Path(c.journal['retirement']['archive']).read_bytes(),before)
        self.assertEqual(c.journal['attempts'],json.loads(before)['attempts'])
        self.assertEqual(c.journal['deadline'],json.loads(before)['deadline'])
        with self.assertRaises(ValueError):self.request()

if __name__=='__main__':unittest.main()
