"""An input-guard deadline must leave a visible error, without issuing input."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import Clicker
from meter_calibration import MeterCalibration
from tools.configure_meter_calibration import RULE
from tests import test_meter_calibration as controller_tests

ROOT=Path(__file__).resolve().parents[1]

class GuardDeadlineTests(unittest.TestCase):
    def run_guard(self, mode, *, fishing=False):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            shutil.copytree(ROOT/'assets',root/'assets')
            config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            config.update(rules=[{**deepcopy(RULE),'enabled':True}],poll_interval=.1)
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            original=json.dumps(dict(version=1,pending=True,rank=38,stage=9,restore_running=True,
                deadline=time.time()+90,attempts={'start':1},sailing_attempts={'stop':1})).encode()
            (root/'meter_calibration_state.json').write_bytes(original)
            image=np.asarray(Image.open(ROOT/'tests/fixtures/fishing-meter/sleep.png').convert('RGB'))
            contacts=[];reservations=[];controllers=[];expired_raw=[]
            class Device:
                def __init__(self,config,stop):
                    self.config,self.stop,self.image=config,stop,None
                    self.io_lock=threading.RLock()
                def screenshot(self):
                    self.image=image.copy();return self.image
                def click(self,button):contacts.append(button)
                def stuck_record_add(self,button):pass
            app=Clicker(root,Device);app.digit_reader=object()
            def make_controller(_,reader,meter_action,sailing_action,event,interval):
                c=MeterCalibration(root,reader,meter_action,sailing_action,event,interval)
                controllers.append(c);c.phase='open'
                def validate(action,fresh):
                    if mode=='expired':
                        c.journal['deadline']=0;c._save();expired_raw.append(c.path.read_bytes())
                    if mode=='user_stop':app.stop_event.set()
                    return mode!='refused'
                c.validate_sailing=validate
                c.reserve_sailing_input=lambda action:reservations.append(action) or True
                c._step=lambda frame:sailing_action('open',233,45,'test')
                if fishing:
                    c.world.sailing.running=lambda _:False
                    c.world.training.state['sailing_enabled']=False
                    c.meter_frame_allowed=lambda fresh:validate('start',fresh)
                    c.fishing.validate=lambda *args:True
                    c._step=lambda frame:meter_action('start',{'value':100,'goal':100,'player':100})
                c.screen=lambda *args,**kw:None
                return c
            try:
                with patch('app.MeterCalibration',side_effect=make_controller):
                    app.start(preview=False,timeout=1)
                    app.thread.join(8)
                self.assertFalse(app.running())
                error=app.state['last_error']
                diagnostics=list((root/'logs').glob('error-*.json'))
                if mode=='expired':
                    c=controllers[0]
                    self.assertTrue(c.awaiting_recovery)
                    self.assertEqual(Path(c.journal['retirement']['archive']).read_bytes(),expired_raw[0])
                    self.assertEqual(c.journal['attempts'],{'start':1})
                    self.assertEqual(c.journal['deadline'],0)
                    self.assertEqual(app.state['stop_reason'],'timeout')
                else:
                    self.assertEqual((root/'meter_calibration_state.json').read_bytes(),original)
                self.assertEqual(contacts,[])
                self.assertEqual(reservations,[])
                return error,[json.loads(p.read_text(encoding='utf-8')) for p in diagnostics]
            finally:
                app.stop()
                if app.thread:app.thread.join(3)
                for handler in list(app.log.handlers):
                    handler.close();app.log.removeHandler(handler)

    def test_expiry_during_fresh_sailing_guard_retires_and_worker_continues(self):
        error,diagnostics=self.run_guard('expired')
        self.assertIsNone(error)
        self.assertEqual(diagnostics,[])

    def test_explicit_user_stop_remains_a_clean_stop(self):
        error,diagnostics=self.run_guard('user_stop')
        self.assertIsNone(error)
        self.assertEqual(diagnostics,[])

    def test_expiry_during_fresh_meter_guard_also_continues_without_input(self):
        error,diagnostics=self.run_guard('expired',fishing=True)
        self.assertIsNone(error)
        self.assertEqual(diagnostics,[])

    def test_refused_frame_without_expiry_does_not_report_a_task_failure(self):
        error,diagnostics=self.run_guard('refused')
        self.assertIsNone(error)
        self.assertEqual(diagnostics,[])

class ExplicitRouteRestoreTests(unittest.TestCase):
    setUpClass=classmethod(controller_tests.MeterCalibrationTests.setUpClass.__func__)
    setUp=controller_tests.MeterCalibrationTests.setUp
    tearDown=controller_tests.MeterCalibrationTests.tearDown
    pending=controller_tests.MeterCalibrationTests.pending
    restart=controller_tests.MeterCalibrationTests.restart

    def saved_failure(self):
        c=self.pending('open')
        image=np.asarray(Image.open(ROOT/'tests/fixtures/meter-resume-deadline/stopped-battle.png').convert('RGB'))
        c.journal.update(rank=38,stage=9,deadline=1000,proof='post_release_fishing',
            attempts={'cancel':1,'drag':1,'start':1},sailing_attempts={'stop':1},
            resume_requested=False,restore_retry=None,
            stage_strokes=c.world._stage_strokes(image).hex())
        c.context=c.fishing._context(image);c.journal['stopped_context']=c.context
        c.world.training.state['rank']=38;c.world.rank=38
        c.retry_path=c.path.with_name('meter_calibration_retry.json')
        c.expired_observe_until=115;c._save()
        (c.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id=RULE['id'],enabled=True)])),encoding='utf-8')
        return c,image,c.path.read_bytes()

    def request(self,c,raw):
        import hashlib
        with patch('meter_calibration.time.time',return_value=1100):
            request=MeterCalibration.request_retry(c.path.parent,hashlib.sha256(raw).hexdigest())
        c.retry=request
        return request

    def confirm_request(self,c,image):
        with patch('meter_calibration.time.time',return_value=1100):
            for now in (100,101.5,103):
                with patch('meter_calibration.time.monotonic',return_value=now):c.step(image.copy())

    def test_no_request_cannot_extend_the_expired_window(self):
        c,image,raw=self.saved_failure()
        with patch('meter_calibration.time.time',return_value=1100), \
             patch('meter_calibration.time.monotonic',return_value=120):
            self.assertFalse(c.step(image))
        self.assertTrue(c.awaiting_recovery)
        self.assertEqual(Path(c.journal['retirement']['archive']).read_bytes(),raw)
        self.assertEqual(self.inputs,[])

    def test_hash_bound_request_keeps_failure_bytes_until_new_stopped_proof(self):
        c,image,raw=self.saved_failure();self.request(c,raw)
        with patch('meter_calibration.time.time',return_value=1100):
            for now in (100,101.5):
                with patch('meter_calibration.time.monotonic',return_value=now):c.step(image.copy())
        self.assertEqual(c.path.read_bytes(),raw)
        self.assertEqual(self.inputs,[])

    def test_requested_main_restore_archives_without_resetting_meter_or_counts(self):
        c,image,raw=self.saved_failure();self.request(c,raw);self.confirm_request(c,image)
        before=json.loads(raw)
        for key in ('deadline','attempts','sailing_attempts','resume_requested'):
            self.assertEqual(c.journal[key],before[key])
        retry=c.journal['restore_retry']
        self.assertEqual(Path(retry['archive']).read_bytes(),raw)
        self.assertEqual(retry['deadline'],1140)
        self.assertTrue(retry['from_stopped_main'])
        self.assertEqual(c.phase,'open')
        self.assertTrue(c.off_verified)
        self.assertEqual(self.inputs,[])
        with patch('meter_calibration.time.time',return_value=1100):
            for action in ('cancel','drag','start','reveal'):
                self.assertFalse(c.reserve_meter_input(action))
            self.assertFalse(c.reserve_sailing_input('stop'))
            self.assertTrue(c.validate_sailing('open',image))
            self.assertTrue(c.reserve_sailing_input('open'))
            self.assertFalse(c.validate_sailing('open',image))
            with self.assertRaisesRegex(RuntimeError,'上限'):c.reserve_sailing_input('open')

    def test_requested_window_expires_without_a_second_extension(self):
        c,image,raw=self.saved_failure();self.request(c,raw);self.confirm_request(c,image)
        saved=c.path.read_bytes()
        with patch('meter_calibration.time.time',return_value=1140):
            self.assertFalse(c.input_allowed())
            self.assertFalse(c.step(image))
        self.assertTrue(c.awaiting_recovery)
        self.assertEqual(Path(c.journal['retirement']['archive']).read_bytes(),saved)
        self.assertEqual(self.inputs,[])

    def test_restart_reconfirms_off_without_renewing_either_deadline(self):
        c,image,raw=self.saved_failure();self.request(c,raw);self.confirm_request(c,image)
        c=self.restart(c.journal,now=1101,monotonic=100)
        self.assertEqual(c.phase,'open');self.assertFalse(c.off_verified)
        with patch('meter_calibration.time.time',return_value=1101):
            for now in (100,101.5):
                with patch('meter_calibration.time.monotonic',return_value=now):c.step(image.copy())
            self.assertFalse(c.off_verified)
            with patch('meter_calibration.time.monotonic',return_value=103):c.step(image.copy())
        self.assertTrue(c.off_verified)
        self.assertEqual(c.journal['deadline'],1000)
        self.assertEqual(c.journal['restore_retry']['deadline'],1140)
        self.assertEqual(self.inputs,[])

    def test_running_changed_rank_world_or_place_cannot_consume_request(self):
        c,image,raw=self.saved_failure();self.request(c,raw)
        variants=[]
        for area in ((8,40,46,80),(72,12,130,50),(286,12,434,34),(194,4,246,42)):
            changed=image.copy();x,y,X,Y=area;changed[y:Y,x:X]=0;variants.append(changed)
        for changed in variants:
            self.assertIsNone(c._restore_repair_context(changed))
        self.assertEqual(c.path.read_bytes(),raw)
        self.assertEqual(self.inputs,[])

    def test_missing_verified_start_does_not_qualify_for_main_restore(self):
        for patch_value in (dict(proof=None),dict(attempts={'start':0}),
                            dict(restore_running=False),dict(resume_requested=True),
                            dict(restore_retry={'deadline':1140})):
            c,image,raw=self.saved_failure();c.journal.update(patch_value);c._save()
            with self.assertRaises(ValueError):self.request(c,c.path.read_bytes())
            self.assertEqual(self.inputs,[])

if __name__=='__main__':unittest.main()
