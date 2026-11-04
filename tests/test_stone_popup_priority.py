"""Retired completed-cycle journals cannot starve a configured task popup."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from app import Clicker
from digit_ocr import DigitReader
from tests.test_task_popup_arrival import image,ROOT,CLOSE,RETURN


class StonePopupPriorityTests(unittest.TestCase):
    def wait_stopped(self,app,limit=120):
        """Wait for the worker instead of guessing one wall-clock budget.

        The virtual clock advances one second per screenshot, so a healthy run is
        a few dozen frames; how long that takes in real time depends on whatever
        else the process is doing. Polling keeps the assertion about the virtual
        deadline honest, and the hard cap still fails loudly on a wedged worker.
        A worker that outlives the cap is stopped here on purpose: a leftover
        thread keeps reading the next case's patched clock and can stretch that
        case past its own budget, which is exactly the cross-talk this suite hit
        when three cases ran in one process.
        """
        end=time.monotonic()+limit
        while app.running() and time.monotonic()<end:time.sleep(.02)
        if app.running():
            app.stop()
            end=time.monotonic()+30
            while app.running() and time.monotonic()<end:time.sleep(.02)
        return not app.running()

    def run_case(self,enabled=True,changed=False):
        clock=[0.]
        modal=image('tests/fixtures/stone-first-result/popup-20261017.png')
        main=image('tests/fixtures/stone-first-result/rank1-20261017.png')
        class Device:
            def __init__(self,config,stop):
                self.stop=stop;self.io_lock=threading.RLock()
                self.n=0;self.taps=[];self.image=None
            def screenshot(self):
                clock[0]+=1;self.n+=1
                self.image=(main if self.taps else modal).copy()
                if changed and self.n>=4:
                    self.image=main.copy()
                    self.stop.set()
                return self.image
            def click(self,button):
                if button.name!=CLOSE:raise AssertionError('Unexpected input')
                self.taps.append((button.name,tuple(button.button)))
            def stuck_record_add(self,_):pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shutil.copytree(ROOT/'assets',root/'assets')
            config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            config['rules']=[deepcopy(r) for r in config['rules']
                             if r['id'] in (CLOSE,RETURN,'driftinggirl_stone_farm')]
            for r in config['rules']:
                r['enabled']=enabled if r['id'] in (CLOSE,RETURN) else True
                if r['id'] in (CLOSE,RETURN):r.update(interval=.2,next_delay=0)
            config['poll_interval']=.1
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            # Intentionally expired first-verify scope. The independent popup
            # closer may run, but it must not renew or complete this journal.
            journal=dict(version=1,pending=True,cycles=0,handoffs=7,recoveries=8,
                reason='cycle',rank=40,from_stage=8,to_stage=9,entry_rank=0,
                phase='first_verify',started_at=1,deadline=2,phase_deadline=2,
                first_deadline=2,return_deadline=2,rank40_verified=True,
                attempts={'first-cast':1},issued_inputs=[dict(kind='first-cast')])
            state=root/'stone_farm_state.json';state.write_text(json.dumps(journal))
            raw=state.read_bytes();app=Clicker(root,Device);app.digit_reader=DigitReader(ROOT)
            try:
                with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])):
                    app.start(preview=False,max_clicks=2,timeout=30)
                    # The worker owns a 30-second virtual deadline. Wait for it
                    # to end instead of betting on a wall-clock join budget.
                    self.assertTrue(self.wait_stopped(app),
                                    'worker did not stop within the virtual deadline')
                self.assertFalse(app.running())
                saved=json.loads(state.read_text(encoding='utf-8'))
                self.assertFalse(saved['pending'])
                self.assertEqual(Path(saved['retirement']['archive']).read_bytes(),raw)
                self.assertEqual(saved['retirement']['source'],'startup_expired')
                for key in set(journal)-{'pending','phase'}:
                    self.assertEqual(saved[key],journal[key],key)
                self.assertEqual(json.loads((root/'config.json').read_text()),config)
                if enabled and not changed:
                    self.assertEqual(app.device.taps,[(CLOSE,(627,412,665,449))])
                    self.assertIsNone(app.state['last_error'])
                    self.assertEqual(app.state['clicks'],1)
                else:
                    self.assertEqual(app.device.taps,[])
            finally:
                app.stop()
                if app.thread:app.thread.join(2)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

    def test_popup_closes_before_expired_cycle_and_failure_is_retained(self):self.run_case()
    def test_disabled_popup_is_not_silently_enabled(self):self.run_case(enabled=False)
    def test_changed_fresh_frame_is_not_clicked(self):self.run_case(changed=True)


if __name__=='__main__':unittest.main()
