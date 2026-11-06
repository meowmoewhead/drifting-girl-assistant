"""Real Maa jobs: once-only guarded execution, stop, scopes and bounded state."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from drift.core.maa_engine import MaaEngine, framework_check
from frame_runtime import FrameRuntime


class MaaEngineTests(unittest.TestCase):
    def setUp(self):
        self.stop=threading.Event()
        self.device=SimpleNamespace(screenshot=Mock(return_value=np.zeros((8,8,3),np.uint8)))
        self.engine=MaaEngine(self.device,self.stop)

    def tearDown(self):
        self.engine.close()

    def test_real_jobs_preserve_false_and_identity_without_extra_capture(self):
        call=Mock(return_value=False)
        self.assertIs(self.engine.call('no_ui',call),False)
        marker=object()
        self.assertIs(self.engine.call('controller',lambda:marker),marker)
        call.assert_called_once_with()
        self.device.screenshot.assert_not_called()
        self.assertEqual(self.engine.snapshot()['completed_jobs'],2)
        self.assertTrue(self.engine.snapshot()['version'].startswith('v5.14.3'))

    def test_callback_exception_is_propagated_once_without_fallback(self):
        call=Mock(side_effect=RuntimeError('uncertain DOWN'))
        with self.assertRaisesRegex(RuntimeError,'uncertain DOWN'):
            self.engine.call('input',call)
        call.assert_called_once_with()
        self.assertEqual(self.engine.snapshot()['failed_jobs'],1)
        self.assertEqual(self.engine.call('read_only',lambda:19),19)

    def test_nested_guard_stays_in_current_job_and_thread(self):
        threads=[]
        def inner():threads.append(threading.get_ident());return 42
        def outer():
            threads.append(threading.get_ident())
            return self.engine.call('new_frame_guard',inner)
        self.assertEqual(self.engine.call('battle',outer),42)
        self.assertEqual(threads[0],threads[1])
        self.assertNotEqual(threads[0],threading.get_ident())
        self.assertEqual(self.engine.snapshot()['completed_jobs'],1)

    def test_generic_maa_input_and_process_actions_cannot_bypass_guard(self):
        ctrl=self.engine._controller
        for name,args in (('click',(1,2)),('swipe',(1,2,3,4,100)),
                          ('touch_down',(0,1,2,1)),('touch_up',(0,)),
                          ('stop_app',('game',)),('start_app',('game',))):
            self.assertFalse(getattr(ctrl,name)(*args))
        self.device.screenshot.assert_not_called()

    def test_stop_before_dispatch_does_not_invoke_callback(self):
        call=Mock()
        self.stop.set();self.engine.request_stop()
        with self.assertRaises(InterruptedError):self.engine.call('input',call)
        call.assert_not_called()

    def test_stop_during_action_waits_for_release_without_repeating(self):
        entered,released=threading.Event(),threading.Event()
        count=[]
        def guarded():
            count.append('DOWN');entered.set()
            try:self.stop.wait(2)
            finally:count.append('UP');released.set()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(self.engine.call,'hold',guarded)
            self.assertTrue(entered.wait(2))
            self.stop.set();began=time.monotonic();self.engine.request_stop()
            self.assertLess(time.monotonic()-began,.2)
            with self.assertRaises(InterruptedError):future.result(timeout=2)
        self.assertTrue(released.is_set())
        self.assertEqual(count,['DOWN','UP'])

    def test_concurrent_submitters_never_share_pending_action(self):
        active=[];seen=[]
        def one(number):
            self.assertEqual(active,[])
            active.append(number);time.sleep(.01)
            seen.append(number);active.pop()
            return number
        with ThreadPoolExecutor(max_workers=2) as pool:
            values=list(pool.map(lambda n:self.engine.call('serial',one,n),range(8)))
        self.assertEqual(values,list(range(8)))
        self.assertEqual(sorted(seen),list(range(8)))

    def test_jobs_and_framework_history_are_bounded(self):
        for i in range(70):self.engine.call('poll',lambda:i)
        state=self.engine.snapshot()
        self.assertEqual(len(state['recent_jobs']),32)
        self.assertEqual(state['registered_nodes'],1)
        self.assertEqual(state['completed_jobs'],70)
        self.assertIsNone(self.engine._tasker.get_task_detail(state['recent_jobs'][0]['job_id']))

    def test_framework_self_check_never_needs_a_game(self):
        self.assertEqual(framework_check()['completed_jobs'],1)

    def test_callback_cannot_close_its_own_executor_and_deadlock(self):
        with self.assertRaisesRegex(RuntimeError,'MaaFramework'):
            self.engine.call('bad_close',self.engine.close)
        self.assertEqual(self.engine.call('still_available',lambda:True),True)

    def test_status_is_available_while_native_shutdown_waits(self):
        original=self.engine._tasker.post_stop
        checked=[]
        def stop_job():
            job=original()
            def wait():
                # A different request thread must not wait for close's state lock.
                with ThreadPoolExecutor(max_workers=1) as pool:
                    checked.append(pool.submit(self.engine.snapshot).result(timeout=1))
                return job.wait()
            return SimpleNamespace(wait=wait)
        with patch.object(self.engine._tasker,'post_stop',side_effect=stop_job):
            self.engine.close()
        self.assertTrue(checked[0]['closed'])

    def test_new_frame_clears_ocr_scope_in_maa_callback_thread(self):
        local=threading.local()
        def begin():local.value=object()
        def end():local.value=None
        runtime=FrameRuntime(self.device,self.stop,lambda:SimpleNamespace(begin_frame=begin,end_frame=end))
        runtime.engine=self.engine
        runtime.capture()
        first=runtime.call('read_scope',lambda:local.value)
        self.assertIsNotNone(first)
        self.assertIsNot(runtime.call('read_again',lambda:local.value),first)
        runtime.capture()
        self.assertIsNot(runtime.call('read_new',lambda:local.value),first)
        def nested_capture():
            previous=local.value
            self.assertIs(runtime.call('same_action_read',lambda:local.value),previous)
            runtime.capture()
            return local.value is not previous
        self.assertTrue(runtime.call('input_guard',nested_capture))
