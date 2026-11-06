import ctypes
import struct
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from app import AdbDevice
from native_capture import MuMuCapture, NativeUnavailable


class Function:
    def __init__(self, call):self.call=call
    def __call__(self,*args):return self.call(*args)


class SDK:
    def __init__(self, pixels, ret=0, delay=0, dimensions=None):
        self.pixels,self.ret,self.delay,self.dimensions=pixels,ret,delay,dimensions
        self.calls,self.closed=0,[]
        self.nemu_connect=Function(lambda *_:11)
        self.nemu_disconnect=Function(self.closed.append)
        self.nemu_get_display_id=Function(lambda *_:0)
        self.nemu_capture_display=Function(self.capture)
        self.contacts=[]
        self.nemu_input_event_touch_down=Function(lambda *args:self.contacts.append(('down',args)) or 0)
        self.nemu_input_event_touch_up=Function(lambda *args:self.contacts.append(('up',args)) or 0)
    def capture(self,handle,display,length,width,height,buffer):
        h,w=self.pixels.shape[:2]
        width.contents.value,height.contents.value=self.dimensions or (w,h)
        if length:
            self.calls+=1
            if self.delay:time.sleep(self.delay)
            if not self.ret:ctypes.memmove(buffer,self.pixels.tobytes(),self.pixels.nbytes)
        return self.ret


class NativeCaptureTests(unittest.TestCase):
    def setUp(self):
        self.config=dict(adb_path='C:/MuMu/nx_main/adb.exe',serial='emulator-5554',resolution=[4,3])
        self.rgb=np.arange(36,dtype=np.uint8).reshape(3,4,3)
        self.raw=np.concatenate((self.rgb[::-1],np.full((3,4,1),255,np.uint8)),axis=2)

    def test_native_new_buffer_preserves_rgb_orientation_and_is_detached(self):
        sdk=SDK(self.raw.copy());capture=MuMuCapture(self.config,lambda _:sdk)
        first=capture.capture(threading.Event())
        np.testing.assert_array_equal(first,self.rgb)
        sdk.pixels[:,:,:3]=77
        second=capture.capture(threading.Event())
        self.assertTrue(np.all(second==77))
        np.testing.assert_array_equal(first,self.rgb)
        self.assertEqual(sdk.calls,2);self.assertEqual(sdk.closed,[11,11])

    def test_native_error_size_change_stop_and_late_calls_cannot_authorize_old_pixels(self):
        for sdk,error in ((SDK(self.raw,ret=4),NativeUnavailable),
                          (SDK(self.raw,dimensions=(3,4)),RuntimeError)):
            capture=MuMuCapture(self.config,lambda _:sdk)
            with self.assertRaises(error):capture.capture(threading.Event())
            self.assertEqual(sdk.closed,[11])
        sdk=SDK(self.raw,delay=.1);capture=MuMuCapture(self.config,lambda _:sdk)
        with self.assertRaises(NativeUnavailable):capture.capture(threading.Event(),timeout=.01)
        time.sleep(.12)
        self.assertEqual(sdk.closed,[11])
        stopped=threading.Event();stopped.set()
        with self.assertRaises(InterruptedError):capture.capture(stopped)
        self.assertEqual(sdk.calls,1)

    def test_other_instance_is_never_guessed(self):
        config={**self.config,'serial':'emulator-5556'}
        loader=lambda _:self.fail('Must not load another instance')
        with self.assertRaises(NativeUnavailable):MuMuCapture(config,loader)

    def test_native_portrait_touch_preserves_axes_and_releases_on_stop(self):
        sdk=SDK(np.zeros((4,3,4),np.uint8))
        capture=MuMuCapture({**self.config,'resolution':[3,4]},lambda _:sdk)
        stop=threading.Event()
        def down(*args):
            sdk.contacts.append(('down',args));stop.set();return 0
        sdk.nemu_input_event_touch_down=Function(down)
        result=capture.touch(2,1,stop,time.monotonic())
        self.assertEqual(sdk.contacts,[('down',(11,0,2,1)),('up',(11,0))])
        self.assertEqual(result['native_xy'],[2,1])
        self.assertEqual(result['input_transport'],'mumu-native')
        self.assertEqual(sdk.closed,[11])
        with self.assertRaises(InterruptedError):capture.touch(2,1,stop,time.monotonic())
        self.assertEqual(len(sdk.contacts),2)

    def test_stale_native_touch_is_reacquired_and_uncertain_input_never_falls_back(self):
        sdk=SDK(np.zeros((4,3,4),np.uint8))
        portrait={**self.config,'resolution':[3,4]}
        capture=MuMuCapture(portrait,lambda _:sdk)
        self.assertIsNone(capture.touch(2,1,threading.Event(),time.monotonic()-.2))
        self.assertEqual(sdk.contacts,[])
        sdk.nemu_input_event_touch_down=Function(lambda *args:sdk.contacts.append(('down',args)) or 1)
        device=AdbDevice(portrait);device.native_capture=capture;device.screenshot_transport='mumu-native'
        with patch.object(device,'command') as command,self.assertRaisesRegex(RuntimeError,'未重复'):
            device.touch_fast(2,1,time.monotonic(),lambda:False)
        command.assert_not_called()
        self.assertEqual(sdk.contacts,[('down',(11,0,2,1)),('up',(11,0))])

    def test_unverified_landscape_touch_does_not_issue_input(self):
        sdk=SDK(self.raw);capture=MuMuCapture(self.config,lambda _:sdk)
        with self.assertRaises(NativeUnavailable):capture.touch(2,1,threading.Event(),time.monotonic())
        self.assertEqual(sdk.contacts,[])

    def test_adapter_uses_native_and_fallback_is_a_new_adb_frame(self):
        device=AdbDevice(self.config)
        sdk=SDK(self.raw);capture=MuMuCapture(self.config,lambda _:sdk)
        with patch('app.MuMuCapture',return_value=capture),patch.object(device,'command') as command:
            np.testing.assert_array_equal(device.screenshot_fast(),self.rgb)
            command.assert_not_called()
        self.assertEqual(device.screenshot_transport,'mumu-native')
        previous=device.image
        fresh=np.full_like(self.raw,63)
        raw=struct.pack('<IIII',4,3,1,0)+fresh.tobytes()
        with patch.object(capture,'capture',side_effect=NativeUnavailable('read timeout')),\
             patch.object(device,'command',return_value=raw) as command:
            np.testing.assert_array_equal(device.screenshot_fast(),fresh[:,:,:3])
        command.assert_called_once_with('exec-out','screencap')
        self.assertTrue(device.native_disabled)
        self.assertEqual(device.screenshot_transport,'adb-raw')
        np.testing.assert_array_equal(previous,self.rgb)

    def test_timed_out_worker_must_finish_before_any_new_probe_and_late_pixels_are_discarded(self):
        release=threading.Event();entered=threading.Event()
        sdk=SDK(self.raw.copy());original=sdk.capture
        def blocked(*args):
            if args[2]:
                entered.set();release.wait(1)
            return original(*args)
        sdk.nemu_capture_display=Function(blocked)
        capture=MuMuCapture(self.config,lambda _:sdk)
        try:
            with self.assertRaises(NativeUnavailable):capture.capture(threading.Event(),timeout=.01)
            self.assertTrue(entered.is_set());self.assertTrue(capture.capture_pending)
            worker=capture._capture_worker
            with self.assertRaisesRegex(NativeUnavailable,'仍未结束'):
                capture.capture(threading.Event(),timeout=.01)
            self.assertIs(capture._capture_worker,worker)
            release.set();worker.join(.5)
            self.assertFalse(capture.capture_pending)
            sdk.pixels[:,:,:3]=92
            frame=capture.capture(threading.Event())
            self.assertTrue(np.all(frame==92))
            self.assertEqual(sdk.calls,2)
            self.assertEqual(sdk.closed,[11,11])
        finally:
            release.set()

    def test_transient_native_failure_reprobes_after_cooldown_without_reusing_fallback(self):
        device=AdbDevice(self.config)
        sdk=SDK(self.raw.copy());capture=MuMuCapture(self.config,lambda _:sdk)
        fallback=np.full_like(self.raw,63)
        raw=struct.pack('<IIII',4,3,1,0)+fallback.tobytes()
        with patch('app.MuMuCapture',return_value=capture),\
             patch.object(device,'command',return_value=raw) as command,\
             patch('drift.device.adb.time.monotonic',return_value=0) as clock:
            with patch.object(capture,'capture',side_effect=NativeUnavailable('temporary')) as probe:
                self.assertTrue(np.all(device.screenshot_fast()==63))
                clock.return_value=4.99
                self.assertTrue(np.all(device.screenshot_fast()==63))
                self.assertEqual(probe.call_count,1)
            clock.return_value=5
            sdk.pixels[:,:,:3]=94
            self.assertTrue(np.all(device.screenshot_fast()==94))
            self.assertEqual(command.call_count,2)
            self.assertIs(device.native_capture,capture)
            self.assertFalse(device.native_disabled)
            self.assertEqual(device.native_status()['recoveries'],1)
            self.assertEqual(device.screenshot_transport,'mumu-native')
            self.assertEqual(sdk.contacts,[])

    def test_persistent_failure_backs_off_to_sixty_seconds_and_stop_never_probes(self):
        device=AdbDevice(self.config);capture=MuMuCapture(self.config,lambda _:SDK(self.raw))
        raw=struct.pack('<IIII',4,3,1,0)+self.raw.tobytes()
        with patch('app.MuMuCapture',return_value=capture),\
             patch.object(device,'command',return_value=raw),\
             patch.object(capture,'capture',side_effect=NativeUnavailable('unavailable')) as probe,\
             patch('drift.device.adb.time.monotonic',return_value=0) as clock:
            for delay in (5,10,20,40,60,60):
                now=clock.return_value
                device.screenshot_fast()
                self.assertEqual(device.native_retry_at,now+delay)
                clock.return_value=now+delay-.01
                count=probe.call_count;device.screenshot_fast()
                self.assertEqual(probe.call_count,count)
                clock.return_value=now+delay
            device.stop.set();count=probe.call_count;device.screenshot_fast()
            self.assertEqual(probe.call_count,count)

    def test_touch_preparation_failure_requires_new_capture_before_recovery_input(self):
        device=AdbDevice(self.config);capture=MuMuCapture(self.config,lambda _:SDK(self.raw))
        device.native_capture=capture;device.screenshot_transport='mumu-native'
        with patch.object(capture,'touch',side_effect=NativeUnavailable('preparation')),\
             patch.object(device,'command') as command:
            self.assertIsNone(device.touch_fast(1,1,time.monotonic(),lambda:False))
            command.assert_not_called()
        self.assertTrue(device.native_disabled)
        self.assertIs(device.native_capture,capture)
        self.assertGreater(device.native_retry_at,time.monotonic())


if __name__=='__main__':unittest.main()
