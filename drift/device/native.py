"""Fresh MuMu SDK screenshots and guarded touches in its desktop session.

Interface reference: vendor/alas/module/device/method/nemu_ipc.py (GPL-3.0).
Each capture owns its handle and buffer, so a timed-out read cannot publish
pixels into a later request. Touches require a recent, matching native capture.

原名 ``native_capture.py``,现为设备层原生通道;根模块仍可导入(兼容代理)。
"""
import ctypes
import os
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np


class NativeUnavailable(Exception):
    pass


class MuMuCapture:
    def __init__(self, config, loader=None):
        # emulator-5554 is the verified primary MuMu instance. Do not guess a
        # different device/instance mapping from a similar screen resolution.
        if config['serial'] != 'emulator-5554':
            raise NativeUnavailable('未验证的模拟器实例')
        adb = Path(config['adb_path'])
        dll = adb.parent/'sdk'/'external_renderer_ipc.dll'
        if loader is None and (os.name != 'nt' or adb.parent.name != 'nx_main' or not dll.is_file()):
            raise NativeUnavailable('MuMu 原生截图接口不可用')
        self.folder = str(adb.parent.parent)
        self.resolution = tuple(config['resolution'])
        self._capture_lock = threading.Lock()
        self._capture_worker = None
        self.lib = (loader or ctypes.CDLL)(str(dll))
        self.lib.nemu_connect.argtypes = [ctypes.c_wchar_p,ctypes.c_int]
        self.lib.nemu_connect.restype = ctypes.c_int
        self.lib.nemu_disconnect.argtypes = [ctypes.c_int]
        self.lib.nemu_disconnect.restype = None
        self.lib.nemu_get_display_id.argtypes = [ctypes.c_int,ctypes.c_char_p,ctypes.c_int]
        self.lib.nemu_get_display_id.restype = ctypes.c_int
        self.lib.nemu_capture_display.argtypes = [ctypes.c_int,ctypes.c_uint,ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_ubyte)]
        self.lib.nemu_capture_display.restype = ctypes.c_int

    def _capture(self, result):
        handle = 0
        try:
            handle = self.lib.nemu_connect(self.folder,0)
            if handle <= 0:
                raise NativeUnavailable('MuMu 原生截图连接失败')
            display = self.lib.nemu_get_display_id(handle,b'com.daerigame.girladriftrm',0)
            if display < 0:
                raise NativeUnavailable('MuMu 当前游戏显示不可用')
            width,height = ctypes.c_int(),ctypes.c_int()
            ret = self.lib.nemu_capture_display(handle,display,0,
                ctypes.pointer(width),ctypes.pointer(height),None)
            if ret != 0:
                raise NativeUnavailable('MuMu 原生尺寸查询失败')
            shape = (width.value,height.value)
            if shape != self.resolution:
                raise RuntimeError(f'MuMu 原生截图尺寸 {shape} 与配置 {self.resolution} 不符')
            buffer = (ctypes.c_ubyte*(shape[0]*shape[1]*4))()
            ret = self.lib.nemu_capture_display(handle,display,len(buffer),
                ctypes.pointer(width),ctypes.pointer(height),buffer)
            if ret != 0:
                raise NativeUnavailable(f'MuMu 原生截图失败：{ret}')
            if (width.value,height.value) != shape:
                raise RuntimeError('MuMu 截图期间分辨率变化')
            rgba = np.ctypeslib.as_array(buffer).reshape(shape[1],shape[0],4)
            # Use vectorized channel conversion while retaining exactly the
            # same fresh RGB pixels, vertical orientation and owned memory.
            result.put(cv2.flip(cv2.cvtColor(rgba,cv2.COLOR_RGBA2RGB),0))
        except Exception as error:
            result.put(error)
        finally:
            if handle > 0:
                self.lib.nemu_disconnect(handle)

    @property
    def capture_pending(self):
        worker = self._capture_worker
        return worker is not None and worker.is_alive()

    def capture(self, stop, timeout=.25):
        # A timed-out SDK call may still own its handle and private buffer.
        # Keep that worker visible to later probes instead of piling up more
        # reads. Once it exits, the next call must capture NEW pixels.
        with self._capture_lock:
            if stop.is_set():
                raise InterruptedError('已停止')
            if self.capture_pending:
                raise NativeUnavailable('上一次 MuMu 原生截图仍未结束')
            return self._capture_once(stop, timeout)

    def _capture_once(self, stop, timeout):
        if stop.is_set():
            raise InterruptedError('已停止')
        result = queue.Queue(maxsize=1)
        worker = threading.Thread(target=self._capture,args=(result,),daemon=True)
        self._capture_worker = worker
        worker.start()
        # A late result stays in this call's queue and is discarded. A later
        # probe never consumes it or uses it to authorize fallback input.
        worker.join(timeout)
        if stop.is_set():
            raise InterruptedError('已停止')
        if worker.is_alive():
            raise NativeUnavailable('MuMu 原生截图超时')
        frame = result.get_nowait()
        if isinstance(frame,Exception):
            raise frame
        return frame

    def touch(self, x, y, stop, captured_at, stopped=None, hold=.04):
        """Send one stationary contact, then always release it.

        This verified portrait display uses Android x/y directly. ALAS's
        landscape rotation must not be applied to a portrait screenshot.
        No timeout thread can issue a late input or retry an uncertain DOWN.
        """
        def cancelled():
            return stop.is_set() or bool(stopped and stopped())
        if cancelled():
            raise InterruptedError('已停止')
        if not (0<=x<self.resolution[0] and 0<=y<self.resolution[1]):
            raise RuntimeError('MuMu 触摸坐标超出新截图范围')
        if self.resolution[0]>=self.resolution[1]:
            raise NativeUnavailable('未验证横屏原生触摸映射')
        down=getattr(self.lib,'nemu_input_event_touch_down',None)
        up=getattr(self.lib,'nemu_input_event_touch_up',None)
        if down is None or up is None:
            raise NativeUnavailable('MuMu 原生触摸接口不可用')
        down.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
        up.argtypes=[ctypes.c_int,ctypes.c_int]
        down.restype=up.restype=ctypes.c_int
        handle=self.lib.nemu_connect(self.folder,0)
        if handle<=0:
            raise NativeUnavailable('MuMu 原生触摸连接失败')
        try:
            display=self.lib.nemu_get_display_id(handle,b'com.daerigame.girladriftrm',0)
            width,height=ctypes.c_int(),ctypes.c_int()
            code=self.lib.nemu_capture_display(handle,display,0,
                ctypes.pointer(width),ctypes.pointer(height),None)
            if code!=0 or (width.value,height.value)!=self.resolution or display<0:
                raise NativeUnavailable('MuMu 原生触摸尺寸或显示校验失败')
            if cancelled():
                raise InterruptedError('已停止')
            # Preparing a handle must not authorize a contact after the short
            # charging target has aged. A refused input is reacquired next frame.
            if time.monotonic()-captured_at>.12:
                return None
            nx,ny=int(x),int(y)
            metadata=dict(input_transport='mumu-native',native_xy=[nx,ny],
                hold_ms=round(hold*1000),touch_down_time=time.time())
            started=time.monotonic()
            try:
                if down(handle,display,nx,ny)!=0:
                    raise RuntimeError('MuMu 原生触摸按下失败；未重复或回退输入')
                metadata['down_call_ms']=round((time.monotonic()-started)*1000,2)
                stop.wait(hold)
            finally:
                metadata['touch_up_time']=time.time()
                if up(handle,display)!=0:
                    raise RuntimeError('MuMu 原生触摸释放失败；已停止后续输入')
            return metadata
        finally:
            self.lib.nemu_disconnect(handle)


