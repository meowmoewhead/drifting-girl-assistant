"""ADB 传输层:原始 screencap 帧优先,PNG 兜底。

``AdbDevice`` 原住 ``app.py``(第 80–221 行),此处为逐字搬迁:构造签名、
公开属性名与全部行为(重试次数、停止语义、原生降级、异常文案)保持不变。
``app.py`` 按原名再导出本类,``from app import AdbDevice`` 照旧可用。
"""
from __future__ import annotations

import subprocess
import logging
import sys
import threading
import time

import cv2
import numpy as np

from drift.core.util import rectangle
from drift.device import native as native_transport
from drift.device.native import MuMuCapture, NativeUnavailable

# 历史补丁位置:AdbDevice 原与 MuMuCapture 同住 app 命名空间,测试用
# ``patch('app.MuMuCapture')`` 替换它。设备层搬进 drift 后补丁只改 app 的绑定,
# 因此构造原生通道时按“app 侧补丁 > 包内模块(含 native_capture 代理)> 本模块导入”
# 的顺序解析类;app 未加载或未再导出该名字时自动回落到包内实现。
# 比较用 id():补丁值是 Mock,不能让比较运算落到被替换的对象上。
_IMPORTED_MUMU_CAPTURE = MuMuCapture


def _mumu_capture_class():
    """返回当前生效的 MuMu 原生截图类,兼容按原路径打的补丁。"""
    candidate = getattr(sys.modules.get("app"), "MuMuCapture", None)
    if candidate is not None and id(candidate) != id(_IMPORTED_MUMU_CAPTURE):
        return candidate
    return getattr(native_transport, "MuMuCapture", _IMPORTED_MUMU_CAPTURE)


class AdbDevice:
    """Native ADB transport; no Windows desktop coordinates or SSH credentials."""

    def __init__(self, config, stop=None):
        self.config = config
        self.stop = stop or threading.Event()
        self.io_lock = threading.RLock()
        self.image = None
        self.native_capture = None
        self.native_disabled = False
        self.native_retry_at = 0.
        self.native_failures = 0
        self.native_consecutive_failures = 0
        self.native_recoveries = 0
        self.native_last_error = None
        self.screenshot_transport = 'adb'

    def _native_failed(self, error):
        self.native_disabled = True
        self.native_last_error = str(error)
        self.native_failures += 1
        self.native_consecutive_failures += 1
        delay = min(60, 5 * 2 ** min(self.native_consecutive_failures - 1, 4))
        self.native_retry_at = time.monotonic() + delay
        logging.warning('原生截图通道暂不可用：%s；%s秒后只读复查，期间只使用新ADB帧', error, delay)

    def native_status(self):
        return dict(disabled=self.native_disabled, failures=self.native_failures,
                    consecutive_failures=self.native_consecutive_failures,
                    recoveries=self.native_recoveries, last_error=self.native_last_error,
                    retry_in_seconds=round(max(0., self.native_retry_at-time.monotonic()), 2)
                        if self.native_disabled else 0.,
                    capture_pending=bool(self.native_capture is not None
                                         and self.native_capture.capture_pending))

    def command(self, *args):
        completed = subprocess.run(
            [self.config["adb_path"], "-s", self.config["serial"], *args],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        error = completed.stderr.decode("utf-8", "replace").strip()
        # MuMu's ADB can report a missing device on stderr with exit code zero.
        if completed.returncode or any(line.lower().startswith(('error:', 'adb: error:'))
                                       for line in error.splitlines()):
            raise RuntimeError(error or "ADB 命令失败")
        return completed.stdout

    def touch_fast(self,x,y,captured_at,stopped):
        if self.stop.is_set() or stopped():
            raise InterruptedError('已停止')
        if self.screenshot_transport=='mumu-native' and self.native_capture is not None:
            try:
                return self.native_capture.touch(x,y,self.stop,captured_at,stopped)
            except NativeUnavailable as error:
                # This read/preparation failure issued no contact. Acquire a
                # new ADB frame before any fallback input, never repeat a DOWN.
                self._native_failed(error)
                return None
        self.command('shell','input','swipe',str(x),str(y),str(x),str(y),'80')
        return dict(input_transport='adb',hold_ms=80)

    def screenshot(self):
        # A stopped worker still permits a read-only screenshot from the UI.
        cancel_on_stop = not self.stop.is_set()
        with self.io_lock:
            for attempt in range(3):
                if cancel_on_stop and self.stop.is_set():
                    raise InterruptedError("已停止")
                raw = self.command("exec-out", "screencap", "-p")
                if cancel_on_stop and self.stop.is_set():
                    raise InterruptedError("已停止")
                try:
                    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
                except cv2.error:
                    bgr = None
                if bgr is not None:
                    height, width = bgr.shape[:2]
                    if [width, height] != self.config["resolution"]:
                        raise RuntimeError(f"截图尺寸 {width}x{height} 与配置 {self.config['resolution']} 不符")
                    self.image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    self.screenshot_transport = 'adb-png'
                    return self.image
                if attempt < 2:
                    delay = .2 * (attempt + 1)
                    if cancel_on_stop:
                        if self.stop.wait(delay):
                            raise InterruptedError("已停止")
                    else:
                        time.sleep(delay)
            raise RuntimeError("ADB 连续返回空或无效截图，请检查模拟器和设备连接")

    def screenshot_fast(self):
        """Use raw pixels for short battle/cancellation windows.

        Accept only complete RGBA/RGBX buffers with a known header. Other
        formats fall back to a new normal capture; never reuse an old image.
        """
        cancel_on_stop = not self.stop.is_set()
        with self.io_lock:
            if cancel_on_stop and self.stop.is_set():
                raise InterruptedError('已停止')
            if cancel_on_stop and (not self.native_disabled or time.monotonic() >= self.native_retry_at):
                try:
                    if self.native_capture is None:
                        self.native_capture = _mumu_capture_class()(self.config)
                    self.image = self.native_capture.capture(self.stop)
                    self.screenshot_transport = 'mumu-native'
                    if self.native_disabled:
                        self.native_recoveries += 1
                        logging.info('原生截图通道已通过新截图恢复；继续逐帧核对输入守卫')
                    self.native_disabled = False
                    self.native_consecutive_failures = 0
                    self.native_retry_at = 0.
                    return self.image
                except (NativeUnavailable,OSError) as error:
                    # Retain the SDK object: a timed-out read must finish
                    # before it permits another worker. No emulator restart,
                    # old-frame reuse, input retry, or deadline extension.
                    self._native_failed(error)
            raw = self.command('exec-out', 'screencap')
            if cancel_on_stop and self.stop.is_set():
                raise InterruptedError('已停止')
            if len(raw) >= 12:
                width, height, fmt = np.frombuffer(raw[:12], dtype='<u4')
                width, height, fmt = int(width), int(height), int(fmt)
                if 0 < width <= 16384 and 0 < height <= 16384 and fmt in (1, 2):
                    if [width, height] != self.config['resolution']:
                        raise RuntimeError(f"截图尺寸 {width}x{height} 与配置 {self.config['resolution']} 不符")
                    header = len(raw)-width*height*4
                    if header in (12, 16):
                        pixels = np.frombuffer(raw, dtype=np.uint8, offset=header)
                        self.image = pixels.reshape(height,width,4)[:,:,:3].copy()
                        self.screenshot_transport = 'adb-raw'
                        return self.image
            return self.screenshot()

    def hold_feedback(self, x, y, milliseconds):
        """Capture during a held gesture, before a fast catch has finished."""
        x, y, milliseconds = int(x), int(y), int(milliseconds)
        if not (0 <= x < 720 and 0 <= y < 1280 and 1000 <= milliseconds <= 3000):
            raise ValueError('钓鱼长按参数无效')
        with self.io_lock:
            if self.stop.is_set():
                raise InterruptedError('已停止')
            # This fixed screenshot file is private to the serialized fishing
            # gesture. Wait for input to release before returning or reading it.
            command = (f'input swipe {x} {y} {x} {y} {milliseconds} >/dev/null & '
                       'taskFishingPid=$!; sleep 0.6; '
                       'screencap -p /sdcard/codex-fishing-start.png; taskCaptureStatus=$?; '
                       'wait $taskFishingPid; [ $? -eq 0 ] && [ $taskCaptureStatus -eq 0 ]')
            self.command('shell', command)
            raw = self.command('exec-out','cat','/sdcard/codex-fishing-start.png')
            bgr = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR) if raw else None
            if bgr is None or bgr.shape[:2] != (1280,720):
                raise RuntimeError('钓鱼长按期间的截图无效')
            return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)

    def game_state(self):
        # 延迟 import(过渡期):game_recovery 仍在根目录,将来归位到 drift.tasks 后
        # 若其内部反向依赖 app/drift.runtime,模块级 import 会形成
        # app → drift.runtime → drift.device.adb → game_recovery 的环。放在调用点
        # 既消除该风险,又让 patch('game_recovery.game_state') 照旧生效。
        from drift.tasks.game_recovery import game_state
        # Exit history is diagnostic only. Recovery now needs just an online
        # emulator and the actual foreground/process state; an unsupported or
        # slow exit-info query must not block reopening the game.
        return game_state(self.command('shell', 'dumpsys', 'activity', 'activities'),
                          self.command('shell', 'ps', '-A', '-o', 'NAME'), '')

    def click(self, button):
        with self.io_lock:
            if self.stop.is_set():
                raise InterruptedError("已停止")
            x1, y1, x2, y2 = rectangle(button.button, *self.config["resolution"])
            x, y = str((x1 + x2) // 2), str((y1 + y2) // 2)
            if getattr(button,'name',None) in ('driftinggirl_guild_potion_select',
                               'driftinggirl_guild_material_select', 'driftinggirl_guild_close',
                               'driftinggirl_mailbox_close', 'driftinggirl_mailbox_limit_close',
                               'driftinggirl_task_popup_close'):
                # A zero-duration ADB tap can be lost by these animated menu
                # controls. One stationary contact, never an automatic retry.
                self.command('shell','input','swipe',x,y,x,y,'80')
            else:
                self.command('shell','input','tap',x,y)

    def stuck_record_add(self, button):
        # Repeated-click detection is handled by the runner, per run.
        pass
