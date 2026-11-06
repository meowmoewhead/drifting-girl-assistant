"""Bounded game launch and title-page recovery while the emulator is online."""
from collections import deque
from datetime import datetime
from pathlib import Path
import re
import time
from types import SimpleNamespace

import cv2
import numpy as np

PACKAGE = 'com.daerigame.girladriftrm'
COMPONENT = PACKAGE + '/com.unity3d.player.UnityPlayerActivity'


def game_state(activity, processes, exits):
    def text(value):
        return value.decode('utf-8', 'replace') if isinstance(value, bytes) else value
    activity, processes, exits = map(text, (activity, processes, exits))
    foreground=None
    for field in ('topResumedActivity','mResumedActivity','ResumedActivity'):
        foreground=re.search(r'\b'+field+r'\s*[:=]\s*ActivityRecord\{[^}\n]*\bu\d+\s+(\S+/\S+)',activity)
        if foreground:
            break
    records = []
    for block in re.split(r'ApplicationExitInfo #\d+:', exits)[1:]:
        process = re.search(r'process=(\S+)\s+reason=(\d+).*?status=(\d+)', block)
        stamp = re.search(r'timestamp=(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)', block)
        if process and stamp and process[1] == PACKAGE:
            try:
                records.append({'timestamp': datetime.fromisoformat(stamp[1]).timestamp(),
                                'reason': int(process[2]), 'status': int(process[3])})
            except ValueError:
                pass
    return {'foreground': foreground[1] if foreground else None,
            'running': any(name == PACKAGE or name.startswith(PACKAGE + ':')
                           for name in processes.split()),
            'exit': max(records, key=lambda r: r['timestamp']) if records else None}


def restartable(state):
    # A fresh screenshot and successful ADB state reads establish that MuMu is
    # online. The user authorizes opening the game from any foreground app,
    # regardless of the existence, reason or age of Android's exit record.
    # am start also brings an existing background game forward without killing
    # it. A live game with a temporarily unknown foreground only waits.
    foreground = state.get('foreground')
    return (state.get('running') is False
            or state.get('running') is True and bool(foreground)
            and not foreground.startswith(PACKAGE + '/'))


class GameRecovery:
    def __init__(self, root, device, event, click=None, capture=None):
        self.device, self.event = device, event
        self.capture = capture or (lambda: device.screenshot())
        self.click = click or (lambda x, y, _: self.device.click(SimpleNamespace(button=(x-5, y-5, x+5, y+5))))
        title = cv2.imread(str(Path(root)/'assets/recovery/login.png'))
        if title is None or title.shape[:2] != (1280, 720):
            raise RuntimeError('游戏恢复登录模板不完整')
        self.title = cv2.cvtColor(title, cv2.COLOR_BGR2RGB)
        network=cv2.imread(str(Path(root)/'assets/recovery/network-error.png'))
        if network is None or network.shape[:2] != (1280,720):
            raise RuntimeError('游戏恢复网络退出提示模板不完整')
        self.network=cv2.cvtColor(network,cv2.COLOR_BGR2RGB)
        self.root=Path(root)
        self.scene_probe=None
        self.phase = 'idle'
        self.missing = self.title_frames = self.ready_frames = 0
        self.started_at = None
        self.restarts = deque()
        self.login_clicked = False
        self.recoveries = 0
        self.network_frames=0
        self.awaiting_self_exit = False

    def scene(self, frame):
        """Recovery also works when only scheduled or reward tasks are enabled."""
        if self.scene_probe is None:
            from drift.tasks.meter_calibration import _ScreenContext
            from drift.tasks.training_sailing import TrainingSailing
            self.scene_probe=_ScreenContext(self.root,None,lambda:None)
            self.scene_probe.sailing=TrainingSailing(self.root,self.scene_probe.training,None,lambda *_:None)
        probe=self.scene_probe
        ready=probe.training.main(frame) or probe.sailing.running(frame) is not None
        return ready or probe._parallel(frame) or probe._observatory(frame),ready

    def _network_error(self, frame):
        if frame.shape[:2] != (1280,720):
            return False
        # The changing countdown digit is deliberately outside these fixed
        # words. Observe the game's own exit; never press its Exit button.
        for x,y,X,Y in ((194,541,547,574),(315,591,465,625),(280,764,428,811)):
            score=cv2.matchTemplate(frame[y:Y,x:X],self.network[y:Y,x:X],cv2.TM_CCOEFF_NORMED)[0,0]
            if not np.isfinite(score) or score<.95:
                return False
        return float((frame[477:520,140:580].min(axis=2)>=140).mean())>=.7

    def _wait_for_exit(self, reason, *, awaiting_self_exit=False):
        if self.started_at is None:
            self.started_at=time.monotonic()
        self.phase='loading'
        self.awaiting_self_exit = awaiting_self_exit
        self.missing=self.title_frames=self.ready_frames=0
        self.login_clicked=False
        self.event(reason+'；等待退出后自动拉起，仍沿用120秒总期限')

    def _login(self, frame):
        if frame.shape[:2] != (1280, 720):
            return False
        for box in ((133, 249, 589, 352), (270, 903, 451, 926)):
            x1, y1, x2, y2 = box
            source = self.title[y1:y2, x1:x2]
            ink = np.all(source >= 220, axis=2).astype(np.uint8)*255
            if float(np.all(frame[y1:y2, x1:x2][ink > 0] >= 210, axis=1).mean()) < .85:
                return False
            mask = np.repeat(cv2.dilate(ink, np.ones((3, 3), np.uint8))[:, :, None], 3, axis=2)
            score = cv2.matchTemplate(frame[y1:y2, x1:x2], source, cv2.TM_CCOEFF_NORMED, mask=mask)[0, 0]
            if not np.isfinite(score) or score < .90:
                return False
        return True

    def _stopped(self):
        if self.device.stop.is_set():
            raise InterruptedError('已停止')

    def _launch(self, state, *, retry=False):
        now = time.monotonic()
        if self.started_at is not None and retry and now-self.started_at>120:
            raise RuntimeError('游戏恢复等待超过 120 秒，已停止并保留当前画面')
        while self.restarts and now - self.restarts[0] >= 1800:
            self.restarts.popleft()
        if len(self.restarts) >= 2:
            raise RuntimeError('游戏在 30 分钟内已恢复两次，已停止以避免反复重启')
        fresh = self.device.game_state()
        self._stopped()
        if fresh != state or not restartable(fresh):
            self.missing = 0
            return False
        now = time.monotonic()
        if self.started_at is not None and retry and now-self.started_at>120:
            raise RuntimeError('游戏恢复等待超过 120 秒，已停止并保留当前画面')
        result = self.device.command('shell', 'am', 'start', '-n', COMPONENT)
        if b'Error:' in result or b'Exception' in result:
            raise RuntimeError('游戏启动失败，已停止')
        self.restarts.append(now)
        if not retry:
            self.started_at = now
        self.phase = 'loading'
        self.missing = self.title_frames = self.ready_frames = 0
        self.login_clicked = False
        action = '切回后台游戏' if fresh['running'] else '打开已退出的游戏'
        self.event('游戏恢复：模拟器在线，'+action+'；30 分钟最多两次')
        return True

    def step(self, frame, known, ready):
        """Return wait/ready while recovering; None leaves ordinary workflows alone."""
        self._stopped()
        if self.phase == 'idle':
            self.network_frames=self.network_frames+1 if self._network_error(frame) else 0
            if self.network_frames:
                if self.network_frames>=3:
                    self._wait_for_exit('游戏恢复：连续确认网络错误退出提示', awaiting_self_exit=True)
                return 'wait'
            if known:
                self.missing = self.title_frames = 0
                self.started_at=None
                self.awaiting_self_exit = False
                return None
            self.title_frames = self.title_frames + 1 if self._login(frame) else 0
            if self.title_frames >= 3:
                # A manually reopened game may already be on the start page.
                # Adopt only fresh, bright title text in the actual game.
                with self.device.io_lock:
                    state = self.device.game_state()
                    self._stopped()
                    fresh = self.capture()
                    self._stopped()
                    if (state['running'] and (state.get('foreground') or '').startswith(PACKAGE + '/')
                            and self._login(fresh)):
                        self.started_at = time.monotonic()
                        self.phase = 'loading'
                        self.login_clicked = False
                        self.ready_frames = self.missing = 0
                        self.event('游戏恢复：已确认开始页面，接续进入主画面')
                return 'wait'
            self.missing += 1
            if self.missing < 3:
                return 'wait'
            with self.device.io_lock:
                self._stopped()
                state = self.device.game_state()
                self._stopped()
                if state['running'] and (state.get('foreground') or '').startswith(PACKAGE + '/'):
                    self.missing=0
                    return None
                if not restartable(state):
                    if not state.get('foreground'):
                        self._wait_for_exit('游戏恢复：前台活动正在切换')
                        return 'wait'
                    raise RuntimeError('未能确认游戏进程状态，已停止并保留当前应用')
                self._launch(state)
            return 'wait'

        if time.monotonic() - self.started_at > 120:
            raise RuntimeError('游戏恢复等待超过 120 秒，已停止并保留当前画面')
        self.ready_frames = self.ready_frames + 1 if ready else 0
        if self.ready_frames >= 3:
            with self.device.io_lock:
                state=self.device.game_state()
                self._stopped()
                if not (state['running'] and (state.get('foreground') or '').startswith(PACKAGE+'/')):
                    self.ready_frames=0
                    return 'wait'
            self.phase, self.missing = 'idle', 0
            self.started_at=None
            self.awaiting_self_exit = False
            self.recoveries += 1
            self.event('游戏恢复：连续确认主画面，重新识别任务并续接保存的练级记录')
            return 'ready'
        self.title_frames = self.title_frames + 1 if self._login(frame) else 0
        self.missing = self.missing + 1 if not ready and not self.title_frames else 0
        if self.missing >= 3:
            with self.device.io_lock:
                self._stopped()
                state = self.device.game_state()
                self._stopped()
                if restartable(state):
                    # Retrying a launch never extends the original 120 seconds.
                    self._launch(state, retry=True)
                    return 'wait'
                if not state.get('foreground'):
                    # Android may temporarily report no resumed Activity
                    # while a crashed process is being removed. No input.
                    self.missing=0
                    return 'wait'
                if not (state['running'] and (state.get('foreground') or '').startswith(PACKAGE + '/')):
                    raise RuntimeError('游戏恢复期间未能确认游戏进程状态，'
                                       f"foreground={state.get('foreground')!r}, "
                                       f"running={state.get('running')!r}, exit={state.get('exit')!r}；"
                                       '已停止并保留当前应用')
            self.missing = 0
        if not self.login_clicked and self.title_frames >= 3:
            with self.device.io_lock:
                state = self.device.game_state()
                self._stopped()
                fresh = self.capture()
                self._stopped()
                if (state['running'] and (state.get('foreground') or '').startswith(PACKAGE + '/')
                        and self._login(fresh)):
                    self.click(360, 914, '游戏恢复：点击 TOUCH TO START')
                    self.login_clicked = True
                    self.event('游戏恢复：已确认登录页，点击 TOUCH TO START')
                else:
                    self.title_frames = 0
        return 'wait'


