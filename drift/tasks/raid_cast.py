"""RAID 甩杆辅助：2026-10-22 更新后，"点击开始"进入战斗场地时 BOSS 不再自动出现，
需要再点一次屏幕、让角色甩杆才会开打。此前流程会在 raid_battle 里干等结果。

进入 raid_battle 后，若仍没有 RAID 战斗血条和沙漏，延迟后尝试抛竿；
最多两次，间隔 15 秒。识别到战斗即停止补点。
"""
from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np


class RaidBattleHud:
    """Identify the fixed RAID battle HUD independently of the dimmed footer."""
    ANCHORS = ((473,93,500,118),(292,164,326,174),(365,230,444,238))

    def __init__(self, root):
        source = cv2.imread(str(Path(root)/'assets/raid/battle.png'))
        if source is None:
            raise RuntimeError('RAID 战斗模板缺失')
        source = cv2.cvtColor(source,cv2.COLOR_BGR2RGB)
        # The hidden-players eye and the bevel on the two health bars are
        # fixed. Exclude the boss, health fill, level and countdown digits.
        self.templates = [source[y:Y,x:X].copy() for x,y,X,Y in self.ANCHORS]
        x,y,X,Y = 471,136,493,159
        self.timer = source[y:Y,x:X].copy()

    @staticmethod
    def _matches(source, template, minimum):
        score = cv2.minMaxLoc(cv2.matchTemplate(source,template,cv2.TM_CCOEFF_NORMED))[1]
        return bool(np.isfinite(score) and score >= minimum)

    def visible(self, frame):
        if frame.shape != (1280,720,3):
            return False
        for (x,y,X,Y), template in zip(self.ANCHORS, self.templates):
            if not self._matches(frame[y-2:Y+3,x-2:X+3],template,.95):
                return False
        return self._matches(frame[125:165,188:610],self.timer,.94)


class RaidCastAssist:
    # 进入战斗场地后等多久再点（给过场/加载留时间）
    FIRST_DELAY = 3.0
    # 第一次点完还没开打，隔多久补第二次
    RETRY_DELAY = 15.0
    # 最多补几次（防止无限点）
    MAX_TAPS = 2
    # 实战确认：合作突袭要抛竿时，必须点中间那个圆形按钮；
    # 点海面(360,900)没反应，只有 (360,1040) 会让角色真的甩杆、BOSS 随之出现。
    AT = (360, 1040)

    def __init__(self, *, battle, cast, event, clock=time.monotonic):
        self.battle, self.cast, self.event = battle, cast, event
        self.clock = clock
        self.armed = False
        self.due = None
        self.taps = 0
        self.hold = None

    def status(self):
        return dict(armed=self.armed, taps=self.taps, due=self.due, hold=self.hold)

    def arm(self):
        """在 raid_wait → raid_battle 的阶段切换时调用。"""
        self.armed, self.taps = True, 0
        self.due = self.clock() + self.FIRST_DELAY
        self.hold = 'RAID：已进入战斗场地，等待确认 BOSS 是否出现'

    def clear(self, reason=None):
        self.armed, self.due = False, None
        if reason:
            self.hold = reason

    def observe_battle(self, frame):
        if not self.battle(frame):
            return False
        if self.armed:
            self.event('RAID：已识别战斗血条与沙漏，停止补点')
            self.clear('RAID：战斗已在进行')
        return True

    def step(self, frame):
        """返回 True 表示这一帧刚发出一次点击（调用方应跳过其它输入）。"""
        if not self.armed:
            return False
        if self.observe_battle(frame):
            return False
        now = self.clock()
        if self.due is None or now < self.due:
            return False
        if self.taps >= self.MAX_TAPS:
            self.event('RAID：战斗场地始终没有出现 BOSS，停止补点，交给原有等待/超时流程')
            self.clear('RAID：补点用尽，未出现 BOSS')
            return False
        if not self.cast(self.AT[0], self.AT[1], 'RAID：战斗场地未出现 BOSS，点击一次让角色甩杆'):
            self.event('RAID：甩杆点击没有发出')
            self.clear('RAID：甩杆输入未发出')
            return False
        self.taps += 1
        self.due = now + self.RETRY_DELAY
        self.hold = f'RAID：已发出第 {self.taps} 次甩杆输入，等待 BOSS 出现'
        return True


