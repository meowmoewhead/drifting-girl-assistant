"""进展检测:抓"页面在两个状态之间来回打转"。

阶段截止时间只能发现"卡在一个步骤里太久";发现不了另一种常见故障 ——
每一步都很快完成,却在 A、B 两个页面之间反复往返(典型:任务弹窗反复出现、
收奖后又被弹回菜单)。ALAS 用 ``GameTooManyClickError`` 从点击侧兜住这类问题;
本项目按**状态迁移序列**判断,不需要改点击路径。

用法:每帧把当前 stage 交给 ``note()``,返回 True 表示判定为打转。
"""

from __future__ import annotations

from collections import deque


class ProgressGuard:
    def __init__(self, clock, *, window=120.0, limit=12):
        self.clock = clock
        self.window = float(window)
        self.limit = int(limit)
        self.last = None
        self.moves = deque()      # (时间, 起点, 终点)
        self.counts = {}          # (起点, 终点) -> 次数

    def reset(self):
        self.last = None
        self.moves.clear()
        self.counts.clear()

    def note(self, stage):
        """记录一次 stage。stage 变化时记一条迁移边,并返回是否判定打转。"""
        now = self.clock()
        if stage == self.last:
            return False
        previous, self.last = self.last, stage
        if previous is None:
            return False
        edge = (previous, stage)
        self.moves.append((now, edge))
        self.counts[edge] = self.counts.get(edge, 0) + 1
        while self.moves and now - self.moves[0][0] >= self.window:
            _, old = self.moves.popleft()
            remaining = self.counts.get(old, 0) - 1
            if remaining > 0:
                self.counts[old] = remaining
            else:
                self.counts.pop(old, None)
        return self.counts.get(edge, 0) > self.limit

    def status(self):
        return dict(window=self.window, limit=self.limit, stage=self.last,
                    edges={f'{a}->{b}': n for (a, b), n in self.counts.items()})
