"""开局把原生航巡按用户要求配好：珍珠 / 琥珀 / 蓝色宝石 / BOSS + 自动售卖 + 自动强化。

为什么单独写一个而不复用次元石控制器：次元石那套只在"有降阶交接"时跑，
而用户要的是**脚本一启动、从主界面就要把航巡开起来**。这里同样是"看清楚了才点"：
- 只在主画面、且航巡确实没在跑时动手；
- 每点一次都要在下一帧重新确认菜单与设置指纹，确认不到就退避；
- 目标值、自动售卖、自动强化逐项比对，只补差的那一项；
- 成功后不再重复，失败最多重试一次。
"""
from __future__ import annotations

import time

# 与次元石控制器一致的目标顺序与期望值
DESIRED = {'pearl': True, 'material': False, 'amber': True, 'gem': True, 'boss': True}
# 与 training_sailing.TARGETS 一致的目标横坐标
TARGET_X = {'pearl': 272, 'material': 345, 'amber': 417, 'gem': 490, 'boss': 562}
TARGET_Y = 458
OPEN_AT = (233, 45)          # 航巡入口（设置菜单）
START_AT = (360, 873)        # 开始航巡
CLOSE_AT = (600, 363)        # 设置菜单右上角红 X（放弃时必须关掉，否则主界面规则认不出）
SALE_AT = (576, 647)         # 自动售卖勾选框
UPGRADE_AT = (576, 761)      # 自动强化勾选框


class CruiseSetup:
    OPEN_WAIT = 20.0         # 等菜单出现
    CONFIRM_FRAMES = 3       # 同一设置连续确认帧数
    START_WAIT = 60.0        # 等航巡变运行
    MAX_TAPS = 6             # 目标/勾选最多补几次
    RETRY_BUDGET = 1         # 整轮失败后最多再试一次

    def __init__(self, *, sailing, click, event, clock=time.monotonic, desired=DESIRED, observe=None):
        self.sailing, self.click, self.event = sailing, click, event
        self.clock, self.desired = clock, dict(desired)
        self.state = 'idle'
        self.deadline = None
        self.taps = 0
        self.failures = 0
        self.done = False
        self.hold = '航巡开局配置：等待主画面'
        self._key = None
        self._seen = 0
        self.observe = observe
        self.input_context = None
        self.scene = None
        self.last_frame = None
        self._leak_closed = False

    # ---- 状态 -----------------------------------------------------
    def status(self):
        return dict(state=self.state, done=self.done, taps=self.taps,
                    failures=self.failures, hold=self.hold)

    @property
    def busy(self):
        return not self.done and self.state in ('open', 'menu', 'starting')

    # ---- 读数 -----------------------------------------------------
    def _selected(self, frame, name):
        return self.sailing.selected(frame, name)

    def _sale(self, frame):
        return self.sailing.sale_selected(frame)

    def _upgrade(self, frame):
        selected = getattr(self.sailing, 'upgrade_selected', None)
        return selected(frame) if callable(selected) else None

    def _mismatch(self, frame):
        """返回第一个需要处理的项：('target', name) / ('sale', None) / ('upgrade', None)。"""
        for name in self.desired:
            actual = self._selected(frame, name)
            if actual is not None and actual is not self.desired[name]:
                return 'target', name
        if self._sale(frame) is False:
            return 'sale', None
        if self._upgrade(frame) is False:
            return 'upgrade', None
        return None, None

    def _all_known(self, frame):
        return (all(self._selected(frame, name) is not None for name in self.desired)
                and self._sale(frame) is not None and self._upgrade(frame) is not None)

    def _confirmed(self, key):
        if key != self._key:
            self._key, self._seen = key, 1
            return False
        self._seen += 1
        return self._seen >= self.CONFIRM_FRAMES

    # ---- 主流程 ---------------------------------------------------
    def step(self, frame):
        """返回 True 表示这一帧由本流程占用（已发出输入）。"""
        if self.done:
            # 收尾守卫：万一放弃时的关闭动作没落地（进程重启、输入被拒），
            # 也不能让设置菜单留在屏幕上把主循环锁死。只补一次，避免连点。
            if not self._leak_closed and self.sailing.menu(frame):
                if self.click(*CLOSE_AT, '航巡开局配置：关闭遗留的设置菜单'):
                    self._leak_closed = True
                    self.event('航巡开局配置：已关闭上一轮遗留的设置菜单')
            return False
        now = self.clock()
        if self.busy and self.deadline is not None and now >= self.deadline:
            self._give_up('当前步骤超过原有时限，未确认航巡启动', frame)
            return False
        if frame is self.last_frame:return False
        self.last_frame=frame
        if self.observe is not None:
            self.input_context = self.observe(frame)
            if self.input_context is None:
                self._key, self._seen = None, 0
                return False
            scene = self.input_context['scene']
            if self.busy and self.scene != scene:
                self._give_up('当前世界或 RANK 已变化', frame)
                return False
            self.scene = scene

        if self.state == 'idle':
            if self.sailing.running(frame) is True:
                # 已经在跑：不动设置，交给已有的航巡守卫
                self.done = True
                self.hold = '航巡已在运行，开局配置跳过'
                return False
            if self.sailing.menu(frame):
                self.state, self.deadline = 'menu', now + self.OPEN_WAIT
                self.hold = '航巡设置已打开，核对目标与自动项'
                return False
            if self.sailing.running(frame) is not False:
                self.hold = '航巡状态未知，等待明确停航主画面'
                self._key, self._seen = None, 0
                return False
            if not self._confirmed(('stopped', self.scene)):
                return False
            if not self.click(*OPEN_AT, '航巡开局配置：打开航巡设置'):
                self._give_up('打开航巡设置的输入没有发出', frame)
                return False
            self.state, self.deadline = 'open', now + self.OPEN_WAIT
            self.hold = '已打开航巡设置，等菜单出现'
            self.event('航巡开局配置：主画面确认航巡未运行，打开设置')
            return True

        if self.state == 'open':
            if not self.sailing.menu(frame):
                if self.deadline is not None and now > self.deadline:
                    self._give_up('航巡设置菜单没有出现', frame)
                return False
            self.state = 'menu'
            self.deadline = now + self.OPEN_WAIT
            self._key = None

        if self.state == 'menu':
            if not self.sailing.menu(frame):
                # 菜单被关掉（战斗/弹窗打断）：放弃本轮，避免盲点
                self._give_up('航巡设置菜单被关闭', frame)
                return False
            if not self._all_known(frame):
                if self.deadline is not None and now > self.deadline:
                    self._give_up('设置项读不清（目标或自动售卖/强化）', frame)
                return False
            kind, name = self._mismatch(frame)
            if kind is not None:
                if self.taps >= self.MAX_TAPS:
                    self._give_up('需要改的项太多，停止追加输入', frame)
                    return False
                if not self._confirmed(('mismatch', kind, name)):
                    return False
                if kind == 'target':
                    x = TARGET_X.get(name)
                    if x is None:
                        self._give_up('目标坐标缺失', frame)
                        return False
                    label = f'航巡开局配置：勾选 {name}'
                    if not self.click(x, TARGET_Y, label):
                        self._give_up('勾选目标的输入没有发出', frame)
                        return False
                elif kind == 'sale':
                    if not self.click(*SALE_AT, '航巡开局配置：勾选自动售卖'):
                        self._give_up('勾选自动售卖的输入没有发出', frame)
                        return False
                else:
                    if not self.click(*UPGRADE_AT, '航巡开局配置：勾选自动强化'):
                        self._give_up('勾选自动强化的输入没有发出', frame)
                        return False
                self.taps += 1
                self._key, self._seen = None, 0
                self.deadline = now + self.OPEN_WAIT
                self.hold = f'已补 {self.taps} 项，重新核对设置'
                return True
            # 全部符合：连续确认后点开始
            key = (tuple(self._selected(frame, n) for n in self.desired),
                   self._sale(frame), self._upgrade(frame))
            if not self._confirmed(key):
                return False
            if not self.click(*START_AT, '航巡开局配置：开始航巡'):
                self._give_up('开始航巡的输入没有发出', frame)
                return False
            self.state, self.deadline = 'starting', now + self.START_WAIT
            self._key, self._seen = None, 0
            self.hold = '设置已符合要求，已点击开始航巡'
            self.event('航巡开局配置：目标与自动项已确认，开始航巡')
            return True

        if self.state == 'starting':
            running = self.sailing.running(frame)
            if running is True and self._confirmed(('started',)):
                self.done = True
                self.hold = '航巡开局配置完成：航巡已运行'
                self.event('航巡开局配置完成：珍珠/琥珀/蓝色宝石/BOSS + 自动售卖 + 自动强化已生效')
                return False
            if running is not True:
                self._key, self._seen = None, 0
                self.hold = '设置菜单已关闭，等待连续确认航巡运行' if not self.sailing.menu(frame) else '等待航巡运行'
            if self.deadline is not None and now > self.deadline:
                self._give_up('点开始后航巡没有变成运行', frame)
            return False

        self._give_up('未知状态', frame)
        return False

    def _give_up(self, reason, frame=None):
        self.failures += 1
        self.hold = '航巡开局配置放弃：' + reason
        # 打开菜单后放弃却不关菜单，会让 stage=main 的规则集永久认不出画面
        # （2026-10-29 12:12 实测卡死 20 分钟）。放弃前必须把菜单收回去。
        if frame is not None and self.sailing.menu(frame):
            if self.click(*CLOSE_AT, '航巡开局配置：放弃并关闭设置菜单'):
                self.hold += '；已关闭设置菜单'
        self.event(self.hold)
        if self.failures > self.RETRY_BUDGET:
            self.done = True
        self.state, self.deadline = 'idle', None
        self._key, self._seen = None, 0
