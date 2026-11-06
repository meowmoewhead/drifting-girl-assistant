"""主循环的会话态。

原先这些变量是 ``Clicker.run()` 的局部变量 —— 四十来个名字散在 1300 行里,
既无法单独测试,也让"把某个阶段抽成函数"变成"传二十几个参数"。
收进这个数据类之后:

- 阶段函数只需要拿 ``LoopState`` + 当前帧 + 控制器句柄;
- 每个字段的初值、类型与生命周期集中在一处;
- 控制器句柄(parallel/equipment/…)**仍留在 run() 里做局部变量**,
  因为测试用 ``patch('app.ParallelWorld')`` 这类方式替换它们。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class LoopState:
    """一次 run() 期间跨帧保留的全部状态。"""

    # --- 规则扫描与动作执行 ---
    stable: dict = field(default_factory=dict)          # 规则 id → 连续命中帧数
    previous: object = None                             # 上一次命中的规则 id
    previous_progress: object = None
    consecutive: int = 0                                # 同一目标连续点击次数
    absent_frames: int = 0                              # 目标消失帧数(满 2 帧重置 previous)
    number_values: dict = field(default_factory=dict)   # 数字条件读值缓存
    target_keys: dict = field(default_factory=dict)     # 绿色奖励列表锁定行
    matchers: dict = field(default_factory=dict)        # 规则 id → (rule, button, offset)

    # --- 分步流程状态机 ---
    stage: str = "main"
    stage_started_at: float = 0.0
    stage_deadline: object = None
    stage_trigger: object = None
    stage_ready_at: float = 0.0
    # 当前步骤中途超时的处理:true = 跳过本次并回主界面(领不上就跳过),false = 停机等人工
    skip_on_timeout: bool = False
    # 页面图导航:当前 stage 已发出的"退回主画面"点击次数,防止在超时分支里无限点
    nav_stage: object = None
    nav_clicks: int = 0
    # 上一次写进日志的 stage:只用于记录阶段迁移,不参与任何判定
    logged_stage: object = None
    daily_cleanup_deadline: object = None
    daily_orphan_key: object = None
    daily_orphan_frames: int = 0

    # --- 中断(任务弹窗)占用 ---
    handling_interrupt: bool = False
    interrupt_started_at: object = None
    interrupt_times: deque = field(default_factory=deque)
    popup_map_frames: int = 0
    popup_panel_context: object = None
    popup_panel_frames: int = 0
    popup_panel_since: object = None
    # 关闭红叉后弹窗仍在时已重按的次数(每次成功关闭都归零)
    popup_reclick: int = 0
    # 弹窗连续可见的帧数:重按前要重新凑够 stable_frames,与首次点击同一套证据
    popup_reclick_frames: int = 0

    # --- 被资源战斗打断的入口恢复 ---
    resource_crown_frames: int = 0
    resource_crown_context: object = None
    resource_crown_since: object = None
    resource_entry_frames: int = 0
    resource_entry_context: object = None
    resource_entry_since: object = None
    crown_return_frames: int = 0
    crown_return_context: object = None
    crown_return_since: object = None
    entry_recoveries: dict = field(default_factory=dict)  # 恢复/重试时间戳 deque

    # --- 每日与定时任务 ---
    recovery_only: bool = False
    daily_period: object = None
    daily_attempt: object = None
    daily_verified: bool = False
    scheduled: bool = False
    scheduled_daily: object = None

    # --- 限额与暂停 ---
    run_deadline: object = None
    defense_pause_at: object = None
    completion_pause_at: object = None
    recovery_pause_at: object = None
    recovery_pause_stage: object = None
    recovery_pause_used: float = 0.0

    def account_recovery_wait(self, result, started, now):
        """Exclude recovery ownership once, capped at 120 seconds per stage.

        Deadlines already expired before the interruption are never revived.
        The formal run deadline and persisted game/skill budgets are untouched.
        """
        identity = (self.stage, self.stage_trigger)
        if self.recovery_pause_stage != identity:
            self.recovery_pause_at = None
            self.recovery_pause_used = 0.0
            self.recovery_pause_stage = identity
        since = self.recovery_pause_at
        if since is None and result == 'wait':
            if self.stage_deadline is not None and started >= self.stage_deadline:
                return 0.0
            since = started
        if since is None:
            return 0.0
        duration = min(max(0.0, now-since), max(0.0, 120-self.recovery_pause_used))
        self.stage_started_at += duration
        if self.stage_ready_at:
            self.stage_ready_at += duration
        if self.stage_deadline is not None:
            self.stage_deadline += duration
        if self.daily_cleanup_deadline is not None:
            self.daily_cleanup_deadline += duration
        self.recovery_pause_used += duration
        self.recovery_pause_at = now if result == 'wait' else None
        return duration

    def reset_after_return(self):
        """回到主界面后清空分步流程状态(原先散落的三处赋值)。"""
        self.stage = "main"
        self.stage_deadline = None
        self.stage_trigger = None
        self.stage_ready_at = 0.0
        self.skip_on_timeout = False
        self.daily_cleanup_deadline = None
        self.daily_orphan_key = None
        self.daily_orphan_frames = 0
        self.recovery_pause_at = self.recovery_pause_stage = None
        self.recovery_pause_used = 0.0


