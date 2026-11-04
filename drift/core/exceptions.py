"""框架异常层次(参照 ALAS ``module/exception.py`` 的分类思路)。

约定:
- ``Stopped``:用户停止或达到限额,属于**正常收尾**,不写错误截图;
- ``RetryableStage``:本步骤可以作废/稍后重试(预算用尽、可跳过任务超时、
  页面反复打转)。命名沿用；带任务身份的实例可被调度层接住继续，但必须
  核实安全交还并保留未完成记录。默认仍是整轮干净停止，不写错误截图;
- ``GameStuck``:同一个动作反复命中却没有进展,或游戏侧卡住;保留现场并建议
  重启游戏,调用方可以据此走恢复流程;
- 其余异常都是真故障,调用方应保留现场。

``DriftError`` 继承 ``RuntimeError``:历史代码与测试大量用 ``except RuntimeError``
捕获,保持兼容,避免引入新类型时改变既有语义。
"""

from __future__ import annotations


class DriftError(RuntimeError):
    """本框架所有可预期错误的基类。"""


class Stopped(DriftError):
    """已请求停止,或达到点击/时长限额。"""


class RetryableStage(DriftError):
    """本步骤可作废:预算用尽、可跳过任务超时、页面反复打转。"""

    def __init__(self, *args, task=None, retry_seconds=None):
        super().__init__(*args)
        self.task = task
        self.retry_seconds = retry_seconds


class GameStuck(DriftError):
    """反复命中同一动作却无进展,或游戏侧卡住。"""


class DeviceError(DriftError):
    """模拟器连接、截图、输入失败。"""


class RuleError(DriftError):
    """识别规则配置不合法。"""


class TaskError(DriftError):
    """任务流程走到无法继续的状态。"""
