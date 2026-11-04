"""任务声明表:功能 ↔ 规则集合 ↔ 执行模式 ↔ 控制器模块。

由 :func:`drift.tasks.base.build_specs` 从 ``drift/tasks/features.py`` 的功能注册表
派生,因此不会与 UI 契约漂移。控制器模块路径在导入期解析一次:已经迁到
``drift.tasks`` 下的用包内路径,尚未迁移的仍指向项目根的旧模块名。
"""

from __future__ import annotations

import importlib

from .base import CONTROLLERS, build_specs
from .daily import DAILY_ENTRIES
from .features import DAILY_FEATURES, FEATURES, REALTIME_CONDITIONS, TIMED_FEATURES


def _resolve_module(controller):
    """控制器当前可用的模块路径(drift.tasks 优先,未迁移则回退根模块)。"""
    migrated = f"drift.tasks.{controller}"
    try:
        importlib.import_module(migrated)
    except ImportError:
        return controller
    return migrated


#: 功能 id → :class:`drift.tasks.base.TaskSpec`
TASK_SPECS = build_specs(FEATURES, TIMED_FEATURES, DAILY_FEATURES, REALTIME_CONDITIONS,
                         resolve_module=_resolve_module)

__all__ = ["TASK_SPECS", "DAILY_ENTRIES", "CONTROLLERS"]
