"""``app`` —— 控制面板入口(兼容外观)。

实现已按 ALAS 分层拆到 ``drift`` 包,这里只做三件事:

1. **再导出**原有公开名字,使 ``from app import Clicker, AdbDevice, ...`` 与打包入口不变;
2. **共享补丁目标**:``time`` / ``subprocess`` / 各控制器类名在此与 ``drift.runtime``
   指向同一对象,``patch('app.time')``、``patch('app.ParallelWorld')`` 依然命中真实现;
3. `serve` / `diagnose` / `click-once` 三个子命令与 HTTP 层分别转发到
   :mod:`drift.web.cli` 与 :mod:`drift.web.server`。

分层对照:``drift/core`` 基础设施 → ``drift/device`` 截图与输入 →
``drift/ui`` 模板与规则 → ``drift/perception`` 识别 → ``drift/tasks`` 任务与调度 →
``drift/web`` 面板 → ``drift/runtime`` 装配。
"""

from __future__ import annotations

# 标准库与第三方:runtime 与外观共享同一模块对象,补丁两边同时生效。
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess
import time
from urllib.parse import urlsplit

# 最先执行:先把系统 MSVC 运行库载入,避免用到打进包里的旧版本(见 drift/core/winruntime.py)。
from drift.core.winruntime import prefer_system_msvc_runtime
prefer_system_msvc_runtime(quiet=True)

# 框架层再导出。
from drift.core.paths import ROOT, BUNDLE, UPSTREAM
from drift.core.util import hsv_mask, number, png_bytes, rectangle
# AdbDevice / make_handler 用模块级 __getattr__ 惰性取用,避免迁移期的循环导入。
try:  # HTTP 与命令行层正在迁往 drift.web;迁移期允许缺失
    from drift.web.server import make_handler
except ImportError:  # pragma: no cover
    make_handler = None
try:
    from drift.web.cli import main
except ImportError:  # pragma: no cover
    def main():  # type: ignore[misc]
        raise RuntimeError("命令行层尚未迁移完成")
from drift.ui.button import CroppedTemplateButton
from drift.runtime import Clicker

# 任务控制器再导出:测试会 patch('app.ParallelWorld') 等,必须同名同对象。
from features import FEATURE_BY_ID, RULE_FEATURE, RECOVERY_RULE_IDS, feature_states
from daily_schedule import DailySchedule, DAILY_ENTRIES, DAILY_RETURNS
from task_schedule import TaskSchedule
from meter_calibration import MeterCalibration
from drift.tasks.stone_farm import StoneFarm
from parallel_world import ParallelWorld
from equipment_upgrade import EquipmentUpgrade
from parallel_skills import ParallelSkills
from boss_training import training_options
from boss_evidence import BossInputEvidence
from native_capture import MuMuCapture, NativeUnavailable
from frame_runtime import FrameRuntime
from backpack_sale import BackpackSale
from game_recovery import GameRecovery, PACKAGE, game_state

# HTTP 层与命令行:实现位于 drift.web,这里按原路径转发,便于打补丁与打包。
# (导入语句已上移到框架层再导出段落,因为迁移期允许缺失。)

# HTTP 层与命令行:实现位于 drift.web,这里按原路径转发,便于打补丁与打包。
# (导入语句已上移到框架层再导出段落,因为迁移期允许缺失。)


def __getattr__(name):
    """迁移期兜底:设备层/HTTP 层就位前后都能按原路径取到名字。"""
    if name == "AdbDevice":
        from drift.device.adb import AdbDevice
        globals()["AdbDevice"] = AdbDevice
        return AdbDevice
    if name == "make_handler":
        from drift.web.server import make_handler
        globals()["make_handler"] = make_handler
        return make_handler
    raise AttributeError(name)


__all__ = [
    "Clicker", "AdbDevice", "CroppedTemplateButton", "make_handler", "main",
    "png_bytes", "number", "rectangle", "hsv_mask", "ROOT", "BUNDLE", "UPSTREAM",
    "ParallelWorld", "EquipmentUpgrade", "ParallelSkills", "BackpackSale", "MeterCalibration",
    "MuMuCapture", "NativeUnavailable", "DailySchedule", "TaskSchedule", "FrameRuntime",
    "BossInputEvidence", "GameRecovery", "PACKAGE", "game_state", "training_options",
    "BaseHTTPRequestHandler", "ThreadingHTTPServer", "urlsplit", "subprocess", "time",
]


if __name__ == "__main__":
    main()
