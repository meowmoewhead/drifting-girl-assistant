"""drift —— 《漂流少女》MuMu 自动化框架(ALAS 风格重构版)。

分层:

- ``drift.core``    路径、数值、日志、例外,无业务依赖
- ``drift.device``  截图与输入的三条通道(原生 / ADB 原始帧 / ADB PNG)
- ``drift.ui``      模板匹配与规则模型
- ``drift.perception`` 数字 OCR、奖励列表、购买校验
- ``drift.tasks``   一个功能一个任务类 + NextRun 调度
- ``drift.web``     控制面板 HTTP 层
- ``drift.runtime`` 装配各层、跑主循环

原有的根模块仍可导入:它们通过 :mod:`drift.compat` 转发到包内实现,
`patch('equipment_upgrade.time')` 这类补丁语义保持不变。
"""

from __future__ import annotations

__all__ = ["core", "compat"]


