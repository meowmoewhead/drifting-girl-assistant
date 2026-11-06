"""任务层:一个游戏功能一个任务类,外加 NextRun 调度。

- ``drift.tasks.features``     功能注册表(开关 ↔ 规则集合 ↔ 执行模式)
- ``drift.tasks.schedule``     实时 / 定时 / 每日三类任务的下次开启时间
- ``drift.tasks.daily``        每日周期的北京时间分界与完成记录
- ``drift.tasks.frame_runtime`` 截图与动作耗时统计
- 其余模块为各功能自己的流程实现(逐轮迁移中)

根目录同名模块(features / daily_schedule / task_schedule / frame_runtime)已改为
:mod:`drift.compat` 代理壳,`patch('features.FEATURES')` 这类补丁仍落到本包实现。
本 ``__init__`` 刻意不在顶层 import 任何子模块,避免导入顺序影响代理绑定时机。
"""


