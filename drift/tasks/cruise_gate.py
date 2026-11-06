"""航巡看门狗与开局配置的放行条件。

这些条件原本是 ``Clicker.run()`` 里的闭包,能直接抓 ``st``/``calibrator``/``stone``
等局部变量,但完全没法单测。2026-10-26 12:38 的航巡停摆就是这里写错了一行:
标尺窗口以 ``workflow_limit`` 归档后 ``awaiting_recovery`` 一直为真,而
``meter_active`` 把它也算成"标尺在跑",于是航巡看门狗和开局配置永远被拦住;
反过来标尺解除归档又要求原生航巡在跑,两边互锁,船在港口停了 40 分钟。

抽成纯函数后,参数全部显式传入,这类回归可以直接钉住。
"""
BLOCKER_KEYS = ('stage', 'handling_interrupt', 'setup_active', 'daily_cleanup',
                'meter_active', 'stone_active', 'other_task')


def _task_active(task):
    active = getattr(task, 'active', False)
    return bool(active() if callable(active) else active)


def cruise_blockers(*, include_setup=True, stage='main', handling_interrupt=False,
                    setup_busy=False, daily_cleanup=False, calibrator=None,
                    stone=None, tasks=()):
    """返回当前阻挡航巡的项;全为 False 才允许看门狗/开局配置动手。

    ``meter_active`` 只认 ``calibrator.active``,也就是"存在还没结束的标尺窗口"。
    已经归档(retired)的标尺只是只读观察、等原生航巡回来,不能再否掉航巡:
    它的释放证明本身就依赖航巡在跑,互相等待就是死锁。
    """
    if calibrator is None:
        meter_active = False
    else:
        active = getattr(calibrator, 'active', False)
        meter_active = bool(active() if callable(active) else active)
    return dict(stage=stage != 'main',
                handling_interrupt=bool(handling_interrupt),
                setup_active=bool(include_setup and setup_busy),
                daily_cleanup=bool(daily_cleanup),
                meter_active=meter_active,
                stone_active=_task_active(stone),
                other_task=any(_task_active(task) for task in tasks))


