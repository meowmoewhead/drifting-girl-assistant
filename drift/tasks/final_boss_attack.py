"""最后两关的独立攻击：守护者只补伤害，海神保留反击和补伤害。

独立于全局开关，识别真实 BOSS 后立即生效。

RANK 39 的任务点也会出现普通鱼。只有本次新截图通过 ``BossDefense.encounter``
的海神名字、倒计时、世界、RANK 与任务骷髅守卫，才额外开启两种攻击。
守护者对应 RANK 38→39；经独立的守护者遭遇守卫，只额外开启点击补伤害。
解析结果只供这一帧使用，不改写用户配置，也不带入下一帧或连续观察窗口。
"""

from __future__ import annotations

#: 打 40 BOSS 的那一关(平行世界最后一个 RANK;通关后 RANK 才变 40)
FINAL_BOSS_RANK = 39
GUARDIAN_BOSS_RANK = 38
#: 面板上显示的名字,与功能说明保持一致
FEATURE_ID = 'final_boss_attack'
FEATURE_NAME = '40级BOSS攻击'


def is_final_boss(rank):
    """是否处于最后一关 RANK；单凭此值不能授权攻击。"""
    return type(rank) is int and rank == FINAL_BOSS_RANK


def forced_attacks(training, rank, *, encounter=False, guardian=False):
    """本次是否由 BOSS 专用规则额外开启了攻击开关。

    返回 ``(boss_interrupt, battle_tap)``:为真表示该攻击是被本功能打开的(配置里
    原本是关的);配置本来就开着时返回假 —— 那样它不算本功能的功劳,状态显示也更诚实。
    """
    if type(rank) is int and rank == GUARDIAN_BOSS_RANK and guardian is True:
        return (False, training.options.get('battle_tap') is not True)
    if not is_final_boss(rank) or encounter is not True:
        return (False, False)
    options = training.options
    return (options.get('boss_interrupt') is not True, options.get('battle_tap') is not True)


def options_for(training, rank, *, encounter=False, guardian=False):
    """返回本次判定该用的选项视图。

    - 新截图确认海神遭遇且处于平行世界 RANK 39:``boss_interrupt``(组织攻击)与
      ``battle_tap``(点击攻击)**强制为真** —— 这一关这两个攻击就是功能本身,
      不需要用户再去别的地方打开,配置里怎么设都不影响;
    - 新截图确认 RANK 38 的守护者:只额外开启 ``battle_tap``，不启用触手反击；
    - 其它画面或未经确认:原样返回 ``training.options``。

    普通鱼继续使用全局配置，不继承上一张 BOSS 画面的额外开关。
    """
    options = training.options
    if type(rank) is int and rank == GUARDIAN_BOSS_RANK and guardian is True:
        if options.get('battle_tap') is True:
            return options
        return dict(options, battle_tap=True)
    if not is_final_boss(rank) or encounter is not True:
        return options
    if options.get('boss_interrupt') is True and options.get('battle_tap') is True:
        return options
    view = dict(options)
    view['boss_interrupt'] = True
    view['battle_tap'] = True
    return view


