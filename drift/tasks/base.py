"""任务基类与注册表。

ALAS 的风格是"一个功能一个类,自己管 `run()` / `NextRun`"。本项目的功能分两类,
但它们共用同一套描述信息与调度语义,所以用 :class:`TaskSpec` 统一声明:

- **规则驱动任务**(10 个):流程完全由 `config.json` 的 stage 图与识别规则描述,
  例如救济品、法力补充、皇冠、邮箱、每日奖励、活动药水、公会商店、RAID、两种广告。
  它们在主循环里共用同一套"扫描 → 稳定帧投票 → 点击"引擎,不需要各自的类。
- **控制器任务**(5 个):有跨帧状态机,各自一个类,已分别在自己的模块里实现
  (鱼饼强化、平行世界技能、满背包出售、标尺校准、平行世界)。

`TaskSpec` 把两类任务在"开关 ↔ 规则集合 ↔ 执行模式 ↔ 依赖"上的差异集中成数据,
`:data:`TASK_SPECS` 是这份数据的单一来源。`drift/tasks/features.py` 里的
`FEATURES` 仍是 UI 契约(面板与测试读它),两者的一致性由测试
`tests/test_task_registry.py` 校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 执行模式:realtime 按画面条件触发;timed 由 NextRun 定时;daily 按北京时间刷新。
MODE_REALTIME = "realtime"
MODE_TIMED = "timed"
MODE_DAILY = "daily"


@dataclass(frozen=True)
class TaskSpec:
    """一个功能的声明式描述。

    ``kind`` 为 ``rule`` 时表示流程由规则图描述并在主循环里共用扫描引擎;
    为 ``controller`` 时表示由 ``drift/tasks`` 下的同名类自行实现。
    """

    id: str
    name: str
    kind: str
    mode: str
    rules: tuple
    module: str = ""
    controller: str = ""
    main_guard: str = ""
    depends_on: tuple = field(default_factory=tuple)
    note: str = ""

    @property
    def is_controller(self):
        return self.kind == "controller"


def build_specs(features, timed, daily, realtime_conditions, resolve_module=None):
    """由功能注册表派生任务声明,保证 key 集合完全一致。

    ``resolve_module`` 把一个控制器名解析成"当前可用"的模块路径:迁移完成的控制器
    在 ``drift.tasks`` 下,尚未迁移的仍在项目根目录,由调用方决定。
    """
    resolve_module = resolve_module or (lambda name: f"drift.tasks.{name}")
    specs = []
    for feature in features:
        identifier = feature["id"]
        mode = (MODE_DAILY if identifier in daily
                else MODE_TIMED if identifier in timed
                else MODE_REALTIME)
        controller = CONTROLLERS.get(identifier, "")
        specs.append(TaskSpec(
            id=identifier,
            name=feature["name"],
            # 控制器持有界面并暂停主界面目标的功能(哪怕流程本身由规则图描述)
            kind="controller" if identifier in MAIN_GUARDS else "rule",
            mode=mode,
            rules=tuple(feature["rules"]),
            module=resolve_module(controller) if controller else "drift.core.matcher",
            controller=controller,
            main_guard=MAIN_GUARDS.get(identifier, ""),
            depends_on=DEPENDENCIES.get(identifier, ()),
            note=realtime_conditions.get(identifier, "")))
    return {spec.id: spec for spec in specs}


# 有独立状态机、在 drift/tasks 下各自成模块的功能(其余走共用规则引擎)。
CONTROLLERS = {
    "pass_daily": "pass_daily",
    "equipment_upgrade": "equipment_upgrade",
    "parallel_skills": "parallel_skills",
    "backpack_sell": "backpack_sale",
    "meter_calibration": "meter_calibration",
    "stone_farm": "stone_farm",
    "parallel_world": "parallel_world",
}

# 主循环里这些控制器一旦"持有界面",就暂停主界面目标;名字对应 state 里的字段。
MAIN_GUARDS = {
    "pass_daily": "pass_daily",
    "equipment_upgrade": "equipment",
    "parallel_skills": "skills",
    "backpack_sell": "backpack",
    "meter_calibration": "meter",
    "stone_farm": "stone",
    "parallel_world": "parallel",
}

# 功能之间的显式依赖(原先只由 config.json 的 stage 图隐含表达)。
DEPENDENCIES = {
    "pass_daily": ("task_popup",),
    "parallel_world": ("equipment_upgrade", "parallel_skills", "backpack_sell", "meter_calibration"),
    "backpack_sell": ("parallel_world",),
    "meter_calibration": ("parallel_world",),
    "raid": ("task_popup",),
    "daily_reward": ("task_popup",),
    "guild_shop": ("task_popup",),
    "event_potion": ("task_popup",),
    "growth_ad": ("task_popup",),
    "potion_ad": ("task_popup",),
    "mailbox": ("task_popup",),
    "achievements": ("task_popup",),
}


