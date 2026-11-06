"""User-facing tasks; each switch owns its complete recognition workflow.

本文件是"功能 ↔ 规则集合"的**单一事实来源**:``FEATURES`` 的每一条都把开关、
识别规则、执行模式绑在一起。根模块 ``features`` 现在只是转发到这里的兼容代理。
"""

FEATURES = (
    dict(id="return_parallel", name="返回平行世界", icon="parallel", category="日常收取",
         description="普通世界停船且航巡停止后，航行到传送观测所进入次元门。开关独立于次元石循环模式。",
         summary="普通世界停船时检查 · 航行至观测所 · 进入次元门",
         steps=["确认普通世界已停船且不在战斗", "打开地图，最多向两侧各平移六次寻找观测所",
                "选择观测所并关闭地图，等待航行到达", "进入已有次元门，未激活时先打开新门",
                "确认平行世界画面后完成；超时保留记录"],
         rules=(), enabled=False),
    dict(id="restaurant_upgrade", name="餐品强化", icon="growth", category="日常收取",
         description="定时前往当前世界的餐馆，在料理图鉴中长按交付四种食材，交满后确认强化永久效果。可只强化指定类别。不会制作或食用料理。",
         summary="从左到右逐个强化 · 四种食材 · 满级或缺料后继续下一个",
         steps=["等待当前事务结束，保存原地点和航巡设置", "在当前世界前往餐馆并打开料理图鉴",
                "依次检查九类已解锁餐品，可只强化指定类别，跳过未选类别和满级", "长按缺少的食材并核对实际交付数量，食材不足跳过",
                "四项交满后确认强化，核对升级后继续该餐品，直到满级或无法继续交付", "关闭窗口并恢复原地点或原航巡，记录下次执行时间"],
         rules=("driftinggirl_restaurant_upgrade",)),
    dict(id="guild_daily", name="公会任务提交", icon="guild", category="日常收取",
         description="每天北京时间18:00前往公会，点击紫色与蓝色两类任务的已完成指标进行提交，跳过已领取和未达成项目，结束后恢复原世界和航巡。",
         summary="每天18:00 · 两类任务 · 保留原活动",
         steps=["等待当前事务结束并保存原世界和航巡", "安全停航停钓，必要时返回普通世界", "直达公会，打开公会任务",
                "检查两类任务并提交可领取项目", "返回原世界并恢复原航巡", "记录当天完成，等待次日18:00"],
         rules=("driftinggirl_guild_daily",)),
    dict(id="task_popup", name="任务弹窗处理", icon="screen", category="日常收取",
         description="识别任务或每日刷新弹窗，关闭后重新确认当前页面，接续被打断的任务。",
         summary="任务及每日刷新弹窗 · 关闭后接续", steps=["优先识别当前弹窗", "核对关闭按钮后处理", "确认弹窗消失及原任务页面", "接续未完成任务"],
         rules=("driftinggirl_task_popup_close", "driftinggirl_task_popup_return")),
    dict(id="update_prompt", name="更新提示处理", icon="screen", category="日常收取",
         description="出现「发现新版本。现在更新吗？」时点取消继续游戏，避免弹窗盖住主界面导致其它任务全部识别失败。",
         summary="识别更新提示 · 点取消 · 回主界面", steps=["识别标题与按钮", "连续确认后点取消", "确认弹窗消失并回到主界面"],
         rules=("driftinggirl_update_prompt", "driftinggirl_update_prompt_return")),
    dict(id="relief", name="救济品", icon="parachute", category="日常收取",
         description="发现上半屏降落伞，自动收取水滴与星形救济品。",
         summary="两种降落伞 · 看到就领取", steps=["识别伞面与挂坠", "确认连续出现", "点击当前位置"],
         rules=("driftinggirl_relief_parachute", "driftinggirl_relief_parachute_star")),
    dict(id="mana", name="法力补充", icon="potion", category="日常收取",
         description="法力不足时使用一瓶药水，完成后自动回到游戏。",
         summary="低于设定值时补充 · 每轮一瓶", steps=["读取左下法力", "打开技能面板", "检查药水库存", "使用一瓶药水", "关闭面板并返回"],
         rules=("driftinggirl_mana_open", "driftinggirl_mana_potion", "driftinggirl_mana_confirm",
                "driftinggirl_mana_close", "driftinggirl_mana_return")),
    dict(id="achievements", name="皇冠奖励", icon="crown", category="日常收取",
         description="出现绿色 N 时进入奖励列表，逐项领取所有绿色按钮。",
         summary="绿色按钮全部领取 · 自动返回", steps=["发现绿色 N", "打开皇冠列表", "逐个领取绿色奖励", "关闭列表并收起菜单"],
         rules=("driftinggirl_achievements_open", "driftinggirl_achievements_ready", "driftinggirl_achievements_crown",
                "driftinggirl_achievements_claim", "driftinggirl_achievements_close", "driftinggirl_achievements_fold",
                "driftinggirl_achievements_return", "driftinggirl_achievements_resume",
                "driftinggirl_achievements_recover", "driftinggirl_achievements_battle_recover",
                "driftinggirl_achievements_parallel_battle_recover", "driftinggirl_achievements_claim_menu_recover",
                "driftinggirl_achievements_parallel_entry_recover")),
    dict(id="mailbox", name="邮箱奖励", icon="mail", category="日常收取",
         description="每天08:00刷新后，有新邮件提示时一键领取；本周期领空后不再领取。物品达到获取上限时保留邮件，15分钟后再检查。",
         summary="每天 08:00 刷新 · 新邮件提示触发 · 一键领取",
         steps=["北京时间 08:00 刷新后检查邮箱", "发现新邮件提示后打开邮箱", "全部领取并确认领空", "确认返回主界面后记录本周期完成",
                "本周期已领取则等到下一次 08:00 再检查"],
         rules=("driftinggirl_mailbox_open", "driftinggirl_mailbox_claim_all", "driftinggirl_mailbox_close", "driftinggirl_mailbox_limit_close",
                "driftinggirl_mailbox_return", "driftinggirl_mailbox_recover", "driftinggirl_mailbox_battle_recover",
                "driftinggirl_mailbox_parallel_battle_recover", "driftinggirl_mailbox_parallel_return")),
    dict(id="daily_reward", name="每日奖励", icon="gift", category="日常收取",
         description="从购物车进入 PREMIUM 页，分别检查免费奖励和附赠奖励：附赠卡明亮可领时领取，灰色直接跳过，确认返回后记录完成。",
         summary="每天 08:00 刷新 · 附赠可领就领、灰色跳过",
         steps=["北京时间 08:00 刷新后检查商店", "进入 PREMIUM 页", "领取免费奖励或识别已领取",
                "独立检查右侧附赠奖励，明亮可领才点击一次，灰色不点击",
                "每次点击前重新截图，领取后确认变灰；不点击购买券加号",
                "关闭商店并确认返回，保留已有免费完成记录，另记附赠已检查"],
         rules=("driftinggirl_daily_open", "driftinggirl_daily_premium", "driftinggirl_daily_claim",
                "driftinggirl_daily_bonus",
                "driftinggirl_daily_unavailable", "driftinggirl_daily_close", "driftinggirl_daily_return")),
    dict(id="pass_daily", name="PASS每日通行证", icon="gift", category="日常收取",
         description="每日08:00重置，10:00后检查；钓满120条鱼才领取，确认四档免费奖励绿勾并返回。",
         summary="每日10:00后 · 钓满120鱼 · 一次领满",
         steps=["北京时间08:00重置，等待两小时至10:00", "打开右上PASS并确认每日页",
                "读取本日钓鱼数，未满120则关闭，稍后再查", "只点击一次左侧免费奖励，不进入高级购买",
                "确认30/60/90/120四档绿勾并返回", "保留完成记录，等待明日10:00"],
         rules=("driftinggirl_pass_daily",)),
    dict(id="event_potion", name="活动药水", icon="coin", category="日常收取",
         description="当期活动材料超过 500 时，购买商店第一个药水并返回游戏。",
         summary="材料 > 500 · 每轮购买一瓶", steps=["连续读取活动材料", "超过 500 时打开活动商店", "选择第一个 500 材料药水", "确认购买", "关闭商店并返回主界面"],
         rules=("driftinggirl_event_potion_open", "driftinggirl_event_potion_select", "driftinggirl_event_potion_confirm",
                "driftinggirl_event_potion_close", "driftinggirl_event_potion_return")),
    dict(id="guild_shop", name="公会点数商品", icon="coin", category="日常收取",
         description="进入 SPECIAL 公会商店，将药水和天赋值加至可购买上限后兑换。",
         summary="每天 08:00 刷新 · 药水与天赋值", steps=["北京时间 08:00 刷新后进入 SPECIAL", "切换公会点数页", "药水逐次加一至可购买上限", "点击公会币并核对确认金额", "天赋值逐次加一并确认购买", "核对两种商品买满后记录本周期完成"],
         rules=("driftinggirl_guild_open", "driftinggirl_guild_special", "driftinggirl_guild_tab",
                "driftinggirl_guild_potion_select", "driftinggirl_guild_potion_skip", "driftinggirl_guild_potion_plus", "driftinggirl_guild_potion_buy", "driftinggirl_guild_potion_confirm",
                "driftinggirl_guild_material_select", "driftinggirl_guild_material_skip", "driftinggirl_guild_material_plus", "driftinggirl_guild_material_buy", "driftinggirl_guild_material_confirm",
                "driftinggirl_guild_close", "driftinggirl_guild_return")),
    dict(id="raid", name="RAID 合作突袭", icon="swords", category="战斗任务",
         description="进入合作突袭并准备，开始后核对战斗场地；BOSS未出现时点击圆盘抛竿，结算后接续原任务。",
         summary="准备与开始 · 场内抛竿 · 战斗结算", steps=["发现可参加的 RAID 并进入准备页", "等待开始文字出现且按钮变蓝后参战", "进入场地后核对真实战斗；未开打时按限次点击圆盘抛竿", "出现真实战斗后等待结算", "确认结算或已自动返回主界面，接续原任务"],
         rules=("driftinggirl_raid_open", "driftinggirl_raid_prepare", "driftinggirl_raid_start",
                "driftinggirl_raid_confirm", "driftinggirl_raid_return", "driftinggirl_raid_recover",
                "driftinggirl_raid_entry_recover")),
    dict(id="backpack_sell", name="满背包出售", icon="coin", category="日常收取",
         description="由脚本处理钓鱼材料满包：停航后前往港口，只出售第一标签的未锁定材料，再返回原地点。与游戏航巡中的自动贩卖分别控制。",
         summary="脚本港口出售 · 保留锁定材料 · 返回原地点", steps=["连续确认右下钓鱼材料背包已满", "确认自动航行停止，保存原地点并选择附近港口", "到达后切换背包第一标签", "出售未锁定的钓鱼材料", "返回原地点，按已保存的活动状态接续"],
         rules=("driftinggirl_backpack_sell",)),
    dict(id="equipment_upgrade", name="鱼饼武器强化", icon="swords", category="战斗任务",
         description="脚本定时打开鱼饼面板，检查并强化五种武器。归在次元石循环组内，仍使用独立开关和间隔。",
         summary="五种武器各检查一次 · 按保存间隔执行", steps=["到达下次执行时间后检查", "避开实际战斗，确认底部入口后打开鱼饼面板", "确认武器名称、鱼饼和绿色升级箭头", "五种武器各检查一次，可强化时长按一次", "检查一遍后关闭面板并安排下次执行", "入口未展开且连续确认主画面后等待下个间隔"],
         rules=("driftinggirl_equipment_upgrade",)),
    dict(id="parallel_skills", name="巡航设置", icon="growth", category="战斗任务",
         description="设置航巡目标、自动贩卖与强化、补票阈值及三种巡航技能；保存后自动同步。",
         summary="航巡选项 · 自动补票 · 技能独立开关", steps=["保存的修改在当前任务结束后自动应用", "重新核对停航和菜单，按勾选目标与自动项设置后恢复航巡", "售卖券低于阈值时长按加号，逐次确认数量增加；0表示不补票", "平行世界内分别核对钓鱼、移动、时间加速的自动续与强化", "经验加速仅在到期且余额足够时用200技能点购买1小时，保留有效购买记录"],
         rules=("driftinggirl_parallel_skills",)),
    dict(id="meter_calibration", name="标尺等级校准", icon="growth", category="战斗任务",
         description="平行世界中，标尺落后角色超过触发差距时校准。保留航巡，只暂停持续钓鱼，调整到当前可用范围内的角色等级附近。",
         summary="等级差触发 · 保留航巡 · 成功后按设置冷却", steps=["冷却结束后连续读取角色等级与标尺，差距严格超过触发值才开始", "保留航巡，仅在普通鱼准备阶段停持续钓鱼；任务BOSS、港口或未知画面等待", "连续确认未钓鱼后读取当前范围，地点变化时重新核对", "圆球AUTO开启时调到可用范围内的角色等级附近", "长按启动后，以连续新图的真实填充或战斗确认恢复", "保留原航巡选项，成功后按保存间隔冷却"],
         rules=("driftinggirl_meter_calibration",)),
    dict(id="stone_farm", name="次元石：降阶刷最后BOSS", icon="portal", category="战斗任务",
         description="游戏自行航巡；连续确认当前阶段；RANK39保留进度降一阶，通关后重开本轮原阶段。",
         summary="当前阶段RANK39 → 降一阶通关 → 重开原阶段", steps=["正常钓鱼、练级与BOSS挑战交给游戏航巡", "连续确认当前阶段与RANK39后，安全停航停钓，再返回观测所", "只降一阶并保留RANK39，重新进入本轮传送门", "按巡航设置恢复目标与自动项；未启用管理时沿用兼容路线", "降阶后的世界实际RANK40后，安全停航停钓并返回，确认旧门关闭", "打开新门并核对本轮原阶段，确认新轮后启动钓鱼及航巡"],
         rules=("driftinggirl_stone_farm",)),
    dict(id="parallel_world", name="任务导航与练级（旧模式）", icon="portal", category="战斗任务",
         description="沿地图任务升至 RANK 40，确认通关后关闭并重开次元门，循环获取次元石。",
         summary="前期 BOSS 航行 · 后期三资源练级 · RANK 40 循环", steps=["可设置前期仅 BOSS 自动航行的截止 RANK，0 为关闭", "达到截止 RANK 后连续确认停航，交回任务与练级流程", "普通 BOSS 实际钓鱼失败达到设定次数后开始练级", "最后一个 BOSS 默认练到同级再挑战", "练级只勾选珍珠、琥珀、蓝色宝石", "钓鱼期间继续已开启的武器强化", "达到目标等级后先停止自动航行，再返回 BOSS", "RANK 40 连续确认后返回并重开次元门"],
         rules=("driftinggirl_parallel_world",)),
    dict(id="growth_ad", name="成长广告", icon="growth", category="广告奖励",
         description="识别右下 50% 成长图标，按游戏显示的每日次数领取。",
         summary="50% 成长加速 · 每日上限 10 次", steps=["识别图标与每日计数", "点击继续", "确认加速生效并返回"],
         rules=("driftinggirl_bottom_right_arrow", "driftinggirl_ad_confirm", "driftinggirl_ad_reward_active",
                "driftinggirl_ad_entry_recover")),
    dict(id="potion_ad", name="药水广告", icon="flask", category="广告奖励",
         description="识别右下药水图标，领取奖励后等待返回主画面。",
         summary="药水奖励 · 每日上限 40 次", steps=["识别图标与每日计数", "点击继续", "确认入口消失并返回"],
         rules=("driftinggirl_potion_ad_open", "driftinggirl_potion_ad_confirm", "driftinggirl_potion_ad_return",
                "driftinggirl_potion_ad_entry_recover")),
)

TASK_GROUPS = (
    dict(id='cruise', name='巡航与补给', icon='sail',
         description='设置游戏航巡目标、自动贩卖/强化、补票及技能；标尺、法力和脚本满包出售分别控制。',
         features=('parallel_skills', 'meter_calibration', 'mana', 'backpack_sell')),
    dict(id='stone', name='次元石循环', icon='portal',
         description='选择降阶刷最后BOSS或任务导航练级；自动返回平行世界、鱼饼武器强化分别开启。',
         features=('stone_farm', 'parallel_world', 'return_parallel', 'equipment_upgrade')),
    dict(id='restaurant', name='餐品强化', icon='growth',
         description='按保存的间隔强化餐品，保留餐品选择、食材交付进度和下次执行时间。',
         features=('restaurant_upgrade',)),
    dict(id='rewards', name='奖励领取', icon='gift',
         description='每日奖励、PASS、邮箱分别记录领取；皇冠和救济品按画面实时触发。',
         features=('daily_reward', 'pass_daily', 'mailbox', 'achievements', 'relief')),
    dict(id='guild', name='公会事务', icon='coin',
         description='公会任务18点后提交，商店08点刷新；两项保留独立开关和完成记录。',
         features=('guild_daily', 'guild_shop')),
    dict(id='activities', name='活动与广告', icon='flask',
         description='活动兑换与两种广告分别开启，按材料、可领取入口及各自限额执行。',
         features=('event_potion', 'growth_ad', 'potion_ad')),
    dict(id='raid', name='RAID', icon='swords',
         description='可参加时进入合作突袭，结算返回后接续原任务。', features=('raid',)),
    dict(id='recovery', name='后台恢复', icon='screen',
         description='设置任务/每日刷新弹窗和更新提示处理。游戏启动与重连由脚本内置恢复流程处理。',
         features=('task_popup', 'update_prompt')),
)

# Only the presentation changes. Rule IDs, execution modes and journals stay stable.
for group in TASK_GROUPS:
    for feature in FEATURES:
        if feature['id'] in group['features']:
            feature.update(group=group['id'], category=group['name'])

FEATURE_BY_ID = {feature["id"]: feature for feature in FEATURES}
RULE_FEATURE = {rule: feature["id"] for feature in FEATURES for rule in feature["rules"]}
RECOVERY_RULE_IDS = FEATURE_BY_ID['task_popup']['rules']

# Timing applies to starting a workflow, never to its follow-up or safety steps.
TIMED_FEATURES = {'equipment_upgrade', 'parallel_skills', 'restaurant_upgrade'}
# 每天只触发一次的功能:北京时间 08:00 刷新,本周期完成后等待下一次刷新。
# 邮箱由用户指定为每日一次(原先按新邮件提示实时触发,会一天领很多轮)。
DAILY_FEATURES = {'daily_reward', 'guild_shop', 'mailbox', 'pass_daily', 'guild_daily'}
REALTIME_CONDITIONS = {
    'task_popup': '任务弹窗出现时处理',
    'update_prompt': '出现「发现新版本」提示时点取消',
    'relief': '降落伞出现时收取',
    'mana': '法力低于设定阈值时补充',
    'achievements': '皇冠出现新奖励提示时领取',
    'event_potion': '活动材料超过设定值时兑换',
    'raid': '可参加的 RAID 入口出现时处理',
    'backpack_sell': '背包满时处理',
    'meter_calibration': '冷却结束且角色与标尺的差距超过触发值时校准',
    'stone_farm': '当前阶段RANK39降一阶；RANK40后重开原阶段',
    'parallel_world': '持续检查任务、练级与次元门状态',
    'growth_ad': '可领取的成长广告入口出现时处理',
    'potion_ad': '可领取的药水广告入口出现时处理',
    'return_parallel': '普通世界停船且航巡停止时，航行到观测所进入次元门',
}
# 不靠识别规则驱动的功能:它们由运行时直接判定条件(例如 Rank 与所在世界),
# 没有对应的 scan_rules 条目。列在这里是为了让"一功能一规则集合"的守卫明确
# 知道这类功能是漏登记(需要修)还是设计如此(允许 rules 为空)。
RULELESS_FEATURES = {'return_parallel'}
# 每日功能的触发条件说明(它们不走 REALTIME_CONDITIONS)。
DAILY_CONDITIONS = {
    'guild_daily': '每天北京时间18:00提交紫色和蓝色两类公会任务',
    'daily_reward': '每天08:00刷新后检查免费及附赠奖励',
    'guild_shop': '每天 08:00 刷新后购买药水与天赋值',
    'mailbox': '每天08:00刷新后检查新邮件，本周期领空后等待下次刷新',
    'pass_daily': '每天08:00重置，10:00后且钓满120条鱼领取',
}


def execution_mode(feature):
    return 'daily' if feature in DAILY_FEATURES else 'timed' if feature in TIMED_FEATURES else 'realtime'


def _duration(seconds):
    value = float(seconds)
    for unit, label in ((86400, '天'), (3600, '小时'), (60, '分钟')):
        if value >= unit and value % unit == 0:
            return f'{value / unit:g}{label}'
    return f'{value:g}秒'


def feature_states(config, schedules=None, executions=None):
    rules = {rule["id"]: rule for rule in config["rules"]}
    result = []
    for feature in FEATURES:
        members = [rules[key] for key in feature["rules"] if key in rules]
        count = sum(bool(rule["enabled"]) for rule in members)
        complete = len(members) == len(feature["rules"])
        entry = rules.get(feature["rules"][0], {}) if feature["rules"] else {}
        result.append({**feature, "available": complete, "enabled": complete and count == len(members),
                       "partial": 0 < count < len(feature["rules"]), "enabled_count": count,
                       "interval": entry.get("interval"),
                       "mana_threshold": entry.get("number_condition", {}).get("value"),
                       "daily_limit": entry.get("counter_condition", {}).get("expected_total"),
                       "schedule": (schedules or {}).get(feature['id']),
                       "execution": (executions or {}).get(feature['id'], {
                           'mode': execution_mode(feature['id']),
                           'condition': REALTIME_CONDITIONS.get(feature['id'])
                                        or DAILY_CONDITIONS.get(feature['id'])})})
        if feature['id'] == 'return_parallel':
            enabled = config.get('return_parallel_enabled') is True
            result[-1].update(enabled=enabled, enabled_count=int(enabled))
            result[-1]['execution'] = {**result[-1]['execution']}
            if not enabled:
                result[-1]['execution']['status'] = 'disabled'
        if feature['id'] == 'parallel_world':
            # 函数内延迟导入:features 与 boss_training 互相引用,只能在调用时解析。
            # 迁移过渡期 drift.tasks.boss_training 可能尚未落地,回退到根代理。
            try:
                from drift.tasks.boss_training import training_options
            except ImportError:
                from drift.tasks.boss_training import training_options
            result[-1]['training'] = training_options(entry.get('training'))
        if feature['id'] == 'parallel_skills':
            result[-1]['growth_points'] = entry.get('growth_points', True)
            from drift.tasks.cruise_options import from_rule
            result[-1]['cruise_settings'] = from_rule(entry)
        if feature['id'] == 'restaurant_upgrade' and 'targets' in entry:
            result[-1]['targets'] = list(entry['targets'])
        if feature['id'] == 'restaurant_upgrade':
            result[-1]['summary'] = f"当前间隔{_duration(entry.get('interval', 259200))} · 按所选类别从左到右，逐个强化"
        if feature['id'] == 'equipment_upgrade':
            from drift.tasks.equipment_upgrade import EquipmentUpgrade
            seconds = EquipmentUpgrade.HOLD_SECONDS
            result[-1]['hold_seconds'] = seconds
            result[-1]['summary'] = f"当前间隔{_duration(entry.get('interval', 900))} · 每种长按{seconds:g}秒一次"
            result[-1]['steps'] = list(feature['steps'])
            result[-1]['steps'][3] = f'五种武器各检查一次，可强化时长按{seconds:g}秒一次'
        if feature['id'] == 'meter_calibration':
            from drift.tasks.meter_calibration import MeterCalibration
            from drift.tasks.fishing_meter import FishingMeter
            gap = MeterCalibration.GAP
            interval = entry.get('interval', 60)
            result[-1]['calibration'] = dict(trigger_gap=gap, tolerance=FishingMeter.LEVEL_GAP,
                                            failure_backoff=min(interval, MeterCalibration.FAILURE_BACKOFF))
            result[-1]['summary'] = f'差距 > {gap}级 · 当前成功冷却{_duration(interval)}'
            result[-1]['execution'] = {**result[-1]['execution'],
                'condition': f'冷却结束且角色等级比标尺高出超过{gap}级时校准（等于{gap}不触发）'}
    native = next(f for f in result if f['id'] == 'stone_farm')['enabled']
    for feature in result:
        feature['effective_enabled'] = feature['enabled'] and not (feature['id'] == 'parallel_world' and native)
        if feature['id'] == 'parallel_world' and feature['enabled'] and native:
            feature['inactive_reason'] = '当前使用降阶刷最后BOSS，旧模式设置保留'
    return result


# 只读预留表:feature id → 计划中的任务类路径(将来"一功能一类"用)。
# 目前全部留空,且**没有任何现有逻辑读取本表**;填值前请勿在别处引用。
TASK_CLASSES: dict[str, str] = {feature["id"]: "" for feature in FEATURES}


