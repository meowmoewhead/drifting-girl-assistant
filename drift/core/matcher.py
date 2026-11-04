"""规则扫描:每帧按 stage 过滤、连续帧投票、数字条件校验,得出本帧唯一动作。

从 ``Clicker.run()`` 的主循环里提取,行为逐字保持。时钟由调用方注入
(runtime 的 ``_time`` 会先看 app 侧是否被 patch),因此本模块不自行 import time。

返回 ``(candidate, grace_visible, previous_visible)``;candidate 为
``(rule, button, offset)`` 或 None。stage 超时且无宽限画面时由本函数抛错,
与提取前一致。
"""

from __future__ import annotations


def popup_reclick_decision(notice_visible, return_ready, reclick_count, limit=3):
    """Task notice still on screen after the close tap: tap the cross again.

    ``task_popup_close`` is a ``stage="*"`` interrupt, so once the runner moves
    to ``task_popup_return`` the rule scan no longer offers it again: the
    wildcard only matches while it is the current interruption.  A swallowed
    tap (input landed during the notice's own slide-in lock) therefore used to
    idle until the 15 s stage deadline stopped the whole run.  Report what the
    runtime should do instead of silently waiting:

    - ``wait``: the notice is gone, or the post-close delay has not elapsed;
    - ``reclick``: press the cross again (bounded by ``limit``);
    - ``give_up``: the cross was pressed ``limit`` times and the notice stays.
    """
    if not notice_visible or not return_ready:
        return 'wait'
    return 'reclick' if reclick_count < limit else 'give_up'


def observe_popup_equipment_return(frame, st, equipment, *, notice_gone, stop, clock):
    """A closed task notice may reveal the already active fishcake panel."""
    now=clock().monotonic()
    close=st.matchers.get('driftinggirl_task_popup_close')
    context=None
    if (not stop.is_set() and notice_gone and st.handling_interrupt
            and st.stage=='task_popup_return' and close is not None
            and st.stage_trigger==close[0]['name'] and st.stage_deadline is not None
            and st.stage_ready_at<=now<st.stage_deadline
            and equipment is not None and equipment.panel(frame)):
        context=(st.stage_trigger,st.stage_started_at,equipment.phase)
    if context is None:
        st.popup_panel_context,st.popup_panel_frames,st.popup_panel_since=None,0,None
        return False
    if context!=st.popup_panel_context:
        st.popup_panel_context,st.popup_panel_frames,st.popup_panel_since=context,1,now
    else:
        st.popup_panel_frames+=1
    now=clock().monotonic()
    return bool(st.popup_panel_frames>=3 and now-st.popup_panel_since>=3
                and not stop.is_set() and now<st.stage_deadline)


def observe_native_crown_return(frame, st, module, stone, *, preview, interruption,
                                stop, clock):
    """Confirm a closed crown page before combat clears ordinary rule votes.

    This is read-only and uses the unchanged full return template as well as
    the current native fishing/world/RANK context. Occlusion breaks the proof;
    an expired step cannot be revived, and no reward or route record is saved.
    """
    returning = st.matchers.get('driftinggirl_achievements_return')
    folded = st.matchers.get('driftinggirl_achievements_fold')
    ready = st.matchers.get('driftinggirl_achievements_ready')
    closed = st.matchers.get('driftinggirl_achievements_close')
    context = None
    now = clock().monotonic()
    triggers = {item[0]['name'] for item in (folded, closed) if item is not None}
    if (not preview and not stop.is_set() and not st.handling_interrupt
            and interruption is None and stone is not None
            and st.stage == 'achievements_return' and st.stage_trigger in triggers
            and st.stage_deadline is not None and st.stage_ready_at <= now < st.stage_deadline
            and returning is not None and folded is not None and ready is not None
            and not stone.defense.encounter(frame,39)):
        rule, button, offset = returning
        if (rule.get('enabled') and rule.get('stage') == 'achievements_return'
                and rule.get('action') == 'observe' and rule.get('next_stage') == 'main'
                and module.appear(button, offset=offset, similarity=rule['threshold'])
                and not module.appear(folded[1], offset=folded[2], similarity=folded[0]['threshold'])
                and not module.appear(ready[1], offset=ready[2], similarity=ready[0]['threshold'])):
            actual = stone.native_entry_recovery_context(frame)
            if actual is not None:
                # Battle pauses move stage_started_at; that does not change
                # the already closed page or this fresh visual context.
                context = (st.stage_trigger, actual)
    if context is None:
        st.crown_return_context, st.crown_return_frames, st.crown_return_since = None, 0, None
        return False
    if context != st.crown_return_context:
        st.crown_return_context, st.crown_return_frames = context, 1
        st.crown_return_since = clock().monotonic()
    else:
        st.crown_return_frames += 1
    now = clock().monotonic()
    return bool(st.crown_return_frames >= max(3, returning[0]['stable_frames'])
                and st.crown_return_since is not None and now-st.crown_return_since >= 3
                and not stop.is_set() and now < st.stage_deadline)


def scan_rules(frame, rules, module, matchers, *, load_rule, preview, interruption, stage,
               stage_expired, stage_deadline, stage_started_at, stage_trigger, stable,
               number_values, target_keys, previous, scheduled, daily_schedule, task_schedule,
               config, state, lock, number_condition, counter_condition, purchase_condition,
               daily_entries, clock):
    """扫描一帧,返回 (candidate, grace_visible, previous_visible)。"""
    candidate = None
    grace_visible = False
    previous_visible = False
    for rule, button, offset in rules:
        key = rule["id"]
        # Preview may inspect any supplied template without changing the UI.
        # Live clicks follow the sequence and suspend main-screen targets in a menu.
        if not preview and (key != interruption if interruption is not None
                            else rule.get("stage", "main") != stage):
            stable[key] = 0
            continue
        # A frame captured at the timeout boundary may already show a
        # valid return. Only configured observations may finish their
        # consecutive-frame check; no delayed clicks or renewed deadline.
        if stage_expired and (not rule.get('timeout_grace')
                or clock().monotonic() >= stage_deadline + rule['timeout_grace']):
            stable[key] = 0
            continue
        if not preview and clock().monotonic() - stage_started_at < float(rule.get('min_stage_seconds', 0)):
            stable[key] = 0
            continue
        if not preview and key in daily_entries and daily_schedule.is_complete(daily_entries[key]):
            stable[key] = 0
            continue
        if scheduled and key in daily_entries and not task_schedule.ready(
                daily_entries[key], daily_schedule.status()[daily_entries[key]]):
            stable[key] = 0
            continue
        matched = module.appear(button, offset=offset, similarity=rule["threshold"])
        bonus_visual = getattr(button, 'daily_bonus_visual', None)
        if matched and bonus_visual is not None:
            availability = bonus_visual.state(frame)
            matched = availability in ('ready', 'gray')
            if availability != number_values.get(key):
                stable[key] = 0
            number_values[key] = availability
        if matched and rule.get("absent_rule"):
            other = matchers.get(rule["absent_rule"])
            if other is None:
                other_rule = next((r for r in config["rules"] if r["id"] == rule["absent_rule"]), None)
                if other_rule is None:
                    raise ValueError("消失检测模板不存在")
                other = load_rule(other_rule)
                matchers[rule["absent_rule"]] = other
            guard_rule, guard_button, guard_offset = other
            module.appear(guard_button, offset=guard_offset, similarity=guard_rule["threshold"])
            # Presence checks depend on the picture, not an optional external tap location.
            matched = guard_button.last_similarity <= guard_rule["threshold"]
        if matched and any(rule.get(field) for field in ('number_condition', 'counter_condition', 'purchase_condition', 'quantity_condition', 'confirmation_condition')):
            check = (purchase_condition if any(rule.get(field) for field in ('purchase_condition', 'quantity_condition', 'confirmation_condition')) else
                     counter_condition if rule.get("counter_condition") else number_condition)
            matched, value = check(rule, button, frame)
            if value != number_values.get(key):
                stable[key] = 0
            number_values[key] = value
        elif not matched:
            number_values.pop(key, None)
        if matched and key == previous:
            previous_visible = True
        if matched and hasattr(button, "target_key"):
            if target_keys.get(key) != button.target_key:
                stable[key] = 0
            target_keys[key] = button.target_key
        elif not matched:
            target_keys.pop(key, None)
        stable[key] = stable[key] + 1 if matched else 0
        if stage_expired and matched:
            grace_visible = True
        if matched and candidate is None:
            with lock:
                state["last_match"] = {"name": rule["name"],
                    "score": round(button.last_similarity, 4),
                    "box": [int(v) for v in button.button]}
                if hasattr(button, "green_count"):
                    state["last_match"]["green_count"] = button.green_count
                    state["last_match"]["row"] = button.target_key + 1
            if stable[key] >= rule["stable_frames"]:
                timer = module.interval_timer.get(key)
                if scheduled and key in daily_entries or timer is None or timer.reached():
                    candidate = (rule, button, offset)
    return candidate, grace_visible, previous_visible

