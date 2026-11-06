"""Shared inputs extracted from the aligned runtime; original guards retained."""
from contextlib import nullcontext

def _scope(ctx,name,owner=None):
    ui=getattr(ctx['runtime'],'ui',None)
    if ui is None:return nullcontext()
    current=ui.current_actor()
    if current in ui.MANAGED or current in ui.PRIORITY:return nullcontext()
    mapping={'parallel_click':'parallel','parallel_swipe':'parallel','portal_click':'stone_farm',
        'completion_click':'completion','defense_click':'battle','sailing_click':'parallel',
        'fishing_action':'meter_calibration' if ctx.get('calibrator') and ctx['calibrator'].active else 'parallel',
        'equipment_hold':'equipment_upgrade','backpack_click':'backpack_sale','backpack_swipe':'backpack_sale',
        'skill_click':'parallel_skills','pass_click':'pass_daily','restaurant_click':'restaurant_upgrade',
        'return_parallel_action':'return_parallel','guild_daily_click':'guild_daily'}
    if owner is not None:
        for key,task in ui.tasks.items():
            if task is owner:return ui.scope(key)
    return ui.scope(mapping[name])

def build_handlers(_bindings,_globals):
    def parallel_click(x, y, label):
        _ctx=_bindings()
        self=_ctx['self']
        SimpleNamespace=_globals['SimpleNamespace']
        with _scope(_ctx,'parallel_click'):
            x, y = int(x), int(y)
            self.device.click(SimpleNamespace(button=(max(0, x-5), max(0, y-5),
                                                      min(720, x+5), min(1280, y+5))))
            with self.lock:
                self.state['clicks'] += 1
                self.state['matches'] += 1
                self.state['last_match'] = {'name': label, 'score': 1.,
                                            'box': [x-5, y-5, x+5, y+5]}
            self.event(f'点击：{label}，坐标 ({x}, {y})')

    def parallel_swipe(x1, y1, x2, y2, label):
        _ctx=_bindings()
        self=_ctx['self']
        with _scope(_ctx,'parallel_swipe'):
            with self.device.io_lock:
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                self.device.command('shell', 'input', 'swipe', str(x1), str(y1),
                                    str(x2), str(y2), '480')
            with self.lock:
                self.state['clicks'] += 1
            self.event(label)

    def portal_click(action, owner=None):
        _ctx=_bindings()
        parallel=_ctx['parallel']
        self=_ctx['self']
        max_clicks=_ctx['max_clicks']
        st=_ctx['st']
        stone=_ctx['stone']
        runtime=_ctx['runtime']
        parallel_click=_ctx['parallel_click']
        _lookup=_globals['_lookup']
        _time=_globals['_time']
        uuid=_globals['uuid']
        with _scope(_ctx,'portal_click',owner=owner):
            owner = parallel if owner is None else owner
            with self.device.io_lock:
                if (self.stop_event.is_set()
                        or max_clicks is not None and self.state['clicks'] >= max_clicks
                        or st.run_deadline is not None and _time().monotonic() >= st.run_deadline):
                    raise InterruptedError('已停止')
                returning = action['kind'].startswith('return-')
                abandoning_boss = owner is stone and action['kind'] in ('lower-boss-abandon','lower-boss-abandon-menu')
                fast_contact = returning or abandoning_boss
                cancelling = owner is stone and action['kind'] == 'handoff-cancel'
                def stopped():
                    return (self.stop_event.is_set()
                        or max_clicks is not None and self.state['clicks'] >= max_clicks
                        or st.run_deadline is not None and _time().monotonic() >= st.run_deadline)
                started = _time().monotonic()
                capture = runtime.capture
                fresh = capture()
                if cancelling:
                    scan_deadline = started+6
                    reserved = False
                    for index in range(40):
                        if stopped() or _time().monotonic() >= scan_deadline or not owner.same_cancel_context(action,fresh):
                            return False
                        if owner.validate_portal(action,fresh) and _time().monotonic()-started <= .12:
                            if reserved:
                                break
                            # Persist the conservative attempt before contact,
                            # then capture again. Disk writes may outlive the
                            # 120 ms input frame; never spend that stale frame
                            # or reserve a second attempt for the same scan.
                            if not owner.reserve_input(action):
                                return False
                            reserved = True
                        if index == 39:
                            return False
                        started = _time().monotonic()
                        fresh = capture()
                frame_id = self.remember(fresh)
                # The cancelling loop just checked this exact fresh frame.
                # Repeating its image work eats the fleeting cast window.
                if not cancelling and not owner.validate_portal(action,fresh):
                    return False
                if (self.stop_event.is_set()
                        or max_clicks is not None and self.state['clicks'] >= max_clicks
                        or st.run_deadline is not None and _time().monotonic() >= st.run_deadline):
                    raise InterruptedError('已停止')
                if (fast_contact or cancelling) and _time().monotonic()-started > .12:
                    return False
                if cancelling and _time().monotonic() >= scan_deadline:
                    return False
                if not cancelling and hasattr(owner, 'reserve_input') and not owner.reserve_input(action):
                    return False
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                if (fast_contact or cancelling) and _time().monotonic()-started > .12:
                    return False
                if cancelling and _time().monotonic() >= scan_deadline:
                    return False
                metadata = dict(input_id=uuid.uuid4().hex,frame_id=frame_id,
                    kind=action['kind'],x=action['x'],y=action['y'],
                    label=action['label'],rank=owner.rank,
                    completed_rank_40=owner.completed_rank_40,
                    controller='stone_farm' if owner is stone else 'parallel_world',
                    input_begin_time=_time().time(),capture_age_ms=round((_time().monotonic()-started)*1000,2))
                if cancelling:
                    metadata['continuous_cast_stop'] = action.get('continuous_stop') is True
                if fast_contact or cancelling:
                    if action['kind']=='return-confirm' or owner is stone and action['kind']=='lower-boss-abandon-menu':
                        # Unity's modal button needs a complete stationary
                        # contact. Keep the battle SDK's 40 ms release intact;
                        # this is the existing 80 ms ADB menu gesture instead.
                        if stopped():
                            raise InterruptedError('已停止')
                        self.device.command('shell','input','swipe',str(action['x']),str(action['y']),
                                            str(action['x']),str(action['y']),'80')
                        contact = dict(input_transport='adb-stationary',hold_ms=80)
                    else:
                        contact = self.device.touch_fast(action['x'],action['y'],started,stopped)
                    if contact is None:
                        return False
                    metadata.update(contact)
                    with self.lock:
                        self.state['clicks'] += 1
                        self.state['matches'] += 1
                        self.state['last_match'] = {'name':action['label'],'score':1.,
                            'box':[action['x']-5,action['y']-5,action['x']+5,action['y']+5]}
                    self.event(f"点击：{action['label']}，坐标 ({action['x']}, {action['y']})")
                elif owner is stone and action['kind'] == 'first-pan':
                    X,Y,milliseconds=action['swipe']
                    self.device.command('shell','input','swipe',str(action['x']),str(action['y']),
                                        str(X),str(Y),str(milliseconds))
                    metadata.update(input_transport='adb-swipe',swipe=action['swipe'])
                    with self.lock:
                        self.state['clicks'] += 1
                        self.state['matches'] += 1
                    self.event('滑动：'+action['label'])
                else:
                    parallel_click(action['x'],action['y'],action['label'])
                    metadata['input_transport'] = 'adb'
                if hasattr(owner, 'input_issued'):
                    owner.input_issued(action, metadata)
                after = None
                if not self.stop_event.is_set():
                    after = capture()
                    metadata['after_frame_id'] = self.remember(after)
                    metadata['after_time'] = _time().time()
                if self.portal_evidence is None:
                    self.portal_evidence = _lookup("BossInputEvidence")(self.root,self.event,
                        folder='portal-input',label='传送门')
                self.portal_evidence.record(metadata,fresh.copy(),after.copy() if after is not None else None)
            return True

    def completion_click():
        _ctx=_bindings()
        self=_ctx['self']
        max_clicks=_ctx['max_clicks']
        st=_ctx['st']
        runtime=_ctx['runtime']
        parallel=_ctx['parallel']
        parallel_click=_ctx['parallel_click']
        equipment=_ctx['equipment']
        _time=_globals['_time']
        with _scope(_ctx,'completion_click'):
            with self.device.io_lock:
                if (self.stop_event.is_set()
                        or max_clicks is not None and self.state['clicks'] >= max_clicks
                        or st.run_deadline is not None and _time().monotonic() >= st.run_deadline):
                    raise InterruptedError('已停止')
                fresh = runtime.capture()
                self.remember(fresh)
                if not parallel.completion_visible(fresh):
                    return False
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                parallel_click(360,840,'平行世界：确认本阶段探索结束')
                if equipment is not None:
                    equipment.yielded_for_completion()
            return True

    def defense_click(expected, owner=None, rank=None):
        _ctx=_bindings()
        parallel=_ctx['parallel']
        self=_ctx['self']
        stone=_ctx['stone']
        st=_ctx['st']
        runtime=_ctx['runtime']
        max_clicks=_ctx['max_clicks']
        _lookup=_globals['_lookup']
        _time=_globals['_time']
        uuid=_globals['uuid']
        with _scope(_ctx,'defense_click',owner=owner):
            owner = parallel if owner is None else owner
            attack_rank = owner.rank if rank is None else rank
            with self.device.io_lock:
                captured = {}
                pending = []
                if self.boss_evidence is None:
                    self.boss_evidence = _lookup("BossInputEvidence")(self.root,self.event)
                def capture_defense_frame():
                    started = _time().monotonic()
                    fresh = runtime.capture()
                    identifier = self.remember(fresh)
                    if pending:
                        metadata,before = pending.pop()
                        metadata.update(after_frame_id=identifier,after_time=_time().time())
                        self.boss_evidence.record(metadata,before,fresh.copy())
                    captured.update(frame=fresh,frame_id=identifier,started=started)
                    return fresh
                def defense_stopped():
                    return (self.stop_event.is_set()
                            or owner is stone and not owner.enabled()
                            or owner is stone and owner.active and not owner.input_allowed()
                            or max_clicks is not None and self.state['clicks'] >= max_clicks
                            or st.run_deadline is not None and _time().monotonic() >= st.run_deadline)
                def hit_defense_target(target):
                    if defense_stopped():
                        raise InterruptedError('已停止')
                    # This applies to both native and fresh ADB transports.
                    # A slow capture must never authorize a late contact.
                    if _time().monotonic()-captured['started'] > .12:
                        return False
                    x,y = target['x'],target['y']
                    metadata = dict(input_id=uuid.uuid4().hex,frame_id=captured['frame_id'],
                        rank=attack_rank,controller_rank=attack_rank,
                        controller='stone_farm' if owner is stone else 'parallel_world',
                        kind=target.get('kind','defense'),
                        target=dict(target),input_begin_time=_time().time(),
                        transport=getattr(self.device,'screenshot_transport','test'),
                        capture_to_command_ms=round((_time().monotonic()-captured['started'])*1000,1),
                        gesture='stationary_touch',hold_ms=80,result='issued_only')
                    # A DOWN/UP within one Unity frame can be lost by gameplay
                    # input polling. Hold for several frames, with no movement
                    # or unguarded repeat at the old coordinates.
                    if hasattr(self.device,'touch_fast'):
                        contact=self.device.touch_fast(x,y,captured['started'],defense_stopped)
                        if contact is None:
                            return False
                        metadata.update(contact)
                    else:
                        self.device.command('shell','input','swipe',str(x),str(y),str(x),str(y),'80')
                        metadata.update(input_transport='adb')
                    metadata['input_end_time'] = _time().time()
                    pending.append((metadata,captured['frame'].copy()))
                    with self.lock:
                        self.state['clicks'] += 1
                        self.state['matches'] += 1
                        self.state['last_match'] = {'name':('战斗：点击屏幕补充伤害' if target.get('kind') == 'damage'
                                                             else '海神：连续点击发黄光的触手武器'),
                            'score':1.,'box':[x-5,y-5,x+5,y+5],
                            'frame_id':captured['frame_id'],'input_id':metadata['input_id'],
                            'hold_ms':metadata['hold_ms'],'transport':metadata['transport'],
                            'input_transport':metadata['input_transport'],'result':'issued_only'}
                try:
                    # First-cast result votes need fresh outer frames before
                    # their original deadline; a four-second burst must not
                    # consume the remaining verification window. Other fights
                    # retain the existing burst and per-input guards.
                    limits = {'seconds':.5} if owner is stone and owner.phase=='first_verify' else {}
                    taps = owner.defense.burst(capture_defense_frame,hit_defense_target,
                                              attack_rank,defense_stopped,**limits)
                finally:
                    if pending:
                        metadata,before = pending.pop()
                        metadata.update(after_frame_id=None,after_reason='burst_ended')
                        self.boss_evidence.record(metadata,before)
            counts = owner.defense.last_burst
            if counts.get('defense_taps'):
                self.event(f"海神反击：连续核对新截图后点击充能武器 {counts['defense_taps']} 次")
            if counts.get('damage_taps'):
                self.event(f"战斗补伤害：新截图核对后点击屏幕 {counts['damage_taps']} 次；充能触手优先")
            return taps

    def sailing_click(action, x, y, label):
        _ctx=_bindings()
        self=_ctx['self']
        runtime=_ctx['runtime']
        parallel=_ctx['parallel']
        parallel_click=_ctx['parallel_click']
        _lookup=_globals['_lookup']
        _time=_globals['_time']
        uuid=_globals['uuid']
        with _scope(_ctx,'sailing_click'):
            with self.device.io_lock:
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                fresh = runtime.capture()
                self.remember(fresh)
                controller = parallel.sailing
                if not controller.validate(action,fresh):
                    self.event('自动航行：点击前画面或目标变化，等待重新识别')
                    return False
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                controller.before_click(action)
                metadata = (dict(input_id=uuid.uuid4().hex,frame_id=self.state.get('frame_id'),
                    action=action,profile=controller.profile,rank=parallel.rank,
                    x=x,y=y,time=_time().time(),result='issued_only')
                    if controller.profile == 'boss' else None)
                parallel_click(x,y,label)
                if metadata is not None:
                    after = None
                    if not self.stop_event.is_set():
                        after = runtime.capture()
                        metadata['after_frame_id'] = self.remember(after)
                    if self.sailing_evidence is None:
                        self.sailing_evidence = _lookup("BossInputEvidence")(self.root,self.event,
                            folder='sailing-input',label='自动航行')
                    self.sailing_evidence.record(metadata,fresh.copy(),after.copy() if after is not None else None)
            return True

    def fishing_action(action, data, owner=None, guard=None, reserve=None, allowed=None,
                       route_expected=None):
        _ctx=_bindings()
        parallel=_ctx['parallel']
        self=_ctx['self']
        runtime=_ctx['runtime']
        calibrator=_ctx['calibrator']
        st=_ctx['st']
        max_clicks=_ctx['max_clicks']
        MeterWorkflowLimit=_globals['MeterWorkflowLimit']
        _lookup=_globals['_lookup']
        _time=_globals['_time']
        uuid=_globals['uuid']
        with _scope(_ctx,'fishing_action',owner=owner):
            owner = parallel if owner is None else owner
            observed = []
            def fishing_stopped():
                return (self.stop_event.is_set()
                         or guard is not None and not (allowed or calibrator.input_allowed)()
                        or max_clicks is not None and self.state['clicks'] >= max_clicks
                        or st.run_deadline is not None and _time().monotonic() >= st.run_deadline)
            def raise_fishing_stop():
                if (guard is not None and allowed is None and not self.stop_event.is_set()
                        and not (max_clicks is not None and self.state['clicks'] >= max_clicks)
                        and not (st.run_deadline is not None and _time().monotonic() >= st.run_deadline)
                        and not calibrator.input_allowed()):
                    raise MeterWorkflowLimit('标尺输入前校准窗口已到期或功能已关闭')
                raise InterruptedError('已停止')
            with self.device.io_lock:
                if fishing_stopped():
                    raise_fishing_stop()
                meter = owner.fishing
                # Only the independent calibration opts into preserving native
                # sailing. Navigation/training/stone handoffs still require OFF.
                keep_route = bool(guard is not None and callable(route_expected)
                                  and route_expected() is True)
                cancelling = action in ('cancel','scan_cancel')
                captured = {}
                def capture_cancel_frame():
                    started = _time().monotonic()
                    image = runtime.capture()
                    identifier = self.remember(image)
                    captured.update(frame=image,frame_id=identifier,started=started)
                    return image
                if cancelling:
                    if self.fishing_evidence is None:
                        self.fishing_evidence = _lookup("BossInputEvidence")(self.root,self.event,
                            folder='fishing-input',label='钓鱼计量')
                    def usable_cancel_frame(image):
                        return _time().monotonic()-captured['started'] <= .12
                    def stopped_route(image):
                        if (not owner._parallel(image) or not keep_route
                                and owner.training.options['auto_sailing']
                                and owner.training.state.get('sailing_enabled')
                                or guard is not None and not guard(image)):
                            return None
                        return owner.sailing.running(image)
                    scan_options = {'expected_route': True} if keep_route else {}
                    fresh = meter.scan_cancel_frame(capture_cancel_frame,stopped_route,
                                                    data,fishing_stopped,usable_cancel_frame,**scan_options)
                    if fresh is None:
                        return False
                else:
                    capture_started = _time().monotonic()
                    fresh = runtime.capture()
                    self.remember(fresh)
                    route = owner.sailing.running(fresh) if owner.sailing else False
                    if ((route is not True if keep_route else route is True or owner.sailing
                            and owner.training.options['auto_sailing']
                            and (route is not False or owner.training.state.get('sailing_enabled')))
                            or guard is not None and not guard(fresh)):
                        self.event('钓鱼计量：航巡状态或校准上下文变化，等待重新确认')
                        return False
                    if not meter.validate(action, fresh, data):
                        self.event('钓鱼计量：操作前画面变化，等待重新识别')
                        return False
                    actual_start = getattr(meter,'validated_start',None) if action == 'start' else None
                    if actual_start is not None and actual_start[0] is fresh:
                        # Input receipt and log must describe this fresh frame,
                        # rather than the slightly different planned selection.
                        data = actual_start[1]
                if fishing_stopped():
                    raise_fishing_stop()
                if action == 'wake' and _time().monotonic()-capture_started > .12:
                    return False
                if reserve is not None and not reserve(action):
                    return False
                if fishing_stopped():
                    raise_fishing_stop()
                calibration_metadata = None
                if guard is not None and not cancelling:
                    calibration_metadata = dict(input_id=uuid.uuid4().hex,
                        frame_id=self.state.get('frame_id'),kind='meter_'+action,
                        rank=owner.training.state.get('rank'),input_begin_time=_time().time(),
                        target=dict(value=data.get('value'),goal=data.get('goal')),
                        result='issued_only')
                    calibration_before = fresh.copy()
                if cancelling:
                    # The X is the left-hand label; target the fish-shaped
                    # cancellation button itself, rather than that label.
                    x,y = 690,1090
                    if _time().monotonic()-captured['started'] > .12:
                        return False
                    before = fresh.copy()
                    metadata = dict(input_id=uuid.uuid4().hex,frame_id=captured['frame_id'],
                        kind='meter_cancel',rank=owner.training.state.get('rank'),
                        target=dict(x=x,y=y),input_begin_time=_time().time(),
                        transport=getattr(self.device,'screenshot_transport','test'),
                        capture_to_command_ms=round((_time().monotonic()-captured['started'])*1000,1),
                        result='issued_only')
                    if hasattr(self.device,'touch_fast'):
                        contact = self.device.touch_fast(x,y,captured['started'],fishing_stopped)
                        if contact is None:
                            return False
                        metadata.update(contact)
                    else:
                        self.device.command('shell','input','tap',str(x),str(y))
                        metadata.update(input_transport='adb',hold_ms=0)
                    metadata['input_end_time'] = _time().time()
                    label = '钓鱼计量：尝试取消已确认的普通鱼准备状态，等待休眠证据'
                    if not self.stop_event.is_set():
                        observed.append(runtime.capture())
                        metadata.update(after_frame_id=self.remember(observed[-1]),after_time=_time().time())
                    self.fishing_evidence.record(metadata,before,observed[-1].copy() if observed else None)
                elif action == 'wake':
                    x, y = 360, 1010
                    if hasattr(self.device, 'touch_fast'):
                        contact = self.device.touch_fast(x, y, capture_started, fishing_stopped)
                        if contact is None:
                            return False
                        if calibration_metadata is not None:
                            calibration_metadata.update(contact)
                    else:
                        self.device.command('shell','input','swipe',str(x),str(y),str(x),str(y),'80')
                    if calibration_metadata is not None:
                        calibration_metadata['gesture'] = dict(x=x,y=y,duration_ms=40 if hasattr(self.device,'touch_fast') else 80)
                    label = '钓鱼计量：单次短点唤醒Zzz，等待新画面，不当作钓鱼已启动'
                elif action == 'resume_cast':
                    x, y = 360, 1010
                    if calibration_metadata is not None:
                        calibration_metadata['gesture'] = dict(x=x,y=y,duration_ms=80)
                    self.device.command('shell', 'input', 'swipe', str(x), str(y),
                                        str(x), str(y), '80')
                    label = '标尺校准：仅短点灰色鱼钩恢复钓鱼，等待真实填充或战斗，不计校准成功'
                elif action in ('drag','reveal'):
                    x, y, to_y = data['x'], data['y'], data['to_y']
                    if calibration_metadata is not None:
                        calibration_metadata['gesture'] = dict(x=x,y=y,to_y=to_y,duration_ms=meter.DRAG_MS)
                    self.device.command('shell', 'input', 'swipe', str(x), str(y),
                                        str(x), str(to_y), str(meter.DRAG_MS))
                    label = ('钓鱼计量：移开遮挡范围标签的数字框，重新读取上下限' if action == 'reveal'
                             else f"钓鱼计量：拖动数字框 {data['value']} → 目标 {data['goal']}，角色 Lv{data['player']}")
                else:
                    x, y = 360, 1010
                    hold_feedback = getattr(self.device,'hold_feedback',None)
                    if callable(hold_feedback):
                        # The gesture owns its during-hold capture. No numeric
                        # cache may survive across those separately acquired pixels.
                        runtime.end_frame()
                        observed.append(hold_feedback(x,y,meter.HOLD_MS))
                    else:
                        self.device.command('shell', 'input', 'swipe', str(x), str(y),
                                            str(x), str(y), str(meter.HOLD_MS))
                    label = f"钓鱼计量：已调到 {data['value']}，长按中央圆盘 {meter.HOLD_MS/1000:g} 秒开始钓鱼"
                    # Fast fishing can finish before the next OCR-heavy loop.
                    # Capture immediately while the same device lock is held.
                    if not self.stop_event.is_set():
                        observed.append(runtime.capture())
                        self.remember(observed[-1])
                if action in ('resume_cast','start') and not fishing_stopped():
                    # 取消钓鱼的 X 只在抛竿后约 1.5~2 秒内出现(2026-10-23 实测):
                    # 抛完立刻补一帧,把这个短窗口写进证据,但不拖住主循环。
                    probe = runtime.capture()
                    self.remember(probe)
                    if meter.cancel_button(probe):
                        self.event('钓鱼计量：抛竿后捕捉到取消 X（约 2 秒窗口）')
                if calibration_metadata is not None:
                    calibration_metadata['input_end_time'] = _time().time()
                    if not observed and not fishing_stopped():
                        observed.append(runtime.capture())
                        self.remember(observed[-1])
                    if self.fishing_evidence is None:
                        self.fishing_evidence = _lookup("BossInputEvidence")(self.root,self.event,
                            folder='fishing-input',label='钓鱼计量')
                    after_id = self.state.get('frame_id') if observed else None
                    calibration_metadata['after_frame_id'] = after_id
                    self.fishing_evidence.record(calibration_metadata,calibration_before,
                                                observed[-1].copy() if observed else None)
                    if action == 'start' and len(observed) >= 2:
                        feedback = dict(input_id=uuid.uuid4().hex,
                            parent_input_id=calibration_metadata['input_id'],kind='meter_hold_feedback',
                            frame_id=uuid.uuid4().hex,after_frame_id=after_id,
                            hold_ms=meter.HOLD_MS,time=_time().time(),result='observed_only')
                        self.fishing_evidence.record(feedback,observed[0].copy(),observed[-1].copy())
            with self.lock:
                self.state['clicks'] += 1
                self.state['matches'] += 1
                self.state['last_match'] = {'name': label, 'score': 1.,
                                           'box': [x-5, y-5, x+5, y+5]}
            self.event(label)
            return observed if observed else True

    def equipment_hold(x, y, seconds, weapon, label):
        _ctx=_bindings()
        self=_ctx['self']
        runtime=_ctx['runtime']
        equipment=_ctx['equipment']
        parallel=_ctx['parallel']
        stone=_ctx['stone']
        st=_ctx['st']
        max_clicks=_ctx['max_clicks']
        _time=_globals['_time']
        def budget_available():
            budget=getattr(equipment,'hold_budget_available',None)
            return ((not callable(budget) or budget(seconds))
                    and (st.run_deadline is None or _time().monotonic()+seconds <= st.run_deadline))
        with _scope(_ctx,'equipment_hold'):
            with self.device.io_lock:
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                if max_clicks is not None and self.state['clicks'] >= max_clicks:
                    raise InterruptedError('已达到点击次数限制')
                if not budget_available():
                    return False
                fresh = runtime.capture()
                self.remember(fresh)
                if (not equipment.validate_hold(fresh, weapon, x, y)
                        or parallel and not parallel.boss_sailing_hold_allowed(fresh)
                        or parallel and parallel.defense and parallel.defense.owns(fresh,parallel.rank)
                        or stone and (stone.defense.encounter(fresh,39)
                                      or stone.defense._match(fresh,'timer',(188,125,610,165),.94)
                                      or stone.defense._match(fresh,'swords',(307,962,415,1057),.96))):
                    self.event('武器强化：长按前画面变化，等待重新识别')
                    return False
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                if max_clicks is not None and self.state['clicks'] >= max_clicks:
                    raise InterruptedError('已达到点击次数限制')
                # Full visual validation can take time; check again immediately
                # before contact, not only at controller/capture entry.
                if not budget_available():
                    return False
                self.device.command('shell', 'input', 'swipe', str(x), str(y),
                                    str(x), str(y), str(int(seconds * 1000)))
            with self.lock:
                self.state['clicks'] += 1
                self.state['matches'] += 1
                self.state['last_match'] = {'name': label, 'score': 1.,
                                           'box': [x-5, y-5, x+5, y+5]}
            self.event(f'{label}，{seconds} 秒，坐标 ({x}, {y})')
            return True

    def backpack_click(x, y, label, action):
        _ctx=_bindings()
        self=_ctx['self']
        max_clicks=_ctx['max_clicks']
        st=_ctx['st']
        runtime=_ctx['runtime']
        parallel=_ctx['parallel']
        backpack=_ctx['backpack']
        parallel_click=_ctx['parallel_click']
        _time=_globals['_time']
        with _scope(_ctx,'backpack_click'):
            with self.device.io_lock:
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                if max_clicks is not None and self.state['clicks'] >= max_clicks:
                    raise InterruptedError('已达到点击次数限制')
                if st.run_deadline is not None and _time().monotonic() >= st.run_deadline:
                    raise InterruptedError('已达到运行时间限制')
                fresh = runtime.capture()
                self.remember(fresh)
                if (parallel is not None and parallel.sailing is not None
                        and parallel.sailing.running(fresh) is None):
                    # A recognized bag/map covers gameplay's lower AUTO. Its
                    # fresh top OFF icon and current world/RANK must positively
                    # establish stopping; unknown icons never authorize input.
                    panel_stopped = backpack.stopped_panel(fresh)
                    same_world_rank = parallel.sailing_context('bag',fresh)
                    if not panel_stopped or not same_world_rank:
                        self.event('背包出售：航行状态未知，拒绝输入并等待重新确认停止')
                        return False
                if (parallel is not None and parallel.sailing is not None
                        and (parallel.sailing.active or parallel.training.state.get('sailing_enabled')
                             or parallel.sailing.running(fresh) is True)):
                    self.event('背包出售：航行尚未停稳，等待重新确认')
                    return False
                if not backpack.validate(action, fresh, x, y):
                    self.event('背包出售：操作前画面变化，等待重新识别')
                    return False
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                if max_clicks is not None and self.state['clicks'] >= max_clicks:
                    raise InterruptedError('已达到点击次数限制')
                if st.run_deadline is not None and _time().monotonic() >= st.run_deadline:
                    raise InterruptedError('已达到运行时间限制')
                parallel_click(x, y, label)
            return True

    def backpack_swipe(x1, y1, x2, y2, label):
        _ctx=_bindings()
        self=_ctx['self']
        max_clicks=_ctx['max_clicks']
        st=_ctx['st']
        runtime=_ctx['runtime']
        parallel=_ctx['parallel']
        backpack=_ctx['backpack']
        parallel_swipe=_ctx['parallel_swipe']
        _time=_globals['_time']
        with _scope(_ctx,'backpack_swipe'):
            with self.device.io_lock:
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                if max_clicks is not None and self.state['clicks'] >= max_clicks:
                    raise InterruptedError('已达到点击次数限制')
                if st.run_deadline is not None and _time().monotonic() >= st.run_deadline:
                    raise InterruptedError('已达到运行时间限制')
                fresh = runtime.capture()
                self.remember(fresh)
                if (parallel is not None and parallel.sailing is not None
                        and (parallel.sailing.active or parallel.training.state.get('sailing_enabled'))):
                    self.event('背包出售：航行尚未停稳，等待重新确认地图')
                    return False
                if not backpack.validate_scan(fresh):
                    self.event('背包出售：滑动前世界、RANK、地点或停航状态变化，等待新图')
                    return False
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                if max_clicks is not None and self.state['clicks'] >= max_clicks:
                    raise InterruptedError('已达到点击次数限制')
                if st.run_deadline is not None and _time().monotonic() >= st.run_deadline:
                    raise InterruptedError('已达到运行时间限制')
                parallel_swipe(x1, y1, x2, y2, label)
            return True

    def skill_click(x, y, label, action):
        _ctx=_bindings()
        self=_ctx['self']
        runtime=_ctx['runtime']
        skills=_ctx['skills']
        parallel_click=_ctx['parallel_click']
        cruise_options=_globals['cruise_options']
        with _scope(_ctx,'skill_click'):
            with self.device.io_lock:
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                fresh = runtime.capture()
                self.remember(fresh)
                if action in ('open_growth','confirm') and getattr(skills,'options',None) is not None:
                    live=next((r for r in self.config['rules'] if r['id']=='driftinggirl_parallel_skills'),{})
                    desired=cruise_options(live)['skills']
                    if not live.get('enabled') or not desired['enabled'] or not desired['growth_points']:
                        skills.growth_points=False
                        return False  # Next owned step cancels an open purchase or closes the panel.
                if not skills.validate(action, fresh, x, y):
                    self.event('技能加速：点击前画面或余额变化，等待重新识别')
                    return False
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                skills.before_click(action, fresh)
                parallel_click(x, y, label)
            return True

    def pass_click(action,x,y,context):
        _ctx=_bindings()
        self=_ctx['self']
        runtime=_ctx['runtime']
        pass_task=_ctx['pass_task']
        parallel_click=_ctx['parallel_click']
        max_clicks=_ctx['max_clicks']
        st=_ctx['st']
        _lookup=_globals['_lookup']
        _time=_globals['_time']
        uuid=_globals['uuid']
        with _scope(_ctx,'pass_click'):
            with self.device.io_lock:
                def stopped():
                    return (self.stop_event.is_set() or _time().monotonic()>=pass_task.deadline
                        or max_clicks is not None and self.state['clicks']>=max_clicks
                        or st.run_deadline is not None and _time().monotonic()>=st.run_deadline)
                if stopped():return False
                fresh=runtime.capture();before_id=self.remember(fresh)
                period=self.daily_schedule.period()
                if not pass_task.validate(action,fresh,context,period) or stopped():return False
                if not pass_task.reserve(action):return False
                metadata=dict(input_id=uuid.uuid4().hex,action=action,frame_id=before_id,
                    x=x,y=y,period=period,time=_time().time(),result='reserved')
                after=None
                try:
                    if stopped():return False
                    metadata['result']='uncertain'
                    parallel_click(x,y,'PASS每日奖励：'+{'open':'打开PASS','tab':'选择每日页','claim':'领取免费奖励','close':'关闭PASS'}[action])
                    metadata['result']='issued_only'
                    after=runtime.capture();metadata['after_frame_id']=self.remember(after)
                finally:
                    if self.pass_evidence is None:
                        self.pass_evidence=_lookup('BossInputEvidence')(self.root,self.event,limit=16,folder='pass-daily',label='PASS每日奖励')
                    self.pass_evidence.record(metadata,fresh,after)
                return True

    def restaurant_click(action):
        _ctx=_bindings()
        self=_ctx['self']
        runtime=_ctx['runtime']
        restaurant=_ctx['restaurant']
        st=_ctx['st']
        max_clicks=_ctx['max_clicks']
        BossInputEvidence=_globals['BossInputEvidence']
        _time=_globals['_time']
        fresh_cruise_input=_globals['fresh_cruise_input']
        uuid=_globals['uuid']
        with _scope(_ctx,'restaurant_click'):
            with self.device.io_lock:
                def stopped():
                    return (self.stop_event.is_set() or not restaurant.allowed()
                        or max_clicks is not None and self.state['clicks']>=max_clicks
                        or st.run_deadline is not None and _time().monotonic()>=st.run_deadline)
                if stopped():return False
                if action['kind']=='stop':
                    def stop_allowed():
                        return (not stopped() and restaurant.enabled()
                            and restaurant.journal.get('attempts',{}).get('stop',0)<restaurant.input_limit('stop'))
                    def stop_touch(fresh,began):
                        if not restaurant.reserve(action) or not stop_allowed_after_reserve():return None
                        return self.device.touch_fast(action['x'],action['y'],began,
                            lambda:stopped() or not restaurant.enabled())
                    def stop_allowed_after_reserve():
                        return not stopped() and restaurant.enabled()
                    result=fresh_cruise_input(capture=runtime.capture,
                        allowed=stop_allowed,
                        validate=lambda fresh:restaurant.validate(action,fresh),
                        touch=stop_touch,clock=_time().monotonic)
                    if result is None:return False
                    fresh,contact=result;before_id=self.remember(fresh)
                    receipt=dict(input_id=uuid.uuid4().hex,frame_id=before_id,action=action,
                        time=_time().time(),result='issued_only',controller='restaurant_upgrade',**contact)
                    restaurant.input_issued(action,receipt)
                    with self.lock:self.state['clicks']+=1
                    self.event('餐品强化输入：stop (%s,%s) 40ms native/fresh' % (action['x'],action['y']))
                    after=None
                    if not self.stop_event.is_set():
                        after=runtime.capture();receipt['after_frame_id']=self.remember(after)
                    if self.restaurant_evidence is None:
                        self.restaurant_evidence=BossInputEvidence(self.root,self.event,limit=180,
                            folder='restaurant-input',label='餐品强化')
                    self.restaurant_evidence.record(receipt,fresh,after)
                    return True
                fresh=runtime.capture();before_id=self.remember(fresh)
                if not restaurant.validate(action,fresh) or stopped():return False
                if not restaurant.reserve(action) or stopped():return False
                receipt=dict(input_id=uuid.uuid4().hex,frame_id=before_id,action=action,
                    time=_time().time(),result='uncertain',controller='restaurant_upgrade')
                after=None
                try:
                    x,y=action['x'],action['y']
                    X,Y,ms=action.get('swipe',(x,y,action.get('hold_ms',80)))
                    self.device.command('shell','input','swipe',str(x),str(y),str(X),str(Y),str(ms))
                    with self.lock:self.state['clicks']+=1
                    receipt['result']='issued_only'
                    restaurant.input_issued(action,receipt)
                    self.event('餐品强化输入：'+action['kind']+f' ({x},{y}) {ms}ms')
                    if not self.stop_event.is_set():
                        after=runtime.capture();receipt['after_frame_id']=self.remember(after)
                        if action['kind'].startswith('scan:'):restaurant.observe_map_scan(fresh,after)
                finally:
                    if self.restaurant_evidence is None:
                        self.restaurant_evidence=BossInputEvidence(self.root,self.event,limit=180,
                            folder='restaurant-input',label='餐品强化')
                    self.restaurant_evidence.record(receipt,fresh,after)
                return True

    def return_parallel_action(action):
        _ctx=_bindings()
        self=_ctx['self']
        runtime=_ctx['runtime']
        return_parallel=_ctx['return_parallel']
        st=_ctx['st']
        max_clicks=_ctx['max_clicks']
        return_parallel_enabled=_ctx['return_parallel_enabled']
        _time=_globals['_time']
        with _scope(_ctx,'return_parallel_action'):
            with self.device.io_lock:
                def stopped():
                    return (self.stop_event.is_set() or not return_parallel_enabled()
                            or max_clicks is not None and self.state['clicks'] >= max_clicks
                            or st.run_deadline is not None and _time().monotonic() >= st.run_deadline)
                if stopped():return False
                fresh = runtime.capture()
                self.remember(fresh)
                if not return_parallel.validate(action, fresh) or stopped():return False
                x, y = action['x'], action['y']
                if action.get('swipe'):
                    X, Y, ms = action['swipe']
                    self.device.command('shell', 'input', 'swipe', str(x), str(y), str(X), str(Y), str(ms))
                else:
                    # 2026-10-27：与本文件 guild_daily_click 一致，用 80 毫秒定点点按。
                    # 零时长的 input tap 会被这里带过渡动画的地图/弹窗控件吞掉。
                    self.device.command('shell', 'input', 'swipe', str(x), str(y), str(x), str(y), '80')
                with self.lock:self.state['clicks'] += 1
                self.event('返回平行世界输入：' + action['kind'] + f' ({x},{y})')
                return True

    def guild_daily_click(action):
        _ctx=_bindings()
        self=_ctx['self']
        runtime=_ctx['runtime']
        guild_daily=_ctx['guild_daily']
        max_clicks=_ctx['max_clicks']
        st=_ctx['st']
        BossInputEvidence=_globals['BossInputEvidence']
        _time=_globals['_time']
        uuid=_globals['uuid']
        with _scope(_ctx,'guild_daily_click'):
            with self.device.io_lock:
                def stopped():
                    return (self.stop_event.is_set() or not guild_daily.allowed()
                        or max_clicks is not None and self.state['clicks']>=max_clicks
                        or st.run_deadline is not None and _time().monotonic()>=st.run_deadline)
                if stopped():return False
                began=_time().monotonic();fresh=runtime.capture();before_id=self.remember(fresh)
                if not guild_daily.validate(action,fresh) or stopped():return False
                fast=action['kind']=='cancel'
                if fast and _time().monotonic()-began>.12:return False
                if not guild_daily.reserve(action) or stopped():return False
                receipt=dict(input_id=uuid.uuid4().hex,frame_id=before_id,action=action,
                    time=_time().time(),result='uncertain',controller='guild_daily')
                after=None
                try:
                    x,y=action['x'],action['y']
                    if fast:
                        contact=self.device.touch_fast(x,y,began,stopped)
                        if contact is None:return False
                        receipt.update(contact)
                        with self.lock:self.state['clicks']+=1
                    elif action.get('swipe'):
                        X,Y,ms=action['swipe']
                        self.device.command('shell','input','swipe',str(x),str(y),str(X),str(Y),str(ms))
                        with self.lock:self.state['clicks']+=1
                    else:
                        self.device.command('shell','input','swipe',str(x),str(y),str(x),str(y),'80')
                        with self.lock:self.state['clicks']+=1
                    receipt['result']='issued_only'
                    guild_daily.input_issued(action,receipt)
                    self.event('公会任务输入：'+action['kind']+f' ({x},{y})')
                    if not self.stop_event.is_set():
                        after=runtime.capture();receipt['after_frame_id']=self.remember(after)
                finally:
                    if self.guild_daily_evidence is None:
                        self.guild_daily_evidence=BossInputEvidence(self.root,self.event,limit=80,
                            folder='guild-daily-input',label='公会任务')
                    self.guild_daily_evidence.record(receipt,fresh,after)
                return True

    return {'parallel_click':parallel_click,'parallel_swipe':parallel_swipe,'portal_click':portal_click,'completion_click':completion_click,'defense_click':defense_click,'sailing_click':sailing_click,'fishing_action':fishing_action,'equipment_hold':equipment_hold,'backpack_click':backpack_click,'backpack_swipe':backpack_swipe,'skill_click':skill_click,'pass_click':pass_click,'restaurant_click':restaurant_click,'return_parallel_action':return_parallel_action,'guild_daily_click':guild_daily_click}
