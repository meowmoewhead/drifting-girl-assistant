"""Shared tasks extracted from the aligned runtime; original guards retained."""
def build_services(_bindings,_globals):
    def finish_scheduled_daily(success=False):
        _ctx=_bindings()
        st=_ctx['st']
        self=_ctx['self']
        if st.scheduled_daily is not None:
            feature, interval, started_period = st.scheduled_daily
            daily = self.daily_schedule.status()[feature]
            if daily['period'] != started_period:
                daily = {**daily, 'period': started_period, 'completed': False}
            self.task_schedule.finish(feature, interval, daily, success=success, outcome='complete' if success else 'skipped' if st.skip_on_timeout else 'retry')
            st.scheduled_daily = None

    def step_scheduled_controller(feature, label, controller, image):
        _ctx=_bindings()
        self=_ctx['self']
        st=_ctx['st']
        runtime=_ctx['runtime']
        invoke = lambda fn, picture: runtime.call(label, fn, picture)
        if feature == 'parallel_skills' and getattr(controller,'force_check',False):
            owned=invoke(controller.step,image)
            if controller.cycles:
                controller.force_check=False
                self.task_schedule.finish(feature, controller.interval, success=True, controller=controller)
            return owned
        if st.scheduled:
            return self.task_schedule.step_controller(feature, controller, image, invoke)
        return invoke(controller.step, image)

    def step_restaurant(image,allow_start=True):
        _ctx=_bindings()
        restaurant=_ctx['restaurant']
        self=_ctx['self']
        runtime=_ctx['runtime']
        st=_ctx['st']
        stop_on_main=_ctx['stop_on_main']
        RESTAURANT_FAILURE_LIMIT=_globals['RESTAURANT_FAILURE_LIMIT']
        was_active=restaurant.active
        due=allow_start and self.task_schedule.ready('restaurant_upgrade')
        try:
            owned=runtime.call('restaurant_upgrade',restaurant.step,image,due=due)
        except RuntimeError as exc:
            # Count failed runs on the runner, not frames or controllers:
            # waiting and config reloads must not permit infinite retries.
            self.restaurant_failures+=1
            restaurant.abandon(str(exc))
            self.task_schedule.finish('restaurant_upgrade', restaurant.interval, success=False, retry_seconds=1800, controller=restaurant)
            st.stable={key:0 for key in st.stable}
            st.number_values.clear();st.target_keys.clear()
            with self.lock:
                self.state['restaurant_upgrade']=restaurant.status()
                self.state['stage']='main'
            if self.restaurant_failures>=RESTAURANT_FAILURE_LIMIT:
                self.event(f'餐品强化：连续失败达到 {RESTAURANT_FAILURE_LIMIT} 次，停止运行（{exc}）')
                raise
            self.event(f'餐品强化：本轮中止，30 分钟后重试，运行继续（{exc}）')
            return False
        if not was_active and restaurant.active:
            self.task_schedule.begin('restaurant_upgrade',restaurant.interval)
        if restaurant.done is not None:
            if restaurant.done:self.restaurant_failures=0
            # Every incomplete run, including successful UI cleanup after
            # a confirmation timeout, retries without the full interval.
            retry=1800 if not restaurant.done else None
            if retry:
                self.task_schedule.finish('restaurant_upgrade', restaurant.interval, success=bool(restaurant.done), retry_seconds=retry, controller=restaurant)
            else:
                self.task_schedule.finish('restaurant_upgrade', restaurant.interval, success=bool(restaurant.done), controller=restaurant)
            if stop_on_main:
                with self.lock:self.state['stop_reason']='cycle_complete' if restaurant.done else 'observation_complete'
                self.stop_event.set()
        with self.lock:
            self.state['restaurant_upgrade']=restaurant.status()
            if owned:self.state['stage']='restaurant_'+restaurant.phase
        return owned

    def step_guild_daily(image,allow_start=True):
        _ctx=_bindings()
        self=_ctx['self']
        guild_daily=_ctx['guild_daily']
        runtime=_ctx['runtime']
        daily=self.daily_schedule.status()['guild_daily']
        was_active=guild_daily.active
        due=(allow_start and not daily['completed'] and self.task_schedule.ready('guild_daily',daily))
        try:
            owned=runtime.call('guild_daily',guild_daily.step,image,due=due,period=daily['period'])
        except RuntimeError as exc:
            # 单个任务超时不该停掉整个 runner:2026-10-26 18:12 公会往返超时直接把
            # 主循环带走,18:12:20→21:43:25 停摆 3 h 31 min,期间所有任务全丢。
            retry=1800
            self.event(f'公会任务：本次往返中止，{retry//60} 分钟后重试，运行继续（{exc}）')
            try:guild_daily.abandon()
            except Exception:self.event('公会任务：中止时清理记录失败，按原样保留')
            self.task_schedule.finish('guild_daily', guild_daily.interval, daily, success=False, retry_seconds=retry, controller=guild_daily)
            with self.lock:self.state['guild_daily']=guild_daily.status()
            return False
        if not was_active and guild_daily.active:
            self.task_schedule.begin('guild_daily',guild_daily.interval,daily)
        if guild_daily.done is not None:
            if guild_daily.done:self.daily_schedule.mark_complete('guild_daily',guild_daily.started_period)
            self.task_schedule.finish('guild_daily', guild_daily.interval, self.daily_schedule.status()['guild_daily'], success=bool(guild_daily.done), controller=guild_daily)
        with self.lock:
            self.state['guild_daily']=guild_daily.status()
            if owned:self.state['stage']='guild_daily_'+guild_daily.phase
        return owned

    def step_pass(image):
        _ctx=_bindings()
        self=_ctx['self']
        pass_task=_ctx['pass_task']
        st=_ctx['st']
        runtime=_ctx['runtime']
        PASS_ID=_globals['PASS_ID']
        daily=self.daily_schedule.status()['pass_daily']
        due=(not daily['completed'] and self.task_schedule.ready('pass_daily',daily))
        was_active=pass_task.active
        if was_active and not any(r['id']==PASS_ID and r['enabled'] for r in self.config['rules']):
            pass_task.claim_allowed=False
        ready=(pass_task.main_visible(image) or
               (self.daily_return_visible(image,st.matchers) if 'driftinggirl_daily_return' in st.matchers else
                pass_task.entry(image)))
        owned=runtime.call('pass_daily',pass_task.step,image,due=due,period=daily['period'],main_ready=ready)
        if not was_active and pass_task.active:
            self.task_schedule.begin('pass_daily',pass_task.interval,daily)
        if pass_task.done is not None:
            if pass_task.done:
                self.daily_schedule.mark_complete('pass_daily',pass_task.started_period)
            self.task_schedule.finish('pass_daily', pass_task.interval, self.daily_schedule.status()['pass_daily'], success=bool(pass_task.done), retry_seconds=None if pass_task.done else pass_task.interval, controller=pass_task)
        return owned

    return {'finish_scheduled_daily':finish_scheduled_daily,'step_scheduled_controller':step_scheduled_controller,'step_restaurant':step_restaurant,'step_guild_daily':step_guild_daily,'step_pass':step_pass}
