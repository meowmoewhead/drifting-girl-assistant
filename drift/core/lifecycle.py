"""Shared UI ownership, interruption context and read-only business health.

This layer never clicks, changes task journals or extends a controller deadline.
The controllers retain their scene-specific input guards and reservations.
"""
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import time

from drift.core.continuity import FreshConfirmation
from drift.core.workflow import WorkflowContext

_actor=ContextVar('drift_input_actor',default=None)


class InputOwnershipError(RuntimeError):
    pass


def active(task):
    value=getattr(task,'active',False)
    return bool(value() if callable(value) else value)


class UiOwnership:
    MANAGED={'equipment_upgrade','backpack_sale','backpack','parallel_skills','pass_daily',
             'guild_daily','restaurant_upgrade','return_parallel','meter_calibration',
             'stone_farm','cruise_setup','cruise_watchdog','parallel'}
    ALIASES={'backpack':'backpack_sale','equipment':'equipment_upgrade','skills':'parallel_skills',
             'stone':'stone_farm','boss_route_stop':'parallel','meter_stop_before_battle':'meter_calibration'}
    PRIORITY={'battle','completion','game_recovery','raid_cast','cleanup'}

    def __init__(self, snapshot=lambda:('main',{},False), clock=time.monotonic):
        self.snapshot=snapshot;self.clock=clock
        self.owners={};self.stage='main';self.popup=False
        self.interrupted=None;self.last_resume=None
        self.last_denial=None
        self.history=deque(maxlen=8)
        self.tasks={};self.rule=None;self.checkpoint=None;self.budget=lambda:{}

    def refresh(self):
        stage,tasks,popup=self.snapshot()
        self.tasks=tasks
        owners={}
        for name,task in tasks.items():
            if task is None:continue
            busy_value=getattr(task,'busy',False)
            busy=active(task) or bool(busy_value() if callable(busy_value) else busy_value)
            # Legacy parallel detects/plays continuously without owning a panel.
            if name=='parallel':busy=getattr(task,'phase','detect') not in ('detect','play')
            if name=='game_recovery':busy=getattr(task,'started_at',None) is not None
            if not busy:continue
            journal=getattr(task,'journal',{})
            owners[name]={'phase':getattr(task,'phase',getattr(task,'state',None)),
                'deadline':journal.get('deadline') if isinstance(journal,dict) else None,
                'attempts_sha256':hashlib.sha256(repr(sorted(journal.get('attempts',{}).items())).encode()).hexdigest()
                    if isinstance(journal,dict) else None}
        if popup and not self.popup:
            self.interrupted={'stage':stage,'owners':owners,'at':self.clock()}
        elif not popup and self.popup:
            self.last_resume={'stage':stage,'previous':self.interrupted,
                'at':self.clock(),'requires_fresh_controller_validation':True}
            self.history.append(self.last_resume)
            self.interrupted=None
        self.stage,self.owners,self.popup=stage,owners,bool(popup)
        return self

    def can_call(self,name):
        name=self.ALIASES.get(name,name)
        if name not in self.MANAGED:return True
        self.refresh();name=self.ALIASES.get(name,name)
        # Existing workflows finish/validate their original reserved attempt.
        # A new controller cannot steal a panel from one of those workflows.
        held=(self.checkpoint and (self.checkpoint.blocked or self.checkpoint.pending_stage) and name not in self.owners)
        allowed=(not held and not self.popup and (name in self.owners or self.stage=='main' and not self.owners))
        if not allowed:
            self.last_denial={'task':name,'stage':self.stage,'owners':list(self.owners),'popup':self.popup}
        return bool(allowed)

    def flush(self):
        self.refresh()
        if self.checkpoint:self.checkpoint.capture(self.tasks,stage=self.stage,popup=self.popup,budget=self.budget())

    @contextmanager
    def scope(self,name):
        actor='rule:'+name[5:] if name.startswith('rule_') and name!='rule_scan' else self.ALIASES.get(name,name)
        token=_actor.set(actor)
        try:yield
        finally:_actor.reset(token)

    def current_actor(self):
        return _actor.get()

    def select_rule(self,rule):
        self.rule={k:rule.get(k) for k in ('id','stage','interrupt','action','next_stage')}
        self.rule['stage']=rule.get('stage','main')

    def authorize(self,kind,args):
        self.refresh()
        actor=self.ALIASES.get(_actor.get(),_actor.get())
        if actor is None and self.rule:actor='rule:'+self.rule['id']
        rule=self.rule if self.rule and actor=='rule:'+self.rule['id'] else None
        if kind=='launch' and actor!='game_recovery':
            raise InputOwnershipError('游戏拉起只能由恢复控制器执行')
        if actor=='game_recovery' and self.checkpoint and actor in self.checkpoint.blocked:
            allowed=False
        elif actor in self.PRIORITY:
            allowed=True
        elif rule:
            allowed=bool(rule.get('interrupt') or (not self.owners and rule.get('stage') in (self.stage,'*')
                and (not self.popup or self.stage!='main')))
            if self.checkpoint and self.checkpoint.pending_stage and not rule.get('interrupt'):allowed=False
        else:
            allowed=(actor in self.MANAGED and not self.popup
                and (actor in self.owners or self.stage=='main' and not self.owners)
                and not (self.checkpoint and actor in self.checkpoint.blocked))
        if not allowed:
            self.last_denial={'task':actor,'stage':self.stage,'owners':list(self.owners),'popup':self.popup}
            raise InputOwnershipError('当前界面不属于本次输入，已拒绝发送：'+str(actor))
        # Authority-journalled tasks already reserve on disk before DOWN. Battle
        # windows retain their native limits without adding disk I/O to 120 ms.
        entry=(self.checkpoint.data['tasks'].get(actor,{}) if self.checkpoint else {})
        if self.checkpoint and (actor not in self.PRIORITY or actor=='game_recovery') and not entry.get('authority_journal'):
            self.checkpoint.capture(self.tasks,stage=self.stage,popup=self.popup,budget=self.budget())
            self.checkpoint.reserve(actor,kind,args,self.tasks,self.budget())
            # A finite sharing retry must not move an input past its original
            # deadline. The reservation remains spent when this final check fails.
            entry=self.checkpoint.data['tasks'].get(actor,{})
            budget=self.budget()
            limits=[entry.get('deadline'),budget.get('run_deadline_epoch')]
            if rule:limits.append(budget.get('stage_deadline_epoch'))
            if any(type(limit) in (int,float) and self.checkpoint.wall()>=limit for limit in limits):
                raise InputOwnershipError('原输入窗口已到期，保留预留记录并拒绝发送：'+str(actor))
            return self.checkpoint.data['last_input']['at']
        return None

    def issued(self,token):
        if token is not None and self.checkpoint:self.checkpoint.issued(token)

    def status(self):
        return {'stage':self.stage,'owners':self.owners,'popup':self.popup,
                'interrupted':self.interrupted,'last_resume':self.last_resume,
                'conflict':len(self.owners)>1,'last_denial':self.last_denial}


class BusinessProgress:
    def __init__(self, clock=time.monotonic, event=lambda _:None, *, idle_limit=90.):
        self.clock,self.event,self.idle_limit=clock,event,float(idle_limit)
        self.proof=FreshConfirmation(max_gap=15.)
        self.last_id=None;self.last_fishing=None;self.last_sample=None
        self.idle_since=None;self.context=None;self.last_notice=None
        self.value={'state':'unknown','reason':'等待新的实际钓鱼画面','evidence':None}
        self.evidence=deque(maxlen=8)

    def observe(self,frame_id,*,context=None,kind='unknown',owner=None):
        now=self.clock()
        if not frame_id or frame_id==self.last_id:return self.status()
        self.last_id,self.last_sample=frame_id,now
        if owner:
            self.proof.reset();self.idle_since=None
            self.value={'state':'occupied','reason':'当前任务占用界面：'+owner,'evidence':frame_id}
        elif kind in ('fill','battle','success'):
            self.last_fishing=now;self.idle_since=None;self.proof.reset()
            self.evidence.append({'frame_id':frame_id,'kind':kind,'at':now})
            self.value={'state':'fishing','reason':{'fill':'已见圆内真实填充','battle':'已见实际战斗',
                         'success':'已见钓鱼成功'}[kind],'evidence':frame_id}
        elif context is None or kind=='unknown':
            self.proof.reset();self.idle_since=None
            self.value={'state':'unknown','reason':'画面遮挡或状态未知，等待新图','evidence':frame_id}
        elif kind in ('idle','port'):
            if context!=self.context or self.idle_since is None:self.idle_since=now
            self.context=context
            confirmed=self.proof.observe((context,kind),frame_id,now)
            seconds=now-self.idle_since
            stalled=confirmed and seconds>=self.idle_limit
            reason=('港口停留且未见实际钓鱼' if kind=='port' else '持续灰色鱼钩或休眠，未见实际钓鱼')
            self.value={'state':'stalled' if stalled else 'waiting','reason':reason,
                        'idle_seconds':round(seconds,1),'evidence':frame_id}
            if stalled and (self.last_notice is None or now-self.last_notice>=300):
                self.event('实际进展观测：'+reason+'；保留记录，等待原安全恢复流程，不重启或补点')
                self.last_notice=now
        else:
            self.proof.reset();self.idle_since=None
            self.value={'state':'waiting','reason':'航行或抛竿准备中，尚未确认实际钓鱼','evidence':frame_id}
        return self.status()

    def status(self):
        now=self.clock()
        return {**self.value,'sample_age_seconds':None if self.last_sample is None else round(now-self.last_sample,1),
            'last_fishing_seconds_ago':None if self.last_fishing is None else round(now-self.last_fishing,1),
            'recent_evidence':list(self.evidence)}


class RuntimeLifecycle:
    def __init__(self,snapshot,*,event=lambda _:None,clock=time.monotonic,root=None,serial=None,budget=lambda:{}):
        self.clock=clock;self.ui=UiOwnership(snapshot,clock)
        self.health=BusinessProgress(clock,event)
        self.sampled_at=None
        self.context=WorkflowContext(root,serial,event,monotonic=clock) if root is not None else None
        self.ui.checkpoint=self.context;self.ui.budget=budget
        self.registered={}
        self.release_proof=FreshConfirmation(max_gap=8.)

    def observe(self,frame,frame_id,*,probe=None,enabled=True):
        self.ui.refresh()
        if self.context is not None:
            for name,task in self.ui.tasks.items():
                if task is not None and self.registered.get(name)!=id(task):
                    self.context.restore(name,task);self.registered[name]=id(task)
            self.ui.refresh()
            self.context.capture(self.ui.tasks,stage=self.ui.stage,popup=self.ui.popup,budget=self.ui.budget())
        owner=('弹窗恢复' if self.ui.popup else self.ui.stage if self.ui.stage!='main'
               else ', '.join(self.ui.owners))
        now=self.clock()
        if owner:
            self.health.observe(frame_id,owner=owner);return
        if self.sampled_at is not None and now-self.sampled_at<3:return
        self.sampled_at=now
        if not enabled or probe is None:
            self.health.observe(frame_id);return
        meter=getattr(probe,'fishing',None)
        if (meter is None or not meter._main_visible(frame)
                or getattr(probe,'sailing',None) is not None and probe.sailing.menu(frame)
                or callable(getattr(probe,'_map',None)) and probe._map(frame)):
            self.health.observe(frame_id);return
        # Fixed text strokes only; no countdown or animated-background pixels.
        strokes=b''.join((frame[y:Y,x:X].max(axis=2)<70).tobytes()
            for x,y,X,Y in ((71,9,136,59),(40,45,68,71),(286,12,434,34)))
        context=hashlib.sha256(strokes).hexdigest()
        if meter.real_battle(frame):kind='battle'
        elif meter.blue_fill(frame):kind='fill'
        elif meter.at_port(frame):kind='port'
        elif callable(getattr(probe,'_moving',None)) and probe._moving(frame):kind='travel'
        elif meter.idle_icon(frame) in ('sleep','hook'):kind='idle'
        elif meter.cast_preparation(frame,allow_task=True):kind='preparation'
        else:kind='unknown'
        self.health.observe(frame_id,context=context,kind=kind)
        if self.context is not None:
            if self.release_proof.observe(context,frame_id,now):self.context.release_verified_main()

    def status(self):
        entry=getattr(self.ui.tasks.get('stone_farm'),'entry_recovery',None)
        return {'ui':self.ui.status(),'health':self.health.status(),
                'continuity':self.context.status() if self.context else None,
                'entry_recovery':entry.status() if entry else None}
