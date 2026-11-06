"""Apply saved native cruise options at a safe boundary; bound coupon holds."""
from pathlib import Path
import hashlib
import json
import time

import cv2
import numpy as np

from drift.tasks.cruise_options import TARGET_NAMES, fingerprint
from drift.tasks.cruise_setup import OPEN_AT, START_AT, CLOSE_AT, SALE_AT, UPGRADE_AT, TARGET_X, TARGET_Y
from drift.tasks.cruise_watchdog import CruiseView
from drift.tasks.guild_daily import GuildVisual
from drift.tasks.parallel_skills import ParallelSkills
from drift.core.journal import archive_bytes

TICKET_AT = (576, 697)


class SettingsView(CruiseView):
    def __init__(self, owner, root, reader):
        super().__init__(owner)
        self.reader = reader
        self.guild = GuildVisual(root)
        self.original_seen = False
        self.original_rank_ink = None
        self.original_location = None
        self.resume_scene = None
        self.ticket_templates = {}
        world_battle=cv2.imread(str(Path(root)/'assets/navigation/world-battle.png'))
        self.whale=(cv2.cvtColor(world_battle,cv2.COLOR_BGR2RGB)[156:205,116:184].copy()
                    if world_battle is not None else None)
        for name in ('ticket-label', 'ticket-plus'):
            image = cv2.imread(str(Path(root)/'assets/cruise-settings'/f'{name}.png'))
            if image is None:
                raise RuntimeError('巡航补票识别模板缺失：'+name)
            self.ticket_templates[name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def rank_ink(frame):
        crop = frame[9:59,71:136].min(axis=2)
        return (crop >= crop.max()*.84).tobytes()

    def restore_context(self, scene):
        # A receipt binds identity, never coordinates or permission to repeat an input.
        # Adoption still requires a fresh complete menu, OFF, and matching RANK ink.
        if (isinstance(scene,(list,tuple)) and len(scene)==2 and scene[0]=='original'
                and isinstance(scene[1],str) and len(scene[1])==64):
            self.resume_scene=tuple(scene)

    def observe_setup(self, frame):
        who = self.owner
        if frame.shape != (1280,720,3) or who.active or not who.enabled():
            return None
        if who._parallel(frame):
            self.original_seen = False
            self.resume_scene = None
            return super().observe_setup(frame)
        journal=getattr(who,'journal',{})
        retirement=journal.get('retirement') or {}
        if (journal.get('reason')=='lower' and retirement and not retirement.get('released')
                and not journal.get('entered_verified_at')):
            # An expired lower handoff still owes a portal return. Do not turn
            # a stopped original-world recovery into an unrelated farm route.
            return None
        menu = who.sailing.menu(frame)
        if not menu:
            if (not who.fishing._main_visible(frame) or who._battle(frame)
                    or not (self.guild.point(frame,'g-icon') or who._observatory(frame)
                            or self.original_seen and self.rank_ink(frame)==self.original_rank_ink)):
                # Battle/animation can briefly hide the menu. Keep the proved
                # identity; every later menu/input still checks fresh fixed pixels.
                return None
            self.original_seen = True
            self.original_rank_ink = self.rank_ink(frame)
        else:
            current_ink=self.rank_ink(frame)
            if not who._off_icon(frame):return None
            if not self.original_seen:
                if self.resume_scene != ('original',hashlib.sha256(current_ink).hexdigest()):return None
                self.original_seen=True
                self.original_rank_ink=current_ink
            if current_ink != self.original_rank_ink:return None
        return dict(scene=('original', hashlib.sha256(self.original_rank_ink).hexdigest()), menu=menu,
                    signature=self.signature(frame, menu))

    def safe_main(self, frame):
        who = self.owner
        # A task skull can remain throughout travel and between encounters.
        # Navigation settings do not cancel fishing or touch either fishing X.
        return who.fishing._main_visible(frame) and not who._battle(frame)

    def fishing_scope(self, frame, context):
        return json.dumps([context['scene'],self.owner.fishing._context(frame)])

    def fishing_idle(self, frame, context):
        who, meter = self.owner, self.owner.fishing
        if (context['menu'] or not self.safe_main(frame) or who.training.boss_mode(frame)
                or who._no_destination(frame)
                or meter.at_port(frame) or who._moving(frame) or meter.busy(frame)
                or meter.cancel_button(frame) or meter.cast_preparation(frame,allow_task=True)):
            return None
        icon = meter.idle_icon(frame)
        return (self.fishing_scope(frame,context),icon) if icon in ('sleep','hook') else None

    def fishing_evidence(self, frame, context):
        if context['menu'] or self.owner.fishing.at_port(frame):return False
        meter=self.owner.fishing
        if meter.blue_fill(frame) or meter.real_battle(frame):return True
        # Original-world whale rescue has no small HP sword. Its own fixed
        # whale icon, timer AND full crossed-weapons footer are all required.
        if context['scene'][0]!='original' or self.whale is None:return False
        score=cv2.minMaxLoc(cv2.matchTemplate(frame[153:208,113:187],self.whale,
                                             cv2.TM_CCOEFF_NORMED))[1]
        defense=meter.start_defense
        return bool(np.isfinite(score) and score>=.96
                    and defense._match(frame,'timer',(188,125,610,165),.94)
                    and defense._battle_footer(frame))

    def observe_fishing(self, frame, scene):
        # Read-only proof for the owned start. Battles remain forbidden for menu inputs.
        who = self.owner
        if (frame.shape != (1280,720,3) or not who.enabled() or who.active
                or who.sailing.menu(frame) or who.sailing.running(frame) is None
                or not who.fishing._main_visible(frame)):
            return None
        if scene[0] != 'original':
            return self.observe_setup(frame)
        if (who._parallel(frame) or not self.original_seen
                or scene != ('original',hashlib.sha256(self.rank_ink(frame)).hexdigest())):
            return None
        return dict(scene=scene,menu=False,signature=self.signature(frame,False))

    def fishing_guard(self, frame):
        meter = self.owner.fishing
        cast = meter.cast_data(frame)
        if cast is None:
            return None
        parts = [np.all(frame[y:Y,x:X]>=200,axis=2).tobytes() for x,y,X,Y in
                 ((237,930,283,954),(234,1070,284,1100),(76,1248,137,1278))]
        return {**{k:cast[k] for k in ('context','box','digit_strokes','world_strokes')},
                'numbers':hashlib.sha256(b''.join(parts)).hexdigest()}

    def start_reading(self, frame):
        """Starting the existing selection does not drag or infer covered limits."""
        meter=self.owner.fishing
        data=meter.measure(frame,remember=False)
        if data is not None:return data
        if (not self.safe_main(frame) or not meter.world_matches(frame)
                or self.owner.training.boss_mode(frame) or meter.at_port(frame)
                or not meter.adaptive_auto_on(frame)):
            return None
        box=meter.box(frame)
        if box is None:return None
        value=meter.selected(frame,box)
        player=meter.training._number(frame,((76,1248,137,1278),(79,1249,136,1277)))
        if value is None or player is None:return None
        return dict(value=value,player=player,low=None,high=None,unchanged_selection=True)

    def validate_action(self, frame, expected, action, options):
        who = self.owner
        if (expected is None or frame.shape != (1280,720,3) or not who.enabled() or who.active):
            return False
        menu = who.sailing.menu(frame)
        if (menu != expected['menu'] or self.signature(frame,menu) != expected['signature']
                or who._parallel(frame) != (expected['scene'][0] != 'original')):
            return False
        if action == 'stop':
            return not menu and self.safe_main(frame) and who.sailing.running(frame) is True
        if not who._off_icon(frame):
            return False
        if action == 'fish-start':
            # The signature above compares fresh RANK, stage and location ink
            # with this iteration's fully decoded scene. OCR here would age
            # every otherwise valid native frame beyond the 120 ms limit.
            current=dict(scene=expected['scene'],menu=False,signature=expected['signature'])
            return bool(self.fishing_idle(frame,current) is not None
                        and who.sailing.running(frame) is False
                        and self.fishing_guard(frame)==expected.get('fishing_guard'))
        if action == 'open':
            return not menu and self.safe_main(frame) and who.sailing.running(frame) is False
        if not menu:
            return False
        if action == 'close':
            return True
        values = self.values(frame)
        if values is None:
            return False
        if action == 'start':
            return values == self.wanted(options)
        if action == 'tickets':
            return (options['auto_sale'] and options['ticket_min'] > 0
                    and type(expected.get('ticket_count')) is int
                    and 0 <= expected['ticket_count'] < options['ticket_min']
                    and who.sailing.sale_selected(frame) is True and self.ticket_anchors(frame)
                    and expected.get('ticket_ink') == self.ticket_ink(frame))
        if action in TARGET_NAMES:
            return values['targets'][action] is not options['targets'][action]
        return action in ('auto_sale','auto_upgrade') and values[action] is not options[action]

    def values(self, frame):
        sail = self.owner.sailing
        values = dict(targets={n:sail.selected(frame,n) for n in TARGET_NAMES},
                      auto_sale=sail.sale_selected(frame), auto_upgrade=sail.upgrade_selected(frame))
        return None if (None in values['targets'].values() or values['auto_sale'] is None
                        or values['auto_upgrade'] is None) else values

    @staticmethod
    def wanted(options):
        return {k:options[k] for k in ('targets','auto_sale','auto_upgrade')}

    def ticket_anchors(self, frame):
        if not self.owner.sailing.menu(frame):
            return False
        for name, (x,y,X,Y) in {'ticket-label':(280,681,455,716), 'ticket-plus':(556,676,597,719)}.items():
            # The count and its label are right aligned together: digit-width changes
            # move the fixed label, while the purchase button stays in place.
            dx=40 if name=='ticket-label' else 2
            score=cv2.minMaxLoc(cv2.matchTemplate(frame[y-2:Y+2,x-dx:X+dx],
                self.ticket_templates[name],cv2.TM_CCOEFF_NORMED))[1]
            if not np.isfinite(score) or score < .96:
                return False
        return True

    def tickets(self, frame):
        if not self.ticket_anchors(frame):
            return None
        text=ParallelSkills._digits(self,frame[683:715,385:517],(35,90),r'[0-9]{1,5}',
                                   crop_views=((100,2),(0,3)))
        return int(text) if text is not None else None

    @staticmethod
    def ticket_ink(frame):
        hsv=cv2.cvtColor(frame[683:715,385:517],cv2.COLOR_RGB2HSV)
        return ((hsv[:,:,0]>=35)&(hsv[:,:,0]<=90)&(hsv[:,:,1]>=60)&(hsv[:,:,2]<210)).tobytes()


class CruiseSettings:
    """Separate application receipts from game progress/purchase validity journals."""
    LIMIT = 180
    MAX_HOLDS = 10
    HOLD_MS = 2000
    TICKET_CONFIRM_SECONDS = 10
    ROUTE_RESERVE_SECONDS = 20

    def __init__(self, *, root, view, options, click, event, interval=300, clock=time.time):
        self.view, self.options, self.click, self.event = view, options, click, event
        self.clock, self.interval = clock, max(60,float(interval))
        self.path=Path(root)/'cruise_settings_state.json'
        self.journal=json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
        self.policy=fingerprint(options)
        self.state='idle'
        self.input_context=None
        self.scene=None
        self.key=None
        self.seen=0
        self.last_frame=None
        self.hold='等待明确主画面'
        self.deadline=None
        self.done=False
        self.retry_at=0
        self.ticket_before=None
        self.fish_key=self.fish_since=self.fish_seen_at=None
        self.fish_frames=0
        self.input_rejections={};self.last_input_rejection=None;self.rejection_logged_at=None
        restore=getattr(self.view,'restore_context',None)
        if callable(restore):restore(self.journal.get('scene'))
        if self.journal.get('pending'):
            # A restart archives the old input budget; no spent contact is replayed.
            self._end(False, '启动时归档未完成设置，本轮按原次数保留')
        self._release_unspent_start()

    def _release_unspent_start(self):
        old=self.journal.get('last_result') or {}
        scope=self.journal.get('fishing_blocked_scope')
        if (not scope or old.get('fishing_start_scope')!=scope or old.get('success') is not False
                or old.get('attempts',{}).get('fish-start',0)>0
                or type(old.get('deadline')) not in (int,float)
                or self.clock()<old['deadline']):return
        history=self.path.with_suffix('.history.jsonl')
        # Read the complete bounded journal; an omitted old input cannot be
        # used to justify release. Large/unknown histories remain blocked.
        if not history.is_file() or history.stat().st_size>1048576:return
        try:
            with history.open('rb') as stream:
                chunks=[stream.read(65536) for _ in range(16)]
                if stream.read(1):return
            rows=[json.loads(line) for line in b''.join(chunks).decode('utf-8').splitlines() if line]
        except (ValueError,OSError):return
        if old not in rows or any(row.get('fishing_start_scope')==scope
                and row.get('attempts',{}).get('fish-start',0)>0 for row in rows):return
        raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        archive_bytes(self.path.parent/'logs/cruise-settings-retired'/(digest+'.json'),raw,'巡航设置')
        if self.path.read_bytes()!=raw:raise RuntimeError('巡航记录已变化，拒绝清理未输入的启动等待')
        self.journal.pop('fishing_blocked_scope')
        self.journal['unspent_start_release']=dict(at=self.clock(),previous_sha256=digest,
            scope=scope,previous_deadline=old['deadline'],read_only=True)
        self._save()
        self.event('开局钓鱼：旧窗口未发出任何启动输入，归档后解除误锁；旧失败、截止及已用航巡次数保留')

    @property
    def busy(self):
        return self.state != 'idle'

    def observe_input_rejection(self, action, detail):
        key=action+':'+detail['reason']
        if key not in self.input_rejections and len(self.input_rejections)>=24:key='other'
        self.input_rejections[key]=self.input_rejections.get(key,0)+1
        self.last_input_rejection=dict(action=action,**detail)
        now=self.clock()
        if self.rejection_logged_at is None or now-self.rejection_logged_at>=30:
            self.rejection_logged_at=now
            self.event('巡航设置输入拒绝：'+json.dumps(self.last_input_rejection,ensure_ascii=False))

    def status(self):
        return dict(state=self.state, done=False, hold=self.hold, deadline=self.deadline,
                    policy=self.policy, applied_policy=self.journal.get('applied_policy'),
                    tickets=self.journal.get('tickets'), attempts=self.journal.get('attempts',{}),
                    last_result=self.journal.get('last_result'),
                    fishing_requested=self.journal.get('fishing_start_requested',False),
                    fishing_verified=self.journal.get('fishing_start_verified',False),
                    fishing_blocked_scope=self.journal.get('fishing_blocked_scope'),
                    input_rejections=dict(self.input_rejections),
                    last_input_rejection=self.last_input_rejection,
                    rank_observation=getattr(self.view.owner,'menu_rank_probe',None))

    def _save(self):
        from drift.core.persistence import atomic_write_text
        atomic_write_text(self.path, json.dumps(self.journal, ensure_ascii=False, indent=2))

    def _seen(self, key):
        self.seen=self.seen+1 if self.key==key else 1
        self.key=key
        return self.seen >= 3

    def _set(self, state):
        self.state=state
        self.seen=0;self.key=None

    def _fishing_proof(self, key, now):
        if key is None:
            self.fish_key=self.fish_since=self.fish_seen_at=None;self.fish_frames=0
            return False
        if key!=self.fish_key or self.fish_seen_at is None or now-self.fish_seen_at>8:
            self.fish_key,self.fish_since,self.fish_frames=key,now,1
        else:self.fish_frames+=1
        self.fish_seen_at=now
        return self.fish_frames>=3 and now-self.fish_since>=3

    def allowed(self, action):
        if not self.busy or self.deadline is None or self.clock()>=self.deadline:
            return False
        if (action=='tickets' and self.deadline-self.clock()
                < self.HOLD_MS/1000+self.TICKET_CONFIRM_SECONDS+self.ROUTE_RESERVE_SECONDS):
            return False
        limit=self.MAX_HOLDS if action=='tickets' else 1
        return self.journal.get('attempts',{}).get(action,0)<limit

    def reserve(self, action):
        if not self.allowed(action):return False
        counts=self.journal.setdefault('attempts',{})
        counts[action]=counts.get(action,0)+1
        self._save()
        return True

    def _input(self, action):
        point={'stop':OPEN_AT,'open':OPEN_AT,'start':START_AT,'close':CLOSE_AT,
               'auto_sale':SALE_AT,'auto_upgrade':UPGRADE_AT,'tickets':TICKET_AT,
               'fish-start':(360,1010)}.get(action)
        if action in TARGET_NAMES:point=(TARGET_X[action],TARGET_Y)
        if not self.allowed(action):
            if (action=='tickets' and self.deadline is not None and self.deadline-self.clock()
                    < self.HOLD_MS/1000+self.TICKET_CONFIRM_SECONDS+self.ROUTE_RESERVE_SECONDS):
                self.hold='剩余时间不足以补票并核对票数，本轮先恢复航巡；下次按实际票数检查'
                self.event('巡航设置：'+self.hold)
            else:
                self.hold='已用完本轮 '+action+' 输入次数，等待画面确认'
            return False
        return self.click(action,*point,1500 if action=='fish-start' else self.HOLD_MS if action=='tickets' else 40)

    def _end(self, success, reason):
        now=self.clock()
        snapshot={k:self.journal.get(k) for k in ('policy','deadline','attempts','started_at','scene')}
        snapshot.update({k:self.journal.get(k) for k in
                         ('fishing_start_requested','fishing_start_scope','fishing_start_verified')})
        snapshot.update(success=success,reason=reason,ended_at=now)
        with self.path.with_suffix('.history.jsonl').open('a',encoding='utf-8') as out:
            out.write(json.dumps(snapshot,ensure_ascii=False)+'\n')
        self.journal.update(pending=False,last_result=snapshot,checked_at=now)
        if (self.journal.get('fishing_start_requested') and not self.journal.get('fishing_start_verified')
                and self.journal.get('attempts',{}).get('fish-start',0)>0):
            self.journal['fishing_blocked_scope']=self.journal.get('fishing_start_scope')
        if success:
            self.journal.update(applied_policy=self.policy,applied_scene=self.scene)
        self._save()
        self.hold=reason
        self.event('巡航设置：'+reason)
        self.retry_at=now+self.interval
        self._set('idle')
        self.deadline=None

    def step(self, frame):
        if frame is self.last_frame:return self.busy
        self.last_frame=frame
        now=self.clock()
        if self.busy and now>=self.deadline:
            self._end(False,'本轮设置窗口到期，保留输入记录，稍后重查')
            return False
        if self.state=='fish_verify' and now>=self.journal.get('fishing_verify_deadline',self.deadline):
            self._end(False,'开局长按后未确认实际钓鱼，保留已用次数，不重复长按')
            return False
        context=self.view.observe_setup(frame)
        late_observation=(self.state=='idle' and self.journal.get('fishing_blocked_scope'))
        if context is None and (self.state=='fish_verify' or late_observation):
            observe=getattr(self.view,'observe_fishing',None)
            observed_scene=self.scene or tuple(self.journal.get('scene') or ())
            if callable(observe) and observed_scene:context=observe(frame,observed_scene)
        if context is None:
            self.key=None;self.seen=0
            self._fishing_proof(None,now)
            if self.busy:
                self.hold='等待当前设置页的世界、阶段和RANK复核；未使用缓存数字'
                probe=getattr(self.view.owner,'menu_rank_probe',None)
                if probe and now-getattr(self,'unknown_logged_at',float('-inf'))>=30:
                    self.unknown_logged_at=now
                    self.event('巡航设置等待：'+self.hold+'；'+str(probe))
            return self.busy
        self.input_context=context
        scene=context['scene']
        if self.busy and scene!=self.scene and not (self.state=='verify' and scene[0]==self.scene[0]):
            self._end(False,'世界或RANK变化，结束本轮输入')
            return False
        menu=context['menu'];route=self.view.owner.sailing.running(frame)
        idle_probe=getattr(self.view,'fishing_idle',None)
        evidence_probe=getattr(self.view,'fishing_evidence',None)
        idle=idle_probe(frame,context) if callable(idle_probe) else None
        evidence=bool(evidence_probe(frame,context)) if callable(evidence_probe) else False
        scope=self.view.fishing_scope(frame,context) if callable(getattr(self.view,'fishing_scope',None)) else None
        if self.state in ('fish_idle','fish_verify') and scope!=self.journal.get('fishing_start_scope'):
            self._end(False,'开局钓鱼地点变化，保留次数并结束本轮输入')
            return False
        fish_ready=self._fishing_proof(('idle',idle,route) if idle else
            ('active',scope,route) if evidence else None,now)
        if evidence and fish_ready and self.journal.get('fishing_blocked_scope')==scope:
            self.journal.pop('fishing_blocked_scope')
            self.journal['fishing_late_observed_at']=now
            self._save()
            self.retry_at=now
            self.event('开局钓鱼：只读连续新图确认已有实际钓鱼，保留旧失败、截止和次数；后续仅核对原航巡设置')
        if self.busy and self.clock()>=self.deadline-5 and menu:
            # Close only while the original budget is still valid; never grant a new input window.
            if self._input('close'):
                self._end(False,'设置未能完成，已关闭菜单，保留原输入记录')
            return True
        if self.state=='idle':
            startup=bool(idle and fish_ready and scope!=self.journal.get('fishing_blocked_scope'))
            if idle and not fish_ready:return False
            if now<self.retry_at and not startup:return False
            # Only recheck an established running route for a new policy or a due ticket audit.
            applied=self.journal.get('applied_policy')==self.policy
            same_world=(self.journal.get('applied_scene') or [None])[0]==scene[0]
            due=(self.options['auto_sale'] and self.options['ticket_min']>0
                 and now-self.journal.get('checked_at',0)>=self.interval)
            if applied and same_world and route is True and not due and not startup:return False
            if not menu and (route is None or not self.view.safe_main(frame)):return False
            if not self._seen(('begin',scene,menu,route)):return False
            self.scene=scene
            self.deadline=now+self.LIMIT
            self.journal.update(pending=True,policy=self.policy,scene=scene,deadline=self.deadline,
                                started_at=now,attempts={},fishing_start_requested=startup,
                                fishing_start_scope=scope if startup else None,fishing_start_verified=False)
            self._save()
            self._set('menu' if menu else 'stop' if route else 'fish_idle' if startup else 'open')
        if self.state=='stop':
            if route is False:
                if self._seen(('off',scene)):
                    self._set('fish_idle' if self.journal.get('fishing_start_requested') else 'open')
            elif route is True and self._seen(('running',scene)):
                self._input('stop')
            return True
        if self.state=='fish_idle':
            if route is not False:return True
            if evidence:
                self.journal['fishing_verify_deadline']=min(self.deadline,self.clock()+30)
                self._save()
                self._set('fish_verify');return True
            if not idle or not fish_ready:return True
            read=getattr(self.view,'start_reading',None)
            data=(read(frame) if callable(read) else self.view.owner.fishing.measure(frame,remember=False))
            if (data is None or type(data.get('value')) is not int or not 1<=data['value']<=10000
                    or type(data.get('player')) is not int or not 1<=data['player']<=10000
                    or data.get('low') is not None and data['value']<data['low']
                    or data.get('high') is not None and data['value']>data['high']):
                self.hold='沿用当前标尺启动前，等待读清选值及角色等级；不会拖动标尺'
                return True
            guard=self.view.fishing_guard(frame)
            if guard is None:return True
            self.input_context['fishing_guard']=guard
            self.journal['fishing_start_reading']={k:data[k] for k in ('player','value','low','high')}
            self._save()
            if self._input('fish-start'):
                self.journal['fishing_verify_deadline']=min(self.deadline,self.clock()+30)
                self._save();self._set('fish_verify')
            return True
        if self.state=='fish_verify':
            if evidence and fish_ready and route is False:
                self.journal['fishing_start_verified']=True
                self._save();self._set('open')
                self.event('开局钓鱼：长按后连续新图确认真实填充或战斗，继续原航巡设置；未调整标尺')
            elif now>=self.journal.get('fishing_verify_deadline',self.deadline):
                self._end(False,'开局长按后未确认实际钓鱼，保留已用次数，不重复长按')
            return True
        if self.state=='open':
            if menu:self._set('menu')
            elif route is False and self._seen(('open',scene)):
                self._input('open')
            return True
        if self.state=='menu':
            if not menu:return True
            values=self.view.values(frame)
            if values is None:return True
            if not self._seen(('menu',json.dumps(values,sort_keys=True))):return True
            for name in TARGET_NAMES:
                if values['targets'][name] is not self.options['targets'][name]:
                    if self._input(name):self.key=None
                    return True
            for name in ('auto_sale','auto_upgrade'):
                if values[name] is not self.options[name]:
                    if self._input(name):self.key=None
                    return True
            self._set('tickets' if self.options['auto_sale'] and self.options['ticket_min']>0 else 'start')
        if self.state=='tickets':
            if menu:
                count=self.view.tickets(frame)
                if count is None:
                    if not self._seen(('tickets-unknown',)):return True
                    self.hold='售卖券数字未读清，本轮不购买，继续原航巡'
                    self._set('start');return True
                if not self._seen(('tickets',count)):return True
                self.journal['tickets']=count;self._save()
                if count<self.options['ticket_min']:
                    self.ticket_before=count
                    self.input_context['ticket_ink']=self.view.ticket_ink(frame)
                    self.input_context['ticket_count']=count
                    if self._input('tickets'):self._set('ticket_verify')
                    elif not self.allowed('tickets'):self._set('start')
                    return True
                self._set('start')
            else:return True
        if self.state=='ticket_verify':
            if not menu:return True
            count=self.view.tickets(frame)
            if count is not None and count>self.ticket_before and self._seen(('increase',count)):
                self.journal['tickets']=count;self._save()
                self.event(f'自动贩卖券：{self.ticket_before} → {count}，继续核对阈值')
                self._set('menu')
            return True
        if self.state=='start':
            if not menu:return True
            values=self.view.values(frame)
            if values!=self.view.wanted(self.options):return True
            if self._seen(('start',json.dumps(values,sort_keys=True))) and self._input('start'):
                self._set('verify')
            return True
        if self.state=='verify':
            # RANK can naturally advance once cruise has started. Only the world must match.
            if route is True and self._seen(('verified',scene)):
                self._end(True,'目标、自动贩卖和自动强化已核对，连续确认航巡运行')
            return True
        return self.busy

# Keep legacy task logic; delegate every record read/write to shared storage.
from drift.core.persistence import ReliablePath as Path

# Shared reliability layer; retain all other deployed methods.
def _save(self):
    from drift.core.persistence import atomic_write_text
    atomic_write_text(self.path, json.dumps(self.journal, ensure_ascii=False, indent=2))

CruiseSettings._save = _save



# Module-local compatibility for every authoritative record read/write.
from drift.core.persistence import ReliablePath as Path

# Bounded long-action budgets; preserve all other deployed methods.

CruiseSettings.TICKET_CONFIRM_SECONDS = 10

CruiseSettings.ROUTE_RESERVE_SECONDS = 20

def allowed(self, action):
    if not self.busy or self.deadline is None or self.clock()>=self.deadline:
        return False
    if (action=='tickets' and self.deadline-self.clock()
            < self.HOLD_MS/1000+self.TICKET_CONFIRM_SECONDS+self.ROUTE_RESERVE_SECONDS):
        return False
    limit=self.MAX_HOLDS if action=='tickets' else 1
    return self.journal.get('attempts',{}).get(action,0)<limit

CruiseSettings.allowed = allowed

def _input(self, action):
    point={'stop':OPEN_AT,'open':OPEN_AT,'start':START_AT,'close':CLOSE_AT,
           'auto_sale':SALE_AT,'auto_upgrade':UPGRADE_AT,'tickets':TICKET_AT,
           'fish-start':(360,1010)}.get(action)
    if action in TARGET_NAMES:point=(TARGET_X[action],TARGET_Y)
    if not self.allowed(action):
        if (action=='tickets' and self.deadline is not None and self.deadline-self.clock()
                < self.HOLD_MS/1000+self.TICKET_CONFIRM_SECONDS+self.ROUTE_RESERVE_SECONDS):
            self.hold='剩余时间不足以补票并核对票数，本轮先恢复航巡；下次按实际票数检查'
            self.event('巡航设置：'+self.hold)
        else:
            self.hold='已用完本轮 '+action+' 输入次数，等待画面确认'
        return False
    return self.click(action,*point,1500 if action=='fish-start' else self.HOLD_MS if action=='tickets' else 40)

CruiseSettings._input = _input

def step(self, frame):
    if frame is self.last_frame:return self.busy
    self.last_frame=frame
    now=self.clock()
    if self.busy and now>=self.deadline:
        self._end(False,'本轮设置窗口到期，保留输入记录，稍后重查')
        return False
    if self.state=='fish_verify' and now>=self.journal.get('fishing_verify_deadline',self.deadline):
        self._end(False,'开局长按后未确认实际钓鱼，保留已用次数，不重复长按')
        return False
    context=self.view.observe_setup(frame)
    late_observation=(self.state=='idle' and self.journal.get('fishing_blocked_scope'))
    if context is None and (self.state=='fish_verify' or late_observation):
        observe=getattr(self.view,'observe_fishing',None)
        observed_scene=self.scene or tuple(self.journal.get('scene') or ())
        if callable(observe) and observed_scene:context=observe(frame,observed_scene)
    if context is None:
        self.key=None;self.seen=0
        self._fishing_proof(None,now)
        if self.busy:
            self.hold='等待当前设置页的世界、阶段和RANK复核；未使用缓存数字'
            probe=getattr(self.view.owner,'menu_rank_probe',None)
            if probe and now-getattr(self,'unknown_logged_at',float('-inf'))>=30:
                self.unknown_logged_at=now
                self.event('巡航设置等待：'+self.hold+'；'+str(probe))
        return self.busy
    self.input_context=context
    scene=context['scene']
    if self.busy and scene!=self.scene and not (self.state=='verify' and scene[0]==self.scene[0]):
        self._end(False,'世界或RANK变化，结束本轮输入')
        return False
    menu=context['menu'];route=self.view.owner.sailing.running(frame)
    idle_probe=getattr(self.view,'fishing_idle',None)
    evidence_probe=getattr(self.view,'fishing_evidence',None)
    idle=idle_probe(frame,context) if callable(idle_probe) else None
    evidence=bool(evidence_probe(frame,context)) if callable(evidence_probe) else False
    scope=self.view.fishing_scope(frame,context) if callable(getattr(self.view,'fishing_scope',None)) else None
    if self.state in ('fish_idle','fish_verify') and scope!=self.journal.get('fishing_start_scope'):
        self._end(False,'开局钓鱼地点变化，保留次数并结束本轮输入')
        return False
    fish_ready=self._fishing_proof(('idle',idle,route) if idle else
        ('active',scope,route) if evidence else None,now)
    if evidence and fish_ready and self.journal.get('fishing_blocked_scope')==scope:
        self.journal.pop('fishing_blocked_scope')
        self.journal['fishing_late_observed_at']=now
        self._save()
        self.retry_at=now
        self.event('开局钓鱼：只读连续新图确认已有实际钓鱼，保留旧失败、截止和次数；后续仅核对原航巡设置')
    if self.busy and self.clock()>=self.deadline-5 and menu:
        # Close only while the original budget is still valid; never grant a new input window.
        if self._input('close'):
            self._end(False,'设置未能完成，已关闭菜单，保留原输入记录')
        return True
    if self.state=='idle':
        startup=bool(idle and fish_ready and scope!=self.journal.get('fishing_blocked_scope'))
        if idle and not fish_ready:return False
        if now<self.retry_at and not startup:return False
        # Only recheck an established running route for a new policy or a due ticket audit.
        applied=self.journal.get('applied_policy')==self.policy
        same_world=(self.journal.get('applied_scene') or [None])[0]==scene[0]
        due=(self.options['auto_sale'] and self.options['ticket_min']>0
             and now-self.journal.get('checked_at',0)>=self.interval)
        if applied and same_world and route is True and not due and not startup:return False
        if not menu and (route is None or not self.view.safe_main(frame)):return False
        if not self._seen(('begin',scene,menu,route)):return False
        self.scene=scene
        self.deadline=now+self.LIMIT
        self.journal.update(pending=True,policy=self.policy,scene=scene,deadline=self.deadline,
                            started_at=now,attempts={},fishing_start_requested=startup,
                            fishing_start_scope=scope if startup else None,fishing_start_verified=False)
        self._save()
        self._set('menu' if menu else 'stop' if route else 'fish_idle' if startup else 'open')
    if self.state=='stop':
        if route is False:
            if self._seen(('off',scene)):
                self._set('fish_idle' if self.journal.get('fishing_start_requested') else 'open')
        elif route is True and self._seen(('running',scene)):
            self._input('stop')
        return True
    if self.state=='fish_idle':
        if route is not False:return True
        if evidence:
            self.journal['fishing_verify_deadline']=min(self.deadline,self.clock()+30)
            self._save()
            self._set('fish_verify');return True
        if not idle or not fish_ready:return True
        read=getattr(self.view,'start_reading',None)
        data=(read(frame) if callable(read) else self.view.owner.fishing.measure(frame,remember=False))
        if (data is None or type(data.get('value')) is not int or not 1<=data['value']<=10000
                or type(data.get('player')) is not int or not 1<=data['player']<=10000
                or data.get('low') is not None and data['value']<data['low']
                or data.get('high') is not None and data['value']>data['high']):
            self.hold='沿用当前标尺启动前，等待读清选值及角色等级；不会拖动标尺'
            return True
        guard=self.view.fishing_guard(frame)
        if guard is None:return True
        self.input_context['fishing_guard']=guard
        self.journal['fishing_start_reading']={k:data[k] for k in ('player','value','low','high')}
        self._save()
        if self._input('fish-start'):
            self.journal['fishing_verify_deadline']=min(self.deadline,self.clock()+30)
            self._save();self._set('fish_verify')
        return True
    if self.state=='fish_verify':
        if evidence and fish_ready and route is False:
            self.journal['fishing_start_verified']=True
            self._save();self._set('open')
            self.event('开局钓鱼：长按后连续新图确认真实填充或战斗，继续原航巡设置；未调整标尺')
        elif now>=self.journal.get('fishing_verify_deadline',self.deadline):
            self._end(False,'开局长按后未确认实际钓鱼，保留已用次数，不重复长按')
        return True
    if self.state=='open':
        if menu:self._set('menu')
        elif route is False and self._seen(('open',scene)):
            self._input('open')
        return True
    if self.state=='menu':
        if not menu:return True
        values=self.view.values(frame)
        if values is None:return True
        if not self._seen(('menu',json.dumps(values,sort_keys=True))):return True
        for name in TARGET_NAMES:
            if values['targets'][name] is not self.options['targets'][name]:
                if self._input(name):self.key=None
                return True
        for name in ('auto_sale','auto_upgrade'):
            if values[name] is not self.options[name]:
                if self._input(name):self.key=None
                return True
        self._set('tickets' if self.options['auto_sale'] and self.options['ticket_min']>0 else 'start')
    if self.state=='tickets':
        if menu:
            count=self.view.tickets(frame)
            if count is None:
                if not self._seen(('tickets-unknown',)):return True
                self.hold='售卖券数字未读清，本轮不购买，继续原航巡'
                self._set('start');return True
            if not self._seen(('tickets',count)):return True
            self.journal['tickets']=count;self._save()
            if count<self.options['ticket_min']:
                self.ticket_before=count
                self.input_context['ticket_ink']=self.view.ticket_ink(frame)
                self.input_context['ticket_count']=count
                if self._input('tickets'):self._set('ticket_verify')
                elif not self.allowed('tickets'):self._set('start')
                return True
            self._set('start')
        else:return True
    if self.state=='ticket_verify':
        if not menu:return True
        count=self.view.tickets(frame)
        if count is not None and count>self.ticket_before and self._seen(('increase',count)):
            self.journal['tickets']=count;self._save()
            self.event(f'自动贩卖券：{self.ticket_before} → {count}，继续核对阈值')
            self._set('menu')
        return True
    if self.state=='start':
        if not menu:return True
        values=self.view.values(frame)
        if values!=self.view.wanted(self.options):return True
        if self._seen(('start',json.dumps(values,sort_keys=True))) and self._input('start'):
            self._set('verify')
        return True
    if self.state=='verify':
        # RANK can naturally advance once cruise has started. Only the world must match.
        if route is True and self._seen(('verified',scene)):
            self._end(True,'目标、自动贩卖和自动强化已核对，连续确认航巡运行')
        return True
    return self.busy

CruiseSettings.step = step
