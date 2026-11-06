"""Bounded continuous cookbook enhancement, all four slots required per level.

Stays in the current world. No crafting, food use, ingredient buying, portal,
fishing cancel, or manual changes to another controller's save records.
"""
from pathlib import Path
import base64
import hashlib
import json
import time
import uuid
import cv2
import numpy as np
from drift.core.journal import archive_bytes
from drift.perception.restaurant import RestaurantVisual, CATEGORIES, INGREDIENT_X
from drift.tasks.backpack_sale import BackpackSale
from drift.tasks.guild_daily import GuildVisual
from drift.tasks.stone_farm import StoneFarm
from drift.tasks.training_sailing import TARGETS

RESTAURANT_ID='driftinggirl_restaurant_upgrade'
# 720x1280 固定坐标；2026-10-29 现场取证：(233,45) 随上下文打开
# 航巡设置、料理强化面板或目的地地图，不能当作航巡开关。
CRUISE_TOGGLE=(360,880)
MENU_CLOSE=(600,363)
# 默认三行、每行八次小幅横拖，两次纵向换行，共 26 步；40 次额度兼容路径重放和边界改向。
# 2026-10-29 实测：480px/1500ms 的地图拖动会甩出惯性，一步实际位移超过一屏宽度，
# 相邻两帧之间留缝，餐厅图标正好落进缝里（扫 11~30 次一次未命中）；
# 240px/3000ms 慢拖跟手，两帧有重叠，从地图最左端 6 步即命中 0.9421。
SCAN_LIMIT=40
SCAN_ROWS=3
SCAN_STEPS_PER_ROW=8
SCAN_DRAG_MS=3000  # 500ms 不动、1500ms 甩过头，均实测；3000ms 才跟手。
SCAN_MAP_AREA=(20,175,700,1115)
# 地图区域 8 位灰度绝对差的均值；低于一个灰度级视为静止，忽略轻微截图噪声。
SCAN_STILL_MEAN=1.0
SCAN_DRAGS={'east':((480,650),(240,650)), 'west':((240,650),(480,650)),
            'south':((360,1000),(360,300)), 'north':((360,300),(360,1000))}
SCAN_NAMES={'east':'东','west':'西','south':'南','north':'北'}


def pack(image):return base64.b64encode(cv2.imencode('.png',image)[1]).decode()
def unpack(data):return cv2.imdecode(np.frombuffer(base64.b64decode(data),np.uint8),cv2.IMREAD_COLOR)


class RestaurantUpgrade:
    LIMIT=7200
    STALL_LIMIT=3
    # 开菜单、停航、关菜单及取原画面共用此上限，不随输入或识别失败续期。
    STOP_ORIGIN_LIMIT=90
    STOP_ENTRY_LIMIT=90
    MENU_CLOSE_LIMIT=3
    # 恢复原航巡的总上限：owned 阶段不能无限期挡住标尺校准等其它任务。
    RESTORE_LIMIT=60
    PHASES={'idle','stop','settings_open','settings','settings_close','map_open','map',
            'map_close','travel','open','atlas','align','list','detail','feed_wait',
            'upgrade_wait','detail_close','category_next','close','restore','restore_menu',
            'restore_verify','cleanup','blocked','menu_retire'}

    def __init__(self,root,reader,action,event,serial,interval=259200,*,targets=None,state_path=None,clock=time.time):
        self.root,self.reader,self.action,self.event,self.serial=Path(root),reader,action,event,serial
        self.targets=frozenset(targets or CATEGORIES)
        self.path=Path(state_path) if state_path else self.root/'restaurant_upgrade_state.json'
        self.clock,self.interval=clock,float(interval);self.enabled=lambda:True
        self.journal=dict(version=1,serial=serial,pending=False,phase='idle')
        if self.path.exists():
            saved=json.loads(self.path.read_text(encoding='utf-8-sig'))
            if (saved.get('version')!=1 or saved.get('serial')!=serial
                    or type(saved.get('pending')) is not bool or saved.get('phase') not in self.PHASES):
                raise RuntimeError('餐品强化记录无效，保留原记录')
            self.journal=saved
        self.phase=self.journal['phase'] if self.journal['pending'] else 'idle'
        self._check_startup=self.active
        self.visual=RestaurantVisual(root,reader);self.guild=GuildVisual(root)
        self.view=StoneFarm(root,reader,lambda _:False,event,state_path=self.root/'restaurant-probe-unused.json')
        self.bag=BackpackSale(root,lambda *a:False,event,reader,state_path=self.root/'restaurant-bag-unused.json')
        self.sailing,self.fishing=self.view.sailing,self.view.fishing
        self.key=self.since=self.last_seen=self.last_frame=None;self.seen=0
        self.next_at=0;self.done=None;self.hold=None
        self.popup_frame=None;self.popup_since=None;self.popup_frames=0
        self._observed_frame=self._menu_close_frame=self._stop_close_frame=None
        self._retire_probe_until=None

    @property
    def active(self):return self.journal.get('pending',False)

    def status(self):
        return dict(phase=self.phase,pending=self.active,hold=self.hold,
            upgraded=self.journal.get('upgraded',0),submitted=self.journal.get('submitted',0),
            checked=self._checked_unique(),checked_entries=len(self.journal.get('visited',[])),
            categories=self.journal.get('categories',[]),incomplete_categories=self.journal.get('incomplete_categories',[]),
            skipped=self.journal.get('skipped',[]),deadline=self.journal.get('deadline'),
            origin={k:v for k,v in self.journal.get('origin',{}).items() if k not in ('header','place')},interval_seconds=self.interval,
            failure=self.journal.get('failure'),last_failure=self.journal.get('last_failure'),
            retirement=self.journal.get('retirement'),cleanup_deadline=self.journal.get('cleanup_deadline'),
            attempts=self.journal.get('attempts',{}),order='left_to_right_complete_one',
            order_hold=getattr(self,'order_hold',None),
            current_dish={k:v for k,v in self.journal.get('card',{}).items() if k!='art'})

    def _checked_unique(self):
        from drift.tasks.restaurant_list import count
        return count(self)

    def _save(self):
        tmp=self.path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(self.journal,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        tmp.replace(self.path)

    def _set(self,phase,seconds=60):
        if self.phase==phase:return
        self.phase=phase;self.journal.update(phase=phase,phase_deadline=min(self.execution_deadline(),self.clock()+seconds))
        if phase=='stop' and 'origin_deadline' not in self.journal:
            self.journal['origin_deadline']=min(self.journal['phase_deadline'],self.clock()+self.STOP_ORIGIN_LIMIT)
        if (phase in ('restore','restore_menu','restore_verify')
                and 'restore_deadline' not in self.journal):
            # 恢复原航巡只保留一分钟：确认不了就退休，把运行期还给其它任务。
            self.journal['restore_deadline']=min(self.journal['phase_deadline'],
                self.clock()+self.RESTORE_LIMIT)
        self.key=None;self.seen=0;self._save();self.event('餐品强化：'+phase)

    def _confirmed(self,key,frame,seconds=1.2):
        now=self.clock()
        if frame is self.last_frame:return False
        if key is None or key!=self.key or self.last_seen is None or now-self.last_seen>12:
            self.key,self.since,self.seen=key,now,1 if key else 0
        else:self.seen+=1
        self.last_frame,self.last_seen=frame,now
        return key is not None and self.seen>=3 and now-self.since>=seconds

    @staticmethod
    def header(frame):
        return [pack(frame[y:Y,x:X]) for x,y,X,Y in ((71,9,136,59),(13,45,68,72))]

    @staticmethod
    def context_key(context):
        if not context:return None
        key=(context['parallel'],tuple(context['header']),context['route'],
             context.get('at_restaurant'))
        if 'place' in context:
            # Animated highlights/ellipsis change PNG bytes; only the fixed
            # dark location strokes establish identity, including on startup.
            strokes=unpack(context['place']).min(axis=2)<70
            if int(strokes.sum())<30:return None
            key+=(hashlib.sha256(strokes.tobytes()).hexdigest(),)
        return key

    def same_world(self,frame):
        origin=self.journal.get('origin',self.journal.get('startup',{}))
        if frame.shape!=(1280,720,3) or self.view._parallel(frame)!=origin.get('parallel'):return False
        for data,(x,y,X,Y) in zip(origin.get('header',[]),((71,9,136,59),(13,45,68,72))):
            reference=unpack(data);current=frame[y:Y,x:X]
            score=float(cv2.matchTemplate(current,reference,cv2.TM_CCOEFF_NORMED)[0,0])
            if not np.isfinite(score) or score<.97:return False
        return len(origin.get('header',[]))==2

    def at_restaurant(self,frame):
        # The destination label is fixed, while the right-hand entry icon
        # bounces and can briefly fail matching. Confirm arrival from the
        # location and stopped main screen; opening still requires a newly
        # recognized entry icon in validate().
        return bool(self.visual.find(frame,'location')
                    and self.fishing._main_visible(frame) and not self.bag.moving(frame))

    def ordinary_world(self,frame):
        # Resource/RAID encounters temporarily display the ordinary RANK.
        # Only a real normal-world arrival entry may start this navigation.
        return bool(self.at_restaurant(frame) or self.guild.point(frame,'g-icon')
                    or self.view._observatory(frame))

    def start_screen(self,frame,*,observe_battle=False):
        # World/RANK and the route can be confirmed while sailing; the place cannot.
        # Unknown cruise readings wait for evidence in stop; only a positively
        # running cruise may open the context menu. The map finds the restaurant.
        if (not self.fishing._main_visible(frame)
                or (self.view._battle(frame) and not observe_battle)):return None
        route=self.sailing.running(frame)
        parallel=self.view._parallel(frame)
        if parallel:
            s=self.view.screen(frame)
            # RANK never blocks the cookbook: the world can sit at any RANK
            # and the restaurant stays reachable from the map.
            if not s or s['completion'] or self.view.training.boss_mode(frame):return None
        elif not self.ordinary_world(frame):return None
        return dict(parallel=parallel,header=self.header(frame),route=route)

    def panel_start_screen(self,frame):
        # A finished failed run may leave the known restaurant container open.
        # Admit a NEW due round from that UI; never revive its expired journal.
        if (getattr(frame,'shape',None)!=(1280,720,3)
                or not self.visual.panel(frame) or self.visual.detail(frame)
                or not self.visual.find(frame,'location')
                or not self.view._off_icon(frame) or self.sailing.running(frame) is True
                or self.sailing.menu(frame) or self.bag.moving(frame)
                or self.view._battle(frame)):return None
        return dict(parallel=self.view._parallel(frame),header=self.header(frame),route=False)

    def origin_screen(self,frame,*,startup=None,observe_battle=False):
        # 移动中不取原画面：地点像素会变。停航巡后本函数会被逐帧重试，
        # 只有确实取不到时才由 stop 阶段超时归档退出。
        if self.bag.moving(frame):return None
        origin=self.start_screen(frame,observe_battle=observe_battle) if startup is None else startup
        if origin is None:return None
        return dict(**origin,at_restaurant=self.at_restaurant(frame),place=pack(frame[12:36,286:434]))

    def route_was_running(self):
        # Older pending journals stored the restoration intent in origin.route.
        return self.journal.get('resume_route',self.journal.get('origin',{}).get('route',False))

    def route_off(self,frame,run):
        # 航巡读数读不出来时，用本任务对所有非 stop 输入本来就要求的
        # 「停航主画面」证据（AUTO 标签 + 非激活面板 + 无激活箭头）判定没在跑。
        return run is False or (run is None and bool(self.view._off_icon(frame)))

    def _stop_resume(self):
        # 只有这一轮确实动过航巡才恢复；从已停画面误开设置面板时必须保持停着。
        if self.journal.get('resume_route') is False:return False
        if self.route_was_running():return True
        startup=self.journal.get('startup')
        return (bool(startup) and startup.get('route') is None
                and self.journal.get('attempts',{}).get('stop-toggle',0)>0)

    def _stop_origin_expired(self,frame=None):
        if self.phase!='stop':return False
        now=self.clock();close=self.journal.get('stop_context_close')
        if close and now>=close['deadline'] and not self._stop_context_closed(frame):
            return self.abandon('误开窗口关闭未确认，停止本轮输入',frame=frame)
        attempts=self.journal.get('attempts',{})
        if (attempts.get('stop') and not attempts.get('stop-toggle')
                and not attempts.get('stop-close') and not self.journal.get('stop_entry_confirmed')):
            # 新记录在预约输入时落盘；旧记录沿用原输入时间，重启不续期。
            last=self.journal.get('last_input',{})
            started=(last['at'] if last.get('kind')=='stop' else
                     self.journal.get('started_at',self.journal.get('origin_deadline',now)-self.STOP_ORIGIN_LIMIT))
            deadline=self.journal.get('stop_entry_deadline',started+self.STOP_ENTRY_LIMIT)
            if (now<=deadline and frame is not None and self.same_world(frame)
                    and (self.at_restaurant(frame) or self.sailing.menu(frame) or self.map_ready(frame)
                         # 与巡航设置相同：入口点后航巡已停且不在移动中，也算入口生效。
                         or (not self.bag.moving(frame)
                             and self.route_off(frame,self.sailing.running(frame))))):
                self.journal['stop_entry_confirmed']=True;self._save()
            elif now>=deadline:
                # 误开地图/面板时 _stop 会先关再继续；仍停在航巡主画面则放弃本轮，稍后重试。
                reason='移动中航巡入口无响应，本轮放弃稍后重试' if (frame is not None and self.bag.moving(frame)) else '入口(233,45)未打开航巡菜单，画面语义不符'
                return self.abandon(reason,frame=frame)
        deadline=self.journal.get('origin_deadline')
        if deadline is None:
            self.journal['origin_deadline']=min(self.journal['phase_deadline'],
                self.execution_deadline(),self.clock()+self.STOP_ORIGIN_LIMIT)
            self._save();return False
        if self.clock()<deadline:return False
        return self.abandon('停航后未能确认原画面，停止本轮输入',frame=frame)

    def _restore_expired(self,frame=None):
        if self.phase not in ('restore','restore_menu','restore_verify'):return False
        now=self.clock();deadline=self.journal.get('restore_deadline')
        if deadline is None:
            # 只在内存里记，不落盘：归档证据必须还是进入恢复前的原始记录。
            self.journal['restore_deadline']=min(self.execution_deadline(),now+self.RESTORE_LIMIT)
            return False
        if now<deadline:return False
        return self.abandon('航巡恢复确认超时',frame=frame,message=(
            '餐品强化：安全返回未完成，60 秒内没有确认实际恢复钓鱼，'
            '放弃本轮并保留记录；不再占住运行期，交给航巡守卫'))

    def observe_popup_return(self,frame,*,notice_gone):
        ready=(self.active and notice_gone and self.same_world(frame)
               and self.view._off_icon(frame)
               and (self.visual.detail(frame) or self.visual.atlas(frame)))
        if not ready:
            self.popup_since=None;self.popup_frames=0;self.popup_frame=None
            return False
        if frame is self.popup_frame:return False
        self.popup_frame=frame
        if self.popup_since is None:self.popup_since=self.clock()
        self.popup_frames+=1
        return self.popup_frames>=3 and self.clock()-self.popup_since>=3

    def settings(self,frame):
        if not self.sailing.menu(frame):return None
        values={name:self.sailing.selected(frame,name) for name in TARGETS}
        values.update(sale=self.sailing.sale_selected(frame),upgrade=self.sailing.upgrade_selected(frame))
        if any(v is None for v in values.values()):return None
        values['priority']=hashlib.sha256((frame[515:579,250:555].min(axis=2)>210).tobytes()).hexdigest()
        return values

    def map_ready(self,frame):
        # stop 尚未取 origin 时，只借 startup 的世界身份识别并关闭误开的地图。
        origin=self.journal.get('origin') or (self.journal.get('startup') if self.phase=='stop' else None)
        if not origin:return False
        self.bag.parallel=origin['parallel']
        return bool(self.same_world(frame) and self.view._off_icon(frame) and self.bag._scan_map(frame,allow_moving=True))

    def target(self,frame):
        if not self.map_ready(frame):return None
        if self.journal['destination']=='restaurant':return self.visual.point(frame,'map-restaurant')
        if not self.journal.get('origin_map'):return None
        self.bag.origin=unpack(self.journal['origin_map']);return self.bag._origin_target(frame)

    def allowed(self):
        # runtime 的输入门禁会先查 allowed()。过期任务仅给红 X 一个独立、
        # 不可续期的短窗口；validate/reserve 在此阶段拒绝所有其它输入。
        if self.phase=='menu_retire':
            return self.active and self.clock()<self.journal['menu_close']['deadline']
        if self.phase=='stop' and self.clock()>=self.journal.get('origin_deadline',float('inf')):
            return False
        return (self.active and self.phase!='blocked' and self.clock()<self.execution_deadline()
                and self.clock()<self.journal['phase_deadline'])

    def execution_deadline(self):
        return self.journal.get('cleanup_deadline',self.journal.get('deadline',0))

    def _begin_cleanup(self,reason):
        if (self._menu_visible(getattr(self,'_observed_frame',None)) or self.journal.get('cleanup_deadline')
                or self.phase in ('restore','restore_menu','restore_verify','blocked','menu_retire')):
            self.abandon(reason)
            return False
        raw=self.path.read_bytes()
        archive_bytes(self.root/'logs/restaurant-history'/(hashlib.sha256(raw).hexdigest()+'.json'),raw,'餐品强化')
        self.journal.update(failure=dict(reason=reason,at=self.clock(),phase=self.phase,
            original_deadline=self.journal.get('deadline'),phase_deadline=self.journal.get('phase_deadline')),
            cleanup_deadline=self.clock()+900)
        self._set('cleanup',120)
        self.event('餐品强化：'+reason+'；本轮停止提交食材，只关闭已识别窗口并恢复原活动')
        return True

    def retire_expired_on_start(self,frame=None):
        # Only inspect saved deadlines, never the file's modification time. A
        # live cleanup window may resume; an expired one must not be renewed.
        self._check_startup=False
        if self.phase=='menu_retire':
            self._confirm_menu_retirement(frame)
            return True
        if not self.active or self.allowed():return False
        self.abandon('启动时发现上次执行已过期',
                     message='餐品强化：上次执行已过期，已归档并重新排队',frame=frame)
        return True

    def _menu_visible(self,frame):
        return (getattr(frame,'shape',None)==(1280,720,3)
                and self.sailing.menu(frame) is True)

    def _start_menu_retirement(self,frame,message,deadline=None):
        self.phase='menu_retire'
        self.journal.update(pending=True,phase=self.phase,menu_close=dict(
            deadline=self.clock()+self.MENU_CLOSE_LIMIT if deadline is None else deadline,
            result='reserved',message=message))
        self._menu_close_frame=frame
        self._save()  # 先持久化一次性关闭意图；重启只确认，不重放点击。
        try:
            issued=self._act('menu-leak-close',MENU_CLOSE,frame)
        except RuntimeError as exc:
            self.journal['menu_close']['error']=str(exc);issued=False
        self.journal['menu_close']['result']='issued_only' if issued else 'rejected'
        if issued:self._save()
        else:self._finish(False,message=message)

    def _confirm_menu_retirement(self,frame):
        close=self.journal['menu_close']
        if (frame is not None and frame is not self._menu_close_frame
                and not self._menu_visible(frame) and self.fishing._main_visible(frame)):
            close['result']='closed'
        elif self.clock()<close['deadline']:return
        else:close['result']='unconfirmed'
        self._finish(False,message=close['message'])

    def abandon(self,reason='本轮中止',*,message=None,frame=None):
        """Archive before retiring; failed archival must leave the record intact."""
        if self.phase=='menu_retire':return True
        # Recognition may fail before the first on-disk journal is created.
        raw=(self.path.read_bytes() if self.path.exists() else
             (json.dumps(self.journal,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
        digest=hashlib.sha256(raw).hexdigest()
        archive=self.root/'logs/restaurant-history'/(digest+'.json')
        archive_bytes(archive,raw,'餐品强化')
        failure=self.journal.pop('failure',None) or dict(reason=reason,at=self.clock(),
            phase=self.phase,original_deadline=self.journal.get('deadline'),
            phase_deadline=self.journal.get('phase_deadline'))
        self.journal.update(last_failure=failure,retirement=dict(reason=reason,at=self.clock(),
            previous_phase=self.phase,sha256=digest,archive=str(archive)))
        self.key=self.since=self.last_seen=self.last_frame=None;self.seen=0
        self.next_at=0;self.hold=None
        message=message or f'餐品强化：安全返回未完成，放弃本轮并保留记录；食材提交已停止，拒绝未知画面输入（{reason}）'
        frame=getattr(self,'_observed_frame',None) if frame is None else frame
        if self._menu_visible(frame):
            self._start_menu_retirement(frame,message)
        else:
            # 工厂启动检查没有截图；第一张新帧只补一次识别，且仍受短期限约束。
            if frame is None:self._retire_probe_until=self.clock()+self.MENU_CLOSE_LIMIT
            self._finish(False,message=message)
        return True

    @staticmethod
    def input_limit(kind):
        if kind in ('stop','stop-toggle','stop-close','menu-leak-close'):return 1
        if kind.startswith('feed:'):return 24
        if kind.startswith('upgrade:'):return 1
        if kind.startswith('scan:'):return SCAN_LIMIT
        if kind.startswith(('scroll:','align:')):return 5
        if kind.startswith('dish:'):return 2
        return 3

    def reserve(self,action):
        if not self.allowed():return False
        kind=action['kind'];attempts=self.journal['attempts']
        if self.phase=='menu_retire' and kind!='menu-leak-close':return False
        if attempts.get(kind,0)>=self.input_limit(kind):return False
        attempts[kind]=attempts.get(kind,0)+1
        if kind=='stop':
            self.journal.update(resume_route=True,stop_entry_deadline=min(
                self.journal.get('origin_deadline',self.clock()+self.STOP_ENTRY_LIMIT),
                self.clock()+self.STOP_ENTRY_LIMIT))
        self.journal['last_input']=dict(kind=kind,state='reserved',at=self.clock())
        self._save();return True

    def input_issued(self,action,receipt):
        self.journal['last_input'].update(state='issued_only',input_id=receipt['input_id'])
        self._save()

    def _act(self,kind,point,frame,**extra):
        if point is None:return False
        # Check the existing reservation budget only when another input is
        # needed. The last issued attempt may still be verified successfully.
        if ((self.journal.get('cleanup_deadline') or kind in ('restore-open','restore-start'))
                and self.journal['attempts'].get(kind,0)>=self.input_limit(kind)):
            self.abandon(kind+' 输入次数已达上限',frame=frame)
            return False
        request=dict(kind=kind,x=int(point[0]),y=int(point[1]),phase=self.phase,**extra)
        if not self.action(request):return False
        self.key=None;self.seen=0;self.next_at=self.clock()+.8;return True

    def validate(self,a,frame):
        if not self.allowed() or a['phase']!=self.phase:return False
        k=a['kind'];p=(a['x'],a['y'])
        if self.journal['attempts'].get(k,0)>=self.input_limit(k):return False
        if self.phase=='menu_retire':
            return k=='menu-leak-close' and p==MENU_CLOSE and self._menu_visible(frame)
        if not self.same_world(frame):return False
        if self.journal.get('cleanup_deadline'):
            permitted=('cleanup-detail-close','cleanup-panel-close','cleanup-map-close',
                'cleanup-settings-close','restore-open','restore-start',
                'map-open:origin','map-select:origin','map-close:origin','scan:origin')
            if k not in permitted:return False
        if k=='stop':return (self.phase=='stop' and self.enabled()
                            and self.fishing._main_visible(frame) and not self.view._battle(frame)
                            and not self.view.training.boss_mode(frame)
                            and (self.view._parallel(frame) or self.ordinary_world(frame))
                            and not self.sailing.menu(frame) and not self.at_restaurant(frame)
                            and self.sailing.running(frame) is True and p==(233,45))
        if k in ('stop-toggle','stop-close'):
            attempts=self.journal['attempts']
            ready=(self.phase=='stop' and not self.view._battle(frame)
                   and self.sailing.menu(frame) and attempts.get('stop',0)>0)
            if k=='stop-toggle':
                return (ready and self.route_was_running() and not attempts.get('stop-close')
                        and self.sailing.running(frame) is not False and p==CRUISE_TOGGLE)
            return (ready and (attempts.get('stop-toggle') or self.sailing.running(frame) is False)
                    and p==MENU_CLOSE)
        if self.phase=='stop':
            close=self.journal.get('stop_context_close')
            if (not close or k!=close['kind'] or self.clock()>=close['deadline']
                    or self.view._battle(frame)):return False
            if k=='cleanup-panel-close':
                return (self.visual.panel(frame) and not self.visual.detail(frame)
                        and p==self.visual.point(frame,'panel-close'))
            if k=='cleanup-map-close':return self.map_ready(frame) and p==(668,1187)
            return False
        if not self.view._off_icon(frame):return False
        if k=='cleanup-detail-close':return self.visual.detail(frame) and p==self.visual.point(frame,'detail-close')
        if k=='cleanup-panel-close':return self.visual.panel(frame) and not self.visual.detail(frame) and p==self.visual.point(frame,'panel-close')
        if k=='cleanup-map-close':return self.map_ready(frame) and p==(668,1187)
        if k=='cleanup-settings-close':return self.sailing.menu(frame) and self.settings(frame)==self.journal.get('settings') and p==MENU_CLOSE
        if k in ('settings-open','restore-open'):
            # 航巡读数可能恰好读不出来；上面的 _off_icon 已证明是停航主画面。
            return p==(233,45)
        if k in ('settings-close','restore-start'):
            return (self.sailing.menu(frame) and self.settings(frame)==self.journal.get('settings')
                    and p==(MENU_CLOSE if k=='settings-close' else (360,873)))
        if k.startswith('map-open:'):
            return self.fishing._main_visible(frame) and not self.view._battle(frame) and p==(350,30)
        if k.startswith('scan:'):return self.map_ready(frame) and self.target(frame) is None
        if k.startswith('map-select:'):return self.map_ready(frame) and self.target(frame)==p
        if k.startswith('map-close:'):return self.map_ready(frame) and p==(668,1187)
        if k=='restaurant-open':return self.at_restaurant(frame) and p==self.visual.point(frame,'entry')
        if k=='atlas-tab':return self.visual.panel(frame) and not self.visual.detail(frame) and p==self.visual.point(frame,'atlas-tab')
        if k.startswith('detail-close:'):return self.visual.detail(frame) and p==self.visual.point(frame,'detail-close')
        if k=='panel-close':return self.visual.panel(frame) and not self.visual.detail(frame) and p==self.visual.point(frame,'panel-close')
        if k.startswith(('scroll:','align:','next:','dish:')):
            if self.visual.category(frame)!=self.journal.get('category'):return False
            if k.startswith(('scroll:','align:','dish:')) and 'list_key' in a:
                from drift.tasks.restaurant_list import fresh_list
                if not fresh_list(self,a,frame):return False
            if k.startswith('next:'):return p==self.visual.point(frame,'next')
            if k.startswith('dish:'):
                art=unpack(a['art'])
                card=next((c for c in self._ordered_cards(frame)
                           if not self._exhausted_card(c) and not self._visited(c) and c['level']!=20),None)
                return bool(self.enabled() and card is not None and card['level'] is not None
                    and card['point']==p and card['level']==a['level']<20
                    and self.visual.art_matches(card['art'],art))
            return not self.visual.detail(frame)
        if k.startswith(('feed:','upgrade:')):
            v=self.visual.detail_values(frame)
            if (not self.enabled() or not v or v!=a['before']
                    or v['identity'] in self.journal.get('exhausted_dishes',[])):return False
            if k.startswith('feed:'):
                i=a['ingredient']
                return (i not in self.journal.get('blocked_ingredients',[])
                        and self.visual.ingredient_ready(frame,i,v)
                        and p==(INGREDIENT_X[i],647) and a.get('hold_ms')==3000)
            return (not self._upgrade_attempted(v['identity'],v['level'])
                    and self.visual.upgrade_ready(frame,v) and p==(604,752))
        return False

    def _nav(self,destination):
        self.journal.update(destination=destination,scans=0)
        self.journal.pop('scan_state',None);self.journal.pop('scan_pending',None)
        self._scan_frames=None;self._scan_before=None
        self._set('map_open',120)

    @staticmethod
    def _scan_image(frame):
        x,y,X,Y=SCAN_MAP_AREA
        return cv2.cvtColor(frame[y:Y,x:X],cv2.COLOR_RGB2GRAY)

    def observe_map_scan(self,before,after):
        # 输入回调只交回实际前后帧；回调返回后才识别/点击，避免在设备锁内递归输入。
        self._scan_frames=(before,after)

    def _scan_state(self):
        if 'scan_state' not in self.journal:
            hint=self.journal.get('scan_hint',[]) if self.journal['destination']=='restaurant' else []
            hint=[list(s) for s in hint if isinstance(s,(list,tuple)) and len(s)==2
                  and s[0]=='drag' and isinstance(s[1],str) and s[1] in SCAN_DRAGS][:SCAN_LIMIT] if isinstance(hint,list) else []
            # 旧记录没有方向历史，按计数续接网格，但不把不完整路径覆盖为成功 hint。
            n=self.journal['scans'];row,step=divmod(n,SCAN_STEPS_PER_ROW+1)
            self.journal['scan_state']=dict(row=row,step=step,hint=hint,hint_index=0,
                path=[],complete=n==0,horizontal_edges=[],vertical_edges=[],last='地图初始画面')
            self._save()
            if hint:self.event(f'餐品强化：按上次成功的拖动路径重放（{len(hint)} 步）')
        return self.journal['scan_state']

    def _select_map_target(self,point,frame):
        if self._act('map-select:'+self.journal['destination'],point,frame):
            state=self.journal.get('scan_state',{})
            if self.journal['destination']=='restaurant':
                if state.get('complete',self.journal['scans']==0):
                    self.journal['scan_hint']=[s[:] for s in state.get('path',[])]
                self.event('餐品强化：地图上找到餐厅图标（'+state.get('last','地图初始画面')+'）')
            self._set('map_close')

    def _scan_result(self,frame):
        pending=self.journal.get('scan_pending')
        if not pending:return False
        # 没有回传截图时，等待第一张有效新帧；绝不拿拖动前帧冒充拖动后帧。
        if frame is getattr(self,'_scan_before',None) or not self.map_ready(frame):return True
        point=self.target(frame)
        self.journal.pop('scan_pending')
        if point is not None:
            self._save()
            # target 已匹配拖后帧，_act 的运行时门禁还会再匹配一次新帧的同一坐标。
            self._select_map_target(point,frame)
            return True
        state=self._scan_state();direction=pending['direction']
        horizontal=direction in ('east','west')
        edges=state['horizontal_edges' if horizontal else 'vertical_edges']
        before=unpack(pending['before'])[:,:,0]
        still=float(cv2.absdiff(before,self._scan_image(frame)).mean())<SCAN_STILL_MEAN
        if still:
            if direction not in edges:edges.append(direction)
            opposite={'east':'west','west':'east','south':'north','north':'south'}[direction]
            following=('改走'+SCAN_NAMES[opposite]+'侧' if opposite not in edges else
                       '尝试纵向换行' if horizontal else '该列纵向已到头')
            self.event(f'餐品强化：地图向{SCAN_NAMES[direction]}到头，{following}')
        else:
            # 横移后是新列、纵移后是新行，对应轴的旧边界证据不能沿用。
            state['vertical_edges' if horizontal else 'horizontal_edges']=[]
            if not horizontal and pending['source']=='grid':
                state['row']+=1;state['step']=0
                self.event(f'餐品强化：地图搜索换到第 {state["row"]+1} 行')
        self._save()
        if len(state['horizontal_edges'])==2 and len(state['vertical_edges'])==2:
            self._begin_cleanup('餐馆地图搜索横纵方向均已到头')
            return True
        return False

    def _scan_map(self,frame):
        if max(self.journal['scans'],self.journal['attempts'].get('scan:'+self.journal['destination'],0))>=SCAN_LIMIT:
            self._begin_cleanup('餐馆地图搜索已达上限');return
        state=self._scan_state();source='grid';direction=None
        # hint 与网格共用方向边界和输入额度，已证实不动的方向直接跳过。
        while state['hint_index']<len(state['hint']):
            candidate=state['hint'][state['hint_index']][1]
            edges=state['horizontal_edges' if candidate in ('east','west') else 'vertical_edges']
            if candidate not in edges:
                direction=candidate;source='hint';break
            state['hint_index']+=1
        if direction is None:
            if state['row']<SCAN_ROWS and state['step']<SCAN_STEPS_PER_ROW:
                east=state['row']%2==0
                direction=next((d for d in (('east','west') if east else ('west','east'))
                                if d not in state['horizontal_edges']),None)
            if direction is None and state['row']<SCAN_ROWS-1:
                direction=next((d for d in ('south','north') if d not in state['vertical_edges']),None)
            if direction is None:
                self._begin_cleanup('餐馆地图搜索网格已完成或方向到头，未找到目标');return
        point,end=SCAN_DRAGS[direction]
        self._scan_frames=None;self._scan_before=frame
        if not self._act('scan:'+self.journal['destination'],point,frame,swipe=[*end,SCAN_DRAG_MS]):return
        self.journal['scans']+=1;state['path'].append(['drag',direction])
        if source=='hint':
            state['hint_index']+=1
            label=f'路径重放第 {state["hint_index"]} 步'
        elif direction in ('east','west'):
            state['step']+=1
            label=f'第 {state["row"]+1} 行 第 {state["step"]} 步'
        else:label=f'换到第 {state["row"]+2} 行'
        state['last']=label
        self.event(f'餐品强化：地图搜索 {label}（向{SCAN_NAMES[direction]}）')
        frames=self._scan_frames
        before=frames[0] if frames is not None else frame
        self.journal['scan_pending']=dict(direction=direction,source=source,before=pack(self._scan_image(before)))
        self._save()
        if frames is not None:self._scan_result(frames[1])
        self._scan_frames=None

    def _finish(self,success,*,message=None):
        self.done=bool(success and not self.journal.get('failure') and not self.journal.get('incomplete_categories'));self.phase='idle'
        self.journal.update(pending=False,phase='idle',finished_at=self.clock(),success=self.done)
        label='本轮完成' if self.done else '本轮未完成，保留失败记录'
        self._save();self.event(message or f'餐品强化：{label}，已验证升级 '+str(self.journal.get('upgraded',0))+' 级')

    def _skip(self,reason,*,exhausted=False):
        self.journal['skipped'].append(dict(identity=self.journal.get('dish_identity'),
            category=self.journal.get('category'),reason=reason,exhausted=exhausted))
        if exhausted:
            identity=self.journal.get('dish_identity')
            dishes=self.journal.setdefault('exhausted_dishes',[])
            if identity and identity not in dishes:
                dishes.append(identity)
                count=self.journal.get('dish_upgrades',{}).get(identity,0)
                self.event(f'餐品强化：「{identity}」本轮材料用完（已升 {count} 级）')
        self._set('detail_close')

    def _stall(self):
        # Count completed, confirmed no-progress cycles, never individual frames
        # or issued-only receipts. Retrying needs three fresh matching frames.
        self.journal['stall']=self.journal.get('stall',0)+1
        self.key=None;self.seen=0
        if self.journal['stall']>=self.STALL_LIMIT:
            self._skip(f'连续 {self.STALL_LIMIT} 次详情循环无进展，停止本餐品',exhausted=True)
        else:
            self._set('detail',300);self._save()

    def _category_exhausted(self):
        from drift.tasks.restaurant_list import exhausted
        return exhausted(self)

    def _category_stats(self):
        return self.journal.setdefault('category_stats',{}).setdefault(
            str(self.journal.get('category')),dict(upgraded=[],submitted=0))

    def _complete_category(self):
        category=self.journal['category']
        if not self._category_exhausted():
            self.order_hold='尚未覆盖九种餐品，继续有界扫描'
            self._set('list',1200)
            return
        if category not in self.journal['categories']:
            self.journal['categories'].append(category)
            stats=self._category_stats()
            self.event(f'餐品强化：「{CATEGORIES[category]}」本轮结束，升级 {len(stats["upgraded"])} 个餐品，提交 {stats["submitted"]} 份食材')
        self._set('category_next')

    def _visited(self,card):
        return any(self.visual.art_matches(card['art'],unpack(v['art']))
                   for v in self.journal.get('visited',[]) if v['category']==self.journal['category'])

    def _ordered_cards(self,frame):
        return sorted(self.visual.cards(frame,strict_order=True),key=lambda card:card['point'][0])


    def _exhausted_card(self,card):
        exhausted=self.journal.get('exhausted_dishes',[])
        if card.get('identity') in exhausted:return True
        # Atlas cards expose art, while stable name identities come from detail.
        # Persist that association without changing perception or templates.
        return any(v.get('identity') in exhausted
                   and self.visual.art_matches(card['art'],unpack(v['art']))
                   for v in self.journal.get('visited',[]) if v['category']==self.journal['category'])

    def _upgrade_attempted(self,identity,level):
        # The name is stable across artwork tints; each level has its own budget,
        # including reserved inputs restored from a pending journal.
        return self.journal.get('attempts',{}).get(f'upgrade:{identity}:{level}',0)>0

    def _remember_card(self):
        if self.journal.get('card'):
            self.journal['visited'].append(dict(category=self.journal['category'],**self.journal.pop('card')))
            self._save()

    def _open_at_restaurant(self,frame):
        # 餐厅上下文已成立，无需停航证据；先保留旧 origin.route 的恢复意图。
        if self.route_was_running():self.journal['resume_route']=True
        if not self.journal.get('origin'):
            self.journal['origin']=dict(parallel=self.view._parallel(frame),header=self.header(frame),
                route=False,at_restaurant=True,place=pack(frame[12:36,286:434]))
        self.journal.pop('stop_context_close',None)
        self._set('open')

    def _stop_context_closed(self,frame):
        return (frame is not None and frame is not self._stop_close_frame and self.same_world(frame)
                and self.fishing._main_visible(frame) and not self.sailing.menu(frame)
                and not self.visual.panel(frame) and not self.map_ready(frame))

    def _close_stop_context(self,kind,point,frame,*,restaurant=False):
        # 与 menu-leak-close 一样，先记一次性关闭意图，再用新帧确认；重启不重放。
        close=dict(kind=kind,deadline=self.clock()+self.MENU_CLOSE_LIMIT,
                   restaurant=restaurant,result='reserved')
        self.journal['stop_context_close']=close;self._stop_close_frame=frame;self._save()
        try:
            issued=self._act(kind,point,frame)
        except RuntimeError as exc:
            close['error']=str(exc);issued=False
        close['result']='issued_only' if issued else 'rejected';self._save()
        if not issued:self.abandon('误开窗口关闭输入未获确认，停止本轮输入',frame=frame)

    def _stop(self,frame):
        if self.at_restaurant(frame):
            self._open_at_restaurant(frame);return
        close=self.journal.get('stop_context_close')
        if close:
            if self._stop_context_closed(frame):
                if close['restaurant']:self._open_at_restaurant(frame)
                else:
                    # Closing an unexpected page does not undo a spent contact.
                    self.journal.pop('stop_context_close',None);self._save()
            return
        run=self.sailing.running(frame);menu=self.sailing.menu(frame)
        attempts=self.journal['attempts']
        opened=attempts.get('stop',0);toggled=attempts.get('stop-toggle',0)
        closed=attempts.get('stop-close',0)
        if menu:
            self._confirmed(None,frame)
            if not opened or not self.route_was_running():
                self.abandon('停航阶段出现未由本轮打开的航巡菜单',frame=frame)
            elif not toggled and not closed and run is not False:
                self._act('stop-toggle',CRUISE_TOGGLE,frame)
            elif not closed:
                self._act('stop-close',MENU_CLOSE,frame)
            return
        if not self.fishing._main_visible(frame) or self.map_ready(frame):
            self._confirmed(None,frame)
            if self.visual.panel(frame) and not self.visual.detail(frame):
                self._close_stop_context('cleanup-panel-close',self.visual.point(frame,'panel-close'),frame)
            elif self.map_ready(frame):
                self._close_stop_context('cleanup-map-close',(668,1187),frame,
                                         restaurant=bool(opened and not toggled and not closed))
            return
        if opened and not toggled and not closed:
            # 入口点后：若航巡已停（与巡航设置同一点位的常见结果），不要死等菜单。
            # 菜单/地图/餐馆语义仍由 _stop_origin_expired 的入口确认负责。
            if not (self.route_off(frame,run) and not self.bag.moving(frame)):
                self._confirmed(None,frame);return
        if not opened and run is True:
            # 与巡航设置一致：先连续确认仍在跑，再点入口；移动中也可点，但需稳定帧。
            if not self._confirmed(('stop-ready',run,bool(self.bag.moving(frame))),frame):
                return
            self._act('stop',(233,45),frame);return
        # 按钮若自行收起菜单，不在主画面盲点红 X；仍需完整静止证据。
        origin=self.origin_screen(frame,observe_battle=True) if self.route_off(frame,run) else None
        key=self.context_key(origin)
        if self._confirmed(key,frame,3) and origin and not self.view._battle(frame):
            self.journal['resume_route']=self._stop_resume()
            self.journal['origin']=origin
            if self.route_was_running():self._set('settings_open')
            elif origin['at_restaurant']:self._set('open')
            else:self._nav('restaurant')

    def step(self,frame,*,due=True):
        self.done=None
        self._observed_frame=frame
        if self.phase=='menu_retire':
            self._check_startup=False;self._confirm_menu_retirement(frame)
            return self.active
        if getattr(self,'_retire_probe_until',None) is not None:
            deadline=self._retire_probe_until;self._retire_probe_until=None
            if self.clock()<deadline and self._menu_visible(frame):
                self._start_menu_retirement(frame,'餐品强化：遗留菜单已处理，本轮仍按失败归档',deadline)
                return self.active
        if getattr(self,'_check_startup',False) and self.retire_expired_on_start(frame):return self.active
        if not self.active:
            if not due or not self.enabled():return False
            # Read-only identity may span ordinary fishing. Input validation
            # still requires its own new, non-battle frame. Never take a BOSS.
            panel_start=self.panel_start_screen(frame)
            startup=panel_start or self.start_screen(frame,observe_battle=True)
            origin=(dict(**panel_start,at_restaurant=True,place=pack(frame[12:36,286:434]))
                    if panel_start else self.origin_screen(frame,startup=startup)
                    if startup and startup['route'] is False else None)
            # 航巡读数未知也先接管：origin 要等停航巡后画面稳定才能取。
            context=startup if startup and startup['route'] is not False else origin
            key=self.context_key(context)
            if not self._confirmed(key,frame,3) or self.view._battle(frame):return False
            if self.path.exists():
                raw=self.path.read_bytes();archive_bytes(self.root/'logs/restaurant-history'/(hashlib.sha256(raw).hexdigest()+'.json'),raw,'餐品强化')
            scan_hint=self.journal.get('scan_hint')
            self.journal=dict(version=1,serial=self.serial,pending=True,phase='idle',run_id=uuid.uuid4().hex,
                started_at=self.clock(),deadline=self.clock()+self.LIMIT,startup=startup,
                resume_route=startup['route'],
                attempts={},visited=[],categories=[],skipped=[],upgraded=0,submitted=0,
                exhausted_dishes=[],dish_upgrades={},category_stats={},stall=0)
            if scan_hint is not None:self.journal['scan_hint']=scan_hint
            if origin is not None:self.journal['origin']=origin
            self._set('open' if panel_start else 'stop',self.STOP_ORIGIN_LIMIT);return True
        if self._stop_origin_expired(frame) or self._restore_expired(frame):return self.active
        if not self.allowed():
            reason=('执行窗口到期' if self.clock()>=self.execution_deadline()
                    else f'{self.phase} 阶段确认超时')
            if not self._begin_cleanup(reason):return self.active
        if not self.enabled() and not self.journal.get('cleanup_deadline'):
            if not self._begin_cleanup('用户关闭功能'):return self.active
        if self.clock()<self.next_at:return True
        if not self.same_world(frame):
            self._confirmed(None,frame)
            self.hold='等待原世界与RANK画面，拒绝其他世界输入';return True
        self.hold=None
        phase=self.phase
        if phase=='cleanup':
            if self.visual.detail(frame):self._act('cleanup-detail-close',self.visual.point(frame,'detail-close'),frame)
            elif self.visual.panel(frame):self._act('cleanup-panel-close',self.visual.point(frame,'panel-close'),frame)
            elif self.map_ready(frame):self._act('cleanup-map-close',(668,1187),frame)
            elif self.sailing.menu(frame):
                value=self.settings(frame)
                if value and ('settings' not in self.journal or value==self.journal['settings']):
                    if 'settings' not in self.journal:
                        self.journal['settings']=value;self._save()
                    self._act('cleanup-settings-close',MENU_CLOSE,frame)
            elif self.fishing._main_visible(frame):
                origin=self.journal.get('origin',{});run=self.sailing.running(frame)
                if self.route_was_running():
                    self._set('restore',180)
                elif not self.bag.moving(frame) and self.route_off(frame,run):
                    place=origin.get('place')
                    if place is None:
                        # 线上事故：origin 为空时这里会 KeyError 打崩 runner。
                        self.abandon('缺少原画面记录，无法确认是否回到原地点');return False
                    same_place=float(cv2.matchTemplate(frame[12:36,286:434],unpack(place),cv2.TM_CCOEFF_NORMED)[0,0])>=.97
                    if same_place:self._finish(False)
                    elif self.journal.get('origin_map'):self._nav('origin')
        elif phase=='stop':
            self._stop(frame)
        elif phase=='settings_open':
            if self.sailing.menu(frame):self._set('settings')
            else:self._act('settings-open',(233,45),frame)
        elif phase=='settings':
            value=self.settings(frame)
            if self._confirmed(str(value) if value else None,frame):
                self.journal['settings']=value;self._set('settings_close')
        elif phase=='settings_close':
            if self.sailing.menu(frame):self._act('settings-close',MENU_CLOSE,frame)
            elif self.fishing._main_visible(frame):
                if 'origin' not in self.journal:self._set('stop')
                elif self.journal['origin']['at_restaurant']:self._set('open')
                else:self._nav('restaurant')
        elif phase=='map_open':
            if self.map_ready(frame):self._set('map',120)
            else:self._act('map-open:'+self.journal['destination'],(350,30),frame)
        elif phase=='map':
            if not self.map_ready(frame):return True
            if not self.journal.get('origin_map') and not self.route_was_running():
                player=(self.bag._find(frame,'map-player-current',(20,175,700,1115),.94)
                        or self.bag._find(frame,'map-player',(20,175,700,1115),.94))
                if not player or not(90<=player[0]<=630 and 230<=player[1]<=1090):return True
                x,y=player;self.journal['origin_map']=pack(frame[y-52:y-20,x-70:x+70]);self._save()
            if self._scan_result(frame):return self.active
            point=self.target(frame)
            if point and self._confirmed(('target',point),frame):
                self._select_map_target(point,frame)
            elif point is None and self._confirmed('search',frame):
                self._scan_map(frame)
        elif phase=='map_close':
            if self.map_ready(frame):self._act('map-close:'+self.journal['destination'],(668,1187),frame)
            elif self.fishing._main_visible(frame):self._set('travel',600)
        elif phase=='travel':
            arrived=self.at_restaurant(frame) if self.journal['destination']=='restaurant' else (
                self.fishing._main_visible(frame) and not self.bag.moving(frame)
                and cv2.matchTemplate(frame[12:36,286:434],unpack(self.journal['origin']['place']),cv2.TM_CCOEFF_NORMED)[0,0]>=.97)
            if self._confirmed('arrived' if arrived else None,frame,3):
                self._set('open' if self.journal['destination']=='restaurant' else 'restore')
        elif phase=='open':
            if self.visual.atlas(frame):self._set('atlas')
            elif self.visual.panel(frame):self._act('atlas-tab',self.visual.point(frame,'atlas-tab'),frame)
            else:self._act('restaurant-open',self.visual.point(frame,'entry'),frame)
        elif phase=='atlas':
            category=self.visual.category(frame)
            if self._confirmed(category if category is not None else None,frame):
                self.journal.update(category=category,scrolls=0,aligns=0,previous_list=None)
                if CATEGORIES[category] not in self.targets:
                    # Persist traversal separately from completed categories/cards.
                    skipped=self.journal.setdefault('skipped_categories',[])
                    if category not in skipped:
                        skipped.append(category)
                        self.event(f'餐品强化：跳过「{CATEGORIES[category]}」（不在已选类别内）')
                    self._set('category_next')
                elif category in self.journal['categories']:self._set('category_next')
                else:self._set('align')
        elif phase in ('align','list'):
            from drift.tasks.restaurant_list import step_list
            return step_list(self,frame)
        elif phase=='detail':
            if not self.enabled():self._skip('功能已关闭');return True
            if self.visual.max_detail(frame) is True:
                if self.visual.max_level(frame) is True and self._confirmed('max-level',frame,3):
                    self.journal['card']['level']=20
                    self._skip('餐品已达Lv20，关闭详情后继续图鉴')
                return True
            values=self.visual.detail_values(frame)
            if not values or not values.get('identity'):
                self._confirmed(None,frame);return True
            if not self._confirmed(str(values),frame):return True
            self.journal['dish_identity']=values['identity']
            card=self.journal['card']
            if card.get('identity') and card['identity']!=values['identity']:
                self._skip('餐品身份与当前详情不一致');return True
            card['identity']=values['identity']
            if values['identity'] in self.journal.get('exhausted_dishes',[]):
                self._skip('食材不足、已交满；本餐品本轮已耗尽',exhausted=True);return True
            if values['level']<card['level']:
                self._skip('餐品等级与选中卡片不一致');return True
            if (self.visual.upgrade_ready(frame,values)
                    and not self._upgrade_attempted(values['identity'],values['level'])):
                kind='upgrade:'+values['identity']+':'+str(values['level'])
                if self._act(kind,(604,752),frame,before=values):
                    if values['level']==19:
                        self.journal['awaiting_name']=pack(frame[480:510,209:530])
                    self.journal['awaiting']=values;self._set('upgrade_wait',60)
                else:self._stall()
                return True
            for i in range(4):
                used=self.journal['feed_counts'].get(str(i),0)
                if i in self.journal['blocked_ingredients']:continue
                if self.visual.ingredient_ready(frame,i,values):
                    kind=f"feed:{values['identity']}:{values['level']}:{i}:{used}"
                    if self._act(kind,(INGREDIENT_X[i],647),frame,before=values,ingredient=i,hold_ms=3000):
                        self.journal['feed_counts'][str(i)]=used+1
                        self.journal.update(awaiting=values,ingredient=i);self._set('feed_wait',45)
                    else:self._stall()
                    return True
            self._skip('食材不足、已交满或无可验证的后续操作',exhausted=True)
        elif phase=='feed_wait':
            v=self.visual.detail_values(frame);before=self.journal['awaiting'];i=self.journal['ingredient']
            if not v or v['identity']!=before['identity'] or v['level']!=before['level']:
                self._confirmed(None,frame);return True
            if not self._confirmed(str(v),frame,3):return True
            if any(v['counters'][j]!=tuple(before['counters'][j]) for j in range(4) if j!=i):
                self._skip('其他食材进度意外变化');return True
            delta=v['counters'][i][0]-before['counters'][i][0]
            if delta>0:
                self.journal['submitted']+=delta;self.event(f'餐品强化：第{i+1}类食材实际提交{delta}份')
                self._category_stats()['submitted']+=delta
                self.journal['stall']=0;self._set('detail',300)
            else:
                if i not in self.journal['blocked_ingredients']:self.journal['blocked_ingredients'].append(i)
                self._stall()
        elif phase=='upgrade_wait':
            v=self.visual.detail_values(frame);before=self.journal['awaiting']
            maxed=(before['level']==19 and self.visual.max_detail(frame) is True)
            if maxed:
                name=self.journal.get('awaiting_name')
                if (name and self.visual.max_name_matches(frame,unpack(name)) is True
                        and self._upgrade_attempted(before['identity'],19)):
                    v=dict(identity=before['identity'],level=20,counters=[])
                else:
                    self._confirmed(None,frame);return True
            if (v and v['identity']==before['identity'] and v['level']==before['level']+1
                    and self._confirmed(str(v),frame,3)):
                self.journal['upgraded']+=1
                counts=self.journal.setdefault('dish_upgrades',{})
                counts[v['identity']]=counts.get(v['identity'],0)+1
                upgraded=self._category_stats()['upgraded']
                if v['identity'] not in upgraded:upgraded.append(v['identity'])
                self.journal['stall']=0
                # Only an independently verified new level starts its own four
                # ingredient checks. All old per-level reservations stay saved.
                if isinstance(self.journal.get('card'),dict):
                    self.journal['card']['level']=v['level']
                self.journal['feed_counts']={}
                self.journal['blocked_ingredients']=[]
                self.event(f"餐品强化：实际Lv{before['level']}→Lv{v['level']}，继续检查该餐品")
                if maxed:
                    self.journal['card']['level']=20
                    self._skip('餐品已达Lv20，关闭详情后继续图鉴')
                else:self._set('detail',300)
            elif v and v['identity']==before['identity'] and v['level']==before['level']:
                if self._confirmed(str(v),frame,3):self._stall()
            elif not v or v['identity']!=before['identity'] or v['level']!=before['level']+1:
                self._confirmed(None,frame)
        elif phase=='detail_close':
            if self.visual.detail(frame):
                # Budget per dish; closing a later dish is a different action.
                self._act('detail-close:'+self.journal.get('dish_identity','unknown'),self.visual.point(frame,'detail-close'),frame)
            elif self.visual.atlas(frame):
                self._remember_card()
                if self._category_exhausted():self._complete_category()
                else:self._set('list',1200)
        elif phase=='category_next':
            incomplete={v['category'] for v in self.journal.get('incomplete_categories',[])}
            remaining={i for i,name in enumerate(CATEGORIES) if name in self.targets} - set(self.journal['categories']) - incomplete
            if not remaining or not self.enabled():self._set('close');return True
            category=self.visual.category(frame)
            if category is not None and category!=self.journal['category']:self._set('atlas')
            else:self._act('next:'+str(self.journal['category']),self.visual.point(frame,'next'),frame)
        elif phase=='close':
            if self.visual.detail(frame):self._act('detail-close:'+self.journal.get('dish_identity','unknown'),self.visual.point(frame,'detail-close'),frame)
            elif self.visual.panel(frame):self._act('panel-close',self.visual.point(frame,'panel-close'),frame)
            elif self._confirmed('closed' if self.at_restaurant(frame) else None,frame,3):
                if self.route_was_running() or self.journal.get('origin',{}).get('at_restaurant'):self._set('restore')
                else:self._nav('origin')
        elif phase in ('restore','restore_verify'):
            if not self.fishing._main_visible(frame):return True
            wanted=self.route_was_running();actual=self.sailing.running(frame)
            off=self.route_off(frame,actual)
            if not wanted and self._confirmed('off-restored' if off else None,frame,3):self._finish(True)
            elif wanted and off:
                if self._act('restore-open',(233,45),frame):self._set('restore_menu')
            elif wanted and actual is True:
                healthy=bool(self.fishing.blue_fill(frame) or self.fishing.real_battle(frame))
                if self._confirmed('running-fishing' if healthy else None,frame,3):self._finish(True)
        elif phase=='restore_menu':
            value=self.settings(frame)
            if value and 'settings' not in self.journal and self.journal.get('cleanup_deadline'):
                # Startup may expire before the first settings read. No option was changed.
                if self._confirmed(('restore-settings',str(value)),frame):
                    self.journal['settings']=value;self._save()
            elif value and value==self.journal.get('settings'):
                if self._confirmed('settings-restored',frame) and self._act('restore-start',(360,873),frame):
                    self._set('restore_verify',180)
        return self.active

# Shared reliability layer; retain all other deployed methods.
def _save(self):
    from drift.core.persistence import atomic_write_text
    atomic_write_text(self.path, json.dumps(self.journal, ensure_ascii=False, indent=2) + '\n')

RestaurantUpgrade._save = _save



# Module-local compatibility for every authoritative record read/write.
from drift.core.persistence import ReliablePath as Path
