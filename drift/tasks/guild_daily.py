"""18:00 guild errands with a separate, durable round-trip journal.

No guild-shop purchases, progress menu, portal close/search, or meter changes.
The runtime reacquires a frame and calls validate before reserving every input.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from drift.tasks.backpack_sale import BackpackSale
from drift.tasks.stone_farm import StoneFarm
from drift.tasks.training_sailing import TARGETS
from drift.core.journal import archive_and_replace, archive_bytes
from drift.perception.purchase_checks import currency_floor

GUILD_ID = 'driftinggirl_guild_daily'


class GuildVisual:
    """Only verified guild labels/buttons; dynamic sidebar positions are searched."""
    def __init__(self, root):
        self.root = Path(root)/'assets/guild-tasks'
        self.spec = json.loads((self.root/'visuals.json').read_text(encoding='utf-8'))
        self.images = {}
        self._frame=None;self._cache={}
        for name, spec in self.spec.items():
            image = cv2.imread(str(self.root/spec['file']))
            if image is None:
                raise RuntimeError('公会任务识别样本缺失：'+name)
            self.images[name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def find(self, frame, name):
        if frame is not self._frame:self._frame=frame;self._cache={}
        if name in self._cache:return self._cache[name]
        if frame.shape != (1280,720,3) or name not in self.spec:
            return None
        spec = self.spec[name]
        x,y,X,Y = spec['area']
        template = self.images[name]
        gray = cv2.cvtColor(frame[y:Y,x:X], cv2.COLOR_RGB2GRAY)
        best = None
        for scale in spec.get('scales',[1]):
            patch = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            h,w = patch.shape[:2]
            if h > Y-y or w > X-x:
                continue
            scores = cv2.matchTemplate(gray, cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY),cv2.TM_CCOEFF_NORMED)
            _, score, _, (px,py) = cv2.minMaxLoc(np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1))
            if score >= spec.get('threshold',.94) and (best is None or score > best[2]):
                best = ((x+px+w//2,y+py+h//2), (x+px,y+py,w,h),score)
        self._cache[name]=best
        return best

    # 公会入口的粉色 G 徽标会做脉冲动画（实测 34px ↔ 38px），固定尺度模板在
    # 动画胀大相位只有 0.60~0.92，低于阈值；这时改用「粉色像素量 + 连通域形状」
    # 兜底。实测：真实画面 288~352 像素（外接框 28x27 左右、中心恒定），活动商店
    # 与地图帧为 0 像素，因此不会误命中。
    PINK_MIN_PIXELS = 180
    PINK_MIN_WIDTH = 18
    PINK_MAX_WIDTH = 40

    def pink_badge(self, frame):
        if frame.shape != (1280,720,3):
            return None
        spec = self.spec.get('g-icon')
        if spec is None:
            return None
        x,y,X,Y = spec['area']
        roi = frame[y:Y,x:X]
        r = roi[:,:,0].astype(int); g = roi[:,:,1].astype(int); b = roi[:,:,2].astype(int)
        mask = (r > 150) & (b > 150) & (r - g > 45) & (b - g > 45)
        count = int(mask.sum())
        if count < self.PINK_MIN_PIXELS:
            return None
        ys,xs = np.nonzero(mask)
        w = int(xs.max() - xs.min() + 1); h = int(ys.max() - ys.min() + 1)
        if not (self.PINK_MIN_WIDTH <= w <= self.PINK_MAX_WIDTH):
            return None
        if not (self.PINK_MIN_WIDTH <= h <= self.PINK_MAX_WIDTH):
            return None
        center = (int(xs.mean()) + x, int(ys.mean()) + y)
        return (center, (int(xs.min()) + x, int(ys.min()) + y, w, h), float(count))

    def point(self, frame, name):
        found = self.find(frame,name)
        if found is None and name == 'g-icon':
            found = self.pink_badge(frame)
            if found is not None:
                self._cache['g-icon'] = found
        return found[0] if found else None

    def panel(self, frame):
        return all(self.find(frame,key) for key in ('tab-home','tab-members','tab-tasks','panel-close'))

    def task_page(self, frame):
        return self.panel(frame) and all(self.find(frame,key) for key in ('purple','blue'))

    def category(self, frame):
        if not self.task_page(frame):return None
        values = []
        for key in ('purple','blue'):
            hit = self.find(frame,key)
            x,y,w,h = hit[1]
            # Only the tile's blank margin, away from the colored G emblem.
            strip=frame[max(0,y-7):y+3,max(0,x-12):min(720,x+w+12)]
            values.append(float(strip.mean()))
        if abs(values[0]-values[1]) < 20:return None
        return 0 if values[0]>values[1] else 1

    def claim(self, frame):
        if not self.task_page(frame):return None
        # In this panel "已完成" means the objective is ready to submit.
        # Successful submission removes its row; it is not a claimed marker.
        return self.point(frame,'completed')

    @staticmethod
    def row_identity(frame,point):
        y=point[1]
        crop=frame[y-35:y-4,248:463]
        mask=(cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)<100).astype('uint8')*255
        return base64.b64encode(cv2.imencode('.png',mask)[1]).decode()

    @staticmethod
    def row_gone(frame,identity):
        patch=cv2.imdecode(np.frombuffer(base64.b64decode(identity),np.uint8),cv2.IMREAD_GRAYSCALE)
        if patch is None or cv2.countNonZero(patch)<30:return False
        mask=(cv2.cvtColor(frame[415:986,240:471],cv2.COLOR_RGB2GRAY)<100).astype('uint8')*255
        scores=cv2.matchTemplate(mask,patch,cv2.TM_CCOEFF_NORMED)
        return float(cv2.minMaxLoc(scores)[1])<.90

    def completed_near(self,frame,point):
        hit=self.find(frame,'completed')
        return bool(hit and abs(hit[0][1]-point[1])<45)

    def claimed_only(self,frame):
        if not self.task_page(frame) or self.claim(frame) is not None:return False
        hit=self.find(frame,'claimed')
        if hit is None:return False
        x,y,w,h=hit[1]
        return float(frame[y:y+h,x:x+w].mean())<145

    def teleport_prompt(self,frame):
        return (frame.shape==(1280,720,3)
                and float((frame[495:535,140:580].min(axis=2)>=140).mean())>=.8
                and all(self.point(frame,name) for name in ('teleport-question','teleport-currency',
                                                          'teleport-continue','teleport-cancel')))

    def restore_success(self,frame):
        if frame.shape!=(1280,720,3):return False
        if float((frame[1147:1168,558:585].max(axis=2)>=210).mean())<.30:return False
        def green(image):
            rgb=image.astype(np.int16)
            return ((rgb[:,:,1]>100)&(rgb[:,:,1]-rgb[:,:,0]>10)
                    &(rgb[:,:,1]-rgb[:,:,2]>35)).astype('uint8')*255
        template=green(self.images['restore-success'])
        region=green(frame[960:1058,290:435])
        for scale in np.linspace(.80,1.10,31):
            patch=cv2.resize(template,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
            if np.count_nonzero(patch)<100:continue
            scores=cv2.matchTemplate(region,patch,cv2.TM_CCOEFF_NORMED)
            if float(np.nan_to_num(scores,nan=-1,posinf=-1).max())>=.96:return True
            # One-pixel raster changes during the result animation must still
            # retain the complete check stroke and reject extra green shapes.
            _,score,_,(x,y)=cv2.minMaxLoc(np.nan_to_num(scores,nan=-1,posinf=-1))
            if score<.85:continue
            observed=region[y:y+patch.shape[0],x:x+patch.shape[1]]>0
            kernel=np.ones((3,3),np.uint8)
            core=cv2.erode(patch,kernel)>0;allowed=cv2.dilate(patch,kernel)>0
            amount=int(observed.sum());expected=int(np.count_nonzero(patch))
            if (int(core.sum())>=50 and .7*expected<=amount<=1.3*expected
                    and float(observed[core].mean())>=.97
                    and float((observed&~allowed).sum())/max(1,amount)<=.03):return True
        return False

    def unknown_enabled(self,frame):
        """A green action without a known submit label is not a completed page."""
        hsv=cv2.cvtColor(frame[300:1140,450:680],cv2.COLOR_RGB2HSV)
        mask=((hsv[:,:,0]>=30)&(hsv[:,:,0]<=90)&(hsv[:,:,1]>=65)&(hsv[:,:,2]>=150)).astype('uint8')
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        return any(cv2.contourArea(c)>500 for c in contours) and self.claim(frame) is None

    @staticmethod
    def content_key(frame):
        # Guild list occupies this area; ignore top notifications and sidebar.
        gray=cv2.cvtColor(frame[421:986,123:596],cv2.COLOR_RGB2GRAY)
        return hashlib.sha256(cv2.resize(gray,(136,195)).tobytes()).hexdigest()


class GuildDaily:
    LIMIT = 1800
    COLLECTION_LIMIT = 360  # both categories, bounded by the original round-trip end
    INTERVAL = 86400
    PHASES = {'idle','stop','snapshot_open','snapshot','stop_fishing','return',
              'return_confirm','guild_icon','guild_menu','guild_prompt','guild_cancel','guild_travel','guild_at',
              'tasks','collect','close','map_open','map_select','map_close','travel',
              'gate','enter','arrive','resume','resume_menu','resume_verify','blocked'}
    def __init__(self, root, reader, action, event, serial, *, clock=time.time,state_path=None,collection_only=False):
        self.root,self.reader,self.action,self.event=Path(root),reader,action,event
        self.serial,self.clock=serial,clock
        self.path=Path(state_path) if state_path else self.root/'guild_daily_state.json'
        self.collection_only=collection_only
        self.journal={'version':1,'serial':serial,'pending':False,'phase':'idle'}
        if self.path.exists():
            saved=json.loads(self.path.read_text(encoding='utf-8-sig'))
            if (saved.get('version')!=1 or saved.get('serial')!=serial
                    or saved.get('phase') not in self.PHASES or type(saved.get('pending')) is not bool):
                raise RuntimeError('公会任务往返记录无效，保留原记录')
            self.journal=saved
        # Isolated read-only probes: never read or modify production stone/bag journals.
        self.view=StoneFarm(root,reader,lambda _:False,event,state_path=self.root/'guild-probe-unused.json')
        self.bag=BackpackSale(root,lambda *a:False,event,reader,state_path=self.root/'guild-bag-unused.json')
        self.visual=GuildVisual(root)
        self.sailing,self.fishing=self.view.sailing,self.view.fishing
        self.phase=self.journal['phase'] if self.active else 'idle'
        self.enabled=lambda:True
        self.done=None;self.hold=None;self.next_at=0
        self.reject_reason=None
        self.key=self.since=self.last_seen=self.last_frame=None;self.seen=0
        self.off_verified=False;self.idle_verified=False
        self.interval=self.INTERVAL

    @property
    def active(self):return self.journal.get('pending',False)

    @property
    def started_period(self):return self.journal.get('period')

    def status(self):
        return dict(phase=self.phase,pending=self.active,period=self.started_period,
                    deadline=self.journal.get('deadline'),hold=self.hold,
                    submitted=self.journal.get('submitted',0),categories=self.journal.get('categories',[]),
                    origin=self.journal.get('origin'),target=self.journal.get('target'),
                    farm_origin=self.journal.get('farm_origin'),
                    farm_restored_at=self.journal.get('farm_restored_at'),attempts=self.journal.get('attempts',{}))

    def _save(self):
        tmp=self.path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(self.journal,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        tmp.replace(self.path)

    def retire_expired_on_start(self):
        if not self.active or not self.path.is_file():return False
        if self.phase!='blocked' and self.allowed():return False
        raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        archive=self.path.parent/'logs/guild-daily-retired'/(digest+'.json')
        updated=dict(self.journal,pending=False,phase='idle',retirement=dict(
            at=self.clock(),sha256=digest,archive=str(archive),previous_phase=self.phase))
        archive_and_replace(self.path,raw,updated,archive,digest,'公会任务','guild_daily_retry.json')
        self.journal=updated;self.phase='idle'
        self.event('公会任务：过期执行已归档，原截止和已用次数保留，不重放旧输入')
        return True

    def abandon(self):
        """中止本次往返:保留证据,但不再让同一份过期记录反复抛错。

        2026-10-26 的教训:阶段超时会把 pending 留在 true、把下次运行推到 24 小时后,
        并且这个 RuntimeError 一路冒到 runner 主循环,整整停摆 3 h 31 min。
        """
        previous=self.phase
        if self.path.is_file():
            raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            archive_bytes(self.path.parent/'logs/guild-daily-retired'/(digest+'.json'),raw,'公会任务')
        self.journal.update(pending=False,phase='idle',abandoned_at=self.clock(),
                            abandoned_phase=previous,failure=self.hold)
        self.phase='idle';self.done=False;self._save()
        self.event('公会任务:'+str(previous)+'阶段已中止,保留记录等待有界重试')
        return True

    def _set(self,phase,seconds=120):
        if self.phase==phase:return
        self.phase=phase
        self.journal.update(phase=phase,phase_deadline=min(self.journal['deadline'],self.clock()+seconds))
        self.key=None;self.seen=0
        self._save();self.event('公会任务：'+phase)

    def _confirmed(self,key,frame,seconds=3,*,max_gap=10):
        now=self.clock()
        if frame is self.last_frame:return False
        if key is None or key!=self.key or self.last_seen is None or now-self.last_seen>max_gap:
            self.key,self.since,self.seen=key,now,1 if key else 0
        else:self.seen+=1
        self.last_seen,self.last_frame=now,frame
        return key is not None and self.seen>=3 and now-self.since>=seconds

    def signature(self,frame):return self.view.signature(frame)

    def origin_screen(self,frame):
        self.reject_reason=None
        if self.view._parallel(frame):
            current=self.view.screen(frame)
            if current is not None:
                if current['rank']>=40:self.reject_reason='rank>=40'
                elif current['stage']==9 and current['rank']>=39:self.reject_reason='final-stage'
                elif current['completion']:self.reject_reason='completion'
            if self.view.training.boss_mode(frame):self.reject_reason='boss'
            if self.reject_reason or current is None or current['menu']:return None
            return dict(parallel=True,stage=current['stage'],rank=current['rank'],route=current['route'],
                        fishing=not self.view._idle_fishing(frame))
        # Resource/RAID excursions show an original-world RANK temporarily.
        # Require a real guild entry or observatory, not a missing parallel badge.
        if (self.fishing._main_visible(frame) and not self.view._battle(frame)
                and (self.visual.point(frame,'g-icon') or self.view._observatory(frame))):
            route=self.sailing.running(frame)
            if route is not None:return dict(parallel=False,stage=None,rank=None,route=route,
                                            at_guild=self.visual.point(frame,'guild-location') is not None,
                                            fishing=not self.view._idle_fishing(frame))
        return None

    @staticmethod
    def valid_farm_origin(origin):
        return (isinstance(origin,dict) and origin.get('parallel') is True
                and type(origin.get('stage')) is int and 1<=origin['stage']<=99
                and type(origin.get('rank')) is int and 0<=origin['rank']<40
                and not (origin['stage']==9 and origin['rank']>=39)
                and type(origin.get('route')) is bool)

    def return_target(self):
        # 旧的在途记录仍可完成；新往返把实际出发地与恢复目标分开保存。
        return self.journal.get('target') or self.journal.get('origin') or {}

    def target_settings(self):
        target=self.return_target()
        return target.get('settings',self.journal.get('settings'))

    def same_origin(self,frame,*,menu=False,target=None):
        origin=self.return_target() if target is None else target
        if not origin:return False
        if self.view._parallel(frame)!=origin['parallel']:return False
        if origin['parallel']:
            rank=self.view._sailing_menu_rank(frame) if menu else self.view._rank_value(frame)
            return self.view._stage(frame)==origin['stage'] and rank==origin['rank']
        return bool(self.sailing.menu(frame) if menu else self.fishing._main_visible(frame))

    def restored_settings(self,frame):
        # 旧记录的 resume_verify 已经核对过选项；新记录必须保留本轮返程核对证据。
        if 'target' not in self.journal:return True
        settings=self.target_settings()
        current=self.settings(frame)
        return bool(settings and (current if current is not None else self.journal.get('resume_settings'))==settings)

    def restored_state(self,frame):
        target=self.return_target()
        if not self.same_origin(frame) or not self.restored_settings(frame):return False
        if self.sailing.running(frame) is not target['route']:return False
        if target.get('fishing',target['route']):
            success=getattr(getattr(self,'visual',None),'restore_success',None)
            return bool(self.fishing.blue_fill(frame) or self.fishing.real_battle(frame)
                        or callable(success) and success(frame) is True)
        return target['route'] or self.view._idle_fishing(frame)

    def _finish_restored(self):
        if self.journal.get('farm_origin')==self.return_target():
            self.journal['farm_restored_at']=self.clock()
            self.journal['resume_pending']=False
        self._finish(len(set(self.journal['categories']))==2)

    @property
    def read_only_return_pending(self):
        origin,target=self.journal.get('origin'),self.return_target()
        return bool(not self.active and self.journal.get('resume_pending')
            and self.valid_farm_origin(origin) and self.valid_farm_origin(target)
            and (origin['stage']!=target['stage'] or self.journal.get('return_target_reconciliation')))

    def _observe_stale_return(self,frame,period):
        """Reconcile an old-stage target without renewing or replaying its trip."""
        j=self.journal
        origin,target=j.get('origin') or {},self.return_target()
        settings=j.get('settings')
        proof=j.get('return_target_reconciliation')
        if (self.active or self.collection_only or j.get('period')!=period
                or not j.get('resume_pending') or not self.valid_farm_origin(origin)
                or not self.valid_farm_origin(target)):
            return False
        if not proof:
            attempts=j.get('attempts',{})
            if (origin['stage']==target['stage'] or target!=j.get('farm_origin')
                    or not j.get('failed_at') or self.clock()<j.get('phase_deadline',float('inf'))
                    or j.get('last_input',{}).get('kind')!='enter-confirm'
                    or j.get('last_input',{}).get('state')!='issued_only'
                    or not all(attempts.get(k,0)>0 for k in
                               ('snapshot-open','snapshot-close','return-confirm','enter-confirm'))
                    or not isinstance(settings,dict)
                    or not all(type(settings.get(k)) is bool for k in (*TARGETS,'sale','upgrade'))
                    or not isinstance(settings.get('priority'),str) or not settings['priority']):
                return False
        menu=self.sailing.menu(frame)
        rank=((self.view._sailing_menu_rank(frame) if menu else self.view._rank_value(frame))
              if self.view._parallel(frame) else None)
        ready=(type(rank) is int and origin['rank']<=rank<40
               and self.view._stage(frame)==origin['stage']
               and (menu or self.fishing._main_visible(frame))
               and (self.view._off_icon(frame) if menu else self.sailing.running(frame) is not None))
        if not ready:
            self._confirmed(None,frame)
            return True
        if not proof:
            if self._confirmed(('stale-return-target',origin['stage'],rank),frame):
                raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
                archive_bytes(self.path.parent/'logs/guild-daily-retired'/(digest+'.json'),raw,'公会任务')
                if self.path.read_bytes()!=raw:raise RuntimeError('公会返程记录已变化，拒绝覆盖')
                corrected=dict(origin,settings=deepcopy(settings))
                j.update(target=corrected,farm_origin=deepcopy(corrected),
                         return_target_reconciliation=dict(at=self.clock(),previous_sha256=digest,
                             previous_target=deepcopy(target),read_only=True))
                self._save();self.key=None;self.seen=0
                self.event('公会任务：实际已回本轮出发阶段，只读修正跨阶段旧目标；原截止、提交及输入次数保留')
            return True
        current=self.settings(frame)
        if current is not None:
            # Observation grants no input permission. Native cruise may leave
            # this fully identified matching page before three seconds elapse;
            # restoration still needs three new actual fishing frames below.
            if current==self.target_settings() and j.get('resume_settings')!=current:
                j['resume_settings']=deepcopy(current);self._save()
            return True
        restored=(self.restored_settings(frame) and self.sailing.running(frame) is origin['route']
                  and (self.fishing.blue_fill(frame) or self.fishing.real_battle(frame)))
        if self._confirmed(('stale-return-restored',origin['stage']) if restored else None,frame,max_gap=60):
            j['restoration_verification']=dict(at=self.clock(),read_only=True,stage=origin['stage'],rank=rank,
                frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest())
            self._finish_restored()
            self.event('公会任务：只读确认当前阶段航巡和实际钓鱼，完成返程；未重放提交、瞬移或旧窗口输入')
        return True

    def settings(self,frame):
        if not self.sailing.menu(frame):return None
        values={name:self.sailing.selected(frame,name) for name in TARGETS}
        values.update(sale=self.sailing.sale_selected(frame),upgrade=self.sailing.upgrade_selected(frame))
        if any(v is None for v in values.values()):return None
        values['priority']=hashlib.sha256((frame[515:579,250:555].min(axis=2)>210).tobytes()).hexdigest()
        return values

    def allowed(self):
        return (self.active and self.phase!='blocked' and self.clock()<self.journal['deadline']
                and self.clock()<self.journal['phase_deadline'])

    @staticmethod
    def input_limit(kind):
        if kind.startswith('submit:'):return 1
        if kind.startswith('scroll:'):return 6
        if kind.startswith('map-scan:'):return 16
        return {'stop':2,'cancel':2,'return-request':3,'return-confirm':2,'cast':1,
                'route-start':1,'snapshot-open':2,'snapshot-close':2,
                'teleport-confirm':1,'teleport-cancel':1}.get(kind,3)

    def reserve(self,action):
        if not self.allowed():return False
        kind=action['kind'];attempts=self.journal['attempts']
        if attempts.get(kind,0)>=self.input_limit(kind):return False
        attempts[kind]=attempts.get(kind,0)+1
        self.journal['last_input']={'kind':kind,'time':self.clock(),'state':'reserved'}
        self._save();return True

    def input_issued(self,action,receipt):
        self.journal['last_input'].update(state='issued_only',input_id=receipt['input_id'])
        self._save()

    def _click(self,kind,point,frame,**extra):
        if point is None:return False
        request=dict(kind=kind,x=int(point[0]),y=int(point[1]),signature=self.signature(frame),
                     phase=self.phase,**extra)
        if self.action(request) is False:return False
        self.key=None;self.seen=0;self.next_at=self.clock()+1.5
        return True

    def normal_idle(self,frame):
        return (not self.view._parallel(frame) and self.view._off_icon(frame)
                and self.fishing._main_visible(frame) and not self.view._battle(frame)
                and not self.bag.moving(frame))

    def teleport_quote(self,frame):
        if not self.visual.teleport_prompt(frame) or self.view._parallel(frame) or not self.view._off_icon(frame):return None
        raw=frame[429:466,183:381]
        ink=(raw.max(axis=2)<100).astype('uint8')
        count,_,stats,_=cv2.connectedComponentsWithStats(ink,8)
        zero=(sum(stats[i,cv2.CC_STAT_AREA]>=8 for i in range(1,count))==1
                and self.visual.point(frame,'balance-zero') is not None)
        crops=[]
        green=(raw[:,:,1].astype(float)>raw[:,:,0]*1.15)&(raw[:,:,1].astype(float)>raw[:,:,2]*1.15)
        green &= (raw[:,:,1]<190)&(raw[:,:,0]<150)&(raw[:,:,2]<150)
        for threshold in (70,100):
            # The abbreviated k is green. Cropping only black pixels clipped
            # it from 1.16k and left a non-integer 1.16, blocking the whole trip.
            ink=((raw.max(axis=2)<threshold)|green).astype('uint8')
            if cv2.countNonZero(ink)<8:return None
            x,y,w,h=cv2.boundingRect(ink)
            crops.append(raw[max(0,y-3):min(raw.shape[0],y+h+3),max(0,x-4):min(raw.shape[1],x+w+4)])
        price=frame[575:617,438:518]
        price_green=(price[:,:,1].astype(float)>price[:,:,0]*1.15)&(price[:,:,1].astype(float)>price[:,:,2]*1.15)
        price_green &= (price[:,:,1]<190)&(price[:,:,0]<150)&(price[:,:,2]<150)
        if int(price_green.sum())<8:return None
        x,y,w,h=cv2.boundingRect(price_green.astype('uint8'))
        price_crops=[price[max(0,y-3):min(price.shape[0],y+h+3),max(0,x-4):min(price.shape[1],x+w+4)],
                     frame[573:619,440:520]]
        readings=self.reader.read_many(([] if zero else crops)+price_crops)
        if len(readings)!=(2 if zero else 4) or any(not r.text.isascii() or r.confidence<.95 for r in readings):return None
        costs=[r.text.strip() for r in readings[-2:]]
        if costs[0]!=costs[1] or not costs[0].isdigit() or not 1<=int(costs[0])<=9999:return None
        cost=int(costs[0])
        if zero:return dict(balance=0,cost=cost)
        texts=[r.text.strip().lower() for r in readings[:2]]
        values=[currency_floor(r,.95) for r in readings[:2]]
        if texts[0]!=texts[1] or None in values or values[0]!=values[1]:return None
        if texts[0].endswith('k') and int(green.sum())<8:return None
        # Distance changes the blue-coin price (observatory 10, New Delhi 50).
        # The old fixed words template does not establish the dynamic amount.
        return dict(balance=values[0],cost=cost)

    def _checkpoints(self):
        candidates=[self.journal]
        folder=self.path.parent/'logs/guild-daily-retired'
        if folder.is_dir():
            paths=sorted(folder.glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:24]
            for path in paths:
                if path.stat().st_size>131072:continue
                try:
                    raw=path.read_bytes()
                    if hashlib.sha256(raw).hexdigest()!=path.stem:continue
                    saved=json.loads(raw.decode('utf-8-sig'))
                except (OSError,ValueError):continue
                if not isinstance(saved,dict):continue
                candidates.append(saved)
        return candidates

    def _prompt_checkpoint(self,period):
        candidates=self._checkpoints()
        owned=[]
        for saved in candidates:
            last=saved.get('last_input',{})
            if (saved.get('serial')!=self.serial or saved.get('period')!=period
                    or 'guild_prompt' not in str(saved.get('failure',''))
                    or last.get('kind')!='direct' or last.get('state')!='issued_only'):
                continue
            owned.append(saved)
        if not owned:return None
        # Even an uncertain issued confirmation remains spent across retries.
        if any(v.get('attempts',{}).get('teleport-confirm',0) for v in candidates
               if v.get('serial')==self.serial and v.get('period')==period):return None
        return max(owned,key=lambda v:v.get('started_at',0))

    def _departure_target(self,period,target):
        """Recover an owed home only from a matching, verified departure."""
        if not self.valid_farm_origin(target):return target
        candidates=[]
        for saved in self._checkpoints():
            origin=saved.get('origin');settings=saved.get('settings')
            if (saved.get('serial')!=self.serial or saved.get('period')!=period
                    or saved.get('target')!=target or not self.valid_farm_origin(origin)
                    or origin['stage']!=target['stage'] or origin['rank']<target['rank']
                    or not saved.get('attempts',{}).get('return-confirm')
                    or saved.get('farm_restored_at') is not None
                    or not isinstance(settings,dict)
                    or any(type(settings.get(k)) is not bool for k in (*TARGETS,'sale','upgrade'))
                    or not isinstance(settings.get('priority'),str)):
                continue
            candidates.append(saved)
        if not candidates:return target
        saved=max(candidates,key=lambda v:v.get('started_at',0))
        return dict(deepcopy(saved['origin']),settings=deepcopy(saved['settings']))

    def _adopt_teleport_prompt(self,frame,period):
        """Adopt our known dialog within a fresh normal retry's existing budget.

        No old execution deadline is renewed, and every spent input is carried
        forward. A prior departure checkpoint supplies the actual return RANK.
        """
        if (self.phase!='stop' or self.journal.get('origin') is not None
                or self.journal.get('prompt_recovery') or self.visual.teleport_prompt(frame) is not True
                or self.view._parallel(frame) or not self.view._off_icon(frame)):
            return None
        previous=self._prompt_checkpoint(period)
        if previous is None:return None
        digest=hashlib.sha256(json.dumps(previous,sort_keys=True).encode()).hexdigest()
        if not self._confirmed(('resume-teleport',digest,tuple(self.signature(frame))),frame):return True
        attempts=deepcopy(self.journal.get('attempts',{}))
        for name,count in previous.get('attempts',{}).items():attempts[name]=max(attempts.get(name,0),count)
        target=deepcopy(self.return_target())
        origin=previous.get('origin');settings=previous.get('settings')
        if (self.valid_farm_origin(origin) and self.valid_farm_origin(target)
                and origin['stage']==target['stage'] and origin['rank']>=target['rank']
                and isinstance(settings,dict) and all(settings.get(k) is not None
                    for k in (*TARGETS,'sale','upgrade','priority'))):
            target=dict(deepcopy(origin),settings=deepcopy(settings))
        remaining=self.journal['phase_deadline']-self.clock()
        self.journal.update(attempts=attempts,target=target,farm_origin=deepcopy(target),
            settings=deepcopy(settings) if isinstance(settings,dict) else self.journal.get('settings'),
            origin=deepcopy(origin),prompt_recovery=dict(at=self.clock(),
                previous_started_at=previous.get('started_at'),previous_deadline=previous.get('deadline'),
                previous_phase_deadline=previous.get('phase_deadline'),
                previous_attempts=deepcopy(previous.get('attempts',{}))))
        self.off_verified=self.idle_verified=True
        self._set('guild_prompt',max(0,min(60,remaining)))
        self.event('公会任务：本次新重试接续已验证的瞬移页，沿用已用次数，不重复出发；原记录和截止保留')
        return True

    def stopped_map(self,frame):
        return (self.view._map(frame) and self.view._off_icon(frame)
                and self.view._match(frame,'world-badge',(10,43,42,75),.92) is None
                and self.bag._find(frame,'map-legend',(9,1136,405,1241),.93) is not None)

    def map_target(self,frame):
        if not self.stopped_map(frame):return None
        target=self.journal.get('map_goal')
        if target=='observatory':
            found=self.view._match(frame,'map-observatory',(20,175,700,1115),.94)
            return found[:2] if found else None
        if target=='origin':
            raw=self.journal.get('origin_map')
            if not raw:return None
            self.bag.origin=cv2.imdecode(np.frombuffer(base64.b64decode(raw),np.uint8),cv2.IMREAD_COLOR)
            return self.bag._origin_target(frame)
        ports=self.bag._ports(frame)
        return ports[0] if ports else None

    def validate(self,action,frame):
        if self.journal.get('return_only'):
            if action['kind'] in ('direct','teleport-confirm'):return False
            if not self.journal.get('collect_here') and (action['kind'].startswith('submit:')
                    or action['kind'] in ('tab-tasks','g-icon')):return False
        if not self.allowed() or action['phase']!=self.phase:return False
        kind=action['kind'];point=(action['x'],action['y'])
        if self.journal['attempts'].get(kind,0)>=self.input_limit(kind):return False
        if kind=='cancel':
            return (self.off_verified and self.view._off_icon(frame)
                and self.signature(frame)==action['signature'] and self.sailing.running(frame) is False
                and not self.view.training.boss_mode(frame) and not self.view._battle(frame)
                and self.fishing.cancel_button(frame) and self.view.continuous_preparation(frame)
                and point==(690,1090))
        if kind=='return-confirm':
            return (self.off_verified and self.idle_verified and self.view._parallel(frame)
                and self.view._off_icon(frame)
                and self.view._fixed(frame,'return-question',(223,556,506,629),.95) is not None
                and point==(208,788))
        if kind=='teleport-confirm':
            quote=self.teleport_quote(frame)
            return (self.enabled() and quote is not None and quote['balance']>=quote['cost']
                    and quote==self.journal.get('teleport_quote')
                    and point==self.visual.point(frame,'teleport-continue'))
        if kind=='teleport-cancel':
            return self.visual.teleport_prompt(frame) and point==self.visual.point(frame,'teleport-cancel')
        if kind.startswith('map-scan:') or kind=='map-select':
            return (self.stopped_map(frame) and self.signature(frame)==action['signature']
                    and (kind!='map-select' or self.map_target(frame)==point))
        if kind=='map-close':
            return (self.phase in ('map_close','travel') and self.stopped_map(frame)
                    and not self.bag.moving(frame)
                    and self.signature(frame)==action['signature'] and point==(668,1187))
        if kind=='enter-confirm':
            return (not self.view._parallel(frame) and self.view._observatory(frame)
                and self.view._off_icon(frame)
                and self.view._fixed(frame,'enter-question',(220,556,502,620),.95) is not None
                and point==(208,787))
        if kind.startswith('submit:'):
            return (self.enabled() and self.visual.category(frame)==self.journal.get('category')
                    and self.visual.claim(frame)==point
                    and self.visual.content_key(frame)==action.get('content'))
        if kind.startswith('scroll:'):
            return self.visual.category(frame)==self.journal.get('category')
        if kind in ('purple','blue','tab-tasks','panel-close'):
            return self.visual.panel(frame) and self.visual.point(frame,kind)==point
        if kind in ('g-icon','direct','guild-tasks','view-guild'):
            return (self.normal_idle(frame) and self.visual.point(frame,kind)==point
                    and (kind=='g-icon' or self.visual.point(frame,'g-icon') is not None))
        if kind in ('snapshot-close','route-start'):
            restoring=self.phase in ('resume_menu','resume_verify')
            settings=self.target_settings() if restoring else self.journal.get('settings')
            return (self.view._off_icon(frame) and self.sailing.menu(frame)
                and bool(settings) and self.settings(frame)==settings
                and (not restoring or self.same_origin(frame,menu=True))
                and point==((600,361) if kind=='snapshot-close' else (360,873)))
        if self.signature(frame)!=action['signature']:return False
        if kind=='stop':
            return (self.sailing.running(frame) is True and point==(233,45)
                    and (self.phase not in ('resume','resume_verify') or self.same_origin(frame)))
        if kind in ('snapshot-open','route-open'):
            return (self.sailing.running(frame) is False and point==(233,45)
                    and (kind!='route-open' or (self.same_origin(frame)
                         and self.fishing._main_visible(frame) and not self.sailing.menu(frame))))
        if kind=='return-request':
            return (self.off_verified and self.idle_verified and self.view._parallel(frame)
                and self.view._off_icon(frame) and self.view._idle_fishing(frame)
                and self.view._return_button(frame)==point)
        if kind=='map-open':return self.normal_idle(frame) and point==(350,30)
        if kind=='enter':return self.view._original(frame) and self.view._gate_active(frame) and point==(354,482)
        if kind=='cast':
            return (self.same_origin(frame) and not self.fishing.at_port(frame)
                and self.view._idle_fishing(frame) and self.fishing.idle_icon(frame)=='hook'
                and point==(360,1010))
        return False

    def _map_begin(self,goal,after):
        self.journal.update(map_goal=goal,map_after=after,map_scans=0)
        self._set('map_open',180)

    def _finish(self,success):
        self.journal.update(pending=False,phase='idle',finished_at=self.clock(),success=bool(success))
        self.phase='idle';self.done=bool(success);self._save()
        self.event('公会任务：两类任务已检查并恢复原活动' if success else '公会任务：本次未全部核对，已恢复原活动')

    def step(self,frame,*,due,period):
        self.done=None
        if not self.active:
            if self._observe_stale_return(frame,period):return False
            if not due or not self.enabled():return False
            if (self.journal.get('resume_pending') and self.journal.get('period')==period
                    and len(set(self.journal.get('categories',[])))==2
                    and self.journal.get('resume_settings') and self.valid_farm_origin(self.return_target())):
                # A completed hand-in and issued restore must never repeat the
                # departure just because its final observation timed out.
                if self.restored_state(frame):
                    if self._confirmed(('restored-read-only',),frame,max_gap=60):
                        raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
                        archive_bytes(self.path.parent/'logs/guild-daily-retired'/(digest+'.json'),raw,'公会任务')
                        self.journal['restoration_verification']=dict(at=self.clock(),read_only=True,
                            previous_sha256=digest,frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest())
                        self._finish_restored()
                        self.event('公会任务：新截图确认原世界、设置、航巡及实际钓鱼，只读完成返程，不重放输入')
                return False
            if self.collection_only:
                if not self.visual.task_page(frame):return False
                origin=dict(parallel=False,stage=None,rank=None,route=False)
            else:
                origin=self.origin_screen(frame)
                if self.reject_reason:
                    self._confirmed(None,frame);return False
            previous=deepcopy(self.journal)
            farm=deepcopy(self.journal.get('farm_origin'))
            restored_at=self.journal.get('farm_restored_at')
            # Legacy journals with an unrestored farm also owe the original activity.
            resume_pending=self.journal.get('resume_pending',self.valid_farm_origin(farm) and restored_at is None)
            refresh=(not self.collection_only and self.valid_farm_origin(origin)
                     and (not self.valid_farm_origin(farm) or restored_at is not None or not resume_pending
                          or ((origin['stage'],origin['rank'])!=(farm['stage'],farm['rank']) and origin['route'] is True
                              and origin.get('fishing') is True)))
            if origin and origin.get('parallel') and not self.valid_farm_origin(origin):
                self._confirmed(None,frame);return False
            if refresh:
                target=dict(origin,settings=None);farm=deepcopy(target)
            elif not self.collection_only and self.valid_farm_origin(farm):
                target=self._departure_target(period,deepcopy(farm));farm=deepcopy(target)
            elif self.collection_only:target=dict(origin,settings=None)
            else:
                self._confirmed(None,frame);return False
            key=('start',json.dumps([origin,target],sort_keys=True))
            if not self._confirmed(key,frame):return False
            return_only=(not self.collection_only and previous.get('period')==period
                and resume_pending and self.valid_farm_origin(farm)
                and ((previous.get('retirement') or {}).get('previous_phase') in ('collect','close')
                     or previous.get('return_only') is True)
                and previous.get('attempts',{}).get('teleport-confirm',0)>0)
            collect_here=bool(return_only and origin and origin.get('at_guild') and origin['route'] is False)
            # 2026-10-26 18:02 失败后，22:40 补跑把公会误当终点；农场起点不能随新一轮日志丢失。
            self.journal=dict(version=1,serial=self.serial,period=period,pending=True,
                phase='tasks' if self.collection_only else 'stop',origin=origin,target=target,
                farm_origin=farm,farm_restored_at=restored_at if self.collection_only else None,
                resume_pending=resume_pending if self.collection_only else True,
                capture_target_settings=refresh or not self.valid_farm_origin(farm),
                scope='collection' if self.collection_only else 'round_trip',
                started_at=self.clock(),deadline=self.clock()+(600 if self.collection_only else self.LIMIT),
                phase_deadline=self.clock()+180,attempts={},categories=[],submitted=0)
            if return_only:
                self.journal.update(return_only=True,collect_here=collect_here,attempts=deepcopy(previous.get('attempts',{})),
                    categories=deepcopy(previous.get('categories',[])),submitted=previous.get('submitted',0),
                    retry_of=deepcopy(previous.get('retirement') or previous.get('retry_of')))
                self.event('公会任务：已在公会，沿用已用次数提交剩余任务，不再付费瞬移' if collect_here else
                    '公会任务：新重试仅接续中断后的返程，继承已用次数；不再付费瞬移或重复提交')
            if (not return_only or collect_here) and not self.collection_only and origin and origin.get('at_guild') and origin['route'] is False:
                self.journal['phase']='guild_at'
            self.phase=self.journal['phase'];self._save()
            self.event('公会任务：18:00每日检查开始，保存原世界及航巡状态')
        if not self.allowed():
            limit='总流程' if self.clock()>=self.journal['deadline'] else self.phase+'阶段'
            self.hold=f'公会往返{limit}超过原有时限；保留当前阶段和输入记录'
            self.journal.update(phase='blocked',failure=self.hold,failed_at=self.clock())
            self.phase='blocked';self._save()
            raise RuntimeError(self.hold)
        if self.clock()<self.next_at:return True
        phase=self.phase
        if phase=='stop':
            adopted=self._adopt_teleport_prompt(frame,period)
            if adopted is not None:return adopted
            origin=self.journal.get('origin')
            if origin is None:
                if self.visual.panel(frame):self._set('close' if self.journal.get('return_only') else 'tasks');return True
                if (self.journal.get('return_only') and self.valid_farm_origin(self.return_target())
                        and self.stopped_map(frame) and not self.bag.moving(frame)):
                    if self._confirmed(('pending-return-map',tuple(self.signature(frame))),frame):
                        remaining=max(0,self.journal['phase_deadline']-self.clock())
                        self.journal.update(map_goal='observatory',map_after='gate',
                            map_scans=self.journal.get('attempts',{}).get('map-scan:observatory',0))
                        self._set('map_select',min(180,remaining))
                    return True
                if (self.journal.get('return_only') and self.valid_farm_origin(self.return_target())
                        and self.sailing.menu(frame) and self.view._off_icon(frame)):
                    settings=self.settings(frame)
                    key=('pending-cruise-settings',json.dumps(settings,sort_keys=True),tuple(self.signature(frame))) if settings else None
                    if self._confirmed(key,frame):
                        remaining=max(0,self.journal['phase_deadline']-self.clock())
                        self._set('snapshot',min(120,remaining))
                    return True
                if (self.valid_farm_origin(self.return_target()) and not self.view._parallel(frame)
                        and self.view._observatory(frame) and self.view._off_icon(frame)
                        and self.view._fixed(frame,'enter-question',(220,556,502,620),.95) is not None):
                    # An abandoned arrival can leave its dialog covering the departure screen.
                    if self._confirmed(('pending-enter',tuple(self.signature(frame))),frame):
                        self._set('arrive',self.journal['phase_deadline']-self.clock())
                    return True
                origin=self.origin_screen(frame)
                if origin is None:
                    self._confirmed(None,frame);return True
                self.journal['origin']=origin;self._save()
            if not self.same_origin(frame,target=origin):return True
            route=self.sailing.running(frame)
            if route is True:self._click('stop',(233,45),frame)
            elif self._confirmed(('off',) if route is False else None,frame):
                self.off_verified=True
                self._set('stop_fishing' if self.journal.get('return_only') else 'snapshot_open')
        elif phase=='snapshot_open':
            if self.sailing.menu(frame):self._set('snapshot')
            elif self._click('snapshot-open',(233,45),frame):self._set('snapshot')
        elif phase=='snapshot':
            settings=self.settings(frame)
            target=self.return_target()
            # Retain the owed activity while completing only a verified home snapshot.
            capture=(self.same_origin(frame,menu=True,target=target)
                     and (self.journal.get('capture_target_settings')
                          or (self.valid_farm_origin(target) and target.get('settings') is None
                              and target==self.journal.get('farm_origin'))))
            key=('settings',json.dumps(settings,sort_keys=True),capture) if settings else None
            if self._confirmed(key,frame):
                self.journal['settings']=settings
                if capture:
                    self.journal['target']['settings']=deepcopy(settings)
                    if self.valid_farm_origin(self.journal['target']):
                        self.journal['farm_origin']=deepcopy(self.journal['target'])
                self._save()
                if self._click('snapshot-close',(600,361),frame):self._set('stop_fishing',180)
        elif phase=='stop_fishing':
            if self.view._idle_fishing(frame) and self.view._off_icon(frame):
                if self._confirmed(('idle',tuple(self.signature(frame))),frame):
                    self.idle_verified=self.off_verified=True
                    if self.journal.get('return_only'):
                        if self.same_origin(frame):self._set('resume',180)
                        elif self.view._npc_ready(frame):self._set('gate',180)
                        else:self._map_begin('observatory','gate')
                    elif self.journal['origin']['parallel']:self._set('return',180)
                    elif self.journal['origin'].get('at_guild'):self._set('guild_at')
                    else:self._map_begin('nearby','guild_icon')
            elif (self.off_verified and self.view._off_icon(frame) and not self.view.training.boss_mode(frame)
                    and self.fishing.cancel_button(frame) and self.view.continuous_preparation(frame)):
                self._click('cancel',(690,1090),frame)
        elif phase in ('return','return_confirm'):
            if self.view._original(frame):
                if self._confirmed(('original',),frame):self._set('guild_icon')
            elif self.view._fixed(frame,'return-question',(223,556,506,629),.95):
                if self._click('return-confirm',(208,788),frame):self._set('return_confirm',120)
            elif phase=='return':self._click('return-request',self.view._return_button(frame),frame)
        elif phase in ('guild_icon','guild_at'):
            if self.normal_idle(frame):
                if self._click('g-icon',self.visual.point(frame,'g-icon'),frame):
                    self._set('guild_menu' if phase=='guild_icon' else 'tasks')
        elif phase=='guild_menu':
            if self.visual.panel(frame):self._set('tasks')
            elif self._click('direct',self.visual.point(frame,'direct'),frame):self._set('guild_prompt',60)
        elif phase=='guild_prompt':
            quote=self.teleport_quote(frame)
            if quote and self._confirmed(('quote',quote['balance'],quote['cost']),frame):
                self.journal['teleport_quote']=quote;self._save()
                if quote['balance']<quote['cost']:
                    self.hold=f'蓝色公会币不足{quote["cost"]}，取消瞬移并恢复原活动，本日未记完成'
                    if self._click('teleport-cancel',self.visual.point(frame,'teleport-cancel'),frame):
                        self._set('guild_cancel')
                elif self._click('teleport-confirm',self.visual.point(frame,'teleport-continue'),frame):
                    self._set('guild_travel',900)
        elif phase=='guild_cancel':
            if self._confirmed(('cancelled',) if self.normal_idle(frame) else None,frame):
                if self.return_target()['parallel'] and self.view._npc_ready(frame):self._set('gate')
                elif self.return_target()['parallel']:self._map_begin('observatory','gate')
                else:self._map_begin('origin','resume')
        elif phase=='guild_travel':
            here=(self.normal_idle(frame) and self.visual.point(frame,'g-icon')
                  and self.visual.point(frame,'guild-location'))
            if self._confirmed(('guild',) if here else None,frame):
                self._set('guild_at')
            elif here and self.clock()>=self.journal['phase_deadline']-30:
                # 单帧判据已成立、且离阶段截止不到 30 秒:直接放行。2026-10-26 18:12
                # 就是"首帧成立、三帧确认还差两帧"时被阶段截止判死(整整晚 600 秒)。
                # guild_at 会用 normal_idle 复核后才点击,早放行的代价只是多等一帧。
                self.event('公会任务：临近阶段截止，按单帧判定已到达公会')
                self._set('guild_at')
            elif self.clock()>=getattr(self,'travel_note_at',0):
                # 这个分支原先一行日志都没有,600 秒里发生了什么无从判断(2026-10-26 事故)。
                self.travel_note_at=self.clock()+30
                self.event('公会任务：等待到达公会 主界面=%s 入口=%s 地点=%s 确认帧=%s/%s' % (
                    self.normal_idle(frame),bool(self.visual.point(frame,'g-icon')),
                    bool(self.visual.point(frame,'guild-location')),self.seen,
                    round(self.clock()-self.since,1) if self.since else None))
        elif phase=='tasks':
            if self.visual.task_page(frame):
                self.journal.update(category=0,scrolls=0,previous_content=None,unchanged=0)
                self._set('collect',self.COLLECTION_LIMIT)
            elif self.visual.panel(frame):self._click('tab-tasks',self.visual.point(frame,'tab-tasks'),frame)
            else:
                point=self.visual.point(frame,'guild-tasks')
                if point:self._click('guild-tasks',point,frame)
                else:self._click('view-guild',self.visual.point(frame,'view-guild'),frame)
        elif phase=='collect':
            if not self.visual.task_page(frame):return True
            category=self.journal['category']
            if self.visual.category(frame)!=category:
                name=('purple','blue')[category];self._click(name,self.visual.point(frame,name),frame);return True
            content=self.visual.content_key(frame)
            previous=self.journal.get('await_claim')
            if previous:
                if (content==previous['content'] or not (self.visual.row_gone(frame,previous['identity'])
                                                        or self.visual.claimed_only(frame))
                        or not self._confirmed(('submitted',content),frame)):
                    return True
                self.journal['submitted']+=1;self.journal.pop('await_claim',None);self._save()
            point=self.visual.claim(frame)
            if point and self.enabled():
                kind='submit:'+str(category)+':'+content
                if self._click(kind,point,frame,content=content):
                    self.journal['await_claim']={'content':content,'point':point,
                                                'identity':self.visual.row_identity(frame,point)};self._save()
                return True
            if self.visual.unknown_enabled(frame):
                self.hold='发现尚未识别的可用按钮，保留截图，不能当作已领取'
                return True
            if not self._confirmed(('list',category,content),frame):return True
            if content==self.journal.get('previous_content'):
                self.journal['unchanged']=self.journal.get('unchanged',0)+1
            else:self.journal['unchanged']=0
            if self.journal['unchanged']>=1:
                self.journal['categories'].append(category)
                if category==0:
                    self.journal.update(category=1,scrolls=0,previous_content=None,unchanged=0)
                    self._save();self.key=None
                else:self._set('close')
            elif self.journal.get('scrolls',0)<6:
                if self._click('scroll:'+str(category),(360,980),frame,swipe=[360,500,500]):
                    self.journal.update(previous_content=content,scrolls=self.journal.get('scrolls',0)+1)
                    self._save()
            else:
                self.hold='列表滚动达到本轮上限，未把未知页记作已完成';self._set('close')
        elif phase=='close':
            if self.visual.panel(frame):self._click('panel-close',self.visual.point(frame,'panel-close'),frame)
            else:
                target=self.return_target()
                home=not self.collection_only and target['parallel'] and self.same_origin(frame)
                restored=home and self.restored_state(frame)
                key=('restored',) if restored else ('home',) if home else ('closed',) if self.normal_idle(frame) else None
                if self._confirmed(key,frame):
                    if restored:self._finish_restored()
                    elif self.collection_only or not target['parallel'] and target.get('at_guild'):
                        self._finish(len(set(self.journal['categories']))==2)
                    elif home:self._set('resume',180)
                    else:self._map_begin('observatory' if target['parallel'] else 'origin',
                                        'gate' if target['parallel'] else 'resume')
        elif phase=='map_open':
            if self.stopped_map(frame):self._set('map_select',180)
            else:self._click('map-open',(350,30),frame)
        elif phase=='map_select':
            if not self.stopped_map(frame):return True
            if not self.return_target()['parallel'] and not self.journal.get('origin_map'):
                player=(self.bag._find(frame,'map-player-current',(20,175,700,1115),.94)
                        or self.bag._find(frame,'map-player',(20,175,700,1115),.94))
                if player is None:return True
                x,y=player
                if not (90<=x<=610 and 230<=y<=1090):return True
                self.journal['origin_map']=base64.b64encode(cv2.imencode('.png',frame[y-52:y-20,x-70:x+70])[1]).decode()
                self._save()
            point=self.map_target(frame)
            if point:
                if self._click('map-select',point,frame):
                    # The stopped-map guard waits for manual travel to finish.
                    # A distant observatory return needs the same travel budget
                    # as the next phase, still capped by this trip's total end.
                    self._set('map_close',600 if self.journal['map_goal']=='observatory' else 120)
            elif self.journal['map_scans']<16:
                n=self.journal['map_scans']
                # The observatory is in the western extension. The real sixth
                # scan from Colombo had not reached that extension yet;
                # reversing there searched the already-seen eastern map again.
                right=self.journal['map_goal']=='observatory' or n<6
                if self._click('map-scan:'+self.journal['map_goal'],(200 if right else 520,650),frame,
                               swipe=[520 if right else 200,650,400]):
                    self.journal['map_scans']+=1;self._save()
        elif phase=='map_close':
            if self.view._map(frame):
                self._close_return_map(frame)
            elif self._confirmed(('map-closed',tuple(self.signature(frame)))
                                 if self.normal_idle(frame) else None,frame):
                self.hold=None;self._set('travel',600)
            else:
                self.hold='等待连续确认地图已关闭；未知画面不作为抵达依据'
        elif phase=='travel':
            # An older saved phase can still have the map open. Close only a
            # freshly confirmed stopped map; never renew this travel deadline.
            if self.view._map(frame):
                if self.journal.get('attempts',{}).get('map-select',0)>0:
                    self._close_return_map(frame)
                else:self._confirmed(None,frame)
                return True
            ready=self.normal_idle(frame)
            if self.journal['map_goal']=='observatory':ready=ready and self.view._npc_ready(frame)
            elif self.journal['map_goal']=='nearby':ready=ready and self.visual.point(frame,'g-icon') is not None
            if not ready and self.clock()>=getattr(self,'travel_note_at',0):
                # 2026-10-27 事故：travel 阶段无日志，600 秒超时后无法判断卡在哪个判据。
                self.travel_note_at=self.clock()+30
                self.event('公会任务：等待抵达港口 主界面=%s 入口=%s 移动=%s 确认帧=%s/%s'
                           % (self.normal_idle(frame),self.visual.point(frame,'g-icon') is not None,
                              self.bag.moving(frame),self.seen,3))
            if self._confirmed(('arrived',) if ready else None,frame):self._set(self.journal['map_after'],180)
        elif phase=='gate':
            if self.view._original(frame) and self.view._gate_active(frame):
                if self._confirmed(('gate',),frame):self._set('enter')
        elif phase=='enter':
            if self.view._parallel(frame):self._set('arrive')
            elif self.view._fixed(frame,'enter-question',(220,556,502,620),.95):
                if self._click('enter-confirm',(208,787),frame):self._set('arrive',180)
            else:self._click('enter',(354,482),frame)
        elif phase=='arrive':
            question=self.view._fixed(frame,'enter-question',(220,556,502,620),.95)
            if question is not None and not self.view._parallel(frame):
                ready=self.view._observatory(frame) and self.view._off_icon(frame)
                key=('enter-question',tuple(self.signature(frame))) if ready else None
                if (self._confirmed(key,frame)
                        and self.journal['attempts'].get('enter-confirm',0)<self.input_limit('enter-confirm')):
                    self._click('enter-confirm',(208,787),frame)
            elif self._confirmed(('origin',) if self.same_origin(frame) else None,frame):self._set('resume',180)
        elif phase in ('resume','resume_verify'):
            if not self.same_origin(frame):
                self._confirmed(None,frame);return True
            running=self.sailing.running(frame)
            target=self.return_target();wanted=target['route']
            previously_fishing=target.get('fishing',wanted)
            settings_ok=self.restored_settings(frame)
            if self.restored_state(frame):
                # Fast ordinary fights can end between captures. Require three
                # actual fill/battle frames, but allow the normal cast cycle
                # between them. The original phase/overall deadlines still win.
                if self._confirmed(('resumed',),frame,max_gap=60 if previously_fishing else 10):
                    self._finish_restored()
            elif running is True and (not wanted or not settings_ok):
                self._confirmed(None,frame)
                self._click('stop',(233,45),frame)
            elif running is False and (wanted or not settings_ok):
                self._confirmed(None,frame)
                if self._click('route-open',(233,45),frame):
                    # Persisted attempts distinguish reentry, including after a restart.
                    remaining=self.journal['phase_deadline']-self.clock()
                    self._set('resume_menu',remaining if self.journal['attempts'].get('route-open',0)>1
                              else min(120,remaining))
            elif (previously_fishing and running is wanted and self.view._idle_fishing(frame)
                    and not self.fishing.at_port(frame)):
                if self._confirmed(('idle-cast',tuple(self.signature(frame))),frame):self._click('cast',(360,1010),frame)
            elif running is not wanted:self._confirmed(None,frame)
        elif phase=='resume_menu':
            if self.sailing.menu(frame) and self.same_origin(frame,menu=True):
                settings=self.target_settings()
                if not settings or self.settings(frame)!=settings:
                    self._confirmed(None,frame)
                    self.hold='游戏航巡选项与离开前不一致，保留当前选项并等待核对';return True
                self.hold=None
                if self._confirmed(('route-settings',json.dumps(settings,sort_keys=True)),frame):
                    wanted=self.return_target()['route']
                    if self._click('route-start' if wanted else 'snapshot-close',
                                   (360,873) if wanted else (600,361),frame):
                        self.journal['resume_settings']=deepcopy(settings)
                        self._set('resume_verify',180)
            elif (not self.sailing.menu(frame) and self.same_origin(frame)
                    and self.fishing._main_visible(frame)):
                running=self.sailing.running(frame)
                key=('resume-main',tuple(self.signature(frame)),running) if running is not None else None
                if self._confirmed(key,frame):
                    if (running is False and not self.restored_state(frame)
                            and self.journal['attempts'].get('route-open',0)<self.input_limit('route-open')):
                        self._click('route-open',(233,45),frame)
                    else:
                        # A missed menu input must not renew either restoration deadline.
                        self._set('resume',self.journal['phase_deadline']-self.clock())
            else:self._confirmed(None,frame)
        return True

    def _close_return_map(self,frame):
        ready=self.stopped_map(frame) and not self.bag.moving(frame)
        key=('stopped-return-map',tuple(self.signature(frame))) if ready else None
        self.hold='地图仍打开，等待停航和固定地点连续确认后关闭'
        if self._confirmed(key,frame) and self._click('map-close',(668,1187),frame):
            self.hold=None
