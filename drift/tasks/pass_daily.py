"""Daily PASS: wait until 10:00 Beijing and 120 fish, claim once, verify return."""
import hashlib
from pathlib import Path
import time
import cv2
import numpy as np
from drift.perception.digit_ocr import read_numeric_views

PASS_ID='driftinggirl_pass_daily'


class PassDaily:
    LIMIT=60
    RETRY_SECONDS=600
    def __init__(self,root,reader,action,event):
        self.reader,self.action,self.event=reader,action,event
        self.assets={}
        for name in ('entry','daily-ready','daily-claimed'):
            data=cv2.imread(str(Path(root)/'assets/pass'/(name+'.png')))
            if data is None or data.shape[:2]!=(1280,720):
                raise RuntimeError('PASS每日奖励模板不完整')
            self.assets[name]=cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
        ref=self.assets['daily-ready'][251:312,605:677]
        red=((ref[:,:,0]>=180)&(ref[:,:,0]>ref[:,:,1].astype(float)*1.4)).astype('uint8')
        self.cross_mask=cv2.dilate(red,np.ones((3,3),np.uint8))
        tick=self.assets['daily-claimed'][507:553,169:238]
        hsv=cv2.cvtColor(tick,cv2.COLOR_RGB2HSV)
        self.tick_ink=(hsv[:,:,0]>=35)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>=90)&(hsv[:,:,2]>=120)
        self.tick_mask=cv2.dilate(self.tick_ink.astype('uint8'),np.ones((3,3),np.uint8))
        self.phase='idle';self.deadline=None;self.seen=0;self.key=None
        self.started_period=None;self.claim_allowed=False;self.claim_issued=False
        self.claim_verified=False;self.done=None;self.fish_count=None;self.attempts={}
        self.cycles=0
        self.last_action_at=None
        self.main_proof_key=None;self.main_proof_since=None;self.main_proof_seen=0
        self.main_proof_frame=None

    @property
    def active(self):return self.phase!='idle'

    def status(self):
        return dict(phase=self.phase,fish_count=self.fish_count,required_fish=120,
                    claim_issued=self.claim_issued,verified=self.claim_verified,cycles=self.cycles)

    def _fixed(self,frame,name,box,threshold=.96):
        if frame.shape!=(1280,720,3):return False
        x,y,X,Y=box
        score=cv2.matchTemplate(frame[y:Y,x:X],self.assets[name][y:Y,x:X],cv2.TM_CCOEFF_NORMED)[0,0]
        return np.isfinite(score) and score>=threshold

    def _cross(self,frame):
        if frame.shape!=(1280,720,3):return False
        crop=frame[244:320,598:684];ref=self.assets['daily-ready'][251:312,605:677]
        scores=cv2.matchTemplate(cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY),cv2.cvtColor(ref,cv2.COLOR_RGB2GRAY),
                                cv2.TM_CCOEFF_NORMED,mask=self.cross_mask)
        _,score,_,(x,y)=cv2.minMaxLoc(np.where(np.isfinite(scores),scores,-1))
        sample=crop[y:y+ref.shape[0],x:x+ref.shape[1]].astype(np.int16)
        red=(sample[:,:,0]>=180)&(sample[:,:,0]>sample[:,:,1]*1.4)&(sample[:,:,0]>sample[:,:,2]*1.4)
        return score>=.97 and int((red&(self.cross_mask>0)).sum())>=1200

    def panel(self,frame):
        return (self._fixed(frame,'daily-ready',(27,282,146,331),.95)
                and self._fixed(frame,'daily-ready',(380,215,459,245),.95)
                and self._cross(frame))

    def daily_page(self,frame):
        return (self.panel(frame) and self._fixed(frame,'daily-ready',(283,306,436,335))
                and self._fixed(frame,'daily-ready',(142,406,239,440))
                and self._fixed(frame,'daily-ready',(126,952,576,977)))

    def entry(self,frame):
        return (self._fixed(frame,'entry',(624,174,693,201),.96)
                and self._fixed(frame,'entry',(208,1137,269,1192),.91)
                and self._fixed(frame,'entry',(539,1137,600,1186),.91)
                and float((frame[1147:1168,558:585].max(axis=2)>=210).mean())>=.3)

    def main_visible(self,frame):
        # PASS and the right sidebar disappear during ordinary battles. The
        # fixed MENU label and bright bottom icons remain visible on main.
        return (self._fixed(frame,'entry',(631,14,695,59),.96)
                and self._fixed(frame,'entry',(208,1137,269,1192),.91)
                and self._fixed(frame,'entry',(539,1137,600,1186),.91)
                and float((frame[1147:1168,558:585].max(axis=2)>=210).mean())>=.3
                and not self.panel(frame))

    def _main_return_confirmed(self,frame,visible,now):
        context=self.context(frame) if visible else None
        if context is None:
            self.main_proof_key=None;self.main_proof_since=None;self.main_proof_seen=0
            self.main_proof_frame=None
            return False
        if frame is self.main_proof_frame:return False
        if context!=self.main_proof_key:
            self.main_proof_key=context;self.main_proof_since=now;self.main_proof_seen=1
        else:self.main_proof_seen+=1
        self.main_proof_frame=frame
        return self.main_proof_seen>=3 and now-self.main_proof_since>=3

    def _finish(self,period):
        self.done=self.claim_verified and self.started_period==period
        self.cycles+=int(self.done);self.phase='idle'
        self.event('PASS每日奖励：确认已领满并返回主画面' if self.done else
                   'PASS每日奖励：已返回主画面，未确认领满，保留每日记录并按间隔再检查')

    def count(self,frame):
        if not self.daily_page(frame):return None
        views=[]
        for x,y,X,Y in ((307,407,411,446),(310,409,408,443)):
            ink=frame[y:Y,x:X].min(axis=2)>=225
            views.append(np.repeat(np.where(ink,0,255).astype('uint8')[:,:,None],3,axis=2))
        votes=read_numeric_views(self.reader,views)
        if (any(v.value is None or not 0<=v.value<=1000000 or v.confidence<.97 for v in votes)
                or votes[0].value!=votes[1].value):return None
        return votes[0].value

    def ready(self,frame):
        return self.daily_page(frame) and self._fixed(frame,'daily-ready',(142,480,275,582),.96)

    def claimed(self,frame):
        if not self.daily_page(frame):return False
        ref=self.assets['daily-claimed'][507:553,169:238]
        for offset in (0,120,240,360):
            crop=frame[503+offset:557+offset,165:242]
            scores=cv2.matchTemplate(cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY),cv2.cvtColor(ref,cv2.COLOR_RGB2GRAY),
                                    cv2.TM_CCOEFF_NORMED,mask=self.tick_mask)
            _,score,_,(x,y)=cv2.minMaxLoc(np.where(np.isfinite(scores),scores,-1))
            hsv=cv2.cvtColor(crop[y:y+46,x:x+69],cv2.COLOR_RGB2HSV)
            green=(hsv[:,:,0]>=35)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>=90)&(hsv[:,:,2]>=120)
            if score<.95 or int((green&self.tick_ink).sum())<int(self.tick_ink.sum()*.8):return False
        return True

    @staticmethod
    def context(frame):
        if frame.shape!=(1280,720,3):return None
        rank=frame[10:62,72:136].min(axis=2)>=130
        if int(rank.sum())<50:return None
        badge=frame[45:71,40:68].astype(np.int16)
        world=(badge[:,:,1]-badge[:,:,0]>=10)&(badge[:,:,2]-badge[:,:,0]>=15)&(badge.max(axis=2)>=90)
        return hashlib.sha256(rank.tobytes()+world.tobytes()).hexdigest()

    def _stable(self,frame,label):
        context=self.context(frame)
        key=(label,context) if context else None
        self.seen=self.seen+1 if key and key==self.key else 1 if key else 0
        self.key=key
        return self.seen>=3

    def validate(self,action,frame,context,period):
        if (not self.active or time.monotonic()>=self.deadline or period!=self.started_period
                or self.context(frame)!=context):return False
        if action=='open':return self.entry(frame) and not self.attempts.get('open')
        if action=='tab':return self.panel(frame) and not self.attempts.get('tab')
        if action=='close':return self.panel(frame) and not self.attempts.get('close')
        count=self.count(frame) if action=='claim' else None
        return bool(action=='claim' and self.claim_allowed and not self.claim_issued
                    and count is not None and count>=120 and self.ready(frame))

    def reserve(self,action):
        if self.attempts.get(action):return False
        self.attempts[action]=1
        if action=='claim':self.claim_issued=True
        return True

    def _click(self,action,xy,frame):
        if self.action(action,*xy,self.context(frame)) is not False:
            self.seen=0;self.key=None
            self.last_action_at=time.monotonic()
            self.phase={'open':'page','tab':'page','claim':'claimed','close':'return'}[action]

    def suspend(self,seconds):
        if self.active:self.deadline+=max(0,seconds)

    def step(self,frame,*,due,period,main_ready):
        self.done=None
        if not self.active:
            if not due or not (self.panel(frame) or self.entry(frame)):return False
            self.phase='page' if self.panel(frame) else 'open'
            self.last_action_at=time.monotonic()
            self.deadline=self.last_action_at+self.LIMIT;self.started_period=period
            self.claim_allowed=True;self.claim_issued=self.claim_verified=False;self.attempts={}
            self.seen=0;self.key=None
            self.main_proof_key=None;self.main_proof_since=None;self.main_proof_seen=0
            self.main_proof_frame=None
            self.event('PASS每日奖励：已到重置后两小时的检查时间，核对120条鱼及免费奖励')
        if time.monotonic()>=self.deadline:
            raise RuntimeError('PASS每日奖励60秒内未确认返回，已停止并保留每日完成记录')
        if period!=self.started_period:self.claim_allowed=False
        if self.phase=='open':
            if self.entry(frame) and self._stable(frame,'entry'):self._click('open',(654,188),frame)
        elif self.phase=='return':
            if main_ready and not self.panel(frame) and self._stable(frame,'return'):
                self._finish(period)
        elif self.panel(frame):
            self._main_return_confirmed(frame,False,time.monotonic())
            if not self.daily_page(frame):
                if self._stable(frame,'tab'):self._click('tab',(292,217),frame)
            else:
                count=self.count(frame);self.fish_count=count
                if count is None:
                    # The real two-digit 17 counter is below the strict OCR
                    # threshold. An unknown count cannot authorize a claim,
                    # but a stable, identified daily page can safely close.
                    # Keep the daily entry incomplete and its normal retry.
                    if self._stable(frame,'unreadable-close'):
                        self.event('PASS每日奖励：进度未能高置信度读取，安全关闭后按原间隔重查')
                        self._click('close',(637,280),frame)
                    return True
                if count>=120 and self.claimed(frame):
                    self.claim_verified=True
                    if self._stable(frame,'close'):self._click('close',(637,280),frame)
                elif count<120 or not self.claim_allowed:
                    if self._stable(frame,'incomplete-close'):self._click('close',(637,280),frame)
                elif self.ready(frame) and not self.claim_issued:
                    if self._stable(frame,'claim'):self._click('claim',(205,530),frame)
        elif self.phase in ('page','claimed'):
            now=time.monotonic()
            visible=now-self.last_action_at>=3 and self.main_visible(frame)
            if self._main_return_confirmed(frame,visible,now):
                # An interrupted/failed entry is not a claim. This path only
                # releases the menu owner and schedules the normal retry.
                self._finish(period)
        return True
