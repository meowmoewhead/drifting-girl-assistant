"""Durable interruption context for every task, separate from input journals.

Authority journals remain authoritative. Volatile controllers can resume only
their existing unexpired cycle; expired/uncertain work is observed, not renewed.
"""
from copy import deepcopy
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import uuid

from drift.core.persistence import atomic_write_json,read_bytes
from drift.core.journal import archive_bytes
from drift.core.continuity import FreshConfirmation

TIME_FIELDS={'started_at','phase_since','deadline','next_at','next_action_at','next_check_at','step_deadline'}
FIELDS={'phase','state','started_at','phase_since','deadline','next_at','next_action_at','next_check_at',
        'step_deadline','page','index','swipes','holds','passes','cycles','pass_holds','last_weapon',
        'interrupted','attempts','period','started_period','enhance_attempts','close_attempts',
        'opening_attempts','attempted','capacity_before','capacity_after','pending','login_clicked',
        'recoveries','awaiting_self_exit'}


def classify(record,now):
    if not isinstance(record,dict):return 'new'
    if not record.get('pending'):return 'finished'
    deadlines=[v for k,v in record.items() if k.endswith('deadline') and type(v) in (int,float)]
    if deadlines and now>=min(deadlines):return 'observe_only'
    last=record.get('last_input',{})
    if last.get('state') in ('reserved','uncertain'):
        return 'verify_issued_input'
    return 'resume_existing_window'


class WorkflowContext:
    def __init__(self,root,serial,event=lambda _:None,wall=time.time,monotonic=time.monotonic):
        self.root,self.serial,self.event=Path(root),serial,event
        self.wall,self.monotonic=wall,monotonic
        self.clock_offset=wall()-monotonic()
        self.observed_at=wall()
        self.path=self.root/'workflow_context_state.json';self.raw=None
        self.data={'version':1,'serial':serial,'pending':False,'tasks':{},'stage':None}
        self.attached=set();self.blocked={}
        if self.path.exists():
            self.raw=read_bytes(self.path);self.data=json.loads(self.raw.decode('utf-8-sig'))
            if (self.data.get('version')!=1 or self.data.get('serial')!=serial
                    or not isinstance(self.data.get('tasks'),dict)):
                raise ValueError('任务接续上下文无效，原记录保留')
        stage=self.data.get('stage') or {}
        self.pending_stage=deepcopy(stage) if stage.get('stage') not in (None,'main') else None
        self.stage_proof=FreshConfirmation(max_gap=8.)
        self.main_proof=FreshConfirmation(max_gap=8.)
        self.stage_result=None
        self.outcomes=[]
        self._recovery_start_processed=False

    def retire_expired_recovery_on_start(self):
        """A new worker run may request a fresh bounded login, never renew one.

        Archive the old context before changing its disposition. Launch receipts,
        including a launch reserved just before a crash, still count toward the
        cross-run 30-minute limit. Ordinary restore/observe never calls this hook.
        """
        if self._recovery_start_processed:return False
        self._recovery_start_processed=True
        saved=self.data['tasks'].get('game_recovery')
        if not saved or not saved.get('pending') or saved.get('authority_journal'):return False
        deadline=saved.get('deadline')
        if type(deadline) not in (int,float) or not self.wall()>=deadline:return False
        values=saved.get('values') or {}
        stamps=[stamp for stamp in values.get('restarts_epoch',[]) if type(stamp) in (int,float)]
        for key,receipt in (saved.get('spent') or {}).items():
            stamp=receipt.get('at') if isinstance(receipt,dict) else None
            if key.startswith('launch:') and type(stamp) in (int,float):
                if all(abs(stamp-old)>.01 for old in stamps):stamps.append(stamp)
        self._archive()
        archive=hashlib.sha256(self.raw).hexdigest()
        retired={'cycle_id':saved.get('cycle_id'),'deadline':deadline,
                 'spent':deepcopy(saved.get('spent',{})),
                 'at':self.wall(),'archive':archive,'reason':'new_worker_start'}
        self.data.setdefault('recovery_retirements',[]).append(retired)
        # Older summaries remain in the immutable full-context archives.
        self.data['recovery_retirements']=self.data['recovery_retirements'][-16:]
        saved['pending']=False
        saved['disposition']='retired_on_new_run'
        saved['values']={**values,'restarts_epoch':sorted(stamps)}
        self.data['pending']=any(task.get('pending') for task in self.data['tasks'].values()) or bool(self.pending_stage)
        self.save()
        self.event('游戏恢复：新一轮启动已归档过期登录窗口，保留原截止、输入记录和30分钟拉起次数；重新确认画面后才建立新窗口')
        return True

    def save(self):
        atomic_write_json(self.path,self.data,expected=self.raw)
        self.raw=read_bytes(self.path)

    def _archive(self):
        if self.raw is None:return
        digest=hashlib.sha256(self.raw).hexdigest()
        archive_bytes(self.root/'logs/workflow-context-retired'/(digest+'.json'),self.raw,'任务接续')

    def pack(self,task):
        values={}
        for key in FIELDS:
            value=getattr(task,key,None)
            if type(value) not in (str,int,float,bool,type(None),dict,list,tuple):continue
            try:encoded=json.dumps(value)
            except (TypeError,ValueError):continue
            if len(encoded)>32768:continue
            if key in TIME_FIELDS and type(value) in (int,float) and 0<value<100000000:
                values[key]={'epoch':round(self.clock_offset+value,6),'clock':'monotonic'}
            else:values[key]=deepcopy(value)
        restarts=getattr(task,'restarts',None)
        if isinstance(restarts,deque):
            values['restarts_epoch']=[round(self.clock_offset+v,6) for v in restarts]
        return values

    def capture(self,tasks,*,stage=None,popup=False,budget=None):
        self.observed_at=self.wall()
        changed=False
        for name,task in tasks.items():
            if task is None:continue
            if name in self.blocked:continue
            is_active=getattr(task,'active',False)
            is_active=bool(is_active() if callable(is_active) else is_active)
            busy=getattr(task,'busy',False);is_active=is_active or bool(busy() if callable(busy) else busy)
            if name=='parallel':is_active=getattr(task,'phase','detect') not in ('detect','play')
            if name=='game_recovery':is_active=getattr(task,'started_at',None) is not None
            previous=self.data['tasks'].get(name,{})
            # Idle controllers do not overwrite a still-unverified interrupted cycle.
            if not is_active and previous.get('pending') and name not in self.attached:continue
            journal=getattr(task,'journal',None)
            authority=bool(isinstance(journal,dict) and journal.get('version')==1)
            values={} if authority else self.pack(task)
            phase=getattr(task,'phase',getattr(task,'state',None))
            payload={'pending':is_active,'authority_journal':authority,'values':values,
                'phase':phase if isinstance(phase,str) else None}
            if authority:
                payload['disposition']=classify(journal,self.wall())
                payload['deadline']=journal.get('deadline')
                payload['attempts_sha256']=hashlib.sha256(json.dumps(journal.get('attempts',{}),sort_keys=True).encode()).hexdigest()
            elif is_active:
                started=values.get('started_at')
                if isinstance(started,dict):payload['deadline']=started['epoch']+180 if name=='equipment_upgrade' else None
                if isinstance(started,dict) and name=='game_recovery':payload['deadline']=started['epoch']+120
                deadline=values.get('deadline')
                if isinstance(deadline,dict):payload['deadline']=deadline['epoch']
                elif type(deadline) in (int,float) and deadline>0:payload['deadline']=deadline
                if payload.get('deadline') is None and isinstance(values.get('phase_since'),dict):
                    payload['deadline']=values['phase_since']['epoch']+float(getattr(task,'STEP_SECONDS',30))
            same_cycle=bool(previous.get('pending') or not is_active)
            payload['cycle_id']=previous.get('cycle_id',uuid.uuid4().hex) if same_cycle else uuid.uuid4().hex
            payload['spent']=deepcopy(previous.get('spent',{})) if same_cycle else {}
            if payload!=previous:self.data['tasks'][name]=payload;changed=True
        context={'stage':stage,'popup':bool(popup),'budget':deepcopy(budget or {})}
        if self.pending_stage is None and self.data.get('stage')!=context:self.data['stage']=context;changed=True
        pending=any(v.get('pending') for v in self.data['tasks'].values()) or bool(popup) or stage not in (None,'main')
        if self.data.get('pending')!=pending:self.data['pending']=pending;changed=True
        if changed:self.save()

    def restore(self,name,task):
        self.attached.add(name)
        saved=self.data['tasks'].get(name)
        if not saved:return 'new'
        values=saved.get('values',{})
        if name=='game_recovery':
            task.restarts=deque(self.monotonic()+stamp-self.wall() for stamp in values.get('restarts_epoch',[]))
        if not saved.get('pending'):return 'new'
        if saved.get('authority_journal'):
            return classify(getattr(task,'journal',{}),self.wall())
        deadline=saved.get('deadline')
        if type(deadline) not in (int,float) or self.wall()>=deadline:
            self.blocked[name]='原窗口已到期，只读等待安全主画面，不续期或重放输入'
            return 'observe_only'
        # Unknown legacy navigation requires its own scene reconstruction, never
        # fabricate omitted image/context fields from a partial primitive snapshot.
        if name not in ('equipment_upgrade','parallel_skills','pass_daily','backpack_sale','game_recovery'):
            self.blocked[name]='待原控制器以新画面核对中断位置'
            return 'observe_only'
        for key,value in values.items():
            if key not in FIELDS or key in ('state','pending'):continue
            if key in TIME_FIELDS and isinstance(value,dict) and value.get('clock')=='monotonic':
                value=self.monotonic()+value['epoch']-self.wall()
            setattr(task,key,deepcopy(value))
        if name=='equipment_upgrade' and saved.get('spent'):
            indices=[int(k.split(':')[1]) for k in saved['spent'] if k.startswith('hold:')]
            old_index=task.index
            task.index=max(task.index,max(indices,default=-1)+1)
            task.holds=max(task.holds,len(indices))
            if task.index>old_index:task.interrupted=True
        if name=='game_recovery':
            spent=saved.get('spent',{})
            launch_at=max([v['at'] for k,v in spent.items() if k.startswith('launch:')],default=0)
            if spent.get('login',{}).get('at',-1)>=launch_at:task.login_clicked=True
            for key,entry in spent.items():
                if key.startswith('launch:'):
                    stamp=self.monotonic()+entry['at']-self.wall()
                    if all(abs(stamp-v)>.01 for v in task.restarts):task.restarts.append(stamp)
            task.title_frames=task.ready_frames=task.missing=0
        # No saved frame/confirmation contributes to a fresh input proof.
        for key,value in (('stable',0),('seen',0),('last_frame',None),('key',None),('since',None)):
            if hasattr(task,key):setattr(task,key,value)
        self.event('任务接续：'+name+' 沿用原阶段、原截止和已用次数，等待新画面核对')
        return 'resume_existing_window'

    def reserve(self,actor,kind,args,tasks,budget):
        if actor=='equipment_upgrade' and actor in self.data['tasks']:
            # A held weapon is spent even if the process dies before holds += 1.
            if len(args)>=8 and tuple(args[:3])==('shell','input','swipe') and int(args[-1])>=5000:
                task=tasks.get(actor);entry=self.data['tasks'][actor]
                entry.setdefault('spent',{})['hold:'+str(task.index)]={'state':'reserved','at':self.wall()}
        if actor=='game_recovery' and actor in self.data['tasks']:
            entry=self.data['tasks'][actor]
            key='launch:'+str(self.wall()) if kind=='launch' else 'login'
            entry.setdefault('spent',{})[key]={'state':'reserved','at':self.wall()}
            if kind=='launch':
                values=entry.setdefault('values',{})
                if values.get('started_at') is None:
                    values['started_at']={'epoch':self.observed_at,'clock':'monotonic'}
                    entry['deadline']=self.observed_at+120
                values['phase']='loading';values['login_clicked']=False
                entry['pending']=True;self.data['pending']=True
        self.data['last_input']={'actor':actor,'kind':kind,'state':'reserved','at':self.wall(),
            'budget':deepcopy(budget or {}),'stage':(self.data.get('stage') or {}).get('stage')}
        self.save()

    def issued(self,token):
        if self.data.get('last_input',{}).get('at')==token:
            self.data['last_input']['state']='issued_only';self.save()

    def release_verified_main(self):
        if not self.blocked:return
        self._archive()
        for name in self.blocked:
            if name in self.data['tasks']:self.data['tasks'][name]['pending']=False
            self.outcomes.append({'feature':name,'outcome':'blocked','detail':self.blocked[name]})
        self.blocked.clear();self.save()

    def reconcile_stage(self,loop,matchers,module,frame,frame_id,*,main=False):
        saved=self.pending_stage
        if not saved:return False
        stage=saved['stage'];budget=saved.get('budget') or {};now=self.wall()
        deadline=budget.get('stage_deadline_epoch')
        last=self.data.get('last_input') or {}
        uncertain=(last.get('stage')==stage and last.get('state') in ('reserved','issued_only','uncertain'))
        candidates=[]
        if type(deadline) in (int,float) and now<deadline:
            for rule,button,offset in matchers.values():
                if rule.get('stage')!=stage or rule.get('interrupt'):continue
                # A result observation can reconcile an issued input. The same
                # uncertain purchase/claim button cannot authorize its replay.
                if uncertain and rule.get('action','click')!='observe':continue
                if module.appear(button,offset=offset,similarity=rule['threshold']):
                    candidates.append(rule['id'])
        if self.stage_proof.observe((stage,tuple(candidates)) if candidates else None,frame_id,self.monotonic()):
            self._archive()
            loop.stage=stage;loop.stage_trigger=budget.get('stage_trigger')
            loop.stage_deadline=self.monotonic()+deadline-now
            for key in ('stage_started_at','stage_ready_at'):
                value=budget.get(key+'_epoch')
                if type(value) in (int,float):setattr(loop,key,self.monotonic()+value-now)
            for key in ('consecutive','nav_clicks','previous','previous_progress','skip_on_timeout','daily_verified'):
                if key in budget:setattr(loop,key,deepcopy(budget[key]))
            for key in ('scheduled_daily','daily_attempt'):
                if budget.get(key) is not None:setattr(loop,key,tuple(budget[key]))
            self.pending_stage=None;self.stage_result='resume_existing_window'
            self.event('任务接续：连续新图确认原界面，沿用原阶段截止与已用次数；不重放不确定输入')
            return True
        if self.main_proof.observe('main' if main else None,frame_id,self.monotonic()):
            self._archive();self.pending_stage=None;self.stage_result='retired_on_verified_main'
            self.event('任务接续：连续新图确认已在安全主画面；旧未完成步骤归档，不补记完成或重放输入')
            self.data['stage']={'stage':'main','popup':False,'budget':{}}
            self.data['pending']=any(t.get('pending') for t in self.data['tasks'].values())
            scheduled=budget.get('scheduled_daily')
            if scheduled:
                self.outcomes.append({'feature':scheduled[0],'interval':scheduled[1],
                    'period':scheduled[2],'outcome':'blocked','detail':'中断步骤未确认完成；实际已回主画面'})
            self.save()
        return False

    def drain_outcomes(self):
        outcomes,self.outcomes=self.outcomes,[]
        return outcomes

    def status(self):
        return {'pending':self.data.get('pending',False),'tasks':{
            name:{k:v for k,v in item.items() if k in ('pending','phase','deadline','disposition','cycle_id')}
            for name,item in self.data['tasks'].items()},'blocked':dict(self.blocked),
            'last_input':self.data.get('last_input'),'pending_stage':self.pending_stage,
            'stage_reconciliation':self.stage_result}
