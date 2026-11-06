"""Write-ahead recovery of the first task in one *verified* new world.

The normal StoneFarm controller remains the only input owner. This independent
record keeps its exact committed phase and spent inputs if its main save fails.
It never allocates a new deadline or authorizes another cast after expiration.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

from drift.core.persistence import atomic_write_json,atomic_write_bytes,read_bytes,record_lock
from drift.core.journal import archive_bytes


def identity(record,serial):
    if (record.get('reason')!='cycle' or record.get('rank')!=40
            or record.get('entry_rank')!=0 or type(record.get('from_stage')) is not int
            or record.get('to_stage')!=record['from_stage']+1):return None
    entered=record.get('entered_verified_at')
    if type(entered) not in (int,float) or not math.isfinite(entered):return None
    receipts=[r for r in record.get('issued_inputs',[]) if r.get('kind')=='enter-confirm' and r.get('input_id')]
    if not receipts:return None
    value=[serial,record.get('started_at'),record['from_stage'],record['to_stage'],entered,receipts[-1]['input_id']]
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def payload(record):
    fields={'reason','rank','entry_rank','from_stage','to_stage','started_at','deadline','phase_deadline',
            'phase','pending','entered_verified_at','verified_gate_stage','attempts','issued_inputs','last_input',
            'fishing_recovery','selected_fishing_verified_at'}
    return deepcopy({k:v for k,v in record.items() if k in fields or k.startswith('first_')})


class EntryRecoveryJournal:
    def __init__(self,root,serial,event=lambda _:None,clock=time.time):
        self.root,self.serial,self.event,self.clock=Path(root),serial,event,clock
        self.path=self.root/'entry_recovery_state.json'
        self.saved=None;self.raw=None;self.last_result=None
        if self.path.exists():
            self.raw=read_bytes(self.path);self.saved=json.loads(self.raw.decode('utf-8-sig'))
            if (self.saved.get('version')!=1 or self.saved.get('serial')!=serial
                    or type(self.saved.get('pending')) is not bool):
                raise ValueError('新轮启动记录无效，保留原记录并停止输入')

    def capture(self,controller):
        source=controller.journal;key=identity(source,self.serial)
        if key is None:return
        first=str(source.get('phase','')).startswith('first_')
        if not first and not (self.saved and self.saved.get('entry_id')==key):return
        before=read_bytes(controller.path) if controller.path.exists() else None
        state={'version':1,'serial':self.serial,'pending':bool(source.get('pending')),
               'entry_id':key,'saved_at':self.clock(),'journal':payload(source),
               'source_before_sha256':hashlib.sha256(before).hexdigest() if before is not None else None,
               'requires_fresh_visual_validation':True}
        # A new entry archives the exact preceding record, including every used input.
        if self.saved and self.saved.get('entry_id')!=key and self.raw is not None:
            digest=hashlib.sha256(self.raw).hexdigest()
            archive=self.root/'logs/entry-recovery-retired'/(digest+'.json')
            archive_bytes(archive,self.raw,'新轮启动')
        atomic_write_json(self.path,state,expected=self.raw)
        self.saved,self.raw=state,read_bytes(self.path)

    def reconcile(self,controller):
        record=controller.journal;saved=self.saved
        if (not saved or not record.get('pending') or saved.get('entry_id')!=identity(record,self.serial)):
            return 'none'
        committed=saved.get('journal',{});now=self.clock()
        old_raw=read_bytes(controller.path)
        # Only a failed main-file commit can restore a forward phase. A normal
        # later main save, manual edits or another entry never match this hash.
        failed_commit=hashlib.sha256(old_raw).hexdigest()==saved.get('source_before_sha256')
        deadline=min(record.get('deadline',0),committed.get('deadline',0))
        if not saved.get('pending') or not failed_commit or now>=deadline:
            self.last_result='observe_only';return self.last_result
        phase=committed.get('phase','')
        phase_limit=committed.get('first_deadline',0)
        if phase=='first_close':phase_limit=committed.get('first_travel_deadline',phase_limit)
        elif phase=='first_idle':phase_limit=committed.get('first_cast_deadline',phase_limit)
        elif phase=='first_verify':phase_limit=committed.get('first_observation_deadline',phase_limit)
        if (not phase.startswith('first_') or now>=min(committed.get('phase_deadline',0),phase_limit)
                or saved.get('saved_at',float('inf'))>record.get('phase_deadline',0)):
            self.last_result='observe_only';return self.last_result
        updated=deepcopy(record)
        for k,v in committed.items():
            if k.startswith('first_') or k in ('phase','phase_deadline','fishing_recovery','last_input','issued_inputs',
                                               'entered_verified_at','verified_gate_stage'):
                updated[k]=deepcopy(v)
        # Reservations are monotone, including an uncertain/failed native DOWN.
        used=updated.setdefault('attempts',{})
        for k,v in committed.get('attempts',{}).items():used[k]=max(used.get(k,0),v)
        updated['deadline']=deadline
        for k in ('first_deadline','first_travel_deadline','first_cast_deadline','first_observation_deadline'):
            if k in record and k in updated:updated[k]=min(record[k],updated[k])
        digest=hashlib.sha256(old_raw).hexdigest()
        archive=self.root/'logs/entry-recovery-retired'/(digest+'-main.json')
        archive_bytes(archive,old_raw,'新轮启动')
        updated['entry_reconciliation']={'source_sha256':digest,'entry_id':saved['entry_id'],
            'ledger_sha256':hashlib.sha256(self.raw).hexdigest(),'at':now,'input_replayed':False}
        atomic_write_json(controller.path,updated,expected=old_raw)
        controller.journal,controller.phase=updated,updated['phase']
        self.last_result='resume_existing_phase'
        self.event('新轮启动：独立记录恢复已提交的阶段与已用次数；保留原截止，仍须新图核对，不追加抛竿次数')
        return self.last_result

    def status(self):
        saved=self.saved or {};record=saved.get('journal',{})
        return {'entry_id':saved.get('entry_id'),'pending':saved.get('pending',False),
                'phase':record.get('phase'),'deadline':record.get('deadline'),
                'attempts':record.get('attempts',{}),'reconciliation':self.last_result}


def install(cls):
    if getattr(cls,'_entry_record_installed',False):return
    previous=cls._save
    def _save(self):
        journal=getattr(self,'entry_recovery',None)
        if journal is not None:journal.capture(self)
        return previous(self)
    cls._save=_save;cls._entry_record_installed=True


def attach(controller,serial):
    # Read-only guild/cruise probes never attach this writer.
    journal=EntryRecoveryJournal(controller.root,serial,controller.event)
    journal.reconcile(controller)
    controller.entry_recovery=journal
    return controller
