"""Shared fresh-frame confirmation and conservative departure reconciliation."""
from copy import deepcopy


class FreshConfirmation:
    def __init__(self, *, frames=3, seconds=3., max_gap=6.):
        self.frames,self.seconds,self.max_gap=frames,seconds,max_gap
        self.reset()

    def reset(self):
        self.key=self.first=self.last=self.frame_id=None
        self.count=0

    def observe(self, key, frame_id, now):
        if key is None or frame_id is None:
            self.reset();return False
        if frame_id==self.frame_id:return False
        if key!=self.key or self.last is None or now-self.last>self.max_gap:
            self.key,self.first,self.count=key,now,0
        self.frame_id,self.last=frame_id,now
        self.count+=1
        return self.count>=self.frames and now-self.first>=self.seconds


def departure_refresh(origin, farm, *, valid, restored_at=None, resume_pending=False):
    """A known running departure replaces stale home; unknown pixels never do."""
    return bool(valid(origin) and (not valid(farm) or restored_at is not None
        or not resume_pending or ((origin['stage'],origin['rank'])!=(farm['stage'],farm['rank'])
                                 and origin['route'] is True and origin.get('fishing') is True)))


def reconcile_late_departure(journal, origin, *, valid):
    """Repair only an unissued departure, never an already-owned return trip.

    This handles legacy pending records created before the departure was known.
    It deliberately leaves the original deadline and all reservations intact.
    """
    if journal.get('return_only') or not valid(origin):return False
    attempts=journal.get('attempts',{})
    if any(v for k,v in attempts.items() if k not in ('stop',)):return False
    target=journal.get('target') or journal.get('farm_origin')
    if not departure_refresh(origin,target,valid=valid,
            restored_at=journal.get('farm_restored_at'),resume_pending=journal.get('resume_pending',False)):
        return False
    journal['departure_reconciliation']={'previous_target':deepcopy(target),
        'actual_origin':deepcopy(origin),'before_departure':True}
    journal['target']=dict(deepcopy(origin),settings=None)
    journal['farm_origin']=deepcopy(journal['target'])
    journal['capture_target_settings']=True
    journal['farm_restored_at']=None
    return True
