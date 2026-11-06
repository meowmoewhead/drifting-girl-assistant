"""Strict outlined RANK fallback for the recognized native cruise menu."""
from functools import wraps

import cv2
import numpy as np

from drift.perception.digit_ocr import read_numeric_views


def _strong(readings):
    return {r.value for r in readings if type(r.value) is int and r.confidence >= .95}


def outlined_menu_rank(owner, frame, previous=()):
    if (frame.shape != (1280,720,3) or not owner.sailing
            or not owner.sailing.menu(frame) or not owner._parallel(frame)):
        return None
    # Re-read both complete fixed crops: an older caller's diagnostic may
    # contain only its final group, so it cannot replace these conflict checks.
    binary=[]
    for x,y,X,Y in ((71,9,136,59),(80,9,130,59),(70,8,137,60)):
        for threshold in (110,130):
            ink=np.all(frame[y:Y,x:X]>=threshold,axis=2)
            binary.append(np.repeat(np.where(ink,0,255).astype('uint8')[:,:,None],3,axis=2))
    prior=list(previous)+list(read_numeric_views(owner.reader,binary))
    if len(_strong(prior))>1:
        return None
    crop=frame[9:59,71:136]
    boxes=[]
    for threshold in (130,140):
        ink=np.all(crop>=threshold,axis=2).astype('uint8')
        count,labels,stats,_=cv2.connectedComponentsWithStats(ink,8)
        components=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=30
                    and stats[i,cv2.CC_STAT_HEIGHT]>=18]
        if not 1<=len(components)<=2:
            return None
        x,y,w,h=cv2.boundingRect(np.isin(labels,components).astype('uint8'))
        if x<=1 or y<=1 or x+w>=crop.shape[1]-1 or y+h>=crop.shape[0]-1:
            return None
        boxes.append((x,y,w,h))
    # Existing outlined 5 needs wider margins. Single 7 needs compact framing.
    # Neither profile may override a conflicting confident reading elsewhere.
    profiles=[]
    for margins in (((8,9),(8,9)),((3,4),(5,6))):
        views=[]
        for (x,y,w,h),padding in zip(boxes,margins):
            for margin in padding:
                views.append(crop[max(0,y-margin):min(crop.shape[0],y+h+margin),
                                  max(0,x-margin):min(crop.shape[1],x+w+margin)])
        profiles.append(views)
    decoded=list(read_numeric_views(owner.reader,[v for group in profiles for v in group]))
    if len(decoded)!=8 or len(_strong(prior+decoded))!=1:
        return None
    for index,views in enumerate(profiles):
        group=decoded[index*4:index*4+4]
        if (len({(v.shape,v.tobytes()) for v in views})!=4
                or any(type(r.value) is not int or not 0<=r.value<=40 or r.confidence<.95 for r in group)
                or len({r.value for r in group})!=1):
            continue
        value=group[0].value
        if _strong(prior+decoded)!={value}:
            return None
        owner.menu_rank_probe=dict(method='outlined_digit',profile='wide' if index==0 else 'compact',
            views=[dict(value=r.value,confidence=round(r.confidence,4)) for r in group])
        return value
    return None


def install(cls):
    original=cls._sailing_menu_rank
    if getattr(original,'_outlined_menu_fallback',False):
        return
    @wraps(original)
    def read(self,frame):
        old_probe=getattr(self,'menu_rank_probe',None)
        value=original(self,frame)
        if value is not None:
            return value
        from types import SimpleNamespace
        probe=getattr(self,'menu_rank_probe',None)
        probe=probe if isinstance(probe,dict) and probe is not old_probe else {}
        # An existing outlined decoder already examined these raw strokes and
        # rejected a conflict/low view. Do not vote its explicit rejection away.
        if probe.get('method') in ('outlined_digit','outlined_digit_unverified'):
            return None
        previous=[SimpleNamespace(value=r.get('value'),confidence=r.get('confidence',0))
                  for r in probe.get('views',()) if isinstance(r,dict)]
        return outlined_menu_rank(self,frame,previous)
    read._outlined_menu_fallback=True
    cls._sailing_menu_rank=read
