"""Fixed card outlines and independent high-confidence numeral consensus."""
import cv2
import numpy as np


def cards(self,frame,*,strict_order=False):
    if not self.atlas(frame):return []
    # Max-level cards are green and other cards are beige. Match only the
    # fixed dark Lv outline, never the changing card background.
    def strokes(image):
        return cv2.GaussianBlur(((image.max(axis=2)<100)*255).astype('uint8'),(3,3),.7)
    source=strokes(frame[447:486,106:623])
    results=[]
    for name in ('level-label','level-label-half'):
        template=strokes(self.images[name])
        results.append(cv2.matchTemplate(source,template,cv2.TM_CCOEFF_NORMED))
    scores=np.nan_to_num(np.maximum(*results),nan=-1)
    positions=[]
    for _ in range(6):
        _,score,_,(x,y)=cv2.minMaxLoc(scores)
        if score<.94:break
        x+=106
        # Require the full dish artwork and numeral crop. The last card's
        # decorative border can touch x623 while its art ends at x615.
        if 115<=x and x+90<=623:positions.append(x)
        scores[:,max(0,x-106-50):x-106+51]=-1
    positions.sort();views=[]
    for x in positions:
        views.extend(self.binary(frame[453:483,x+37:x+81],t,True) for t in (215,230))
    values=self.reader.read_many(views) if views else []
    def consensus(readings):
        confident=[r for r in readings if r.confidence>=.95]
        if (len(confident)<2 or any(r.value is None or not 1<=r.value<=20 for r in confident)
                or len({r.value for r in confident})!=1):return None
        return confident[0].value
    pairs=[values[2*i:2*i+2] for i in range(len(positions))]
    unresolved=[i for i,pair in enumerate(pairs) if consensus(pair) is None]
    if unresolved:
        extra=self.reader.read_many([self.binary(frame[453:483,positions[i]+37:positions[i]+81],t,True)
                                    for i in unresolved for t in (205,220,240)])
        if len(extra)==3*len(unresolved):
            for n,i in enumerate(unresolved):pairs[i].extend(extra[3*n:3*n+3])
    cards=[]
    for i,x in enumerate(positions):
        level=consensus(pairs[i])
        valid=level is not None
        if not valid and not strict_order:continue
        # Art is stable while a card's level and completion tint can change.
        art=frame[364:428,x+3:x+88]
        # Keep an unread slot as an explicit barrier for sequential runs;
        # dropping it would allow a later, readable card to overtake it.
        cards.append(dict(point=(x+42,398),level=level,art=art.copy()))
    return cards

