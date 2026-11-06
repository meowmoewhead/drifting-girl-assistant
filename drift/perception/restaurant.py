"""Restaurant cookbook recognition. Crafting and ingredient shops never qualify."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from drift.perception.digit_ocr import parse_counter

CATEGORIES = ('干货','炖煮','生鱼片','油炸','炒饭','面食','寿司','汤','烧烤')
INGREDIENT_X = (142,255,368,480)


class RestaurantVisual:
    def __init__(self, root, reader):
        folder=Path(root)/'assets/restaurant'
        self.spec=json.loads((folder/'visuals.json').read_text(encoding='utf-8'))
        self.images={}
        for name,spec in self.spec.items():
            image=cv2.imread(str(folder/spec['file']))
            if image is None:raise RuntimeError('餐品强化模板缺失：'+name)
            self.images[name]=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.reader=reader
        self.frame=None;self.cache={}

    def find(self, frame, name, area=None):
        if frame is not self.frame:self.frame=frame;self.cache={}
        key=(name,tuple(area) if area else None)
        if key in self.cache:return self.cache[key]
        if frame.shape!=(1280,720,3):return None
        spec=self.spec[name];x,y,X,Y=area or spec['area']
        source=cv2.cvtColor(frame[y:Y,x:X],cv2.COLOR_RGB2GRAY)
        best=None
        for scale in spec.get('scales',[1.]):
            patch=cv2.resize(self.images[name],None,fx=scale,fy=scale)
            patch=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY);h,w=patch.shape
            if h>Y-y or w>X-x:continue
            scores=np.nan_to_num(cv2.matchTemplate(source,patch,cv2.TM_CCOEFF_NORMED),nan=-1,posinf=-1,neginf=-1)
            _,score,_,(px,py)=cv2.minMaxLoc(scores)
            if score>=spec['threshold'] and (best is None or score>best[2]):
                best=((x+px+w//2,y+py+h//2),(x+px,y+py,w,h),score)
        self.cache[key]=best;return best

    def point(self,frame,name):
        if name=='detail-close' and self.max_detail(frame):
            hit=self.find(frame,'max-detail-close')
            return hit[0] if hit else None
        hit=self.find(frame,name);return hit[0] if hit else None

    def detail(self,frame):
        return bool(self.max_detail(frame) or (
                    all(self.find(frame,k) for k in ('detail-book','detail-level','detail-close'))
                    and float((frame[480:800,210:530].max(axis=2)>210).mean())>.40))

    def max_detail(self,frame):
        # At Lv20 the four ingredient counters disappear and the detail card
        # becomes shorter. Its foreground X is below the atlas' background X.
        return bool(all(self.find(frame,k) for k in
                    ('max-detail-book','max-detail-level','max-detail-close','max-detail-full'))
                    and float((frame[535:750,210:530].max(axis=2)>210).mean())>.55
                    and float((frame[676:691,220:520,1].astype(int)
                               -frame[676:691,220:520,0].astype(int)>30).mean())>.60)

    def max_level(self,frame):
        if not self.max_detail(frame):return False
        if 'max_level' not in self.cache:
            pair=self.reader.read_many([self.binary(frame[672:702,141:177],t) for t in (110,130)])
            self.cache['max_level']=(len(pair)==2 and all(r.value==20 and r.confidence>=.95 for r in pair))
        return self.cache['max_level']

    def max_name_matches(self,frame,reference):
        # Preserve the pre-upgrade full name. Layout changes rasterize its text
        # at a fractional vertical offset; compare fixed strokes, not the art.
        if reference is None or reference.shape!=(30,321,3) or not self.max_level(frame):return False
        source=frame[540:587,204:535]
        for threshold in (110,130):
            def strokes(crop):
                return cv2.GaussianBlur((crop.max(axis=2)<threshold).astype('float32'),(7,7),1.5)
            needle=strokes(reference)
            if np.count_nonzero(needle) < 100:return False
            scores=cv2.matchTemplate(strokes(source),needle,cv2.TM_CCOEFF_NORMED)
            score=float(np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1).max())
            if score<.96:return False
        return True

    def panel(self,frame):
        return bool(self.find(frame,'panel-close') and self.find(frame,'atlas-tab')
                    and self.find(frame,'ingredients-title'))

    def atlas(self,frame):
        return bool(self.panel(frame) and not self.detail(frame) and self.find(frame,'atlas-title')
                    and self.find(frame,'next') and self.find(frame,'prev')
                    and float((frame[200:600,95:620].max(axis=2)>210).mean())>.45)

    def category(self,frame):
        if not self.atlas(frame):return None
        hits=[i for i in range(9) if self.find(frame,'category-'+str(i))]
        return hits[0] if len(hits)==1 else None

    @staticmethod
    def identity(frame):
        # Full fixed dish name, excluding the changing level, arrow and art.
        ink=(frame[480:510,209:530].max(axis=2)<100).astype('uint8')
        return hashlib.sha256(ink.tobytes()).hexdigest() if ink.sum()>25 else None

    @staticmethod
    def binary(crop,threshold,white=False):
        ink=(crop.min(axis=2)>threshold) if white else (crop.max(axis=2)<threshold)
        if white and ink.any():
            y,x=np.where(ink);ink=ink[y.min():y.max()+1,x.min():x.max()+1]
        return np.repeat(np.pad(np.where(ink,0,255).astype('uint8'),3,constant_values=255)[:,:,None],3,axis=2)

    def detail_values(self,frame):
        if not self.detail(frame) or self.max_detail(frame):return None
        if 'values' in self.cache:return self.cache['values']
        crops=[frame[738:768,141:176]]+[frame[687:703,x:x+70] for x in (109,222,335,447)]
        views=[self.binary(crops[0],t) for t in (110,130)]
        views.extend(self.binary(c,t) for c in crops[1:] for t in (80,110,130))
        readings=self.reader.read_many(views)
        level=readings[:2]
        if any(r.value is None or not 1<=r.value<=20 or r.confidence<.95 for r in level) or level[0].value!=level[1].value:return None
        counters=[]
        unresolved=[]
        for i in range(4):
            values=[parse_counter(r.text,r.confidence) for r in readings[2+3*i:5+3*i]]
            # Tiny counter glyphs can lose an antialiased stroke at one cutoff.
            # Require two independent high-confidence decodes; any conflicting
            # high-confidence read invalidates the frame rather than voting it away.
            confident=[r for r in values if r.confidence>=.95 and r.current is not None]
            if (any(r.total not in (8,12,16) for r in confident)
                    or len({(r.current,r.total) for r in confident})>1):return None
            counters.append(confident)
            if len(confident)<2:unresolved.append(i)
        if unresolved:
            # A completed counter's green tick casts a moving shadow on its
            # tiny digits. Recover those strokes at two brighter cutoffs;
            # keep .95 confidence, independent agreement and conflict vetoes.
            extra=self.reader.read_many([self.binary(crops[i+1],t)
                                         for i in unresolved for t in (140,150)])
            if len(extra)!=2*len(unresolved):return None
            for n,i in enumerate(unresolved):
                counters[i].extend(r for reading in extra[2*n:2*n+2]
                    if (r:=parse_counter(reading.text,reading.confidence)).confidence>=.95
                    and r.current is not None)
        if any(len(v)<2 or any(r.total not in (8,12,16) for r in v)
               or len({(r.current,r.total) for r in v})!=1 for v in counters):return None
        counters=[(v[0].current,v[0].total) for v in counters]
        if len({v[1] for v in counters})!=1:return None
        result=dict(identity=self.identity(frame),level=level[0].value,counters=counters)
        self.cache['values']=result;return result

    def ingredient_ready(self,frame,index,values):
        return (values is not None and values['level']<20 and 0<=index<4
                and values['counters'][index][0]<values['counters'][index][1]
                and self.find(frame,'ingredient-'+str(index)) is not None
                and self.available_arrow(frame,index))

    @staticmethod
    def available_arrow(frame,index):
        # The four gold availability arrows bob vertically; the border below
        # them does not. Recognize the full gold downward triangle, excluding
        # that border and the ingredient art below it.
        x=INGREDIENT_X[index]
        hsv=cv2.cvtColor(frame[580:625,x-20:x+20],cv2.COLOR_RGB2HSV)
        mask=cv2.inRange(hsv,(15,100,80),(40,255,255))
        for contour in cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
            X,Y,w,h=cv2.boundingRect(contour)
            if not(20<=w<=32 and 8<=h<=16 and 90<=cv2.contourArea(contour)<=230):continue
            points=cv2.approxPolyDP(contour,.08*cv2.arcLength(contour,True),True).reshape(-1,2)
            if len(points)!=3:continue
            points=points[np.argsort(points[:,1])]
            if (abs(int(points[0,1])-int(points[1,1]))<=3
                    and points[2,1]-points[1,1]>=6 and abs(int(points[2,0])-20)<=4):return True
        return False

    def upgrade_ready(self,frame,values):
        return bool(values and values['level']<20
                    and all(a==b for a,b in values['counters'])
                    and self.find(frame,'upgrade-ready')
                    and float((frame[744:762,590:619,1].astype(int)-frame[744:762,590:619,0].astype(int)>30).mean())>.15)

    def cards(self,frame,*,strict_order=False):
        from drift.perception.restaurant_cards import cards
        return cards(self,frame,strict_order=strict_order)

    @staticmethod
    def art_matches(left,right):
        if left is None or right is None or left.shape!=right.shape:return False
        score=float(cv2.matchTemplate(left,right,cv2.TM_CCOEFF_NORMED)[0,0])
        return bool(np.isfinite(score) and score>=.92)

    @staticmethod
    def list_key(frame):
        # Used only to prove that a bounded scroll reached the end.
        image=cv2.cvtColor(frame[335:485,107:620],cv2.COLOR_RGB2GRAY)
        return cv2.resize(image,(257,75))


# Reviewed source-sprite recognition; controller guards and input logic unchanged.
from drift.perception.original_templates import install_original_templates as _install_original_templates
_install_original_templates(RestaurantVisual, 'restaurant', 'find')

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
    cards=[]
    for i,x in enumerate(positions):
        pair=values[2*i:2*i+2]
        valid=(len(pair)==2 and all(r.value is not None and 1<=r.value<=20 and r.confidence>=.95 for r in pair)
               and pair[0].value==pair[1].value)
        if not valid and not strict_order:continue
        # Art is stable while a card's level and completion tint can change.
        art=frame[364:428,x+3:x+88]
        # Keep an unread slot as an explicit barrier for sequential runs;
        # dropping it would allow a later, readable card to overtake it.
        cards.append(dict(point=(x+42,398),level=pair[0].value if valid else None,art=art.copy()))
    return cards

RestaurantVisual.cards = cards

def detail_values(self,frame):
    if not self.detail(frame) or self.max_detail(frame):return None
    if 'values' in self.cache:return self.cache['values']
    crops=[frame[738:768,141:176]]+[frame[687:703,x:x+70] for x in (109,222,335,447)]
    views=[self.binary(crops[0],t) for t in (110,130)]
    views.extend(self.binary(c,t) for c in crops[1:] for t in (80,110,130))
    readings=self.reader.read_many(views)
    level=readings[:2]
    if any(r.value is None or not 1<=r.value<=20 or r.confidence<.95 for r in level) or level[0].value!=level[1].value:return None
    counters=[]
    unresolved=[]
    for i in range(4):
        values=[parse_counter(r.text,r.confidence) for r in readings[2+3*i:5+3*i]]
        # Tiny counter glyphs can lose an antialiased stroke at one cutoff.
        # Require two independent high-confidence decodes; any conflicting
        # high-confidence read invalidates the frame rather than voting it away.
        confident=[r for r in values if r.confidence>=.95 and r.current is not None]
        if (any(r.total not in (8,12,16) for r in confident)
                or len({(r.current,r.total) for r in confident})>1):return None
        counters.append(confident)
        if len(confident)<2:unresolved.append(i)
    if unresolved:
        # A completed counter's green tick casts a moving shadow on its
        # tiny digits. Recover those strokes at two brighter cutoffs;
        # keep .95 confidence, independent agreement and conflict vetoes.
        extra=self.reader.read_many([self.binary(crops[i+1],t)
                                     for i in unresolved for t in (140,150)])
        if len(extra)!=2*len(unresolved):return None
        for n,i in enumerate(unresolved):
            counters[i].extend(r for reading in extra[2*n:2*n+2]
                if (r:=parse_counter(reading.text,reading.confidence)).confidence>=.95
                and r.current is not None)
    if any(len(v)<2 or any(r.total not in (8,12,16) for r in v)
           or len({(r.current,r.total) for r in v})!=1 for v in counters):return None
    counters=[(v[0].current,v[0].total) for v in counters]
    if len({v[1] for v in counters})!=1:return None
    result=dict(identity=self.identity(frame),level=level[0].value,counters=counters)
    self.cache['values']=result;return result

RestaurantVisual.detail_values = detail_values


# Independent numeral consensus; keep every other deployed perception method.
from drift.perception.restaurant_cards import cards as _restaurant_cards
RestaurantVisual.cards = _restaurant_cards


# Reviewed source-sprite recognition; controller guards and input logic unchanged.
from drift.perception.original_templates import install_original_templates as _install_original_templates
_install_original_templates(RestaurantVisual, 'restaurant', 'find')
