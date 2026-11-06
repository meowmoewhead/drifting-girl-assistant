"""Adjust the draggable fishing level and confirm a real fishing start."""
from pathlib import Path
import hashlib
import time
import cv2
import numpy as np
from drift.tasks.boss_defense import BossDefense
from drift.tasks.meter_limits import MeterWorkflowLimit
from drift.perception.digit_ocr import read_numeric_views


class FishingMeter:
    LEVEL_GAP = 10
    TOP, BOTTOM = 936, 1085
    HOLD_MS = 1500
    DRAG_MS = 900
    IDLE_SECONDS = 3
    WAKE_SECONDS = 15
    ENDPOINT_SLOP = 8
    MAX_ENDPOINT_SLOP = 1
    MAX_DRAG_OVERSHOOT = 16
    RECALIBRATE_GAP = 20

    def __init__(self, root, training, action, event):
        self.training, self.action, self.event = training, action, event
        self.assets = {}
        for name in ('sleep', 'hook', 'waiting-dot', 'waiting-dot-filled', 'cancel-cast', 'port-anchor', 'bite-warning', 'bite-warning-night'):
            data = cv2.imread(str(Path(root)/'assets/fishing-meter'/(name+'.png')))
            if data is None:
                raise RuntimeError('钓鱼计量模板缺失：'+name)
            self.assets[name] = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        auto = cv2.imread(str(Path(root)/'assets/training-sailing/main-auto-sailing.png'))
        if auto is None:
            raise RuntimeError('钓鱼自适应勾选模板缺失')
        self.adaptive_auto_template = cv2.cvtColor(auto,cv2.COLOR_BGR2RGB)
        self.port_assets = {}
        for name in ('header-port',):
            data = cv2.imread(str(Path(root)/'assets/backpack'/(name+'.png')))
            if data is None:
                raise RuntimeError('钓鱼港口识别模板缺失：'+name)
            self.port_assets[name] = cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
        self.phase = 'detect'
        self.current = self.player = self.goal = None
        self.stable, self.key = 0, None
        self.next_action_at = self.since = 0.
        self.drags = self.starts = 0
        self.wakes = 0
        self.cancels = 0
        # 取消 X 只在抛竿后约 1.5~2 秒内出现(2026-10-23 实测:50 帧/70 秒 0 次,
        # 另一段 40 帧里只有连续 2 帧命中),所以「看见过」要留一个短记忆,
        # 不能只看当前帧。
        self.confirmed_starts = 0
        self.start_fill_verified = False
        self.start_anchor = None
        self.start_seen_at = self.start_last_seen = None
        self.start_seen = 0
        self.start_last_frame = None
        self.start_proof = None
        self.start_defense = BossDefense(root, training, None, event)
        self.start_route_assets = {}
        for name in ('auto-label', 'inactive-face', 'active-arrow'):
            raw = cv2.imread(str(Path(root)/'assets/training-sailing'/(name+'.png')))
            if raw is None:
                raise RuntimeError('启动验证缺少停航模板：'+name)
            self.start_route_assets[name] = cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
        self.saved_player_gap = False
        self.last_warning_at = -60.
        self.idle_since = None
        self.filled = False
        self.dial_state = 'unknown'
        self.range_cache = self.training.state.get('fishing_meter_limits', {})
        self._world_frame = None
        self._world_rank = None
        self._world_ok = False
        self._rank_reference = None
        warning = cv2.cvtColor(self.assets['bite-warning'],cv2.COLOR_RGB2HSV)
        colors = warning[:,:,0][(warning[:,:,1]>=100)&(warning[:,:,2]>=170)]
        self.warning_hue = float(np.median(colors))
        night=self.assets['bite-warning-night']
        night_hsv=cv2.cvtColor(night,cv2.COLOR_RGB2HSV)
        background=np.median(np.concatenate((night_hsv[:,:2,2],night_hsv[:,-2:,2]),axis=1),axis=1)
        night_ink=((abs(night_hsv[:,:,0].astype(float)-self.warning_hue)<=10)
                   &(night_hsv[:,:,1]>=70)&(night_hsv[:,:,2]>=background[:,None]+12))
        self.night_warning_mask=cv2.dilate(night_ink.astype('uint8'),np.ones((3,3),np.uint8))
        self.waiting_shapes = []
        for name, minimum in (('waiting-dot',150),('waiting-dot-filled',100)):
            template = self.assets[name]
            hsv = cv2.cvtColor(template,cv2.COLOR_RGB2HSV)
            ink = ((hsv[:,:,0]>=17)&(hsv[:,:,0]<=37)&(hsv[:,:,1]>=60)
                   &(hsv[:,:,2]>=minimum)).astype('uint8')
            # Include the dot's outline while omitting the moving dial fill.
            mask = cv2.dilate(ink,np.ones((3,3),np.uint8))
            self.waiting_shapes.append((cv2.cvtColor(template,cv2.COLOR_RGB2GRAY),mask))
        hook = cv2.cvtColor(self.assets['hook'][30:],cv2.COLOR_RGB2HSV)
        self.hook_mask = cv2.dilate(((hook[:,:,1]<50)&(hook[:,:,2]>=150)).astype('uint8'),
                                   np.ones((3,3),np.uint8))
        cancel = cv2.cvtColor(self.assets['cancel-cast'],cv2.COLOR_RGB2HSV)
        self.cancel_mask = cv2.dilate(((cancel[:,:,0]<=30)&(cancel[:,:,1]>=100)
                                      &(cancel[:,:,2]>=150)).astype('uint8'),
                                     np.ones((3,3),np.uint8))

    def cancel_button(self, frame):
        """Keep the original match, with a foreground-only X for moving glow."""
        if frame.shape[:2] != (1280,720):
            return False
        crop = frame[1066:1110,634:682]
        template = self.assets['cancel-cast']
        score = cv2.minMaxLoc(cv2.matchTemplate(crop,template,cv2.TM_CCOEFF_NORMED))[1]
        if np.isfinite(score) and score >= .94:
            return True
        scores = cv2.matchTemplate(cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY),
                                  cv2.cvtColor(template,cv2.COLOR_RGB2GRAY),
                                  cv2.TM_CCOEFF_NORMED,mask=self.cancel_mask)
        scores = np.where(np.isfinite(scores),scores,-1)
        _,matched,_,(x,y) = cv2.minMaxLoc(scores)
        h,w = template.shape[:2]
        hsv = cv2.cvtColor(crop[y:y+h,x:x+w],cv2.COLOR_RGB2HSV)
        orange = ((hsv[:,:,0]<=30)&(hsv[:,:,1]>=100)&(hsv[:,:,2]>=150))
        if matched >= .97 and int((orange & (self.cancel_mask>0)).sum()) >= 100:
            return True
        return False

    def status(self):
        return dict(phase=self.phase, value=self.current, player_level=self.player,
                    target=self.goal, target_mode='player_level', level_gap=self.LEVEL_GAP,
                    drags=self.drags, wakes=self.wakes, starts=self.starts, confirmed_starts=self.confirmed_starts,
                    start_fill_verified=self.start_fill_verified,
                    start_proof=self.start_proof, post_release_frames=self.start_seen,
                    blue_fill=self.filled, dial_state=self.dial_state)

    def waiting_cast(self, frame, *, allow_fill=False, allow_task=False):
        """Only the three-dot cast has a cancellable, non-combat fishing X."""
        if (not self._main_visible(frame) or self.training.boss_mode(frame) and not allow_task
                or self.blue_fill(frame) and not allow_fill):
            return False
        dot = self.assets['waiting-dot']
        for x in (322,360,398):
            crop = frame[997:1024,x-13:x+13]
            score = cv2.minMaxLoc(cv2.matchTemplate(crop,dot,cv2.TM_CCOEFF_NORMED))[1]
            if np.isfinite(score) and score >= .94:
                continue
            # The old low-meter fill tints the dots, especially the right one.
            # Callers must separately authorise stopping an existing fill.
            # The ordinary preparation and visible cancel-X guards still apply.
            if not allow_fill:
                return False
            found = False
            gray = cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
            for shape,mask in self.waiting_shapes:
                scores = cv2.matchTemplate(gray,shape,cv2.TM_CCOEFF_NORMED,mask=mask)
                scores = np.where(np.isfinite(scores),scores,-1)
                _,matched,_,(a,b) = cv2.minMaxLoc(scores)
                if matched < .97:
                    continue
                h,w = shape.shape
                hsv = cv2.cvtColor(crop[b:b+h,a:a+w],cv2.COLOR_RGB2HSV)
                gold = ((hsv[:,:,0]>=17)&(hsv[:,:,0]<=37)&(hsv[:,:,1]>=60)&(hsv[:,:,2]>=100))
                if int(gold.sum()) >= 90:
                    found = True
                    break
            if not found:
                return False
        return self.cancel_button(frame)

    def at_port(self, frame, *, main_known=False):
        if frame.shape[:2] != (1280,720) or not (main_known or self._main_visible(frame)):
            return False
        for template,(x,y,X,Y),minimum in ((self.port_assets['header-port'],(248,33,295,82),.92),
                                          (self.assets['port-anchor'],(310,954,416,1064),.94)):
            scores = cv2.matchTemplate(frame[y:Y,x:X],template,cv2.TM_CCOEFF_NORMED)
            score = cv2.minMaxLoc(scores)[1]
            if not np.isfinite(score) or score < minimum:
                return False
        return True

    def cancellable_cast(self, frame):
        # A pending low-meter calibration may stop the old continuous fishing
        # at its positively identified pre-battle X. Fill still prohibits all
        # dragging/starting, and a newly confirmed higher start must be adopted.
        return self.ordinary_preparation(frame, allow_fill=self._saved_low_correction())

    def ordinary_preparation(self, frame, *, allow_fill=False):
        """Read-only preparation probe; permission belongs to its controller."""
        return self.cast_preparation(frame,allow_fill=allow_fill)

    def cast_preparation(self, frame, *, allow_fill=False, allow_task=False):
        # Only the portal handoff opts into task-location preparation. Meter
        # calibration always uses ordinary_preparation and rejects task BOSS.
        if self.waiting_cast(frame,allow_fill=allow_fill,allow_task=allow_task):
            return True
        if (not self._main_visible(frame) or self.training.boss_mode(frame) and not allow_task
                or self.blue_fill(frame) and not allow_fill):
            return False
        # The yellow ! warns that a fish has bitten; combat replaces it with
        # crossed weapons. This ordinary pre-battle phase has the same stop X.
        # The ! pulses in size. Require its tall stem and separate aligned dot
        # instead of matching one animation frame. Remove the moving gold line.
        crop = cv2.cvtColor(frame[950:1062,330:390],cv2.COLOR_RGB2HSV)
        ink = ((abs(crop[:,:,0].astype(float)-self.warning_hue)<=10)
               &(crop[:,:,1]>=100)&(crop[:,:,2]>=150)).astype('uint8')
        box = self.box(frame)
        if box:
            y=(box[1]+box[3])//2-950
            ink[max(0,y-3):max(0,min(112,y+4))]=0
        _,_,components,_ = cv2.connectedComponentsWithStats(ink)
        warning = False
        for x,y,w,h,area in components[1:]:
            if not (10<=w<=35 and 35<=h<=100 and 15<=x+w/2<=45 and area>=w*h*.5):
                continue
            for a,b,W,H,size in components[1:]:
                if (8<=W<=28 and 5<=H<=22 and size>=W*H*.45
                        and abs(a+W/2-x-w/2)<=5 and 2<=b-y-h<=20):
                    warning = True
        if not warning and allow_fill:
            # The real night-time stem has a 146-value gradient, below the
            # ordinary 150 color cutoff. Match its fixed complete stem and
            # dot instead, retaining .97 and the independent ordinary X.
            template=self.assets['bite-warning-night']
            gray=cv2.cvtColor(frame[950:1062,330:390],cv2.COLOR_RGB2GRAY)
            scores=cv2.matchTemplate(gray,cv2.cvtColor(template,cv2.COLOR_RGB2GRAY),
                cv2.TM_CCOEFF_NORMED,mask=self.night_warning_mask)
            scores=np.where(np.isfinite(scores),scores,-1)
            _,score,_,(a,b)=cv2.minMaxLoc(scores)
            h,w=template.shape[:2]
            sample=crop[b:b+h,a:a+w]
            gold=((abs(sample[:,:,0].astype(float)-self.warning_hue)<=10)
                  &(sample[:,:,1]>=70)&(sample[:,:,2]>=100))
            warning=(score>=.97 and int((gold&(self.night_warning_mask>0)).sum())>=500)
        return warning and self.cancel_button(frame)

    def _saved_low_correction(self):
        return bool(self.training.options['meter_auto_adjust']
                    and self.training.state.get('sailing_sleep_recovery')
                    and self.training.state.get('sailing_meter_adjustment')
                    and self.current is not None
                    and (self.saved_player_gap or self.goal is not None
                         and self.goal-self.current >= self.RECALIBRATE_GAP))

    def prepare(self, frame):
        if time.monotonic() < self.next_action_at:
            return True
        # A newly issued start owns its verification. Its fresh fill must
        # never be cancelled by the older low-meter recovery request.
        if self.starts or self.wakes or self.start_fill_verified:
            return self.step(frame)
        if self.blue_fill(frame) and self._saved_low_correction():
            if self.cancels >= 2:
                raise MeterWorkflowLimit('计量校准取消等待抛竿后仍未休眠，已保留练级记录')
            data = self.cast_data(frame)
            if data is not None and self.action('scan_cancel',data):
                self.cancels += 1
                self.idle_since = None
                self.phase = 'detect'
                self.next_action_at = time.monotonic()+1.5
            return True
        if self.cancellable_cast(frame):
            if self.cancels >= 2:
                raise MeterWorkflowLimit('计量校准取消等待抛竿后仍未休眠，已保留练级记录')
            data = self.cast_data(frame)
            if data is not None and self.action('cancel',data):
                self.cancels += 1
                self.idle_since = None
                self.phase = 'detect'
                self.next_action_at = time.monotonic()+1.5
            return True
        return self.step(frame)

    def scan_cancel_frame(self, capture, route, expected, stopped, usable=None, *, expected_route=False):
        """Capture near the input until a short cancellation window appears."""
        deadline = time.monotonic()+6
        for _ in range(40):
            if stopped() or time.monotonic() >= deadline:
                break
            fresh = capture()
            if stopped() or time.monotonic() >= deadline:
                break
            current = self.cast_data(fresh)
            if current is None or any(current[key] != expected.get(key)
                                      for key in ('value','context','box','digit_strokes','world_strokes')):
                break
            sailing = route(fresh)
            if type(sailing) is bool and sailing is not expected_route:
                break
            if sailing is not expected_route:
                # A flying reward may briefly cover a main-screen guard. The
                # current digits/world/place still match, but this frame may
                # not authorize input. Observe another NEW frame in the same
                # six-second/40-frame window without losing prior OFF proof.
                continue
            if self.validate('cancel',fresh,expected):
                if stopped() or time.monotonic() >= deadline:
                    break
                if usable is None or usable(fresh):
                    return fresh
        return None

    def cast_data(self, frame):
        # Cancelling a positively identified waiting cast needs its current
        # dial and location, not the slower player/range OCR used for dragging.
        if not self.world_matches(frame):
            return None
        box = self.box(frame)
        if box is None:
            return None
        x,y,X,Y = box
        hsv = cv2.cvtColor(frame[y+5:Y-3,x+4:X-7],cv2.COLOR_RGB2HSV)
        ink = ((hsv[:,:,0]>=15)&(hsv[:,:,0]<=35)&(hsv[:,:,1]>60)&(hsv[:,:,2]>170))
        if int(ink.sum()) < 8:
            return None
        # Keep the fixed cyan stage-number strokes separate from the range
        # cache. A new gate/stage must abandon this short input window.
        badge = frame[45:71,40:68].astype(np.int16)
        world = ((badge[:,:,1]-badge[:,:,0]>=20)&(badge[:,:,2]-badge[:,:,0]>=30)
                 &(badge.max(axis=2)>=170))
        return dict(value=self.current,context=self._context(frame),box=box,
                    digit_strokes=hashlib.sha256(ink.tobytes()).hexdigest(),
                    world_strokes=hashlib.sha256(world.tobytes()).hexdigest())

    def observe(self, frame, *, main_known=False):
        """Report the dial independently of AUTO sailing and experience gains."""
        visible = frame.shape[:2] == (1280,720) and (main_known or self._main_visible(frame))
        self.filled = bool(visible and self.blue_fill(frame,main_known=main_known))
        if not visible:
            self.dial_state = 'unknown'
        elif self.at_port(frame,main_known=main_known):
            self.dial_state = 'docked'
        elif self.filled:
            self.dial_state = 'filled'
        else:
            icon = self.idle_icon(frame,main_known=main_known)
            self.dial_state = ('sleeping' if icon == 'sleep' else 'busy' if self.busy(frame,main_known=main_known)
                               else 'idle' if icon == 'hook' else 'unknown')
        return self.dial_state

    def idle_icon(self, frame, *, main_known=False):
        if frame.shape[:2] != (1280,720) or not (main_known or self._main_visible(frame)):
            return None
        for name in ('sleep','hook'):
            template = self.assets[name]
            if name == 'hook':
                # The gold connector crosses the handle at a different height
                # for every selected level. Match the invariant lower curve.
                template, crop = template[30:], frame[980:1062,316:405]
            else:
                crop = frame[947:1056,308:398]
            score = cv2.minMaxLoc(cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED))[1]
            if np.isfinite(score) and score >= .94:
                return name
            if name == 'hook':
                # At a middle depth the connector crosses the lower curve too.
                # Match the fixed gray curve/outline and omit that moving row.
                mask = self.hook_mask.copy()
                box = self.box(frame)
                if box:
                    row = (box[1]+box[3])//2-993
                    mask[max(0,row-4):max(0,min(mask.shape[0],row+5))] = 0
                if int(mask.sum()) >= 250:
                    matched = cv2.matchTemplate(frame[993:1048,330:390],template,
                                               cv2.TM_CCOEFF_NORMED,mask=mask)[0,0]
                    if np.isfinite(matched) and matched >= .96:
                        return 'hook'
        return None

    def adaptive_auto_on(self, frame):
        """Keep the existing full match; handle glow with the sailing guards.

        The fixed check strokes and surrounding checkbox must both match.
        The sailing AUTO arrow does not establish this separate checkbox.
        """
        if frame.shape[:2] != (1280,720):
            return False
        if self.training._match(frame,'main-auto',(276,881,310,913),.91):
            return True
        check=cv2.minMaxLoc(cv2.matchTemplate(frame[887:910,283:305],
            self.adaptive_auto_template[8:27,9:27],cv2.TM_CCOEFF_NORMED))[1]
        outline=cv2.minMaxLoc(cv2.matchTemplate(frame[879:915,274:312],
            self.adaptive_auto_template,cv2.TM_CCOEFF_NORMED))[1]
        return bool(np.isfinite(check) and check>=.95 and np.isfinite(outline) and outline>=.88)

    def _main_visible(self, frame):
        if frame.shape[:2] != (1280,720):
            return False
        # A floating reward can briefly cover the fishcake entry. Confirmation
        # uses the other three main guards and brightness. Numeric and idle
        # guards still independently reject menus, casts, and combat gestures.
        boxes = {'guard-star': (208,1137,269,1192),
                 'guard-diamond': (539,1137,600,1186)}
        return (all(self.training._match(frame,name,box,.91) for name,box in boxes.items())
                and self.adaptive_auto_on(frame)
                and float((frame[1147:1168,558:585].max(axis=2) >= 210).mean()) >= .30)

    def blue_fill(self, frame, *, main_known=False):
        if frame.shape[:2] != (1280,720) or not (main_known or self._main_visible(frame)):
            return False
        # A fill crosses both sides of the dial at the same height. Restrict
        # the sample to its interior, away from the cyan Zzz/hook in the middle
        # and the blue effects outside the rim. Partial fills must also count.
        crop = frame[982:1068,301:420]
        hsv = cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
        rgb = crop.astype(np.int16)
        y,x = np.ogrid[982:1068,301:420]
        inside = (x-360)**2+(y-1010)**2 <= 65**2
        blue = ((hsv[:,:,0] >= 75) & (hsv[:,:,0] <= 105)
                & (hsv[:,:,1] >= 20) & (hsv[:,:,2] >= 65)
                & (rgb[:,:,1]-rgb[:,:,0] >= 10) & (rgb[:,:,2]-rgb[:,:,0] >= 7))
        # Night lighting tints the same fill olive/gray. Real paired native
        # frames retain its bright, low-saturation surface on both inside
        # sides; the empty brown dial, gold connector and outside effects do
        # not. Keep the original six-row, same-height geometry for both hues.
        warm = ((hsv[:,:,0] >= 18) & (hsv[:,:,0] <= 75)
                & (hsv[:,:,1] >= 20) & (hsv[:,:,1] <= 110)
                & (hsv[:,:,2] >= 100) & (rgb[:,:,1]-rgb[:,:,0] >= -4)
                & (rgb[:,:,2]-rgb[:,:,0] >= -32)
                & (rgb.max(axis=2)-rgb.min(axis=2) <= 50))
        mask = (blue | warm) & inside
        left,right = inside[:,:20].sum(axis=1),inside[:,-20:].sum(axis=1)
        rows = ((left >= 8) & (right >= 8)
                & (mask[:,:20].sum(axis=1) >= left*.6)
                & (mask[:,-20:].sum(axis=1) >= right*.6))
        run = 0
        for row in rows:
            run = run+1 if row else 0
            if run >= 6:
                return True
        return False

    def busy(self, frame, *, main_known=False):
        # A hook can remain visible after casting, and the AUTO text can become
        # bright between catches. Either a fill or dimmed AUTO prevents input.
        return ((main_known or self._main_visible(frame)) and (self.blue_fill(frame, main_known=main_known)
                or float((frame[885:907,212:274].max(axis=2) >= 210).mean()) < .08))

    def box(self, frame):
        hsv = cv2.cvtColor(frame[915:1112, 202:281], cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([15, 100, 155]), np.array([35, 255, 255]))
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        candidates = []
        for x, y, width, height, area in stats[1:]:
            if (15 <= width <= 79 and 34 <= height <= 48 and area >= 90
                    and self.TOP-5 <= y+915+height/2 <= self.BOTTOM+5):
                candidates.append((int(x+202), int(y+915), int(x+202+width), int(y+915+height)))
        return candidates[0] if len(candidates) == 1 else None

    def selected(self, frame, box):
        """读圆盘当前选中的数字。

        两个视图(裁图位置略不同)分别校验后投票:
        - 单视图可信、另一视图没读出 → 用可信的那个;
        - 两视图都读出且一致 → 用它;不一致 → 判为读不清。

        原先是"任一视图不合格就整帧放弃",于是出现实测到的现象:视图0 置信 0.964
        (差一点没过 0.97)直接把整帧判死,而视图1 其实是 1.000。数字框顶端被金框或
        指针压住一点点时就会这样,**读数长期随机失败**,校准跟着一直报"读不清"。
        """
        x, y, X, Y = box
        views = []
        for crop in (frame[y+5:Y-3, x+4:X-7], frame[y+6:Y-5, x+5:X-10]):
            if min(crop.shape[:2]) < 4:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            ink = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 35)
                   & (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 170))
            ys, xs = np.where(ink)
            if len(xs) < 8:
                continue
            binary = np.repeat(np.where(ink, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
            binary = binary[max(0, ys.min()-2):min(binary.shape[0], ys.max()+3),
                            max(0, xs.min()-2):min(binary.shape[1], xs.max()+3)]
            views.append(binary)
        votes = []
        for result in read_numeric_views(self.training.reader, views):
            if result.value is None or not 1 <= result.value <= 10000 or result.confidence < .97:
                continue
            votes.append(result.value)
        if not votes:
            return None
        if len(set(votes)) != 1:
            return None
        return votes[0]

    def _context(self, frame):
        # The solid digit/name strokes stay fixed while the translucent paper
        # and the travel countdown animate. Exclude those changing pixels.
        rank = np.all(frame[12:50,72:130] >= 240,axis=2)
        # Animated lighting can move one antialiased edge pixel across 70
        # within an unchanged place name. Keep its darker fixed letter cores;
        # the rank strokes, selected digits and world/OFF guards stay exact.
        location = np.max(frame[12:34,286:434],axis=2) < 60
        return hashlib.sha256(rank.tobytes()+location.tobytes()).hexdigest()

    def world_matches(self, frame):
        expected = self.training.state.get('rank') if self.training.state.get('mode') == 'farming' else None
        if expected is None:
            return True
        if frame is not self._world_frame or expected != self._world_rank:
            self._world_frame,self._world_rank = frame,expected
            strokes = (hashlib.sha256(np.all(frame[12:50,72:130]>=240,axis=2).tobytes()).digest()
                       if frame.shape[:2] == (1280,720) else None)
            if strokes is not None and self._rank_reference == (expected,strokes):
                self._world_ok = True
                return True
            actual = (self.training._number(frame,((71,9,136,59),(80,9,130,59)))
                      if frame.shape[:2] == (1280,720) else None)
            self._world_ok = actual == expected
            if self._world_ok:
                self._rank_reference = expected,strokes
        return self._world_ok

    def _covered(self, box, area):
        x,y,X,Y = box
        a,b,A,B = area
        overlap = max(0,min(X,A)-max(x,a))*max(0,min(Y,B)-max(y,b))
        return overlap >= (A-a)*(B-b)*.15

    def _cached_limit(self, context, name):
        c = self.range_cache
        if not isinstance(c,dict):
            return None
        value, observed = c.get(name),c.get(name+'_at')
        return value if (c.get('version') == 1 and c.get('context') == context
                         and isinstance(value,int) and not isinstance(value,bool) and 1 <= value <= 10000
                         and isinstance(observed,(float,int)) and not isinstance(observed,bool)
                         and 0 <= time.time()-observed <= 180) else None

    def _remember_limits(self, context, low, high, low_clear, high_clear):
        if not high_clear and not low_clear:
            return
        previous = self.range_cache if isinstance(self.range_cache,dict) else {}
        cache = dict(previous) if previous.get('context') == context else {}
        cache.update(version=1,context=context)
        if low_clear:
            cache.update(low=low,low_at=time.time())
        if high_clear:
            cache.update(high=high,high_at=time.time())
        self.range_cache = cache
        if self.training.state.get('sailing_sleep_recovery') and self.training.state.get('fishing_meter_limits') != cache:
            self.training.state['fishing_meter_limits'] = cache
            self.training._save()

    def measure(self, frame, *, main_known=False, remember=True):
        """Reading an active dial is allowed; gestures still require true idle."""
        if (frame.shape[:2] != (1280,720) or self.training.boss_mode(frame)
                or not (main_known or self._main_visible(frame)) or not self.world_matches(frame)
                or self.at_port(frame,main_known=main_known)
                or not self.adaptive_auto_on(frame)):
            return None
        box = self.box(frame)
        if box is None:
            return None
        selected = self.selected(frame, box)
        player = self.training._number(frame, ((76,1248,137,1278), (79,1249,136,1277)))
        if selected is None or player is None:
            return None
        center = (box[1]+box[3])//2
        # The visible frame center shifts by six pixels at the lower stop.
        # Its gold value covers the endpoint label; reading that cropped label
        # can turn 1359 into 359. The lower stop uses its full selected value;
        # the upper limit always needs an independent label reading.
        # Keep both complete text views inside the label. The wider second
        # region included an animated ring edge, creating a conflicting vote.
        low = selected if center >= self.BOTTOM-self.ENDPOINT_SLOP else self.training._number(
            frame, ((236,1070,283,1099), (236,1071,282,1097)))
        # The first old crop included two rows above the actual white label;
        # on the real 1604 frame its confidence fell to .63 while the complete
        # tighter view read .995. Keep two independent full-number views and
        # the same conflict/confidence guards, excluding the animated margin.
        high = self.training._number(frame, ((237,931,282,953), (239,930,280,952)))
        context = self._context(frame)
        low_covered = self._covered(box,(234,1070,284,1100))
        high_covered = self._covered(box,(237,929,282,953))
        low_clear, high_clear = low is not None and not low_covered, high is not None and not high_covered
        # A cropped label is not a new verified endpoint, even if OCR happens
        # to recover its value. Only a recent same-location record can replace
        # a hidden upper label; a nearby selected value cannot prove MAX.
        if low_covered and center < self.BOTTOM-self.ENDPOINT_SLOP:
            low = self._cached_limit(context,'low')
        # Proximity to the stop cannot establish the numeric upper limit:
        # the actual 1806 selection at y937 still had a verified MAX of 1810.
        # If the label is hidden, retain only a recent independent reading;
        # otherwise reveal it before starting, even at the top of the track.
        if high_covered:
            high = self._cached_limit(context,'high')
        if (low is None and not low_covered or high is None and not high_covered
                or low is not None and selected < low or high is not None and selected > high
                or low is not None and high is not None and low > high):
            return None
        if remember:
            self._remember_limits(context,low,high,low_clear,high_clear)
        # Target the player within the independently read available range.
        # Keep adaptive AUTO unchanged; its failures may lower the selection.
        goal = (None if high is None else min(high,player) if low is None
                else max(low+1,min(high,player)) if low < high else low)
        # Keep the existing low-meter trigger relative to the player. The
        # game's adaptive holder lowers its selection after failures; treating
        # every such drop from the maximum as stale would fight that setting.
        recalibration_goal = (None if high is None else min(high,player-self.LEVEL_GAP//2)
            if low is None else max(low+1,min(high,player-self.LEVEL_GAP//2)) if low < high else low)
        return dict(value=selected, player=player, low=low, high=high,
                    x=(box[0]+box[2])//2, y=center, goal=goal, context=context,
                    adaptive=True,recalibration_goal=recalibration_goal)

    def read(self, frame):
        if self.busy(frame) or not self.idle_icon(frame):
            return None
        return self.measure(frame)

    def close_enough(self, data):
        current, player, low, high = (data[k] for k in ('value','player','low','high'))
        if low is None or high is None:
            return False
        if data.get('adaptive') is True:
            target = max(low+1,min(high,player)) if low < high else low
            return low <= current <= high and (current > low or low == high) and abs(current-target) <= self.LEVEL_GAP
        return (current > low or low == high) and (
            0 <= player-current <= self.LEVEL_GAP
            or player >= high and current == high
            or player <= low and current == min(low+1, high))

    def plan(self, data):
        if data['low'] is None or data['high'] is None:
            # Only read()'s spatially verified occlusion can reach this branch.
            # Move inside the existing track to reveal both labels; never start
            # fishing before they have been read and the level reselected.
            return {**data,'to_y':(self.TOP+self.BOTTOM)//2,'kind':'reveal'}
        if data['high'] <= data['low']:
            return None
        if data.get('adaptive') is True and data['goal'] == data['high']:
            # The one-pixel endpoint correction can equal the handle's current
            # center, leaving MAX-1 after a stationary touch. Give the final drag
            # enough travel to cross the stop;
            # the widget clamps it. Numeric MAX and the handle's actual endpoint
            # are still independently verified after release.
            target = (self.TOP-self.MAX_DRAG_OVERSHOOT if data['value'] < data['high']
                      and data['y'] <= self.TOP+self.ENDPOINT_SLOP else self.TOP)
            return {**data,'to_y':target,'kind':'drag'}
        delta = (data['goal']-data['value']) * (self.BOTTOM-self.TOP)/(data['high']-data['low'])
        y = max(self.TOP, min(self.BOTTOM, round(data['y']-delta)))
        if y == data['y']:
            y += -1 if data['goal'] > data['value'] else 1
        return {**data, 'to_y': y, 'kind':'drag'}

    def start_reading(self, frame, expected):
        """Validate one fresh idle reading and return its actual numbers.

        The adaptive handle can settle by a couple of levels after dragging.
        Preserve exact context/range/AUTO checks and current-player tolerance;
        a small allowed settling change is not a reason to recast or drag.
        """
        if frame.shape[:2] != (1280,720) or self.training.boss_mode(frame):
            return None
        data = self.read(frame)
        if (data is None or not self.close_enough(data)
                or self.idle_icon(frame) not in ('sleep','hook')
                or any(data[key] != expected[key] for key in
                       ('context','low','high','adaptive') if key in expected)):
            return None
        previous = expected.get('value',data['value'])
        if (data['value'] != previous
                and not (data.get('adaptive') is True and expected.get('adaptive') is True
                         and type(previous) is int
                         and abs(data['value']-previous) <= self.LEVEL_GAP)):
            return None
        return data

    def resume_cast_data(self, frame):
        """A restoration tap needs actual idle, never a skull-only inference.

        The calibration controller separately owns the saved cancel, remaining
        budget, current rank/stage/route and three-second context consensus.
        This does not authorize any task-location drag, hold or cancellation.
        """
        if (frame.shape[:2] != (1280,720) or not self._main_visible(frame)
                or not self.world_matches(frame) or self.at_port(frame)
                or self.idle_icon(frame) != 'hook' or self.busy(frame)
                or self.real_battle(frame) or self.cancel_button(frame)):
            return None
        return self.cast_data(frame)

    def validate(self, action, frame, expected):
        if action == 'wake':
            if (not getattr(self, 'allow_sleep_wake', False) or self.starts or self.wakes
                    or frame.shape[:2] != (1280,720) or self.training.boss_mode(frame)
                    or not self._main_visible(frame) or not self.world_matches(frame)
                    or self.at_port(frame) or self.idle_icon(frame) != 'sleep'
                    or self.busy(frame) or self.real_battle(frame) or self.cancel_button(frame)):
                return False
            actual = self.cast_data(frame)
            return bool(actual is not None and all(actual[key] == expected.get(key)
                for key in ('context','box','digit_strokes','world_strokes')))
        if action == 'resume_cast':
            actual = self.resume_cast_data(frame)
            return bool(actual is not None and all(actual[key] == expected.get(key)
                for key in ('context','box','digit_strokes','world_strokes')))
        if frame.shape[:2] != (1280,720) or self.training.boss_mode(frame):
            return False
        if action == 'start':
            if (expected.get('idle_icon') is not None
                    and self.idle_icon(frame) != expected['idle_icon']):
                self.validated_start = None
                return False
            actual = self.start_reading(frame,expected)
            # A bounded, single-frame receipt snapshot, never a reading for
            # another capture. The runtime still uses validate() as its gate.
            self.validated_start = (frame,actual) if actual is not None else None
            return actual is not None
        if action == 'cancel':
            data = self.cast_data(frame)
            return (self.cancellable_cast(frame) and data is not None
                    and data['context'] == expected.get('context')
                    and data['value'] == expected.get('value')
                    and data['box'] == expected.get('box')
                    and data['world_strokes'] == expected.get('world_strokes')
                    and data['digit_strokes'] == expected.get('digit_strokes'))
        data = self.read(frame)
        if data is None:
            return False
        if action not in ('drag','reveal') or self.close_enough(data):
            return False
        plan = self.plan(data)
        return (plan is not None and plan['kind'] == action
                and data['context'] == expected.get('context')
                and data['value'] == expected['value'] and data['low'] == expected['low']
                and data['high'] == expected['high'] and abs(data['y']-expected['y']) <= 3
                and data['player'] == expected['player'] and data['goal'] == expected['goal']
                and data['adaptive'] == expected.get('adaptive')
                and data['x'] == expected['x'] and abs(plan['to_y']-expected['to_y']) <= 3
                and (self.TOP-self.MAX_DRAG_OVERSHOOT if data['adaptive'] is True
                     else self.TOP-1) <= expected['to_y'] <= self.BOTTOM)

    def _seen(self, key):
        self.stable = self.stable+1 if key == self.key else 1
        self.key = key
        return self.stable >= 2

    def _start_fill(self, frame):
        return (frame.shape[:2] == (1280,720) and not self.training.boss_mode(frame)
                and self.world_matches(frame) and not self.at_port(frame) and self.blue_fill(frame))

    def real_battle(self, frame):
        """Require the HP sword, timer and masked crossed weapons together."""
        return bool(frame.shape[:2] == (1280,720) and self._main_visible(frame)
            and self.start_defense._match(frame,'timer',(188,125,610,165),.94)
            and self.start_defense._match(frame,'damage-sword',(153,205,192,245),.96)
            and self.start_defense._battle_footer(frame))

    def _start_identity(self, frame):
        if frame.shape[:2] != (1280,720):
            return None
        badge = frame[45:71,40:68].astype(np.int16)
        ink = ((badge[:,:,1]-badge[:,:,0] >= 10) & (badge[:,:,2]-badge[:,:,0] >= 15)
               & (badge.max(axis=2) >= 110))
        if int(ink.sum()) < 8:
            return None
        return self._context(frame), ink.tobytes()

    def post_start_evidence(self, frame):
        """Read-only proof; AUTO, a hook or a result alone never prove a start."""
        if (frame.shape[:2] != (1280,720) or self.training.boss_mode(frame)
                or not self.world_matches(frame) or not self._main_visible(frame)
                or self.at_port(frame)
                or not self.start_defense._match(frame,'world',(8,40,46,80),.94)):
            return False
        def fixed(name, box, threshold):
            x,y,X,Y = box
            scores = cv2.matchTemplate(frame[y-2:Y+2,x-2:X+2],self.start_route_assets[name],cv2.TM_CCOEFF_NORMED)
            score = cv2.minMaxLoc(scores)[1]
            return np.isfinite(score) and score >= threshold
        running = getattr(self, 'start_route_expected', False) is True
        return bool(fixed('auto-label',(186,34,253,65),.90)
            and fixed('active-arrow' if running else 'inactive-face',(194,4,246,42),.96)
            and not fixed('inactive-face' if running else 'active-arrow',(194,4,246,42),.96)
            and (self.blue_fill(frame) or self.real_battle(frame)))

    def arm_start_observation(self, frame):
        self.phase = 'starting'
        self.since = time.monotonic()
        self.start_anchor = self._start_identity(frame)
        self.start_fill_verified = False
        self.start_proof = None
        self._reset_start_observation()

    def _reset_start_observation(self):
        self.start_seen_at = self.start_last_seen = self.start_last_frame = None
        self.start_seen = 0

    def confirm_start(self, frame):
        if self.phase not in ('starting', 'waking'):
            return False
        now = time.monotonic()
        if (self.start_anchor is None or self._start_identity(frame) != self.start_anchor
                or not self.post_start_evidence(frame)):
            self._reset_start_observation()
            return False
        # The caller supplies a newly captured frame. Reject reusing the same
        # array, long observation gaps and zero-time duplicate callbacks.
        if frame is self.start_last_frame or self.start_last_seen == now:
            return False
        if self.start_last_seen is None or now-self.start_last_seen > 6:
            self._reset_start_observation()
            self.start_seen_at = now
        self.start_last_frame, self.start_last_seen = frame, now
        self.start_seen += 1
        if self.start_seen < 3 or now-self.start_seen_at < self.IDLE_SECONDS:
            return False
        self.start_fill_verified = True  # Legacy public field: verified start.
        self.start_proof = 'post_release_fishing'
        self.filled = self.blue_fill(frame)
        self.dial_state = 'filled' if self.filled else 'battle'
        self.confirmed_starts += 1
        self.phase = 'active'
        self.drags = self.starts = 0
        self.stable, self.key = 0,None
        self.event('已确认开始钓鱼：长按结束后连续新图至少3秒确认真实填充或战斗')
        return True

    def step(self, frame):
        now = time.monotonic()
        if (frame.shape[:2] != (1280,720) or self.training.boss_mode(frame)
                or not self.world_matches(frame) or self.at_port(frame)):
            self.stable, self.key = 0, None
            self.idle_since = None
            self._reset_start_observation()
            return False
        if self.phase == 'waking':
            # A wake tap can start fishing directly. Observe it without another
            # hold, cancellation or drag. Otherwise require a freshly idle hook
            # and the normal three-second numeric consensus before the hold.
            if self.start_anchor is None:
                self.start_anchor = self._start_identity(frame)
            if self.confirm_start(frame):
                return False
            if (self._start_identity(frame) == self.start_anchor
                    and self._main_visible(frame) and self.idle_icon(frame) == 'hook'
                    and not self.busy(frame) and not self.real_battle(frame)):
                self.phase = 'detect'
                self.idle_since = now
                self.stable, self.key = 0, None
                self._reset_start_observation()
                self.event('钓鱼计量：单次唤醒后已见灰色鱼钩，重新确认休眠与读数后再长按')
                return True
            if now-self.since > self.WAKE_SECONDS:
                raise MeterWorkflowLimit('单次唤醒Zzz后未确认灰色鱼钩或真实钓鱼，保留已用次数和原截止时间')
            return True
        if self.phase == 'starting':
            if self.confirm_start(frame):
                return False
            if now-self.since > 30:
                raise MeterWorkflowLimit('长按后未连续确认真实填充或战斗，已停止并保留画面')
            return True  # Never drag, cancel or repeat the hold while verifying.
        self.observe(frame)
        if self.busy(frame):
            self.idle_since = None
            if self.phase == 'starting' and not self.confirm_start(frame):
                if now-self.since > 30:
                    raise MeterWorkflowLimit('长按后未确认大球浅蓝色填充，已停止并保留画面')
                self.stable, self.key = 0,None
                return True
            self.phase = 'active'
            self.stable, self.key = 0, None
            self.drags = self.starts = 0
            return False
        if not self._main_visible(frame):
            self.stable, self.key = 0, None
            self.idle_since = None
            return False
        if not self.idle_icon(frame):
            if self.confirm_start(frame):
                return False
            self.stable, self.key = 0, None
            self.idle_since = None
            if self.phase == 'starting':
                if now-self.since > 30:
                    raise MeterWorkflowLimit('长按后圆盘状态持续无法确认，已停止并保留画面')
                return True
            self.phase = 'unknown'
            return False
        if now < self.next_action_at:
            return True
        if self.idle_since is None:
            self.idle_since = now
        data = self.read(frame)
        if data is None:
            if now-self.last_warning_at >= 60:
                self.event('钓鱼计量或上下限未读清，暂不拖动或开始')
                self.last_warning_at = now
            self.stable, self.key = 0, None
            self.idle_since = None
            return False
        self.current, self.player, self.goal = data['value'], data['player'], data['goal']
        if not self._seen((data['value'],data['player'],data['y'],data['low'],data['high'],
                           data['adaptive'],data['context'])):
            return True
        if now-self.idle_since < self.IDLE_SECONDS:
            return True
        if not self.close_enough(data):
            plan = self.plan(data)
            if plan is None:
                return False
            if self.drags >= 4:
                raise MeterWorkflowLimit('钓鱼计量连续调整后仍未接近角色等级，已停止并保留画面')
            kind = plan['kind']
            if self.action(kind, plan):
                self.drags += 1
                self.phase = 'revealing' if kind == 'reveal' else 'adjusting'
                self.since = now
                self.next_action_at = now+1.5
        else:
            if getattr(self, 'allow_sleep_wake', False) and self.idle_icon(frame) == 'sleep':
                if self.wakes or self.starts:
                    raise MeterWorkflowLimit('本轮Zzz唤醒已用，拒绝重复唤醒或重试已用长按')
                signature = self.cast_data(frame)
                if signature is None or signature['context'] != data['context']:
                    return True
                result = self.action('wake', {**data, **signature})
                if result is not False and result is not None:
                    self.wakes += 1
                    self.arm_start_observation(frame)
                    self.phase = 'waking'
                    self.idle_since = None
                    self.stable, self.key = 0, None
                    self.event('钓鱼计量：只唤醒Zzz一次，等待灰色鱼钩或真实填充/战斗，不计启动成功')
                return True
            if self.phase == 'starting' and now-self.since < 8:
                return True
            if self.starts >= 3:
                raise MeterWorkflowLimit('长按开始钓鱼后未确认浅蓝色填充，已停止并保留画面')
            expected_start = ({**data, 'idle_icon': self.idle_icon(frame)}
                              if getattr(self, 'allow_sleep_wake', False) else data)
            result = self.action('start', expected_start)
            if result is not False and result is not None:
                self.starts += 1
                self.arm_start_observation(frame)
                self.idle_since = None
                self.next_action_at = self.since+1.5
                feedback = [result] if isinstance(result,np.ndarray) else result if isinstance(result,(list,tuple)) else []
                # The during-hold capture remains diagnostic only. Begin the
                # continuous proof with the immediate post-release capture.
                if len(feedback) >= 2:
                    self.confirm_start(feedback[-1])
        self.stable, self.key = 0, None
        return True

FishingMeter.WAKE_SECONDS = 15
