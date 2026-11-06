"""Guardian damage taps and Sea God counters, guarded from each new frame.

Strategy: https://gamerch.com/drifting-girl/992620
Clicks are attempts, never evidence by themselves that an attack was cancelled.
"""
from pathlib import Path
import time

import cv2
import numpy as np

from drift.tasks.final_boss_attack import options_for


class BossDefense:
    def __init__(self, root, training, action, event, scan=False, *, final_boss_attacks=False):
        self.training,self.action,self.event = training,action,event
        self.final_boss_attacks = final_boss_attacks
        self.assets = {}
        for name,path in (('world',Path(root)/'assets/parallel/world-badge.png'),
                          ('rank38',Path(root)/'assets/boss-defense/rank38.png'),
                          ('rank39',Path(root)/'assets/boss-defense/rank39.png'),
                          ('timer',Path(root)/'assets/boss-defense/timer.png'),
                          ('sea-god',Path(root)/'assets/boss-defense/sea-god-name.png'),
                          ('guardian',Path(root)/'assets/boss-defense/guardian-name.png'),
                          ('swords',Path(root)/'assets/boss-defense/swords.png'),
                          ('damage-sword',Path(root)/'assets/boss-defense/damage-sword.png')):
            raw = cv2.imread(str(path))
            if raw is None:
                raise RuntimeError('海神反击模板缺失：'+name)
            self.assets[name] = cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
        sword = cv2.cvtColor(self.assets['swords'],cv2.COLOR_RGB2HSV)
        # Only the white blades and red grip are stable; low saturation alone
        # also included the animated brown/grey disc behind the weapons.
        self.sword_mask = (((sword[:,:,1] < 60) & (sword[:,:,2] >= 160))
                           | ((sword[:,:,0] < 12) & (sword[:,:,1] >= 140)
                              & (sword[:,:,2] >= 145))).astype(np.uint8)
        # Native combat frames pulse this glyph down to 85/90/95 percent.
        # Retain only blade/handle strokes at each scale; animated disc pixels
        # and a lowered correlation threshold cannot authorize an input.
        self.sword_variants = []
        for scale in (1., .95, .90, .85):
            template = cv2.resize(self.assets['swords'],None,fx=scale,fy=scale)
            pixels = cv2.cvtColor(template,cv2.COLOR_RGB2HSV)
            mask = (((pixels[:,:,1] < 45) & (pixels[:,:,2] >= 175))
                    | ((pixels[:,:,0] < 12) & (pixels[:,:,1] >= 140)
                       & (pixels[:,:,2] >= 150))).astype(np.uint8)
            self.sword_variants.append((template,mask))
        name = cv2.cvtColor(self.assets['sea-god'],cv2.COLOR_RGB2HSV)
        self.name_mask = ((name[:,:,1] < 100) | (name[:,:,2] < 90)).astype(np.uint8)
        guardian = cv2.cvtColor(self.assets['guardian'],cv2.COLOR_RGB2HSV)
        self.guardian_mask = ((guardian[:,:,1] < 100) | (guardian[:,:,2] < 90)).astype(np.uint8)
        meter = cv2.cvtColor(self.assets['damage-sword'],cv2.COLOR_RGB2HSV)
        self.meter_mask = ((meter[:,:,0] >= 10) & (meter[:,:,0] <= 45)
                           & (meter[:,:,1] >= 80) & (meter[:,:,2] >= 100)).astype(np.uint8)
        self.rank_reference = None
        self.damage_taps = 0
        self.next_damage_at = 0.
        self.scan = scan
        self.active = False
        self.active_rank = None
        self.bursts,self.taps = 0,0
        self.target = None
        self.last_side = None
        self.next_at = 0.
        self.last_burst = dict(taps=0,elapsed_ms=0,tap_gaps_ms=[])

    def status(self):
        return dict(enabled=self.training.options['boss_interrupt'] or self.final_boss_attacks,active=self.active,
                    configured_enabled=self.training.options['boss_interrupt'],
                    scope='guardian_tap_and_final_boss' if self.final_boss_attacks else 'configured',
                    active_rank=self.active_rank,
                    target=self.target,bursts=self.bursts,taps=self.taps,
                    battle_tap=self.training.options['battle_tap'] or self.final_boss_attacks,
                    configured_battle_tap=self.training.options['battle_tap'],damage_taps=self.damage_taps,
                    last_burst=self.last_burst)

    def _match(self, frame, name, box, threshold=.94):
        x,y,X,Y = box
        if name == 'swords':
            for template,mask in self.sword_variants:
                scores = cv2.matchTemplate(frame[y:Y,x:X],template,cv2.TM_CCOEFF_NORMED,mask=mask)
                scores = np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1)
                if cv2.minMaxLoc(scores)[1] >= threshold:
                    return True
            return False
        mask = (self.meter_mask if name == 'damage-sword' else self.name_mask if name == 'sea-god'
                else self.guardian_mask if name == 'guardian' else None)
        scores = cv2.matchTemplate(frame[y:Y,x:X],self.assets[name],cv2.TM_CCOEFF_NORMED,mask=mask)
        scores = np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1)
        return cv2.minMaxLoc(scores)[1] >= threshold

    def owns(self, frame, rank):
        encounter = self.encounter(frame,rank)
        return self.attack_options(frame,rank,encounter=encounter)['boss_interrupt'] and encounter

    def attack_options(self, frame, rank, *, encounter=None):
        """Resolve additional boss attacks from this fresh frame only."""
        if not self.final_boss_attacks:
            return self.training.options
        if encounter is None:
            encounter = self.encounter(frame,rank)
        return options_for(self.training,rank,encounter=encounter,
                           guardian=self.guardian_encounter(frame,rank))

    def guardian_encounter(self, frame, rank):
        """RANK38→39 guardian; the task marker alone also matches ordinary fish."""
        if frame.shape[:2] != (1280,720) or type(rank) is not int or rank != 38:
            return False
        return (self.training.boss_mode(frame)
                and self._match(frame,'world',(8,40,46,80),.94)
                and self._match(frame,'rank38',(65,9,139,63),.97)
                and self._match(frame,'guardian',(187,168,254,199),.97)
                and self._match(frame,'timer',(188,125,610,165),.94)
                and float((np.max(frame[174:194,195:246],axis=2) >= 210).mean()) >= .05)

    def encounter_rank(self, frame):
        """Select the native-route attack rank from the image, never an old journal."""
        if self.encounter(frame,39):
            return 39
        if self.guardian_encounter(frame,38):
            return 38
        return None

    def encounter(self, frame, rank):
        """Read-only encounter guard, also usable when tapping is disabled."""
        if (frame.shape[:2] != (1280,720)
                or (rank if rank is not None else self.training.state.get('rank')) != 39):
            return False
        # A task skull also remains during ordinary fish. Require the actual
        # Sea God name, timer and undimmed world before holding the encounter.
        return (self.training.boss_mode(frame)
                and self._match(frame,'world',(8,40,46,80),.94)
                and self._match(frame,'rank39',(65,9,139,63),.97)
                and self._match(frame,'sea-god',(187,168,240,199),.97)
                and self._match(frame,'timer',(188,125,610,165),.94)
                and float((np.max(frame[174:194,195:227],axis=2) >= 210).mean()) >= .05)

    def battle(self, frame, rank):
        if not self.owns(frame,rank):
            return False
        return self._battle_footer(frame)

    def _battle_footer(self, frame):
        footer = (all(self.training._match(frame,name,box,.91) for name,box in
                      (('guard-star',(208,1137,269,1192)),
                       ('guard-diamond',(539,1137,600,1186)),
                       ('main-auto',(276,881,310,913))))
                  and float((np.max(frame[1147:1168,558:585],axis=2) >= 210).mean()) >= .30)
        return footer and self._match(frame,'swords',(307,962,415,1057),.96)

    @staticmethod
    def _rank_strokes(frame):
        return np.all(frame[12:50,72:130] >= 240,axis=2).tobytes()

    def confirm_rank(self, frame, rank):
        # Called only after the existing three-frame OCR consensus. The current
        # pixels must still agree before every generic fish-battle contact.
        if (frame.shape[:2] == (1280,720) and type(rank) is int and 0 <= rank < 40
                and self._match(frame,'world',(8,40,46,80),.94)):
            self.rank_reference = (rank,self._rank_strokes(frame))
        else:
            self.rank_reference = None

    def damage_battle(self, frame, rank):
        if (not self.attack_options(frame,rank)['battle_tap'] or frame.shape[:2] != (1280,720)
                or type(rank) is not int or not 0 <= rank < 40):
            return False
        rank_ok = (self._match(frame,'rank39',(65,9,139,63),.97) if rank == 39 else
                   True if self.final_boss_attacks and self.guardian_encounter(frame,rank) else
                   self.rank_reference == (rank,self._rank_strokes(frame)))
        return (rank_ok and self._match(frame,'world',(8,40,46,80),.94)
                and self._match(frame,'timer',(188,125,610,165),.94)
                and self._match(frame,'damage-sword',(153,205,192,245),.96)
                and self._battle_footer(frame))

    @staticmethod
    def _combat_context(frame):
        # Only fixed rank/location/title strokes. Never use the animated fish,
        # HP amount, countdown, progress-bar background or old coordinates.
        location = np.max(frame[12:34,286:434],axis=2) < 80
        title = np.all(frame[174:194,194:330] >= 210,axis=2)
        task = np.all(frame[38:75,254:289] >= 210,axis=2)
        return (BossDefense._rank_strokes(frame),location.tobytes(),title.tobytes(),task.tobytes())

    @staticmethod
    def damage_target():
        # Empty gameplay space below the HP bars, outside all menu, skill,
        # AUTO, fish-cancel and central dial buttons on the guarded portrait UI.
        return dict(kind='damage',side='screen',x=360,y=330)

    def locate(self, frame, rank):
        if not self.battle(frame,rank):
            return None
        return self._locate_core(frame)

    def _locate_core(self, frame):
        candidates = []
        # Exclude the character/ship below, side menus, and the central face.
        # The strategy's cue is the raised weapon glowing yellow on either side.
        for side,(x,y,X,Y) in (('left',(90,275,238,625)),('right',(430,410,635,705))):
            crop = frame[y:Y,x:X]
            hsv = cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
            # Real charging weapon tips have a broad pale gold core. Narrow
            # sparks and the saturated orange fire on the ship are not cues.
            warm = (((hsv[:,:,0] >= 10) & (hsv[:,:,0] <= 38)
                     & (hsv[:,:,1] >= 15) & (hsv[:,:,1] <= 135)
                     & (hsv[:,:,2] >= 175))
                    | ((hsv[:,:,1] < 30) & (hsv[:,:,2] >= 240))).astype(np.uint8)
            joined = cv2.morphologyEx(warm,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
            if side == 'right':
                valid = self._right_cores(frame,x,y,hsv,warm,joined)
                if len(valid) <= 3:
                    candidates.extend(valid)
                continue
            count,labels,stats,_ = cv2.connectedComponentsWithStats(joined)
            valid = []
            for index in range(1,count):
                a,b,w,h,area = stats[index]
                pixels = (labels == index) & (warm != 0)
                size = int(pixels.sum())
                # The pale cap has a darker shaded half. Locate its whole
                # interior, but still require a substantial bright charge.
                bright = int((pixels & (hsv[:,:,2] >= 205)).sum())
                # Native input replays exposed 30-38 px lamp reflections and
                # thin weapon fragments after the arm had already dropped.
                # Require a broad core before authorizing a stationary touch.
                if (1200 <= size <= 18000 and 50 <= w <= 140 and 40 <= h <= 200
                        and bright >= 500 and .45 <= h/w <= 3 and size/(w*h) >= .30):
                    ys,xs = np.where(pixels)
                    mx,my = x+int(np.median(xs)),y+int(np.median(ys))
                    if my <= 590 and mx <= 225:
                        # Separate x/y medians can land in a hole between
                        # disconnected highlights. Tap inside the actual core.
                        # A cap touching the ROI edge must not make that edge
                        # look farther inside the weapon than its real center.
                        interior = cv2.distanceTransform(np.pad(pixels.astype(np.uint8),1),
                            cv2.DIST_L2,5)[1:-1,1:-1]
                        _,radius,_,point = cv2.minMaxLoc(interior)
                        if radius >= 12:
                            valid.append(dict(side=side,x=x+point[0],y=y+point[1],size=size,
                                core_radius=round(radius,1),box=[int(x+a),int(y+b),int(x+a+w),int(y+b+h)]))
            # Corrupt captures or broad yellow flashes are not weapon targets.
            if len(valid) <= 3:
                candidates.extend(valid)
        if not candidates:
            return None
        candidates.sort(key=lambda c:(c['side'] == self.last_side,-c['size']))
        return candidates[0]

    @staticmethod
    def _right_cores(frame, x, y, hsv, warm, joined):
        # Touch rings cut holes through a real cap; a white combat arc can also
        # join its edge to the background. Keep the original colour/area/pose
        # guards, reconstruct only a compact current component, and select a
        # visible gold pixel inside it. No coordinates or masks survive a frame.
        for separated in (False, True):
            mask = (cv2.morphologyEx(joined,cv2.MORPH_OPEN,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
                    if separated else joined)
            count,labels,stats,_ = cv2.connectedComponentsWithStats(mask)
            valid = []
            for index in range(1,count):
                a,b,w,h,_ = map(int,stats[index])
                pixels = (labels == index) & (warm != 0)
                size = int(pixels.sum())
                bright = int((pixels & (hsv[:,:,2] >= 205)).sum())
                if not (1200 <= size <= 18000 and 50 <= w <= 140 and 40 <= h <= 200
                        and bright >= 500 and .45 <= h/w <= 1.25 and size/(w*h) >= .30):
                    continue
                ys,xs = np.where(pixels)
                mx,my = x+int(np.median(xs)),y+int(np.median(ys))
                if not (455 <= mx and 435 <= my <= 660):
                    continue
                if my > 590 and not (500 <= mx <= 555 and .90 <= h/w):
                    continue
                left,right = mx-max(1,int(w*.3)),mx+max(1,int(w*.3))
                top,bottom = y+b+h+4,y+b+h+44
                shaft = cv2.cvtColor(frame[top:bottom,left:right],cv2.COLOR_RGB2HSV)
                dark = ((shaft[:,:,0] < 40) & (shaft[:,:,1] > 70) & (shaft[:,:,2] < 80))
                shoulder = cv2.cvtColor(frame[
                    my+max(1,int(h*.20)):my+max(2,int(h*.75)),
                    mx+max(1,int(w*.08)):mx+max(2,int(w*.42))],cv2.COLOR_RGB2HSV)
                attached = ((shoulder[:,:,0] < 40) & (shoulder[:,:,1] > 70)
                            & (shoulder[:,:,2] < 80))
                if (float(dark.mean()) < .28
                        and not (h/w <= 1.20 and float(attached.mean()) >= .40)):
                    continue
                contours,_ = cv2.findContours((labels == index).astype(np.uint8),
                                               cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                hull = cv2.convexHull(max(contours,key=cv2.contourArea))
                solid = np.zeros_like(warm)
                cv2.fillConvexPoly(solid,hull,1)
                if size/float(solid.sum()) < .55:
                    continue
                interior = cv2.distanceTransform(np.pad(solid,1),cv2.DIST_L2,5)[1:-1,1:-1]
                # A translucent touch dot is often nearly grey. Do not choose
                # its centre or a point fabricated by the convex hull: retain
                # a 3x3 patch of actual bright gold in this very screenshot.
                gold = (pixels & (hsv[:,:,0] >= 10) & (hsv[:,:,0] <= 38)
                        & (hsv[:,:,1] >= 35) & (hsv[:,:,2] >= 205))
                interior *= cv2.erode(gold.astype(np.uint8),np.ones((3,3),np.uint8))
                _,radius,_,point = cv2.minMaxLoc(interior)
                if radius >= 12:
                    valid.append(dict(side='right',x=x+point[0],y=y+point[1],size=size,
                        core_radius=round(radius,1),
                        box=[x+a,y+b,x+a+w,y+b+h]))
            # Opening is only a fallback for a joined arc, never a way to
            # replace an already identified cap with another bright object.
            if valid:
                return valid
        return []

    def validate(self, frame, rank, expected):
        target = self.locate(frame,rank)
        if target is None or target['side'] != expected.get('side'):
            return None
        if abs(target['x']-expected['x']) > 32 or abs(target['y']-expected['y']) > 32:
            return None
        return target

    def burst(self, capture, hit, rank, stopped, seconds=4., max_frames=40, max_taps=12):
        """Observe a short charging window; each hit uses that fresh frame.

        The old loop found a target, then took a second picture to track the
        old coordinate. It missed short or moving cues. Reacquire the raised
        weapon from every new capture, never reuse a point from an old frame.
        """
        started = time.monotonic()
        deadline = started+seconds
        taps = 0
        tap_times = []
        defense_taps = damage_taps = 0
        context = None
        for _ in range(max_frames):
            if stopped() or time.monotonic() >= deadline:
                break
            fresh = capture()
            if stopped() or time.monotonic() >= deadline:
                break
            encounter = self.encounter(fresh,rank)
            options = self.attack_options(fresh,rank,encounter=encounter)
            owned = options['boss_interrupt'] and encounter
            footer = self._battle_footer(fresh) if owned else False
            damage = (bool(options['battle_tap'] and footer
                           and self._match(fresh,'damage-sword',(153,205,192,245),.96))
                      if owned else self.damage_battle(fresh,rank))
            if not owned and not damage:
                break
            current = self._combat_context(fresh)
            if context is not None and current != context:
                break
            context = current
            # Charging weapons always take precedence over a neutral damage
            # contact on this same new image; an obscured footer only waits.
            target = self._locate_core(fresh) if footer else None
            if target is None and damage and time.monotonic() >= self.next_damage_at:
                target = self.damage_target()
            if target is None:
                # A single animation frame may hide the glow. Keep looking
                # within this bounded window; it never authorizes a stale tap.
                continue
            if stopped():
                break
            touch_at=time.monotonic()
            if hit(target) is False:
                continue
            tap_times.append(touch_at)
            self.target = target
            if target.get('kind') == 'damage':
                damage_taps += 1
                self.next_damage_at = touch_at+.15
            else:
                defense_taps += 1
                self.last_side = target['side']
            taps += 1
            if taps >= max_taps:
                break
        self.last_burst = dict(taps=taps,defense_taps=defense_taps,damage_taps=damage_taps,
            elapsed_ms=round((time.monotonic()-started)*1000,1),
            tap_gaps_ms=[round((b-a)*1000,1) for a,b in zip(tap_times,tap_times[1:])])
        return taps

    def step(self, frame, rank):
        self.active = self.owns(frame,rank) or self.damage_battle(frame,rank)
        self.active_rank = rank if self.active else None
        self.target = None
        if not self.active:
            return False
        target = self.locate(frame,rank)
        self.target = target
        # Hold the battle UI even while neither weapon is charging.
        if (target or self.scan) and time.monotonic() >= self.next_at:
            taps = self.action(target) if self.action else 0
            if taps:
                self.bursts += 1
                self.taps += self.last_burst.get('defense_taps',taps)
                self.damage_taps += self.last_burst.get('damage_taps',0)
                if target and not self.scan:
                    self.last_side = target['side']
            self.next_at = time.monotonic()+.02
        return True


