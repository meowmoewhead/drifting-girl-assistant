"""Configure verified resource or early BOSS AUTO routes with a stopped handoff."""
from pathlib import Path
import time

import cv2
import numpy as np


TARGETS = {'pearl': (272, True, '珍珠'), 'material': (345, False, '天赋材料'),
           'amber': (417, True, '琥珀'), 'gem': (490, True, '蓝色宝石'),
           'boss': (562, False, 'BOSS')}


class TrainingSailing:
    def __init__(self, root, training, action, event, fishing=None, context_check=None):
        self.training, self.action, self.event = training, action, event
        self.fishing = fishing
        self.context_check = context_check
        self.assets = {}
        for path in (Path(root)/'assets/training-sailing').glob('*.png'):
            data = cv2.imread(str(path))
            if data is None:
                raise RuntimeError('自动航行识别模板损坏')
            self.assets[path.stem] = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        required = {'auto-label', 'inactive-face', 'active-arrow', 'menu-target',
                    'menu-destination', 'menu-start', 'menu-close', 'checkbox-off', 'checkbox-on',
                    'main-auto-sailing','main-bottle'}
        required.update(name+'-icon' for name in TARGETS)
        if set(self.assets) != required:
            raise RuntimeError('自动航行识别模板不完整')
        self.phase = 'idle'
        self.since = time.monotonic()
        self.next_at = 0.
        self.stable = 0
        self.cycles = 0
        self.toggles = 0
        self.sleep_since = None
        self.wake_off_seen = 0
        self.wake_confirmed_before = 0
        self.meter_since = None
        self.meter_key = None
        self.next_meter_check = 0.
        self.next_level_check = 0.
        self.profile = training.state.get('sailing_profile','resources')
        self.route_rank = training.state.get('sailing_started_rank')

    def targets(self):
        return {name:(x, name == 'boss' if self.profile == 'boss' else selected, label)
                for name,(x,selected,label) in TARGETS.items()}

    @property
    def active(self):
        return self.phase != 'idle'

    def status(self):
        return {'phase': self.phase, 'running': self.training.state.get('sailing_enabled', False),
                'profile': self.profile,
                'targets': ['BOSS'] if self.profile == 'boss' else ['珍珠', '琥珀', '蓝色宝石'], 'cycles': self.cycles,
                'fishing_preparation': bool(self.training.state.get('sailing_sleep_recovery'))}

    def _fixed(self, frame, name, box, threshold=.94):
        x, y, X, Y = box
        if frame.shape[:2] != (1280,720):
            return False
        template = self.assets[name]
        crop = frame[max(0,y-2):min(1280,Y+2),max(0,x-2):min(720,X+2)]
        score = cv2.minMaxLoc(cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED))[1]
        return np.isfinite(score) and score >= threshold

    def running(self, frame):
        if frame.shape[:2] != (1280,720):
            return None
        # Sailing changes the lower AUTO icon, and flying fishcakes hide the
        # center footer button. Use three unobscured footer controls here while
        # retaining the lower AUTO and brightness guards. Fishing's separate
        # idle/casting checks are unchanged.
        main = self.training.main(frame)
        if not main:
            boxes = {'guard-star':(208,1137,269,1192),'guard-diamond':(539,1137,600,1186)}
            # The glow changes the checkbox background; its check strokes
            # remain fixed. Require those and the full surrounding outline.
            template = self.assets['main-auto-sailing'][8:27,9:27]
            check = cv2.minMaxLoc(cv2.matchTemplate(frame[887:910,283:305],template,
                                                  cv2.TM_CCOEFF_NORMED))[1]
            lower = (self.training._match(frame,'main-auto',(276,881,310,913),.91)
                     or np.isfinite(check) and check>=.95
                     and self._fixed(frame,'main-auto-sailing',(276,881,310,913),.88))
            main = (lower
                    and self._fixed(frame,'main-bottle',(25,1147,70,1185),.95)
                    and all(self.training._match(frame,name,box,.91)
                            for name,box in boxes.items())
                    and float((np.max(frame[1147:1168,558:585],axis=2)>=210).mean())>=.30)
        if not main or not self._fixed(frame,'auto-label',(186,34,253,65),.90):
            return None
        if self._fixed(frame,'active-arrow',(194,4,246,42),.96):
            return True
        if self._fixed(frame,'inactive-face',(194,4,246,42),.96):
            return False
        return None

    def menu(self, frame):
        return (self._fixed(frame,'menu-target',(150,439,207,475))
                and self._fixed(frame,'menu-destination',(128,531,226,564))
                and self._fixed(frame,'menu-start',(246,850,491,895))
                and self._fixed(frame,'menu-close',(564,330,639,397))
                and float((frame[850:895,246:491].min(axis=2) >= 245).mean()) >= .04)

    def selected(self, frame, name):
        x = TARGETS[name][0]
        if not self.menu(frame) or not self._fixed(frame,name+'-icon',(x-24,432,x+24,484),.92):
            return None
        # The tile's corners brighten from V~193 to V~254 when selected.
        # Require all four corners, rather than the gem's own bright center.
        values = [cv2.cvtColor(frame[y:y+8,X:X+8],cv2.COLOR_RGB2HSV)[:,:,2].mean()
                  for y in (428,477) for X in (x-26,x+17)]
        if all(value >= 240 for value in values):
            return True
        if all(160 <= value <= 215 for value in values):
            return False
        return None

    def extras_off(self, frame):
        return (self._fixed(frame,'checkbox-off',(557,627,595,667),.97)
                and self._fixed(frame,'checkbox-off',(557,741,595,781),.97))

    def sale_selected(self, frame):
        """Read the actual sale checkbox; a missing OFF match is not ON proof."""
        if not self.menu(frame):
            return None
        on = self._fixed(frame,'checkbox-on',(557,627,595,667),.97)
        off = self._fixed(frame,'checkbox-off',(557,627,595,667),.97)
        return on if on != off else None

    def upgrade_selected(self, frame):
        """Read the auto-upgrade checkbox; same convention as sale_selected."""
        if not self.menu(frame):
            return None
        on = self._fixed(frame, 'checkbox-on', (557, 741, 595, 781), .97)
        off = self._fixed(frame, 'checkbox-off', (557, 741, 595, 781), .97)
        return on if on != off else None

    def configured(self, frame):
        return (self.menu(frame) and self.extras_off(frame)
                and all(self.selected(frame,name) is wanted for name,(_,wanted,_) in self.targets().items()))

    def validate(self, action, frame):
        if self.context_check is not None and not self.context_check(action,frame):
            return False
        if action == 'open':
            return self.running(frame) is False
        if action == 'stop':
            return self.running(frame) is True
        if action == 'close':
            return self.menu(frame)
        if action == 'start':
            return self.configured(frame)
        if action.startswith('target:'):
            name = action.split(':',1)[1]
            if name in TARGETS:
                selected = self.selected(frame,name)
                return selected is not None and selected is not self.targets()[name][1]
        return False

    def before_click(self, action):
        if action == 'start':
            # A restart between the input and confirmation must still know to
            # stop a potentially active route before returning to the boss.
            self.training.set_sailing(True, verified=False,profile=self.profile,rank=self.route_rank)

    def _set(self, phase):
        if self.phase != phase:
            if self.phase == 'idle' or phase == 'idle':
                self.since = time.monotonic()
            self.phase, self.stable = phase,0

    def _click(self, action, x, y, label, phase):
        if not self.active:
            self._set('validating')
        if self.action(action,x,y,label) is not False:
            self._set(phase)
            self.stable = 0
        self.next_at = time.monotonic()+1.5
        return True

    def _timeout(self):
        limit = 90 if self.phase.startswith('wake_') else 40
        if time.monotonic()-self.since > limit:
            raise RuntimeError('自动航行画面或选项未能确认，已停止并保留练级记录')

    def _save_wake(self, wanted):
        if bool(self.training.state.get('sailing_sleep_recovery')) == wanted:
            return
        if wanted:
            self.training.state['sailing_sleep_recovery'] = True
        else:
            self.training.state.pop('sailing_sleep_recovery', None)
            self.training.state.pop('sailing_meter_adjustment', None)
        self.training._save()

    def _finish_wake(self):
        self._save_wake(False)
        self.sleep_since = None
        self.wake_off_seen = 0
        self.next_meter_check = time.monotonic()+30
        self.meter_key,self.meter_since = None,None
        self._set('idle')
        self.event('睡眠恢复：已确认大球浅蓝填充，重新核对三个资源目标并恢复航行')

    def _adopt_new_start(self, frame, wanted):
        # A held gesture can finish just before the workflow's next deadline
        # check. Commit its verified result before that check discards it.
        # This path issues no input or further meter preparation. Uncompleted
        # workflows retain their original deadline and pending journal.
        if (not wanted or self.phase != 'wake_prepare' or self.fishing is None
                or not self.training.state.get('sailing_sleep_recovery')
                or (self.training.state.get('sailing_meter_adjustment')
                    and not self.training.options['meter_auto_adjust'])
                or self.wake_off_seen < 2
                or not self.fishing.start_fill_verified
                or self.fishing.confirmed_starts <= self.wake_confirmed_before):
            return False
        if (self.context_check is None or not self.context_check('recover',frame)
                or self.running(frame) is not False or not self.fishing.world_matches(frame)
                or self.training.boss_mode(frame) or self.fishing.at_port(frame)
                or not (self.fishing.blue_fill(frame) or self.fishing.real_battle(frame))):
            return False
        self._finish_wake()
        return True

    def _wake(self, frame, wanted, running):
        """A running route is not proof of fishing; wake only after stopping it."""
        pending = bool(self.training.state.get('sailing_sleep_recovery'))
        if self.training.state.get('sailing_meter_adjustment') and not self.training.options['meter_auto_adjust']:
            wanted = False
        if not wanted:
            self.sleep_since = None
            if pending:
                self._save_wake(False)
                self.wake_off_seen = 0
                self._set('await_off')
            return None
        if self.fishing is None:
            if pending:
                raise RuntimeError('睡眠恢复缺少钓鱼控制器，已保留练级记录')
            return None
        if running is not None and not self.fishing.world_matches(frame):
            self.sleep_since = self.meter_since = self.meter_key = None
            if self.wake_off_seen < 2:
                self.wake_off_seen = 0
            self.fishing.idle_since = None
            return True
        if self.fishing.at_port(frame,main_known=running is not None):
            self.sleep_since = self.meter_since = self.meter_key = None
            self.next_meter_check = 0.
            if pending:
                self._save_wake(False)
                self.wake_off_seen = 0
                self.fishing.idle_since = None
                self.fishing.phase = 'detect'
                self.fishing.drags = self.fishing.starts = self.fishing.cancels = 0
                self._set('idle')
                self.event('港口圆盘显示船锚：先恢复资源航行，离港后再校准钓鱼计量')
            return None
        if not pending:
            known = running is not None
            sleeping = (known and not self.training.boss_mode(frame)
                        and self.fishing.idle_icon(frame, main_known=known) == 'sleep'
                        and not self.fishing.busy(frame, main_known=known))
            now = time.monotonic()
            adjustment = False
            if (known and self.training.options['meter_auto_adjust']
                    and now >= self.next_meter_check and not self.training.boss_mode(frame)):
                data = self.fishing.measure(frame,main_known=True)
                trigger_goal = data.get('recalibration_goal',data['goal']) if data is not None else None
                low = (trigger_goal is not None
                       and trigger_goal-data['value'] >= self.fishing.RECALIBRATE_GAP)
                key = (data['value'],data['context']) if low else None
                if key is not None:
                    self.fishing.current,self.fishing.player,self.fishing.goal = data['value'],data['player'],data['goal']
                    if key != self.meter_key:
                        self.meter_key,self.meter_since = key,now
                    adjustment = now-self.meter_since >= self.fishing.IDLE_SECONDS
                    if not adjustment and not sleeping:
                        return True
                else:
                    self.meter_key,self.meter_since = None,None
                    self.next_meter_check = now+(30 if data is not None else 2)
            if not sleeping and not adjustment:
                self.sleep_since = None
                return None
            if sleeping and not adjustment:
                if self.sleep_since is None:
                    self.sleep_since = now
                if now-self.sleep_since < self.fishing.IDLE_SECONDS:
                    return True
            if adjustment:
                self.training.state['sailing_meter_adjustment'] = True
            self._save_wake(True)
            self._set('wake_stop')
            self.wake_off_seen = 0
            self.wake_confirmed_before = self.fishing.confirmed_starts
            self.fishing.idle_since = None
            self.fishing.cancels = 0
            self.event('连续确认钓鱼计量偏低：先停止航行，等待休眠后调高并长按钓鱼'
                       if adjustment else '连续确认角色睡眠：先停止航行，再调计量并长按启动钓鱼')
        elif not self.phase.startswith('wake_'):
            self._set('wake_stop')
            self.wake_off_seen = 0
            self.wake_confirmed_before = self.fishing.confirmed_starts

        self._timeout()
        if self.menu(frame):
            self.wake_off_seen = 0
            return self._click('close',600,360,'睡眠恢复：关闭航行设置，核对停止状态','wake_stop')
        if running is True:
            self.wake_off_seen = 0
            if self.phase == 'wake_wait_off':
                return True
            return self._click('stop',233,45,'睡眠恢复：停止航行后启动钓鱼','wake_wait_off')
        if running is not False:
            # After two confirmed OFF frames, an obscuring catch animation
            # blocks input but need not undo that proof. A fresh ON observation
            # does reset it above; every gesture still rechecks current OFF.
            if self.wake_off_seen < 2:
                self.wake_off_seen = 0
            self.fishing.idle_since = None
            return True
        if self.wake_off_seen < 2:
            self.wake_off_seen += 1
            if self.wake_off_seen < 2:
                return True
            self.training.set_sailing(False)
            self.fishing.idle_since = None
            self._set('wake_prepare')
            self.event('睡眠恢复：已连续确认航行停止，开始核对计量和钓鱼状态')
        # A cast, bright AUTO text or an XP gain cannot finish preparation.
        # A newly issued hold needs continuous post-release fishing proof.
        adjusting = self.training.state.get('sailing_meter_adjustment',False)
        filled = self.fishing.blue_fill(frame)
        current = self.fishing.measure(frame) if adjusting and filled else None
        if current is not None:
            self.fishing.current,self.fishing.player,self.fishing.goal = current['value'],current['player'],current['goal']
        pending_start = self.fishing.phase == 'starting' or self.fishing.starts > 0
        if pending_start:
            self.fishing.confirm_start(frame)
        if (filled and not pending_start and (not adjusting or current is not None and self.fishing.close_enough(current))
                or self.fishing.confirmed_starts > self.wake_confirmed_before):
            self._finish_wake()
            return True
        (self.fishing.prepare if adjusting else self.fishing.step)(frame)
        # Finish only the continuously verified post-release observation.
        if (self.fishing.start_fill_verified
                and self.fishing.confirmed_starts > self.wake_confirmed_before):
            self._finish_wake()
        return True

    def step(self, frame, wanted, *, profile='resources', rank=None):
        now = time.monotonic()
        if profile not in ('resources','boss'):
            raise ValueError('自动航行路线无效')
        if wanted and profile != self.profile:
            if self.active or self.training.state.get('sailing_enabled'):
                # Never reinterpret a green arrow as proof of another route.
                return self.step(frame,False,profile=self.profile)
            self.profile = profile
        if wanted and self.profile == 'boss':
            if type(rank) is not int or not 0 <= rank < 39:
                raise ValueError('BOSS 自动航行当前 RANK 无效')
            self.route_rank = rank
        resource = self.profile == 'resources'
        if resource and self._adopt_new_start(frame,wanted):
            return True
        if self.active:
            self._timeout()
        if now < self.next_at:
            return self.active
        menu, running = self.menu(frame), self.running(frame)
        if menu and self.context_check is not None and not self.context_check('menu',frame):
            # A recognized settings panel owns the UI, even when its dimmed
            # background cannot prove the current world/RANK. Wait boundedly;
            # never let another workflow or a saved RANK authorize its inputs.
            if not self.active:
                self._set('validating')
            self.stable = 0
            self._timeout()
            return True
        if (resource and wanted and self.training.state.get('sailing_sleep_recovery') and running is not None
                and self.fishing and self.fishing.world_matches(frame)):
            if (self.training.state.get('sailing_meter_adjustment') and running is False
                    and self.wake_off_seen >= 2
                    and self.fishing.confirmed_starts <= self.wake_confirmed_before
                    and self.fishing.cancellable_cast(frame)):
                self._timeout()
                return self.fishing.prepare(frame)
            if now >= self.next_level_check:
                self.training.levels(frame,main_known=True)
                self.next_level_check = now+8
        if resource and wanted and self.training.ready_to_retry():
            wanted = False
        wake = self._wake(frame, wanted, running) if resource else None
        if wake is not None:
            return wake
        owned = self.training.state.get('sailing_enabled',False)
        if not wanted:
            if menu:
                return self._click('close',600,360,'自动航行：关闭设置，交还导航控制','await_off')
            if running is True:
                if self.phase == 'await_off':
                    self._timeout()
                    return True
                return self._click('stop',233,45,'自动航行：停止当前路线，交还导航控制','await_off')
            if running is False:
                if not owned and not self.active:
                    return False
                self.stable += 1
                if self.stable >= 2:
                    self.training.set_sailing(False)
                    self._set('idle')
                    self.event('已确认自动航行停止，可以执行其他导航任务')
                    return False
                return True
            self.stable = 0
            if owned or self.active:
                self._set('await_off')
                self._timeout()
                return True
            return False

        if running is True:
            if self.phase == 'await_active':
                self.stable += 1
                if self.stable >= 2:
                    self.training.set_sailing(True, verified=True,profile=self.profile,rank=self.route_rank)
                    self.cycles += 1
                    self.toggles = 0
                    self._set('idle')
                    self.event('已确认 BOSS 自动航行：仅 BOSS 目标' if not resource else
                               '已确认自动航行练级：珍珠、琥珀、蓝色宝石')
                    return False
                return True
            if owned and self.training.state.get('sailing_verified') and not self.active:
                return False
            if self.phase == 'await_config_off':
                self._timeout()
                return True
            # An unverified pre-existing route may include the boss. Stop it
            # once, configure the exact resource targets, and resume it safely.
            return self._click('stop',233,45,'自动航行：暂停旧路线，核对练级目标','await_config_off')
        if menu:
            self._set('configure')
            states = {name:self.selected(frame,name) for name in TARGETS}
            if any(value is None for value in states.values()) or not self.extras_off(frame):
                self.stable = 0
                self._timeout()
                return True
            for name,(x,selected,label) in self.targets().items():
                if states[name] is not selected:
                    if self.toggles >= 10:
                        raise RuntimeError('自动航行目标多次切换仍未确认，已停止')
                    self.toggles += 1
                    return self._click('target:'+name,x,458,
                                       '自动航行：'+('勾选' if selected else '取消')+label,'configure')
            self.stable += 1
            if self.stable >= 2:
                return self._click('start',360,872,('自动航行：开始仅 BOSS 目标的前期路线' if not resource
                                   else '自动航行：开始三个资源目标的练级路线'),'await_active')
            return True
        if self.phase == 'await_active':
            self.stable = 0
            self._timeout()
            return True
        if running is False:
            self.training.set_sailing(False)
            return self._click('open',233,45,'自动航行：打开'+('BOSS' if not resource else '练级')+'目标设置','opening')
        self.stable = 0
        if not self.active:
            return False
        self._timeout()
        return True
