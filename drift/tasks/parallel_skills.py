"""Own the parallel-world skill panel and guard talent-point purchases."""
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np

from equipment_upgrade import EquipmentUpgrade
from drift.tasks.cruise_options import SKILL_NAMES


class ParallelSkills:
    COST = 200
    DURATION = 3600
    STEP_SECONDS = 30
    OBSERVATION_GRACE = 8
    BOXES = {
        'title': (30, 795, 115, 837),
        'fishing': (84, 863, 225, 884),
        'growth': (80, 1050, 200, 1079),
        'points': (490, 1063, 576, 1090),
        'clock': (207, 1049, 232, 1074),
        'confirm-currency': (289, 528, 337, 573),
        'confirm-cost': (344, 540, 400, 570),
        'confirm-growth': (195, 579, 475, 616),
        'continue': (125, 764, 300, 804),
    }
    ACTIONS = {'open': (45, 1166), 'enhance': (500, 886), 'enhance2': (500, 951), 'enhance3': (500, 1015),
               'auto1': (558, 886), 'auto2': (558, 951), 'auto3': (558, 1015),
               'open_growth': (530, 1077), 'confirm': (210, 788),
               'cancel': (510, 788), 'cancel_recovery': (510, 788), 'close': (360, 700)}

    def __init__(self, root, click, event, reader, interval=300, growth_points=True,
                 main_guard=None, options=None):
        if not isinstance(growth_points, bool):
            raise ValueError('技能点加速开关必须为布尔值')
        self.root = Path(root)
        self.path = self.root / 'parallel_skill_state.json'
        self.click, self.event, self.reader = click, event, reader
        self.main_guard = main_guard or (lambda frame: False)
        self.interval, self.growth_points = float(interval), growth_points
        self.options = options
        if options is not None:
            self.growth_points = options['growth_points']
        self.screen = EquipmentUpgrade(root, None, None, None, event)
        self.assets = {}
        for name in self.BOXES:
            data = cv2.imread(str(self.root / 'assets' / 'parallel-skills' / (name+'.png')))
            if data is None:
                raise RuntimeError('平行世界技能模板缺失或损坏：'+name)
            self.assets[name] = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = cv2.imread(str(self.root / 'assets' / 'parallel' / 'world-badge.png'))
        if data is None:
            raise RuntimeError('缺少平行世界标记模板')
        self.badge = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        self.journal = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
        if not isinstance(self.journal, dict):
            raise RuntimeError('技能点购买记录无效，请检查记录文件')
        self.phase = 'idle'
        self.since = time.monotonic()
        self.next_check_at = self.next_action_at = 0.
        self.stable = 0
        self.key = None
        self.cycles = 0
        self.percent = self.balance = None
        self.opening_interrupted = False

    @property
    def active(self):
        return self.phase != 'idle'

    def status(self):
        return dict(phase=self.phase, cycles=self.cycles, fishing_percent=self.percent,
                    balance=self.balance, growth_points=self.growth_points,
                    growth_until=self.journal.get('growth_until'))

    def _save(self):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.journal, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        temporary.replace(self.path)

    def _set(self, phase):
        self.phase, self.since = phase, time.monotonic()
        self.stable, self.key = 0, None
        self.opening_interrupted = False

    def suspend(self, seconds):
        """Exclude verified battle ownership from this UI step's deadline."""
        if self.active and seconds > 0:
            self.since += seconds
            self.stable, self.key = 0, None
            if self.phase == 'opening':
                self.opening_interrupted = True

    def _seen(self, key):
        self.stable = self.stable + 1 if key == self.key else 1
        self.key = key
        return self.stable >= 2

    def _match(self, frame, name, threshold=.94):
        x, y, X, Y = self.BOXES[name]
        crop = frame[max(0, y-2):Y+2, max(0, x-2):X+2]
        score = cv2.minMaxLoc(cv2.matchTemplate(crop, self.assets[name], cv2.TM_CCOEFF_NORMED))[1]
        return bool(np.isfinite(score) and score >= threshold)

    def world(self, frame):
        crop = frame[43:76, 9:43]
        score = cv2.minMaxLoc(cv2.matchTemplate(crop, self.badge, cv2.TM_CCOEFF_NORMED))[1]
        return bool(np.isfinite(score) and score >= .92)

    def panel(self, frame):
        return (self.world(frame) and all(self._match(frame, name) for name in ('title', 'fishing', 'growth', 'points'))
                and float((frame[780:792, 120:340].max(axis=2) > 200).mean()) > .8)

    def modal(self, frame):
        return (self.world(frame) and all(self._match(frame, name, .96)
                for name in ('confirm-currency', 'confirm-cost', 'confirm-growth', 'continue'))
                and float((frame[448:477, 160:560].max(axis=2) > 200).mean()) > .8)

    def purchase_prompt(self, frame):
        # Falling rewards can cover the growth/duration text. These anchors
        # permit cancelling our interrupted prompt, never confirming a charge.
        return (self.world(frame) and all(self._match(frame,name,.96)
                for name in ('confirm-currency','confirm-cost','continue'))
                and float((frame[448:477,160:560].max(axis=2) > 200).mean()) > .8)

    def owns_purchase_prompt(self):
        balance, opened = self.journal.get('balance_before'), self.journal.get('opened_at')
        return (isinstance(balance,int) and not isinstance(balance,bool) and balance >= self.COST
                and isinstance(opened,(int,float)) and not isinstance(opened,bool)
                and np.isfinite(opened) and 0 <= opened <= time.time())

    def main(self, frame):
        return (self.world(frame) and not self.panel(frame) and not self.modal(frame)
                and not self.purchase_prompt(frame)
                and (self.screen.main(frame) or self.main_guard(frame)))

    @staticmethod
    def toggle(frame, enhancement=False, row=1):
        x = 482 if enhancement else 539
        y = (886, 951, 1015)[row-1]
        hsv = cv2.cvtColor(frame[y-21:y+22, x:x+5], cv2.COLOR_RGB2HSV)
        bright = float((hsv[:, :, 2] > 220).mean())
        saturated = float((hsv[:, :, 1] > 140).mean()) if enhancement else 0.
        if max(bright, saturated) >= .6:
            return True
        if max(bright, saturated) <= .1:
            return False
        return None

    def _digits(self, crop, hue, pattern, crop_views=None):
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        votes = []
        green = hue == (35, 90)
        views = set()
        passes = crop_views or (((100, 2), (0, 5)) if green else ((100, 2), (60, 4), (130, 4)))
        for value, padding in passes:
            ink = ((hsv[:, :, 0] >= hue[0]) & (hsv[:, :, 0] <= hue[1])
                   & (hsv[:, :, 1] >= (60 if green else 50)) & (hsv[:, :, 2] >= value))
            if green:
                ink &= hsv[:, :, 2] < 210
            ink = ink.astype(np.uint8)
            y, x = np.where(ink)
            if len(x) < 8:
                if green:
                    return None
                continue
            binary = np.repeat(np.where(ink, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
            binary = binary[max(0, y.min()-padding):min(binary.shape[0], y.max()+padding+1),
                            max(0, x.min()-padding):min(binary.shape[1], x.max()+padding+1)]
            key = (binary.shape, binary.tobytes())
            if key in views:
                continue
            views.add(key)
            result = self.reader.read(binary)
            if result.confidence < .97:
                if green:
                    return None
                continue
            if not re.fullmatch(pattern, result.text):
                return None
            votes.append(result.text)
        # Dark outlines need the lower threshold. A weak extra view cannot
        # veto two strong reads, but any strong disagreement blocks spending.
        return votes[0] if len(votes) >= 2 and len(set(votes)) == 1 else None

    def points(self, frame):
        text = self._digits(frame[1193:1230, 201:289], (110, 155), r'[0-9]{1,8}')
        return int(text) if text is not None else None

    def effect(self, frame):
        # The percentage moves with the translated subtitle. Green ink separates it.
        text = self._digits(frame[887:914, 80:390], (35, 90), r'[0-9]{1,4}%')
        return int(text[:-1]) if text else None

    def growth_active(self, frame):
        return self._match(frame, 'clock', .9)

    def _purchase_allowed(self):
        return self.growth_points and time.time() >= self.journal.get('growth_until', 0)

    def _pending(self):
        return (self._purchase_allowed() and self.journal.get('pending') is True
                and 0 <= time.time() - self.journal.get('opened_at', 0) <= 90)

    def validate(self, action, frame, x, y):
        if frame.shape[:2] != (1280, 720) or self.ACTIONS.get(action) != (int(x), int(y)):
            return False
        if action == 'open':
            return self.main(frame)
        if action == 'cancel_recovery':
            return self.phase == 'cancel_recovery' and self.owns_purchase_prompt() and self.purchase_prompt(frame)
        if action in ('confirm', 'cancel'):
            return self.modal(frame) and (self._pending() if action == 'confirm' else not self._pending())
        if not self.panel(frame):
            return False
        if action.startswith('enhance'):
            row = int(action[-1]) if action[-1].isdigit() else 1
            wanted = self.options[SKILL_NAMES[row-1]]['enhance'] if self.options else True
            actual = self.toggle(frame, True, row=row)
            return actual is not None and actual is not wanted and self.effect(frame) is not None
        if action.startswith('auto'):
            row = int(action[-1])
            wanted = self.options[SKILL_NAMES[row-1]]['auto'] if self.options else row == 1
            actual = self.toggle(frame, row=row)
            return actual is not None and actual is not wanted
        if action == 'open_growth':
            balance = self.points(frame)
            return self._purchase_allowed() and not self.growth_active(frame) and balance is not None and balance >= self.COST
        return action == 'close'

    def before_click(self, action, frame):
        # Called under the device lock, after fresh validation and before any input.
        if action == 'open_growth':
            self.journal.update(pending=True, opened_at=time.time(), balance_before=self.points(frame))
            self._save()
        elif action == 'confirm':
            self.journal.update(pending=False, growth_until=time.time()+self.DURATION, confirmed=False)
            self._save()
        elif action in ('cancel','cancel_recovery'):
            self.journal['pending'] = False
            self._save()

    def _tap(self, action, label, phase=None):
        x, y = self.ACTIONS[action]
        if self.click(x, y, label, action):
            if phase:
                self._set(phase)
            self.next_action_at = time.monotonic()+1.5
        self.stable, self.key = 0, None

    def _finish(self):
        self.cycles += 1
        self.next_check_at = time.monotonic()+self.interval
        self._set('idle')
        self.event('平行世界技能：已回到主画面，继续钓鱼与武器强化')

    def step(self, frame):
        if frame.shape[:2] != (1280, 720):
            raise RuntimeError('技能加速只支持 720×1280 模拟器画面')
        now = time.monotonic()
        if not self.active:
            if not self.world(frame):
                return False
            if self.modal(frame):
                self._set('confirm')
            elif self.owns_purchase_prompt() and self.purchase_prompt(frame):
                self._set('cancel_recovery')
            elif self.panel(frame):
                self._set('configure')
            elif now < self.next_check_at:
                return False
            elif self.main(frame):
                if self._seen('main'):
                    self._tap('open', '打开平行世界技能面板', 'opening')
                return True
            else:
                return False
        if now < self.next_action_at:
            return True
        if now-self.since > self.STEP_SECONDS:
            if self.phase == 'confirm' and self.owns_purchase_prompt():
                self.event('技能点确认未能完成，等待识别货币和取消按钮后关闭，本轮稍后重试')
                self._set('cancel_recovery')
                return True
            # A slow capture can arrive after the deadline with the expected
            # page already visible. Only finish its consecutive observations;
            # purchase phases never gain a renewed deadline.
            visible = ((self.phase == 'opening' and (self.panel(frame)
                        or (self.opening_interrupted and self.main(frame))))
                       or (self.phase == 'await_main' and self.main(frame)))
            if not visible or now-self.since > self.STEP_SECONDS+self.OBSERVATION_GRACE:
                # Navigation timeouts: abandon this round and retry later.
                # Raising here used to hard-stop the whole panel via record_failure.
                if self.phase in ('opening', 'await_main', 'closing', 'configure'):
                    delay = 30
                    phase = self.phase
                    self.next_check_at = now + delay
                    self._set('idle')
                    self.event(f'平行世界技能步骤超时（{phase}）：本轮放弃，{int(delay)}秒后重新检查')
                    return True
                raise RuntimeError('平行世界技能步骤超时：'+self.phase)
        if self.phase == 'opening':
            if self.panel(frame):
                if self._seen('panel'):
                    self._set('configure')
            elif self.opening_interrupted and self.main(frame):
                if self._seen('interrupted-opening-main'):
                    # A verified battle can dismiss a panel before we adopt it.
                    # End only this uncompleted opening attempt; do not count a
                    # skill cycle, reopen immediately, or alter a purchase record.
                    self.next_check_at = now+30
                    self._set('idle')
                    self.event('技能面板打开被战斗打断：已连续确认主画面，本轮未完成，30秒后重新检查')
            else:
                self.stable, self.key = 0, None
        elif self.phase == 'configure':
            if not self.panel(frame):
                return True
            self.percent, self.balance = self.effect(frame), self.points(frame)
            if self.options is None:
                switches = [('enhance', 1, True, True), ('auto1', 1, False, True),
                            ('auto2', 2, False, False), ('auto3', 3, False, False)]
            else:
                switches = [(('enhance' if row == 1 else 'enhance'+str(row)) if enhance else 'auto'+str(row),
                             row, enhance, self.options[name]['enhance' if enhance else 'auto'])
                            for row, name in enumerate(SKILL_NAMES, 1) for enhance in (True, False)]
            states = tuple(self.toggle(frame, enhancement=e, row=r) for _,r,e,_ in switches)
            if None in states or self.percent is None:
                if now-self.since > 10:
                    self.event('平行世界技能开关未读清，本轮跳过切换')
                    self._set('closing')
                return True
            mismatch = next((item for item,actual in zip(switches,states) if actual is not item[3]), None)
            action = mismatch[0] if mismatch else None
            if action:
                if self._seen((action, states, self.percent)):
                    _, row, enhancement, wanted = mismatch
                    self._tap(action, ('开启' if wanted else '关闭') + ('钓鱼加速','移动加速','时间加速')[row-1]
                              + ('技能强化' if enhancement else '自动续'))
                return True
            if self._seen(('configured', states)):
                self.event(f'巡航技能已按保存的开关核对，钓鱼速度 {self.percent}%')
                if self.growth_active(frame):
                    self.event('50%成长加速仍生效，本轮不消耗技能点')
                    self._set('closing')
                elif (self._purchase_allowed() and self.balance is not None and self.balance >= self.COST):
                    self._tap('open_growth', '使用200技能点补充1小时50%成长加速', 'confirm')
                else:
                    self.event('50%成长加速：余额未满足或购买记录仍在有效期，本轮跳过')
                    self._set('closing')
        elif self.phase == 'confirm':
            if self.modal(frame) and self._seen('confirm'):
                if self._pending():
                    self._tap('confirm', '确认200技能点兑换50%成长加速', 'verify')
                else:
                    self._tap('cancel', '取消重复或过期的技能点购买确认', 'configure')
            elif not self.modal(frame):
                self.stable, self.key = 0, None
        elif self.phase == 'cancel_recovery':
            if self.purchase_prompt(frame):
                if self._seen('cancel_recovery'):
                    self._tap('cancel_recovery','取消未完成的技能点加速，本轮稍后重试','closing')
            else:
                self.stable, self.key = 0, None
        elif self.phase == 'verify':
            if self.panel(frame) and self.growth_active(frame) and self._seen('active'):
                self.journal['confirmed'] = True
                self._save()
                self.event('已验证50%成长加速生效，1小时内不再购买')
                self._set('closing')
        elif self.phase == 'closing':
            if self.panel(frame) and self._seen('close'):
                self._tap('close', '关闭平行世界技能面板', 'await_main')
            elif self.main(frame) and self._seen('main'):
                self._finish()
        elif self.phase == 'await_main':
            if self.main(frame):
                if self._seen('main'):
                    self._finish()
            elif self.panel(frame) and now-self.since > 5:
                self._set('closing')
            else:
                self.stable, self.key = 0, None
        return True

# Shared reliability layer; retain all other deployed methods.
def _save(self):
    from drift.core.persistence import atomic_write_text
    atomic_write_text(self.path, json.dumps(self.journal, ensure_ascii=False, indent=2) + '\n')

ParallelSkills._save = _save



# Module-local compatibility for every authoritative record read/write.
from drift.core.persistence import ReliablePath as Path
