"""Bounded fish-cake equipment enhancement, using confirmed game controls."""
from pathlib import Path
import time

import cv2
import numpy as np


WEAPONS = {
    'rod': ('钓鱼竿', (29, 856, 90, 916), (103, 864, 177, 889), 886),
    'harpoon': ('渔叉炮塔', (29, 944, 87, 1005), (103, 951, 203, 977), 974),
    'cannon': ('大炮', (29, 1032, 82, 1093), (103, 1040, 151, 1067), 1062),
    'turret': ('机关炮塔', (29, 860, 91, 921), (103, 871, 204, 898), 892),
    'magic': ('魔法石', (29, 950, 84, 1009), (103, 960, 178, 986), 980),
}
PAGES = {'top': ('rod', 'harpoon', 'cannon'), 'bottom': ('turret', 'magic')}


class EquipmentUpgrade:
    HOLD_SECONDS = 12
    CYCLE_SECONDS = 180
    MAX_HOLDS = 5
    CLOSE_RESERVE_SECONDS = 10
    ACTION_SETTLE_SECONDS = 1

    def __init__(self, root, click, swipe, hold, event, interval=900, main_guard=None,
                 capture=None, scroll_guard=None, opening_guard=None):
        self.assets = {}
        for path in (Path(root) / 'assets' / 'equipment').glob('*.png'):
            data = cv2.imread(str(path))
            if data is None:
                raise RuntimeError('武器强化识别模板损坏')
            self.assets[path.stem] = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        required = {'entry', 'guard-star', 'guard-diamond', 'header-currency',
                    'header-tap', 'header-attack', 'arrow', 'row-currency',
                    'main-auto'}
        required.update(key + suffix for key in WEAPONS for suffix in ('-icon', '-name'))
        if set(self.assets) != required:
            raise RuntimeError('武器强化识别模板不完整')
        # Long prices move the currency icon into the arrow's lower right edge.
        # Its upper portion remains separate from both the currency and digits.
        self.assets['arrow'] = self.assets['arrow'][:22]
        # Numeric groups center their fixed titles at fractional pixels. Match
        # the title strokes without treating that antialiasing shift as a new UI.
        self.glyphs = {
            name: cv2.GaussianBlur(cv2.cvtColor(self.assets[name], cv2.COLOR_RGB2GRAY), (3, 3), .7)
            for name in ('header-tap', 'header-attack', 'row-currency')
        }
        # A new parallel-world round uses short values that center this fixed
        # title at a different fractional pixel. Preserve the same threshold
        # and all other panel guards, using its observed title strokes only.
        variant = Path(root) / 'assets' / 'recovery' / 'equipment-tap-title.png'
        if variant.exists():
            title = cv2.imread(str(variant), cv2.IMREAD_GRAYSCALE)
            if title is None or title.shape != self.glyphs['header-tap'].shape:
                raise RuntimeError('武器强化低等级标题模板损坏')
            self.glyphs['header-tap-low-level'] = cv2.GaussianBlur(title, (3, 3), .7)
        self.click, self.swipe, self.hold, self.event = click, swipe, hold, event
        self.main_guard = main_guard
        self.capture,self.scroll_guard = capture,scroll_guard
        self.opening_guard = opening_guard
        self.interval = float(interval)
        self.phase = 'idle'
        self.phase_since = time.monotonic()
        self.next_check_at = 0.
        self.next_action_at = 0.
        self.started_at = 0.
        self.stable = 0
        self.page = 'top'
        self.index = 0
        self.swipes = 0
        self.holds = 0
        self.passes = 0
        self.cycles = 0
        self.pass_holds = 0
        self.last_weapon = None
        self.interrupted = False

    @property
    def active(self):
        return self.phase != 'idle'

    def status(self):
        return {'phase': self.phase, 'weapon': self.last_weapon, 'holds': self.holds,
                'passes': self.passes, 'cycles': self.cycles,
                'hold_seconds': self.HOLD_SECONDS, 'interval_seconds': self.interval,
                'max_holds_per_cycle': self.MAX_HOLDS,
                'next_check_in_seconds': max(0,round(self.next_check_at-time.monotonic(),1))}

    def _set(self, phase):
        self.phase = phase
        self.phase_since = time.monotonic()
        self.stable = 0

    def suspend(self, seconds):
        if self.active and seconds > 0:
            self.phase_since += seconds
            if self.started_at:
                self.started_at += seconds
            self.stable = 0

    def validate_battle_close(self, frame):
        return (frame.shape[:2] == (1280,720) and self.panel(frame)
                and self._fixed(frame,'entry',(330,1143,375,1189),.9))

    def yielded_for_battle(self):
        # Wait for the actual panel disappearance before releasing ownership.
        # An interrupted pass is not counted as a completed upgrade cycle.
        self.interrupted = True
        self._set('close')
        self.next_action_at = time.monotonic()+1.2
        self.event('海神战斗：已收起鱼饼强化面板，等待主画面确认后继续')

    def yielded_for_completion(self):
        self.interrupted = True
        self._set('close')
        self.next_action_at = time.monotonic()+1.2
        self.event('平行世界探索结束：先收起未完成的强化面板，再确认通关和停航')

    def yielded_for_sailing(self):
        self.interrupted = True
        self._set('await_main')
        self.next_action_at = time.monotonic()+1.2
        self.event('BOSS 航行交接：已收起强化面板，等待主画面后确认停航；未完成强化不计完成')

    def _fixed(self, frame, name, box, threshold=.92, margin=3):
        x1, y1, x2, y2 = box
        return self._within(frame, name,
                            (max(0, x1-margin), max(0, y1-margin),
                             min(720, x2+margin), min(1280, y2+margin)), threshold)

    def _within(self, frame, name, box, threshold=.92, *, text=False):
        x1, y1, x2, y2 = box
        template = self.glyphs[name] if text else self.assets[name]
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
            return False
        if text:
            crop = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), (3, 3), .7)
        score = cv2.minMaxLoc(cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED))[1]
        return bool(np.isfinite(score) and score >= threshold)

    @staticmethod
    def _bright(frame, box, minimum=.3):
        x1, y1, x2, y2 = box
        return float((np.max(frame[y1:y2, x1:x2], axis=2) >= 210).mean()) >= minimum

    def panel(self, frame):
        return (self._fixed(frame, 'header-currency', (30, 800, 60, 829))
                # Header groups center their labels and changing numeric values together.
                # Search only the header band; upgraded damage can shift both labels.
                and (self._within(frame, 'header-tap', (285, 797, 440, 831), .95, text=True)
                     or 'header-tap-low-level' in self.glyphs
                     and self._within(frame, 'header-tap-low-level', (285, 797, 440, 831), .95, text=True))
                and self._within(frame, 'header-attack', (470, 797, 595, 831), .95, text=True)
                and self._bright(frame, (335, 784, 410, 799), .8))

    def main(self, frame):
        return (not self.panel(frame)
                and ((self._fixed(frame, 'entry', (330, 1143, 375, 1189), .9)
                and self._fixed(frame, 'guard-star', (211, 1140, 266, 1189), .93)
                and self._fixed(frame, 'guard-diamond', (542, 1140, 597, 1183), .93)
                and self._fixed(frame, 'main-auto', (279, 884, 307, 910), .92)
                and self._bright(frame, (558, 1147, 585, 1168)))
                or self.main_guard is not None and self.main_guard(frame)))

    def weapon_visible(self, frame, key):
        _, icon, name, _ = WEAPONS[key]
        return (self._fixed(frame, key+'-icon', icon, .9)
                and self._fixed(frame, key+'-name', name, .91))

    def page_visible(self, frame, page):
        return self.panel(frame) and all(self.weapon_visible(frame, key) for key in PAGES[page])

    def affordable(self, frame, key):
        if not self.panel(frame) or not self.weapon_visible(frame, key):
            return False
        y = WEAPONS[key][3]
        arrow = (535, y-36, 575, y-14)
        # The currency icon shifts horizontally as the price gains digits.
        currency = cv2.GaussianBlur(cv2.cvtColor(frame[y-22:y+24, 555:650], cv2.COLOR_RGB2GRAY),
                                    (3,3), .7)
        currency_score = cv2.minMaxLoc(cv2.matchTemplate(
            currency, self.glyphs['row-currency'], cv2.TM_CCOEFF_NORMED))[1]
        if not (self._fixed(frame, 'arrow', arrow, .87, 5)
                and np.isfinite(currency_score) and currency_score >= .82):
            return False
        crop = cv2.cvtColor(frame[y-32:y+9, 539:571], cv2.COLOR_RGB2HSV)
        green = ((crop[:, :, 0] >= 35) & (crop[:, :, 0] <= 90)
                 & (crop[:, :, 1] >= 65) & (crop[:, :, 2] >= 100))
        return float(green.mean()) >= .3

    def validate_hold(self, frame, key, x, y):
        return (frame.shape[:2] == (1280, 720) and key in WEAPONS
                and (int(x), int(y)) == (620, WEAPONS[key][3])
                and self.affordable(frame, key))

    def _begin(self):
        self.interrupted = False
        self.started_at = time.monotonic()
        self.holds = self.passes = self.pass_holds = 0
        self.index = self.swipes = 0
        self.page = 'top'
        self._set('to_top')
        self.event(f'鱼饼武器强化：五种武器各检查一次，每次长按 {self.HOLD_SECONDS} 秒，完成一遍即关闭')

    def _finish(self, interrupted=False):
        interrupted = interrupted or self.interrupted
        if not interrupted:
            self.cycles += 1
        self.next_check_at = time.monotonic() + self.interval
        self._set('idle')
        self.interrupted = False
        self.event('武器强化面板已关闭，继续其他任务' if not interrupted
                   else '武器强化面板已消失，稍后重新检查')

    def hold_budget_available(self, seconds):
        # A still-open cycle is not enough: leave time for release, settling
        # and closing the panel. This does not renew the original cycle.
        return (self.phase == 'upgrade' and 0 < seconds <= self.HOLD_SECONDS
                and time.monotonic() + seconds + self.ACTION_SETTLE_SECONDS
                    + self.CLOSE_RESERVE_SECONDS <= self.started_at + self.CYCLE_SECONDS)

    def _close_for_budget(self):
        self.interrupted = True
        self._set('close')
        self.event('武器强化：剩余时间不足以完成长按和收起面板，本轮先关闭；未完成强化不计完成')

    def step(self, frame):
        if frame.shape[:2] != (1280, 720):
            raise RuntimeError('武器强化只支持 720×1280 模拟器画面')
        now = time.monotonic()
        if now < self.next_action_at:
            return self.active
        panel = self.panel(frame)
        if self.phase == 'idle':
            if now < self.next_check_at:
                return False
            if panel:
                self._begin()
            else:
                self.stable = self.stable + 1 if self.main(frame) else 0
                if self.stable < 2:
                    return False
                if self.capture is not None:
                    frame = self.capture()
                    if frame.shape[:2] != (1280,720) or self.panel(frame) or not self.main(frame):
                        self.stable = 0
                        return False
                if self.opening_guard is not None and not self.opening_guard(frame):
                    self.stable = 0
                    return False
                self.click(360, 1166, '武器强化：打开鱼饼面板')
                self._set('opening')
                self.next_action_at = time.monotonic() + 1.2
                return True

        if self.phase == 'opening':
            if panel:
                self._begin()
            else:
                self.stable = self.stable+1 if self.main(frame) else 0
                if self.stable >= 3:
                    self._finish(True)
                    self.event('武器强化入口未展开：已连续确认主画面，等待下个检查间隔；本次不记强化完成')
                    return False
                if now-self.phase_since > 25:
                    raise RuntimeError('武器强化入口后未确认鱼饼面板')
                return True

        if not panel:
            self.stable = self.stable + 1 if self.main(frame) else 0
            if self.stable >= 3:
                self._finish(self.phase != 'await_main')
                return False
            if now-self.phase_since > 35:
                raise RuntimeError('武器强化界面被遮挡或未返回主画面，已停止')
            return True

        if self.phase == 'await_main':
            if now-self.phase_since > 30:
                raise RuntimeError('武器强化面板未关闭，已停止')
            return True

        if (self.phase != 'close' and self.holds < self.MAX_HOLDS
                and time.monotonic()-self.started_at >= self.CYCLE_SECONDS-self.CLOSE_RESERVE_SECONDS):
            self._close_for_budget()
        if self.holds >= self.MAX_HOLDS:
            self._set('close')

        if self.phase == 'close':
            self.click(360, 1166, '武器强化：关闭鱼饼面板')
            self._set('await_main')
            self.next_action_at = now + 1.2
            return True

        if self.phase in ('to_top', 'to_bottom'):
            self.page = 'top' if self.phase == 'to_top' else 'bottom'
            if self.capture is not None:
                # Header OCR can consume the whole settling interval while the
                # outer frame still shows the previous scroll's elastic bounce.
                # Observe NEW pixels after that interval, before another swipe
                # or exhausting the original three-swipe limit.
                frame = self.capture()
                if frame.shape[:2] != (1280,720) or not self.panel(frame) or (self.scroll_guard is not None
                                             and not self.scroll_guard(frame)):
                    return True
                if time.monotonic()-self.started_at >= self.CYCLE_SECONDS:
                    self._close_for_budget()
                    return True
            if self.page_visible(frame, self.page):
                self.index = self.swipes = 0
                self._set('upgrade')
                return True
            if self.swipes >= 3:
                raise RuntimeError('武器强化列表滚动后未确认武器名称，已停止')
            self.swipe(360, 870 if self.page == 'top' else 1070,
                       360, 1070 if self.page == 'top' else 870,
                       '武器强化：滚动至' + ('钓鱼竿' if self.page == 'top' else '机关炮塔和魔法石'))
            self.swipes += 1
            # Count from the completed gesture, not from controller entry.
            self.next_action_at = time.monotonic() + 1.
            return True

        if self.phase == 'upgrade':
            if not self.page_visible(frame, self.page):
                if now-self.phase_since > 25:
                    raise RuntimeError('武器强化列表位置变化，已停止')
                return True
            keys = PAGES[self.page]
            if self.index >= len(keys):
                if self.page == 'top':
                    self._set('to_bottom')
                else:
                    self.passes += 1
                    self._set('close')
                    self.pass_holds = 0
                return True
            key = keys[self.index]
            self.stable += 1
            if self.stable < 2:
                return True
            if self.affordable(frame, key):
                if not self.hold_budget_available(self.HOLD_SECONDS):
                    self._close_for_budget()
                    return True
                name, _, _, y = WEAPONS[key]
                if not self.hold(620, y, self.HOLD_SECONDS, key, '武器强化：长按'+name):
                    self.stable = 0
                    return True
                self.last_weapon = name
                self.holds += 1
                self.pass_holds += 1
                self.next_action_at = time.monotonic() + .8
            self.index += 1
            self.stable = 0
            self.phase_since = time.monotonic()
            return True
        return True


# Reviewed source-sprite recognition; controller guards and input logic unchanged.
from drift.perception.original_templates import install_original_templates as _install_original_templates
_install_original_templates(EquipmentUpgrade, 'equipment', '_within')

# Reviewed source-sprite recognition; controller guards and input logic unchanged.
from drift.perception.original_templates import install_original_templates as _install_original_templates
_install_original_templates(EquipmentUpgrade, 'equipment', '_within')

# Bounded long-action budgets; preserve all other deployed methods.

EquipmentUpgrade.CLOSE_RESERVE_SECONDS = 10

EquipmentUpgrade.ACTION_SETTLE_SECONDS = 1

def step(self, frame):
    if frame.shape[:2] != (1280, 720):
        raise RuntimeError('武器强化只支持 720×1280 模拟器画面')
    now = time.monotonic()
    if now < self.next_action_at:
        return self.active
    panel = self.panel(frame)
    if self.phase == 'idle':
        if now < self.next_check_at:
            return False
        if panel:
            self._begin()
        else:
            self.stable = self.stable + 1 if self.main(frame) else 0
            if self.stable < 2:
                return False
            if self.capture is not None:
                frame = self.capture()
                if frame.shape[:2] != (1280,720) or self.panel(frame) or not self.main(frame):
                    self.stable = 0
                    return False
            if self.opening_guard is not None and not self.opening_guard(frame):
                self.stable = 0
                return False
            self.click(360, 1166, '武器强化：打开鱼饼面板')
            self._set('opening')
            self.next_action_at = time.monotonic() + 1.2
            return True

    if self.phase == 'opening':
        if panel:
            self._begin()
        else:
            self.stable = self.stable+1 if self.main(frame) else 0
            if self.stable >= 3:
                self._finish(True)
                self.event('武器强化入口未展开：已连续确认主画面，等待下个检查间隔；本次不记强化完成')
                return False
            if now-self.phase_since > 25:
                raise RuntimeError('武器强化入口后未确认鱼饼面板')
            return True

    if not panel:
        self.stable = self.stable + 1 if self.main(frame) else 0
        if self.stable >= 3:
            self._finish(self.phase != 'await_main')
            return False
        if now-self.phase_since > 35:
            raise RuntimeError('武器强化界面被遮挡或未返回主画面，已停止')
        return True

    if self.phase == 'await_main':
        if now-self.phase_since > 30:
            raise RuntimeError('武器强化面板未关闭，已停止')
        return True

    if (self.phase != 'close' and self.holds < self.MAX_HOLDS
            and time.monotonic()-self.started_at >= self.CYCLE_SECONDS-self.CLOSE_RESERVE_SECONDS):
        self._close_for_budget()
    if self.holds >= self.MAX_HOLDS:
        self._set('close')

    if self.phase == 'close':
        self.click(360, 1166, '武器强化：关闭鱼饼面板')
        self._set('await_main')
        self.next_action_at = now + 1.2
        return True

    if self.phase in ('to_top', 'to_bottom'):
        self.page = 'top' if self.phase == 'to_top' else 'bottom'
        if self.capture is not None:
            # Header OCR can consume the whole settling interval while the
            # outer frame still shows the previous scroll's elastic bounce.
            # Observe NEW pixels after that interval, before another swipe
            # or exhausting the original three-swipe limit.
            frame = self.capture()
            if frame.shape[:2] != (1280,720) or not self.panel(frame) or (self.scroll_guard is not None
                                         and not self.scroll_guard(frame)):
                return True
            if time.monotonic()-self.started_at >= self.CYCLE_SECONDS:
                self._close_for_budget()
                return True
        if self.page_visible(frame, self.page):
            self.index = self.swipes = 0
            self._set('upgrade')
            return True
        if self.swipes >= 3:
            raise RuntimeError('武器强化列表滚动后未确认武器名称，已停止')
        self.swipe(360, 870 if self.page == 'top' else 1070,
                   360, 1070 if self.page == 'top' else 870,
                   '武器强化：滚动至' + ('钓鱼竿' if self.page == 'top' else '机关炮塔和魔法石'))
        self.swipes += 1
        # Count from the completed gesture, not from controller entry.
        self.next_action_at = time.monotonic() + 1.
        return True

    if self.phase == 'upgrade':
        if not self.page_visible(frame, self.page):
            if now-self.phase_since > 25:
                raise RuntimeError('武器强化列表位置变化，已停止')
            return True
        keys = PAGES[self.page]
        if self.index >= len(keys):
            if self.page == 'top':
                self._set('to_bottom')
            else:
                self.passes += 1
                self._set('close')
                self.pass_holds = 0
            return True
        key = keys[self.index]
        self.stable += 1
        if self.stable < 2:
            return True
        if self.affordable(frame, key):
            if not self.hold_budget_available(self.HOLD_SECONDS):
                self._close_for_budget()
                return True
            name, _, _, y = WEAPONS[key]
            if not self.hold(620, y, self.HOLD_SECONDS, key, '武器强化：长按'+name):
                self.stable = 0
                return True
            self.last_weapon = name
            self.holds += 1
            self.pass_holds += 1
            self.next_action_at = time.monotonic() + .8
        self.index += 1
        self.stable = 0
        self.phase_since = time.monotonic()
        return True
    return True

EquipmentUpgrade.step = step

def hold_budget_available(self, seconds):
    # A still-open cycle is not enough: leave time for release, settling
    # and closing the panel. This does not renew the original cycle.
    return (self.phase == 'upgrade' and 0 < seconds <= self.HOLD_SECONDS
            and time.monotonic() + seconds + self.ACTION_SETTLE_SECONDS
                + self.CLOSE_RESERVE_SECONDS <= self.started_at + self.CYCLE_SECONDS)

EquipmentUpgrade.hold_budget_available = hold_budget_available

def _close_for_budget(self):
    self.interrupted = True
    self._set('close')
    self.event('武器强化：剩余时间不足以完成长按和收起面板，本轮先关闭；未完成强化不计完成')

EquipmentUpgrade._close_for_budget = _close_for_budget
