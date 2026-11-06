"""Bounded recovery of an idle native route, without changing its options."""
from __future__ import annotations

import hashlib
import json
import time


class CruiseView:
    """Read actual world/RANK once; bind each input to new fixed-text pixels."""

    def __init__(self, owner):
        self.owner = owner

    def handoff_due(self, current):
        check = getattr(self.owner, 'handoff_due', None)
        return (check(current) if callable(check) else
                current['stage'] == self.owner.START_STAGE
                and current['rank'] >= self.owner.HANDOFF_RANK)

    def signature(self, frame, menu):
        parts = []
        for x, y, X, Y in ((71, 9, 136, 59), (80, 9, 130, 59)):
            crop = frame[y:Y, x:X].min(axis=2)
            parts.append((crop >= crop.max() * .84).tobytes())
        crop = frame[45:71, 40:68].astype('int16')
        for threshold in ((80, 90) if menu else (110, 130)):
            ink = ((crop[:, :, 1] - crop[:, :, 0] >= 10)
                   & (crop[:, :, 2] - crop[:, :, 0] >= 15)
                   & (crop.max(axis=2) >= threshold))
            parts.append(ink.tobytes())
        parts.append((frame[12:34, 286:434].max(axis=2) < (35 if menu else 60)).tobytes())
        return hashlib.sha256(b''.join(parts)).hexdigest()

    def observe(self, frame):
        who = self.owner
        if not who.enabled() or who.active:
            return None
        current = who.screen(frame)
        if (current is None or current['completion'] or current['rank'] >= 40
                or self.handoff_due(current)):
            # Portal handoffs have priority even before their journal starts.
            return None
        menu = current['menu']
        settings = who.settings_key(frame) if menu else None
        if menu and (settings is None or who.sailing.sale_selected(frame) is not True):
            return None
        return dict(scene=(current['stage'], current['rank']), menu=menu,
                    route=current['route'], settings=settings,
                    signature=self.signature(frame, menu))

    def validate(self, kind, frame, expected):
        who = self.owner
        if (not who.enabled() or who.active or frame.shape != (1280, 720, 3)
                or not who._parallel(frame) or not who._off_icon(frame)):
            return False
        menu = kind == 'cruise-start'
        if (who.sailing.menu(frame) != menu
                or self.signature(frame, menu) != expected['signature']):
            return False
        if menu:
            return (expected['settings'] is not None
                    and who.settings_key(frame) == expected['settings']
                    and who.sailing.sale_selected(frame) is True)
        return who.sailing.running(frame) is False

    def observe_setup(self, frame):
        who=self.owner
        if not who.enabled() or who.active:return None
        current=who.screen(frame)
        if (current is None or current['completion'] or current['rank']>=40
                or self.handoff_due(current)):
            return None
        menu=current['menu']
        if not menu and not who.fishing._main_visible(frame):return None
        return dict(scene=(current['stage'],current['rank']),menu=menu,
                    signature=self.signature(frame,menu))

    def validate_setup(self, frame, expected, point, desired):
        from drift.tasks.cruise_setup import OPEN_AT,START_AT,SALE_AT,UPGRADE_AT,TARGET_X,TARGET_Y
        who=self.owner
        if (expected is None or frame.shape!=(1280,720,3) or not who.enabled() or who.active
                or not who._parallel(frame) or not who._off_icon(frame)):
            return False
        menu=who.sailing.menu(frame)
        if menu!=expected['menu'] or self.signature(frame,menu)!=expected['signature']:return False
        if point==OPEN_AT:
            return not menu and who.fishing._main_visible(frame) and who.sailing.running(frame) is False
        if not menu:return False
        values={name:who.sailing.selected(frame,name) for name in desired}
        sale,upgrade=who.sailing.sale_selected(frame),who.sailing.upgrade_selected(frame)
        if any(v is None for v in values.values()) or sale is None or upgrade is None:return False
        if point==START_AT:return values==desired and sale is True and upgrade is True
        if point==SALE_AT:return sale is False
        if point==UPGRADE_AT:return upgrade is False
        return any(point==(TARGET_X[name],TARGET_Y) and values[name] is not desired[name] for name in desired)


def fresh_cruise_input(*, capture, validate, allowed, touch, clock=time.monotonic, rejected=None):
    """One new frame and at most one contact; rejection never replays a tap."""
    began = captured = validated = None
    def reject(reason):
        if rejected is not None:
            now = clock()
            detail = dict(reason=reason,
                capture_ms=None if captured is None else round((captured-began)*1000, 3),
                validation_ms=None if validated is None else round((validated-captured)*1000, 3),
                elapsed_ms=None if began is None else round((now-began)*1000, 3))
            try:
                rejected(detail)
            except Exception:
                pass  # An optional diagnostic cannot turn a refusal into input/failure.
        return None
    if not allowed():
        return reject('gate_before_capture')
    began = clock()
    fresh = capture()
    captured = clock()
    if not allowed():
        return reject('gate_after_capture')
    valid = validate(fresh)
    validated = clock()
    if not valid:
        return reject('frame_guard')
    if not allowed():
        return reject('gate_after_validate')
    if clock() - began > .120:
        return reject('stale_frame')
    try:
        receipt = touch(fresh, began)
    except InterruptedError:
        # The native transport cancels before DOWN when a deadline/stop
        # changes during handle preparation. It always releases after DOWN.
        # The caller's stop flag/total limits remain effective; no retry here.
        return reject('transport_cancelled')
    return (fresh, receipt) if receipt is not None else reject('no_receipt')


class CruiseWatchdog:
    QUIET_SECONDS = 90.0
    RETRY_SECONDS = 600.0
    MENU_WAIT = 20.0
    START_WAIT = 30.0
    CONFIRM_FRAMES = 3
    CONFIRM_SECONDS = 3.0
    MAX_FRAME_GAP = 8.0
    OPEN_AT = (233, 45)
    START_AT = (360, 873)

    def __init__(self, *, running, observe, click, event, clock=time.monotonic):
        self.running, self.observe, self.click = running, observe, click
        self.event, self.clock = event, clock
        self.state = 'idle'
        self.since = self.deadline = None
        self.retry_at = 0.0
        self.failures = 0
        self.hold = None
        self.scene = None
        self.expected = None
        self.input_rejections = {}
        self.last_input_rejection = None
        self.rejection_report_at = None
        self._clear_confirmation()

    @property
    def busy(self):
        return self.state in ('opening', 'starting')

    def status(self):
        return dict(state=self.state, since=self.since, deadline=self.deadline,
                    retry_at=self.retry_at, failures=self.failures, hold=self.hold,
                    confirmed_frames=self._seen, input_rejections=dict(self.input_rejections),
                    last_input_rejection=dict(self.last_input_rejection) if self.last_input_rejection else None)

    def observe_input_rejection(self, kind, detail):
        # This runs only AFTER refusing the input, never inside the 120 ms contact path.
        key = kind+':'+detail['reason']
        if key not in self.input_rejections and len(self.input_rejections) >= 24:
            key = 'other'
        self.input_rejections[key] = self.input_rejections.get(key, 0)+1
        self.last_input_rejection = dict(kind=kind, **detail)
        now = self.clock()
        if self.rejection_report_at is None or now-self.rejection_report_at >= 60:
            self.rejection_report_at = now
            self.event('航巡输入拒绝观测：'+json.dumps(dict(
                **self.last_input_rejection, counts=self.input_rejections),
                ensure_ascii=False, separators=(',', ':')))

    def _clear_confirmation(self):
        self._key = self._frame = self._last_seen = self._first_seen = None
        self._seen = 0

    def _confirmed(self, key, frame, seconds):
        now = self.clock()
        if key is None:
            self._clear_confirmation()
            return False
        if frame is self._frame:
            return False
        if (key != self._key or self._last_seen is None
                or now - self._last_seen > self.MAX_FRAME_GAP):
            self._key, self._first_seen, self._seen = key, now, 1
        else:
            self._seen += 1
        self._frame, self._last_seen = frame, now
        return self._seen >= self.CONFIRM_FRAMES and now - self._first_seen >= seconds

    def _reset(self):
        self.state, self.since, self.deadline = 'idle', None, None
        self.scene = self.expected = None
        self._clear_confirmation()

    def suspend(self):
        # Pauses cannot turn old observations into consecutive proof or renew
        # an already opened menu's original deadline.
        self._clear_confirmation()
        self.since = None
        if not self.busy:
            self.scene = self.expected = None

    def _fail(self, reason):
        self.failures += 1
        self.retry_at = self.clock() + self.RETRY_SECONDS
        self.hold = '航巡看门狗放弃本轮：' + reason
        self.event(self.hold)
        self._reset()
        return False

    def input_allowed(self, kind):
        phase = {'cruise-open': 'opening', 'cruise-start': 'opening'}.get(kind)
        return (phase is not None and self.state == phase and self.expected is not None
                and self.deadline is not None and self.clock() < self.deadline)

    def step(self, frame):
        now = self.clock()
        # Check expiry before recognition, even on an otherwise valid third
        # menu frame. Slow OCR/input guards re-check the same deadline.
        if self.busy and (self.deadline is None or now >= self.deadline):
            return self._fail('航巡设置菜单未及时确认' if self.state == 'opening'
                              else '点开始后航巡没有变成运行')
        if not self.busy and (now < self.retry_at or self.running(frame) is not False):
            self._reset()
            return False
        view = self.observe(frame)
        if self.busy and self.clock() >= self.deadline:
            return self._fail('识别期间原航巡恢复窗口已到期')
        if view is None:
            self.suspend()
            return self.busy
        if self.busy:
            return self._advance(frame, view)
        if view['menu'] or view['route'] is not False:
            self._reset()
            return False
        key = (view['scene'], view['signature'])
        ready = self._confirmed(key, frame, self.QUIET_SECONDS)
        self.since = self._first_seen
        self.hold = f'航巡已停，{max(0, int(self.QUIET_SECONDS-self.clock()+self.since))} 秒后尝试重新打开'
        if not ready:
            return False
        self.scene, self.expected = view['scene'], view
        self.state, self.deadline = 'opening', self.clock() + self.MENU_WAIT
        self._clear_confirmation()
        if not self.click('cruise-open', *self.OPEN_AT, '打开航巡设置', view):
            return self._fail('新截图或时限守卫拒绝打开航巡设置')
        self.hold = '航巡看门狗：已请求打开航巡设置，等菜单出现'
        self.event(self.hold)
        return True

    def _advance(self, frame, view):
        if view['scene'] != self.scene:
            return self._fail('当前世界、阶段或RANK变化，交还主流程')
        if self.state == 'opening':
            key = ((view['scene'], view['signature'], view['settings'])
                   if view['menu'] and view['settings'] is not None else None)
            if self._confirmed(key, frame, self.CONFIRM_SECONDS):
                self.expected = view
                if not self.input_allowed('cruise-start'):
                    return self._fail('开始航巡前原窗口已到期')
                if not self.click('cruise-start', *self.START_AT, '启动航巡', view):
                    return self._fail('新截图、设置或时限守卫拒绝开始航巡')
                self.state, self.deadline = 'starting', self.clock() + self.START_WAIT
                self._clear_confirmation()
                self.hold = '航巡看门狗：已点开始，等航巡变运行'
                self.event(self.hold)
            return True  # Own the menu on waiting frames too.
        key = ('started', view['scene']) if not view['menu'] and view['route'] is True else None
        if self._confirmed(key, frame, self.CONFIRM_SECONDS):
            self.retry_at = self.clock() + self.RETRY_SECONDS
            self.hold = '航巡看门狗：航巡已恢复运行'
            self.event('航巡看门狗：连续新图跨三秒确认航巡恢复运行，不再重复输入')
            self._reset()
            return False
        return True


