"""One bounded, opt-in voyage from the ordinary world to its existing portal.

Only read-only probes are borrowed from the farming controllers. Their rank
loop, gate-discard operation, fishing and native cruise never run here.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from drift.core.exceptions import RetryableStage
from drift.tasks.backpack_sale import BackpackSale
from drift.tasks.parallel_screen import ParallelScreen
from drift.tasks.parallel_world import ParallelWorld


class ReturnParallel(ParallelScreen):
    PHASE_TIMEOUTS = dict(idle=120, map_open=120, map_find=120, map_close=30,
                          travel=600, npc=60, manager=60, search=60, new_gate=60,
                          panel_close=30, enter=45, enter_confirm=45, arrive=90)
    INPUT_LIMITS = {'map-open': 3, 'map-pan-left': 6, 'map-pan-right': 6,
                    'map-select': 3, 'map-close': 3, 'teleport-cancel': 3,
                    'npc': 3, 'activate': 1, 'search-confirm': 1,
                    'panel-close': 2, 'enter': 2, 'enter-confirm': 2}
    LIMIT = 1500

    # Reuse the exact map/world/location and manager/portal probes, without
    # constructing StoneFarm or touching its independent RANK40 journal.
    _portal_words = ParallelWorld._portal_words
    _gate_manager = ParallelWorld._gate_manager
    _gate_active = staticmethod(ParallelWorld._gate_active)
    _find = BackpackSale._find
    _bright = staticmethod(BackpackSale._bright)
    tab = BackpackSale.tab
    main = BackpackSale.main
    moving = BackpackSale.moving

    def __init__(self, root, action, event, serial='emulator-5554', *,
                 state_path=None, clock=time.time):
        super().__init__(root, None)
        self.root, self.action, self.event = Path(root), action, event
        self.clock, self.serial = clock, serial
        self.path = Path(state_path) if state_path else self.root / 'return_parallel_state.json'
        self.enabled = lambda: True  # Runtime supplies the explicit opt-in switch.
        self.main_guard = lambda frame: False
        sources = {'moving': 'backpack', 'tab-first': 'backpack',
                   'entry': 'equipment', 'guard-star': 'equipment',
                   'guard-diamond': 'equipment', 'main-auto': 'equipment',
                   'auto-label': 'training-sailing', 'inactive-face': 'training-sailing',
                   'active-arrow': 'training-sailing', 'timer': 'boss-defense',
                   'swords': 'boss-defense', 'teleport-currency': 'guild-tasks',
                   'teleport-continue': 'guild-tasks', 'teleport-cancel': 'guild-tasks'}
        for name, folder in sources.items():
            self.assets[name] = self._load(f'assets/{folder}/{name}.png')
        self.portal_frames = {name: self._load(f'assets/recovery/{name}.png') for name in
                              ('parallel-manager', 'parallel-manager-inactive',
                               'observatory-npc', 'observatory-npc-day')}
        now = self.clock()
        self.journal = dict(version=1, serial=serial, pending=False, done=None,
                            phase='idle', deadline=now + self.LIMIT,
                            phase_deadline=now + self.PHASE_TIMEOUTS['idle'],
                            attempts={}, target=None, hold=None, failure=None, next_at=0)
        if self.path.is_file():
            try:
                saved = json.loads(self.path.read_text(encoding='utf-8-sig'))
                if not self._valid_record(saved):
                    raise ValueError('invalid record')
                self.journal = saved
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise RetryableStage('返回平行世界记录无效，保留原记录', task='return_parallel') from exc
        self.phase = self.journal['phase']
        self.seen, self.last_frame = 0, None
        self.note_at = {}

    def _load(self, relative):
        raw = cv2.imread(str(self.root / relative))
        if raw is None:
            raise RetryableStage('返回平行世界缺少模板：' + relative, task='return_parallel')
        return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    def _valid_record(self, saved):
        return (isinstance(saved, dict) and saved.get('version') == 1
                and saved.get('serial') == self.serial
                and type(saved.get('pending')) is bool
                and (saved.get('done') is None or type(saved['done']) is bool)
                and saved.get('phase') in {*self.PHASE_TIMEOUTS, 'done', 'failed'}
                and all(type(saved.get(k)) in (int, float) and math.isfinite(saved[k])
                        for k in ('deadline', 'phase_deadline', 'next_at'))
                and isinstance(saved.get('attempts'), dict)
                and all(k in self.INPUT_LIMITS and type(v) is int
                        and 0 <= v <= self.INPUT_LIMITS[k] for k, v in saved['attempts'].items())
                and (saved['done'] is None and saved['phase'] in self.PHASE_TIMEOUTS
                     and saved['pending'] == (saved['phase'] != 'idle')
                     or saved['done'] is not None and not saved['pending']
                     and saved['phase'] == ('done' if saved['done'] else 'failed')))

    @property
    def active(self):
        return self.journal['pending'] and self.done is None

    @property
    def done(self):
        return self.journal['done']

    def status(self):
        return dict(phase=self.phase, active=self.active, done=self.done,
                    hold=self.journal['hold'], target=self.journal['target'],
                    deadline=self.journal['deadline'], phase_deadline=self.journal['phase_deadline'],
                    attempts=dict(self.journal['attempts']), failure=self.journal['failure'],
                    failed_phase=self.journal.get('failed_phase'), confirmation_frames=self.seen)

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            from drift.core.persistence import atomic_write_text
            atomic_write_text(self.path, json.dumps(self.journal, ensure_ascii=False, indent=2))
        except OSError as exc:
            raise RetryableStage('返回平行世界记录无法保存，停止本任务输入', task='return_parallel') from exc

    def _set(self, phase):
        if phase == self.phase:
            return
        self.phase = phase
        self.journal.update(phase=phase, pending=True, hold=None,
                            phase_deadline=min(self.journal['deadline'],
                                               self.clock() + self.PHASE_TIMEOUTS[phase]))
        self.seen, self.last_frame = 0, None
        self._save()

    def _finish(self, success, failure=None):
        if self.done is not None:
            return
        self.journal['failed_phase'] = None if success else self.phase
        self.phase = 'done' if success else 'failed'
        self.journal.update(pending=False, done=success, phase=self.phase,
                            failure=failure, hold=failure, finished_at=self.clock())
        self._save()
        self.event('返回平行世界已完成' if success else '返回平行世界中止：' + failure)

    def _note(self, key, text):
        self.journal['hold'] = text
        if self.clock() >= self.note_at.get(key, float('-inf')):
            self.note_at[key] = self.clock() + 30
            self.event('返回平行世界：' + text)

    def _off_icon(self, frame):
        return bool(self._fixed(frame, 'auto-label', (186,34,253,65), .90)
                    and self._fixed(frame, 'inactive-face', (194,4,246,42), .96)
                    and not self._fixed(frame, 'active-arrow', (194,4,246,42), .96))

    def _battle(self, frame):
        # Same timer/swords guards as StoneFarm._battle, with read-only assets.
        return bool(self.match(frame, 'timer', (188,125,610,165), .94)
                    or self.match(frame, 'swords', (307,962,415,1057), .96))

    def normal_idle(self, frame):
        return (not self._parallel(frame) and not self._map(frame) and self.main(frame)
                and self._off_icon(frame) and not self.moving(frame) and not self._battle(frame))

    def _npc_ready(self, frame):
        return (self.normal_idle(frame) and self._observatory(frame)
                and any(self._portal_words(frame, name, ((460,515,507,624),))
                        for name in ('observatory-npc', 'observatory-npc-day')))

    def _map_target(self, frame):
        return self.match(frame, 'map-observatory', (0,175,720,1120), .94)

    def _enter_question(self, frame):
        return bool(self._fixed(frame, 'enter-question', (220,556,502,620), .95)
                    and self._fixed(frame, 'continue', (63,741,356,834), .96))

    def _search_question(self, frame):
        return bool(self._observatory(frame)
                    and self._fixed(frame, 'search-question', (194,555,525,620), .95)
                    and self._fixed(frame, 'continue', (63,741,356,834), .96))

    def _paid_prompt(self, frame):
        # Currency + both opposite buttons, independent of the quoted amount.
        # Search the whole dialog for the currency: map teleport and guild
        # teleport do not necessarily place the balance on the same text row.
        for name, box in (('teleport-currency', (100,400,620,760)),
                          ('teleport-continue', (70,730,340,900)),
                          ('teleport-cancel', (390,730,650,900))):
            patch = self.assets[name]
            found = any(self._find(frame, cv2.resize(patch, None, fx=scale, fy=scale,
                                                    interpolation=cv2.INTER_CUBIC), box, .94)
                        for scale in (.95, 1, 1.05))
            if not found:
                return False
        return True

    def validate(self, action, frame):
        """Recheck a fresh frame immediately before the runtime issues input."""
        if (not self.enabled() or self.done is not None or frame.shape != (1280,720,3)
                or action.get('phase') != self.phase
                or self.clock() >= min(self.journal['deadline'], self.journal['phase_deadline'])):
            return False
        kind, point = action.get('kind'), (action.get('x'), action.get('y'))
        paid = self._paid_prompt(frame)
        if paid:
            return kind == 'teleport-cancel' and point == (540,833) and not action.get('swipe')
        if self._parallel(frame) or kind == 'teleport-cancel':
            return False
        if kind == 'map-open':
            return self.phase == 'map_open' and point == (350,30) and self.normal_idle(frame)
        if kind in ('map-pan-left', 'map-pan-right', 'map-select', 'map-close'):
            if not self._map(frame):
                return False
            if kind == 'map-close':
                return self.phase in ('map_close', 'travel') and point == (655,1180)
            if self.phase != 'map_find':
                return False
            target = self._map_target(frame)
            if kind == 'map-select':
                return target is not None and point == tuple(target[:2]) and not action.get('swipe')
            start, end = ((300,640), (690,640)) if kind == 'map-pan-left' else ((690,640), (300,640))
            return target is None and point == start and action.get('swipe') == (*end, 300)
        if kind == 'npc':
            return self.phase == 'npc' and point == (482,548) and self._npc_ready(frame)
        if kind == 'activate':
            return self.phase == 'manager' and point == (355,803) and self._gate_manager(frame) is False
        if kind == 'search-confirm':
            return self.phase == 'search' and point == (208,786) and self._search_question(frame)
        if kind == 'panel-close':
            return self.phase == 'panel_close' and point == (590,395) and self._gate_manager(frame) is True
        if kind == 'enter':
            return (self.phase in ('enter', 'enter_confirm') and point == (354,482)
                    and self.normal_idle(frame) and self._observatory(frame) and self._gate_active(frame))
        if kind == 'enter-confirm':
            return (self.phase in ('enter_confirm', 'arrive') and point == (208,787)
                    and self._observatory(frame) and self._enter_question(frame))
        return False

    def _input(self, kind, point, *, swipe=None):
        attempts = self.journal['attempts']
        if attempts.get(kind, 0) >= self.INPUT_LIMITS[kind]:
            self._finish(False, f'{self.phase} 输入 {kind} 达到次数上限')
            return False
        request = dict(kind=kind, x=int(point[0]), y=int(point[1]), phase=self.phase)
        if swipe is not None:
            request['swipe'] = swipe
        # Reserve durably before the callback, including fresh-guard refusals.
        # Restart or an uncertain receipt can never refund an input budget.
        attempts[kind] = attempts.get(kind, 0) + 1
        self.journal.update(next_at=self.clock() + 1.5,
                            last_input=dict(action=deepcopy(request), at=self.clock(), state='reserved'))
        self._save()
        try:
            accepted = self.action(request) is not False
        except InterruptedError:
            return False
        except (RuntimeError, OSError) as exc:
            self._finish(False, f'{self.phase} 输入未确认：{exc}')
            return False
        self.journal['last_input']['state'] = 'issued_only' if accepted else 'rejected'
        self._save()
        return accepted

    def step(self, frame):
        if not self.enabled() or self.done is not None:
            return False
        now = self.clock()
        if now >= min(self.journal['phase_deadline'], self.journal['deadline']):
            self._finish(False, f'{self.phase} 阶段超时')
            return True
        if frame.shape != (1280,720,3):
            self._note('unknown', '等待720×1280游戏画面')
            return self.active
        if self._paid_prompt(frame):
            if now < self.journal['next_at']:
                return True
            if self._input('teleport-cancel', (540,833)):
                self.event('返回平行世界：放弃了代价传送，已点击取消')
                if self.phase in ('map_close', 'travel'):
                    self.journal['target'] = None
                    self._set('map_find')
            return True
        if self._parallel(frame):
            self._finish(True)
            return True
        if now < self.journal['next_at']:
            return self.active

        phase = self.phase
        if phase == 'idle':
            if self.normal_idle(frame):
                self._set('map_open')
                return True
            self._note('idle', '等待普通世界已停船主界面（非战斗、航巡停止）')
            return False
        if phase == 'map_open':
            if self._map(frame):
                self._set('map_find')
            elif self.normal_idle(frame):
                self._input('map-open', (350,30))
        elif phase == 'map_find':
            if not self._map(frame):
                self._set('map_open')
                return True
            target = self._map_target(frame)
            if target is not None:
                if self._input('map-select', target[:2]):
                    self.journal['target'] = dict(name='传送观测所', point=list(target[:2]))
                    self._set('map_close')
            elif self.journal['attempts'].get('map-pan-left', 0) < 6:
                self._input('map-pan-left', (300,640), swipe=(690,640,300))
            elif self.journal['attempts'].get('map-pan-right', 0) < 6:
                self._input('map-pan-right', (690,640), swipe=(300,640,300))
            else:
                self._finish(False, '地图上找不到传送观测所')
        elif phase == 'map_close':
            if not self._map(frame) or self._input('map-close', (655,1180)):
                self._set('travel')
        elif phase == 'travel':
            if self._map(frame):
                self._input('map-close', (655,1180))
                return True
            here = self._observatory(frame) and self.normal_idle(frame)
            if not here:
                self.seen = 0
            elif frame is not self.last_frame:
                self.seen += 1
            self.last_frame = frame
            if self.seen >= 3:
                self._set('npc')
            else:
                self._note('travel', '等待到达传送观测所 主界面=%s 移动中=%s 地名=%s 确认帧数=%s/3' %
                           (bool(self.main(frame)), self.moving(frame),
                            '传送观测所' if self._observatory(frame) else '未匹配', self.seen))
        elif phase == 'npc':
            if self._gate_manager(frame) is not None:
                self._set('manager')
            elif self.normal_idle(frame) and self._observatory(frame):
                if self._gate_active(frame):
                    self._set('enter')
                elif self._npc_ready(frame):
                    self._input('npc', (482,548))
        elif phase == 'manager':
            active = self._gate_manager(frame)
            if active is True:
                # Ordinary-world Rank>=40 is not proof of completed parallel
                # RANK40. Reenter the existing gate, never discard its progress.
                self._set('panel_close')
            elif active is False and self._input('activate', (355,803)):
                self._set('search')
        elif phase == 'search':
            if self._gate_manager(frame) is True:
                self._set('new_gate')
            elif self._search_question(frame) and self._input('search-confirm', (208,786)):
                self._set('new_gate')
        elif phase == 'new_gate':
            if self._gate_manager(frame) is True:
                self._set('panel_close')
        elif phase == 'panel_close':
            if self._gate_manager(frame) is True:
                if self._input('panel-close', (590,395)):
                    self._set('enter')
            elif self.normal_idle(frame) and self._observatory(frame):
                self._set('enter')
        elif phase == 'enter':
            if self._enter_question(frame):
                self._set('enter_confirm')
            elif self.normal_idle(frame) and self._observatory(frame) and self._gate_active(frame):
                if self._input('enter', (354,482)):
                    self._set('enter_confirm')
        elif phase in ('enter_confirm', 'arrive'):
            if self._observatory(frame) and self._enter_question(frame):
                if self._input('enter-confirm', (208,787)) and phase != 'arrive':
                    self._set('arrive')
            elif phase == 'enter_confirm' and self.normal_idle(frame) and self._observatory(frame):
                if self._gate_active(frame):
                    self._input('enter', (354,482))
        return True

# Keep legacy task logic; delegate every record read/write to shared storage.
from drift.core.persistence import ReliablePath as Path

# Shared reliability layer; retain all other deployed methods.
def _save(self):
    try:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        from drift.core.persistence import atomic_write_text
        atomic_write_text(self.path, json.dumps(self.journal, ensure_ascii=False, indent=2))
    except OSError as exc:
        raise RetryableStage('返回平行世界记录无法保存，停止本任务输入', task='return_parallel') from exc

ReturnParallel._save = _save



# Module-local compatibility for every authoritative record read/write.
from drift.core.persistence import ReliablePath as Path
