"""Collect the visible daily RAID quests, independently of battle loot."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from drift.core.continuity import FreshConfirmation
from drift.tasks.daily import game_day_window
from drift.tasks.training_sailing import TARGETS


RAID_ENTRY = 'driftinggirl_raid_open'
AUTO = (218, 29)
AUTO_START = (360, 873)
AUTO_CLOSE = (600, 363)


def crop(frame, box):
    x, y, right, bottom = box
    return frame[y:bottom, x:right]


@dataclass(frozen=True)
class RaidQuest:
    tier: int
    close: tuple
    route: tuple | None
    finish: tuple | None


class RaidQuestVisual:
    def __init__(self, root):
        def read(path):
            with Image.open(Path(root) / 'assets' / path) as image:
                return np.array(image.convert('RGB'))

        card = read('task-popup/kraken-arrival.png')
        self.templates = {
            'title': crop(card, (464, 462, 538, 489)),
            'defeat': crop(card, (338, 462, 387, 489)),
            'stage': crop(card, (554, 462, 604, 489)),
            'fish': crop(card, (207, 670, 304, 699)),
            'count': crop(card, (309, 669, 387, 701)),
            'alchemy': crop(card, (235, 722, 309, 749)),
            'potion': crop(card, (167, 806, 195, 837)),
            'close': crop(card, (624, 386, 663, 427)),
            'preamble': crop(card, (271, 523, 302, 597)),
            'paper': crop(card, (106, 808, 159, 835)),
            'route': crop(card, (568, 717, 603, 752)),
            'check': crop(card, (564, 671, 606, 701)),
            'finish': read('raid-reward/finish-green.png'),
            'roster': read('raid-reward/quest-roster.png'),
            'reward-effect': read('raid-reward/potion-effect.png'),
        }
        self.numbers = {3: crop(card, (539, 463, 553, 489))}
        # The same game font is used for the parenthesized progress digits.
        self.kinds = {3: crop(card, (200, 807, 220, 837))}

    def find(self, frame, name, box, score=.94):
        return self.locate(frame, self.templates[name], box, score)

    @staticmethod
    def locate(frame, template, box, score=.94):
        source = crop(frame, box)
        h, w = template.shape[:2]
        if source.shape[0] < h or source.shape[1] < w:
            return None
        _, value, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED))
        found = source[y:y+h, x:x+w]
        if value < score or abs(float(found.mean()) - float(template.mean())) > 22:
            return None
        return (box[0] + x + w//2, box[1] + y + h//2)

    def card(self, frame):
        if frame.shape != (1280, 720, 3):
            return None
        # Complete quests have a second, fixed arrival row. Reject the one-row
        # notices while the required catch count is still incomplete.
        anchors = (
            ('defeat', (334, 458, 392, 493)), ('preamble', (268, 520, 305, 600)),
            ('fish', (203, 665, 339, 703)), ('count', (305, 665, 419, 706)),
            ('alchemy', (231, 719, 313, 752)),
            ('paper', (102, 805, 164, 841)), ('potion', (164, 802, 198, 842)),
        )
        close = self.find(frame, 'close', (620, 382, 668, 432))
        if close is None or not all(self.find(frame, name, area) for name, area in anchors):
            return None
        title = self.find(frame, 'title', (460, 458, 568, 493))
        if title is None:
            return None
        dx = title[0]-501
        if (not self.locate(frame, self.numbers[3], (536+dx, 460, 556+dx, 492))
                or not self.find(frame, 'stage', (551+dx, 458, 607+dx, 493))
                or not self.locate(frame, self.kinds[3], (198, 803, 223, 842))
                or not self.find(frame, 'check', (558, 663, 610, 708), .92)):
            return None
        route = self.find(frame, 'route', (561, 711, 610, 759))
        arrived = self.find(frame, 'check', (558, 713, 610, 759), .92)
        finish = self.find(frame, 'finish', (470, 790, 623, 849)) if arrived else None
        return RaidQuest(3, close, route if not arrived else None, finish)

    def roster(self, frame):
        for top in range(265, 817, 110):
            point = self.find(frame, 'roster', (22, top, 94, top+95), .92)
            if point is None:
                continue
            # The blue bar has to be filled to the top. A matching portrait
            # by itself is shared with unrelated alchemy quests.
            x, y = point
            bar = frame[y-30:y-19, 8:19].astype('int16')
            blue = (bar[:, :, 2] > bar[:, :, 0] + 55) & (bar[:, :, 1] > bar[:, :, 0] + 40)
            if float(blue.mean()) >= .88:
                return x, y
        return None


class RaidReward:
    """Run only from a verified quest card or a completed RAID's quest entry."""
    LIMIT = 420
    PROBE_LIMIT = 12
    ALLOWED = {'idle', 'read', 'close', 'stop', 'reopen', 'route', 'travel',
               'claim', 'verify', 'restore', 'menu', 'restored'}

    def __init__(self, root, scene, moving, action, event, enabled, serial, *, clock=time.time):
        self.path = Path(root) / 'raid_reward_state.json'
        self.visual = RaidQuestVisual(root)
        self.scene, self.moving = scene, moving
        self.action, self.event, self.enabled, self.clock = action, event, enabled, clock
        self.serial = serial
        self.proof = FreshConfirmation(seconds=1.2)
        self.arm_until = 0.
        self.next_at = 0.
        self.journal = dict(version=1, serial=serial, pending=False, phase='idle', attempts={}, rewards={})
        self.invalid = False
        try:
            if self.path.is_file():
                saved = json.loads(self.path.read_text(encoding='utf-8-sig'))
                if (not isinstance(saved, dict) or saved.get('version') != 1 or saved.get('serial') != serial
                        or saved.get('phase') not in self.ALLOWED or not isinstance(saved.get('rewards'), dict)
                        or not isinstance(saved.get('attempts'), dict)):
                    raise ValueError('invalid record')
                if (saved['phase'] != 'idle' and (not isinstance(saved.get('reward_key'), str)
                        or not isinstance(saved.get('screen'), str)
                        or not all(isinstance(saved.get(k),(int,float)) for k in ('deadline','step_deadline')))):
                    raise ValueError('invalid pending record')
                self.journal = saved
        except (OSError, TypeError, ValueError):
            self.invalid = True
            self.event('RAID任务：保存记录无法读取，保留记录并跳过领取')

    @property
    def phase(self):
        return self.journal['phase']

    @property
    def active(self):
        return self.phase != 'idle'

    def status(self):
        return dict(phase=self.phase, pending=self.active, deadline=self.journal.get('deadline'),
                    attempts=dict(self.journal['attempts']), rewards=dict(self.journal['rewards']),
                    result=self.journal.get('result'))

    def arm(self):
        self.arm_until = self.clock() + 70

    def offered(self, card, frame):
        return bool(self.enabled() and not self.invalid and card and (card.finish or card.route)
                    and self._screen(frame) is not None
                    and (self.active or self._period()+':raid-3' not in self.journal['rewards']))

    def _period(self):
        return game_day_window(datetime.fromtimestamp(self.clock(), timezone.utc))[0]

    def _save(self):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.journal, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        tmp.replace(self.path)

    def _set(self, phase, seconds):
        self.journal.update(phase=phase, step_deadline=min(self.journal['deadline'], self.clock()+seconds))
        self.proof.reset()
        self._save()

    def _finish(self, reason):
        # Keep all input reservations from the completed attempt in the record.
        self.journal.update(pending=False, phase='idle', result=reason, finished_at=self.clock())
        self.arm_until, self.next_at = 0., self.clock()+60
        self.proof.reset()
        self._save()
        self.event('RAID任务：'+reason)

    def _entry(self, frame):
        return self.visual.roster(frame) if self.scene.sailing.running(frame) is not None else None

    @staticmethod
    def _screen(frame):
        # Fixed normalized RANK strokes and the cyan world/phase badge survive
        # quest-panel dimming. Do not use the moving scene or timer as identity.
        if frame.shape != (1280,720,3):
            return None
        rank = frame[9:58, 70:136].min(axis=2)
        peak = int(rank.max())
        stroke = rank >= .90 * peak
        if peak < 100 or not 160 < stroke.sum() < 1300:
            return None
        badge = frame[45:70, 14:69].astype('int16')
        blue = (badge[:,:,1]-badge[:,:,0] > .12*peak) & (badge[:,:,2]-badge[:,:,0] > .12*peak)
        return ':'.join(np.packbits(mask).tobytes().hex() for mask in (stroke, blue))

    def _same_screen(self, frame):
        current = self._screen(frame)
        saved = self.journal.get('screen')
        if current is None or saved is None:
            return False
        if saved == current:
            return True
        try:
            return all((int(a,16)^int(b,16)).bit_count() <= limit
                       for a,b,limit in zip(current.split(':'), saved.split(':'), (24, 15), strict=True))
        except ValueError:
            return False

    def _options(self, frame):
        auto = self.scene.sailing
        if not self.scene._off_icon(frame) or not auto.menu(frame):
            return None
        settings = self.scene.settings_key(frame)
        extras = auto.sale_selected(frame), auto.upgrade_selected(frame)
        if settings is None or None in extras:
            return None
        return tuple(settings[0]), settings[1], extras

    def _target(self, frame, card=None):
        if not self._same_screen(frame):
            return None
        phase, attempts = self.phase, self.journal['attempts']
        auto = self.scene.sailing
        card = card if card is not None else self.visual.card(frame)
        if card is not None:
            if self.scene._battle(frame):
                return None
            if phase in ('read', 'route', 'travel', 'claim') and card.finish and not attempts.get('finish'):
                return 'finish', card.finish
            if phase == 'close' or (phase == 'travel' and attempts.get('route')):
                return ('close-entry' if phase == 'close' else 'close-travel'), card.close
            if (phase in ('route', 'read') and card.route and self.scene._off_icon(frame)
                    and not self.moving(frame) and not attempts.get('route')):
                return 'route', card.route
            return None
        options = self._options(frame) if phase in ('menu','restored') else None
        if options:
            if not attempts.get('start'):
                return 'start', AUTO_START
            return ('dismiss-auto', AUTO_CLOSE) if self.clock() - attempts['start'] >= 5 else None
        if not self.scene.fishing._main_visible(frame) or self.scene._battle(frame):
            return None
        state = auto.running(frame)
        if phase == 'stop' and state is True:
            return 'stop', AUTO
        travelled = self.journal.get('travel_seen') or (self.journal.get('departure') is not None
                      and self.place(frame) is not None and self.place(frame) != self.journal['departure'])
        if ((phase == 'reopen' or phase == 'travel' and travelled)
                and state is False and not self.moving(frame)):
            point = self._entry(frame)
            if point:
                return ('reopen' if phase == 'reopen' else 'arrive'), point
        if phase == 'read' and not attempts.get('inspect'):
            point = self._entry(frame)
            if point:
                return 'inspect', point
        if phase == 'restore' and state is False:
            return 'menu', AUTO
        return None

    def in_window(self, kind):
        return (self.active and self.clock() < self.journal['step_deadline']
                and (self.enabled() or kind in ('menu','start','dismiss-auto')))

    def allowed(self, kind):
        return (self.in_window(kind)
                and not self.journal['attempts'].get(kind)
                and (kind != 'finish' or self.journal['reward_key'] not in self.journal['rewards']))

    def validate(self, kind, point, frame, options):
        return (self.allowed(kind) and (kind, point) == self._target(frame)
                and (kind != 'start' or self._options(frame) == options))

    @staticmethod
    def place(frame):
        # This band contains only the fixed name, above the moving countdown.
        ink = frame[12:35, 287:437].max(axis=2) < 60
        return np.packbits(ink).tobytes().hex() if ink.sum() >= 28 else None

    def reserve(self, kind, frame=None):
        if not self.allowed(kind):
            return False
        self.journal['attempts'][kind] = self.clock()
        if kind == 'route' and frame is not None:
            self.journal['departure'] = self.place(frame)
        if kind == 'finish':
            self.journal['rewards'][self.journal['reward_key']] = 'issued_unverified'
        self._save()
        return True

    def _begin(self, frame, card):
        screen = self._screen(frame)
        if screen is None:
            return False
        period = self._period()
        key = period + ':raid-3'
        if key in self.journal['rewards']:
            return False
        now = self.clock()
        self.journal = dict(version=1,serial=self.serial,pending=True,phase='read',
                            attempts={}, rewards=self.journal['rewards'], reward_key=key,
                            screen=screen, deadline=now+self.LIMIT, started_at=now)
        self._set('read', 20 if card else self.PROBE_LIMIT)
        return True

    def step(self, frame, frame_id):
        now = self.clock()
        if self.invalid or frame.shape != (1280,720,3):
            return False
        card = self.visual.card(frame)
        if not self.active:
            if not self.enabled():
                return False
            offer = self.offered(card, frame)
            if not offer and (now < self.next_at or now >= self.arm_until or not self._entry(frame)):
                return False
            if not self._begin(frame, card):
                return False
        j = self.journal
        restoring = self.phase in ('restore','menu','restored')
        expired = now >= j['step_deadline'] or not restoring and self._period() != j['reward_key'].split(':', 1)[0]
        if expired or not restoring and not self.enabled():
            if j['attempts'].get('stop') and self.phase not in ('restore','menu','restored') and now+5 < j['deadline']:
                self.event('RAID任务：本次未确认完整领取，接续恢复原航巡')
                self._set('restore', 40)
            else:
                self._finish('本次等待结束，保留已有操作和领奖记录')
                return False
        if not self._same_screen(frame):
            self.proof.reset()
            return True
        phase = self.phase
        auto = self.scene.sailing
        main = self.scene.fishing._main_visible(frame) and not self.scene._battle(frame)
        if phase == 'travel' and main and self.moving(frame) and not j.get('travel_seen'):
            j['travel_seen'] = True
            self._save()
        if phase == 'read' and card and card.route and not self.scene._off_icon(frame):
            self._set('close', 18)
        elif phase == 'stop' and main and auto.running(frame) is False:
            if self.proof.observe('off', frame_id, now):
                self._set('reopen', 65)
            return True
        elif phase == 'verify':
            # Never replay a submitted claim after an interruption. Evidence
            # records show the post-input frame even when no reward is visible.
            if self.proof.observe('returned' if card is None and main else None, frame_id, now):
                if j['rewards'].get(j['reward_key']) != 'confirmed':
                    self.event('RAID任务：提交后任务卡关闭；药水到账仍待核对')
                if j['attempts'].get('stop'):
                    self._set('restore', 40)
                else:
                    self._finish('提交输入完成，已回主画面')
            return self.active
        elif phase in ('restore','restored') and main and auto.running(frame) is True:
            if self.proof.observe('on', frame_id, now):
                self._finish('已连续确认恢复原航巡')
            return self.active
        target = self._target(frame, card)
        if target is None or not self.allowed(target[0]):
            self.proof.reset()
            return True
        kind, point = target
        options = self._options(frame) if kind == 'start' else None
        if not self.proof.observe((target, options), frame_id, now):
            return True
        if not self.action(kind, point, options):
            return True
        transitions = {'inspect': ('read', 12), 'close-entry': ('stop', 40),
                       'stop': ('stop', 40), 'reopen': ('route', 30),
                       'route': ('travel', 235), 'close-travel': ('travel', 235),
                       'arrive': ('claim', 35), 'finish': ('verify', 18),
                       'menu': ('menu', 20), 'start': ('restored', 20),
                       'dismiss-auto': ('restored', 12)}
        self.proof.reset()
        next_phase, seconds = transitions[kind]
        if next_phase != self.phase:
            self._set(next_phase, seconds)
        return True

    def observe_claim(self, frame, before):
        """Only record payment from a fresh post-submit reward animation."""
        key = self.journal.get('reward_key')
        if (key is None or self.journal['rewards'].get(key) != 'issued_unverified'
                or not self._same_screen(frame) or self.visual.card(frame) is not None):
            return False
        area = (240,450,620,855)
        if (self.visual.find(frame,'reward-effect',area,.96) is None
                or self.visual.find(before,'reward-effect',area,.96) is not None):
            return False
        self.journal['rewards'][key] = 'confirmed'
        self._save()
        self.event('RAID任务：提交后确认获得药水')
        return True
