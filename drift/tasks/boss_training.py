"""Persisted fishing and equipment recovery between parallel-world boss attempts."""
from collections import deque
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np


# Keep old gain/time keys readable when updating an existing runtime host configuration.
# They no longer end a farming session before the requested boss-level target.
DEFAULT_TRAINING = {'enabled': True, 'failure_limit': 2, 'level_gain': 20,
                    'minutes': 10, 'retry_level_gap': 20,
                    'final_boss_retry_level_gap': 0, 'auto_sailing': True,
                    'meter_auto_adjust': True, 'boss_interrupt': True, 'battle_tap': False,
                    'parallel_stage_limit': 0, 'boss_auto_sailing_until_rank': 0}


def training_options(value=None):
    if value is None:
        return DEFAULT_TRAINING.copy()
    if not isinstance(value, dict) or not set(value) <= set(DEFAULT_TRAINING):
        raise ValueError('练级设置无效')
    result = {**DEFAULT_TRAINING, **value}
    if any(not isinstance(result[key], bool) for key in
           ('enabled','auto_sailing','meter_auto_adjust','boss_interrupt','battle_tap')):
        raise ValueError('练级开关必须为布尔值')
    for key, low, high in (('failure_limit', 1, 10), ('level_gain', 1, 200),
                           ('minutes', 1, 60), ('retry_level_gap', 0, 200),
                           ('final_boss_retry_level_gap', 0, 200), ('parallel_stage_limit', 0, 100),
                           ('boss_auto_sailing_until_rank', 0, 39)):
        item = result[key]
        if isinstance(item, bool) or not isinstance(item, int) or not low <= item <= high:
            raise ValueError(f'练级参数 {key} 必须是 {low} 到 {high} 的整数')
    return result


class BossTraining:
    TRIAL_SECONDS = 90
    FINAL_BOSS_RANK = 39

    def __init__(self, root, reader, event, options=None, state_path=None):
        root = Path(root)
        self.options = training_options(options)
        self.reader, self.event = reader, event
        self.path = Path(state_path) if state_path is not None else root/'training_state.json'
        self.assets = {}
        for key in ('map-pin', 'map-player', 'failure-cross', 'success-check'):
            self.assets[key] = self._load(root/'assets/training'/f'{key}.png')
        pin = cv2.cvtColor(self.assets['map-pin'], cv2.COLOR_RGB2HSV)
        # Labels and terrain change between chapters; retain only the red pin's
        # shading for the fallback, then check its full shape and location below.
        self.pin_mask = (((pin[:, :, 0] <= 15) | (pin[:, :, 0] >= 170))
                         & (pin[:, :, 1] >= 80) & (pin[:, :, 2] >= 100)).astype(np.uint8)
        for key in ('entry', 'guard-star', 'guard-diamond', 'main-auto'):
            self.assets[key] = self._load(root/'assets/equipment'/f'{key}.png')
        self.assets['boss-mode'] = self._load(root/'assets/navigation/parallel-boss-battle.png')[38:75, 254:289]
        # The large result X also appears over a changing holder/progress bar.
        # Keep its colored strokes and narrow border. A wider dilation includes
        # the moving holder bar, which hid real RANK 35 failures below .93.
        cross = cv2.cvtColor(self.assets['failure-cross'], cv2.COLOR_RGB2HSV)
        strokes = ((cross[:, :, 0] <= 25) & (cross[:, :, 1] >= 80)
                   & (cross[:, :, 2] >= 100)).astype(np.uint8)
        self.failure_mask = cv2.dilate(strokes, np.ones((3, 3), np.uint8))
        self.state = {'mode': 'idle'}
        if self.path.is_file():
            try:
                saved = json.loads(self.path.read_text(encoding='utf-8-sig'))
                if saved.get('version') == 1 and saved.get('mode') == 'idle':
                    # Early boss navigation owns AUTO outside a farming session.
                    # Preserve ownership across restarts, including an issued but
                    # not yet verified start, so it must stop before any map input.
                    if saved.get('sailing_enabled'):
                        if (saved.get('sailing_profile') != 'boss'
                                or type(saved.get('sailing_started_rank')) is not int
                                or not 0 <= saved['sailing_started_rank'] < 39
                                or any(type(saved.get(key)) is not bool
                                       for key in ('sailing_enabled','sailing_verified'))):
                            raise RuntimeError('BOSS 自动航行保存记录无效，已停止以避免遗漏退出')
                        self.state.update({key:saved[key] for key in
                            ('sailing_enabled','sailing_verified','sailing_profile','sailing_started_rank')})
                    if type(saved.get('boss_sailing_sleep_rank')) is int and 0 <= saved['boss_sailing_sleep_rank'] < 39:
                        self.state['boss_sailing_sleep_rank'] = saved['boss_sailing_sleep_rank']
                    if 'boss_sailing_handoff_rank' in saved:
                        handoff = saved['boss_sailing_handoff_rank']
                        if type(handoff) is not int or not 0 <= handoff < 40:
                            raise RuntimeError('BOSS 航行交接记录无效')
                        self.state['boss_sailing_handoff_rank'] = handoff
                if (saved.get('version') == 1 and saved.get('mode') in ('farming', 'trial')
                        and isinstance(saved.get('rank'), int) and 0 <= saved['rank'] < 40
                        and isinstance(saved.get('start_level'), int) and 1 <= saved['start_level'] <= 10000
                        and isinstance(saved.get('boss_level'), int) and 0 <= saved['boss_level'] <= 10000
                        and isinstance(saved.get('started_at'), (int, float))
                        and np.isfinite(saved['started_at'])
                        and 0 <= time.time()-saved['started_at']):
                    self.state = saved
                    if any(key in saved and not isinstance(saved[key],bool)
                           for key in ('sailing_enabled','sailing_verified')):
                        raise RuntimeError('自动航行保存记录无效，已停止以避免遗漏退出')
                cache = saved.get('quest_boss_level')
                if (saved.get('version') == 1 and saved.get('mode') in ('idle','farming','trial')
                        and self._valid_boss_cache(cache)):
                    # Idle encounters need their verified target after a restart,
                    # too. Never import a target from another farming quest.
                    if self.state.get('rank',cache['rank']) == cache['rank']:
                        self.state['quest_boss_level'] = cache
            except (OSError, ValueError, TypeError):
                self.event('练级记录无法读取，按当前画面重新判断')
        self.player_level = None
        cache = self.state.get('quest_boss_level')
        self.boss_level = self.state.get('boss_level') or (cache['level'] if cache else None)
        self.boss_rank = self.state.get('rank') if self.state.get('boss_level') else (cache['rank'] if cache else None)
        self.boss_header_readings = deque(maxlen=2)
        self.last_missing_boss_at = -60.
        self.readings = deque(maxlen=2)
        self.failures = 0
        self.failure_latched = False
        self.outcome_rank = None

    @staticmethod
    def _load(path):
        data = cv2.imread(str(path))
        if data is None:
            raise RuntimeError(f'练级识别模板缺失：{path.name}')
        return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    def _match(self, frame, name, box, threshold=.9):
        x1, y1, x2, y2 = box
        template = self.assets[name]
        crop = frame[y1:y2, x1:x2]
        if min(crop.shape[:2]) < 1 or crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
            return None
        if name == 'failure-cross':
            scores = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED, mask=self.failure_mask)
            scores = np.nan_to_num(scores, nan=-1, posinf=-1, neginf=-1)
        else:
            scores = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, position = cv2.minMaxLoc(scores)
        if not np.isfinite(score) or score < threshold:
            return None
        return x1+position[0]+template.shape[1]//2, y1+position[1]+template.shape[0]//2

    def main(self, frame):
        boxes = {'entry': (327, 1140, 378, 1192), 'guard-star': (208, 1137, 269, 1192),
                 'guard-diamond': (539, 1137, 600, 1186), 'main-auto': (276, 881, 310, 913)}
        return (all(self._match(frame, name, box, .8 if name == 'entry' else .91) for name, box in boxes.items())
                and float((np.max(frame[1147:1168, 558:585], axis=2) >= 210).mean()) >= .30)

    def boss_mode(self, frame):
        return self._match(frame, 'boss-mode', (248, 32, 295, 81), .92) is not None

    def observe_outcome(self, frame, rank):
        if self.outcome_rank != rank:
            self.failures = 0
            self.failure_latched = False
            self.outcome_rank = rank
            if self.boss_rank != rank:
                self.boss_level = None
                self.boss_rank = None
                self.boss_header_readings.clear()
        if not self.main(frame):
            return
        failed = self._match(frame, 'failure-cross', (290, 955, 432, 1060), .93) is not None
        succeeded = self._match(frame, 'success-check', (290, 955, 432, 1060), .92) is not None
        # Menus and long enhancement holds can hide results for minutes. Keep
        # confirmed failures in this quest until an actual success/new rank;
        # elapsed time alone says nothing about whether the next fish succeeds.
        if succeeded:
            self.failures = 0
        if failed and not succeeded and not self.failure_latched:
            # After a quest succeeds, ordinary fish at its old location must
            # not prevent navigation to the next marked boss.
            if self.boss_mode(frame) and self.state['mode'] != 'farming':
                self.failures += 1
                self.event(f'确认任务点钓鱼失败：{self.failures}/{self.options["failure_limit"]}')
        self.failure_latched = failed

    def _number(self, frame, boxes):
        votes = {}
        for index, (x1, y1, x2, y2) in enumerate(boxes):
            crop = frame[y1:y2, x1:x2]
            ink = np.all(crop >= 200, axis=2)
            binary = np.repeat(np.where(ink, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
            for view in (crop, binary):
                reading = self.reader.read(view)
                if reading.value is not None and 1 <= reading.value <= 10000 and reading.confidence >= .93:
                    votes.setdefault(reading.value, set()).add(index)
        # Even a strong conflicting view invalidates this frame.
        if len(votes) == 1:
            value, regions = next(iter(votes.items()))
            if len(regions) == 2:
                return value
        return None

    def levels(self, frame, *, main_known=False):
        if not main_known and not self.main(frame):
            self.readings.clear()
            return None
        # Exclude the vertical XP-bar border, which OCR can append as a 7.
        player = self._number(frame, ((76, 1248, 137, 1278), (79, 1249, 136, 1277)))
        boss = (self._cached_boss(self.outcome_rank)
                or self._number(frame, ((238, 929, 285, 952), (241, 930, 281, 950)))) if self.boss_mode(frame) else None
        self.readings.append((player, boss))
        if player is None or len(self.readings) < 2:
            return None
        previous, _ = self.readings[0]
        if previous is None or not 0 <= player-previous <= 10:
            return None
        self.player_level = player
        if boss is not None:
            self.boss_level = boss
            self.boss_rank = self.outcome_rank
        return player, boss

    @staticmethod
    def _valid_boss_cache(cache):
        return (isinstance(cache,dict) and cache.get('source') == 'sea_god_header'
                and type(cache.get('rank')) is int and cache['rank'] == 39
                and type(cache.get('level')) is int and 1 <= cache['level'] <= 10000)

    def _cached_boss(self, rank):
        cache = self.state.get('quest_boss_level')
        return cache['level'] if self._valid_boss_cache(cache) and cache['rank'] == rank else None

    def observe_boss_header(self, frame, rank, *, selected=False):
        """Remember a target only after the caller verifies the real encounter.

        The gold meter can cover the old range label. Read the explicit Lv
        prefix in two views of the battle title, in two consecutive frames.
        Ordinary fish, dark menus and a previous RANK cannot supply a target.
        """
        if (not selected or rank != self.FINAL_BOSS_RANK or frame.shape[:2] != (1280,720)
                or self.state.get('rank',rank) != rank or not self.boss_mode(frame)):
            self.boss_header_readings.clear()
            return None
        values = []
        for x,y,X,Y in ((232,172,294,196),(233,171,293,198)):
            reading = self.reader.read(frame[y:Y,x:X])
            match = re.fullmatch(r'Lv\.?\s*(\d{1,5})',reading.text)
            if match is None or reading.confidence < .95:
                self.boss_header_readings.clear()
                return None
            values.append(int(match[1]))
        if values[0] != values[1] or not 1 <= values[0] <= 10000:
            self.boss_header_readings.clear()
            return None
        self.boss_header_readings.append((rank,values[0]))
        if len(self.boss_header_readings) < 2 or len(set(self.boss_header_readings)) != 1:
            return None
        boss = values[0]
        self.boss_level,self.boss_rank = boss,rank
        cache = {'rank':rank,'level':boss,'source':'sea_god_header'}
        changed = self.state.get('quest_boss_level') != cache
        if self.state.get('rank',rank) == rank:
            self.state['quest_boss_level'] = cache
            if self.state['mode'] != 'idle' and not self.state.get('boss_level'):
                self.state['boss_level'] = boss
                changed = True
            if changed:
                self._save()
                self.event(f'已记录本关海神等级 Lv{boss}（RANK {rank}），保留原练级和航行进度')
        return boss

    def _save(self):
        saved = {'version': 1, **self.state}
        from drift.core.persistence import atomic_write_text
        atomic_write_text(self.path, json.dumps(saved, ensure_ascii=False, indent=2) + '\n')

    def clear(self):
        if self.state.get('sailing_enabled'):
            raise RuntimeError('必须确认自动航行停止后才能清除练级记录')
        if self.state['mode'] != 'idle' or self.state.get('quest_boss_level'):
            self.state = {'mode': 'idle'}
            self._save()
        self.boss_level = self.boss_rank = None
        self.boss_header_readings.clear()
        self.failures = 0

    def set_sailing(self, enabled, verified=False, *, profile=None, rank=None):
        extra = {}
        if enabled and profile is not None:
            if profile not in ('resources','boss') or (profile == 'boss' and
                    (type(rank) is not int or not 0 <= rank < 39)):
                raise ValueError('自动航行路线或当前 RANK 无效')
            extra['sailing_profile'] = profile
            if profile == 'boss':
                extra['sailing_started_rank'] = rank
        if (self.state.get('sailing_enabled',False) != enabled
                or self.state.get('sailing_verified',False) != (enabled and verified)
                or any(self.state.get(key) != value for key,value in extra.items())):
            self.state.update(sailing_enabled=enabled,sailing_verified=enabled and verified,**extra)
            self._save()

    def retry_gap(self, rank=None):
        if rank is None:
            rank = self.state.get('rank', self.outcome_rank)
        key = 'final_boss_retry_level_gap' if rank == self.FINAL_BOSS_RANK else 'retry_level_gap'
        return self.options[key]

    def retry_level(self, boss=None, rank=None):
        if boss is None:
            boss = self.state.get('boss_level') if self.state['mode'] != 'idle' else self.boss_level
        return max(1, boss-self.retry_gap(rank)) if boss else None

    def ready_to_retry(self):
        target = self.retry_level()
        return target is not None and self.player_level is not None and self.player_level >= target

    def begin(self, rank, player, boss, *, reason='failure'):
        cache = self.state.get('quest_boss_level')
        boss = self._cached_boss(rank) or boss or (self.boss_level if self.boss_rank == rank else None)
        self.state = {'mode': 'farming', 'rank': rank, 'start_level': player,
                      'boss_level': boss or 0, 'started_at': time.time()}
        if self._valid_boss_cache(cache) and cache['rank'] == rank:
            self.state['quest_boss_level'] = cache
        self._save()
        self.failures = 0
        target = self.retry_level(boss, rank)
        trigger = '最后一个 BOSS 尚未达到挑战等级' if reason == 'final_boss' else '已确认钓鱼失败'
        self.event(f'{trigger}，BOSS Lv{boss or "未知"} / 角色 Lv{player}：开始钓鱼练级，'
                   f'{"达到 Lv"+str(target)+" 后试打" if target else "等待确认 BOSS 等级后试打"}；继续已开启的武器强化')

    def trial(self, seconds=None):
        self.failures = 0
        self.failure_latched = False
        self.state['mode'] = 'trial'
        self.state['started_at'] = time.time()
        self.state['trial_seconds'] = seconds or self.TRIAL_SECONDS
        self._save()

    def evaluate(self, frame, rank, *, main_known=False, quest_selected=True):
        self.observe_outcome(frame, rank)
        levels = self.levels(frame,main_known=True) if main_known else self.levels(frame)
        if levels is None:
            return None
        player, boss = levels
        if self.state.get('rank', rank) != rank:
            self.clear()
        mode, at_boss = self.state['mode'], self.boss_mode(frame)
        if not self.options['enabled']:
            self.clear()
            return 'retry' if mode == 'farming' and not at_boss else None
        if mode == 'farming':
            gain = player-self.state['start_level']
            # Freeze the quest boss level saved before moving; a normal fish at
            # the training location is never the retry target. Missing OCR waits.
            target = self.state.get('boss_level') or None
            if target is None and at_boss and boss:
                self.state['boss_level'] = target = boss
                self._save()
            if target and player >= self.retry_level(target, rank):
                self.trial()
                self.event(f'练级本轮提升 {max(0, gain)} 级（当前 Lv{player} / BOSS Lv{target}，'
                           f'等级差不超过 {self.retry_gap(rank)}），返回任务 BOSS 试打')
                return 'retry'
            return 'train' if at_boss else 'farming'
        # RANK 39 -> 40 is the user's final-boss exception: read its actual
        # level at the quest location, then train before further attempts.
        # Earlier quests still require confirmed fishing failures.
        if (quest_selected and rank == self.FINAL_BOSS_RANK and at_boss and boss
                and player < self.retry_level(boss, rank)):
            self.begin(rank, player, boss, reason='final_boss')
            return 'train'
        if mode == 'trial':
            if not at_boss:
                return 'retry'
            if self.failures >= self.options['failure_limit']:
                if not self._known_boss(rank,boss):
                    return self._wait_for_boss_level()
                self.begin(rank, player, boss)
                return 'train'
            if time.time()-self.state['started_at'] >= self.state.get('trial_seconds', self.TRIAL_SECONDS):
                self.clear()
                return None
            return 'trial'
        if at_boss and self.failures >= self.options['failure_limit']:
            if not self._known_boss(rank,boss):
                return self._wait_for_boss_level()
            self.begin(rank, player, boss)
            return 'train' if at_boss else 'farming'
        return None

    def _known_boss(self, rank, value=None):
        return self._cached_boss(rank) or value or (self.boss_level if self.boss_rank == rank else None)

    def _wait_for_boss_level(self):
        now = time.monotonic()
        if now-self.last_missing_boss_at >= 60:
            self.event('本关 BOSS 等级未读清，保留失败次数并留在任务点等待；不开始未知目标的练级航行')
            self.last_missing_boss_at = now
        return None

    def training_target(self, frame, boss=None):
        # Only red fishing-location pins; ports and facilities have different icons.
        area = (20, 175, 700, 1115)
        x1, y1, x2, y2 = area
        template = self.assets['map-pin']
        crop = frame[y1:y2, x1:x2]
        player = self._match(frame, 'map-player', area, .86)
        if boss is None and player is None:
            return None
        anchor = boss[:2] if boss is not None else player
        full_scores = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        full_scores = np.nan_to_num(full_scores, nan=-1, posinf=-1, neginf=-1)
        for fallback in (False, True):
            # Preserve existing choices when the full template identifies a safe
            # pin. Only then fall back to its red shading, retaining a full-shape
            # check at the same position and all existing color/anchor guards.
            if fallback:
                scores = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED, mask=self.pin_mask)
                scores = np.nan_to_num(scores, nan=-1, posinf=-1, neginf=-1)
                scores[full_scores < .60] = -1
            else:
                scores = full_scores.copy()
            candidates = []
            for _ in range(30):
                _, score, _, position = cv2.minMaxLoc(scores)
                if score < (.91 if fallback else .83):
                    break
                x, y = position
                center = (x1+x+template.shape[1]//2, y1+y+template.shape[0]//2)
                scores[max(0, y-30):y+31, max(0, x-30):x+31] = -1
                patch = cv2.cvtColor(frame[center[1]-10:center[1]+10, center[0]-10:center[0]+10], cv2.COLOR_RGB2HSV)
                red = ((patch[:, :, 0] <= 15) | (patch[:, :, 0] >= 170)) & (patch[:, :, 1] >= 80) & (patch[:, :, 2] >= 100)
                if float(red.mean()) < .3:
                    continue
                if any(np.hypot(center[0]-point[0], center[1]-point[1]) < 100 for point in (boss, player) if point is not None):
                    continue
                distance = np.hypot(center[0]-anchor[0], center[1]-anchor[1])
                candidates.append((distance, center))
            if candidates:
                return min(candidates)[1]
        return None

    def status(self):
        mode = self.state['mode']
        boss = self.state.get('boss_level') or self.boss_level
        target = self.retry_level(boss)
        return {'enabled': self.options['enabled'], 'mode': mode, 'player_level': self.player_level,
                'boss_level': self.boss_level, 'start_level': self.state.get('start_level'),
                'level_gain': self.options['level_gain'], 'minutes': self.options['minutes'],
                'retry_level_gap': self.retry_gap(), 'retry_level': target,
                'final_boss': self.state.get('rank',self.outcome_rank) == self.FINAL_BOSS_RANK,
                'final_boss_retry_level_gap': self.options['final_boss_retry_level_gap'],
                'failures': self.failures, 'failure_limit': self.options['failure_limit'],
                'elapsed': max(0, int(time.time()-self.state['started_at'])) if mode != 'idle' else 0}

# Keep legacy task logic; delegate every record read/write to shared storage.
from drift.core.persistence import ReliablePath as Path

# Shared reliability layer; retain all other deployed methods.
def _save(self):
    saved = {'version': 1, **self.state}
    from drift.core.persistence import atomic_write_text
    atomic_write_text(self.path, json.dumps(saved, ensure_ascii=False, indent=2) + '\n')

BossTraining._save = _save



# Module-local compatibility for every authoritative record read/write.
from drift.core.persistence import ReliablePath as Path
