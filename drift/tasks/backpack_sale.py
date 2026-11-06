"""Sell a full fishing-loot bag at a port, preserving locks and the origin."""
import base64
from collections import deque
import json
from pathlib import Path
import time

import cv2
import numpy as np


class BackpackSale:
    def __init__(self, root, click, event, reader, interval=60, state_path=None, swipe=None):
        self.root = Path(root)
        self.click, self.event, self.reader = click, event, reader
        self.swipe = swipe
        self.enabled = lambda: True
        self.main_guard = lambda frame: False
        self.before_route = lambda frame: True
        self.interval = float(interval)
        self.path = Path(state_path) if state_path else self.root/'bag_sale_state.json'
        self.assets = {}
        paths = {name: self.root/'assets/backpack'/f'{name}.png' for name in (
            'bag-icon', 'map-port', 'header-port', 'port-trade', 'moving',
            'bag-title', 'tab-first', 'sell', 'map-player-current')}
        paths.update({name: self.root/'assets/equipment'/f'{name}.png'
                      for name in ('entry', 'guard-star', 'guard-diamond', 'main-auto')})
        paths.update({'map-close': self.root/'assets/parallel/map-close.png',
                      'world-badge': self.root/'assets/parallel/world-badge.png',
                      'map-player': self.root/'assets/training/map-player.png',
                      'map-legend': self.root/'assets/task-popup/map-legend.png',
                      'auto-label': self.root/'assets/training-sailing/auto-label.png',
                      'inactive-face': self.root/'assets/training-sailing/inactive-face.png',
                      'active-arrow': self.root/'assets/training-sailing/active-arrow.png',
                      'port-anchor': self.root/'assets/fishing-meter/port-anchor.png'})
        for name, path in paths.items():
            data = cv2.imread(str(path))
            if data is None:
                raise RuntimeError('背包出售识别模板不完整')
            self.assets[name] = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        if self.assets['map-port'].shape[:2] != (45, 43):
            raise RuntimeError('背包出售港口图标尺寸未验证')
        # Keep the anchor pole, both blue arms, bowl and their outline. The
        # original crop also contains city-name strokes and animated scenery.
        self.port_mask = np.zeros((45, 43), dtype=np.uint8)
        cv2.fillPoly(self.port_mask, [np.array(((19,12),(27,12),(28,18),(40,20),
            (42,28),(34,39),(24,44),(11,41),(0,30),(2,21),(13,20),(16,12)))], 255)
        self.phase, self.since = 'idle', time.monotonic()
        self.next_action = self.next_check = 0.
        self.stable = 0
        self.capacity_readings = deque(maxlen=3)
        self.capacity_before = None
        self.capacity_after = None
        self.origin = None
        self.parallel = None
        self.attempted = False
        self.cycles = 0
        self.checkpoint = None
        self.scan_swipes = self.scan_frames = 0
        self.scan_context = None
        self.route_context = None
        self.return_scan_east = False
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text(encoding='utf-8-sig'))
                if saved.get('version') == 1 and saved.get('pending'):
                    image = cv2.imdecode(np.frombuffer(base64.b64decode(saved['origin']), dtype=np.uint8), cv2.IMREAD_COLOR) if saved.get('origin') else None
                    if image is not None and image.shape != (32, 140, 3):
                        raise ValueError('Invalid return landmark')
                    if not isinstance(saved.get('parallel'), bool) or saved.get('total') != 30 or not isinstance(saved.get('attempted'), bool):
                        raise ValueError('Invalid sale checkpoint')
                    self.origin, self.parallel = image, saved['parallel']
                    self.capacity_before, self.attempted = saved['total'], saved['attempted']
                    self.phase = 'resume'
            except (OSError, ValueError, TypeError, KeyError, cv2.error):
                raise RuntimeError('背包出售返回记录损坏，保留当前画面以免误选地点')

    @property
    def active(self):
        return self.phase != 'idle'

    def suspend(self, seconds):
        if self.active and seconds > 0:
            self.since += seconds
            self.stable = 0

    def status(self):
        return {'phase': self.phase, 'cycles': self.cycles,
                'capacity_before': self.capacity_before, 'capacity_after': self.capacity_after,
                'map_swipes': self.scan_swipes}

    def _find(self, frame, name, box, threshold=.92, mask=None):
        patch = self.assets[name] if isinstance(name, str) else name
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < patch.shape[0] or crop.shape[1] < patch.shape[1]:
            return None
        scores = cv2.matchTemplate(crop, patch, cv2.TM_CCOEFF_NORMED, mask=mask)
        scores = np.nan_to_num(scores, nan=-1, posinf=-1, neginf=-1)
        _, score, _, pos = cv2.minMaxLoc(scores)
        return (x1+pos[0]+patch.shape[1]//2, y1+pos[1]+patch.shape[0]//2) if np.isfinite(score) and score >= threshold else None

    @staticmethod
    def _bright(frame, box):
        x1, y1, x2, y2 = box
        return float((np.max(frame[y1:y2, x1:x2], axis=2) >= 210).mean()) >= .30

    def tab(self, frame):
        return self._find(frame, 'tab-first', (12, 700, 94, 910), .82)

    def first(self, frame):
        return bool(self._find(frame, 'bag-title', (38, 844, 164, 893), .94)
                    and self._find(frame, 'sell', (550, 842, 691, 888), .93)
                    and self._bright(frame, (555, 848, 685, 883)))

    def main(self, frame):
        return (self.tab(frame) is None
                and (self.main_guard(frame) or all(self._find(frame, name, box, score) for name, box, score in (
                    ('entry', (327, 1140, 378, 1192), .8),
                    ('guard-star', (208, 1137, 269, 1192), .91),
                    ('guard-diamond', (539, 1137, 600, 1186), .91),
                    ('main-auto', (276, 881, 310, 913), .91)))
                and self._bright(frame, (558, 1147, 585, 1168))))

    def map(self, frame):
        return bool(self._find(frame, 'map-close', (629, 1148, 708, 1228), .93)
                    and self._bright(frame, (644, 1170, 686, 1210)))

    def moving(self, frame):
        return self._find(frame, 'moving', (283, 38, 480, 79), .9) is not None

    def port(self, frame):
        return ((self._find(frame, 'header-port', (248, 33, 295, 82), .92) is not None
                 or self.main(frame) and self._find(frame, 'port-trade', (632, 1048, 718, 1132), .91) is not None)
                and not self.moving(frame))

    def dock(self, frame):
        # The trade icon rotates between two otherwise valid fresh frames.
        # Require the fixed port marker and central docked anchor together;
        # a trade animation or a header marker alone cannot prove arrival.
        static = (self._find(frame,'header-port',(248,33,295,82),.92) is not None
                  and self._find(frame,'port-anchor',(310,954,416,1064),.94) is not None)
        trade = self.port(frame) and self._find(frame,'port-trade',(632,1048,718,1132),.91) is not None
        return self.main(frame) and not self.moving(frame) and (static or trade)

    def stopped_panel(self, frame):
        """Read a positive OFF icon on a recognized port bag or stopped map.

        Gameplay's lower AUTO is covered by these panels. An unknown top icon
        still rejects every input; saved flags alone cannot authorize a click.
        """
        if frame.shape[:2] != (1280,720) or self.parallel is None:
            return False
        if self._find(frame,'active-arrow',(192,2,248,44),.96) is not None:
            return False
        if self._scan_map(frame,allow_moving=True):
            return True
        return (self.tab(frame) is not None
                and self._find(frame,'header-port',(248,33,295,82),.92) is not None
                and self._bright(frame,(558,1147,585,1168))
                and self._find(frame,'auto-label',(184,32,255,67),.90) is not None
                and self._find(frame,'inactive-face',(192,2,248,44),.96) is not None
                and (self._find(frame,'world-badge',(10,43,42,75),.92) is not None)==self.parallel)

    def capacity(self, frame):
        votes = {}
        for index, (x1, y1, x2, y2) in enumerate(((625, 1210, 706, 1238), (637, 1212, 704, 1236))):
            crop = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            ink = (np.max(crop, axis=2) < 160) | ((hsv[:, :, 0] < 19) & (hsv[:, :, 1] > 80))
            binary = np.repeat(np.where(ink, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
            for kind, view in enumerate((crop, binary)):
                reading = self.reader.read_counter(view)
                if reading.current is not None and reading.total == 30 and reading.confidence >= .93:
                    votes.setdefault((reading.current, reading.total), set()).add((index, kind))
        if len(votes) == 1:
            value, views = next(iter(votes.items()))
            if len(views) >= 2:
                return value
        return None

    def unlocked(self, frame):
        # Only border strips determine locks; green material artwork is irrelevant.
        for row in range(3):
            for col in range(10):
                x, y = 28+68*col, 915+68*row
                border = np.concatenate((frame[y:y+5, x:x+56].reshape(-1, 3),
                                         frame[y:y+56, x:x+5].reshape(-1, 3)))
                hsv = cv2.cvtColor(border.reshape(1, -1, 3), cv2.COLOR_RGB2HSV)[0]
                green = (hsv[:, 0] >= 40) & (hsv[:, 0] <= 95) & (hsv[:, 1] >= 70) & (hsv[:, 2] >= 80)
                if float(green.mean()) < .15:
                    return True
        return False

    def _set(self, phase):
        self.phase, self.since, self.stable = phase, time.monotonic(), 0
        self.capacity_readings.clear()
        self.scan_swipes = self.scan_frames = 0
        self.scan_context = None

    def _confirmed(self, condition):
        self.stable = self.stable+1 if condition else 0
        return self.stable >= 3

    def _save(self, pending=True):
        image = base64.b64encode(cv2.imencode('.png', self.origin)[1]).decode('ascii') if self.origin is not None else None
        saved = {'version': 1, 'pending': pending, 'parallel': self.parallel,
                 'origin': image, 'total': self.capacity_before, 'attempted': self.attempted}
        from drift.core.persistence import atomic_write_text
        atomic_write_text(self.path, json.dumps(saved) + '\n')

    def _finish(self):
        self._save(False)
        self.cycles += 1
        self.next_check = time.monotonic()+max(self.interval, 900 if self.capacity_after == self.capacity_before else self.interval)
        self._set('idle')
        self.event('背包出售：已返回原地点，继续练级、强化和其他任务')

    def _origin_target(self, frame):
        if self.origin is None:
            return None
        # Match the saved place-name lettering, excluding shifting map scenery.
        ink = np.all(self.origin >= 190, axis=2).astype(np.uint8)
        if int(ink.sum()) < 30:
            return None
        mask = cv2.dilate(ink, np.ones((3, 3), np.uint8))*255
        scores = cv2.matchTemplate(frame[175:1115, 20:700], self.origin, cv2.TM_CCOEFF_NORMED, mask=mask)
        scores = np.nan_to_num(scores, nan=-1, posinf=-1, neginf=-1)
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        # Fractional map panning changes font antialiasing. In the native replay
        # the exact name scores .878; every other name scores below .287.
        # Also reject a second plausible name rather than choosing between them.
        if score < .85:
            return None
        scores[max(0, y-45):y+46, max(0, x-100):x+101] = -1
        if cv2.minMaxLoc(scores)[1] >= .75:
            return None
        return 20+x+70, 175+y+16+36

    def _ports(self, frame):
        scores = cv2.matchTemplate(frame[175:1115, 20:700], self.assets['map-port'],
                                   cv2.TM_CCOEFF_NORMED, mask=self.port_mask)
        scores = np.nan_to_num(scores, nan=-1, posinf=-1, neginf=-1)
        ports = []
        for _ in range(16):
            _, score, _, (x, y) = cv2.minMaxLoc(scores)
            if score < .94:
                break
            ports.append((20+x+self.assets['map-port'].shape[1]//2,
                          175+y+self.assets['map-port'].shape[0]//2))
            scores[max(0, y-30):y+31, max(0, x-30):x+31] = -1
        return ports

    def _scan_map(self, frame, *, allow_moving=False):
        return (frame.shape[:2] == (1280, 720) and self.map(frame)
                and (allow_moving or not self.moving(frame))
                and self._find(frame, 'map-legend', (9, 1136, 405, 1241), .93) is not None
                and self._find(frame, 'auto-label', (184, 32, 255, 67), .90) is not None
                and self._find(frame, 'inactive-face', (192, 2, 248, 44), .96) is not None
                and (self._find(frame, 'world-badge', (10, 43, 42, 75), .92) is not None) == self.parallel)

    @staticmethod
    def _context(frame):
        # Fixed RANK strokes, world/tier badge and current place name only.
        # Map scenery, travelling countdowns and player animations are excluded.
        return tuple(frame[y:Y, x:X].copy() for x, y, X, Y in
                     ((71, 9, 136, 59), (10, 43, 60, 75), (300, 12, 418, 36)))

    @staticmethod
    def _matches_context(expected_context, frame):
        if expected_context is None or frame.shape[:2] != (1280, 720):
            return False
        for expected, current in zip(expected_context, BackpackSale._context(frame)):
            score = cv2.matchTemplate(current, expected, cv2.TM_CCOEFF_NORMED)[0, 0]
            if not np.isfinite(score) or score < .98:
                return False
        return True

    def _same_context(self, frame):
        return self._matches_context(self.scan_context, frame)

    def validate_scan(self, frame):
        if (self.origin is None or not self._scan_map(frame) or not self._same_context(frame)):
            return False
        if self.phase == 'find_port':
            return self.enabled() and not self._ports(frame)
        if self.phase == 'find_origin':
            return self._origin_target(frame) is None
        return False

    def _scan(self, frame):
        if self.swipe is None:
            raise RuntimeError('背包出售当前地图未识别目标，已停止并保留返回记录')
        if not self._scan_map(frame):
            self.scan_frames = 0
            self.scan_context = None
            return True
        if not self._same_context(frame):
            self.scan_context = self._context(frame)
            self.scan_frames = 0
        self.scan_frames += 1
        if self.scan_frames < 3:
            return True
        if self.scan_swipes >= 12:
            raise RuntimeError('背包出售地图搜索达到12次上限，已停止并保留返回记录')
        # Search east from the initial viewport, then back across it to the west.
        # Do not reset the original 45-second phase deadline after a pan.
        east_first = self.phase == 'find_port' or self.return_scan_east
        east_now = east_first == (self.scan_swipes < 4)
        coordinates = (600, 600, 120, 600) if east_now else (120, 600, 600, 600)
        target = '港口' if self.phase == 'find_port' else '原地点'
        if self.swipe(*coordinates, f'背包出售：滑动地图搜索{target}') is False:
            self.scan_frames = 0
            return True
        self.scan_swipes += 1
        if self.phase == 'find_port':
            self.return_scan_east = not east_now
        self.scan_frames = 0
        self.next_action = time.monotonic()+1.2
        return True

    def validate(self, action, frame, x, y):
        if (self.parallel is not None and (self.main(frame) or self.map(frame))
                and (self._find(frame, 'world-badge', (10, 43, 42, 75), .92) is not None) != self.parallel):
            return False
        if action == 'map_open':
            return self.main(frame) and (self.phase != 'prepare' or not self.port(frame))
        if action == 'map_close':
            return (self._scan_map(frame, allow_moving=True)
                    and (self.route_context is None or self._matches_context(self.route_context, frame)))
        if action == 'bag_open':
            return self.dock(frame)
        if action == 'bag_close':
            return self.tab(frame) is not None and self._bright(frame, (558, 1147, 585, 1168))
        if action == 'tab_first':
            point = self.tab(frame)
            return point is not None and np.hypot(point[0]-x, point[1]-y) < 8
        if action == 'sell':
            return (self.enabled() and self.first(frame) and self.port(frame) and self.capacity(frame) == (30, 30)
                    and self.unlocked(frame))
        if action in ('port_select', 'origin_select'):
            if (not self._scan_map(frame) or self.route_context is not None
                    and not self._matches_context(self.route_context, frame)):
                return False
            point = (self._origin_target(frame) if action == 'origin_select'
                     else self._find(frame, 'map-port', (x-30, y-32, x+30, y+32), .94, self.port_mask))
            return point is not None and np.hypot(point[0]-x, point[1]-y) < 8
        return False

    def _act(self, action, x, y, label, phase):
        if self.click(x, y, label, action) is False:
            self.stable = 0
            return True
        self._set(phase)
        self.next_action = time.monotonic()+1.2
        return True

    def step(self, frame, rank=None):
        now = time.monotonic()
        if now < self.next_action:
            return self.active
        if self.phase == 'idle':
            if not self.enabled() or now < self.next_check or rank == 40 or not self.main(frame) or self.moving(frame):
                self.capacity_readings.clear()
                return False
            value = self.capacity(frame)
            self.capacity_readings.append(value)
            if len(self.capacity_readings) < 3 or any(v != (30, 30) for v in self.capacity_readings):
                return False
            self.capacity_before, self.capacity_after, self.origin, self.attempted = 30, None, None, False
            self.parallel = self._find(frame, 'world-badge', (10, 43, 42, 75), .92) is not None
            # Checkpoint before handing navigation over from game AUTO. A
            # restart during its stop confirmation must resume this sale.
            self._set('prepare')
            self._save()

        # AUTO must be confirmed off before any map, bag or sale input. This
        # also covers pending-sale recovery after restarting the worker.
        if not self.before_route(frame):
            if now-self.since > 45:
                raise RuntimeError('背包出售等待停航超时，已停止并保存返回记录')
            return True

        current_world = self._find(frame, 'world-badge', (10, 43, 42, 75), .92) is not None
        if current_world != self.parallel and (self.main(frame) or self.map(frame)):
            raise RuntimeError('背包出售期间世界发生变化，已停止以免使用旧地点')
        if now-self.since > (600 if self.phase in ('travel_port', 'travel_origin') else 45):
            raise RuntimeError('背包出售等待超时，已停止并保存返回记录')
        if self.map(frame):
            self.route_context = self._context(frame)
        if not self.enabled() and not self.attempted:
            self.attempted, self.capacity_after = True, self.capacity_before
            self._save()
            self.event('背包出售已关闭：取消出售并返回原地点')
            if self.tab(frame):
                return self._act('bag_close', 674, 1174, '背包出售：取消后关闭背包', 'open_return')
            if self.map(frame):
                self._set('find_origin' if self.origin is not None else 'close_origin_map')
            elif self.main(frame):
                self._set('open_return')
        if self.phase == 'prepare':
            if not self.main(frame) or self.moving(frame):
                self.stable = 0
                return True
            if self.port(frame):
                if self._confirmed(self.dock(frame)):
                    return self._act('bag_open', 674, 1174, '背包出售：港口打开满背包', 'bag')
                return True
            self.event('背包已满 30/30：前往当前世界的港口，保留锁定材料并返回原地点')
            return self._act('map_open', 350, 30, '背包出售：打开港口地图', 'find_port')
        if self.phase == 'resume':
            if self.map(frame):
                self._set('find_origin' if self.attempted else 'find_port')
            elif self.tab(frame):
                self._set('bag')
            elif self.main(frame):
                if self.port(frame):
                    self._set('open_return' if self.attempted else 'travel_port')
                elif self.attempted:
                    if self.origin is None:
                        self._finish()
                    else:
                        return self._act('map_open', 350, 30, '背包出售：恢复原地点返回路线', 'find_origin')
                else:
                    return self._act('map_open', 350, 30, '背包出售：续接港口路线', 'find_port')
            return True
        if self.phase == 'find_port' and self.main(frame):
            return self._act('map_open', 350, 30, '背包出售：重新打开港口地图', 'find_port')
        if self.phase == 'find_port' and self.map(frame):
            player = (self._find(frame, 'map-player-current', (20, 175, 700, 1115), .9)
                      or self._find(frame, 'map-player', (20, 175, 700, 1115), .86))
            if self.origin is None:
                if not player or not 90 <= player[0] <= 630 or player[1] < 230:
                    raise RuntimeError('背包出售未识别原地点，已停止而不离开')
                self.origin = frame[player[1]-52:player[1]-20, player[0]-70:player[0]+70].copy()
                self._save()
            ports = self._ports(frame)
            if not ports:
                return self._scan(frame)
            # The preserved origin remains valid even when a pan hides the player.
            anchor = player or (360, 645)
            point = min(ports, key=lambda p: np.hypot(p[0]-anchor[0], p[1]-anchor[1]))
            return self._act('port_select', *point, '背包出售：选择附近港口', 'close_port_map')
        if self.phase in ('close_port_map', 'close_origin_map'):
            target = 'travel_port' if self.phase == 'close_port_map' else 'travel_origin'
            if self.map(frame):
                return self._act('map_close', 668, 1187, '背包出售：关闭地图等待航行', target)
            self._set(target)
            return True
        if self.phase == 'travel_port' and self._confirmed(self.dock(frame)):
            return self._act('bag_open', 674, 1174, '背包出售：到达港口后打开背包', 'bag')
        if self.phase == 'bag':
            if self.attempted:
                if self.first(frame):
                    value = self.capacity(frame)
                    if self._confirmed(value is not None):
                        self.capacity_after = value[0]
                        self.event(f'背包出售后容量 {value[0]}/{value[1]}；锁定材料保留')
                        return self._act('bag_close', 674, 1174, '背包出售：关闭港口背包', 'open_return')
            elif self.first(frame):
                value = self.capacity(frame)
                if self._confirmed(value == (30, 30) and self.port(frame)):
                    if not self.unlocked(frame):
                        self.attempted, self.capacity_after = True, 30
                        self._save()
                        self.event('满背包均为锁定材料，保留物品并暂缓出售')
                        return self._act('bag_close', 674, 1174, '背包出售：保留锁定材料并关闭', 'open_return')
                    # Persist before committing; resume never submits this sale twice.
                    self.attempted = True
                    self._save()
                    if self.click(620, 866, '背包出售：出售第一标签的未锁定材料', 'sell') is False:
                        self.attempted = False
                        self._save()
                        self.stable = 0
                    else:
                        self._set('bag')
                        self.next_action = now+3
                    return True
            else:
                point = self.tab(frame)
                if point:
                    return self._act('tab_first', *point, '背包出售：切换钓鱼材料第一标签', 'bag')
        if self.phase == 'open_return' and self._confirmed(self.main(frame)):
            if self.origin is None:
                self._finish()
                return True
            return self._act('map_open', 350, 30, '背包出售：打开返回原地点的地图', 'find_origin')
        if self.phase == 'find_origin' and self.map(frame):
            point = self._origin_target(frame)
            if point:
                return self._act('origin_select', *point, '背包出售：重新识别并返回原地点', 'close_origin_map')
            return self._scan(frame)
        if self.phase == 'travel_origin' and self._confirmed(self.main(frame) and not self.moving(frame)):
            self._finish()
        return True

# Keep legacy task logic; delegate every record read/write to shared storage.
from drift.core.persistence import ReliablePath as Path

# Shared reliability layer; retain all other deployed methods.
def _save(self, pending=True):
    image = base64.b64encode(cv2.imencode('.png', self.origin)[1]).decode('ascii') if self.origin is not None else None
    saved = {'version': 1, 'pending': pending, 'parallel': self.parallel,
             'origin': image, 'total': self.capacity_before, 'attempted': self.attempted}
    from drift.core.persistence import atomic_write_text
    atomic_write_text(self.path, json.dumps(saved) + '\n')

BackpackSale._save = _save



# Module-local compatibility for every authoritative record read/write.
from drift.core.persistence import ReliablePath as Path
