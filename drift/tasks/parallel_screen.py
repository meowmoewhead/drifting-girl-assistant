"""平行世界画面探针:判定"当前是不是平行世界"、读 RANK、取阶段笔画。

标尺等级校准原先靠构造一个带空导航回调的 :class:`ParallelWorld` 来拿这几个判定,
顺带把整个平行世界导航/练级状态机拖了进来。这些判定本身与导航无关,抽成基类之后:

- 标尺校准可以直接用它独立成立,不再实例化 ``ParallelWorld``;
- 平行世界功能继承它并复用同一份实现,行为不会漂移。

判定逻辑与 ``ParallelWorld`` 里的原实现逐字一致(badge 判定 / RANK 四视图投票 /
阶段笔画 / 航行菜单里的暗化 RANK)。本模块**只读画面**,不发送任何输入。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from drift.perception.digit_ocr import read_numeric_views

#: 平行世界识别模板(png 文件名即键名)
REQUIRED_ASSETS = ('world-badge', 'next-arrow', 'map-close', 'map-boss',
                   'map-observatory', 'observatory-label', 'gate-close',
                   'gate-activate', 'return-question', 'enter-question',
                   'search-question', 'continue', 'panel-close')


class ParallelScreen:
    """平行世界画面判定基类。

    子类可以覆盖 :meth:`match` 换成自己的模板匹配器(平行世界功能就是这么做的);
    基类自带一个与原实现等价的朴素匹配:同一区域、同一阈值、返回匹配中心。
    ``sailing`` 可选,只有航行菜单里的 RANK 读取需要它。

    方法名沿用 ``ParallelWorld`` 内部原有的下划线命名(``_parallel`` / ``_rank_value`` /
    ``_stage_strokes`` / ``_fixed``),因为平行世界里已有大量调用点,改名只会制造噪音。
    """

    def __init__(self, root, reader, *, match=None, sailing=None, assets_dir='assets/parallel'):
        root = Path(root)
        self.reader = reader
        self.sailing = sailing
        self.assets = {path.stem: cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                       for path in (root / assets_dir).glob('*.png')}
        if (set(self.assets) != set(REQUIRED_ASSETS)
                or any(value is None for value in self.assets.values())):
            raise RuntimeError('平行世界识别模板不完整')
        self._match_override = match

    # --- 模板匹配(子类可覆盖) ---
    def match(self, frame, name, area, threshold=.88):
        if self._match_override is not None:
            return self._match_override(frame, name, area, threshold)
        x1, y1, x2, y2 = area
        patch = self.assets[name]
        if x2 - x1 < patch.shape[1] or y2 - y1 < patch.shape[0]:
            return None
        scores = cv2.matchTemplate(frame[y1:y2, x1:x2], patch, cv2.TM_CCOEFF_NORMED)
        _, score, _, position = cv2.minMaxLoc(scores)
        if np.isfinite(score) and score >= threshold:
            return (x1 + position[0] + patch.shape[1] // 2,
                    y1 + position[1] + patch.shape[0] // 2, score)
        return None

    def _fixed(self, frame, name, box, threshold=.88, margin=7):
        x1, y1, x2, y2 = box
        return self.match(frame, name,
                          (max(0, x1 - margin), max(0, y1 - margin),
                           min(720, x2 + margin), min(1280, y2 + margin)), threshold)

    # --- 判定 ---
    def _parallel(self, frame):
        """画面左上角是否有平行世界徽章。"""
        return self._fixed(frame, 'world-badge', (13, 46, 39, 72), .92) is not None

    def _observatory(self, frame):
        return self._fixed(frame, 'observatory-label', (306, 12, 411, 45), .9) is not None

    def _map(self, frame):
        return (self._fixed(frame, 'map-close', (632, 1151, 705, 1225), .93) is not None
                and float((np.max(frame[1170:1210, 644:686], axis=2) >= 210).mean()) > .30)

    @staticmethod
    def _stage_strokes(frame):
        """阶段徽章的绿色笔画指纹(用于确认"还是同一阶段")。"""
        if frame.shape[:2] != (1280, 720):
            return None
        badge = frame[45:71, 40:68].astype(np.int16)
        ink = ((badge[:, :, 1] - badge[:, :, 0] >= 20) & (badge[:, :, 2] - badge[:, :, 0] >= 30)
               & (badge.max(axis=2) >= 170))
        return ink.tobytes() if int(ink.sum()) >= 8 else None

    def _sailing_menu_rank(self, frame):
        """只在一个完整明亮的 AUTO 面板后面读暗化表头的 RANK。"""
        if not self.sailing or not self.sailing.menu(frame) or not self._parallel(frame):
            return None
        views = []
        for x, y, X, Y in ((71, 9, 136, 59), (80, 9, 130, 59)):
            crop = frame[y:Y, x:X]
            # The panel dims the white digit strokes to 149. Both wide
            # crops and both ink thresholds must independently agree.
            for threshold in (110, 130):
                ink = np.all(crop >= threshold, axis=2)
                binary = np.repeat(np.where(ink, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
                views.append(binary)
        readings = read_numeric_views(self.reader, views)
        self.menu_rank_probe=dict(method='white_ink',views=[dict(value=r.value,
            confidence=round(r.confidence,4)) for r in readings])
        strong={r.value for r in readings if r.value is not None
                and 0<=r.value<=40 and r.confidence>=.95}
        if len(strong)>1:return None
        if len(readings)==4 and all(r.value is not None and 0<=r.value<=40
                and r.confidence>=.95 for r in readings):
            return readings[0].value
        return self._outlined_menu_rank(frame,readings)

    def _outlined_menu_rank(self, frame, previous):
        if (frame.shape!=(1280,720,3) or not self.sailing
                or not self.sailing.menu(frame) or not self._parallel(frame)):return None
        strong={r.value for r in previous if r.value is not None
                and 0<=r.value<=40 and r.confidence>=.95}
        if len(strong)>1:return None
        # Outlined "5" can become "s" after thresholding. Locate the complete
        # current digit foreground, then retain its original outline/colour.
        # No S->5 substitution, saved rank or lower confidence is permitted.
        crop=frame[9:59,71:136];fallback=[];signatures=set()
        for threshold in (130,140):
            ink=np.all(crop>=threshold,axis=2).astype(np.uint8)
            count,labels,stats,_=cv2.connectedComponentsWithStats(ink,8)
            components=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=30
                        and stats[i,cv2.CC_STAT_HEIGHT]>=18]
            if not 1<=len(components)<=2:return None
            selected=np.isin(labels,components).astype(np.uint8)
            x,y,w,h=cv2.boundingRect(selected)
            if x<=1 or y<=1 or x+w>=crop.shape[1]-1 or y+h>=crop.shape[0]-1:return None
            for margin in (8,9):
                sample=crop[max(0,y-margin):min(crop.shape[0],y+h+margin),
                            max(0,x-margin):min(crop.shape[1],x+w+margin)]
                signatures.add((sample.shape,sample.tobytes()));fallback.append(sample)
        if len(signatures)<3:return None
        decoded=read_numeric_views(self.reader,fallback)
        self.menu_rank_probe=dict(method='outlined_digit',views=[dict(value=r.value,
            confidence=round(r.confidence,4)) for r in decoded])
        if (len(decoded)!=4 or any(r.value is None or not 0<=r.value<=40
                or r.confidence<.95 for r in decoded) or len({r.value for r in decoded})!=1):return None
        value=decoded[0].value
        return value if not strong or strong=={value} else None

    def _rank_value(self, frame):
        """读 RANK 数字:四组裁图 × 三种预处理投票,冲突则不算读出。"""
        if self.sailing and self.sailing.menu(frame):
            return self._sailing_menu_rank(frame)
        # The stylized 5 is often read as S from the original tight crop.
        # Compare differently framed views; the wider pair also preserves
        # both digits once RANK reaches 10. Thresholded ink helps with 0/40.
        areas = ((80, 12, 123, 58), (83, 9, 122, 59),
                 (71, 9, 136, 59), (80, 9, 130, 59))
        views = []
        for index, (x1, y1, x2, y2) in enumerate(areas):
            crop = frame[y1:y2, x1:x2]
            views.append((index, 'raw', crop))
            if index in (2, 3):
                ink = np.all(crop >= 200, axis=2)
                binary = np.repeat(np.where(ink, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
                views.append((index, 'white', binary))
            elif index == 1:
                ink = np.max(crop, axis=2) < 80
                binary = np.repeat(np.where(ink, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
                views.append((index, 'black', binary))
        readings = [(index, mode, reading) for (index, mode, _), reading in
                    zip(views, read_numeric_views(self.reader, [view[2] for view in views]))]
        votes = {}
        for index, mode, reading in readings:
            value = reading.value
            if value is None or not 0 <= value <= 40:
                continue
            minimum = .80 if value == 0 else .95 if value == 40 else .90
            if reading.confidence >= minimum:
                votes.setdefault(value, []).append((index, mode, reading.confidence))
        value = None
        for candidate, group in votes.items():
            regions = {index for index, _, _ in group}
            if len(regions) < 2:
                continue
            if candidate >= 10 and not {2, 3}.issubset(regions):
                continue
            if candidate == 40 and len(group) < 2:
                continue
            if value is not None:
                value = None  # Conflicting strong readings are not a rank.
                break
            value = candidate
        return value


from drift.perception.menu_rank import install as _install_menu_rank
_install_menu_rank(ParallelScreen)
del _install_menu_rank

from drift.perception.menu_rank import install as _install_menu_rank
_install_menu_rank(ParallelScreen)
del _install_menu_rank
