"""Locate the achievement page, then select its next green reward button.

SPDX-License-Identifier: GPL-3.0-only
"""
import hashlib
import math

import cv2
import numpy as np

from alas_core.button import Button


class GreenRewardListButton(Button):
    def __init__(self, anchor, layout, source_size, name):
        super().__init__(area=(0, 0, 1, 1), color=(0, 0, 0), button=(0, 0, 1, 1), name=name)
        self.anchor = anchor
        self.row_count = int(layout['rows'])
        self.row_step = float(layout['row_step'])
        self.minimum_fraction = float(layout.get('minimum_green_fraction', 0.65))
        if not 1 <= self.row_count <= 40 or not math.isfinite(self.row_step) or self.row_step <= 0:
            raise ValueError('奖励列表行数或间距无效')
        if not 0.5 <= self.minimum_fraction <= 1:
            raise ValueError('绿色背景比例必须介于 0.5 和 1 之间')
        self.hsv_min = np.asarray(layout.get('hsv_min', [28, 110, 65]), dtype=np.int32)
        self.hsv_max = np.asarray(layout.get('hsv_max', [65, 255, 255]), dtype=np.int32)
        if (self.hsv_min.shape != (3,) or self.hsv_max.shape != (3,)
                or (self.hsv_min < 0).any() or (self.hsv_max > [179, 255, 255]).any()
                or (self.hsv_min > self.hsv_max).any()):
            raise ValueError('奖励按钮 HSV 范围无效')
        self.row_areas = []
        source_width, source_height = source_size
        for index in range(self.row_count):
            regions = []
            for field in ('button_area', 'color_area', 'progress_area'):
                original = layout[field]
                if not isinstance(original, list) or len(original) != 4:
                    raise ValueError('奖励列表区域必须是 [左, 上, 右, 下]')
                x1, y1, x2, y2 = [float(v) for v in original]
                y1 += index * self.row_step
                y2 += index * self.row_step
                if not (0 <= x1 < x2 <= source_width and 0 <= y1 < y2 <= source_height):
                    raise ValueError('奖励列表区域超出源图')
                regions.append((x1, y1, x2, y2))
            self.row_areas.append(regions)
        self.target_key = None
        self.progress_key = None
        self.green_count = 0
        self.last_similarity = -1.0

    def source_to_screen(self, area):
        return self.anchor.source_to_screen(area)

    def match(self, image, offset=0, similarity=0.92):
        self.target_key = None
        self.progress_key = None
        self.green_count = 0
        self.last_similarity = -1.0
        if not self.anchor.match(image, offset=offset, similarity=similarity):
            return False
        height, width = image.shape[:2]
        choices = []
        for index, regions in enumerate(self.row_areas):
            button_box, color_box, progress_box = [self.source_to_screen(area) for area in regions]
            if not all(0 <= b[0] < b[2] <= width and 0 <= b[1] < b[3] <= height
                       for b in (button_box, color_box, progress_box)):
                continue
            x1, y1, x2, y2 = color_box
            hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_RGB2HSV)
            green = ((hsv >= self.hsv_min) & (hsv <= self.hsv_max)).all(axis=2)
            if float(green.mean()) >= self.minimum_fraction:
                choices.append((index, button_box, progress_box))
        self.green_count = len(choices)
        if not choices:
            return False
        index, self._button_offset, progress_box = choices[0]
        self.target_key = index
        x1, y1, x2, y2 = progress_box
        progress = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
        self.progress_key = (index, hashlib.blake2b(progress.tobytes(), digest_size=8).hexdigest())
        self.last_similarity = self.anchor.last_similarity
        return True


