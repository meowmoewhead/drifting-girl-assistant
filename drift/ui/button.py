"""带裁图、多尺度与 HSV 轮廓守卫的模板按钮。

从 app.CroppedTemplateButton 原样搬入;``app`` 仍按原名再导出。
"""

from __future__ import annotations

import cv2
import math

from alas_core.button import Button
from drift.core.util import hsv_mask, number


class CroppedTemplateButton(Button):
    """Match a stable fragment from a supplied icon using ALAS at several scales.

    Cropping keeps changing counters and badges outside the recognition input.
    Optional companion fragments must match near the expected relative position.
    """

    def __init__(self, template, search, scales, name, source_area=None, click_source_area=None,
                 required_templates=(), match_hsv=None, color_condition=None, required_tolerance=2):
        self.match_hsv = match_hsv
        self.color_condition = color_condition
        if match_hsv is not None:
            template = hsv_mask(template, match_hsv)
            if template.std() < 2:
                raise ValueError("颜色轮廓模板必须同时包含目标和背景")
        height, width = template.shape[:2]
        super().__init__(area=(0, 0, width, height), color=(0, 0, 0),
                         button=(0, 0, width, height), name=name)
        self.source_area = source_area or (0, 0, width, height)
        self.click_source_area = click_source_area
        self.matched_box = None
        self.variants = []
        for scale in scales:
            scale = number(scale, 0.5, 2)
            w, h = max(8, round(width * scale)), max(8, round(height * scale))
            if w > search[2] - search[0] or h > search[3] - search[1]:
                continue
            candidate = Button(area=(0, 0, w, h), color=(0, 0, 0),
                               button=(0, 0, w, h), name=f"{name}@{scale}")
            candidate.image = cv2.resize(template, (w, h), interpolation=cv2.INTER_LINEAR)
            candidate._match_init = True
            offset = (search[0], search[1], search[2] - w, search[3] - h)
            companions = []
            sx, sy = w / width, h / height
            for index, (companion_area, companion_image) in enumerate(required_templates):
                if match_hsv is not None:
                    companion_image = hsv_mask(companion_image, match_hsv)
                    if companion_image.std() < 2:
                        raise ValueError("组合颜色轮廓必须同时包含目标和背景")
                ch, cw = companion_image.shape[:2]
                gw, gh = max(8, round(cw * sx)), max(8, round(ch * sy))
                guard = Button(area=(0, 0, gw, gh), color=(0, 0, 0),
                               button=(0, 0, gw, gh), name=f"{name}@{scale}:part{index}")
                guard.image = cv2.resize(companion_image, (gw, gh), interpolation=cv2.INTER_LINEAR)
                guard._match_init = True
                relative_x = round((companion_area[0] - self.source_area[0]) * sx)
                relative_y = round((companion_area[1] - self.source_area[1]) * sy)
                companions.append((guard, relative_x, relative_y, gw, gh, max(1, round(required_tolerance * scale))))
            self.variants.append((candidate, offset, companions))
        if not self.variants:
            raise ValueError("搜索区域小于目标图片")

    def match(self, image, offset=0, similarity=0.92):
        original = image
        if self.match_hsv is not None:
            image = hsv_mask(image, self.match_hsv)
        best = None
        best_score = -1.0
        frame_height, frame_width = image.shape[:2]
        for candidate, search_offset, companions in self.variants:
            candidate.match(image, offset=search_offset, similarity=similarity)
            score = candidate.last_similarity
            if math.isfinite(score) and score > similarity:
                x, y = candidate.button[:2]
                for guard, dx, dy, gw, gh, tolerance in companions:
                    left, top = max(0, x + dx - tolerance), max(0, y + dy - tolerance)
                    right = min(frame_width - gw, x + dx + tolerance)
                    bottom = min(frame_height - gh, y + dy + tolerance)
                    if left > right or top > bottom:
                        score = -1.0
                        break
                    guard.match(image, offset=(left, top, right, bottom), similarity=similarity)
                    guard_score = guard.last_similarity
                    score = min(score, guard_score) if math.isfinite(guard_score) else -1.0
                    if score <= similarity:
                        break
            if math.isfinite(score) and (best is None or score > best_score):
                best = candidate
                best_score = score
        self.last_similarity = best_score
        if best is not None:
            self.matched_box = tuple(best.button)
            self._button_offset = best.button
            if self.color_condition is not None and self.last_similarity > similarity:
                condition = self.color_condition
                x1, y1, x2, y2 = self.source_to_screen(condition['source_area'])
                if not (0 <= x1 < x2 <= original.shape[1] and 0 <= y1 < y2 <= original.shape[0]):
                    self.last_similarity = -1.0
                    return False
                mask = hsv_mask(original[y1:y2, x1:x2], condition['hsv'])
                if float((mask[:, :, 0] > 0).mean()) < condition['minimum_fraction']:
                    self.last_similarity = -1.0
                    return False
            if self.click_source_area is not None:
                box = self.source_to_screen(self.click_source_area)
                height, width = image.shape[:2]
                if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                    return False
                self._button_offset = box
        return self.last_similarity > similarity

    def source_to_screen(self, area):
        if self.matched_box is None:
            raise ValueError("尚未定位目标图案")
        left, top, right, bottom = self.source_area
        x1, y1, x2, y2 = self.matched_box
        sx, sy = (x2 - x1) / (right - left), (y2 - y1) / (bottom - top)
        return (round(x1 + (area[0] - left) * sx), round(y1 + (area[1] - top) * sy),
                round(x1 + (area[2] - left) * sx), round(y1 + (area[3] - top) * sy))
