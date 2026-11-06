"""Detect the large world-map overlay and decide when auto-close is safe.

Distinct from task-popup close (top-right red X on a notice dialog). The world
map uses the bottom-right close control plus the left map legend.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

RULE_ID = "driftinggirl_world_map_close"
CLOSE_XY = (668, 1187)
CLOSE_SEARCH = (632, 1151, 705, 1225)
CLOSE_THRESHOLD = 0.93
RED_Y1, RED_Y2, RED_X1, RED_X2 = 1170, 1210, 644, 686
RED_MIN_FRACTION = 0.30
LEGEND_Y1, LEGEND_Y2, LEGEND_X1, LEGEND_X2 = 1136, 1241, 9, 405
LEGEND_THRESHOLD = 0.93

# Controllers intentionally navigating the world map. Auto-close must not steal
# find_boss / first_map / backpack travel / restaurant origin map, etc.
_MAP_OWNER_MARKERS = (
    "map",
    "find_boss",
    "find_training",
    "find_observatory",
    "find_origin",
    "find_port",
    "quest_",
    "travel",
    "origin_select",
    "port_select",
    "first_select",
    "first_close",
    "first_map",
    "refresh_boss",
    "open_original",
    "open_gate",
    "close_boss",
    "close_training",
    "close_travel",
    "close_origin",
    "close_port",
    "boss_handoff",
)

_close_template = None
_legend_template = None


def _load_templates(root):
    global _close_template, _legend_template
    root = Path(root)
    if _close_template is None:
        with Image.open(root / "assets" / "parallel" / "map-close.png") as image:
            _close_template = np.asarray(image.convert("RGB"))
    if _legend_template is None:
        with Image.open(root / "assets" / "task-popup" / "map-legend.png") as image:
            _legend_template = np.asarray(image.convert("RGB"))
    return _close_template, _legend_template


def world_map_open(frame, root, *, require_legend=True):
    """True when the large world map shows a bright bottom-right close X.

    Optional legend match (default on) keeps this distinct from other red
    buttons and from the task-popup top-right close.
    """
    if frame is None or getattr(frame, "shape", None) is None:
        return False
    if tuple(frame.shape[:2]) != (1280, 720):
        return False
    close, legend = _load_templates(root)
    x1, y1, x2, y2 = CLOSE_SEARCH
    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] < close.shape[0] or roi.shape[1] < close.shape[1]:
        return False
    score = cv2.minMaxLoc(cv2.matchTemplate(roi, close, cv2.TM_CCOEFF_NORMED))[1]
    if not (np.isfinite(score) and score >= CLOSE_THRESHOLD):
        return False
    red = float((np.max(frame[RED_Y1:RED_Y2, RED_X1:RED_X2], axis=2) >= 210).mean())
    if red <= RED_MIN_FRACTION:
        return False
    if require_legend:
        patch = frame[LEGEND_Y1:LEGEND_Y2, LEGEND_X1:LEGEND_X2]
        if patch.shape[0] < legend.shape[0] or patch.shape[1] < legend.shape[1]:
            return False
        legend_score = cv2.minMaxLoc(
            cv2.matchTemplate(patch, legend, cv2.TM_CCOEFF_NORMED)
        )[1]
        if not (np.isfinite(legend_score) and legend_score >= LEGEND_THRESHOLD):
            return False
    return True


def map_navigation_owned(
    *,
    parallel=None,
    stone=None,
    backpack=None,
    restaurant=None,
    return_parallel=None,
):
    """True when a live controller is mid world-map navigation."""
    for controller in (parallel, stone, backpack, restaurant, return_parallel):
        if controller is None:
            continue
        phase = str(getattr(controller, "phase", "") or "")
        if phase and any(marker in phase for marker in _MAP_OWNER_MARKERS):
            return True
    return False
