"""数值、区域、颜色与图片编码等无状态工具。

原先是 app.py 顶部的散装函数,现已归位到框架层;`app` 仍按原名再导出。
"""

from __future__ import annotations

import io
import math

import cv2
import numpy as np
from PIL import Image


def png_bytes(image):
    """把 RGB ndarray 编码成 PNG 字节。"""
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def number(value, low, high):
    """校验有限浮点数并落在 [low, high] 内。"""
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"数值必须介于 {low} 和 {high} 之间")
    return value


def rectangle(value, width, height):
    """校验 [左, 上, 右, 下] 区域并返回整数四元组。"""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("区域必须是 [左, 上, 右, 下]")
    x1, y1, x2, y2 = (int(v) for v in value)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("区域超出截图边界")
    return x1, y1, x2, y2


def hsv_mask(image, bounds):
    """按 HSV 上下界提取二值轮廓,返回三通道图像供模板匹配。"""
    mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_RGB2HSV),
                       np.array(bounds[0], dtype=np.uint8), np.array(bounds[1], dtype=np.uint8))
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
