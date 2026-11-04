"""Offline numeric OCR using the pinned PaddleOCR/RapidAI ONNX model.

SPDX-License-Identifier: GPL-3.0-only
Model input normalization and CTC layout follow the upstream recognition format.
See THIRD_PARTY_NOTICES.md and models/digit-model.json for source and licenses.

路径语义:本模块不做任何 ``__file__`` / ``sys._MEIPASS`` 推导,资源根一律由调用方
以 ``root`` 传入(源码运行是项目根,frozen 运行是 exe 所在目录,即
:data:`drift.core.paths.ROOT` 的两种取值)。因此源码与打包行为一致。
"""
from dataclasses import dataclass
from collections import OrderedDict
import math
from pathlib import Path
import re
import threading
import time
import unicodedata

import cv2
import numpy as np


@dataclass(frozen=True)
class NumberReading:
    text: str
    value: int | None
    confidence: float


@dataclass(frozen=True)
class CounterReading:
    text: str
    current: int | None
    total: int | None
    confidence: float
    reason: str | None = None


def parse_counter(text: str, confidence: float) -> CounterReading:
    text = unicodedata.normalize('NFKC', text).strip()
    match = re.fullmatch(r'([0-9]{1,3})\s*/\s*([0-9]{1,3})', text)
    if match is not None:
        current, total = (int(part) for part in match.groups())
        if 0 <= current <= total and total > 0:
            return CounterReading(text, current, total, confidence)
    return CounterReading(text, None, None, confidence)


def prepare_counter_crop(rgb: np.ndarray, mode: str = 'raw') -> np.ndarray:
    """Isolate white outlined counters from the colored reward icon behind them."""
    if mode == 'raw':
        return rgb
    if mode != 'white_text':
        raise ValueError('未知的每日计数预处理方式')
    # Keep the full fraction, including changing one/two-digit numerators.
    # Every channel must be bright: gold/brown parts of the potion are discarded.
    ink = np.all(rgb >= 200, axis=2)
    return np.repeat(np.where(ink, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)


def read_daily_counter(reader, rgb: np.ndarray, mode: str = 'raw',
                       minimum_confidence: float = .93) -> CounterReading:
    """Require agreement between distinct views of outlined advertisement text.

    Thresholding can break antialiased strokes differently for each numeral.
    Retain both the original view and two white-ink thresholds, with/without
    empty margins. Never choose a number just because one view is confident.
    """
    if mode != 'white_text_consensus':
        return reader.read_counter(prepare_counter_crop(rgb, mode))
    if rgb.ndim != 3 or min(rgb.shape[:2]) < 4:
        return CounterReading('', None, None, 0., '每日计数图像无效')
    variants, padded = [rgb], []
    for threshold in (190, 200):
        ink = np.all(rgb >= threshold, axis=2).astype(np.uint8)
        if not .01 < float(ink.mean()) < .70:
            continue
        binary = np.repeat(np.where(ink, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
        variants.append(binary)
        # A tightly fitted 20px counter can lose its recognition margin even
        # though every stroke is present. Keep the same ink and confidence;
        # only add empty paper around actual separated text components.
        count, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        if sum(stats[i,cv2.CC_STAT_AREA] >= 4 for i in range(1,count)) >= 3:
            padded.append(cv2.copyMakeBorder(binary,3,3,3,3,cv2.BORDER_CONSTANT,
                                             value=(255,255,255)))
        x, y, width, height = cv2.boundingRect(ink)
        if min(width, height) >= 4:
            variants.append(binary[y:y+height, x:x+width])
    readings, seen = [], set()
    for variant in variants:
        signature = (variant.shape, variant.tobytes())
        if signature in seen:
            continue
        seen.add(signature)
        readings.append(reader.read_counter(variant))
    votes = {}
    for reading in readings:
        if (reading.current is not None and reading.total is not None
                and reading.confidence >= minimum_confidence):
            votes.setdefault((reading.current, reading.total), []).append(reading)
    if len(votes) <= 1 and not any(len(group) >= 2 for group in votes.values()):
        # Never use a fallback to vote away an existing strong conflict.
        for variant in padded:
            signature = (variant.shape, variant.tobytes())
            if signature in seen:
                continue
            seen.add(signature)
            reading = reader.read_counter(variant)
            readings.append(reading)
            if (reading.current is not None and reading.total is not None
                    and reading.confidence >= minimum_confidence):
                votes.setdefault((reading.current,reading.total), []).append(reading)
    if len(votes) == 1:
        group = next(iter(votes.values()))
        if len(group) >= 2:
            return CounterReading(group[0].text, group[0].current, group[0].total,
                                  min(item.confidence for item in group))
    best = max(readings, key=lambda item: item.confidence)
    text = ' | '.join(dict.fromkeys(item.text for item in readings if item.text))
    reason = '每日计数识别结果不一致' if len(votes) > 1 else '每日计数未通过两路高置信度复核'
    return CounterReading(text or best.text, None, None, best.confidence, reason)


class DigitReader:
    def __init__(self, root: Path):
        import onnxruntime as ort

        model = Path(root) / 'models' / 'en_PP-OCRv4_rec_mobile.onnx'
        if not model.is_file():
            raise RuntimeError('缺少离线数字识别模型，请重新部署 models 目录')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        characters = self.session.get_modelmeta().custom_metadata_map.get('character')
        if not characters:
            raise RuntimeError('数字识别模型缺少字符表')
        self.characters = [''] + characters.splitlines() + [' ']
        self.input_name = self.session.get_inputs()[0].name
        self._frame = threading.local()
        self._statistics_lock = threading.Lock()
        self._statistics = dict(requests=0,model_calls=0,model_images=0,cache_hits=0,inference_ms=0.)

    def begin_frame(self):
        """Start an empty, thread-local recognition scope for a new capture."""
        self._frame.cache = OrderedDict()

    def end_frame(self):
        self._frame.cache = None

    def statistics(self):
        with self._statistics_lock:
            return dict(self._statistics)

    @staticmethod
    def _key(rgb):
        # Exact crop pixels, shape and type, never a similar-image/RANK cache.
        # Large images are not retained by this small numeric ROI cache.
        return (rgb.shape,rgb.dtype.str,rgb.tobytes()) if rgb.nbytes <= 65536 else None

    def read(self, rgb: np.ndarray) -> NumberReading:
        return self.read_many([rgb])[0]

    @staticmethod
    def _prepare(rgb):
        # Keep the entire number. A little border avoids cropping tight glyphs.
        padding = max(2, round(rgb.shape[0] * 0.15))
        background = tuple(int(v) for v in np.median(rgb.reshape(-1, 3), axis=0))
        padded = cv2.copyMakeBorder(rgb, padding, padding, padding, padding,
                                    cv2.BORDER_CONSTANT, value=background)
        bgr = cv2.cvtColor(padded, cv2.COLOR_RGB2BGR)
        resized_width = min(320, math.ceil(48 * bgr.shape[1] / bgr.shape[0]))
        resized = cv2.resize(bgr, (resized_width, 48), interpolation=cv2.INTER_LINEAR)
        tensor = np.zeros((1, 3, 48, 320), dtype=np.float32)
        tensor[0, :, :, :resized_width] = resized.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0
        return tensor[0]

    def _decode(self, probabilities):
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.characters):
            raise RuntimeError('数字识别模型的输出与字符表不一致')
        if not np.isfinite(probabilities).all():
            return NumberReading('', None, 0.0)
        indices = probabilities.argmax(axis=1)
        selected = (indices != 0)
        selected[1:] &= indices[1:] != indices[:-1]
        decoded = ''.join(self.characters[int(index)] for index in indices[selected])
        text = unicodedata.normalize('NFKC', decoded).strip()
        confidence = float(probabilities.max(axis=1)[selected].min()) if selected.any() else 0.0
        # Never turn an empty result, a letter, or a fraction into zero.
        value = int(text) if re.fullmatch(r'[0-9]{1,4}', text) else None
        return NumberReading(text, value, confidence)

    def read_many(self, images) -> list[NumberReading]:
        """Batch this frame's independent numeric ROIs, retaining all votes."""
        images = list(images)
        result = [None]*len(images)
        cache = getattr(self._frame,'cache',None)
        jobs = OrderedDict()
        hits = 0
        for index,rgb in enumerate(images):
            if rgb.ndim != 3 or min(rgb.shape[:2]) < 4:
                result[index] = NumberReading('',None,0.)
                continue
            key = self._key(rgb)
            if cache is not None and key is not None and key in cache:
                result[index] = cache[key]
                cache.move_to_end(key)
                hits += 1
                continue
            job_key = key if key is not None else index
            if job_key in jobs:
                jobs[job_key][2].append(index)
            else:
                jobs[job_key] = (key,rgb,[index])
        jobs = list(jobs.values())
        model_calls = model_images = 0
        elapsed = 0.
        for start in range(0,len(jobs),8):
            batch = jobs[start:start+8]
            tensor = np.stack([self._prepare(rgb) for _,rgb,_ in batch])
            began = time.perf_counter()
            probabilities = self.session.run(None,{self.input_name:tensor})[0]
            elapsed += (time.perf_counter()-began)*1000
            model_calls += 1;model_images += len(batch)
            if probabilities.ndim != 3 or probabilities.shape[0] != len(batch):
                raise RuntimeError('数字识别模型的批次输出无效')
            for (key,_,indices),prediction in zip(batch,probabilities):
                reading = self._decode(prediction)
                for index in indices:result[index] = reading
                if cache is not None and key is not None:
                    cache[key] = reading
                    while len(cache)>128:cache.popitem(last=False)
        with self._statistics_lock:
            self._statistics['requests'] += len(images)
            self._statistics['cache_hits'] += hits
            self._statistics['model_calls'] += model_calls
            self._statistics['model_images'] += model_images
            self._statistics['inference_ms'] += elapsed
        return result

    def read_counter(self, rgb: np.ndarray) -> CounterReading:
        reading = self.read(rgb)
        return parse_counter(reading.text, reading.confidence)


def read_numeric_views(reader, images):
    """Keep simple readers compatible while batching the production model."""
    batch = getattr(type(reader),'read_many',None)
    return batch(reader,images) if callable(batch) else [reader.read(image) for image in images]
