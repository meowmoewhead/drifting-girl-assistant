"""Reviewed source-sprite templates, with explicit per-controller adapters.

Only manifest-listed static icons use this path. Context, brightness, colour,
currency, OCR, fresh-frame and input guards stay in the owning controllers.
Coordinates refer to the old crop canvas, including transparent padding.
"""
from functools import wraps
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


class OriginalTemplates:
    def __init__(self, root, group):
        root = Path(root)
        folder = root / 'assets' / 'original-templates'
        self.entries = {}
        self.variants = {}
        manifest = folder / 'manifest.json'
        if not manifest.exists():
            return  # Existing installations without this optional asset set.
        data = json.loads(manifest.read_text(encoding='utf-8'))
        if data.get('schema') != 1:
            raise ValueError('Unsupported original-template manifest')
        for item in data['templates']:
            if item['group'] != group:
                continue
            name = item['name']
            # An updated local legacy template must be reviewed again. Do not
            # silently override a newer user's/maintainer's recognition asset.
            legacy = root / 'assets' / group / (name + '.png')
            if (not legacy.is_file() or
                    hashlib.sha256(legacy.read_bytes()).hexdigest() != item['legacy_sha256']):
                continue
            relative = Path(item['file'])
            if relative.is_absolute() or len(relative.parts) != 1:
                raise ValueError('Invalid original-template path')
            path = folder / relative
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != item['sha256']:
                raise ValueError('Original template integrity mismatch: ' + name)
            image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None or image.ndim != 3 or image.shape[2] != 4:
                raise ValueError('Original template requires RGBA: ' + name)
            old = cv2.imread(str(legacy))
            if old is None or old.shape[:2] != image.shape[:2]:
                raise ValueError('Original template coordinate canvas changed: ' + name)
            mask = np.where(image[:, :, 3] == 255, 255, 0).astype(np.uint8)
            if np.count_nonzero(mask) < 50:
                raise ValueError('Original template has insufficient foreground: ' + name)
            self.entries[name] = (cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB), mask)

    def has(self, name):
        return name in self.entries

    def find(self, frame, name, area, threshold, *, scales=(1.,), gray=False):
        if frame.shape != (1280, 720, 3) or frame.dtype != np.uint8:
            return None
        x, y, X, Y = map(int, area)
        if not (0 <= x < X <= 720 and 0 <= y < Y <= 1280):
            return None
        source = frame[y:Y, x:X]
        if gray:
            source = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        best = None
        for scale in scales:
            if not np.isfinite(scale) or not .5 <= scale <= 2:
                continue
            key = (name, float(scale), gray)
            if key not in self.variants:
                image, mask = self.entries[name]
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                if gray:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                self.variants[key] = image, mask
            template, mask = self.variants[key]
            h, w = template.shape[:2]
            if h > Y-y or w > X-x:
                continue
            scores = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            # OpenCV masked matching can produce inf/nan on uniform negatives.
            scores = np.nan_to_num(scores, nan=-1., posinf=-1., neginf=-1.)
            _, score, _, (px, py) = cv2.minMaxLoc(scores)
            if score >= threshold and (best is None or score > best[2]):
                best = ((x+px+w//2, y+py+h//2), (x+px, y+py, w, h), score)
        return best


def install_original_templates(cls, group, method):
    """Adapt one named helper; never alter cv2 or global template matching."""
    if getattr(cls, '_original_templates_adapter', None) == (group, method):
        return
    old_init = cls.__init__
    old_match = getattr(cls, method)

    @wraps(old_init)
    def initialize(self, root, *args, **kwargs):
        old_init(self, root, *args, **kwargs)
        self._source_sprite_templates = OriginalTemplates(root, group)

    if method == '_fixed':
        @wraps(old_match)
        def match(self, frame, name, box, threshold=.94):
            bank = getattr(self, '_source_sprite_templates', None)
            if bank is None or not bank.has(name):
                return old_match(self, frame, name, box, threshold)
            x, y, X, Y = box
            return bank.find(frame, name, (max(0,x-2), max(0,y-2), min(720,X+2), min(1280,Y+2)), threshold) is not None
    elif method == '_within':
        @wraps(old_match)
        def match(self, frame, name, box, threshold=.92, *, text=False):
            bank = getattr(self, '_source_sprite_templates', None)
            if text or bank is None or not bank.has(name):
                return old_match(self, frame, name, box, threshold, text=text)
            return bank.find(frame, name, box, threshold) is not None
    elif method == 'find':
        @wraps(old_match)
        def match(self, frame, name, area=None):
            bank = getattr(self, '_source_sprite_templates', None)
            if bank is None or not bank.has(name):
                return old_match(self, frame, name, area)
            if frame is not getattr(self, '_source_sprite_frame', None):
                self._source_sprite_frame, self._source_sprite_cache = frame, {}
            key = (name, tuple(area) if area else None)
            if key not in self._source_sprite_cache:
                spec = self.spec[name]
                self._source_sprite_cache[key] = bank.find(frame, name, area or spec['area'], spec['threshold'],
                                                          scales=spec.get('scales', [1.]), gray=True)
            return self._source_sprite_cache[key]
    else:
        raise ValueError('Unknown original-template adapter')
    cls.__init__, cls._original_templates_adapter = initialize, (group, method)
    setattr(cls, method, match)
