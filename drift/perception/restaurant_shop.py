"""Cooking shop perception for the four talent-point ingredient slots."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


MATERIALS = ('柔软食材', '坚硬食材', '蔬菜', '调味料')
DAILY_LIMIT = 5
UNIT_PRICE = 10


def rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def tile(slot):
    # The matching premium-currency cells are the intervening columns.
    return (116, 365)[slot % 2], (254, 419)[slot // 2]


@dataclass(frozen=True)
class IngredientOffer:
    slot: int
    sold: int
    point: tuple[int, int]

@dataclass(frozen=True)
class IngredientQuote:
    count: int
    price: int
    confirm: tuple[int, int]
    maximum: tuple[int, int] | None


class IngredientShopVisual:
    def __init__(self, root, reader):
        self.folder = Path(root) / 'assets/restaurant-shop'
        self.images = {p.stem: rgb(p) for p in self.folder.glob('*.png')}
        self.reader = reader
        self.last_frame = None
        self.last_quotes = {}

    def find(self, frame, name, box, threshold=.94, *, bright=False):
        image = self.images.get(name)
        if image is None or frame.shape != (1280, 720, 3):
            return None
        x, y, far, bottom = box
        patch = frame[y:bottom, x:far]
        h, w = image.shape[:2]
        if patch.shape[0] < h or patch.shape[1] < w:
            return None
        scores = cv2.matchTemplate(patch, image, cv2.TM_CCOEFF_NORMED)
        _, score, _, (u, v) = cv2.minMaxLoc(np.nan_to_num(scores, nan=-1., posinf=-1., neginf=-1.))
        if score < threshold:
            return None
        if bright and abs(float(patch[v:v+h, u:u+w].mean()) - float(image.mean())) > 20:
            return None
        return x+u+w//2, y+v+h//2

    def tab(self, frame):
        area=(440, 75, 543, 164)
        return self.find(frame,'tab',area,.94) or self.find(frame,'tab-inactive',area,.94)

    def panel(self, frame):
        # These controls identify the ingredient shop even if all eight offers
        # are sold out. No currency is inferred from the balance above the grid.
        return bool(self.tab(frame)
                    and self.find(frame,'refresh',(212,613,379,647),bright=True)
                    and self.find(frame,'help',(587,608,641,650),bright=True))

    def _glyph(self, mask):
        def normalized(image):
            return cv2.GaussianBlur(cv2.resize(image.astype('float32'),(24,32)),(5,5),1.4)

        glyph = normalized(mask)
        scores = []
        for value in range(DAILY_LIMIT+1):
            image = self.images.get(f'count-{value}')
            if image is None:
                continue
            ref = normalized(image[:, :, 0] > 127)
            score = float(cv2.matchTemplate(glyph, ref, cv2.TM_CCOEFF_NORMED)[0,0])
            scores.append((score, value))
        scores.sort(reverse=True)
        if not scores or scores[0][0] < .86 or len(scores)>1 and scores[0][0]-scores[1][0] < .12:
            return None
        return scores[0][1]

    def stock(self, frame, slot):
        x, y = tile(slot)
        # Read the count only from the matched /5 row. Each whole digit must
        # agree at two ink thresholds, including on a dimmed 5/5 card.
        if not self.find(frame,f'limit-{slot}',(x+72,y+85,x+109,y+113), .92):
            return None
        gray = frame[y+88:y+108, x+57:x+75].min(axis=2)
        peak = int(gray.max())
        if peak < 110:
            return None
        values = []
        for factor in (.83, .89):
            mask = gray > int(peak*factor)
            points = np.argwhere(mask)
            if not len(points):
                return None
            (top,left), (bottom,right) = points.min(axis=0), points.max(axis=0)
            if not 13 <= bottom-top+1 <= 19 or not 4 <= right-left+1 <= 14:
                return None
            values.append(self._glyph(mask[top:bottom+1,left:right+1]))
        return values[0] if values[0] == values[1] else None

    def offer(self, frame, slot):
        if not self.panel(frame):
            return None
        x, y = tile(slot)
        # Verify the material art and the purple coin plus its actual unit price
        # together. The cheaper blue gem cells never qualify as these offers.
        if not (self.find(frame,f'item-{slot}',(x+9,y+18,x+109,y+92), bright=True)
                and self.find(frame,f'talent-price-{slot}',(x+11,y+115,x+108,y+157),bright=True)):
            return None
        sold = self.stock(frame,slot)
        if sold is None or sold >= DAILY_LIMIT:
            return None
        return IngredientOffer(slot,sold,(x+56,y+137))

    def dialog(self, frame):
        return bool(self.tab(frame)
                    and self.find(frame,'quote-x',(551,312,638,391),bright=True)
                    and self.find(frame,'quote-title',(115,637,405,684),bright=True)
                    and self.find(frame,'quote-border',(106,344,168,393),bright=True))

    def cancel(self, frame, slot):
        if not self.dialog(frame) or not self.find(frame,f'quote-item-{slot}',(268,400,454,581),bright=True):
            return None
        return self.find(frame,'quote-x',(551,312,638,391),bright=True)

    @staticmethod
    def _ink(rgb, threshold):
        mask = np.where(rgb.max(axis=2)<threshold,0,255).astype('uint8')
        return np.repeat(mask[:,:,None],3,axis=2)

    def quote(self, frame, slot):
        if frame is not self.last_frame:
            self.last_frame, self.last_quotes = frame, {}
        if slot not in self.last_quotes:
            self.last_quotes[slot] = self._read_quote(frame,slot)
        return self.last_quotes[slot]

    def _read_quote(self, frame, slot):
        if self.cancel(frame,slot) is None:
            return None
        title = self.find(frame,'quote-title',(115,637,405,684),bright=True)
        coin = self.find(frame,'quote-currency',(275,831,362,911),bright=True)
        if (title is None or coin is None
                or not self.find(frame,'quote-frame',(113,819,164,867),bright=True)):
            return None
        x, _ = title
        cx, _ = coin
        number = frame[643:679, x+74:x+98]
        price = frame[851:899, cx+47:cx+105]
        readings = self.reader.read_many([self._ink(part,t) for part in (number,price) for t in (90,125)])
        if len(readings)!=4 or any(read.value is None or read.confidence<.96 for read in readings):
            return None
        amount, repeat, total, repeated_total = (r.value for r in readings)
        if amount!=repeat or total!=repeated_total or not 1<=amount<=DAILY_LIMIT or total!=UNIT_PRICE*amount:
            return None
        maximum=(465,759) if self.find(frame,'quote-plus',(478,737,565,786),bright=True) else None
        return IngredientQuote(amount,total,(360,874),maximum)
