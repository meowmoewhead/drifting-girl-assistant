"""Fixed UI evidence for delivery orders, recipe books and cooking."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from drift.perception.digit_ocr import read_numeric_views


BOOK_NUMBERS = ((198, 745, 225, 780), (311, 745, 336, 781),
                (423, 745, 450, 781), (531, 745, 558, 781))
CRAFT_NUMBERS = ((532, 272, 561, 305), (533, 368, 560, 401),
                 (532, 463, 559, 496), (532, 556, 560, 590))
INGREDIENT_ROWS = (270, 364, 459, 553)
NPC_KINDS = ('cat', 'ranger', 'silver', 'octopus', 'hooded_cat')
BOOK_POINT = (425, 795)
BOOK_CLOSE = (597, 423)
CRAFT_POINT = (244, 559)
CRAFT_CLOSE = (637, 183)
CONFIRM_ONE = (353, 874)
QUANTITY_CLOSE = (596, 350)


def crop(frame, box):
    x, y, right, bottom = box
    return frame[y:bottom, x:right]


def shift(box, dy):
    x, y, right, bottom = box
    return x, y+dy, right, bottom+dy


def title_strokes(frame, top):
    gray = cv2.cvtColor(frame[top+4:top+53, 179:541], cv2.COLOR_RGB2GRAY)
    ink = (gray < 85).astype('uint8') * 255
    y, x = np.where(ink)
    if len(x) < 120 or x.min() < 3 or x.max() > ink.shape[1]-4:
        return None
    return ink[y.min():y.max()+1, x.min():x.max()+1]


def same_title(expected, actual):
    if actual is None or any(abs(a-b) > 3 for a, b in zip(expected.shape, actual.shape)):
        return False
    fixed = cv2.resize(actual, (expected.shape[1], expected.shape[0]), interpolation=cv2.INTER_NEAREST)
    scores = cv2.matchTemplate(np.pad(fixed, 2), expected, cv2.TM_CCOEFF_NORMED)
    return float(scores.max()) >= .84


@dataclass(frozen=True)
class OrderCard:
    npc: str | None
    top: int
    cooked: bool
    arrived: bool
    travel: bool
    can_finish: bool
    signature: str

    @property
    def book(self):
        return BOOK_POINT[0], BOOK_POINT[1] if self.top > 473 else 820

    @property
    def close(self):
        return 644, self.top-55


class MealDeliveryVisual:
    def __init__(self, root, reader):
        folder = Path(root) / 'assets/meal-delivery'
        self.images = {}
        for path in folder.glob('*.png'):
            with Image.open(path) as image:
                self.images[path.stem] = np.array(image.convert('RGB'))
        self.reader = reader
        self.frame = None
        self.cache = {}

    def _frame(self, frame):
        if self.frame is not frame:
            self.frame, self.cache = frame, {}

    def find(self, frame, name, area, *, threshold=.94, bright=True):
        self._frame(frame)
        key = name, area, threshold, bright
        if key in self.cache:
            return self.cache[key]
        self.cache[key] = None
        template = self.images.get(name)
        if template is None or frame.shape != (1280, 720, 3):
            return None
        x, y, _, _ = area
        source = crop(frame, area)
        height, width = template.shape[:2]
        if source.shape[0] < height or source.shape[1] < width:
            return None
        result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (dx, dy) = cv2.minMaxLoc(result)
        current = source[dy:dy+height, dx:dx+width]
        lit = abs(float(current.mean()) - float(template.mean())) < 24
        if score >= threshold and (not bright or lit):
            self.cache[key] = (x+dx+width//2, y+dy+height//2)
        return self.cache[key]

    def npc_points(self, frame):
        result = {}
        for slot, top in enumerate(range(247,795,110)):
            names = [name for name in NPC_KINDS if self.find(
                frame,name+'_roster',(12,top,102,top+111),bright=False)]
            if len(names)==1:
                result[f'{names[0]}:{slot}']=(50,top+64)
        return result

    def order(self, frame):
        from hashlib import sha256

        title = self.find(frame, 'title', (329, 449, 503, 530))
        if title is None:
            return None
        offset = title[1] - 501
        if not (self.find(frame, 'job_close', shift((604, 396, 686, 464), offset))
                and self.find(frame, 'reward', (156, 773, 299, 852))):
            return None
        npc = next((name for name in NPC_KINDS if self.find(
            frame, name+'_body', shift((102, 453, 264, 659), offset), threshold=.91)), None)
        speech = self.speech(frame, title[1]-16)
        signature = sha256(speech.tobytes()).hexdigest()
        cooked = self.find(frame, 'check', (559, 664, 613, 708)) is not None
        arrived = cooked and self.find(frame, 'arrive_check', (559, 711, 613, 757)) is not None
        travel = cooked and self.find(frame, 'route', (558, 708, 615, 760)) is not None
        can_finish = arrived and self.find(frame, 'finish_green', (469, 782, 625, 857)) is not None
        return OrderCard(npc, title[1]-16, cooked, arrived, travel, can_finish, signature)

    @staticmethod
    def speech(frame, title_top):
        sat = crop(frame, shift((335,538,607,636), title_top-485))
        return (cv2.cvtColor(sat,cv2.COLOR_RGB2GRAY)<105).astype('uint8')*255

    @staticmethod
    def book_button(frame, card):
        x,y=card.book
        roi = cv2.cvtColor(frame[y-29:y+31,x-27:x+31], cv2.COLOR_RGB2HSV)
        brown = cv2.countNonZero(cv2.inRange(roi, (6,95,78), (23,240,198)))
        fish = cv2.countNonZero(cv2.inRange(roi, (62,12,125), (122,107,225)))
        return card.book if 280<=brown<=1390 and 28<=fish<=330 else None

    def book_visible(self, frame):
        return bool(self.find(frame, 'book_close', (549, 379, 646, 464))
                    and self._ingredients(frame, 0))

    def _ingredients(self, frame, dy):
        areas = ((148, 705, 203, 756), (262, 706, 315, 754),
                 (374, 703, 423, 758), (488, 704, 538, 759))
        return all(self.find(frame, 'ingredient_'+str(index), shift(area, dy), threshold=.92)
                   for index, area in enumerate(areas))

    def cooking(self, frame):
        return bool(self.find(frame, 'craft_label', (143, 611, 527, 651))
                    and self.find(frame, 'craft_tab', (184, 80, 282, 162))
                    and self.find(frame, 'craft_button', (185, 537, 311, 586))
                    and self.find(frame, 'craft_close', (592, 146, 685, 220)))

    def one_portion(self, frame):
        return bool(self.find(frame, 'one', (278, 820, 435, 927))
                    and self.find(frame, 'quantity_plus', (479, 743, 568, 791))
                    and self._ingredients(frame, -55))

    def recipe(self, frame, page):
        self._frame(frame)
        key = 'numbers', page
        if key in self.cache:
            return self.cache[key]
        areas = {'book': BOOK_NUMBERS, 'cooking': CRAFT_NUMBERS,
                 'one': tuple(shift(box, -55) for box in BOOK_NUMBERS)}[page]
        values = tuple(self._number(crop(frame, area)) for area in areas)
        result = values if None not in values else None
        self.cache[key] = result
        return result

    @staticmethod
    def _digit_mask(rgb, cutoff):
        low, high = rgb.min(axis=2), rgb.max(axis=2)
        white = ((low >= cutoff) & (high-low < 30)).astype('uint8')
        _, labels, stats, _ = cv2.connectedComponentsWithStats(white)
        glyphs = [(i,x,y,w,h) for i,(x,y,w,h,area) in enumerate(stats[1:],1)
                  if 18 <= h <= 25 and 4 <= w <= 21 and area >= 65
                  and x > 0 and y > 0 and x+w < rgb.shape[1] and y+h < rgb.shape[0]]
        if not glyphs or len(glyphs) > 2:
            return None
        glyphs.sort(key=lambda glyph: glyph[1])
        # The overlay digits lie on the lower right of each ingredient tile.
        # Preserve a two-digit value if both separate strokes are visible.
        x, y = min(v[1] for v in glyphs), min(v[2] for v in glyphs)
        right, bottom = max(v[1]+v[3] for v in glyphs), max(v[2]+v[4] for v in glyphs)
        return np.isin(labels[y:bottom, x:right], [v[0] for v in glyphs]).astype('uint8')

    def _number(self, rgb):
        masks = [self._digit_mask(rgb, cutoff) for cutoff in (222, 233)]
        if any(mask is None for mask in masks):
            return None
        known = [(int(key[6:]), value[...,0] > 0) for key, value in self.images.items()
                 if key.startswith('digit_')]
        guesses = []
        for mask in masks:
            scores = []
            for number, target in known:
                if abs(mask.shape[1] / mask.shape[0] - target.shape[1] / target.shape[0]) > .11:
                    continue
                resized = cv2.resize(mask, (target.shape[1],target.shape[0]), interpolation=cv2.INTER_NEAREST)>0
                scores.append((np.count_nonzero(resized & target) / np.count_nonzero(resized | target), number))
            scores.sort(reverse=True)
            guesses.append(scores[0][1] if scores and scores[0][0] >= .88
                           and (len(scores)==1 or scores[0][0]-scores[1][0] >= .08) else None)
        if guesses[0] is not None and guesses[0] == guesses[1]:
            return guesses[0]
        cells = [np.repeat(np.pad(255 * (1-mask), 3, constant_values=255)[...,None], 3, axis=2)
                 for mask in masks]
        parsed = read_numeric_views(self.reader, cells)
        if (all(v.value is not None and v.confidence >= .93 and v.text.isdecimal() for v in parsed)
                and parsed[0].value == parsed[1].value
                and all(value in (None,parsed[0].value) for value in guesses)):
            return int(parsed[0].value)
        return None

    def arrow(self, frame, slot, increase):
        y = INGREDIENT_ROWS[slot]
        center = 594 if increase else 452
        name = 'right' if increase else 'left'
        area = center-18,y-21,center+20,y+21
        if any(self.find(frame, sample, area) for sample in
               (f'{name}_{slot}', name, f'{name}_0', f'{name}_3')):
            return center,y
        return None

    def other_quest_close(self, frame, npc=None):
        # The known generic NPC sheet after submission has its own gray
        # finish label. The return path only uses the red cross.
        if (not self.find(frame,'job_close',(604,396,686,464))
                or not self.find(frame,'finish_gray',(498,766,599,824))):
            return None
        if npc is not None and not self.find(frame,npc+'_body',(102,453,264,659),threshold=.91):
            return None
        return 644,429

    def new_order(self, frame):
        return self.find(frame,'rewarded_new',(49,395,173,474),threshold=.92) is not None

    def destination_locked(self, frame):
        return self.find(frame,'destination-locked',(240,241,478,287)) is not None

    def arrival_record(self, frame):
        return bool(self.find(frame,'arrival-record-title',(285,394,437,448))
                    and self.find(frame,'arrival-record-ok',(290,817,430,899)))

    def delivery_reward(self, frame):
        return (self.arrival_record(frame)
                and self.find(frame,'reward-food-flying',(56,68,257,363),threshold=.92)
                and self.find(frame,'reward-potion-flying',(70,894,255,1122),threshold=.92))

    def final_reward_burst(self, frame):
        return bool(self.find(frame, 'end-reward-sack', (225,440,515,760), threshold=.95)
                    and self.find(frame, 'end-reward-potion', (270,500,560,850), threshold=.95))
