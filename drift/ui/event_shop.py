"""Visual checks for event-material exchanges and the November toy shop."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from alas_core.button import Button
from drift.perception.digit_ocr import read_numeric_views
from drift.perception.purchase_checks import quantity_crop


PREFIX = 'driftinggirl_event_potion_'
OPEN, SELECT, CONFIRM, CLOSE = (PREFIX + name for name in ('open', 'select', 'confirm', 'close'))
SKIN_PRICE, POTION_PRICE = 2000, 500
TOY_MAIN = 'assets/event-potion/main-toys.png'
TOY_SHOP = 'assets/event-potion/shop-toys.png'
TOY_DETAIL = 'assets/event-potion/toy-skin-detail.png'
TOY_CLOSE = (581, 293, 653, 356)
DETAIL_CLOSE = (533, 315, 608, 374)
DETAIL_BUY = (142, 856, 580, 945)
POTION_BUY = (95, 761, 321, 821)
TOY_HELP = (554, 427, 599, 471)
TOY_CURRENCY = (118, 430, 154, 465)
TOY_PRICE_COIN = (131, 622, 158, 646)
TOY_CAP = (185, 441, 249, 460)
TOY_CAPACITY = 2200
TOY_ENTRY_CROP = (650, 280, 701, 320)


@dataclass(frozen=True)
class EventAction:
    kind: str
    layout: str
    price: int | None = None
    balance: int | None = None
    reason: str = ''

    def confirms(self, selection):
        return (self.kind == 'confirm' and selection is not None
                and selection.kind == 'select' and self.layout == selection.layout
                and self.price == selection.price)


def _rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def _crop(frame, box):
    x, y, right, bottom = box
    return frame[y:bottom, x:right]


def _position(frame, reference, threshold=.95):
    if any(size < needed for size, needed in zip(frame.shape[:2], reference.shape[:2])):
        return None
    score = cv2.matchTemplate(frame, reference, cv2.TM_CCOEFF_NORMED)
    _, best, _, (x, y) = cv2.minMaxLoc(score)
    if best < threshold:
        return None
    height, width = reference.shape[:2]
    if abs(float(frame[y:y+height, x:x+width].mean()) - float(reference.mean())) > 17:
        return None
    return x, y


def _matches(frame, reference, box, threshold=.95):
    return _position(_crop(frame, box), _crop(reference, box), threshold) == (0, 0)


def _owned_badges(frame, row_top, left=235, right=578):
    # Purchased cards have a green check over a disabled, dark card.
    # An affordable or unaffordable price by itself says nothing about ownership.
    row = frame[row_top:row_top+85, left:right]
    hsv = cv2.cvtColor(row, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv, (32, 118, 145), (86, 255, 255))
    _, _, stats, _ = cv2.connectedComponentsWithStats(green)
    return sum(36 <= w <= 98 and 22 <= h <= 65 and 650 <= area <= 2900
               for x, y, w, h, area in stats[1:])


def toy_entry_number(reader, frame, button, number_box):
    icon = getattr(button, 'toy_event_icon', None)
    if icon is None or _position(_crop(frame, button.matched_box), icon) is None:
        return reader.read(_crop(frame, number_box))
    # The old full-width OCR picks up texture around short numbers on the
    # toy label. Read only the complete dark glyphs at two thresholds.
    gray = cv2.cvtColor(_crop(frame, number_box)[2:-1, 3:-3], cv2.COLOR_RGB2GRAY)
    votes = []
    for cutoff in (70, 86):
        binary = np.repeat(np.where(gray < cutoff, 0, 255).astype('uint8')[..., None], 3, axis=2)
        votes.append(reader.read(quantity_crop(binary)))
    if (votes[0].value == votes[1].value and all(v.confidence >= .93 and v.text.isdecimal() for v in votes)):
        return min(votes, key=lambda reading: reading.confidence)
    from drift.perception.digit_ocr import NumberReading
    return NumberReading('', None, 0.0)


def toy_entry_icon(root):
    return _crop(_rgb(Path(root) / TOY_MAIN), TOY_ENTRY_CROP).copy()


class EventShopVisual:
    def __init__(self, root, reader):
        self.reader = reader
        self.toys = _rgb(Path(root) / TOY_SHOP)
        self.toy_cap = _crop(self.toys, TOY_CAP)
        self.toy_cap_soft = cv2.GaussianBlur(self.toy_cap, (5, 5), 1.2)
        self.confirm = _rgb(Path(root) / 'assets/event-potion/confirm.png')
        self.detail = _rgb(Path(root) / TOY_DETAIL)

    def toy_shop(self, frame):
        return (frame.shape == (1280, 720, 3) and
                all(_matches(frame, self.toys, area) for area in (TOY_CLOSE, TOY_HELP, TOY_CURRENCY)))

    def read_int(self, image):
        reader = self.reader()
        if reader is None or min(image.shape[:2]) < 7:
            return None
        reading = reader.read(image)
        if reading.confidence >= .93 and reading.text.isdecimal():
            return int(reading.text)
        return None

    def balance(self, frame):
        pos = _position(frame[441:460,180:351],self.toy_cap)
        if pos is not None:
            edge = 180 + pos[0] - 4
            amount = self.read_int(frame[435:461,159:edge])
            return amount if amount is not None and 0 <= amount <= TOY_CAPACITY else None
        # Proportional digits shift the cap's glyphs by a fraction of a pixel.
        # Locate the same '/ 2200' in the smoothed crop, then verify the cap
        # and two uncropped balance views on the original pixels.
        region = cv2.GaussianBlur(frame[441:460,176:351],(5,5),1.2)
        pos = _position(region,self.toy_cap_soft)
        reader = self.reader()
        if pos is None or reader is None:
            return None
        x = 176 + pos[0]
        views = (frame[435:461,159:x-4],frame[431:460,160:x-3],
                 frame[442:460,x+17:x+64])
        if any(min(view.shape[:2]) < 7 for view in views):
            return None
        first,second,cap = read_numeric_views(reader,views)
        if (any(r.confidence < .93 or not r.text.isdecimal() for r in (first,second,cap))
                or cap.value != TOY_CAPACITY or first.value != second.value
                or first.value is None or not 0 <= first.value <= TOY_CAPACITY):
            return None
        return first.value

    def first_toy_offer(self, frame):
        """Freshly opened row starts with unowned skins, then potion, then owned cards."""
        balance = self.balance(frame)
        if balance is None:
            return EventAction('skip', 'toys', reason='活动材料读数不清'), TOY_CLOSE
        first_price = self.read_int(frame[622:649, 161:216])
        coin = _matches(frame, self.toys, TOY_PRICE_COIN, .94)
        if first_price == SKIN_PRICE and coin and not _owned_badges(frame, 520, 118, 228):
            if balance < SKIN_PRICE:
                return EventAction('skip', 'toys', balance=balance, reason='未拥有皮肤，等待 2000 活动材料'), TOY_CLOSE
            return EventAction('select', 'toys', SKIN_PRICE, balance), (137, 520, 215, 608)
        if first_price == POTION_PRICE and _owned_badges(frame, 530) == 3:
            coin = _position(frame[617:650, 126:171], _crop(self.toys, TOY_PRICE_COIN), .94)
            if coin is not None and balance >= POTION_PRICE:
                return EventAction('select', 'toys', POTION_PRICE, balance), (137, 520, 215, 608)
        return EventAction('skip', 'toys', balance=balance, reason='未能确认皮肤已收齐及药水价格'), TOY_CLOSE

    def skin_detail(self, frame):
        # Currency and 2000 glyphs are checked together, independently of the
        # particular skin name/thumbnail. The gold bundle is a different page.
        return all(_matches(frame, self.detail, area) for area in
                   (DETAIL_CLOSE, (390, 881, 532, 930), (193, 880, 274, 929),
                    (143, 850, 189, 875)))

    def toy_potion_price(self, frame):
        if not all(_matches(frame, self.confirm, area) for area in
                   ((163, 768, 260, 817), (267, 574, 513, 608))):
            return False
        lane = frame[567:609, 227:267]
        coin = _crop(self.toys, TOY_PRICE_COIN)
        return any(_position(lane, cv2.resize(coin, (width, height)), .94) is not None
                   for width, height in ((27,24), (28,26), (30,28)))


class EventShopButton(Button):
    def __init__(self, original, visual, rule_id):
        super().__init__(area=original.area, color=(0, 0, 0), button=original.button,
                         name=original.name)
        self.original = original
        self.variants = original.variants
        self.visual = visual
        self.rule_id = rule_id
        self.event_action = self.target_key = None

    @property
    def matched_box(self):
        return self.original.matched_box

    def source_to_screen(self, area):
        return self.original.source_to_screen(area)

    def _choose(self, action, box, score):
        self.event_action = action
        self._button_offset = tuple(box)
        self.target_key = action, self._button_offset
        self.last_similarity = score
        return True

    def match(self, image, offset=0, similarity=.94):
        self.event_action = self.target_key = self._button_offset = None
        self.last_similarity = -1.0
        if self.rule_id in (SELECT, CLOSE) and self.visual.toy_shop(image):
            if self.rule_id == SELECT:
                action, box = self.visual.first_toy_offer(image)
                return self._choose(action, box, .95)
            return self._choose(EventAction('close', 'toys', balance=self.visual.balance(image)), TOY_CLOSE, .95)
        if self.rule_id == CONFIRM:
            if self.visual.skin_detail(image):
                return self._choose(EventAction('confirm', 'toys', SKIN_PRICE), DETAIL_BUY, .96)
            if self.visual.toy_potion_price(image):
                return self._choose(EventAction('confirm', 'toys', POTION_PRICE), POTION_BUY, .96)
        matched = self.original.match(image, offset=offset, similarity=similarity)
        self.last_similarity = self.original.last_similarity
        if not matched:
            return False
        if self.rule_id == SELECT and _owned_badges(image, 615) != 3:
            return False
        kind = {SELECT: 'select', CONFIRM: 'confirm', CLOSE: 'close'}[self.rule_id]
        return self._choose(EventAction(kind, 'legacy', None if kind == 'close' else POTION_PRICE),
                            self.original.button, self.last_similarity)


def adapt_event_shop(rule, button, root, reader):
    path = Path(root) / TOY_SHOP
    if (rule['id'] not in (SELECT, CONFIRM, CLOSE) or not path.is_file()
            or not (Path(root) / TOY_DETAIL).is_file()):
        return button
    known = {SELECT: ('assets/event-potion/shop.png', (143, 621, 206, 694)),
             CONFIRM: ('assets/event-potion/confirm.png', (267, 574, 513, 608)),
             CLOSE: ('assets/event-potion/shop.png', (557, 521, 597, 564))}
    if known[rule['id']] != (rule.get('asset'), tuple(rule.get('source_crop', ()))):
        return button
    if any(rule.get(field) for field in (
            'number_condition', 'counter_condition', 'purchase_condition',
            'quantity_condition', 'confirmation_condition')):
        return button
    return EventShopButton(button, EventShopVisual(root, reader), rule['id'])
