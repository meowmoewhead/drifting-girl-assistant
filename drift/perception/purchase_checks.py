"""Numeric guards for limited guild-point purchases.

SPDX-License-Identifier: GPL-3.0-only

路径语义:``QuantityTemplates`` 的模板根由调用方以 ``root`` 传入(源码运行是项目根,
frozen 运行是 exe 所在目录),本模块内部不做 ``__file__`` / ``sys._MEIPASS`` 推导。
"""
from decimal import Decimal
from pathlib import Path
import re
import cv2
import numpy as np
from PIL import Image
from drift.perception.digit_ocr import NumberReading, CounterReading


def quantity_outline(rgb):
    ink = (rgb.max(axis=2) < 90).astype(np.uint8)*255
    y,x = np.where(ink)
    if len(x) < 12:
        return None
    return ink[max(0,y.min()-2):min(ink.shape[0],y.max()+3),max(0,x.min()-2):min(ink.shape[1],x.max()+3)]


class QuantityTemplates:
    """Verify stylized counts against calibrated native glyph outlines."""
    def __init__(self, root, prefix):
        self.templates = []
        for path in (Path(root)/'assets'/'guild-shop').glob(prefix+'-*.png'):
            match = re.fullmatch(prefix+r'-(\d{3})',path.stem)
            if match:
                frame = np.asarray(Image.open(path).convert('RGB'))
                outline = quantity_outline(frame[655:724,272:450])
                if outline is not None:
                    self.templates.append((int(match.group(1)),outline))

    def read(self, rgb):
        outline = quantity_outline(rgb)
        scores = []
        if outline is not None:
            for value, reference in self.templates:
                if any(abs(a-b) > max(2,b*.08) for a,b in zip(outline.shape,reference.shape)):
                    continue
                image = cv2.resize(outline,(reference.shape[1],reference.shape[0]),interpolation=cv2.INTER_NEAREST)
                score = float(cv2.matchTemplate(image,reference,cv2.TM_CCOEFF_NORMED)[0,0])
                scores.append((score,value))
        scores.sort(reverse=True)
        if scores and scores[0][0] >= .985 and (len(scores)==1 or scores[0][0]-scores[1][0]>=.05):
            score,value=scores[0]
            return NumberReading(str(value),value,min(score,1.0))
        return NumberReading('',None,0.0)


def quantity_crop(rgb, mode='tight'):
    """Trim changing-width outlined digits; isolate purple talent-point glyphs."""
    if mode == 'tight':
        ink = rgb.max(axis=2) < 90
        image = rgb
    elif mode == 'purple_text':
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        ink = cv2.inRange(hsv, np.array([110, 50, 150]), np.array([155, 255, 255])) > 0
        image = np.repeat(np.where(ink, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    else:
        raise ValueError('未知的购买数量预处理方式')
    y, x = np.where(ink)
    if len(x) < 8:
        return np.full((8, 8, 3), 255, dtype=np.uint8)
    return image[max(0, y.min()-2):min(image.shape[0], y.max()+3),
                 max(0, x.min()-2):min(image.shape[1], x.max()+3)]


def guild_price_crop(rgb):
    # The guild-coin icon and numeric price re-center as the price gains digits.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    blue = cv2.inRange(hsv, np.array([85, 60, 90]), np.array([125, 255, 255])) > 0
    _, x = np.where(blue)
    if len(x) < 20 or x.max()+12 >= rgb.shape[1]:
        return np.full((8, 8, 3), 255, dtype=np.uint8)
    return quantity_crop(rgb[:, x.max()+12:], 'tight')


def read_guild_balance(reader, rgb, confidence=.93):
    primary = reader.read(rgb)
    if primary.confidence >= confidence or not re.fullmatch(r'[0-9]{1,7}', primary.text.strip()):
        return primary
    # A short right-aligned balance moves part of its blue guild coin into
    # the saved numeric crop. Remove only that coloured prefix, then require
    # two independently thresholded readings to agree with the original text.
    # Unknown currency, conflicting digits or low confidence remain rejected.
    clean = guild_price_crop(rgb)
    if clean.shape[:2] == (8, 8):
        return primary
    gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
    readings = []
    for threshold in (60, 90):
        mask = np.repeat(np.where(gray < threshold, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
        reading = reader.read(quantity_crop(mask))
        if (reading.text.strip() != primary.text.strip() or reading.value != primary.value
                or reading.confidence < confidence):
            return primary
        readings.append(reading)
    return NumberReading(primary.text, primary.value, min(r.confidence for r in readings))


def stock_counter_crop(rgb):
    # Sold-out cards dim their background and glyphs without changing the count.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return quantity_crop(np.repeat(mask[:, :, None], 3, axis=2))


def read_stock_counter(reader, rgb, confidence=.93):
    primary = reader.read_counter(stock_counter_crop(rgb))
    if primary.current is not None and primary.confidence >= confidence:
        return primary
    # On the dimmed 05/05 card the slash touches an antialiased digit. Require
    # agreement between two ink thresholds, each above the normal confidence.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    alternatives = []
    for threshold in (30, 60):
        mask = np.repeat(np.where(gray < threshold, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
        reading = reader.read_counter(quantity_crop(mask))
        if reading.current is not None and reading.confidence >= confidence:
            alternatives.append(reading)
    if (len(alternatives) == 2
            and (alternatives[0].current, alternatives[0].total) == (alternatives[1].current, alternatives[1].total)):
        first = alternatives[0]
        return CounterReading(first.text, first.current, first.total, min(r.confidence for r in alternatives))
    return primary


def confirmation_price_crop(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ink = cv2.inRange(hsv, np.array([85, 100, 120]), np.array([115, 255, 255]))
    # Reject clipped fragments of the adjacent currency icon.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    clean = np.zeros(ink.shape, dtype=np.uint8)
    for component in range(1, count):
        _, _, width, height, area = stats[component]
        if width >= 5 and height >= 8 and area >= 20:
            clean[labels == component] = 255
    return quantity_crop(np.repeat((255-clean)[:, :, None], 3, axis=2))


def read_moving_confirmation_price(reader, rgb, currency):
    """Find the blue guild coin on the centered question, then read its price.

    Title/button recognition is separate. A missing/wrong coin cannot authorize
    a purchase merely because the number equals an earlier selection.
    """
    currencies=currency if isinstance(currency,tuple) else (currency,)
    best=None
    for template in currencies:
        h,w=template.shape[:2]
        if rgb.shape[0]<h or rgb.shape[1]<w:continue
        scores=cv2.matchTemplate(rgb,template,cv2.TM_CCOEFF_NORMED)
        _,score,_,(x,y)=cv2.minMaxLoc(np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1))
        if score>=.94 and (best is None or score>best[0]):
            best=score,x,y,h,w
    if best is None:return None
    _,x,y,h,w=best
    # The reference is the coin's fixed inner core; its blue right rim extends
    # beyond it. Keep that rim out of the OCR crop just as in the verified
    # three-digit price crop, translated by the newly located core.
    left,right=x+w+21,min(rgb.shape[1],x+w+80)
    top,bottom=max(0,y-5),min(rgb.shape[0],y+h+7)
    if right-left<50:return None
    return reader.read(confirmation_price_crop(rgb[top:bottom,left:right]))


def currency_floor(reading, confidence):
    """Read a balance, rounding abbreviated thousands down conservatively."""
    if reading is None or reading.confidence < confidence:
        return None
    text = reading.text.strip()
    if re.fullmatch(r'[0-9]{1,7}', text):
        return int(text)
    match = re.fullmatch(r'([0-9]{1,4})(?:\.([0-9]{1,2}))?[kK]', text)
    if match:
        digits = match.group(2) or ''
        displayed = Decimal(match.group(1) + ('.'+digits if digits else ''))
        quantum = Decimal(10) ** -len(digits)
        return max(0, int((displayed - quantum) * 1000))
    return None


def stock_check(counter, balance, condition):
    confidence = condition.get('confidence', .93)
    funds = currency_floor(balance, confidence)
    # 限购读清楚就已经能判定"买满了没有";余额只在还能买的时候才需要
    # (三位数余额在低对比度下置信度会掉到 0.7 左右,不该因此卡住整条流程)。
    counted = (counter is not None and counter.current is not None
               and counter.total == condition['expected_total']
               and counter.confidence >= confidence)
    remaining = counter.total - counter.current if counted else None
    sold_out = counted and remaining == 0
    valid = bool(counted and (sold_out or funds is not None))
    target = min(remaining, funds // condition['unit_price']) if valid and not sold_out else (0 if valid else None)
    passed = valid and ((target > 0) if condition['mode'] == 'available' else target == 0)
    reason = ('未读清限购数量或公会点数' if not valid else
              '该商品已买满' if remaining == 0 else '公会点数不足' if target == 0 else None)
    result = dict(kind='purchase_stock', label=condition['label'], value=counter.current if counter else None,
                  limit=condition['expected_total'], remaining=remaining, target=target,
                  balance=funds, balance_text=balance.text if balance else '',
                  unit_price=condition['unit_price'], valid=bool(valid), passed=bool(passed), skip_reason=reason)
    return bool(passed), (result['value'], result['limit'], funds, target), result


def quantity_check(quantity, price, stock, condition):
    confidence = condition.get('confidence', .93)
    unit_price = condition['unit_price']
    bundle = condition.get('bundle_size', 1)
    target = stock.get('target') if stock and stock.get('valid') and stock.get('unit_price') == unit_price else None
    count = quantity.value if quantity else None
    cost = price.value if price else None
    # A stylized single digit can be ambiguous. Incrementing may use the verified
    # total price to count packs; buying still requires both independent readings.
    quantity_clear = count is not None and quantity.confidence >= confidence
    inferred = False
    price_valid = (target is not None and 1 <= target <= condition['maximum']
                   and cost is not None and price.confidence >= confidence
                   and cost > 0 and cost % unit_price == 0
                   and cost <= stock['balance'])
    if condition['mode'] == 'increment' and not quantity_clear and price_valid:
        count = (cost // unit_price) * bundle
        inferred = True
    valid = (target is not None and 1 <= target <= condition['maximum']
             and count is not None and cost is not None
             and (quantity_clear or inferred) and price_valid
             and count % bundle == 0 and 1 <= count // bundle <= target
             and cost == (count // bundle) * unit_price
             and cost <= stock['balance'])
    maximum = target * bundle if target is not None else None
    passed = valid and ((count < maximum) if condition['mode'] == 'increment' else count == maximum)
    result = dict(kind='purchase_quantity', label=condition['label'], value=count,
                  limit=maximum, price=cost, quantity_source='price' if inferred else 'quantity', valid=bool(valid), passed=bool(passed),
                  skip_reason=None if passed else '数量或总价未匹配购买条件')
    return bool(passed), (count, cost, maximum), result


def confirmation_check(price, selection, condition):
    """Accept only the amount from this run's verified maximum-quantity selection."""
    cost = price.value if price else None
    expected = selection.get('price') if selection else None
    valid_selection = (selection is not None and selection.get('kind') == 'purchase_quantity'
                       and selection.get('valid') and selection.get('passed')
                       and selection.get('value') == selection.get('limit')
                       and isinstance(selection.get('value'), int) and selection['value'] > 0
                       and isinstance(expected, int) and expected > 0)
    passed = (valid_selection and cost == expected and price is not None
              and price.confidence >= condition.get('confidence', .93))
    result = dict(kind='purchase_confirmation', label=condition['label'], value=cost,
                  limit=expected, valid=bool(passed), passed=bool(passed),
                  skip_reason=None if passed else '确认金额与本次已核对的购买总价不符或读数不清')
    return bool(passed), (cost, expected), result


