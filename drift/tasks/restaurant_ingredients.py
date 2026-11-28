"""Optional ingredient purchases inside a restaurant upgrade visit."""
from datetime import datetime, timezone

from drift.perception.restaurant_shop import IngredientShopVisual, MATERIALS, DAILY_LIMIT, UNIT_PRICE
from drift.tasks.daily import game_day_window


class RestaurantIngredients:
    BUY_WINDOW = 120
    RETURN_WINDOW = 25

    def __init__(self, owner, root, reader):
        self.owner = owner
        self.visual = IngredientShopVisual(root, reader)

    def day(self):
        return game_day_window(datetime.fromtimestamp(self.owner.clock(), timezone.utc))[0]

    @property
    def record(self):
        return self.owner.journal.get('ingredient_shop')

    def start(self, frame):
        task = self.owner
        if (self.record is not None or not task.enabled()
                or not self.visual.tab(frame) or task.visual.detail(frame)):
            return False
        task.journal['ingredient_shop'] = dict(day=self.day(), results={}, order=None,
            buy_until=min(task.execution_deadline(), task.clock()+self.BUY_WINDOW))
        task._set('ingredient_shop', self.BUY_WINDOW+self.RETURN_WINDOW)
        return True

    def can_buy(self):
        task, data = self.owner, self.record
        return (data is not None and task.enabled() and not task.journal.get('cleanup_deadline')
                and task.clock() < data['buy_until'] and data['day']==self.day())

    def _attempt(self, kind):
        return self.owner.journal['attempts'].get(kind, 0) > 0

    def _since_input(self, order):
        receipt = self.owner.journal.get('last_input', {})
        kind = receipt.get('kind', '')
        if kind.startswith('ingredient-') and kind.endswith(f':{order["slot"]}'):
            return self.owner.clock() - receipt['at']
        return self.owner.clock() - order['started']

    def _move(self, frame, action):
        task = self.owner
        kind, point, fields = action
        key = (kind, point, tuple(fields.items()))
        return task._confirmed(key, frame) and task._act(kind, point, frame, **fields)

    def _result(self, slot, state, message=None, **extra):
        task = self.owner
        self.record['results'][str(slot)] = dict(state=state, **extra)
        self.record['order'] = None
        task._save()
        task.event(f'餐品采购：{MATERIALS[slot]} {message or state}')

    def _dismiss(self, frame, slot):
        point = self.visual.cancel(frame, slot)
        kind = f'ingredient-dismiss:{slot}'
        if point and not self._attempt(kind):
            self._move(frame, (kind, point, {'slot': slot}))
        else:
            self.owner._confirmed(None, frame)

    def validate(self, action, frame):
        # The caller already checked the restaurant visit's world, RANK,
        # deadline, stop state and original input limit on this new frame.
        kind = action['kind']
        point = (action['x'], action['y'])
        shop, task = self.visual, self.owner
        current = self.record['order'] if self.record else None
        if kind=='ingredient-atlas':
            return (shop.panel(frame) or task.visual.panel(frame)) and not task.visual.atlas(frame) and not shop.dialog(frame) and point==task.visual.point(frame,'atlas-tab')
        if kind.startswith('ingredient-dismiss:'):
            return bool(current and point==shop.cancel(frame,current['slot']))
        if not self.can_buy():
            return False
        if kind=='ingredient-open':
            return task.visual.panel(frame) and not shop.panel(frame) and point==shop.tab(frame)
        if current is None or action.get('slot') != current['slot']:
            return False
        slot = current['slot']
        if kind==f'ingredient-select:{slot}':
            offer = shop.offer(frame,slot)
            return bool(offer and offer.sold==current['sold'] and point==offer.point)
        quote = shop.quote(frame,slot)
        if quote is None or not (1 <= quote.count <= DAILY_LIMIT-current['sold'] and quote.price==UNIT_PRICE*quote.count):
            return False
        if kind==f'ingredient-max:{slot}':
            return point==quote.maximum and quote.count<DAILY_LIMIT-current['sold']
        if kind==f'ingredient-confirm:{slot}':
            return (point==quote.confirm and action.get('quantity')==quote.count
                    and action.get('price')==quote.price and current.get('quote')==quote.count)
        return False

    def _settle(self, frame, order, elapsed):
        task = self.owner
        slot, before, quantity = order['slot'], order['sold'], order['quote']
        after = self.visual.stock(frame, slot)
        if self.record['day'] == self.day() and after == before+quantity:
            if task._confirmed(('ingredient-stock', slot, after), frame):
                points = UNIT_PRICE*quantity
                self._result(slot, '已购买', f'已购买 {quantity} 份，消耗 {points} 天赋点，当日 {before}→{after}/{DAILY_LIMIT}',
                             quantity=quantity, spent=points, before=before, after=after)
        else:
            task._confirmed(None, frame)
            if elapsed >= 12 or not self.can_buy() or self._attempt(f'ingredient-dismiss:{slot}'):
                self._result(slot, '已提交，数量未确认', before=before, requested=quantity)

    def _step_order(self, frame, order, at_shop):
        task, shop = self.owner, self.visual
        slot = order['slot']
        elapsed = self._since_input(order)
        kind = f'ingredient-confirm:{slot}'
        confirm = self._attempt(kind)
        if at_shop:
            if confirm:
                self._settle(frame, order, elapsed)
            elif self._attempt(f'ingredient-dismiss:{slot}'):
                self._result(slot, '未确认可购买数量及价格')
            elif self._attempt(f'ingredient-select:{slot}'):
                task._confirmed(None, frame)
                if elapsed >= 12 or not self.can_buy():
                    self._result(slot, '未打开数量窗口')
            else:
                offer = shop.offer(frame, slot)
                if self.can_buy() and offer and offer.sold == order['sold']:
                    self._move(frame, (f'ingredient-select:{slot}', offer.point, {'slot': slot}))
                else:
                    self._result(slot, '库存或购买窗口已变化')
            return

        if confirm or not self.can_buy():
            if elapsed >= 12 or not self.can_buy():
                self._dismiss(frame, slot)
            else:
                task._confirmed(None, frame)
            return

        quote = shop.quote(frame, slot)
        remaining = DAILY_LIMIT-order['sold']
        if quote is None or not 1 <= quote.count <= remaining or quote.price != UNIT_PRICE*quote.count:
            if elapsed >= 12:
                self._dismiss(frame, slot)
            else:
                task._confirmed(None, frame)
            return

        maximum = f'ingredient-max:{slot}'
        if quote.count < remaining and quote.maximum and not self._attempt(maximum):
            # The quantity rail changes only the preview. The input dispatcher
            # still checks the material, currency and quote on a fresh frame.
            task._act(maximum, quote.maximum, frame, slot=slot)
        elif task._confirmed((kind, quote.count, quote.price, quote.confirm), frame):
            order['quote'] = quote.count
            task._save()
            task._act(kind, quote.confirm, frame, slot=slot, quantity=quote.count, price=quote.price)

    def step(self, frame):
        task, shop, data = self.owner, self.visual, self.record
        if data is None:
            return
        if task.visual.atlas(frame) and (self._attempt('ingredient-atlas') or not self.can_buy()):
            task._set('atlas')
            return
        at_shop = shop.panel(frame)
        if data['order'] is not None:
            self._step_order(frame, data['order'], at_shop)
            return

        if at_shop:
            if self.can_buy():
                for slot in range(len(MATERIALS)):
                    if str(slot) in data['results']:
                        continue
                    offer = shop.offer(frame,slot)
                    if offer is None:
                        continue
                    if task._confirmed(('ingredient-offer',slot,offer.sold),frame):
                        data['order'] = dict(slot=slot,sold=offer.sold,started=task.clock())
                        task._save()
                        task._act(f'ingredient-select:{slot}', offer.point, frame, slot=slot)
                    return
            tab = task.visual.point(frame,'atlas-tab')
            if tab and not self._attempt('ingredient-atlas'):
                self._move(frame,('ingredient-atlas',tab,{}))
            return
        opened = task.journal.get('last_input', {})
        open_expired = (opened.get('kind') == 'ingredient-open' and task.clock()-opened['at'] >= 12)
        if self.can_buy() and not self._attempt('ingredient-open'):
            tab = shop.tab(frame)
            if tab and task.visual.panel(frame):
                self._move(frame,('ingredient-open',tab,{}))
        elif (open_expired or not self.can_buy()) and task.visual.panel(frame) and not task.visual.detail(frame):
            tab = task.visual.point(frame,'atlas-tab')
            if tab:
                self._move(frame,('ingredient-atlas',tab,{}))
