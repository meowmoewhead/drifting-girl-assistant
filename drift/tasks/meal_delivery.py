"""Complete one visible cooking delivery using the restaurant navigation."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import time
import uuid
import cv2
import numpy as np

from drift.core.journal import archive_bytes
from drift.perception.meal_delivery import (
    BOOK_CLOSE, CONFIRM_ONE, CRAFT_CLOSE, CRAFT_POINT, QUANTITY_CLOSE,
    MealDeliveryVisual, same_title, title_strokes,
)
from drift.tasks.daily import BEIJING, game_day_window
from drift.tasks.restaurant_upgrade import RestaurantUpgrade, pack, unpack


MEAL_ID = 'driftinggirl_meal_delivery'
FEATURE = 'meal_delivery'
DAILY_LIMIT = 3
CHECK_SECONDS = 180
MEAL_RULE = dict(id=MEAL_ID, name='料理配送', enabled=False,
                 asset='assets/parallel/world-badge.png', source_crop=[0,0,26,26],
                 search_area=[10,43,70,74], scales=[1], threshold=.95,
                 interval=CHECK_SECONDS, stable_frames=3, stage='controller_meal_delivery',
                 action='observe')


def meal_day_status(delivered, now):
    period, following = game_day_window(now)
    count = sum(entry['period'] == period for entry in delivered)
    return dict(period=period, completed=count >= DAILY_LIMIT, delivered_today=count,
                daily_limit=DAILY_LIMIT, reset_hour=8, timezone='Asia/Shanghai',
                next_reset=following.isoformat())


def saved_meal_day_status(root, serial, now):
    # The delivery receipts remain authoritative, including while the worker is stopped.
    delivered = []
    try:
        saved = json.loads((Path(root)/'meal_delivery_state.json').read_text(encoding='utf-8-sig'))
        if saved.get('serial') == serial:
            delivered = [entry for entry in saved.get('delivered', [])
                         if isinstance(entry, dict) and 'period' in entry]
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return meal_day_status(delivered, now)


class MealDelivery(RestaurantUpgrade):
    LIMIT = 1200
    ROUTING = {'stop','settings_open','settings','settings_close','map_open','map',
               'map_close','travel','restore','restore_menu','restore_verify','menu_retire'}
    PHASES = RestaurantUpgrade.PHASES | {
        'inspect','read','book','book_close','depart','cook','confirm','leave_cook',
        'reopen','delivery','arrive','settle','skip_order','delivery_cleanup',
    }

    def __init__(self, root, reader, action, event, serial, interval=CHECK_SECONDS, *,
                 state_path=None, clock=time.time):
        super().__init__(root, reader, action,
                         lambda message: event(message.replace('餐品强化', '料理配送', 1)),
                         serial, interval, state_path=state_path or Path(root)/'meal_delivery_state.json',
                         clock=clock)
        self.meal = MealDeliveryVisual(root, reader)
        self._check_startup = False
        self._adjust_before = self.journal.get('adjust_before')
        self._reconcile_on_start = True

    def period(self):
        return game_day_window(datetime.fromtimestamp(self.clock(), BEIJING))[0]

    def today(self):
        period = self.period()
        return [entry for entry in self.journal.get('delivered', []) if entry['period']==period]

    def daily_status(self):
        return meal_day_status(self.journal.get('delivered', []),
                               datetime.fromtimestamp(self.clock(), BEIJING))

    def status(self):
        return dict(**self.daily_status(), phase=self.phase, pending=self.active,
                    recipe=self.journal.get('recipe'), npc=self.journal.get('npc'),
                    deadline=self.journal.get('deadline'), attempts=self.journal.get('attempts',{}),
                    delivered=self.journal.get('delivered',[]), failure=self.journal.get('failure'),
                    hold=self.hold, last_input=self.journal.get('last_input'))

    @staticmethod
    def input_limit(kind):
        return 1 if kind.startswith('meal-') else RestaurantUpgrade.input_limit(kind)

    def ordinary_world(self, frame):
        return (super().ordinary_world(frame) or
                bool(self.fishing._main_visible(frame) and self.meal.npc_points(frame)
                     and not self.view._battle(frame)))

    def _begin(self, frame, card, point):
        delivered = self.journal.get('delivered', [])
        blocked = {key: record for key, record in self.journal.get('blocked', {}).items()
                   if record['period'] == self.period()}
        inspected = self.journal.get('inspected', []) if self.journal.get('period')==self.period() else []
        if self.path.is_file():
            raw = self.path.read_bytes()
            archive_bytes(self.root/'logs/meal-delivery-history'/(hashlib.sha256(raw).hexdigest()+'.json'),
                          raw, '料理配送')
        self.journal = dict(version=1, serial=self.serial, pending=True, phase='idle',
                            run_id=uuid.uuid4().hex, started_at=self.clock(),
                            deadline=self.clock()+self.LIMIT, period=self.period(),
                            startup=dict(parallel=self.view._parallel(frame), header=self.header(frame)),
                            attempts={}, delivered=delivered, blocked=blocked, inspected=inspected)
        if point is not None:
            self.journal.update(scan_slot=point,scan_npc=point.split(':',1)[0])
        self._set('read' if card else 'inspect', 35)

    def _same_order(self, frame):
        card = self.meal.order(frame)
        return card if self._matches_order(frame, card, self.journal) else None

    def _matches_order(self,frame,card,record):
        if card is None or not record.get('speech') or card.npc not in (None,record.get('npc')):
            return False
        fixed=unpack(record['speech'])[:,:,0]
        seen=self.meal.speech(frame,card.top)
        if np.count_nonzero(fixed) < 300 or np.count_nonzero(seen) < 300:
            return False
        scores = cv2.matchTemplate(np.pad(seen, 4), fixed, cv2.TM_CCOEFF_NORMED)
        return float(scores.max()) >= .84

    def _record_delivery(self, archived=None):
        source = self.journal if archived is None else archived
        contact = source.get('last_input', {})
        delivered = self.journal.get('delivered', [])
        input_id = contact.get('input_id')
        if not input_id or any(entry.get('input_id')==input_id for entry in delivered):
            return False
        # Confirmation may come after the reset. Use the submission time.
        period = game_day_window(datetime.fromtimestamp(contact['at'], BEIJING))[0]
        count = sum(entry['period']==period for entry in delivered)
        if count >= DAILY_LIMIT:
            return False
        entry = dict(period=period, recipe=source.get('recipe'),
                     npc=source.get('npc'), at=contact['at'], input_id=input_id)
        if archived is None:
            self.journal['settled'] = True
            self.journal['inspected'] = []
        else:
            entry.update(reconciled=True,run_id=source['run_id'])
            if source.get('run_id')==self.journal.get('run_id'):
                self.journal['settled'] = True
        self.journal['delivered'] = [*delivered[-29:], entry]
        self._save()
        verb = '根据已保存前后截图补记' if archived is not None else '已确认交付'
        self.event(f"料理配送：{verb}，{period} 周期完成 {count+1}/{DAILY_LIMIT} 单")
        return True

    def _replacement_order(self, frame, card, record):
        return bool(card is not None and not card.cooked and not card.arrived
                    and not self._matches_order(frame,card,record)
                    and (self.meal.new_order(frame)
                         or card.npc is not None and card.npc!=record.get('npc')))

    def _final_submission_reward(self, before, after, record):
        npc = record.get('npc')
        if not npc or not self._main(after) or not self.meal.final_reward_burst(after):
            return False
        # The last order has no successor card. Confirm both flying rewards
        # and the removal of one portrait of the submitting NPC.
        prefix = npc + ':'
        before_count = sum(slot.startswith(prefix) for slot in self.meal.npc_points(before))
        after_count = sum(slot.startswith(prefix) for slot in self.meal.npc_points(after))
        return before_count == after_count + 1

    def _saved_submission_succeeded(self, record):
        contact=record.get('last_input',{})
        uid=contact.get('input_id','')
        if (record.get('serial')!=self.serial
                or record.get('settled') or contact.get('kind')!='meal-submit'
                or contact.get('state')!='issued_only' or record.get('attempts',{}).get('meal-submit')!=1
                or not isinstance(uid,str) or not re.fullmatch('[a-f0-9]{32}',uid)):
            return False
        folder=self.root/'logs/meal-delivery-input'
        try:
            if game_day_window(datetime.fromtimestamp(contact['at'], BEIJING))[0] != self.period():
                return False
            receipt=json.loads((folder/f'input-{uid}.json').read_text(encoding='utf-8-sig'))
            action=receipt['action']
            if (receipt['input_id']!=uid or receipt['result']!='issued_only'
                    or receipt['controller']!=FEATURE or action['kind']!='meal-submit'
                    or action['phase'] not in ('delivery','arrive') or (action['x'],action['y'])!=(547,820)
                    or abs(receipt['time']-contact['at'])>2
                    or not receipt.get('after_frame_id') or receipt['frame_id']==receipt['after_frame_id']):
                return False
            before,after=(f'{uid}-{when}.png' for when in ('before','after'))
            if receipt.get('before_file')!=before or receipt.get('after_file')!=after:
                return False
            images=[cv2.imdecode(np.frombuffer((folder/name).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
                    for name in (before,after)]
            if any(image is None or image.shape!=(1280,720,3) for image in images):
                return False
            first,last=(cv2.cvtColor(image,cv2.COLOR_BGR2RGB) for image in images)
            card=self.meal.order(first)
            return bool(card and card.can_finish and self._matches_order(first,card,record)
                        and self.same_world(first,journal=record) and self.same_world(last,journal=record)
                        and (self._replacement_order(last,self.meal.order(last),record)
                             or self.meal.delivery_reward(last)
                             or self.meal.order(last) is None
                             and self._final_submission_reward(first,last,record)))
        except (OSError,ValueError,TypeError,KeyError,OverflowError,cv2.error):
            return False

    def _reconcile_recent_submissions(self):
        folder=self.root/'logs/meal-delivery-history'
        records=[self.journal]
        if folder.is_dir():
            for path in sorted(folder.glob('*.json'),key=lambda file:file.stat().st_mtime,reverse=True)[:8]:
                try:
                    record=json.loads(path.read_text(encoding='utf-8-sig'))
                except (OSError,ValueError):
                    continue
                if isinstance(record,dict):
                    records.append(record)
        verified=False
        for record in records:
            if self.daily_status()['completed']:
                break
            if self._saved_submission_succeeded(record):
                verified=self._record_delivery(archived=record) or verified
        return verified

    def _finish(self, success, *, message=None):
        self.done = bool(success and self.journal.get('settled'))
        self.phase = 'idle'
        self.journal.update(phase='idle', pending=False, finished_at=self.clock(), success=self.done)
        self._save()
        if message:
            self.event(message)

    def abandon(self, reason='本轮中止', **_):
        self._block_uncertain()
        self.journal['failure'] = reason
        self._finish(False, message='料理配送：'+reason)
        return False

    def _block_uncertain(self):
        attempts = self.journal.get('attempts',{})
        if (not self.journal.get('settled') and self.journal.get('order') and
                (self.journal.get('destination_locked')
                 or any(attempts.get(key) for key in ('meal-make','meal-submit')))):
            entry = {
                key:self.journal.get(key) for key in ('period','npc','speech')}
            if self.journal.get('destination_locked') and not attempts.get('meal-submit'):
                entry['locked_in_parallel'] = self.journal['startup']['parallel']
            self.journal['blocked'][self.journal['order']] = entry

    def _order_deferred(self, frame, card):
        for record in self.journal.get('blocked',{}).values():
            if record['period']!=self.period() or not self._matches_order(frame,card,record):
                continue
            if ('locked_in_parallel' in record and card.cooked
                    and (card.arrived or record['locked_in_parallel']!=self.view._parallel(frame))):
                continue
            return True
        return False

    def offered(self, frame):
        if not self.enabled() or self.daily_status()['completed']:
            return False
        card=self.meal.order(frame)
        return bool(card is not None and card.npc is not None
                    and not self._order_deferred(frame,card))

    def owns_order_panel(self, frame):
        if not self.active or not self.same_world(frame):
            return False
        card=self.meal.order(frame)
        if card is not None:
            return (self._matches_order(frame,card,self.journal)
                    or self.phase=='inspect' and card.npc==self.journal.get('scan_npc')
                    or self.phase=='read' and 'order' not in self.journal and card.npc is not None
                    or self.phase=='settle' and self._replacement_order(frame,card,self.journal))
        if self.phase=='inspect' and self.journal.get('scan_npc'):
            return bool(self.meal.other_quest_close(frame,self.journal['scan_npc']))
        return bool(self.phase=='reopen' and self._last_reopen_slot()
                    and self.meal.other_quest_close(frame,self.journal.get('npc')))

    def observe_route_reply(self, frame):
        if (frame is not None and self.same_world(frame) and self._same_order(frame)
                and self.meal.destination_locked(frame)):
            self.journal['destination_locked'] = True
            self._save()

    def observe_submit_reply(self, frame, *, before=None):
        if frame is None or not self._issued('meal-submit') or not self.same_world(frame):
            return
        card=self.meal.order(frame)
        if (self._replacement_order(frame,card,self.journal)
                or card is None and (self.meal.delivery_reward(frame)
                                     or before is not None and self.same_world(before)
                                     and self._same_order(before) is not None
                                     and self._final_submission_reward(before,frame,self.journal))):
            self.journal['reward_observed'] = True
            self._save()

    def _begin_cleanup(self, reason):
        if self.journal.get('cleanup_deadline'):
            return self.abandon(reason)
        self._block_uncertain()
        self.journal.update(failure=reason, failed_phase=self.phase,
                            failed_phase_deadline=self.journal.get('phase_deadline'))
        self.journal['cleanup_deadline'] = self.clock()+75
        self._set('delivery_cleanup', 30)
        self.event('料理配送：'+reason+'，保留已用输入，仅收起已识别页面')
        return True

    def _idle(self, frame, due):
        if not due or not self.enabled() or self.daily_status()['completed']:
            return False
        card = self.meal.order(frame)
        if card is not None and card.npc is not None:
            if self._order_deferred(frame,card):
                return False
            point = None
            key = 'order', card.signature
        elif card is None and self._main(frame):
            points = self.meal.npc_points(frame)
            if not points:
                return False
            seen = self.journal.get('inspected',[]) if self.journal.get('period')==self.period() else []
            previous = self.journal.get('scan_slot') if self.journal.get('failure') else None
            if previous in points and previous not in seen:
                point = previous
            else:
                available = [name for name in points if name not in seen]
                if not available:
                    return False
                point = max(available,key=lambda name:points[name][1])
            key = 'roster', point, points[point]
        else:
            self._confirmed(None,frame)
            return False
        if self._confirmed(key,frame):
            self._begin(frame, card, point)
        return self.active

    def _open_at_restaurant(self, frame):
        if self.journal.get('already_cooked'):
            self._set('reopen',90)
        else:
            super()._open_at_restaurant(frame)

    def _nav(self, destination):
        if destination == 'restaurant' and self.journal.get('already_cooked'):
            self._set('reopen',90)
        else:
            super()._nav(destination)

    def needs_map_origin(self):
        # The order supplies its own destination after cooking.
        return False

    def scan_start_direction(self, frame):
        return 'west' if self.meal.find(frame,'map-restaurant-left',(0,195,26,810)) else None

    def _main(self, frame):
        return bool(self.fishing._main_visible(frame) and not self.map_ready(frame)
                    and not self.view._battle(frame) and not self.view.training.boss_mode(frame)
                    and not self.sailing.menu(frame))

    def _restaurant_button(self, frame):
        target = (675,1084)
        button = self.visual.point(frame, 'entry')
        if button is not None and abs(button[0]-target[0])<=12 and abs(button[1]-target[1])<=14:
            return target
        # The plate bounces. Only use its verified fixed touch island after
        # the restaurant text and small location icon have been recognized.
        match = self.meal.find(frame, 'restaurant_header', (247,34,297,79))
        disk = frame[1050:1118,643:713]
        light = ((disk.min(axis=2) >= 155) & (disk[...,0] >= 178)).sum()
        return target if match is not None and light >= 780 else None

    def _last_reopen_slot(self):
        kinds=[kind for kind in self.journal['attempts'] if kind.startswith('meal-reopen:')]
        return kinds[-1].split(':',1)[1] if kinds else None

    def _reopen_action(self, frame, card):
        previous=self._last_reopen_slot()
        npc=self.journal.get('npc')
        attempts=self.journal['attempts']
        if previous is not None and not attempts.get('meal-reopen-dismiss:'+previous):
            if card is None:
                return 'meal-reopen-dismiss:'+previous, self.meal.other_quest_close(frame,npc)
            return None,None
        if card is not None or not npc or not self._main(frame) or attempts.get('meal-reopen'):
            return None,None
        points=self.meal.npc_points(frame)
        available={slot:point for slot,point in points.items()
                   if slot.startswith(npc+':') and not attempts.get('meal-reopen:'+slot)}
        if not available:
            return None,None
        saved=self.journal.get('scan_slot')
        slot=saved if saved in available else max(available,key=lambda key:available[key][1])
        return 'meal-reopen:'+slot, available[slot]

    def _next_action(self, frame):
        phase = self.phase
        card = self.meal.order(frame)
        same = self._matches_order(frame,card,self.journal)
        book = self.meal.book_visible(frame)
        cooking = self.meal.cooking(frame)
        one = self.meal.one_portion(frame)
        if phase=='inspect' and card is None and self._main(frame):
            return 'meal-inspect', self.meal.npc_points(frame).get(self.journal.get('scan_slot'))
        if phase=='reopen':
            return self._reopen_action(frame,card)
        if phase=='inspect' and not card and not book:
            return 'meal-dismiss-other', self.meal.other_quest_close(frame,self.journal['scan_npc'])
        if phase=='skip_order' and same and card:
            return 'meal-dismiss-deferred', card.close
        if phase=='read' and same and not card.cooked and not book and not one:
            return 'meal-book', self.meal.book_button(frame,card)
        if phase=='book_close' and book:
            return 'meal-book-close', BOOK_CLOSE
        if phase=='depart' and same and not book:
            return 'meal-order-close', card.close
        if phase=='open' and not cooking and self.at_restaurant(frame) and self._main(frame):
            return 'meal-restaurant-open', self._restaurant_button(frame)
        if phase=='cook' and cooking and not one and self.visual.find(frame,'location'):
            now = self.meal.recipe(frame,'cooking')
            if now is None:
                return None, None
            target = tuple(self.journal['recipe'])
            if now==target:
                return 'meal-cook-form', CRAFT_POINT
            slot = next(i for i,(a,b) in enumerate(zip(now,target)) if a!=b)
            increase = target[slot] > now[slot]
            return f'meal-adjust-{slot}-{now[slot]}-{"plus" if increase else "minus"}', self.meal.arrow(frame,slot,increase)
        if phase=='confirm' and one:
            shape = title_strokes(frame,379)
            expected = unpack(self.journal['title'])[:,:,0]
            if same_title(expected,shape) and self.meal.recipe(frame,'one')==tuple(self.journal['recipe']):
                return 'meal-make', CONFIRM_ONE
        if phase=='leave_cook' and cooking and not one:
            return 'meal-restaurant-close', CRAFT_CLOSE
        if (phase=='delivery' and same and card.travel and not card.arrived
                and self.view._off_icon(frame) and not self.bag.moving(frame)):
            return 'meal-route', (585,735)
        if (phase in ('delivery','arrive') and same and card.can_finish
                and self.view._off_icon(frame) and not self.bag.moving(frame)):
            return 'meal-submit', (547,820)
        if (phase=='settle' and self.journal.get('settled') and self.route_was_running()
                and self._replacement_order(frame,card,self.journal)):
            return 'meal-close-next', card.close
        if phase=='settle' and self.journal.get('settled') and self.meal.arrival_record(frame):
            return 'meal-record-close', (361,857)
        if phase in ('settle','delivery_cleanup'):
            if phase=='delivery_cleanup' and self.map_ready(frame) and not self.bag.moving(frame):
                return 'meal-clean-map', (668,1187)
            if book:
                return 'meal-clean-book', BOOK_CLOSE
            if one:
                return 'meal-clean-confirm', QUANTITY_CLOSE
            if same and card:
                return 'meal-clean-order',card.close
            if cooking:
                return 'meal-clean-craft',CRAFT_CLOSE
            return 'meal-dismiss-after', self.meal.other_quest_close(frame)
        return None, None

    def validate(self, action, frame):
        kind=action['kind']
        if not kind.startswith('meal-'):
            return super().validate(action,frame)
        if (not self.allowed() or not self.same_world(frame) or action['phase']!=self.phase
                or self.journal['attempts'].get(kind,0) >= self.input_limit(kind)):
            return False
        if not self.enabled() and self.phase!='delivery_cleanup':
            return False
        actual_kind, actual_point = self._next_action(frame)
        if kind!=actual_kind or actual_point != (action['x'],action['y']):
            return False
        if kind in ('meal-make','meal-submit'):
            return (self.journal.get('period')==self.period() and len(self.today())<DAILY_LIMIT
                    and self.view._off_icon(frame) and not self.bag.moving(frame))
        return True

    def _acquire(self, frame, card):
        self.journal['order'] = card.signature
        self.journal['npc'] = card.npc or self.journal.get('scan_npc')
        self.journal['speech'] = pack(self.meal.speech(frame,card.top))
        self._save()

    def _issued(self, kind):
        contact = self.journal.get('last_input', {})
        return contact.get('kind')==kind and contact.get('state')=='issued_only'

    def _apply(self, kind, point, frame):
        if kind is None or point is None or self.journal['attempts'].get(kind,0):
            return False
        if not self._confirmed((self.phase,kind,point),frame):
            return False
        if not self._act(kind,point,frame):
            return False
        if kind.startswith('meal-adjust-'):
            self._adjust_before = list(self.meal.recipe(frame,'cooking'))
            self.journal['adjust_before'] = self._adjust_before
            self.journal['adjust_kind'] = kind
            self._save()
        phases = {'meal-book':'book','meal-book-close':'depart','meal-cook-form':'confirm',
                  'meal-make':'leave_cook','meal-restaurant-close':'reopen',
                  'meal-route':'arrive','meal-submit':'settle'}
        if kind in phases:
            self._set(phases[kind], 480 if kind=='meal-route' else 45)
        if kind in ('meal-dismiss-other','meal-dismiss-deferred'):
            seen=self.journal.get('inspected',[])
            self.journal['inspected']=[*seen,self.journal.get('scan_slot',self.journal['scan_npc'])]
            self._finish(False)
        return True

    def step(self, frame, *, due=True):
        self.done=None
        self._observed_frame=frame
        if not self.active:
            if self.daily_status()['completed']:
                return False
            if self._reconcile_on_start:
                self._reconcile_on_start=False
                if self._reconcile_recent_submissions():
                    self.done=True
                    return False
            return self._idle(frame,due)
        if not self.allowed():
            if not self._begin_cleanup(f'{self.phase} 阶段确认超时'):
                return self.active
        if not self.enabled() and not self.journal.get('cleanup_deadline'):
            self._begin_cleanup('开关已关闭')
        if self.phase in self.ROUTING:
            return super().step(frame,due=False)
        if self.clock()<self.next_at or not self.same_world(frame):
            return True
        phase=self.phase
        card=self.meal.order(frame)
        if phase=='inspect' and card is not None:
            self._acquire(frame,card)
            self._set('skip_order' if self._order_deferred(frame,card) else 'read',35)
            return True
        if phase=='read' and card is not None and 'order' not in self.journal:
            self._acquire(frame,card)
        if phase=='read' and card and card.cooked:
            if self._matches_order(frame,card,self.journal):
                if self.view._off_icon(frame) and self.sailing.running(frame) is not True:
                    self._set('delivery',90)
                else:
                    self.journal['already_cooked']=True
                    self._set('depart',35)
            return True
        if phase=='book':
            shape = title_strokes(frame,449) if self.meal.book_visible(frame) else None
            recipe = self.meal.recipe(frame,'book') if shape is not None else None
            key = (recipe,hashlib.sha256(shape.tobytes()).hexdigest()) if recipe else None
            if self._confirmed(key,frame):
                self.journal.update(recipe=list(recipe),title=pack(shape))
                self.event('料理配送：任务册数量 '+','.join(map(str,recipe)))
                self._set('book_close',25)
            return True
        if phase=='depart' and self._main(frame) and card is None:
            context=self.start_screen(frame)
            if self._confirmed(self.context_key(context),frame,2) and context:
                self.journal['startup']=context
                self.journal['resume_route']=context['route']
                self._set('stop',self.STOP_ORIGIN_LIMIT)
            return True
        if phase=='open' and self.meal.cooking(frame) and self.visual.find(frame,'location'):
            self._set('cook',100)
            return True
        if phase=='cook' and self._adjust_before is not None:
            reading=self.meal.recipe(frame,'cooking') if self.meal.cooking(frame) else None
            _,_,slot,_,direction=self.journal['adjust_kind'].split('-')
            index=int(slot)
            expected=list(self._adjust_before)
            expected[index]+=1 if direction=='plus' else -1
            if reading==tuple(expected):
                self._adjust_before=None
                self.journal.pop('adjust_before',None)
                self.journal.pop('adjust_kind',None)
                self._save()
            return True
        if (phase=='confirm' and self._issued('meal-make') and self.meal.cooking(frame)
                and not self.meal.one_portion(frame)):
            self._set('leave_cook',max(0,self.journal['phase_deadline']-self.clock()))
            return True
        if phase=='reopen' and self._same_order(frame) is not None and card.cooked:
            self._set('delivery',90)
            return True
        if phase=='arrive':
            if not self.journal.get('destination_locked'):
                self.observe_route_reply(frame)
            if self.journal.get('destination_locked'):
                self._begin_cleanup('目标地区尚未发现，保留已制作的订单')
                return self.active
        if phase in ('delivery','arrive') and self._issued('meal-submit'):
            self._set('settle',max(0,self.journal['phase_deadline']-self.clock()))
            return True
        if phase=='settle' and not self.journal.get('settled'):
            if not self.journal.get('reward_observed'):
                self.observe_submit_reply(frame)
            replaced=(self._replacement_order(frame,card,self.journal)
                      or card is None and self.meal.other_quest_close(frame,self.journal.get('npc')))
            if self.journal.get('reward_observed') or self._confirmed('result' if replaced else None,frame,2):
                self._record_delivery()
            return True
        if phase in ('settle','delivery_cleanup') and (self._main(frame) and card is None
                or not self.route_was_running() and card is not None
                and not self._matches_order(frame,card,self.journal)):
            if self.route_was_running():
                self._set('restore',60)
            else:
                self._finish(phase=='settle')
            return self.active
        kind,point=self._next_action(frame)
        self._apply(kind,point,frame)
        return self.active
