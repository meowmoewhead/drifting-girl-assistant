"""Offline continuous-upgrade replay with real confirmation and input gates."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from drift.perception.restaurant import CATEGORIES, INGREDIENT_X, RestaurantVisual
from drift.tasks.restaurant_upgrade import RestaurantUpgrade, pack


class RestaurantContinuousTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.now = 100
        self.allow_input = True
        self.available = set()
        self.green = True
        self.issued = []
        self.phases = []
        task = RestaurantUpgrade.__new__(RestaurantUpgrade)
        self.task = task
        task.root = Path(directory.name)
        task.path = task.root/'restaurant_upgrade_state.json'
        task.clock = lambda:self.now
        task.phase = 'detail'
        task.journal = dict(version=1, serial='offline', pending=True, phase='detail',
            deadline=7300, phase_deadline=400, attempts={}, categories=[], category=0,
            visited=[], skipped=[], upgraded=0, submitted=0, feed_counts={},
            blocked_ingredients=[], exhausted_dishes=[], stall=0, scrolls=0,
            previous_list=None, card=dict(level=5, art=pack(self.art(0))))
        task.key = task.since = task.last_seen = task.last_frame = None
        task.seen = 0
        task.next_at = 0
        task.event = Mock()
        task.enabled = lambda:True
        task.same_world = Mock(return_value=True)
        task.view = Mock()
        task.view._off_icon.return_value = True
        task.visual = Mock(spec=RestaurantVisual)
        task.visual.detail.return_value = True
        task.visual.atlas.return_value = False
        task.visual.category.return_value = 0
        task.visual.cards.return_value = []
        task.visual.list_key.return_value = np.arange(25, dtype=np.uint8).reshape(5, 5)
        task.visual.art_matches.side_effect = np.array_equal
        task.visual.find.return_value = (1, 1)
        task.visual.point.return_value = (650, 100)
        task.visual.available_arrow.side_effect = lambda frame, i:i in self.available
        # Keep the production four-slot / green-arrow and ingredient gates.
        task.visual.upgrade_ready.side_effect = lambda frame, values:RestaurantVisual.upgrade_ready(
            task.visual, frame, values)
        task.visual.ingredient_ready.side_effect = lambda frame, i, values:RestaurantVisual.ingredient_ready(
            task.visual, frame, i, values)
        task.visual.detail_values.return_value = self.values()

        def action(request):
            task = self.task
            if not self.allow_input or not task.validate(request, self.current_frame):
                return False
            if not task.reserve(request):
                return False
            self.issued.append(deepcopy(request))
            task.input_issued(request, {'input_id':'offline'})
            return True

        task.action = Mock(side_effect=action)
        task._save()

    @staticmethod
    def art(seed):
        return np.random.default_rng(seed).integers(0, 256, (8, 8, 3), dtype=np.uint8)

    @staticmethod
    def values(level=5, currents=(0, 0, 0, 0), identity='dish-a'):
        return dict(identity=identity, level=level, counters=[(n, 8) for n in currents])

    def frame(self):
        frame = np.zeros((1280, 720, 3), dtype=np.uint8)
        if self.green:
            frame[744:762, 590:619, 1] = 255
        return frame

    def step(self, frame=None):
        self.now += 1.5
        self.current_frame = self.frame() if frame is None else frame
        self.task.step(self.current_frame, due=False)
        self.phases.append(self.task.phase)

    def cycle(self, values=None):
        if values is not None:
            self.task.visual.detail_values.return_value = values
        for _ in range(3):
            self.step()

    def list_phase(self, cards):
        self.task.journal.pop('card', None)
        self.task._set('list', 1200)
        self.task.visual.detail.return_value = False
        self.task.visual.atlas.return_value = True
        self.task.visual.cards.return_value = cards

    def restart(self):
        old = self.task
        with patch('drift.tasks.restaurant_upgrade.RestaurantVisual'), \
             patch('drift.tasks.restaurant_upgrade.GuildVisual'), \
             patch('drift.tasks.restaurant_upgrade.StoneFarm'), \
             patch('drift.tasks.restaurant_upgrade.BackpackSale'):
            self.task = RestaurantUpgrade(old.root, None, old.action, old.event, 'offline', clock=old.clock)
        self.task.visual = old.visual
        self.task.view = old.view
        self.task.same_world = old.same_world

    def scroll_without_new_cards(self):
        self.list_phase([])
        for index in range(5):
            self.task.visual.list_key.return_value = self.art(index)[:, :, 0]
            self.cycle()
            self.assertEqual(self.task.phase, 'list')
            self.assertEqual(self.task.journal['scrolls'], index+1)

    def test_same_dish_upgrades_5_to_6_to_7_without_closing(self):
        for level in (5, 6):
            self.cycle(self.values(level, (8, 8, 8, 8)))
            self.assertEqual(self.task.phase, 'upgrade_wait')
            self.cycle(self.values(level+1))
            self.assertEqual(self.task.phase, 'detail')
        self.assertEqual(self.task.journal['upgraded'], 2)
        self.assertEqual([a['kind'] for a in self.issued], ['upgrade:dish-a:5', 'upgrade:dish-a:6'])
        self.assertNotIn('detail_close', self.phases)
        self.task.event.assert_any_call('餐品强化：实际Lv5→Lv6，继续检查该餐品')

    def test_same_ingredient_can_be_submitted_three_times(self):
        self.available = {0}
        for count in range(3):
            self.cycle(self.values(currents=(count, 0, 0, 0)))
            self.assertEqual(self.task.phase, 'feed_wait')
            self.cycle(self.values(currents=(count+1, 0, 0, 0)))
            self.assertEqual(self.task.phase, 'detail')
        self.assertEqual(self.task.journal['submitted'], 3)
        self.assertEqual([a['kind'] for a in self.issued],
                         [f'feed:dish-a:5:0:{n}' for n in range(3)])
        self.assertEqual(self.task.journal['feed_counts']['0'], 3)

    def test_three_full_slots_never_upgrade_even_with_green_arrow(self):
        values = self.values(currents=(8, 8, 8, 0))
        self.available = {3}
        self.cycle(values)
        self.assertEqual(self.task.phase, 'feed_wait')
        self.assertEqual([a['kind'] for a in self.issued], ['feed:dish-a:5:3:0'])
        self.task._set('detail', 300)
        request = dict(kind='upgrade:dish-a:5', phase='detail', x=604, y=752, before=values)
        self.assertFalse(self.task.validate(request, self.frame()))

    def test_full_slots_without_green_arrow_never_upgrade(self):
        self.green = False
        self.cycle(self.values(currents=(8, 8, 8, 8)))
        self.assertEqual(self.issued, [])
        self.assertEqual(self.task.phase, 'detail_close')

    def test_no_available_material_records_exhaustion_and_closes(self):
        self.cycle()
        self.assertEqual(self.task.phase, 'detail_close')
        self.assertIn('dish-a', self.task.journal['exhausted_dishes'])
        self.assertEqual(self.issued, [])
        saved = json.loads(self.task.path.read_text(encoding='utf-8'))
        self.assertEqual(saved['exhausted_dishes'], ['dish-a'])
        self.task.event.assert_any_call('餐品强化：「dish-a」本轮材料用完（已升 0 级）')

    def test_all_blocked_ingredients_are_not_retried(self):
        self.available = {0, 1, 2, 3}
        self.task.journal['blocked_ingredients'] = [0, 1, 2, 3]
        self.cycle()
        self.assertEqual(self.issued, [])
        self.assertIn('dish-a', self.task.journal['exhausted_dishes'])
        self.assertEqual(self.task.phase, 'detail_close')

    def test_exhausted_identity_is_skipped_before_opening_list_card(self):
        self.task.journal['exhausted_dishes'] = ['dish-a']
        card = dict(identity='dish-a', level=6, art=self.art(1), point=(200, 398))
        self.list_phase([card])
        self.cycle()
        self.assertFalse(any(a['kind'].startswith('dish:') for a in self.issued))
        self.assertNotEqual(self.task.phase, 'detail')

    def test_exhausted_identity_maps_back_to_art_when_card_has_no_name(self):
        self.task.journal['exhausted_dishes'] = ['dish-a']
        self.task.journal['visited'] = [dict(category=0, identity='dish-a', level=5, art=pack(self.art(0)))]
        self.task._visited = Mock(return_value=False)
        self.list_phase([dict(level=7, art=self.art(0), point=(200, 398))])
        self.cycle()
        self.assertFalse(any(a['kind'].startswith('dish:') for a in self.issued))

    def test_three_rejected_detail_cycles_trigger_stall_fuse(self):
        self.available = {0}
        self.allow_input = False
        for stall in (1, 2, 3):
            self.cycle()
            self.assertEqual(self.task.journal['stall'], stall)
            self.assertEqual(self.task.phase, 'detail_close' if stall == 3 else 'detail')
        self.assertIn('dish-a', self.task.journal['exhausted_dishes'])
        self.assertEqual(self.issued, [])

    def test_three_confirmed_no_gain_feeds_trigger_fuse_without_retrying_slot(self):
        self.available = {0, 1, 2, 3}
        for stall in (1, 2, 3):
            self.cycle()
            self.assertEqual(self.task.phase, 'feed_wait')
            self.cycle()
            self.assertEqual(self.task.journal['stall'], stall)
            self.assertEqual(self.task.phase, 'detail_close' if stall == 3 else 'detail')
        self.assertEqual([a['ingredient'] for a in self.issued], [0, 1, 2])
        self.assertEqual(self.task.journal['blocked_ingredients'], [0, 1, 2])
        self.assertIn('dish-a', self.task.journal['exhausted_dishes'])

    def test_confirmed_feed_progress_resets_stall_but_receipt_does_not(self):
        self.available = {0}
        self.task.journal['stall'] = 2
        self.cycle()
        self.assertEqual(self.task.journal['stall'], 2)
        self.cycle(self.values(currents=(1, 0, 0, 0)))
        self.assertEqual(self.task.journal['stall'], 0)
        self.assertEqual(self.task.phase, 'detail')

    def test_new_dish_clears_stall_and_blocked_slots(self):
        self.task.journal.update(stall=2, blocked_ingredients=[0, 1], feed_counts={'0':9})
        self.list_phase([dict(level=5, art=self.art(1), point=(200, 398))])
        self.cycle()
        self.assertEqual(self.task.phase, 'detail')
        self.assertEqual(self.task.journal['stall'], 0)
        self.assertEqual(self.task.journal['blocked_ingredients'], [])
        self.assertEqual(self.task.journal['feed_counts'], {})

    def test_cached_or_unreadable_frames_do_not_count_as_detail_cycles(self):
        self.available = {0}
        self.allow_input = False
        cached = self.frame()
        for _ in range(5):
            self.step(cached)
        self.assertEqual(self.task.journal['stall'], 0)
        self.task.visual.detail_values.return_value = None
        self.cycle()
        self.assertEqual(self.task.journal['stall'], 0)
        self.assertEqual(self.issued, [])

    def test_previous_level_reservation_allows_next_level_but_not_same_level(self):
        self.task.journal['attempts']['upgrade:dish-a:5'] = 1
        for level, allowed in ((5, False), (6, True)):
            with self.subTest(level=level):
                values = self.values(level, (8, 8, 8, 8))
                self.task.visual.detail_values.return_value = values
                request = dict(kind=f'upgrade:dish-a:{level}', phase='detail', x=604, y=752, before=values)
                self.assertEqual(self.task.validate(request, self.frame()), allowed)
        self.cycle(self.values(6, (8, 8, 8, 8)))
        self.assertEqual([a['kind'] for a in self.issued], ['upgrade:dish-a:6'])

    def test_lower_level_than_selected_card_is_still_rejected(self):
        self.available = {0}
        self.cycle(self.values(4))
        self.assertEqual(self.task.phase, 'detail_close')
        self.assertEqual(self.issued, [])
        self.assertEqual(self.task.journal['exhausted_dishes'], [])

    def test_two_exhausted_dishes_do_not_end_category_with_other_enterable_cards(self):
        self.task.journal['exhausted_dishes'] = ['dish-a', 'dish-b']
        self.task.journal['visited'] = [dict(category=0, identity=name, level=5, art=pack(self.art(i)))
                                      for i, name in enumerate(('dish-a', 'dish-b'))]
        self.task.journal['skipped'] = [dict(category=0, identity=name, exhausted=True)
                                      for name in ('dish-a', 'dish-b')]
        self.task.journal.pop('card')
        self.task._set('detail_close')
        self.task.visual.detail.return_value = False
        self.task.visual.atlas.return_value = True
        self.step()
        self.assertEqual(self.task.phase, 'list')
        self.assertEqual(self.task.journal['categories'], [])
        self.task.visual.cards.return_value = [dict(level=5, art=self.art(2), point=(300, 398))]
        self.cycle()
        self.assertEqual(self.task.phase, 'detail')
        self.assertTrue(self.issued[-1]['kind'].startswith('dish:'))

    def test_all_enterable_dishes_exhausted_with_remaining_cards_at_max_level(self):
        self.task.journal['exhausted_dishes'] = ['dish-a', 'dish-b']
        self.task.journal['visited'] = [dict(category=0, identity=name, level=5, art=pack(self.art(i)))
                                      for i, name in enumerate(('dish-a', 'dish-b'))]
        self.task.journal['visited'].extend(dict(category=0, level=20, art=pack(self.art(i)))
                                           for i in range(2, 9))
        self.task.journal['skipped'] = [dict(category=0, identity=name, exhausted=True)
                                      for name in ('dish-a', 'dish-b')]
        self.assertTrue(self.task._category_exhausted())
        self.task.journal['exhausted_dishes'].remove('dish-b')
        self.task.journal['skipped'].pop()
        self.assertFalse(self.task._category_exhausted())

    def test_four_levels_feed_until_exhaustion_and_category_log_counts_unique_dish(self):
        self.available = {0, 1, 2, 3}
        for level in range(5, 9):
            currents = [0]*4
            for i in range(4):
                self.cycle(self.values(level, currents))
                self.assertEqual(self.task.phase, 'feed_wait')
                currents[i] = 8
                self.cycle(self.values(level, currents))
                self.assertEqual(self.task.phase, 'detail')
            self.cycle(self.values(level, currents))
            self.assertEqual(self.task.phase, 'upgrade_wait')
            self.cycle(self.values(level+1))
            self.assertEqual(self.task.phase, 'detail')
        self.assertGreater(self.now-100, 120)
        self.assertNotIn('detail_close', self.phases)
        self.assertEqual(self.task.journal['upgraded'], 4)
        self.assertEqual(self.task.journal['submitted'], 128)
        self.available = set()
        self.cycle(self.values(9))
        self.assertEqual(self.task.phase, 'detail_close')
        self.task.event.assert_any_call('餐品强化：「dish-a」本轮材料用完（已升 4 级）')
        self.task._complete_category()
        self.task.event.assert_any_call(f'餐品强化：「{CATEGORIES[0]}」本轮结束，升级 1 个餐品，提交 128 份食材')

    def test_budget_allows_24_feed_reservations_but_only_one_upgrade_per_level(self):
        feed = dict(kind='feed:dish-a:5:0:0')
        for _ in range(24):
            self.assertTrue(self.task.reserve(feed))
        self.assertFalse(self.task.reserve(feed))
        self.assertTrue(self.task.reserve(dict(kind='feed:dish-a:5:0:1')))
        self.assertTrue(self.task.reserve(dict(kind='upgrade:dish-a:5')))
        self.assertFalse(self.task.reserve(dict(kind='upgrade:dish-a:5')))
        self.assertTrue(self.task.reserve(dict(kind='upgrade:dish-a:6')))

    def test_new_run_has_two_hour_deadline_and_empty_exhausted_list(self):
        self.task.journal.update(pending=False, exhausted_dishes=['old-dish'])
        self.task.phase = 'idle'
        self.task.serial = 'offline'
        self.task.start_screen = Mock(return_value={'route':False})
        self.task.origin_screen = Mock(return_value={'route':False, 'at_restaurant':True})
        for _ in range(3):
            self.now += 1.5
            self.task.step(self.frame())
        self.assertEqual(self.task.phase, 'stop')
        self.assertEqual(self.task.journal['deadline']-self.task.journal['started_at'], 7200)
        self.assertEqual(self.task.journal['exhausted_dishes'], [])

    def test_pending_feed_restart_preserves_used_count_and_resets_stall_only_on_progress(self):
        self.available = {0}
        self.task.journal['stall'] = 2
        self.cycle()
        self.restart()
        self.assertEqual(self.task.phase, 'feed_wait')
        self.assertEqual(self.task.journal['stall'], 2)
        self.assertEqual(self.task.journal['feed_counts'], {'0':1})
        self.cycle(self.values(currents=(1, 0, 0, 0)))
        self.assertEqual(self.task.phase, 'detail')
        self.assertEqual(self.task.journal['stall'], 0)
        self.cycle()
        self.assertEqual([a['kind'] for a in self.issued], ['feed:dish-a:5:0:0', 'feed:dish-a:5:0:1'])

    def test_pending_upgrade_restart_verifies_old_input_then_allows_next_level(self):
        self.task.journal['stall'] = 2
        self.cycle(self.values(currents=(8, 8, 8, 8)))
        self.restart()
        self.assertEqual(self.task.phase, 'upgrade_wait')
        self.assertEqual(self.task.journal['stall'], 2)
        self.cycle(self.values(6))
        self.assertEqual(self.task.phase, 'detail')
        self.assertEqual(self.task.journal['stall'], 0)
        self.cycle(self.values(6, (8, 8, 8, 8)))
        self.assertEqual([a['kind'] for a in self.issued], ['upgrade:dish-a:5', 'upgrade:dish-a:6'])
        self.assertEqual(self.task.journal['dish_upgrades'], {'dish-a':1})

    def test_unchanged_level_after_upgrade_receipt_never_replays_that_upgrade(self):
        self.cycle(self.values(currents=(8, 8, 8, 8)))
        self.cycle()
        self.assertEqual(self.task.phase, 'detail')
        self.assertEqual(self.task.journal['stall'], 1)
        self.assertEqual(self.task.journal['upgraded'], 0)
        self.cycle()
        self.assertEqual(self.task.phase, 'detail_close')
        self.assertEqual([a['kind'] for a in self.issued], ['upgrade:dish-a:5'])
        self.assertIn('dish-a', self.task.journal['exhausted_dishes'])

    def test_new_card_resets_no_new_scroll_count_and_receives_a_fresh_bounded_budget(self):
        self.scroll_without_new_cards()
        self.task.visual.cards.return_value = [dict(level=5, art=self.art(9), point=(200, 398))]
        self.cycle()
        self.assertEqual(self.task.phase, 'detail')
        self.assertEqual(self.task.journal['scrolls'], 0)
        self.assertIsNone(self.task.journal['previous_list'])
        self.task.visual.detail.return_value = True
        self.cycle()
        self.assertEqual(self.task.phase, 'detail_close')
        self.task.visual.detail.return_value = False
        self.step()
        self.assertEqual(self.task.phase, 'list')
        self.task.visual.cards.return_value = []
        self.task.visual.list_key.return_value = self.art(10)[:, :, 0]
        self.cycle()
        self.assertEqual(self.task.phase, 'list')
        self.assertEqual(self.task.journal['scrolls'], 1)
        scrolls = [a['kind'] for a in self.issued if a['kind'].startswith('scroll:')]
        self.assertEqual(scrolls, ['scroll:0:0']*5 + ['scroll:0:1'])

    def test_five_scrolls_with_no_new_cards_end_category_without_sixth_scroll(self):
        self.scroll_without_new_cards()
        self.task.visual.list_key.return_value = self.art(10)[:, :, 0]
        self.cycle()
        self.assertEqual(self.task.phase, 'category_next')
        self.assertEqual(self.task.journal['categories'], [0])
        self.assertEqual(len(self.issued), 5)


if __name__ == '__main__':
    unittest.main()
