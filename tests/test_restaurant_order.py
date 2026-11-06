from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2

import tests.test_restaurant_continuous as continuous
from drift.perception.restaurant import RestaurantVisual


class RestaurantOrderTests(unittest.TestCase):
    setUp=continuous.RestaurantContinuousTests.setUp
    art=staticmethod(continuous.RestaurantContinuousTests.art)
    values=staticmethod(continuous.RestaurantContinuousTests.values)
    frame=continuous.RestaurantContinuousTests.frame
    step=continuous.RestaurantContinuousTests.step
    cycle=continuous.RestaurantContinuousTests.cycle
    list_phase=continuous.RestaurantContinuousTests.list_phase

    def card(self,x,seed,level=5,**extra):
        return dict(point=(x,398),art=self.art(seed),level=level,**extra)

    def test_selection_uses_screen_order_even_if_detector_list_is_shuffled(self):
        self.list_phase([self.card(520,2),self.card(200,0),self.card(360,1)])
        self.cycle()
        self.assertEqual(self.task.phase,'detail')
        self.assertEqual([a['x'] for a in self.issued],[200])
        self.assertEqual(self.task.journal['card']['point'],[200,398])

    def test_unread_left_card_blocks_right_card_without_input_or_scroll(self):
        self.list_phase([self.card(360,1),self.card(200,0,level=None)])
        before=deepcopy(self.task.journal['attempts'])
        self.cycle();self.cycle()
        self.assertEqual(self.task.phase,'list')
        self.assertEqual(self.issued,[])
        self.assertEqual(self.task.journal['attempts'],before)
        self.assertIn('不越过',self.task.order_hold)

    def test_new_frame_cannot_authorize_right_card_over_an_unread_left_slot(self):
        right=self.card(360,1)
        self.list_phase([right]);self.cycle()
        request=self.issued[-1]
        self.task.phase='list'
        self.task.visual.cards.return_value=[right,self.card(200,0,level=None)]
        self.assertFalse(self.task.validate(request,self.frame()))

    def test_maxed_and_exhausted_left_cards_can_be_skipped_in_order(self):
        self.task.journal['exhausted_dishes']=['done']
        self.list_phase([self.card(520,2),self.card(360,1,identity='done'),self.card(200,0,level=20)])
        self.cycle()
        self.assertEqual([a['x'] for a in self.issued],[520])
        self.assertTrue(any(c['level']==20 for c in self.task.journal['visited']))

    def test_verified_next_level_refreshes_slots_without_erasing_old_attempts_or_deadline(self):
        self.cycle(self.values(5,(8,8,8,8)))
        self.task.journal['feed_counts']={'0':13}
        self.task.journal['blocked_ingredients']=[0,1]
        self.task.journal['attempts']['feed:dish-a:5:0:0']=1
        attempts=deepcopy(self.task.journal['attempts'])
        deadline=self.task.journal['deadline']
        self.cycle(self.values(6))
        self.assertEqual(self.task.journal['card']['level'],6)
        self.assertEqual(self.task.journal['feed_counts'],{})
        self.assertEqual(self.task.journal['blocked_ingredients'],[])
        self.assertEqual(self.task.journal['attempts'],attempts)
        self.assertEqual(self.task.journal['deadline'],deadline)
        self.available={0};self.cycle(self.values(6))
        self.assertEqual(self.issued[-1]['kind'],'feed:dish-a:6:0:0')
        self.assertEqual(self.task.journal['attempts']['feed:dish-a:5:0:0'],1)

    def test_left_dish_finishes_multiple_levels_before_right_dish_is_selected(self):
        left,right=self.card(200,0),self.card(360,1)
        self.list_phase([right,left]);self.cycle()
        self.task.visual.detail.return_value=True
        self.task.visual.atlas.return_value=False
        for level in (5,6):
            self.cycle(self.values(level,(8,8,8,8)))
            self.cycle(self.values(level+1))
            self.assertEqual(self.task.phase,'detail')
            self.assertEqual(sum(a['kind'].startswith('dish:') for a in self.issued),1)
        self.available=set();self.cycle(self.values(7))
        self.assertEqual(self.task.phase,'detail_close')
        self.task.visual.detail.return_value=False
        self.task.visual.atlas.return_value=True
        self.step();self.cycle()
        chosen=[a['x'] for a in self.issued if a['kind'].startswith('dish:')]
        self.assertEqual(chosen,[200,360])
        self.assertEqual(self.task.journal['visited'][0]['level'],7)

    def test_all_four_ingredients_are_fed_from_left_to_right_before_upgrade(self):
        self.available={0,1,2,3};currents=[0,0,0,0]
        for index in range(4):
            self.cycle(self.values(5,currents))
            self.assertEqual(self.issued[-1]['ingredient'],index)
            currents[index]=8
            self.cycle(self.values(5,currents))
        self.assertEqual([a['ingredient'] for a in self.issued],[0,1,2,3])
        self.cycle(self.values(5,currents))
        self.assertEqual(self.issued[-1]['kind'],'upgrade:dish-a:5')


class StrictCardPerceptionTests(unittest.TestCase):
    def test_real_atlas_retains_unread_left_slot_only_for_strict_traversal(self):
        root=Path(__file__).resolve().parents[1]
        fixture=root/'tests/fixtures/restaurant-list/atlas-left-unread.png'
        self.assertTrue(fixture.is_file(),'Missing real atlas fixture')
        frame=cv2.cvtColor(cv2.imread(str(fixture)),cv2.COLOR_BGR2RGB)
        reader=Mock()
        values=[None,None,20,20,20,20,1,1]
        reader.read_many.return_value=[SimpleNamespace(value=v,confidence=.999 if v is not None else 0) for v in values]
        visual=RestaurantVisual(root,reader)
        self.assertEqual([c['level'] for c in visual.cards(frame)],[20,20,1])
        strict=visual.cards(frame,strict_order=True)
        self.assertEqual([c['level'] for c in strict],[None,20,20,1])
        self.assertEqual([c['point'][0] for c in strict],sorted(c['point'][0] for c in strict))


if __name__=='__main__':unittest.main()
