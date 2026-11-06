from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from drift.perception.restaurant import RestaurantVisual
from drift.tasks import restaurant_list as scan
from drift.tasks.restaurant_upgrade import pack
from tests import test_restaurant_continuous as continuous

ROOT=Path(__file__).resolve().parents[1]
FIXTURES=ROOT/'tests/fixtures/restaurant-list'


class RestaurantListTests(unittest.TestCase):
    setUp=continuous.RestaurantContinuousTests.setUp
    art=staticmethod(continuous.RestaurantContinuousTests.art)
    values=staticmethod(continuous.RestaurantContinuousTests.values)
    frame=continuous.RestaurantContinuousTests.frame
    step=continuous.RestaurantContinuousTests.step
    cycle=continuous.RestaurantContinuousTests.cycle
    list_phase=continuous.RestaurantContinuousTests.list_phase

    def test_real_ten_entries_are_only_eight_unique_dishes(self):
        record=json.loads((FIXTURES/'coverage-eight.json').read_text(encoding='utf-8'))
        self.task.journal.update(record)
        self.task.visual.art_matches.side_effect=RestaurantVisual.art_matches
        self.assertEqual(len(record['visited']),10)
        self.assertEqual(scan.count(self.task,0),8)
        self.assertFalse(self.task._category_exhausted())
        self.task._complete_category()
        self.assertEqual(self.task.journal['categories'],[])

    def test_real_inertia_frame_resets_stability_before_scanning(self):
        moving=np.asarray(Image.open(FIXTURES/'moving.png').convert('RGB'))
        stopped=np.asarray(Image.open(FIXTURES/'settled.png').convert('RGB'))
        key1=RestaurantVisual.list_key(moving);key2=RestaurantVisual.list_key(stopped)
        self.assertFalse(scan.same_list(key1,key2))
        for delta,key,frame,expected in [(0,key1,moving,False),(1,key1,moving.copy(),False),
                (2,key2,stopped,False),(3,key2,stopped.copy(),False),(4,key2,stopped.copy(),True)]:
            self.now=100+delta
            self.assertEqual(scan.settled(self.task,key,frame),expected)

    def test_reused_frame_and_slow_drift_cannot_build_still_proof(self):
        frame=self.frame();key=np.arange(25,dtype=np.uint8).reshape(5,5)
        self.assertFalse(scan.settled(self.task,key,frame))
        self.now+=3
        self.assertFalse(scan.settled(self.task,key,frame))
        for index in range(1,8):
            self.now+=1
            self.assertFalse(scan.settled(self.task,key+index,self.frame()))

    def test_scroll_is_short_slow_and_input_rechecks_current_position(self):
        self.list_phase([]);self.cycle()
        action=self.issued[-1]
        self.assertTrue(action['kind'].startswith('scroll:'))
        self.assertEqual([action['x'],action['y'],*action['swipe']],[450,400,270,400,1500])
        self.task.visual.list_key.return_value=np.full((5,5),150,dtype=np.uint8)
        self.assertFalse(self.task.validate(action,self.frame()))

    def test_reaching_end_with_eight_unique_dishes_triggers_backscan(self):
        self.list_phase([])
        self.task.journal.update(json.loads((FIXTURES/'coverage-eight.json').read_text(encoding='utf-8')))
        self.task.visual.art_matches.side_effect=RestaurantVisual.art_matches
        self.task.journal['scrolls']=5
        deadline=self.task.journal['deadline']
        self.task.journal['attempts']['old-input']=2
        self.cycle()
        self.assertEqual(self.task.journal['atlas_scan']['direction'],'backward')
        self.assertEqual(self.task.journal['categories'],[])
        self.assertEqual(self.issued,[])
        self.cycle()
        action=self.issued[-1]
        self.assertTrue(action['kind'].endswith(':back'))
        self.assertEqual([action['x'],*action['swipe']],[270,450,400,1500])
        self.assertEqual(self.task.journal['attempts']['old-input'],2)
        self.assertEqual(self.task.journal['deadline'],deadline)

    def test_backscan_can_select_previously_missing_left_dish(self):
        self.list_phase([]);self.task.journal['scrolls']=5;self.cycle()
        self.task.visual.cards.return_value=[dict(point=(315,398),level=1,art=self.art(17))]
        self.cycle()
        self.assertEqual(self.task.phase,'detail')
        self.assertEqual(self.issued[-1]['x'],315)
        self.assertEqual(self.task.journal['card']['level'],1)

    def test_bounded_backscan_finishes_incomplete_without_sixth_input(self):
        self.list_phase([]);self.task.journal['scrolls']=5;self.cycle()
        self.task.journal['atlas_scan']['back_steps']=scan.BACK_LIMIT
        for _ in range(5):
            self.step()
            if self.task.phase!='list':break
        self.assertEqual(self.task.phase,'category_next')
        self.assertEqual(self.task.journal['categories'],[])
        self.assertEqual(len(self.task.journal['incomplete_categories']),1)
        self.assertEqual(self.issued,[])
        self.task._finish(True)
        self.assertFalse(self.task.done)
        self.assertFalse(self.task.journal['success'])

    def test_nine_unique_verified_dishes_complete_without_backscan(self):
        self.list_phase([])
        self.task.journal['visited']=[dict(category=0,level=20,art=pack(self.art(i))) for i in range(9)]
        self.cycle()
        self.assertEqual(self.task.phase,'category_next')
        self.assertEqual(self.task.journal['categories'],[0])
        self.assertEqual(self.issued,[])

    def test_unstable_frames_do_not_mark_maxed_cards_visited(self):
        self.list_phase([dict(point=(200,398),level=20,art=self.art(1))])
        for i in range(6):
            self.task.visual.list_key.return_value=self.art(i)[:,:,0]
            self.step()
        self.assertEqual(self.task.journal['visited'],[])
        self.assertEqual(self.issued,[])


if __name__=='__main__':unittest.main()
