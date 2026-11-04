"""Actual missed-map frame, bounded retries and fresh-input rejection."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2

from boss_defense import BossDefense
from digit_ocr import DigitReader
from parallel_world import ParallelWorld
from training_sailing import TrainingSailing

ROOT=Path(__file__).resolve().parents[1]
FRAMES=ROOT/'tests/fixtures/parallel'

def frame(name):
    return cv2.cvtColor(cv2.imread(str(FRAMES/(name+'.png'))),cv2.COLOR_BGR2RGB)

class NavigationRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.w=ParallelWorld(ROOT,lambda *a:None,lambda *a:None,lambda _:None,
            DigitReader(ROOT),training_state_path=Path(self.temp.name)/'training.json')
        self.w.sailing=TrainingSailing(ROOT,self.w.training,lambda *a:False,lambda _:None)
        self.w.defense=BossDefense(ROOT,self.w.training,None,lambda _:None)
        self.w.rank=3
        self.w.phase='play'
        self.current=frame('map-open-missed-rank3')
        self.actions=[]
        self.w.portal_action=self.action

    def tearDown(self):self.temp.cleanup()

    def action(self,action):
        accepted=self.w.validate_portal(action,self.current)
        if accepted:self.actions.append(action)
        return accepted

    def test_actual_missed_map_frame_is_stopped_rank3_main_with_arrow(self):
        self.assertTrue(self.w._next_map_ready(self.current))
        self.assertEqual(self.w._rank_value(self.current),3)
        self.assertFalse(self.w._map(self.current))
        self.assertIs(self.w.sailing.running(self.current),False)

    def test_missed_open_retries_at_most_three_without_resetting_deadline(self):
        w=self.w
        for stamp in (100,110,120,129):
            with patch('parallel_world.time.monotonic',return_value=stamp):
                w.next_action_at=0
                self.assertTrue(w.step(self.current))
            self.assertEqual(w.phase_since,100)
        self.assertEqual(len(self.actions),3)
        self.assertEqual(w.map_open_attempts,3)
        with patch('parallel_world.time.monotonic',return_value=131):
            w.next_action_at=0
            with self.assertRaisesRegex(RuntimeError,'下一任务地图.*30 秒'):
                w.step(self.current)
        self.assertEqual(len(self.actions),3)

    def test_fresh_frame_rejection_does_not_count_or_reset_phase(self):
        w=self.w
        w.phase,w.phase_since,w.map_open_attempts='find_boss',100,1
        observed=self.current.copy()
        self.current[12:34,286:434]=255
        with patch('parallel_world.time.monotonic',return_value=120):
            self.assertTrue(w._open_next_map(observed,'retry'))
        self.assertEqual(self.actions,[])
        self.assertEqual(w.map_open_attempts,1)
        self.assertEqual(w.phase_since,100)

    def test_actual_map_stops_retries_and_selects_task(self):
        self.w.phase='find_boss'
        self.current=frame('parallel-map-rank2')
        self.w.rank=2
        self.w.step(self.current)
        self.assertEqual(self.w.phase,'close_boss_map')
        self.assertEqual(self.actions,[])

    def test_unknown_world_dimmed_arrow_footer_and_rank40_refuse(self):
        w=self.w
        good=self.current.copy()
        for box in ((13,46,39,72),(307,64,398,127),(539,1137,600,1186)):
            bad=good.copy();x,y,X,Y=box;bad[y:Y,x:X]=0
            self.assertFalse(w._next_map_ready(bad),box)
        self.assertFalse(w._next_map_ready((good*.6).astype('uint8')))
        w.rank=40
        self.assertFalse(w._next_map_ready(good))
        w.rank=3;w.completed_rank_40=True
        self.assertFalse(w._next_map_ready(good))

    def test_new_search_resets_old_attempts_but_same_phase_does_not(self):
        self.w.map_open_attempts=3
        self.w._set('find_boss')
        self.assertEqual(self.w.map_open_attempts,0)
        self.w.map_open_attempts=2
        started=self.w.phase_since
        self.w._set('find_boss')
        self.assertEqual(self.w.map_open_attempts,2)
        self.assertEqual(self.w.phase_since,started)

    def test_running_or_unknown_sailing_and_battle_refuse(self):
        for running in (True,None):
            with patch.object(self.w.sailing,'running',return_value=running):
                self.assertFalse(self.w._next_map_ready(self.current))
        for marker in ('timer','swords'):
            with patch.object(self.w.defense,'_match',side_effect=lambda f,n,*a:n==marker):
                self.assertFalse(self.w._next_map_ready(self.current))

    def test_input_time_deadline_rank_and_location_checks(self):
        w=self.w
        action=dict(kind='next-map',rank=3,signature=w._return_signature(self.current))
        w.phase,w.phase_since='find_boss',100
        with patch('parallel_world.time.monotonic',return_value=130):
            self.assertTrue(w.validate_portal(action,self.current))
        with patch('parallel_world.time.monotonic',return_value=130.01):
            self.assertFalse(w.validate_portal(action,self.current))
        with patch('parallel_world.time.monotonic',return_value=110):
            action['rank']=2
            self.assertFalse(w.validate_portal(action,self.current))
            action['rank']=3
            with patch.object(w,'_rank_value',return_value=None):
                self.assertFalse(w.validate_portal(action,self.current))

if __name__=='__main__':unittest.main()
