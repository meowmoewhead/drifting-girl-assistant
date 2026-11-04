"""A single cast has transient outcomes, then one persistent RANK advance."""
from copy import deepcopy
import json
from unittest.mock import patch
import unittest

from tests import test_stone_farm as base


class FirstResultContinuityTests(unittest.TestCase):
    setUpClass=classmethod(base.StoneFarmTests.setUpClass.__func__)
    setUp=base.StoneFarmTests.setUp
    tearDown=base.StoneFarmTests.tearDown
    act=base.StoneFarmTests.act
    begin=base.StoneFarmTests.begin

    def prepare(self):
        frame=base.image('tests/fixtures/stone-first-result/rank1-20261017.png')
        self.begin(cycle=True)
        self.w.journal.update(entered_verified_at=self.now,first_deadline=self.now+60,
            first_selected_rank=0,first_signature=self.w.signature(frame),
            attempts={'first-cast':1},issued_inputs=[{'kind':'first-cast'}])
        self.w._set('first_verify',15)
        return frame,deepcopy(self.w.journal)

    def observe(self,frame,seconds=1.6):
        self.now+=seconds
        return self.w.observe_first_cast(frame.copy())

    def test_real_result_then_fresh_same_context_frames_confirm_the_cast(self):
        frame,old=self.prepare()
        obscured=frame.copy()
        # Synthetic flying-reward occlusion of two controls: it cannot count
        # as an outcome. After a real result, the fixed bright controls and
        # fresh header can independently confirm the same stopped context.
        obscured[1135:1200,320:385]=0
        obscured[1143:1190,20:75]=0
        self.assertIsNone(self.w._first_result_screen(obscured))
        self.assertIsNotNone(self.w._first_result_context(obscured,None))
        self.assertFalse(self.observe(frame))
        self.assertFalse(self.observe(obscured,.3))
        self.assertEqual(self.w.seen,2)
        self.assertFalse(self.observe(frame,1.3))
        self.assertFalse(self.observe(obscured,.3))
        self.assertTrue(self.observe(frame,1.3))
        self.assertEqual(self.w.phase,'route_open')
        self.assertEqual(self.w.journal['first_cast_verified_rank'],1)
        for key in ('deadline','first_deadline','started_at','attempts','issued_inputs'):
            self.assertEqual(self.w.journal[key],old[key])
        self.assertEqual(self.actions,[])

    def test_logged_battle_to_next_rank_does_not_discard_same_cast_evidence(self):
        frame,_=self.prepare()
        current=self.w.screen(frame)
        # Replay the observed 04:19:53 -> 59 -> 60 positive sequence. This
        # checks vote transitions, not an invented battle image.
        for offset,rank,expected in ((6.57,0,False),(11.89,1,False),(13.67,1,True)):
            self.now=1000.+offset
            with patch.object(self.w,'_first_result_screen',return_value={**current,'rank':rank}):
                self.assertEqual(self.w.observe_first_cast(frame.copy()),expected)
        self.assertEqual(self.actions,[])

    def test_known_popup_dark_changed_world_rank_location_and_running_reset_votes(self):
        frame,_=self.prepare()
        popup=base.image('tests/fixtures/stone-first-result/popup-20261017.png')
        changed=frame.copy();changed[12:34,286:434]=255
        for invalid in (popup,frame//2,changed):
            self.prepare();self.observe(frame);self.observe(frame)
            self.assertFalse(self.observe(invalid,.1));self.assertEqual(self.w.seen,0)
        for owner,name,value in ((self.w,'_parallel',False),(self.w,'_off_icon',False),
                                  (self.w,'_stage',8),(self.w,'_rank_value',38)):
            self.prepare();self.observe(frame);self.observe(frame)
            with patch.object(owner,name,return_value=value):
                self.assertFalse(self.observe(frame,.1));self.assertEqual(self.w.seen,0)
        self.assertEqual(self.actions,[])

    def test_same_capture_cannot_vote_and_non_results_need_a_prior_real_result(self):
        frame,_=self.prepare()
        with patch.object(self.w,'_first_result_screen',return_value=None):
            for _ in range(3):self.assertFalse(self.observe(frame))
        self.assertEqual(self.w.seen,0)
        frame,_=self.prepare();self.now+=1
        self.assertFalse(self.w.observe_first_cast(frame))
        self.now+=4
        self.assertFalse(self.w.observe_first_cast(frame));self.assertEqual(self.w.seen,1)
        with patch.object(self.w,'_first_result_screen',return_value=None):
            self.observe(frame,5)
        self.assertEqual(self.w.seen,1)  # Long gap restarts the context consensus.
        self.assertEqual(self.actions,[])

    def test_dial_auto_animation_does_not_disprove_an_already_issued_cast(self):
        frame,old=self.prepare()
        with patch.object(self.w.fishing,'adaptive_auto_on',return_value=False):
            self.assertFalse(self.w.fishing._main_visible(frame))
            self.assertIsNotNone(self.w._first_result_context(frame,self.w._first_result_screen(frame)))
            self.assertFalse(self.observe(frame))
            self.assertFalse(self.observe(frame))
            self.assertTrue(self.observe(frame))
        self.assertEqual(self.w.phase,'route_open')
        for key in ('deadline','first_deadline','started_at','attempts','issued_inputs'):
            self.assertEqual(self.w.journal[key],old[key])
        self.assertEqual(self.actions,[])

    def test_actual_timeout_preparation_counts_only_after_a_real_result(self):
        _,old=self.prepare()
        frame=base.image('tests/fixtures/stone-first-result/timeout-rank0-20261019.png')
        self.w.journal['first_signature']=self.w.signature(frame)
        self.assertIsNone(self.w._first_result_screen(frame))
        for _ in range(3):self.assertFalse(self.observe(frame))
        self.assertEqual(self.w.seen,0)
        # The 06:28 log has actual RANK0 battle/success results at 46.594
        # and 47.811. This replays that recorded result, not an invented image.
        screen=self.w.screen(frame)
        with patch.object(self.w,'_first_result_screen',return_value=screen):
            self.assertFalse(self.observe(frame))
        self.assertFalse(self.observe(frame))
        self.assertTrue(self.observe(frame))
        self.assertEqual(self.w.journal['first_cast_verified_rank'],0)
        self.assertEqual(self.w.phase,'route_open')
        for key in ('deadline','first_deadline','started_at','attempts','issued_inputs'):
            self.assertEqual(self.w.journal[key],old[key])
        self.assertEqual(self.actions,[])

    def test_next_cast_starts_with_new_result_identity(self):
        frame,_=self.prepare();self.observe(frame)
        self.assertEqual(self.w.first_result_rank,1)
        self.w.phase='first_idle'
        self.w._set('first_verify',15)
        self.assertIsNone(self.w.first_result_rank)
        self.assertIsNone(self.w.first_result_frame)
        self.assertFalse(self.w.first_result_observed)
        self.assertEqual(self.w.seen,0)

    def test_missing_selected_task_identity_cannot_credit_rank_progress(self):
        frame,_=self.prepare();self.w.journal.pop('first_signature')
        for _ in range(3):self.assertFalse(self.observe(frame))
        self.assertEqual(self.w.phase,'first_verify')
        self.assertEqual(self.w.seen,0)
        self.assertEqual(self.actions,[])

    def test_expiry_after_ocr_or_before_read_cannot_commit_or_extend(self):
        for slow in (False,True):
            frame,_=self.prepare();self.observe(frame);self.observe(frame)
            raw=self.path.read_bytes()
            if slow:
                original=self.w._first_result_context
                def context(*args):
                    result=original(*args);self.now=self.w.journal['phase_deadline']+1
                    return result
                with patch.object(self.w,'_first_result_context',side_effect=context):
                    self.assertFalse(self.observe(frame))
            else:
                self.now=self.w.journal['phase_deadline']+1
                self.assertFalse(self.observe(frame))
            self.assertEqual(self.path.read_bytes(),raw)
            self.assertEqual(self.actions,[])


if __name__=='__main__':unittest.main()
