"""Real restarted task sleep and independently guarded recovery inputs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
from digit_ocr import DigitReader
from parallel_world import ParallelWorld
from fishing_meter import FishingMeter
from training_sailing import TrainingSailing
from boss_defense import BossDefense

ROOT=Path(__file__).resolve().parents[1]

class QuestRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.w=ParallelWorld(ROOT,lambda *a:None,lambda *a:None,lambda _:None,
            DigitReader(ROOT),training_state_path=Path(self.temp.name)/'training.json')
        w=self.w
        w.fishing=FishingMeter(ROOT,w.training,None,lambda _:None)
        w.sailing=TrainingSailing(ROOT,w.training,None,lambda _:None)
        w.defense=BossDefense(ROOT,w.training,None,lambda _:None)
        w.portal_action=lambda a:False
        w.phase='play';w.rank=3
        self.frame=cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/parallel/quest-sleep-rank3.png')),cv2.COLOR_BGR2RGB)

    def tearDown(self):self.temp.cleanup()

    def prepare(self,phase='quest_start'):
        w=self.w
        key=w._quest_idle_context(self.frame)
        self.assertIsNotNone(key)
        w.quest_idle_key=w.quest_recovery_key=key
        w.quest_idle_since=100;w.quest_idle_frames=3
        w.quest_recovery_since=100;w.quest_recovery_rank=3
        w.phase=phase
        return dict(kind='quest-wake',rank=3,x=360,y=1010,signature=w._return_signature(self.frame))

    def test_real_frame_is_rank3_stopped_task_sleep_without_fill(self):
        w=self.w
        self.assertEqual(w._rank_value(self.frame),3)
        self.assertIs(w.sailing.running(self.frame),False)
        self.assertEqual(w.fishing.idle_icon(self.frame),'sleep')
        self.assertFalse(w.fishing.blue_fill(self.frame))
        self.assertIsNotNone(w._quest_idle_context(self.frame))

    def test_missing_target_and_unverified_map_never_authorize_cast(self):
        action=self.prepare()
        with patch('parallel_world.time.monotonic',return_value=105):
            self.assertFalse(self.w.validate_portal(action,self.frame))
            self.w.target_rank=3
            self.assertFalse(self.w.validate_portal(action,self.frame))
            self.w.quest_map_selected=True
            self.assertTrue(self.w.validate_portal(action,self.frame))
            self.w.target_rank=None
            self.assertFalse(self.w.validate_portal(action,self.frame))

    def test_three_frames_and_three_seconds_required_even_after_map(self):
        action=self.prepare();w=self.w
        w.target_rank=3;w.quest_map_selected=True
        with patch('parallel_world.time.monotonic',return_value=102.9):
            self.assertFalse(w.validate_portal(action,self.frame))
        w.quest_idle_frames=2
        with patch('parallel_world.time.monotonic',return_value=104):
            self.assertFalse(w.validate_portal(action,self.frame))

    def test_new_world_location_digit_fill_battle_and_sailing_refuse(self):
        action=self.prepare();w=self.w
        w.target_rank=3;w.quest_map_selected=True
        with patch('parallel_world.time.monotonic',return_value=105):
            for x,y,X,Y in ((13,46,39,72),(286,12,434,34),(72,9,136,59),(258,918,275,953)):
                bad=self.frame.copy();bad[y:Y,x:X]=0
                self.assertFalse(w.validate_portal(action,bad),(x,y,X,Y))
            self.assertFalse(w.validate_portal(action,(self.frame*.6).astype('uint8')))
            for value in (True,None):
                with patch.object(w.sailing,'running',return_value=value):
                    self.assertFalse(w.validate_portal(action,self.frame))
            with patch.object(w.fishing,'blue_fill',return_value=True):
                self.assertFalse(w.validate_portal(action,self.frame))
            with patch.object(w.defense,'_match',return_value=True):
                self.assertFalse(w.validate_portal(action,self.frame))

    def test_final_boss_training_and_old_selected_rank_cannot_enter_recovery(self):
        w=self.w
        for rank in (39,40,None):
            w.rank=rank;self.assertIsNone(w._quest_idle_context(self.frame))
        w.rank=3;w.target_rank=2
        self.assertIsNone(w._quest_idle_context(self.frame))
        w.target_rank=None;w.training.state['mode']='farming'
        self.assertIsNone(w._quest_idle_context(self.frame))

    def test_actual_covered_map_requires_eye_jaw_and_current_task_context(self):
        self.prepare('quest_select');w=self.w
        f=cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/parallel/quest-map-covered-rank3.png')),cv2.COLOR_BGR2RGB)
        self.assertTrue(w._quest_map_ready(f))
        self.assertIsNone(w._match(f,'map-boss',(20,175,700,1115),.88))
        target=w._quest_task_target(f)
        self.assertIsNotNone(target)
        self.assertEqual(target[:2],(377,619))
        w.quest_map_target=target[:2];w.quest_map_frames=3
        action=dict(kind='quest-select',x=377,y=619,rank=3,signature=w._return_signature(f))
        with patch('parallel_world.time.monotonic',return_value=105):
            self.assertTrue(w.validate_portal(action,f))
            for x,y,X,Y in ((385,613,393,630),(366,630,391,636),(248,32,295,81),(632,1151,705,1225)):
                bad=f.copy();bad[y:Y,x:X]=0
                self.assertFalse(w.validate_portal(action,bad),(x,y,X,Y))
            w.quest_map_frames=2
            self.assertFalse(w.validate_portal(action,f))
            w.quest_map_frames=3;action['x']=360
            self.assertFalse(w.validate_portal(action,f))

    def test_map_input_rejection_does_not_create_selected_task(self):
        self.prepare('quest_select');w=self.w
        f=cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/parallel/quest-map-covered-rank3.png')),cv2.COLOR_BGR2RGB)
        w.quest_map_frames=3;w.quest_map_target=(377,619)
        with patch('parallel_world.time.monotonic',return_value=105):
            w._quest_step(f)
        self.assertIsNone(w.target_rank)
        self.assertFalse(w.quest_map_selected)
        self.assertEqual(w.phase,'quest_select')

    def test_input_deadline_and_once_per_rank_cannot_be_reset_by_retries(self):
        action=self.prepare();w=self.w
        w.target_rank=3;w.quest_map_selected=True
        with patch('parallel_world.time.monotonic',return_value=161):
            self.assertFalse(w.validate_portal(action,self.frame))
        with patch('parallel_world.time.monotonic',return_value=105):
            w.quest_cast_ranks.add(3)
            self.assertFalse(w.validate_portal(action,self.frame))
        w.phase='quest_map';w.phase_since=100;w.next_action_at=0
        with patch('parallel_world.time.monotonic',return_value=161):
            with self.assertRaisesRegex(RuntimeError,'60 秒'):
                w.step(self.frame)
        self.assertEqual(w.quest_recovery_since,100)

    def test_sleep_or_input_count_does_not_prove_recovery(self):
        self.prepare('quest_verify')
        self.assertFalse(self.w._quest_progress(self.frame))
        self.assertEqual(self.w.quest_recoveries,0)
        self.w.phase_since=100
        with patch('parallel_world.time.monotonic',return_value=116):
            with self.assertRaisesRegex(RuntimeError,'15 秒'):
                self.w._quest_step(self.frame)

    def test_new_round_clears_attempt_guard_but_preserves_training_journal(self):
        self.w.quest_cast_ranks.add(3)
        self.w.quest_wake_ranks.add(3)
        self.w._entered_parallel()
        self.assertEqual(self.w.quest_cast_ranks,set())
        self.assertEqual(self.w.quest_wake_ranks,set())
        self.assertIsNone(self.w.quest_idle_key)

    def test_real_idle_collection_reuses_only_this_steps_rank_reading(self):
        w=self.w
        with patch.object(w,'_rank_value',wraps=w._rank_value) as read:
            for stamp in (100,102,104):
                fresh=self.frame.copy()
                with patch('parallel_world.time.monotonic',return_value=stamp):
                    w.observe(fresh)
                    w.step(fresh)
            self.assertEqual(read.call_count,3)
        self.assertEqual(w.phase,'quest_map')
        self.assertIsNone(w.target_rank)

    def test_obscuring_panel_breaks_idle_proof_before_controller_waits(self):
        self.prepare();w=self.w
        w.phase='play'
        obscured=self.frame.copy();obscured[780:1120]=0
        w.observe(obscured)
        self.assertIsNone(w.quest_idle_key)
        self.assertEqual(w.quest_idle_frames,0)

    def load(self,name):
        return cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/parallel'/name)),cv2.COLOR_BGR2RGB)

    def test_real_sleep_needs_separate_wake_then_three_new_hook_frames(self):
        self.prepare();w=self.w
        w.target_rank=3;w.quest_map_selected=True
        issued=[]
        current=[self.frame]
        def accept(action):
            self.assertTrue(w.validate_portal(action,current[0]))
            issued.append(action['kind']);return True
        w.portal_action=accept
        with patch('parallel_world.time.monotonic',return_value=104):
            w._quest_step(self.frame)
        self.assertEqual(issued,['quest-wake'])
        self.assertEqual(w.phase,'quest_awake')
        self.assertEqual(w.quest_cast_ranks,set())
        with patch('parallel_world.time.monotonic',return_value=105):
            w._observe_quest_idle(self.frame)
            w._quest_step(self.frame)
        self.assertEqual(w.quest_idle_frames,0)
        hook=self.load('quest-hook-rank3.png');current[0]=hook
        self.assertEqual(w.fishing.idle_icon(hook),'hook')
        for stamp in (106,107,109):
            with patch('parallel_world.time.monotonic',return_value=stamp):
                w._observe_quest_idle(hook.copy())
                w._quest_step(hook)
        self.assertEqual(issued,['quest-wake','quest-cast'])
        self.assertEqual(w.phase,'quest_verify')
        self.assertEqual(w.quest_wake_ranks,{3})
        self.assertEqual(w.quest_cast_ranks,{3})
        self.assertFalse(w._quest_progress(hook))
        battle=self.load('quest-battle-rank3.png')
        self.assertTrue(w._quest_progress(battle))
        self.assertEqual(w.quest_recoveries,1)

    def test_hook_cast_cannot_use_sleep_or_unverified_awake_frames(self):
        action=self.prepare();w=self.w
        w.target_rank=3;w.quest_map_selected=True
        action['kind']='quest-cast'
        with patch('parallel_world.time.monotonic',return_value=105):
            self.assertFalse(w.validate_portal(action,self.frame))
            hook=self.load('quest-hook-rank3.png')
            self.assertTrue(w.validate_portal(action,hook))
            w.phase='quest_awake';w.quest_idle_frames=2
            self.assertFalse(w.validate_portal(action,hook))
            w.quest_idle_frames=3
            self.assertTrue(w.validate_portal(action,hook))
            action['kind']='quest-wake'
            self.assertFalse(w.validate_portal(action,hook))
            w.phase='quest_start';w.quest_wake_ranks.add(3)
            self.assertFalse(w.validate_portal(action,self.frame))

    def test_awake_deadline_and_rejected_wake_keep_original_budget(self):
        self.prepare();w=self.w
        w.target_rank=3;w.quest_map_selected=True
        with patch('parallel_world.time.monotonic',return_value=105):
            w._quest_step(self.frame)
        self.assertEqual(w.quest_wake_ranks,set())
        self.assertEqual(w.phase,'quest_start')
        w.phase='quest_awake';w.phase_since=100
        with patch('parallel_world.time.monotonic',return_value=116):
            with self.assertRaisesRegex(RuntimeError,'15 秒'):
                w._quest_step(self.frame)
        self.assertEqual(w.quest_recovery_since,100)

    def test_map_step_reads_one_rank_but_fresh_input_reads_again(self):
        self.prepare('quest_select');w=self.w
        f=self.load('quest-map-covered-rank3.png')
        w.quest_map_target=(377,619);w.quest_map_frames=3
        action=dict(kind='quest-select',rank=3,x=377,y=619,signature=w._return_signature(f))
        with patch.object(w,'_rank_value',wraps=w._rank_value) as read:
            with patch('parallel_world.time.monotonic',return_value=105):
                w.step(f)
                self.assertEqual(read.call_count,1)
                self.assertTrue(w.validate_portal(action,f.copy()))
                self.assertEqual(read.call_count,2)

    def test_waiting_cast_does_not_repeat_header_ocr_or_claim_recovery(self):
        self.prepare('quest_verify');w=self.w
        hook=self.load('quest-hook-rank3.png')
        with patch.object(w,'_rank_value',wraps=w._rank_value) as read:
            self.assertFalse(w._quest_progress(hook))
            self.assertEqual(read.call_count,0)
            self.assertTrue(w._quest_progress(self.load('quest-battle-rank3.png')))
            self.assertEqual(read.call_count,1)

if __name__=='__main__':unittest.main()
