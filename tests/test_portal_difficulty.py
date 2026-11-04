"""Replay actual stage-8/9 menus and guard the bounded NPC retry."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app import Clicker
from boss_training import training_options
from boss_defense import BossDefense
from digit_ocr import DigitReader, NumberReading
from parallel_world import ParallelWorld
from training_sailing import TrainingSailing


ROOT=Path(__file__).resolve().parents[1]


def frame(name):
    paths={'active9':ROOT/'assets/recovery/parallel-manager.png',
           'question8':ROOT/'assets/recovery/parallel-lower-question.png',
           'npc':ROOT/'assets/recovery/observatory-npc.png',
           'inactive9':ROOT/'tests/fixtures/parallel/gate-inactive-tier9.png',
           'active8':ROOT/'tests/fixtures/parallel/gate-active-tier8.png',
           'parallel':ROOT/'tests/fixtures/parallel/parallel-main.png'}
    return cv2.cvtColor(cv2.imread(str(paths[name])),cv2.COLOR_BGR2RGB)


class PortalDifficultyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.actions=[]
        self.w=ParallelWorld(ROOT,lambda *a:None,lambda *a:None,lambda _:None,
            DigitReader(ROOT),training={'parallel_stage_limit':8},
            training_state_path=Path(self.temp.name)/'training_state.json')
        self.w.sailing=TrainingSailing(ROOT,self.w.training,lambda *a:False,lambda _:None)
        self.w.defense=BossDefense(ROOT,self.w.training,None,lambda _:None)
        self.w.portal_action=self.act

    def tearDown(self):
        self.temp.cleanup()

    def act(self,action):
        self.assertTrue(self.w.validate_portal(action,self.current),action)
        self.actions.append(deepcopy(action))
        return True

    def step(self,name):
        self.current=frame(name)
        self.w.next_action_at=0
        return self.w.step(self.current)

    def test_actual_active_inactive_stage_and_lower_question_are_distinct(self):
        for name,active,stage in [('active9',True,9),('active8',True,8),('inactive9',False,9)]:
            self.assertIs(self.w._gate_manager(frame(name)),active)
            self.assertEqual(self.w._gate_stage(frame(name)),stage)
            self.assertIsNone(self.w._lower_prompt(frame(name)))
        self.assertEqual(self.w._lower_prompt(frame('question8')),8)
        self.assertIsNone(self.w._gate_stage(frame('question8')))
        for name in ('npc','parallel'):
            self.assertIsNone(self.w._gate_manager(frame(name)))
        for name in ('active9','inactive9','question8'):
            self.assertIsNone(self.w._gate_stage(frame(name)//2))
            self.assertIsNone(self.w._lower_prompt(frame(name)//2))

    def test_new_gate_is_lowered_once_and_verified_before_entering(self):
        self.w.phase='verify_gate_stage'
        before=deepcopy(self.w.training.state)
        for _ in range(2):self.step('active9')
        self.assertEqual(self.actions,[])
        self.step('active9')
        self.assertEqual(self.w.phase,'lower_stage_confirm')
        self.assertEqual([(a['kind'],a['stage']) for a in self.actions],[('lower',9)])
        for _ in range(3):self.step('question8')
        self.assertEqual(self.w.phase,'await_lower_stage')
        self.assertEqual([(a['kind'],a['stage']) for a in self.actions],[('lower',9),('lower-confirm',8)])
        for _ in range(5):self.step('active9')
        self.assertEqual(len(self.actions),2)
        self.assertEqual(self.w.phase,'await_lower_stage')
        for _ in range(3):self.step('active8')
        self.assertEqual(self.w.phase,'verify_gate_stage')
        self.step('active8')
        self.assertEqual(self.w.phase,'close_gate_panel')
        self.step('active8')
        self.assertEqual(self.w.phase,'enter_gate')
        self.assertEqual(self.actions[-1]['kind'],'panel-close')
        self.assertEqual(self.w.training.state,before)
        self.assertFalse(self.w.completed_rank_40)
        self.assertEqual(self.w.rounds,0)

    def test_at_or_below_limit_and_default_zero_never_lower(self):
        for limit in (0,8,9):
            self.w.training.options['parallel_stage_limit']=limit
            self.w.phase='verify_gate_stage'
            self.w.gate_stage_readings.clear()
            self.actions.clear()
            for _ in range(3):self.step('active8')
            self.assertEqual(self.w.phase,'close_gate_panel')
            self.assertEqual(self.actions,[])

    def test_fresh_changed_stage_cannot_submit_an_old_lowering_action(self):
        self.w.phase='verify_gate_stage'
        self.w.portal_action=lambda action:self.w.validate_portal(action,frame('active8'))
        for _ in range(3):self.step('active9')
        self.assertEqual(self.w.phase,'verify_gate_stage')
        self.assertIsNone(self.w.lower_from)
        self.assertEqual(self.actions,[])
        self.assertFalse(self.w.validate_portal(dict(kind='lower',stage=9),frame('question8')))

    def test_wrong_prompt_number_and_unknown_frame_keep_original_deadline(self):
        self.w.phase='lower_stage_confirm';self.w.phase_since=100;self.w.lower_from=9
        class Reader:
            def read(self,_):return NumberReading('7',7,.999)
        self.w.reader=Reader()
        with patch('parallel_world.time.monotonic',return_value=105):
            for _ in range(4):self.step('question8')
        self.assertEqual(self.actions,[])
        self.assertEqual(self.w.phase_since,100)
        with patch('parallel_world.time.monotonic',return_value=131):
            with self.assertRaisesRegex(RuntimeError,'确认降低.*30 秒'):
                self.step('npc')
        self.assertEqual(self.actions,[])

    def test_resume_inactive_manager_activates_and_keeps_limit(self):
        self.step('inactive9')
        self.assertEqual(self.actions[0]['kind'],'activate')
        self.assertEqual(self.w.phase,'search_confirm')
        self.assertEqual(self.w.training.options['parallel_stage_limit'],8)
        self.assertFalse(self.w.completed_rank_40)
        self.assertIsNone(self.w.rank)

    def test_cat_retry_has_three_frames_two_retries_and_unchanged_timeout(self):
        self.w.phase='cat_menu';self.w.phase_since=100;self.w.cat_attempts=1
        self.w.rank=40;self.w.completed_rank_40=True
        with patch('parallel_world.time.monotonic',return_value=110):
            for _ in range(12):self.step('npc')
        self.assertEqual([a['kind'] for a in self.actions],['npc','npc'])
        self.assertEqual(self.w.cat_attempts,3)
        self.assertEqual(self.w.phase_since,100)
        with patch('parallel_world.time.monotonic',return_value=131):
            with self.assertRaisesRegex(RuntimeError,'管理菜单.*30 秒'):
                self.step('npc')
        self.assertEqual(len(self.actions),2)

    def test_npc_guard_rejects_running_dim_and_other_world(self):
        a=frame('npc')
        self.assertTrue(self.w._npc_ready(a))
        for other in (a//2,frame('parallel'),frame('active9')):
            self.assertFalse(self.w.validate_portal(dict(kind='npc'),other))
        with patch.object(self.w.sailing,'running',return_value=True):
            self.assertFalse(self.w._npc_ready(a))
        with patch.object(self.w.defense,'_match',return_value=True):
            self.assertFalse(self.w._npc_ready(a))
        changed=a.copy();changed[515:624,460:507]=0
        self.assertFalse(self.w._npc_ready(changed))

    def test_actual_daytime_npc_with_changing_speech_keeps_original_guards(self):
        paths=[ROOT/'assets/recovery/observatory-npc-day.png',
               ROOT/'tests/fixtures/parallel/observatory-day-speech.png',
               ROOT/'tests/fixtures/parallel/observatory-day-speech-second.png']
        for path in paths:
            a=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
            self.assertTrue(self.w._npc_ready(a),path.name)
            self.assertTrue(self.w.validate_portal(dict(kind='npc'),a))
            self.assertFalse(self.w._npc_ready(a//2))
            changed=a.copy();changed[515:624,460:507]=0
            self.assertFalse(self.w._npc_ready(changed))
            with patch.object(self.w.sailing,'running',return_value=True):
                self.assertFalse(self.w._npc_ready(a))
            with patch.object(self.w.defense,'_match',return_value=True):
                self.assertFalse(self.w._npc_ready(a))

    def test_conflicting_stage_crops_do_not_lower(self):
        class Reader:
            def read(self,crop):
                n=9 if crop.shape[1]==85 else 8
                return NumberReading(str(n),n,.999)
        self.w.reader=Reader()
        self.w.phase='verify_gate_stage'
        for _ in range(4):self.step('active9')
        self.assertEqual(self.actions,[])
        self.assertFalse(self.w.gate_stage_verified)

    def test_stage_limit_is_strict_and_other_options_are_preserved(self):
        original=training_options(dict(battle_tap=True,boss_interrupt=False))
        wanted=training_options({**original,'parallel_stage_limit':8})
        self.assertEqual({k:v for k,v in wanted.items() if k!='parallel_stage_limit'},
                         {k:v for k,v in original.items() if k!='parallel_stage_limit'})
        for invalid in (-1,101,8.5,True,'8'):
            with self.assertRaises(ValueError):training_options(dict(parallel_stage_limit=invalid))

    def replay_worker(self,expire=False,changed=False):
        manager=frame('inactive9')
        clock=[0.]
        actions=[]
        class Device:
            def __init__(self,config,stop):
                self.stop=stop;self.io_lock=threading.RLock();self.count=0
            def screenshot(self):
                self.count+=1
                clock[0]+= .5
                if self.count==2 and expire:clock[0]=31
                if self.count%2==0 and changed:return np.zeros_like(manager)
                return manager.copy()
            def click(self,button):
                actions.append(((button.button[0]+button.button[2])//2,
                                (button.button[1]+button.button[3])//2))
            def command(self,*args):raise AssertionError('No extra game or system input')
            def stuck_record_add(self,_):pass
        root=Path(self.temp.name)/'worker';root.mkdir()
        shutil.copytree(ROOT/'assets',root/'assets')
        config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        config['rules']=[deepcopy(r) for r in config['rules'] if r['id']=='driftinggirl_parallel_world']
        config['rules'][0].update(enabled=True,training=training_options())
        config['poll_interval']=.1
        (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
        app=Clicker(root,Device);app.digit_reader=DigitReader(ROOT)
        try:
            with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0],time=lambda:clock[0])), \
                 patch('parallel_world.time',SimpleNamespace(monotonic=lambda:clock[0])):
                app.start(preview=False,max_clicks=1,timeout=30)
                app.thread.join(10)
            self.assertFalse(app.running())
            if expire or changed:self.assertEqual(actions,[])
            else:
                self.assertEqual(actions,[(355,803)])
                self.assertEqual(app.state['stop_reason'],'click_limit')
            self.assertFalse(app.state['parallel']['completed_rank_40'])
        finally:
            app.stop()
            if app.thread:app.thread.join(3)
            for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

    def test_worker_portal_click_keeps_global_single_input_limit(self):self.replay_worker()

    def test_worker_deadline_expiring_during_capture_blocks_input(self):self.replay_worker(expire=True)

    def test_worker_changed_fresh_frame_blocks_activation(self):self.replay_worker(changed=True)


if __name__=='__main__':unittest.main()
