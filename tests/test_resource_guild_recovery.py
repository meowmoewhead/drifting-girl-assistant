"""Replay the RANK37 resource battle interrupting the first guild entry."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import Clicker
from digit_ocr import DigitReader
from drift.tasks.features import FEATURE_BY_ID
from parallel_world import ParallelWorld

ROOT = Path(__file__).resolve().parents[1]


class ResourceGuildRecoveryTests(unittest.TestCase):
    feature = 'guild_shop'
    prefix = 'driftinggirl_guild_'
    main_asset = 'assets/guild-shop/main.png'
    shop_asset = 'assets/guild-shop/open.png'
    battle_fixture = 'tests/fixtures/resource-guild/battle.png'
    next_suffix = 'special'
    entry_interval = 3600
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.reader = DigitReader(ROOT)
        cls.battle = np.asarray(Image.open(ROOT/cls.battle_fixture).convert('RGB'))
        cls.main = np.asarray(Image.open(ROOT/cls.main_asset).convert('RGB'))
        cls.shop = np.asarray(Image.open(ROOT/cls.shop_asset).convert('RGB'))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='resource-guild-')
        self.addCleanup(self.directory.cleanup)
        self.actions = []
        action = lambda *args:self.actions.append(args)
        self.world = ParallelWorld(ROOT,action,action,lambda *args:None,self.reader,
            training_state_path=Path(self.directory.name)/'training_state.json',
            fishing_action=action,sailing_action=action,defense_action=action)
        self.world.rank,self.world.target_rank,self.world.phase = 37,37,'play'
        self.world.training.options.update(enabled=True,auto_sailing=True,
                                           battle_tap=False,boss_interrupt=False)
        self.world.training.state = dict(mode='farming',rank=37,boss_level=1421,
            start_level=1101,started_at=1791595022.248239,sailing_enabled=True,
            sailing_verified=True,sailing_profile='resources')

    def test_actual_current_resource_battle_proof_does_not_input_or_write_records(self):
        before = deepcopy(self.world.training.state)
        context = self.world.resource_battle_recovery_context(self.battle)
        self.assertIsNotNone(context)
        self.assertEqual(context[0],37)
        self.assertEqual(self.world.training.state,before)
        self.assertEqual(self.actions,[])
        self.assertFalse((Path(self.directory.name)/'training_state.json').exists())
        for update in (dict(rank=36),dict(mode='trial'),dict(sailing_verified=False),
                       dict(sailing_enabled=False),dict(boss_level=0)):
            self.world.training.state = {**before,**update}
            self.assertIsNone(self.world.resource_battle_recovery_context(self.battle))
        self.world.training.state = before
        uncertain = np.asarray(Image.open(ROOT/'tests/fixtures/resource-guild/arrow-unknown.png').convert('RGB'))
        self.assertIsNone(self.world.resource_battle_recovery_context(uncertain))

    def replay(self,scenario):
        config = deepcopy(self.config)
        config['poll_interval'] = .1
        config['rules'] = [r for r in config['rules'] if r['id'] in FEATURE_BY_ID[self.feature]['rules']
                           or r['id']=='driftinggirl_parallel_world']
        for rule in config['rules']:
            rule.update(enabled=True,next_delay=0)
        entry_id,next_id = self.prefix+'open',self.prefix+self.next_suffix
        entry = next(r for r in config['rules'] if r['id']==entry_id)
        entry['interval'] = .2 if scenario=='repeat' else self.entry_interval
        clock = [0.]
        world = self.world
        frames = dict(main=self.main,battle=self.battle,shop=self.shop)
        devices = []
        contexts = []
        world.training.state['sailing_verified'] = True
        training_before = deepcopy(world.training.state)
        if scenario=='unverified':world.training.state['sailing_verified'] = False

        class Device:
            def __init__(self,_config,stop):
                self.stop,self.image = stop,None
                self.taps,self.after_frames,self.total_frames = [],0,0
                devices.append(self)
            def screenshot(self):
                clock[0] += 1
                self.total_frames += 1
                if not self.taps:
                    self.image = frames['main'].copy()
                else:
                    self.after_frames += 1
                    if scenario=='expired' and self.after_frames==1:clock[0] += 31
                    if scenario=='unknown':self.image = np.zeros_like(frames['battle'])
                    elif scenario=='flicker' and self.after_frames==3:
                        self.image = (frames['battle']*.6).astype(np.uint8)
                    elif scenario in ('shop','later') and len(self.taps)==1:
                        self.image = frames['shop'].copy()
                    elif scenario in ('repeat','cooldown') and self.after_frames>=8:
                        self.image = frames['main'].copy()
                    else:self.image = frames['battle'].copy()
                    if scenario=='stopped' and self.after_frames==2:self.stop.set()
                    if self.after_frames >= (44 if scenario in
                            ('unknown','unverified','repeat','expired','later') else 16):
                        raise InterruptedError('End of bounded replay')
                return self.image
            def click(self,button):
                self.assert_safe(button)
                self.taps.append(button.name)
                self.after_frames = 0
            def assert_safe(self,button):
                if button.name==entry_id:
                    assert np.array_equal(self.image,frames['main'])
                elif button.name==next_id and scenario in ('shop','later'):
                    assert np.array_equal(self.image,frames['shop'])
                    if scenario=='shop':self.stop.set()
                else:raise AssertionError('Unexpected input: '+button.name)
            def stuck_record_add(self,_button):pass

        class Defense:
            def status(self):return {}
            def owns(self,_frame,_rank):return False
            def step(self,frame,_rank):
                return (scenario in ('preempt','flicker') and
                        world.resource_battle_recovery_context(frame) is not None)

        class Parallel:
            only_sailing_shutdown = False
            def __init__(self,*_args,**_kwargs):
                self.training,self.sailing,self.rank = world.training,world.sailing,world.rank
                self.defense,self.phase = Defense(),'play'
            def observe(self,_frame):pass
            def observe_boss_sailing_rank(self,_frame,**_kwargs):return False
            def boss_sailing_stop_requested(self,*_args):return False
            def completion_step(self,_frame,**_kwargs):return False
            def resource_battle_recovery_context(self,frame):
                context = world.resource_battle_recovery_context(frame)
                contexts.append((clock[0],context))
                if context is not None and sum(c is not None for _t,c in contexts)>=4:
                    if scenario=='boundary':clock[0] += 31
                    if scenario=='stop_during_proof':devices[0].stop.set()
                return context
            def suspend(self,_seconds):pass
            def status(self):return world.status()
            def step(self,_frame):return scenario not in ('repeat','cooldown','shop','later') and bool(devices[0].taps)

        root = Path(self.directory.name)/('runner-'+scenario)
        root.mkdir()
        assets = {r['asset'] for r in config['rules']}
        assets.update(p for r in config['rules'] for p in r.get('asset_variants',[]))
        for path in assets:
            (root/path).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/path,root/path)
        (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
        daily = dict(version=1,serial='emulator-5554',completed={
            'guild_shop':dict(period='2000-01-01',completed_at='2000-01-01T22:29:22+08:00')})
        (root/'daily_state.json').write_text(json.dumps(daily),encoding='utf-8')
        daily_bytes = (root/'daily_state.json').read_bytes()
        clicker = Clicker(root,Device)
        clicker.digit_reader = self.reader
        try:
            with patch('app.ParallelWorld',Parallel),patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])):
                clicker.start(preview=False,timeout=160)
                clicker.thread.join(65)
                self.assertFalse(clicker.running(),f'Replay did not finish: {clicker.state}')
            self.assertEqual((root/'daily_state.json').read_bytes(),daily_bytes)
            expected = {**training_before,'sailing_verified':False} if scenario=='unverified' else training_before
            self.assertEqual(world.training.state,expected)
            self.assertEqual(self.actions,[])
            return deepcopy(clicker.state),devices[0].taps,list(clicker.events),contexts
        finally:
            clicker.stop()
            if clicker.thread:clicker.thread.join(2)
            for handler in list(clicker.log.handlers):
                handler.close()
                clicker.log.removeHandler(handler)

    def test_first_entry_recovers_with_independent_proof_despite_defense_preemption(self):
        state,taps,events,contexts = self.replay('preempt')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['recoveries'],1)
        self.assertEqual(taps,[self.prefix+'open'])
        self.assertTrue(any('本次不记购买完成' in e['message'] for e in events))
        proofs = [t for t,context in contexts if context is not None]
        self.assertGreaterEqual(len(proofs),3)
        self.assertGreaterEqual(proofs[-1]-proofs[0],3)

    def test_occlusion_resets_independent_confirmation(self):
        state,taps,_events,contexts = self.replay('flicker')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['recoveries'],1)
        self.assertEqual(taps,[self.prefix+'open'])
        self.assertIn(None,[context for _t,context in contexts])

    def test_unknown_unverified_and_expired_entry_keep_original_timeout(self):
        for scenario in ('unknown','unverified','expired'):
            with self.subTest(scenario=scenario):
                state,taps,events,_contexts = self.replay(scenario)
                self.assertIn('后续界面',state['last_error'])
                self.assertEqual(state['recoveries'],0)
                self.assertEqual(taps,[self.prefix+'open'])
                self.assertFalse(any('本次不记购买完成' in e['message'] for e in events))

    def test_visible_shop_is_not_abandoned(self):
        state,taps,_events,_contexts = self.replay('shop')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['recoveries'],0)
        self.assertEqual(taps,[self.prefix+'open',self.prefix+self.next_suffix])

    def test_later_purchase_navigation_cannot_use_initial_entry_recovery(self):
        state,taps,_events,_contexts = self.replay('later')
        self.assertIn('后续界面',state['last_error'])
        self.assertEqual(state['recoveries'],0)
        self.assertEqual(taps,[self.prefix+'open',self.prefix+self.next_suffix])

    def test_saved_entry_cooldown_prevents_reopening_after_recovery(self):
        state,taps,_events,_contexts = self.replay('cooldown')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['recoveries'],1)
        self.assertEqual(taps,[self.prefix+'open'])

    def test_repeated_interruptions_are_bounded_to_three_per_five_minutes(self):
        state,taps,_events,_contexts = self.replay('repeat')
        self.assertIn('五分钟内已恢复三次',state['last_error'])
        self.assertEqual(state['recoveries'],3)
        self.assertEqual(taps,[self.prefix+'open']*4)

    def test_stop_request_never_finishes_recovery_or_issues_another_input(self):
        state,taps,events,_contexts = self.replay('stopped')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['recoveries'],0)
        self.assertEqual(taps,[self.prefix+'open'])
        self.assertFalse(any('本次不记购买完成' in e['message'] for e in events))

    def test_expiry_during_last_screenshot_validation_cannot_commit_recovery(self):
        state,taps,events,_contexts = self.replay('boundary')
        self.assertIn('后续界面',state['last_error'])
        self.assertEqual(state['recoveries'],0)
        self.assertEqual(taps,[self.prefix+'open'])
        self.assertFalse(any('本次不记购买完成' in e['message'] for e in events))

    def test_stop_during_last_screenshot_validation_cannot_commit_recovery(self):
        state,taps,events,_contexts = self.replay('stop_during_proof')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['recoveries'],0)
        self.assertEqual(taps,[self.prefix+'open'])
        self.assertFalse(any('本次不记购买完成' in e['message'] for e in events))


if __name__=='__main__':unittest.main()
