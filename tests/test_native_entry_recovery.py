"""An unopened shop may return to game AUTO without legacy training state."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import Clicker
from digit_ocr import DigitReader
from drift.tasks.features import FEATURE_BY_ID
from drift.tasks.stone_farm import StoneFarm

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT/'tests/fixtures/native-entry-recovery'


def image(path):
    return np.asarray(Image.open(ROOT/path).convert('RGB'))


class NativeEntryRecoveryTests(unittest.TestCase):
    feature = 'event_potion'
    prefix = 'driftinggirl_event_potion_'
    main_asset = 'assets/event-potion/main.png'
    shop_asset = 'assets/event-potion/shop.png'
    next_suffix = 'select'

    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)
        cls.config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.battle = image('tests/fixtures/native-entry-recovery/stage9-rank35-battle.png')
        cls.occluded_result = image('tests/fixtures/native-entry-recovery/stage9-rank34-success.png')
        cls.label_variant = image('tests/fixtures/native-entry-recovery/stage9-rank36-label-variant.png')
        cls.battle36 = image('tests/fixtures/native-entry-recovery/stage9-rank36-battle.png')
        cls.main,cls.shop = image(cls.main_asset),image(cls.shop_asset)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='native-entry-')
        self.addCleanup(self.directory.cleanup)
        self.actions = []
        self.journal_path = Path(self.directory.name)/'stone_farm_state.json'
        self.world = StoneFarm(ROOT,self.reader,lambda *a:self.actions.append(a),
                              lambda *a:None,state_path=self.journal_path)

    def test_actual_new_battle_reads_rank_without_old_training_or_cached_rank(self):
        self.world.rank,self.world.current_stage = 33,8
        self.world.training.state.update(mode='trial',rank=35,sailing_verified=False)
        before = deepcopy(self.world.training.state),deepcopy(self.world.journal)
        context = self.world.native_entry_recovery_context(self.battle)
        self.assertEqual(context[:2],(35,9))
        self.assertEqual((self.world.training.state,self.world.journal),before)
        self.assertEqual((self.world.rank,self.world.current_stage),(33,8))
        self.assertEqual(self.actions,[])
        self.assertFalse(self.journal_path.exists())

    def test_actual_result_can_abandon_entry_without_claiming_a_battle(self):
        self.assertEqual(self.world.screen(self.occluded_result)['rank'],34)
        # This read-only entry recovery has always accepted an actual success
        # or inner fill as well as battle; it never credits a new fishing start.
        context = self.world.native_entry_recovery_context(self.occluded_result)
        self.assertEqual(context[:2],(34,9))
        unknown_world = self.occluded_result.copy()
        unknown_world[40:80,8:46] = 0
        self.assertIsNone(self.world.native_entry_recovery_context(unknown_world))
        self.assertEqual(self.actions,[])
        self.assertFalse(self.journal_path.exists())

    def test_unknown_dark_corrupt_stopped_menu_and_completion_are_rejected(self):
        frames = [self.battle//2,np.zeros_like(self.battle),self.battle[:900],
                  image('assets/recovery/parallel-complete.png'),
                  image('tests/fixtures/stone-farm/stage8-menu-0.png')]
        for frame in frames:
            with self.subTest(shape=frame.shape):
                self.assertIsNone(self.world.native_entry_recovery_context(frame))
        for route in (False,None):
            with patch.object(self.world.sailing,'running',return_value=route), \
                 patch.object(self.world.sailing,'_fixed',return_value=False):
                self.assertIsNone(self.world.native_entry_recovery_context(self.battle))

    def test_disabled_handoff_and_unconfirmed_world_or_rank_are_rejected(self):
        self.world.enabled = lambda:False
        self.assertIsNone(self.world.native_entry_recovery_context(self.battle))
        self.world.enabled = lambda:True
        self.world.journal['pending'] = True
        self.assertIsNone(self.world.native_entry_recovery_context(self.battle))
        self.world.journal['pending'] = False
        self.world.phase = 'return'
        self.assertIsNone(self.world.native_entry_recovery_context(self.battle))
        self.world.phase = 'idle'
        for current in (None,dict(stage=9,rank=40,route=True,menu=False,completion=False),
                        dict(stage=10,rank=35,route=True,menu=False,completion=False)):
            with patch.object(self.world,'screen',return_value=current), \
                 patch.object(self.world,'_rank_value',return_value=None):
                self.assertIsNone(self.world.native_entry_recovery_context(self.battle))

    def test_actual_label_variant_keeps_original_input_guards_and_read_only_context(self):
        self.assertIsNone(self.world.sailing.running(self.label_variant))
        self.assertIsNone(self.world.screen(self.label_variant))
        self.assertFalse(self.world.sailing._fixed(self.label_variant,'auto-label',(186,34,253,65),.90))
        context=self.world.native_entry_recovery_context(self.label_variant)
        self.assertEqual(context[:2],(36,9))
        self.assertEqual(context,self.world.native_entry_recovery_context(self.battle36))
        self.assertEqual(self.actions,[])
        self.assertFalse(self.journal_path.exists())

    def test_label_variant_cannot_override_unknown_arrow_off_dark_or_unread_digits(self):
        unknown=self.label_variant.copy()
        unknown[2:44,192:248]=0
        for frame in (unknown,self.label_variant//2,np.zeros_like(self.label_variant)):
            self.assertIsNone(self.world.native_entry_recovery_context(frame))
        with patch.object(self.world.sailing,'running',return_value=False):
            self.assertIsNone(self.world.native_entry_recovery_context(self.label_variant))
        for method in ('_rank_value','_stage'):
            with patch.object(self.world,method,return_value=None):
                self.assertIsNone(self.world.native_entry_recovery_context(self.label_variant))
        with patch.object(self.world.sailing,'menu',return_value=True):
            self.assertIsNone(self.world.native_entry_recovery_context(self.label_variant))

    def test_route_arrow_alone_never_proves_fishing_and_activity_is_read_only(self):
        with patch.object(self.world.defense,'_battle_footer',return_value=False), \
             patch.object(self.world.training,'_match',return_value=False), \
             patch.object(self.world.fishing,'blue_fill',return_value=False):
            self.assertIsNone(self.world.native_entry_recovery_context(self.battle))
        self.assertEqual(self.actions,[])

    def test_context_excludes_battle_countdown_but_detects_fixed_location_change(self):
        expected = self.world.native_entry_recovery_context(self.battle)
        changed = self.battle.copy()
        changed[125:165,188:610] = 0
        self.assertEqual(self.world.native_entry_recovery_context(changed),expected)
        changed = self.battle.copy()
        changed[12:34,286:434] = 255
        self.assertNotEqual(self.world.native_entry_recovery_context(changed),expected)

    def test_fixture_hashes_match_unmodified_live_evidence(self):
        manifest = json.loads((FIXTURES/'manifest.json').read_text(encoding='utf-8'))
        for name,meta in manifest.items():
            self.assertEqual(hashlib.sha256((FIXTURES/name).read_bytes()).hexdigest().upper(),meta['sha256'])

    def replay(self,scenario):
        config = deepcopy(self.config)
        config['poll_interval'] = .1
        config['rules'] = [r for r in config['rules'] if r['id'] in FEATURE_BY_ID[self.feature]['rules']
                           or r['id']=='driftinggirl_stone_farm']
        for rule in config['rules']:
            rule.update(enabled=True,next_delay=0)
            if rule['id'].startswith('driftinggirl_event_potion_'):
                # Isolate entry recovery and its original deadline; default
                # optional cleanup has separate end-to-end replay coverage.
                rule['skip_on_timeout'] = False
        entry_id,next_id = self.prefix+'open',self.prefix+self.next_suffix
        entry = next(r for r in config['rules'] if r['id']==entry_id)
        entry['interval'] = .2 if scenario=='repeat' else 3600
        clock,devices,contexts = [0.],[],[]
        world,main,battle,shop = self.world,self.main,self.battle,self.shop
        label_variant=self.label_variant
        if scenario=='actual_label_variation':battle=self.battle36
        before = deepcopy(world.journal),deepcopy(world.training.state)

        class Device:
            def __init__(self,_config,stop):
                self.stop,self.image,self.taps,self.after_frames = stop,None,[],0
                devices.append(self)
            def screenshot(self):
                clock[0] += 1
                if not self.taps:self.image=main.copy()
                else:
                    self.after_frames += 1
                    if scenario=='expired' and self.after_frames==1:clock[0] += 31
                    if scenario=='unknown':self.image=np.zeros_like(battle)
                    elif scenario=='flicker' and self.after_frames==3:self.image=battle//2
                    elif scenario in ('shop','later') and len(self.taps)==1:self.image=shop.copy()
                    elif scenario in ('repeat','cooldown') and self.after_frames>=8:self.image=main.copy()
                    elif scenario=='actual_label_variation' and self.after_frames%2:
                        self.image=label_variant.copy()
                    else:self.image=battle.copy()
                    if scenario=='changed' and self.after_frames%2:
                        self.image[12:34,286:434]=255
                    if scenario=='stopped' and self.after_frames==2:self.stop.set()
                    if self.after_frames >= (45 if scenario in
                            ('unknown','changed','repeat','expired','later') else 16):
                        raise InterruptedError('End of bounded replay')
                return self.image
            def click(self,button):
                if button.name==entry_id:assert np.array_equal(self.image,main)
                elif button.name==next_id and scenario in ('shop','later'):
                    assert np.array_equal(self.image,shop)
                    if scenario=='shop':self.stop.set()
                else:raise AssertionError('Unexpected input: '+button.name)
                self.taps.append(button.name)
                self.after_frames=0
            def stuck_record_add(self,_button):pass

        class Native:
            phase,active = 'idle',False
            def __init__(self,*_a,**_kw):self.defense=world.defense
            def retire_expired_on_start(self):return False
            def native_entry_recovery_context(self,frame):
                context = world.native_entry_recovery_context(frame)
                contexts.append((clock[0],context))
                if context is not None and sum(c is not None for _t,c in contexts)>=4:
                    if scenario=='boundary':clock[0]+=31
                    if scenario=='stop_during_proof':devices[0].stop.set()
                return context
            def status(self):return dict(phase=self.phase,pending=False)
            def step(self,_frame):
                return bool(devices[0].taps) and scenario not in ('repeat','cooldown','shop','later')

        root = Path(self.directory.name)/('runner-'+scenario)
        root.mkdir()
        assets = {r['asset'] for r in config['rules']}
        assets.update(p for r in config['rules'] for p in r.get('asset_variants',[]))
        for path in assets:
            (root/path).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/path,root/path)
        (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
        daily = json.dumps(dict(version=1,serial='emulator-5554',completed={}))
        (root/'daily_state.json').write_text(daily,encoding='utf-8')
        clicker = Clicker(root,Device)
        clicker.digit_reader=self.reader
        try:
            with patch('app.StoneFarm',Native), \
                 patch('app.ParallelWorld',side_effect=AssertionError('Legacy training must stay disabled')), \
                 patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])):
                clicker.start(preview=False,timeout=160)
                clicker.thread.join(65)
                self.assertFalse(clicker.running(),'Bounded replay did not finish')
            self.assertEqual((root/'daily_state.json').read_text(encoding='utf-8'),daily)
            self.assertEqual((world.journal,world.training.state),before)
            self.assertEqual(self.actions,[])
            self.assertFalse(self.journal_path.exists())
            return deepcopy(clicker.state),devices[0].taps,list(clicker.events),contexts
        finally:
            clicker.stop()
            if clicker.thread:clicker.thread.join(2)
            for handler in list(clicker.log.handlers):
                handler.close()
                clicker.log.removeHandler(handler)

    def test_native_entry_recovery_does_not_require_legacy_controller(self):
        state,taps,events,contexts=self.replay('normal')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['recoveries'],1)
        self.assertEqual(taps,[self.prefix+'open'])
        self.assertTrue(any('游戏航巡和实际钓鱼画面' in e['message'] for e in events))
        proof=[t for t,c in contexts if c is not None]
        self.assertGreaterEqual(len(proof),3)
        self.assertGreaterEqual(proof[-1]-proof[0],3)

    def test_occlusion_resets_three_frame_three_second_proof(self):
        state,taps,_events,contexts=self.replay('flicker')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['recoveries'],1)
        self.assertEqual(taps,[self.prefix+'open'])
        self.assertIn(None,[c for _t,c in contexts])

    def test_real_label_variation_does_not_starve_three_second_recovery(self):
        state,taps,events,contexts=self.replay('actual_label_variation')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['recoveries'],1)
        self.assertEqual(taps,[self.prefix+'open'])
        self.assertTrue(any('游戏航巡和实际钓鱼画面' in e['message'] for e in events))
        self.assertGreaterEqual(len([c for _t,c in contexts if c is not None]),3)

    def test_unknown_changing_location_or_expiry_cannot_release_entry(self):
        for scenario in ('unknown','changed','expired','boundary'):
            with self.subTest(scenario=scenario):
                state,taps,_events,_contexts=self.replay(scenario)
                self.assertIn('后续界面',state['last_error'])
                self.assertEqual(state['recoveries'],0)
                self.assertEqual(taps,[self.prefix+'open'])

    def test_visible_shop_and_later_purchase_keep_original_flow(self):
        state,taps,_events,_contexts=self.replay('shop')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['recoveries'],0)
        self.assertEqual(taps,[self.prefix+'open',self.prefix+self.next_suffix])
        state,taps,_events,_contexts=self.replay('later')
        self.assertIn('后续界面',state['last_error'])
        self.assertEqual(state['recoveries'],0)
        self.assertEqual(taps,[self.prefix+'open',self.prefix+self.next_suffix])

    def test_recovery_keeps_entry_cooldown_and_three_per_five_minutes_limit(self):
        state,taps,_events,_contexts=self.replay('cooldown')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['recoveries'],1)
        self.assertEqual(taps,[self.prefix+'open'])
        state,taps,_events,_contexts=self.replay('repeat')
        self.assertIn('五分钟内已恢复三次',state['last_error'])
        self.assertEqual(state['recoveries'],3)
        self.assertEqual(taps,[self.prefix+'open']*4)

    def test_stop_before_or_during_proof_never_finishes_or_repeats_input(self):
        for scenario in ('stopped','stop_during_proof'):
            with self.subTest(scenario=scenario):
                state,taps,_events,_contexts=self.replay(scenario)
                self.assertIsNone(state['last_error'])
                self.assertEqual(state['recoveries'],0)
                self.assertEqual(taps,[self.prefix+'open'])


class NativeGuildEntryRecoveryTests(NativeEntryRecoveryTests):
    feature = 'guild_shop'
    prefix = 'driftinggirl_guild_'
    main_asset = 'assets/guild-shop/main.png'
    shop_asset = 'assets/guild-shop/open.png'
    next_suffix = 'special'


if __name__=='__main__':unittest.main()
