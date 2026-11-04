"""Boundary, native-route preservation and interrupted-calibration regressions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import shutil
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from digit_ocr import DigitReader
from features import feature_states, execution_mode
from meter_calibration import MeterCalibration
from tools.configure_meter_calibration import add, RULE

ROOT = Path(__file__).resolve().parents[1]


def frame(name):
    folder = 'meter-calibration' if name.startswith('native-') else 'fishing-meter'
    return np.asarray(Image.open(ROOT/'tests/fixtures'/folder/(name+'.png')).convert('RGB'))


class MeterCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.events, self.inputs = [], []
        # A real local diagnostic journal must not leak into unrelated tests.
        journal = ROOT/'meter_calibration_state.json'
        original_is_file = Path.is_file
        with patch.object(Path,'is_file',lambda p: False if p == journal else original_is_file(p)):
            self.controller = MeterCalibration(ROOT, self.reader,
                lambda *args: self.inputs.append(args) or False,
                lambda *args: self.inputs.append(args) or True, self.events.append)
        self.controller.path = Path(self.temp.name)/'meter_calibration_state.json'
        self.controller.world.training._save = self.controller._save

    def tearDown(self):
        self.temp.cleanup()

    def pending(self, phase='prepare', running=True):
        c = self.controller
        c.journal.update(pending=True, rank=38, stage=8, restore_running=running,
                         deadline=1090, stage_strokes=b'stage'.hex(), attempts={}, value=1195,high=1421)
        c.phase, c.off_verified, c.context = phase, True, 'place'
        c.world.training.state = dict(mode='farming',rank=38,sailing_enabled=False,
            sailing_sleep_recovery=True,sailing_meter_adjustment=True)
        return c

    def restart(self, payload, *, now=1000, monotonic=100):
        """Restore a journal without writing the project's actual saved state."""
        saved = json.dumps(payload).encode('utf-8')
        journal = ROOT / 'meter_calibration_state.json'
        original_exists, original_read = Path.is_file, Path.read_text
        with patch.object(Path, 'is_file', lambda p: p == journal or original_exists(p)), \
             patch.object(Path, 'read_text', lambda p, **kw: saved.decode('utf-8')
                          if p == journal else original_read(p, **kw)), \
             patch('meter_calibration.time.time', return_value=now), \
             patch('meter_calibration.time.monotonic', return_value=monotonic):
            c = MeterCalibration(ROOT, self.reader,
                lambda *args: self.inputs.append(args) or False,
                lambda *args: self.inputs.append(args) or False, self.events.append)
        c.path = Path(self.temp.name) / 'restored-meter.json'
        c.path.write_bytes(saved)
        return c

    def test_strict_directional_gap_boundary(self):
        # 阈值 2026-10-11 由 100 改为 50;正好等于阈值不触发,高一级才触发。
        # 边界按常量推导,以后调阈值只需改 MeterCalibration.GAP。
        gap = MeterCalibration.GAP
        cases = ((-101, False), (0, False), (gap - 1, False), (gap, False),
                 (gap + 1, True), (gap + 150, True))
        for difference, wanted in cases:
            with self.subTest(difference=difference):
                data=dict(player=1200+difference,value=1200,high=1800,adaptive=True)
                self.assertEqual(MeterCalibration.needs_adjustment(data),wanted)

    def test_already_at_max_or_unchecked_auto_does_not_retrigger(self):
        for data in (None,dict(player=1800,value=1400,high=1400,adaptive=True),
                     dict(player=1800,value=1400,high=1900,adaptive=False)):
            self.assertFalse(MeterCalibration.needs_adjustment(data))

    def test_config_addition_isolated_idempotent_and_off_by_default(self):
        original=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        original['rules']=[r for r in original['rules'] if r['id']!=RULE['id']]
        before=deepcopy(original)
        updated=add(original,enabled=True)
        self.assertEqual(original,before)
        self.assertEqual(updated['rules'][:-1],before['rules'])
        self.assertEqual(add(updated),updated)
        self.assertTrue(updated['rules'][-1]['enabled'])
        self.assertFalse(add(original)['rules'][-1]['enabled'])
        feature=next(f for f in feature_states(updated) if f['id']=='meter_calibration')
        self.assertTrue(feature['available'])
        self.assertEqual(execution_mode('meter_calibration'),'realtime')

    def test_actual_native_four_targets_and_both_extras_are_accepted_unchanged(self):
        c=self.controller
        native=frame('native-four-targets-menu')
        self.assertEqual(c.settings_key(native)[0],(True,False,True,True,True))
        self.assertFalse(c.sailing.extras_off(native))
        self.assertIsNotNone(c.settings_key(native.copy()))
        self.assertEqual(c.screen(native),dict(rank=38,stage=8,menu=True,route=None))
        self.assertEqual(self.inputs,[])

    def test_real_menu_shadow_does_not_change_the_saved_choices_fingerprint(self):
        c=self.controller
        paths=sorted((ROOT/'tests/fixtures/meter-menu-shadow').glob('*.png'))
        self.assertEqual(len(paths),4)
        images=[np.asarray(Image.open(p).convert('RGB')) for p in paths]
        keys=[c.settings_key(f) for f in images]
        self.assertTrue(all(k==keys[0] for k in keys))
        self.assertEqual(keys[0][0],(True,False,True,True,True))
        for native in images:
            self.assertEqual(c.screen(native),dict(rank=38,stage=9,menu=True,route=None))
        changed=images[0].copy()
        # Swapping selected/unselected priority buttons must change the proof.
        changed[515:573,260:410]=images[0][515:573,440:590]
        self.assertNotEqual(c.settings_key(changed),keys[0])
        for y in (627,741):
            changed=images[0].copy();changed[y:y+40,557:595]=255
            self.assertNotEqual(c.settings_key(changed),keys[0])
        self.assertEqual(self.inputs,[])

    def test_rank0_menu_background_outside_buttons_is_not_a_native_choice(self):
        paths = sorted((ROOT/'tests/fixtures/stone-farm').glob('stage9-rank0-menu-*.png'))
        self.assertEqual(len(paths),3)
        images = [np.asarray(Image.open(p).convert('RGB')) for p in paths]
        c = self.controller
        keys = [c.settings_key(f) for f in images]
        self.assertTrue(all(k == keys[0] for k in keys))
        self.assertEqual(keys[0][0],(True,False,True,True,True))
        for native in images:
            self.assertTrue(c.sailing.extras_off(native))
        # Online changes to the actual priority buttons and either checkbox
        # must still invalidate the saved settings before any input.
        changed = images[0].copy()
        changed[520:574,260:415] = images[0][520:574,434:589]
        self.assertNotEqual(c.settings_key(changed),keys[0])
        for y in (627,741):
            changed = images[0].copy()
            changed[y:y+40,557:595] = 255
            self.assertNotEqual(c.settings_key(changed),keys[0])
        self.assertEqual(self.inputs,[])

    def test_restore_only_restart_keeps_both_original_and_separate_deadlines(self):
        payload=dict(version=1,pending=True,rank=38,stage=9,restore_running=True,
            deadline=900,proof='paired_hold',resume_requested=False,
            attempts={'cancel':1,'drag':2,'start':1},sailing_attempts={'open':1},
            restore_retry=dict(deadline=1040,failed_sha256='failure'))
        c=self.restart(payload)
        self.assertEqual(c.phase,'menu')
        self.assertEqual(c.journal['deadline'],900)
        self.assertEqual(c.journal['restore_retry']['deadline'],1040)
        with patch('meter_calibration.time.time',return_value=1000):
            self.assertTrue(c.input_allowed())
            self.assertFalse(c.reserve_meter_input('start'))
        with patch('meter_calibration.time.time',return_value=1040):
            self.assertFalse(c.input_allowed())
        self.assertEqual(self.inputs,[])

    def test_real_sleep_reading_and_current_world(self):
        c=self.controller
        image=frame('sleep')
        screen=c.screen(image)
        self.assertEqual((screen['rank'],screen['stage']),(37,9))
        data=c.fishing.measure(image)
        self.assertEqual((data['player'],data['value'],data['high']),(1348,1195,1421))
        self.assertTrue(c.needs_adjustment(data))
        self.assertEqual(self.inputs,[])

    def test_current_native_sailing_gap169_is_recognized_with_same_stage(self):
        c=self.controller;native=frame('native-gap169')
        self.assertEqual(c.screen(native),dict(rank=39,stage=8,menu=False,route=True))
        data=c.fishing.measure(native)
        self.assertEqual((data['player'],data['value'],data['high']),(1741,1572,1810))
        self.assertTrue(c.needs_adjustment(data))
        self.assertEqual(self.inputs,[])

    def test_unknown_wrong_world_and_boss_are_read_only(self):
        for image in (np.zeros((1280,720,3),dtype='uint8'),frame('wrong-world')):
            self.assertIsNone(self.controller.screen(image))
            self.assertFalse(self.controller.step(image))
        with patch.object(self.controller.world.training,'boss_mode',return_value=True):
            self.assertIsNone(self.controller.screen(frame('sleep')))
        self.assertEqual(self.inputs,[])

    def test_real_task_boss_is_excluded_without_the_navigation_controller(self):
        c = self.controller
        for name in ('training/boss-main', 'boss-defense/native-right-charge-low'):
            image = np.asarray(Image.open(ROOT / 'tests/fixtures' / f'{name}.png').convert('RGB'))
            c.fishing.current = c.fishing.selected(image, c.fishing.box(image))
            c.fishing.goal = 1810
            self.assertTrue(c.world.training.boss_mode(image), name)
            self.assertIsNone(c.screen(image), name)
            self.assertIsNone(c.fishing.measure(image), name)
            self.assertFalse(c.fishing.waiting_cast(image), name)
            self.assertFalse(c.fishing.cancellable_cast(image), name)
            expected = c.fishing.cast_data(image)
            self.assertFalse(c.fishing.validate('cancel', image, expected), name)
            self.assertFalse(c.step(image), name)
        self.assertEqual(self.inputs, [])

    def test_restart_restores_current_rank_recovery_flags_and_fresh_range_cache(self):
        image = frame('sleep-top-overlap')
        context = self.controller.fishing._context(image)
        training = dict(mode='farming', rank=38, sailing_enabled=False,
                        sailing_sleep_recovery=True, sailing_meter_adjustment=True,
                        fishing_meter_limits=dict(version=1, context=context,
                            low=1350, low_at=1000, high=1604, high_at=1000))
        c = self.restart(dict(version=1, pending=True, rank=38, stage=9,
            restore_running=True, deadline=1090, attempts={'cancel': 1},
            value=1563, high=1604, training=training))
        self.assertEqual(c.world.rank, 38)
        self.assertEqual(c.world.training.state, training)
        self.assertFalse(c.off_verified, 'saved state is not fresh OFF proof')
        with patch('meter_calibration.time.time', return_value=1001):
            self.assertEqual(c.fishing._cached_limit(context, 'high'), 1604)
            self.assertEqual(c.fishing.measure(image, remember=False)['high'], 1604)
        with patch('meter_calibration.time.time', return_value=1181):
            self.assertIsNone(c.fishing._cached_limit(context, 'high'))
        self.assertIsNone(c.fishing._cached_limit('different-place', 'high'))
        c._save()
        self.assertEqual(json.loads(c.path.read_text())['training'], training)
        self.assertEqual(self.inputs, [])

    def test_restart_retains_saved_low_fill_cancellation_context(self):
        training = dict(mode='farming', rank=35, sailing_enabled=False,
                        sailing_sleep_recovery=True, sailing_meter_adjustment=True)
        c = self.restart(dict(version=1, pending=True, rank=35, stage=8,
            restore_running=True, deadline=1090, attempts={'cancel': 1},
            value=934, high=1098, training=training))
        self.assertTrue(c.fishing.cancellable_cast(frame('filled-dots-rank35')))
        self.assertEqual(c.fishing.cancels, 1)
        self.assertEqual(self.inputs, [])

    def test_finished_restart_keeps_range_cache_without_pinning_the_old_rank(self):
        image = frame('sleep')
        context = self.controller.fishing._context(image)
        cache = dict(version=1, context=context, high=1421, high_at=1000)
        c = self.restart(dict(version=1, pending=False, rank=38, stage=9,
            training=dict(mode='farming', rank=38, sailing_sleep_recovery=True,
                          sailing_meter_adjustment=True, fishing_meter_limits=cache)))
        original = c.path.read_bytes()
        self.assertIsNone(c.world.rank)
        self.assertEqual(c.world.training.state['mode'], 'idle')
        self.assertEqual(c.fishing.range_cache, cache)
        self.assertFalse(c.world.training.state.get('sailing_meter_adjustment'))
        with patch('meter_calibration.time.time', return_value=1001):
            self.assertEqual(c.screen(image)['rank'], 37)
            self.assertIsNotNone(c.fishing.measure(image, remember=False))
        self.assertEqual(c.path.read_bytes(), original)
        self.assertEqual(self.inputs, [])

    def test_restart_never_renews_the_saved_deadline_after_off_confirmation(self):
        for deadline in (990, 1090):
            with self.subTest(deadline=deadline):
                c = self.restart(dict(version=1, pending=True, rank=38, stage=9,
                    restore_running=True, deadline=deadline, attempts={'cancel': 1},
                    sailing_attempts={'stop': 1}, value=1195, high=1421))
                with patch('meter_calibration.time.time', return_value=1000):
                    c._set('prepare')
                self.assertEqual(c.journal['deadline'], deadline)
                self.assertEqual(c.journal['attempts'], {'cancel': 1})
                self.assertEqual(c.journal['sailing_attempts'], {'stop': 1})

    def test_expired_restart_with_corrected_meter_still_keeps_required_route_restore(self):
        training = dict(mode='farming', rank=37, sailing_enabled=False,
                        sailing_sleep_recovery=True, sailing_meter_adjustment=True)
        c = self.restart(dict(version=1, pending=True, rank=37, stage=9,
            restore_running=True, deadline=990, attempts={'start': 1},
            sailing_attempts={'stop': 1}, value=1348, high=1421,
            proof='paired_hold', training=training))
        original = c.path.read_bytes()
        image = frame('fill-hook')
        data = c.fishing.measure(image, remember=False)
        self.assertEqual(data['player'] - data['value'], 6)
        self.assertIs(c.sailing.running(image), False)
        self.assertTrue(c.fishing.blue_fill(image))
        with patch('meter_calibration.time.time', return_value=1000), \
             patch('meter_calibration.time.monotonic', return_value=101):
            self.assertTrue(c.step(image))
            self.assertFalse(c.reserve_meter_input('start'))
            self.assertFalse(c.reserve_sailing_input('start'))
        with patch('meter_calibration.time.time', return_value=1000), \
             patch('meter_calibration.time.monotonic', return_value=116):
            self.assertFalse(c.step(image))
        self.assertFalse(c.active)
        self.assertTrue(c.awaiting_recovery)
        self.assertTrue(c.journal['restore_running'])
        self.assertEqual(Path(c.journal['retirement']['archive']).read_bytes(), original)
        self.assertEqual(self.inputs, [])

    def test_current_rank_or_stage_conflict_rejects_all_inputs(self):
        c=self.pending()
        for actual_rank,stage in ((37,8),(38,9),(40,8),(None,8)):
            with patch.object(c.world,'_parallel',return_value=True), \
                 patch.object(c.world.training,'boss_mode',return_value=False), \
                 patch.object(c.sailing,'menu',return_value=False), \
                 patch.object(c.sailing,'running',return_value=False), \
                 patch.object(c.fishing,'at_port',return_value=False), \
                 patch.object(c.world,'_rank_value',return_value=actual_rank), \
                 patch.object(c,'_stage',return_value=stage),patch('meter_calibration.time.time',return_value=1000):
                self.assertIsNone(c.screen(frame('sleep')))
                self.assertFalse(c.validate_sailing('open',frame('sleep')))

    def test_three_frames_and_three_seconds_required(self):
        c=self.controller
        for now,wanted in ((100,False),(101,False),(102.9,False),(103,True)):
            with patch('meter_calibration.time.monotonic',return_value=now):
                self.assertEqual(c._confirmed('same'),wanted)
        with patch('meter_calibration.time.monotonic',return_value=115):
            self.assertFalse(c._confirmed('same'))

    def test_confirmed_candidate_preserves_route_without_immediate_fishing_input(self):
        c=self.controller
        data=dict(player=1348,value=1195,high=1421,context='place',adaptive=True)
        screen=dict(rank=37,stage=8,menu=False,route=True)
        with patch.object(c,'screen',return_value=screen),patch.object(c.fishing,'measure',return_value=data), \
             patch.object(c.world,'_stage_strokes',return_value=b'stage'), \
             patch.object(c.fishing,'prepare',side_effect=AssertionError('trigger must not issue fishing input')), \
             patch('meter_calibration.time.time',return_value=1000):
            for now in (100,101.5,103):
                with patch('meter_calibration.time.monotonic',return_value=now):
                    c.step(frame('sleep'))
        self.assertEqual(self.inputs,[])
        self.assertTrue(c.active)
        self.assertFalse(c.off_verified)
        self.assertTrue(c.route_verified)
        self.assertEqual(c.phase,'prepare')
        self.assertEqual(c.journal['route_policy'],'preserve')
        self.assertEqual(c.journal['deadline'],1090)
        saved_deadline=c.journal['deadline']
        with patch.object(c,'screen',return_value=screen), \
             patch.object(c.fishing,'_context',return_value='new-place'), \
             patch.object(c.fishing,'prepare',side_effect=AssertionError('changed place cannot input')), \
             patch('meter_calibration.time.time',return_value=1001), \
             patch('meter_calibration.time.monotonic',return_value=104):
            c.step(frame('sleep'))
        self.assertFalse(c.route_verified)
        self.assertEqual(c.phase,'route_context')
        self.assertEqual(c.journal['deadline'],saved_deadline)
        self.assertEqual(self.inputs,[])

    def test_stopped_context_requires_its_own_continuous_proof(self):
        c=self.pending('stop');c.off_verified=False
        screen=dict(rank=38,stage=8,menu=False,route=False)
        with patch.object(c,'screen',return_value=screen),patch.object(c.fishing,'_context',return_value='place'), \
             patch('meter_calibration.time.time',return_value=1000):
            for now in (100,101.5):
                with patch('meter_calibration.time.monotonic',return_value=now): c.step(frame('sleep'))
                self.assertFalse(c.off_verified)
            with patch('meter_calibration.time.monotonic',return_value=103): c.step(frame('sleep'))
        self.assertTrue(c.off_verified)
        self.assertEqual(c.phase,'prepare')
        self.assertEqual(self.inputs,[])

    def test_new_running_arrow_invalidates_off_proof(self):
        c=self.pending()
        with patch.object(c,'screen',return_value=dict(rank=38,stage=8,menu=False,route=True)), \
             patch('meter_calibration.time.time',return_value=1000):
            c.step(frame('sleep'))
        self.assertFalse(c.off_verified)
        self.assertEqual(self.inputs[0][0],'stop')

    def test_existing_near_player_fill_can_resume_without_claiming_a_new_hold(self):
        c=self.pending()
        data=dict(player=1348,value=1343,high=1421,low=1195,adaptive=True,y=985)
        with patch.object(c,'screen',return_value=dict(rank=38,stage=8,menu=False,route=False)), \
             patch.object(c,'meter_frame_allowed',return_value=True), \
             patch.object(c.fishing,'measure',return_value=data), \
             patch.object(c.fishing,'blue_fill',return_value=True), \
             patch.object(c.fishing,'prepare',side_effect=AssertionError('existing fill must not cancel/hold')), \
             patch('meter_calibration.time.time',return_value=1000):
            for now in (100,101.5,103):
                with patch('meter_calibration.time.monotonic',return_value=now): c.step(frame('sleep'))
        self.assertEqual(c.phase,'open')
        self.assertEqual(c.journal['proof'],'observed_target_fill')
        self.assertEqual(c.fishing.confirmed_starts,0)
        self.assertEqual(self.inputs,[])

    def test_menu_restore_issues_only_start_without_target_or_extra_clicks(self):
        c=self.pending('menu');native=frame('native-four-targets-menu')
        settings=c.settings_key(native)
        with patch.object(c,'screen',return_value=dict(rank=38,stage=8,menu=True,route=None)), \
             patch('meter_calibration.time.time',return_value=1000):
            for now in (100,101.5,103):
                with patch('meter_calibration.time.monotonic',return_value=now): c.step(native)
        self.assertEqual(c.menu_key,settings)
        self.assertEqual([a[:3] for a in self.inputs],[('start',360,873)])
        self.assertEqual(c.phase,'running')

    def test_end_only_fill_after_our_start_cannot_be_promoted_to_success(self):
        c=self.pending()
        c.journal['attempts']={'start':1}
        c.fishing.starts=1
        data=dict(player=1348,value=1343,high=1421,low=1195,adaptive=True,y=985)
        with patch.object(c,'screen',return_value=dict(rank=38,stage=8,menu=False,route=False)), \
             patch.object(c,'meter_frame_allowed',return_value=True), \
             patch.object(c.fishing,'measure',return_value=data), \
             patch.object(c.fishing,'blue_fill',return_value=True), \
             patch.object(c.fishing,'prepare',return_value=True), \
             patch('meter_calibration.time.time',return_value=1000):
            for now in (100,101.5,103):
                with patch('meter_calibration.time.monotonic',return_value=now): c.step(frame('sleep'))
        self.assertEqual(c.phase,'prepare')
        self.assertIsNone(c.journal.get('proof'))
        self.assertEqual(c.fishing.confirmed_starts,0)
        self.assertEqual(self.inputs,[])

    def test_three_frame_post_release_proof_is_not_confirmed_a_second_time(self):
        c=self.pending();c.journal['attempts']={'start':1}
        c.fishing.start_fill_verified=True
        c.fishing.start_proof='post_release_fishing'
        data=dict(player=1348,value=1343,high=1421,low=1195,adaptive=True,y=985)
        with patch.object(c,'screen',return_value=dict(rank=38,stage=8,menu=False,route=False)), \
             patch.object(c,'meter_frame_allowed',return_value=True), \
             patch.object(c.fishing,'measure',return_value=data), \
             patch.object(c.fishing,'post_start_evidence',return_value=True), \
             patch.object(c,'_confirmed',side_effect=AssertionError('duplicate post-release proof')), \
             patch('meter_calibration.time.time',return_value=1000):
            c.step(frame('sleep'))
        self.assertEqual(c.phase,'open')
        self.assertEqual(c.journal['proof'],'post_release_fishing')
        self.assertEqual(c.journal['deadline'],1090)
        self.assertEqual(c.journal['attempts'],{'start':1})
        self.assertEqual(self.inputs,[])

    def test_verified_post_release_still_rejects_lost_current_frame_evidence(self):
        c=self.pending();c.journal['attempts']={'start':1}
        c.fishing.start_fill_verified=True
        c.fishing.start_proof='post_release_fishing'
        with patch.object(c,'screen',return_value=dict(rank=38,stage=8,menu=False,route=False)), \
             patch.object(c,'meter_frame_allowed',return_value=True), \
             patch.object(c.fishing,'measure',return_value=None), \
             patch.object(c.fishing,'post_start_evidence',return_value=False), \
             patch.object(c.fishing,'prepare',return_value=True), \
             patch('meter_calibration.time.time',return_value=1000):
            c.step(frame('sleep'))
        self.assertEqual(c.phase,'prepare')
        self.assertIsNone(c.journal.get('proof'))
        self.assertEqual(self.inputs,[])

    def test_fresh_input_guard_rejects_changed_saved_menu_choices(self):
        c=self.pending('menu');native=frame('native-four-targets-menu')
        c.menu_key=c.settings_key(native)
        with patch.object(c,'screen',return_value=dict(rank=38,stage=8,menu=True,route=None)), \
             patch('meter_calibration.time.time',return_value=1000):
            self.assertTrue(c.validate_sailing('start',native))
            with patch.object(c,'settings_key',return_value=((True,False,False,True,True),'changed')):
                self.assertFalse(c.validate_sailing('start',native))

    def test_originally_stopped_route_stays_stopped(self):
        c=self.pending(running=False)
        c.journal['proof']='observed_max_fill'
        with patch('meter_calibration.time.time',return_value=1000): c._finish()
        self.assertFalse(c.active)
        self.assertEqual(c.phase,'idle')
        self.assertEqual(self.inputs,[])

    def test_timeout_and_disabling_never_issue_inputs_or_extend_deadline(self):
        c=self.pending();c._save()
        with patch('meter_calibration.time.time',return_value=1090):
            self.assertFalse(c.step(frame('sleep')))
        self.assertTrue(c.awaiting_recovery)
        self.assertEqual(c.journal['deadline'],1090)
        c=self.pending();c.journal['retirement']=None
        c.enabled=lambda:False
        with self.assertRaisesRegex(RuntimeError,'关闭'): c.step(frame('sleep'))
        self.assertEqual(self.inputs,[])

    def test_cancel_attempt_reserved_in_separate_record_before_input(self):
        c=self.pending()
        with patch('meter_calibration.time.time',return_value=1000):
            self.assertTrue(c.reserve_meter_input('scan_cancel'))
        saved=json.loads(c.path.read_text(encoding='utf-8'))
        self.assertEqual(saved['attempts'],{'cancel':1})
        self.assertEqual(saved['deadline'],1090)
        self.assertFalse((c.path.parent/'training_state.json').exists())

    def test_restart_retains_attempts_and_deadline_but_requires_new_off_proof(self):
        c=self.pending();c.journal['attempts']=dict(cancel=1,drag=2,start=1);c._save()
        saved=c.path.read_text(encoding='utf-8')
        original_exists,original_read=Path.is_file,Path.read_text
        journal=ROOT/'meter_calibration_state.json'
        with patch.object(Path,'is_file',lambda p: p==journal or original_exists(p)), \
             patch.object(Path,'read_text',lambda p,**kw: saved if p==journal else original_read(p,**kw)):
            restored=MeterCalibration(ROOT,self.reader,lambda *a:False,lambda *a:False,self.events.append)
        self.assertEqual(restored.journal['deadline'],1090)
        self.assertEqual((restored.fishing.cancels,restored.fishing.drags,restored.fishing.starts),(1,2,1))
        self.assertFalse(restored.off_verified)
        self.assertEqual(restored.phase,'stop')

    def test_reserved_resume_survives_restart_without_a_duplicate_toggle(self):
        c=self.pending('menu')
        with patch('meter_calibration.time.time',return_value=1000):
            self.assertTrue(c.reserve_sailing_input('start'))
            with self.assertRaisesRegex(RuntimeError,'上限'): c.reserve_sailing_input('start')
        self.assertTrue(c.journal['resume_requested'])
        self.assertEqual(c.journal['deadline'],1090)

    def test_pending_restart_in_stopped_menu_reconfirms_before_resume(self):
        c=self.pending('stop');c.off_verified=False
        c.journal.update(proof='observed_max_fill',stopped_context='place')
        native=frame('native-four-targets-menu')
        with patch.object(c,'screen',return_value=dict(rank=38,stage=8,menu=True,route=None)), \
             patch('meter_calibration.time.time',return_value=1000):
            for now in (100,101.5,103):
                with patch('meter_calibration.time.monotonic',return_value=now): c.step(native)
        self.assertTrue(c.off_verified)
        self.assertEqual(c.phase,'menu')
        self.assertEqual(self.inputs,[])

    def test_fast_cancel_rejects_new_sailing_unknown_stage_and_context_changes(self):
        c=self.pending()
        native=frame('sleep')
        with patch('meter_calibration.time.time',return_value=1000), \
             patch.object(c.world,'_parallel',return_value=True), \
             patch.object(c.sailing,'running',return_value=False), \
             patch.object(c.world.training,'boss_mode',return_value=False), \
             patch.object(c.fishing,'at_port',return_value=False), \
             patch.object(c.world,'_stage_strokes',return_value=b'stage'), \
             patch.object(c.fishing,'_context',return_value='place'):
            self.assertTrue(c.meter_frame_allowed(native))
            with patch.object(c.sailing,'running',return_value=True): self.assertFalse(c.meter_frame_allowed(native))
            with patch.object(c.world,'_stage_strokes',return_value=None): self.assertFalse(c.meter_frame_allowed(native))
            with patch.object(c.fishing,'_context',return_value='changed'): self.assertFalse(c.meter_frame_allowed(native))

    def test_worker_with_old_saved_training_does_not_take_over_native_auto(self):
        from app import Clicker
        root=Path(self.temp.name)
        shutil.copytree(ROOT/'assets',root/'assets')
        config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        old=deepcopy(next(r for r in config['rules'] if r['id']=='driftinggirl_parallel_world'))
        old['enabled']=False
        config.update(rules=[old,{**deepcopy(RULE),'enabled':True}],poll_interval=.1)
        (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
        saved=b'{"version":1,"mode":"trial","rank":35,"sailing_enabled":true}'
        (root/'training_state.json').write_bytes(saved)
        image=frame('sleep')
        class Device:
            def __init__(self,_,stop): self.stop,self.io_lock,self.image=stop,threading.RLock(),None
            def screenshot(self): self.image=image.copy();return self.image
            def click(self,_): raise AssertionError('Unexpected native route toggle')
            def stuck_record_add(self,_): pass
        fake=SimpleNamespace(active=False,cycles=0,phase='idle',step=lambda _:False,
            retire_expired_on_start=lambda:False,status=lambda:dict(phase='idle'))
        app=Clicker(root,Device);app.digit_reader=self.reader
        try:
            with patch('app.MeterCalibration',return_value=fake), \
                 patch('app.ParallelWorld',side_effect=AssertionError('Old navigation started')):
                app.start(preview=False,timeout=1)
                app.thread.join(5)
            self.assertFalse(app.running())
            self.assertIsNone(app.state['last_error'])
            self.assertEqual(app.state['clicks'],0)
            self.assertIsNone(app.state['parallel'])
            self.assertEqual(app.state['meter'],dict(phase='idle'))
            self.assertEqual((root/'training_state.json').read_bytes(),saved)
        finally:
            app.stop()
            if app.thread: app.thread.join(3)
            for handler in list(app.log.handlers): handler.close();app.log.removeHandler(handler)

    def test_actual_faded_preparation_observes_fast_window_without_range_ocr(self):
        c=self.pending();native=frame('native-timeout-preparation')
        c.journal.update(rank=39,stage=8,value=1562,high=1810,
                         stage_strokes=c.world._stage_strokes(native).hex())
        c.context=c.fishing._context(native)
        c.world.training.state['rank']=39
        c.fishing.current,c.fishing.goal=1562,1810
        self.assertFalse(c.fishing.blue_fill(native))
        self.assertIsNone(c.fishing.idle_icon(native))
        # This fading exclamation stays a negative input frame. A new bright
        # preparation frame must independently pass the existing X guard.
        self.assertFalse(c.fishing.cancellable_cast(native))
        c.journal['selected_signature']=c._selection_signature(c.fishing.cast_data(native))
        with patch.object(c,'screen',side_effect=AssertionError('slow RANK OCR before short window')), \
             patch.object(c.fishing,'measure',side_effect=AssertionError('slow OCR before short window')), \
             patch('meter_calibration.time.time',return_value=1000):
            c.step(native)
        self.assertEqual([a[0] for a in self.inputs],['scan_cancel'])
        self.assertEqual(c.fishing.cancels,0)  # callback rejected, no actual input
        self.assertEqual(c.journal['deadline'],1090)

    def test_changed_selected_strokes_never_start_fast_window(self):
        c=self.pending();native=frame('native-timeout-preparation')
        c.journal['selected_signature']=dict(box=[1,2,3,4],digit_strokes='changed',world_strokes='stage')
        self.assertFalse(c._observe_cancel_window(native))
        self.assertEqual(self.inputs,[])

    def test_freshly_started_fill_cannot_be_quickly_cancelled(self):
        c=self.pending();native=frame('native-timeout-preparation')
        c.journal['selected_signature']=c._selection_signature(c.fishing.cast_data(native))
        for attempts,verified in (({'start':1},False),({},True)):
            c.journal['attempts']=attempts;c.fishing.start_fill_verified=verified
            self.assertFalse(c._observe_cancel_window(native))
        self.assertEqual(self.inputs,[])

    def test_selection_signature_survives_json_restart(self):
        c=self.controller
        signature=c._selection_signature(dict(box=(210,1045,281,1086),digit_strokes='digits',world_strokes='stage'))
        self.assertEqual(signature,json.loads(json.dumps(signature)))

    def test_actual_externally_restored_route_retires_expired_attempt_without_input(self):
        c=self.pending('stop');native=frame('native-restored-gap-negative')
        c.journal.update(rank=39,stage=8,value=1562,high=1810)
        c.world.training.state['rank']=39
        c.expired_observe_until=115
        c._save();original=c.path.read_bytes()
        with patch('meter_calibration.time.time',return_value=1100):
            self.assertFalse(c.input_allowed())
            for now in (100,101.5,103):
                with patch('meter_calibration.time.monotonic',return_value=now): c.step(native.copy())
        self.assertFalse(c.active)
        self.assertEqual(c.journal['resolution']['kind'],'externally_restored')
        self.assertEqual((c.journal['resolution']['player'],c.journal['resolution']['value']),(1746,1764))
        self.assertEqual(Path(c.journal['resolution']['archive']).read_bytes(),original)
        self.assertEqual(c.journal['deadline'],1090)
        self.assertEqual((c.cycles,c.fishing.confirmed_starts),(0,0))
        self.assertEqual(self.inputs,[])

    def test_expired_unknown_quarantines_without_extending_input_deadline(self):
        c=self.pending();c.expired_observe_until=115;c._save();original=c.path.read_bytes()
        with patch('meter_calibration.time.time',return_value=1100), \
             patch('meter_calibration.time.monotonic',return_value=115):
            self.assertFalse(c.step(np.zeros((1280,720,3),dtype='uint8')))
        self.assertTrue(c.awaiting_recovery)
        self.assertEqual(Path(c.journal['retirement']['archive']).read_bytes(),original)
        self.assertEqual(c.journal['deadline'],1090)
        self.assertEqual(self.inputs,[])

    def test_expired_resolution_needs_fill_and_original_route(self):
        c=self.pending();c._save();original=c.path.read_bytes()
        native=frame('native-restored-gap-negative')
        for route,fill,value in ((False,True,1764),(True,False,1764)):
            with patch.object(c,'screen',return_value=dict(rank=38,stage=8,route=route,menu=False)), \
                 patch.object(c.fishing,'measure',return_value=dict(player=1746,value=value,context='place')), \
                 patch.object(c.fishing,'blue_fill',return_value=fill):
                self.assertFalse(c._resolve_expired(native))
        self.assertEqual(c.path.read_bytes(),original)
        self.assertEqual(self.inputs,[])

    def test_expired_resumed_route_with_large_gap_is_archived_without_calibration_credit(self):
        # GAP 2026-10-23 由 50 调成 80：这里要用明显超过 80 的差距（121）才符合 large gap 语义
        c=self.pending();native=frame('native-restored-gap-negative')
        c.journal.update(attempts={'cancel':2,'drag':1,'start':2},cycles=7)
        c.cycles=7
        c._save();original=c.path.read_bytes()
        with patch.object(c,'screen',return_value=dict(rank=38,stage=8,route=True,menu=False)), \
             patch.object(c.fishing,'measure',return_value=dict(player=1900,value=1779,context='place')), \
             patch.object(c.fishing,'blue_fill',return_value=True), \
             patch('meter_calibration.time.time',return_value=1100):
            for now,wanted in ((100,False),(101.5,False),(103,True)):
                with patch('meter_calibration.time.monotonic',return_value=now):
                    self.assertEqual(c._resolve_expired(native.copy()),wanted)
        self.assertFalse(c.active)
        self.assertEqual(c.journal['resolution']['kind'],'externally_resumed_uncalibrated')
        self.assertFalse(c.journal['resolution']['calibration_completed'])
        self.assertEqual(Path(c.journal['resolution']['archive']).read_bytes(),original)
        self.assertEqual(c.journal['deadline'],1090)
        self.assertEqual(c.journal['attempts'],{'cancel':2,'drag':1,'start':2})
        self.assertEqual(c.journal['next_check'],1100+c.interval)
        self.assertEqual((c.cycles,c.fishing.confirmed_starts),(7,0))
        self.assertEqual(self.inputs,[])

    def test_readonly_measurement_never_updates_range_record(self):
        c=self.controller;native=frame('native-restored-gap-negative')
        before=deepcopy(c.world.training.state)
        with patch.object(c.fishing,'_remember_limits',side_effect=AssertionError('read-only record write')):
            data=c.fishing.measure(native,remember=False)
        self.assertEqual((data['player'],data['value']),(1746,1764))
        self.assertEqual(c.world.training.state,before)


if __name__ == '__main__':
    unittest.main()
