"""A real lowered Sea God battle must not starve its own cast observation."""
from copy import deepcopy
import hashlib
import json
import time
import unittest
from unittest.mock import patch

from drift.tasks.boss_defense import BossDefense
from drift.tasks.stone_farm import StoneFarm
from tests import test_stone_farm as base
from tests import test_stone_boss_defense as defense_tests


class FirstCombatTests(unittest.TestCase):
    setUpClass = classmethod(base.StoneFarmTests.setUpClass.__func__)
    setUp = base.StoneFarmTests.setUp
    tearDown = base.StoneFarmTests.tearDown
    act = base.StoneFarmTests.act
    tick = base.StoneFarmTests.tick
    begin = base.StoneFarmTests.begin

    def lowered(self):
        image = base.image('tests/fixtures/stone-first-combat/lowered-battle.png')
        self.begin()
        self.w.journal.update(verified_gate_stage=8, entered_verified_at=self.now,
                              first_deadline=self.now+60, first_selected_rank=39,
                              first_signature=self.w.signature(image), fishing_recovery=True,
                              attempts={'enter-confirm':1,'route-start':1,'restore-stop':1,
                                        'first-select':1,'first-cast':1},
                              issued_inputs=[dict(kind=k) for k in
                                             ('enter-confirm','route-start','restore-stop',
                                              'first-select','first-cast')])
        self.w._set('first_verify',15)
        self.w._save()
        return image

    def failed(self):
        image = self.lowered()
        self.w.journal['phase_deadline'] = self.now-1
        self.w._save()
        raw = self.path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(
            rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        return image,raw,digest

    def test_real_battle_confirms_cast_without_game_input_or_timer_extension(self):
        image = self.lowered()
        old = deepcopy(self.w.journal)
        self.assertIsNotNone(self.w._first_result_screen(image))
        for n in range(3):
            self.now += 1.6
            self.assertEqual(self.w.observe_first_cast(image.copy()),n==2)
        self.assertEqual(self.w.phase,'route_open')
        self.assertTrue(self.w.first_route_resume_requested)
        self.assertEqual(self.actions,[])
        for key in ('attempts','issued_inputs','started_at','deadline','first_deadline'):
            self.assertEqual(self.w.journal[key],old[key])
        self.assertEqual(self.w.journal['first_cast_verified_rank'],39)

    def test_same_outer_frame_cannot_count_twice(self):
        image = self.lowered()
        for n in range(2):
            fresh = image.copy()
            self.now += 1.6
            self.w.observe_first_cast(fresh)
            self.w.first_preobserved_frame = fresh
            self.w._first_step(fresh)
        self.assertEqual(self.w.seen,2)
        self.assertEqual(self.w.phase,'first_verify')
        self.now += 1.6
        self.assertTrue(self.w.observe_first_cast(image.copy()))
        self.assertEqual(self.actions,[])

    def test_real_first_vote_prioritizes_observation_without_extending_deadlines(self):
        image = self.lowered()
        old = deepcopy(self.w.journal)
        self.assertFalse(self.w.first_result_observation_pending)
        self.now += 1
        self.w.observe_first_cast(image.copy())
        self.assertTrue(self.w.first_result_observation_pending)
        self.assertEqual(self.w.journal, old)
        self.now = old['phase_deadline'] + 1
        self.assertFalse(self.w.first_result_observation_pending)
        self.assertEqual(self.actions, [])

    def test_new_cast_gets_only_readonly_observation_beyond_the_original_input_limit(self):
        image = self.lowered()
        self.w.phase = self.w.journal['phase'] = 'first_idle'
        self.w.journal['first_deadline'] = self.now + 4
        self.w.journal['phase_deadline'] = self.now + 4
        original = deepcopy(self.w.journal)
        with patch.object(self.w, '_first_idle', return_value=('same',)), \
             patch.object(self.w, '_confirmed', return_value=True), \
             patch.object(self.w.fishing, 'idle_icon', return_value='hook'), \
             patch.object(self.w, '_click', return_value=True) as click:
            self.w._first_step(image.copy())
        self.assertEqual(click.call_count, 1)
        self.assertEqual(self.w.phase, 'first_verify')
        self.assertEqual(self.w.journal['first_observation_deadline'], self.now + 15)
        self.assertEqual(self.w.journal['phase_deadline'], self.now + 15)
        for key in ('deadline', 'first_deadline', 'attempts', 'issued_inputs'):
            self.assertEqual(self.w.journal[key], original[key])
        self.now += 5
        self.assertTrue(self.w.input_allowed())  # Read-only workflow is alive.
        for kind in StoneFarm.INPUT_LIMITS:
            action=dict(kind=kind,phase='first_verify',signature=self.w.signature(image),x=360,y=1010)
            self.assertFalse(self.w.validate_portal(action, image), kind)
        for _ in range(3):
            self.now += 1.6
            self.w.observe_first_cast(image.copy())
        self.assertEqual(self.w.phase, 'route_open')
        self.assertEqual(self.actions, [])

    def test_new_observation_is_capped_by_overall_deadline_and_not_reset_on_reentry(self):
        image = self.lowered()
        self.w.journal.update(deadline=self.now+8, first_observation_deadline=self.now+15)
        self.w.phase = 'first_idle'
        self.w._set('first_verify', 15)
        self.assertEqual(self.w.journal['phase_deadline'], self.now+8)
        saved = deepcopy(self.w.journal)
        self.now += 2
        self.w._set('first_verify', 15)
        self.assertEqual(self.w.journal, saved)
        self.now += 7
        self.assertFalse(self.w.observe_first_cast(image.copy()))
        self.assertEqual(self.w.journal, saved)
        self.assertEqual(self.actions, [])

    def test_expired_observation_never_silently_renews_or_credits_cast(self):
        image,raw,_ = self.failed()
        for _ in range(4):
            self.now += 1.6
            self.assertFalse(self.w.observe_first_cast(image.copy()))
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertNotIn('first_cast_verified_at',self.w.journal)
        self.assertFalse(self.w.first_route_resume_requested)
        self.assertEqual(self.actions,[])

    def test_full_current_world_rank_stop_location_and_battle_are_required(self):
        image = self.lowered()
        self.assertIsNone(self.w._first_result_screen(image//2))
        for owner,name,value in (
            (self.w,'screen',None),(self.w,'_off_icon',False),
            (self.w.training,'boss_mode',False),(self.w.defense,'_battle_footer',False),
            (self.w,'signature',['changed'])):
            self.now = 1000.
            with self.subTest(name=name),patch.object(owner,name,return_value=value):
                for _ in range(3):
                    self.now += 1.1
                    self.w.observe_first_cast(image.copy())
                self.assertEqual(self.w.phase,'first_verify')
        self.assertEqual(self.actions,[])

    def test_lower_result_retry_requires_explicit_hash_bound_unused_route_scope(self):
        _,raw,digest = self.failed()
        for options in ({},{'allow_first_result_restore':True,'allow_boss_abandon':True},
                        {'allow_first_result_restore':'yes'}):
            with self.assertRaises(ValueError):
                StoneFarm.request_retry(self.path.parent,digest,**options)
        with self.assertRaises(ValueError):
            StoneFarm.request_retry(self.path.parent,'0'*64,allow_first_result_restore=True)
        self.assertEqual(self.path.read_bytes(),raw)
        request = StoneFarm.request_retry(self.path.parent,digest,allow_first_result_restore=True)
        self.assertEqual(request['mode'],'lower_first_result_restore')
        self.assertEqual(self.path.read_bytes(),raw)

    def test_one_restore_archives_cast_and_only_uses_the_unfinished_route_open(self):
        image,raw,digest = self.failed()
        old = deepcopy(self.w.journal)
        StoneFarm.request_retry(self.path.parent,digest,allow_first_result_restore=True)
        self.w = StoneFarm(base.ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        self.assertTrue(self.w.first_route_resume_requested)
        for _ in range(3):
            self.tick(image.copy())
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertEqual(self.w.phase,'route_menu')
        self.assertEqual(self.w.journal['first_result_repair_runs'],1)
        self.assertEqual(self.w.journal['attempts'],old['attempts'])
        self.assertEqual(self.w.journal['issued_inputs'],old['issued_inputs'])
        self.assertEqual(self.w.journal['first_result_previous_deadline'],old['deadline'])
        self.assertEqual(self.w.journal['first_result_previous_phase_deadline'],old['phase_deadline'])
        self.assertEqual(self.w.journal['first_result_previous_started_at'],old['started_at'])
        self.assertEqual([a['kind'] for a in self.actions],['restore-route-open'])
        self.assertEqual(self.w.journal['first_cast_verified_rank'],39)
        self.assertAlmostEqual(self.w.journal['deadline']-self.now,90)
        self.w.phase=self.w.journal['phase']='first_verify'
        self.w.journal['phase_deadline']=self.now-1
        self.w._save()
        with self.assertRaises(ValueError):
            StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest(),
                                    allow_first_result_restore=True)

    def test_restore_rejects_changed_world_rank_route_and_missing_cast_receipt(self):
        image,raw,digest = self.failed()
        StoneFarm.request_retry(self.path.parent,digest,allow_first_result_restore=True)
        self.w = StoneFarm(base.ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        for screen in (None,dict(stage=9,rank=39,route=False,menu=False,completion=False),
                       dict(stage=8,rank=38,route=False,menu=False,completion=False),
                       dict(stage=8,rank=39,route=True,menu=False,completion=False)):
            with patch.object(self.w,'screen',return_value=screen):
                for _ in range(3):self.tick(image.copy())
            self.assertEqual(self.path.read_bytes(),raw)
            self.assertEqual(self.actions,[])
        saved = deepcopy(self.w.journal)
        for key in ('enter-confirm','first-select','first-cast'):
            self.w.journal=deepcopy(saved)
            self.w.journal['issued_inputs']=[a for a in saved['issued_inputs'] if a['kind']!=key]
            self.w._save()
            with self.assertRaises(ValueError):
                StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest(),
                                        allow_first_result_restore=True)


class FirstCombatRuntimeTests(unittest.TestCase):
    def test_confirming_real_cast_does_not_spend_remaining_time_on_combat_burst(self):
        original_init = StoneFarm.__init__
        calls = []
        def initialize(worker, *args, **kwargs):
            original_init(worker, *args, **kwargs)
            worker.phase = worker.journal['phase'] = 'first_verify'
        def observe(worker, image):
            calls.append('observe')
            worker.seen, worker.first_result_rank = 1, 39
            return False
        def battle(worker, *args):
            calls.append('battle')
            return True
        with patch.object(StoneFarm, '__init__', initialize), \
             patch.object(StoneFarm, 'observe_first_cast', observe), \
             patch.object(StoneFarm, 'input_allowed', return_value=True), \
             patch.object(BossDefense, 'step', battle):
            defense_tests.StoneBossDefenseTests().exercise([defense_tests.frame('native-right-charge-low')])
        self.assertIn('observe', calls)
        self.assertNotIn('battle', calls)

    def test_worker_observes_first_cast_before_combat_can_continue_the_loop(self):
        original_init = StoneFarm.__init__
        calls = []
        def initialize(worker,*args,**kwargs):
            original_init(worker,*args,**kwargs)
            worker.phase=worker.journal['phase']='first_verify'
        def observe(worker,image):
            calls.append('observe')
            return False
        def battle(worker,*args):
            calls.append('battle')
            return True
        with patch.object(StoneFarm,'__init__',initialize), \
             patch.object(StoneFarm,'observe_first_cast',observe), \
             patch.object(BossDefense,'step',battle):
            defense_tests.StoneBossDefenseTests().exercise([defense_tests.frame('native-right-charge-low')])
        self.assertGreaterEqual(calls.count('observe'),1)
        self.assertEqual(calls[0:2],['observe','battle'])
        self.assertTrue(all(calls[i:i+2]==['observe','battle'] for i in range(0,len(calls),2)))
