"""Explicit read-only reconciliation never retries an expired input window."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

from drift.tasks.stone_farm import StoneFarm
from tests import test_stone_farm as base


class ObservedRestoreTests(unittest.TestCase):
    setUpClass=classmethod(base.StoneFarmTests.setUpClass.__func__)
    setUp=base.StoneFarmTests.setUp
    tearDown=base.StoneFarmTests.tearDown
    act=base.StoneFarmTests.act
    tick=base.StoneFarmTests.tick
    begin=base.StoneFarmTests.begin

    def failed(self):
        self.begin()
        self.now+=200
        kinds=('stop','handoff-cancel','return-request','return-confirm','lower',
               'lower-confirm','enter-confirm','route-start')
        self.w.journal.update(phase='route_running',phase_deadline=self.now-90,
            deadline=self.now-1,verified_gate_stage=8,entered_verified_at=self.now-150,
            attempts={k:1 for k in kinds},issued_inputs=[dict(kind=k) for k in kinds],
            cycles=2,handoffs=6,recoveries=4)
        self.w.phase='route_running'
        self.w._save()
        raw=self.path.read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[
            dict(id='driftinggirl_stone_farm',enabled=True)])))
        return raw,digest

    def approved(self):
        raw,digest=self.failed()
        request=StoneFarm.request_retry(self.path.parent,digest,allow_observed_lower_restore=True)
        self.assertFalse(request['allow_ordinary_cancel'])
        self.assertEqual(request['mode'],'observed_lower_restore')
        self.assertEqual(request['expires_at']-request['approved_at'],60)
        self.assertEqual(self.path.read_bytes(),raw)
        self.w=StoneFarm(base.ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        frame=base.image('tests/fixtures/stone-lower-entry/observed-moving-fill.png')
        return frame,raw,digest

    def test_existing_expired_window_still_rejects_default_worker_start(self):
        raw,_=self.failed()
        with self.assertRaises(RuntimeError):self.tick()
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual(self.actions,[])

    def test_three_seconds_of_real_fishing_resolves_without_inputs_or_success_credit(self):
        frame,raw,digest=self.approved()
        old=json.loads(raw)
        self.assertTrue(self.w.first_route_resume_requested)
        for _ in range(2):self.tick(frame.copy())
        self.assertEqual(self.path.read_bytes(),raw)
        self.tick(frame.copy())
        self.assertEqual(self.actions,[])
        self.assertFalse(self.w.active)
        self.assertIsNone(self.w.retry)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        for k in ('started_at','deadline','phase_deadline','return_deadline','attempts',
                  'issued_inputs','inputs','handoffs','cycles'):
            self.assertEqual(self.w.journal.get(k),old.get(k),k)
        self.assertEqual(self.w.journal['recoveries'],old['recoveries']+1)
        self.assertEqual(self.w.journal['resolution'],'observed_fishing_after_expired_lower_handoff')
        self.assertGreaterEqual(self.w.journal['observed_lower_proof']['seconds'],3)
        self.assertEqual(self.w.journal['observed_lower_proof']['game_inputs'],0)
        with self.assertRaises(ValueError):
            StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest(),
                                    allow_observed_lower_restore=True)

    def test_running_arrow_alone_or_changed_world_rank_never_resolves(self):
        frame,raw,_=self.approved()
        valid=self.w.screen(frame)
        for sc in (None,{**valid,'stage':9},{**valid,'rank':40},{**valid,'rank':38},
                   {**valid,'route':False},{**valid,'route':None},{**valid,'menu':True}):
            with self.subTest(screen=sc),patch.object(self.w,'screen',return_value=sc):
                for _ in range(3):self.tick(frame.copy())
                self.assertEqual(self.path.read_bytes(),raw)
        with patch.object(self.w.fishing,'blue_fill',return_value=False), \
             patch.object(self.w.defense,'_battle_footer',return_value=False):
            for _ in range(3):self.tick(frame.copy())
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual(self.actions,[])

    def test_unknown_frame_resets_votes_and_cannot_be_committed(self):
        frame,raw,_=self.approved()
        self.tick(frame.copy());self.tick(frame.copy())
        with patch.object(self.w,'screen',return_value=None):self.tick(frame.copy())
        self.tick(frame.copy());self.tick(frame.copy())
        self.assertEqual(self.path.read_bytes(),raw)
        self.tick(frame.copy())
        self.assertFalse(self.w.active)
        self.assertEqual(self.actions,[])

    def test_battle_with_unrecognized_footer_does_not_bypass_full_screen_guard(self):
        _,raw,_=self.approved()
        frame=base.image('tests/fixtures/stone-lower-entry/observed-moving-battle.png')
        self.assertFalse(self.w.defense._battle_footer(frame))
        for _ in range(3):self.tick(frame.copy())
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual(self.actions,[])

    def test_expired_or_stopped_observation_never_renews_old_deadlines(self):
        frame,raw,_=self.approved()
        self.w.observed_restore_allowed=lambda:False
        with self.assertRaises(RuntimeError):self.tick(frame)
        self.assertEqual(self.path.read_bytes(),raw)
        self.w.observed_restore_allowed=lambda:True
        self.now=self.w.retry['expires_at']+1
        with self.assertRaises(RuntimeError):self.tick(frame)
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual(self.actions,[])

    def test_stop_just_before_commit_preserves_failed_record(self):
        frame,raw,_=self.approved()
        def stop_at_confirmation(*args,**kwargs):
            self.w.observed_restore_allowed=lambda:False
            return True
        with patch.object(self.w,'_confirmed',side_effect=stop_at_confirmation):
            with self.assertRaises(RuntimeError):self.tick(frame)
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual(self.actions,[])

    def test_approval_is_exact_hash_scope_and_cannot_include_new_input_permissions(self):
        raw,digest=self.failed()
        for kwargs in ({'allow_observed_lower_restore':'yes'},
                       {'allow_observed_lower_restore':True,'allow_boss_abandon':True},
                       {'allow_observed_lower_restore':True,'allow_continuous_cancel':True}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                StoneFarm.request_retry(self.path.parent,digest,**kwargs)
        with self.assertRaises(ValueError):
            StoneFarm.request_retry(self.path.parent,'0'*64,allow_observed_lower_restore=True)
        old=deepcopy(self.w.journal)
        for k in ('lower-confirm','enter-confirm','route-start'):
            self.w.journal=deepcopy(old)
            self.w.journal['issued_inputs']=[a for a in old['issued_inputs'] if a['kind']!=k]
            self.w._save()
            with self.assertRaises(ValueError):
                StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest(),
                                        allow_observed_lower_restore=True)
        self.assertEqual(self.actions,[])


class ObservedRestoreEndpointTests(unittest.TestCase):
    setUp=base.StoneRuntimeTests.setUp
    tearDown=base.StoneRuntimeTests.tearDown

    def test_read_only_endpoint_has_no_cancel_or_other_input_permission(self):
        data=dict(failed_sha256='x',allow_observed_lower_restore=True)
        with patch('app.StoneFarm.request_retry',return_value={'approved':True}) as approve:
            self.app.retry_stone_farm(data)
            approve.assert_called_once_with(self.root,'x',allow_observed_lower_restore=True)
        for field in ('allow_ordinary_cancel','allow_boss_abandon','allow_continuous_cancel'):
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.app.retry_stone_farm({**data,field:True})
        with patch.object(self.app,'running',return_value=True),self.assertRaises(ValueError):
            self.app.retry_stone_farm(data)


if __name__=='__main__':
    unittest.main()
