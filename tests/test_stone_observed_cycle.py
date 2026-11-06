"""Hash-bound, input-free adoption of a new round after cast verification expired."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

from drift.tasks.stone_farm import StoneFarm
from tests import test_stone_farm as base


class ObservedCycleTests(unittest.TestCase):
    setUpClass=classmethod(base.StoneFarmTests.setUpClass.__func__)
    setUp=base.StoneFarmTests.setUp
    tearDown=base.StoneFarmTests.tearDown
    act=base.StoneFarmTests.act
    tick=base.StoneFarmTests.tick
    begin=base.StoneFarmTests.begin

    def failed(self):
        self.begin()
        self.now+=200
        kinds=('return-request','return-confirm','enter-confirm','first-map',
               'first-select','first-close','first-cast')
        self.w.journal.update(reason='cycle',rank=40,from_stage=8,to_stage=9,entry_rank=0,
            phase='first_verify',phase_deadline=self.now-30,deadline=self.now-1,
            first_deadline=self.now-25,rank40_verified=True,verified_gate_stage=9,
            entered_verified_at=self.now-100,first_selected_rank=0,
            attempts={k:1 for k in kinds},issued_inputs=[dict(kind=k) for k in kinds],
            cycles=2,handoffs=6,recoveries=7)
        self.w.phase='first_verify'
        self.w._save()
        raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[
            dict(id='driftinggirl_stone_farm',enabled=True)])))
        return raw,digest

    def approved(self):
        raw,digest=self.failed()
        request=StoneFarm.request_retry(self.path.parent,digest,allow_observed_cycle_restore=True)
        self.assertFalse(request['allow_ordinary_cancel'])
        self.assertEqual(request['mode'],'observed_cycle_restore')
        self.assertEqual(request['expires_at']-request['approved_at'],60)
        self.assertEqual(self.path.read_bytes(),raw)
        self.w=StoneFarm(base.ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        frame=base.image('tests/fixtures/stone-first-result/rank9-running-fill.png')
        return frame,raw,digest

    def test_real_rank9_requires_three_seconds_and_never_credits_cast_or_cycle(self):
        frame,raw,digest=self.approved();old=json.loads(raw)
        for _ in range(2):self.tick(frame.copy())
        self.assertEqual(self.path.read_bytes(),raw)
        self.tick(frame.copy())
        self.assertFalse(self.w.active);self.assertEqual(self.actions,[])
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        for key in ('started_at','deadline','phase_deadline','return_deadline','first_deadline',
                    'attempts','issued_inputs','inputs','cycles','handoffs'):
            self.assertEqual(self.w.journal.get(key),old.get(key),key)
        self.assertEqual(self.w.journal['recoveries'],8)
        self.assertEqual(self.w.journal['restored_rank'],9)
        self.assertEqual(self.w.journal['observed_cycle_proof']['game_inputs'],0)
        self.assertGreaterEqual(self.w.journal['observed_cycle_proof']['seconds'],3)
        self.assertNotIn('first_cast_verified_at',self.w.journal)
        self.assertEqual(self.w.journal['resolution'],'observed_fishing_after_expired_first_cast')
        with self.assertRaises(ValueError):
            StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest(),
                                    allow_observed_cycle_restore=True)

    def test_error_frame_is_real_rank1_but_not_permission_to_restart(self):
        raw,_=self.failed()
        frame=base.image('tests/fixtures/stone-first-result/timeout-rank1.png')
        self.assertEqual(self.w._first_result_screen(frame)['rank'],1)
        self.assertFalse(self.w.observe_first_cast(frame))
        with self.assertRaises(RuntimeError):self.tick(frame.copy())
        self.assertEqual(self.path.read_bytes(),raw);self.assertEqual(self.actions,[])

    def test_scope_requires_full_current_fishing_and_same_world(self):
        frame,raw,_=self.approved();valid=self.w.screen(frame)
        for sc in (None,{**valid,'stage':8},{**valid,'rank':0},{**valid,'rank':39},
                   {**valid,'rank':40},{**valid,'rank':51},{**valid,'route':False},
                   {**valid,'route':None},{**valid,'menu':True},{**valid,'completion':True}):
            with self.subTest(screen=sc),patch.object(self.w,'screen',return_value=sc):
                for _ in range(3):self.tick(frame.copy())
                self.assertEqual(self.path.read_bytes(),raw)
        with patch.object(self.w.fishing,'blue_fill',return_value=False), \
             patch.object(self.w.defense,'_battle_footer',return_value=False):
            for _ in range(3):self.tick(frame.copy())
        self.assertEqual(self.path.read_bytes(),raw);self.assertEqual(self.actions,[])

    def test_unknown_or_changed_location_restarts_consensus(self):
        frame,raw,_=self.approved()
        self.tick(frame.copy());self.tick(frame.copy())
        with patch.object(self.w,'screen',return_value=None):self.tick(frame.copy())
        self.tick(frame.copy());self.tick(frame.copy())
        self.assertEqual(self.path.read_bytes(),raw)
        with patch.object(self.w,'signature',return_value=['changed']):self.tick(frame.copy())
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual(self.actions,[])

    def test_stop_expiry_and_changed_journal_block_commit(self):
        frame,raw,_=self.approved()
        self.w.observed_restore_allowed=lambda:False
        with self.assertRaises(RuntimeError):self.tick(frame.copy())
        self.assertEqual(self.path.read_bytes(),raw)
        self.w.observed_restore_allowed=lambda:True
        self.now=self.w.retry['expires_at']+1
        with self.assertRaises(RuntimeError):self.tick(frame.copy())
        self.assertEqual(self.path.read_bytes(),raw);self.assertEqual(self.actions,[])

    def test_hash_change_after_votes_cannot_overwrite_new_record(self):
        frame,_,_=self.approved()
        self.tick(frame.copy());self.tick(frame.copy())
        changed=self.path.read_bytes()+b'\n';self.path.write_bytes(changed)
        with self.assertRaises(RuntimeError):self.tick(frame.copy())
        self.assertEqual(self.path.read_bytes(),changed);self.assertEqual(self.actions,[])

    def test_approval_is_narrow_single_use_and_cannot_reuse_old_lower_approval(self):
        raw,digest=self.failed()
        for options in ({'allow_observed_cycle_restore':'yes'},
                        {'allow_observed_cycle_restore':True,'allow_observed_lower_restore':True},
                        {'allow_observed_cycle_restore':True,'allow_continuous_cancel':True},
                        {'allow_observed_lower_restore':True}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                StoneFarm.request_retry(self.path.parent,digest,**options)
        with self.assertRaises(ValueError):
            StoneFarm.request_retry(self.path.parent,'0'*64,allow_observed_cycle_restore=True)
        old=deepcopy(self.w.journal)
        for key in ('return-confirm','enter-confirm','first-select','first-cast'):
            self.w.journal=deepcopy(old)
            self.w.journal['issued_inputs']=[a for a in old['issued_inputs'] if a['kind']!=key]
            self.w._save()
            with self.assertRaises(ValueError):
                StoneFarm.request_retry(self.path.parent,hashlib.sha256(self.path.read_bytes()).hexdigest(),
                                        allow_observed_cycle_restore=True)
        self.assertEqual(self.actions,[])


class ObservedCycleEndpointTests(unittest.TestCase):
    setUp=base.StoneRuntimeTests.setUp
    tearDown=base.StoneRuntimeTests.tearDown

    def test_read_only_endpoint_rejects_any_input_permission_and_running_worker(self):
        data=dict(failed_sha256='x',allow_observed_cycle_restore=True)
        with patch('app.StoneFarm.request_retry',return_value={'approved':True}) as approve:
            self.app.retry_stone_farm(data)
            approve.assert_called_once_with(self.root,'x',allow_observed_cycle_restore=True)
        for field in ('allow_ordinary_cancel','allow_boss_abandon','allow_observed_lower_restore'):
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.app.retry_stone_farm({**data,field:True})
        with patch.object(self.app,'running',return_value=True),self.assertRaises(ValueError):
            self.app.retry_stone_farm(data)


if __name__=='__main__':unittest.main()
