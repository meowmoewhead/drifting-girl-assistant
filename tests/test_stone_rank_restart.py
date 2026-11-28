"""Verify the early restart uses the existing guarded portal and first cast."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from drift.runtime import Clicker
from drift.tasks.entry_recovery import identity
from drift.tasks.features import feature_states
from drift.tasks.stone_farm import StoneFarm
from tests import test_stone_farm as base


def screen(stage=10, rank=35, *, route=True, completion=False):
    return dict(stage=stage, rank=rank, menu=False, route=route, completion=completion)


class RankRestartTests(unittest.TestCase):
    setUpClass = classmethod(base.StoneFarmTests.setUpClass.__func__)
    setUp = base.StoneFarmTests.setUp
    tearDown = base.StoneFarmTests.tearDown
    act = base.StoneFarmTests.act
    tick = base.StoneFarmTests.tick
    action = base.StoneFarmTests.action

    def test_cutoffs_and_already_passed_rank(self):
        for rank in (1, 30, 35, 39, 40):
            self.w.restart_rank = rank
            self.assertFalse(self.w.handoff_due(screen(rank=rank-1)))
            self.assertTrue(self.w.handoff_due(screen(rank=rank)))
            if rank < 40:
                self.assertTrue(self.w.handoff_due(screen(rank=rank+1)))
            self.assertFalse(self.w.handoff_due(screen(rank=rank, stage=0)))

    def test_rank35_can_close_stage10_gate_and_cannot_lower_or_close_wrong_gate(self):
        self.w._begin(screen())
        self.w._set('manager')
        self.assertTrue(self.w.restarting_gate)
        self.assertFalse(self.w.completed_rank_40)
        manager = base.image('tests/fixtures/stone-stage10/manager-0.png')
        wrong = base.image('assets/recovery/parallel-manager.png')
        for frame, valid in ((manager, True), (manager//2, False), (wrong, False)):
            self.assertEqual(self.w.validate_portal(self.action('close-gate',(354,739),frame),frame),valid)
            self.assertFalse(self.w.validate_portal(self.action('lower',(357,873),frame),frame))
        for _ in range(3):
            self.tick(manager.copy())
        self.assertEqual([a['kind'] for a in self.actions], ['close-gate'])
        self.assertEqual(self.w.phase, 'close_confirm')

    def test_retired_lowering_and_its_approval_are_archived_without_replay(self):
        old = dict(self.w.journal, pending=True, phase='lower_confirm', reason='lower',
                   from_stage=10, to_stage=9, rank=39, entry_rank=39,
                   deadline=self.now+600, phase_deadline=self.now+120,
                   attempts={'lower':1}, issued_inputs=[dict(kind='lower')])
        raw = json.dumps(old).encode()
        digest = hashlib.sha256(raw).hexdigest()
        self.path.write_bytes(raw)
        retry_path = self.path.with_name('stone_farm_retry.json')
        approval = json.dumps(dict(failed_sha256=digest, allow_ordinary_cancel=True,
                                   mode='return_retry',expires_at=self.now+60)).encode()
        retry_path.write_bytes(approval)
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[
            dict(id='driftinggirl_stone_farm',enabled=True)])))
        self.w = StoneFarm(base.ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertFalse(self.w.active)
        self.assertIsNone(self.w.retry)
        with self.assertRaises(ValueError):
            StoneFarm.request_retry(self.path.parent,digest)
        self.assertTrue(self.w.retire_expired_on_start())
        archive=self.path.parent/'logs/stone-farm-retired'
        self.assertEqual((archive/f'{digest}.json').read_bytes(),raw)
        self.assertEqual((archive/f'{digest}-retry.json').read_bytes(),approval)
        self.assertFalse(retry_path.exists())
        self.assertEqual(self.w.journal['attempts'],old['attempts'])
        self.assertEqual(self.w.journal['deadline'],old['deadline'])
        self.assertEqual(self.actions,[])
        with patch.object(self.w,'screen',return_value=screen(rank=39)):
            for _ in range(3):
                self.tick()
        self.assertEqual(self.w.journal['reason'],'restart')
        self.assertEqual(self.w.journal['entry_rank'],0)
        self.assertEqual([a['kind'] for a in self.actions],['stop'])

    def test_new_gate_after_close_accepts_real_stage_and_rejects_lower_stage(self):
        self.w._begin(screen())
        self.w.journal['gate_closed'] = True
        self.w._set('new_gate')
        lower = base.image('assets/recovery/parallel-manager.png')
        for _ in range(3):
            self.tick(lower.copy())
        self.assertEqual(self.w.phase, 'new_gate')
        manager = base.image('tests/fixtures/stone-stage10/manager-0.png')
        for _ in range(3):
            self.tick(manager.copy())
        self.assertEqual((self.w.phase,self.w.journal['verified_gate_stage']), ('panel_close',10))

    def test_forward_rank_during_stop_reconfirms_without_renewing_limits(self):
        self.w._begin(screen())
        self.w._set('stop_fishing')
        self.w.journal['attempts']['stop'] = 1
        self.w.journal['attempts']['handoff-cancel'] = 1
        self.w.off_verified = self.w.fishing_stopped = True
        old = deepcopy(self.w.journal)
        frame = self.main.copy()
        with patch.object(self.w,'screen',return_value=screen(rank=36,route=False)):
            for _ in range(3):
                self.now += 1.6
                self.assertIsNone(self.w._expected(frame,advance=True))
            self.assertEqual(self.w.journal['rank'],35)
            for _ in range(3):
                self.now += 1.6
                self.w._expected(frame.copy(),advance=True)
        self.assertEqual(self.w.journal['rank'],36)
        self.assertEqual(self.w.journal['restart_rank'],35)
        self.assertFalse(self.w.off_verified)
        self.assertFalse(self.w.fishing_stopped)
        for key in ('deadline','phase_deadline','return_deadline','stop_fishing_deadline','attempts'):
            self.assertEqual(self.w.journal[key],old[key])

    def test_new_first_cast_record_can_resume_without_changing_saved_cutoff(self):
        self.w._begin(screen(rank=36))
        self.w.journal.update(gate_closed=True, entered_verified_at=self.now, verified_gate_stage=10,
                              issued_inputs=[dict(kind='enter-confirm', input_id='real-entry')])
        self.w._set('route_open')
        old = self.path.read_bytes()
        restored = StoneFarm(base.ROOT, self.reader, self.act, lambda _:None, state_path=self.path)
        restored.restart_rank = 40
        self.assertTrue(restored._first_scope())
        self.assertEqual(restored.journal['restart_rank'],35)
        self.assertTrue(identity(restored.journal,'emulator'))
        self.assertEqual(self.path.read_bytes(),old)

    def test_unverified_return_does_not_advance_to_closing(self):
        self.w._begin(screen())
        self.w._set('return')
        with patch.object(self.w,'_original',return_value=True):
            self.tick()
            self.assertEqual(self.w.phase,'return')
            self.w.journal['issued_inputs']=[dict(kind='return-confirm')]
            self.tick()
            self.assertEqual(self.w.phase,'original')


class RestartSettingsTests(unittest.TestCase):
    def test_backend_default_and_later_edit_without_rewriting_other_settings(self):
        rule = dict(id='driftinggirl_stone_farm', enabled=True, interval=2)
        current = SimpleNamespace(lock=threading.RLock(), running=lambda: False,
                                  config=dict(rules=[rule], serial='same'), event=lambda _:None)
        current.commit_config=lambda data: setattr(current,'config',data)
        current.feature_status=lambda: feature_states(current.config)
        self.assertEqual(next(f for f in current.feature_status() if f['id']=='stone_farm')['restart_rank'],35)
        for rank in (False,0,41,35.5,'35'):
            with self.assertRaises(ValueError):
                Clicker.change_feature(current,dict(id='stone_farm',restart_rank=rank))
        self.assertEqual(current.config,dict(rules=[rule],serial='same'))
        result=Clicker.change_feature(current,dict(id='stone_farm',restart_rank=32))
        self.assertEqual(result['restart_rank'],32)
        self.assertEqual(current.config,dict(rules=[{**rule,'restart_rank':32}],serial='same'))
