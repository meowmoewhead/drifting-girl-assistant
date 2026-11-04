"""Manual RANK1 progress cannot replay an expired RANK0 task or credit a cast."""
import hashlib
import json
import unittest
from unittest.mock import patch
from drift.tasks.stone_farm import StoneFarm
from tests import test_stone_farm as base

ROOT,image=base.ROOT,base.image
class ManualProgressTests(unittest.TestCase):
    setUpClass=classmethod(base.StoneFarmTests.setUpClass.__func__)
    setUp=base.StoneFarmTests.setUp
    tearDown=base.StoneFarmTests.tearDown
    act=base.StoneFarmTests.act
    tick=base.StoneFarmTests.tick
    begin=base.StoneFarmTests.begin

    def failed(self):
        self.begin(cycle=True)
        kinds=('completion','stop','return-request','return-confirm','activate','enter',
            'enter-confirm','route-start','restore-stop','first-map','first-select','first-close')
        self.w.journal.update(phase='first_close',phase_deadline=self.now-1,
            verified_gate_stage=9,entered_verified_at=self.now-80,first_selected_rank=0,
            first_deadline=self.now-1,attempts={k:1 for k in kinds},
            issued_inputs=[dict(kind=k) for k in kinds],gate_closed=True,
            resumed_after_unverified_close=True)
        self.w.phase='first_close';self.w._save()
        raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        return raw,digest

    def test_explicit_manual_progress_required_and_old_counts_are_preserved(self):
        raw,digest=self.failed()
        with self.assertRaises(ValueError):StoneFarm.request_retry(self.path.parent,digest)
        request=StoneFarm.request_retry(self.path.parent,digest,allow_manual_rank_progress=True)
        self.assertEqual(request['mode'],'manual_rank_progress_restore')
        self.assertEqual(self.path.read_bytes(),raw)
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        frame=image('tests/fixtures/stone-farm/stage9-rank1-port-manual.png')
        with patch.object(self.w,'_rank_value',return_value=0):
            self.tick(frame)
        self.assertEqual(self.path.read_bytes(),raw)
        for _ in range(3):self.tick(frame)
        self.assertIsNone(self.w.retry)
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertEqual(self.w.journal['attempts'],json.loads(raw)['attempts'])
        self.assertEqual(self.w.journal['return_deadline'],json.loads(raw)['return_deadline'])
        self.assertEqual(self.w.journal['first_deadline'],json.loads(raw)['first_deadline'])
        self.assertNotIn('first_cast_verified_at',self.w.journal)
        self.assertEqual(self.w.journal['manual_rank_progress_observed_rank'],1)
        self.assertEqual(self.w.phase,'route_menu')
        self.assertEqual([a['kind'] for a in self.actions],['restore-route-open'])
        self.assertFalse(self.w.reserve_meter_input('start'))
        self.assertFalse(self.w.reserve_meter_input('drag'))

    def test_prior_cast_or_second_manual_restore_is_rejected(self):
        _,_=self.failed()
        for change in ('first-cast','manual_rank_progress_runs'):
            with self.subTest(change=change):
                saved=json.loads(self.path.read_text())
                if change=='first-cast':self.w.journal['attempts'][change]=1
                else:self.w.journal[change]=1
                self.w._save()
                with self.assertRaises(ValueError):StoneFarm.request_retry(self.path.parent,
                    hashlib.sha256(self.path.read_bytes()).hexdigest(),allow_manual_rank_progress=True)
                self.w.journal=saved;self.w._save()

    def test_only_real_fishing_finishes_and_never_repeats_old_task_or_meter(self):
        self.failed()
        self.w.journal.update(manual_rank_progress_runs=1,manual_rank_progress_observed_rank=1,
            phase='route_running',phase_deadline=self.now+90)
        self.w.phase='route_running'
        frame=image('tests/fixtures/stone-farm/stage9-rank1-moving-filled.png')
        empty=frame.copy();empty[935:1090,282:439]=0
        with patch.object(self.w.fishing,'blue_fill',return_value=False),patch.object(self.w,'_battle',return_value=False):
            for _ in range(3):self.tick(frame)
        self.assertTrue(self.w.active)
        self.assertEqual(self.actions,[])
        for _ in range(3):self.tick(frame)
        self.assertFalse(self.w.active)
        self.assertEqual(self.w.journal['cycles'],0)
        self.assertEqual(self.w.journal['recoveries'],1)
        self.assertEqual(self.w.journal['restored_rank'],1)
        self.assertEqual(self.w.journal['resolution'],'route_restored_after_manual_rank_progress')
        self.assertNotIn('first_cast_verified_at',self.w.journal)
        self.assertEqual(self.actions,[])

    def test_manual_progress_never_counts_as_full_cycle_even_with_gate_input_proof(self):
        self.failed()
        self.w.journal.pop('resumed_after_unverified_close')
        self.w.journal['issued_inputs'].append(dict(kind='close-gate'))
        self.w.journal.update(manual_rank_progress_runs=1,manual_rank_progress_observed_rank=1,
            phase='route_running',phase_deadline=self.now+90)
        self.w.phase='route_running'
        frame=image('tests/fixtures/stone-farm/stage9-rank1-moving-filled.png')
        for _ in range(3):self.tick(frame)
        self.assertFalse(self.w.active)
        self.assertEqual(self.w.journal['cycles'],0)
        self.assertEqual(self.w.journal['recoveries'],1)
        self.assertEqual(self.w.journal['restored_rank'],1)
        self.assertEqual(self.w.journal['resolution'],'route_restored_after_manual_rank_progress')
        self.assertNotIn('first_cast_verified_at',self.w.journal)
        self.assertEqual(self.actions,[])

    def test_new_gate_selects_first_task_before_rare_first_sailing(self):
        self.begin(cycle=True)
        self.w.journal.update(verified_gate_stage=9)
        self.w._set('arrive',90)
        frame=image('tests/fixtures/stone-farm/stage9-rank0-unselected-stopped.png')
        for _ in range(3):self.tick(frame)
        self.assertEqual(self.w.phase,'first_map')
        self.assertNotIn('route-start',self.w.journal['attempts'])
        self.assertNotIn('meter_attempts',self.w.journal)
        self.assertLessEqual(self.w.journal['first_deadline']-self.now,60+1e-9)
        self.assertEqual(self.actions,[])
