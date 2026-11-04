"""Real selected-map replay and worker-level first-cast deadline guards."""
from copy import deepcopy
import time
import unittest
from unittest.mock import patch

from drift.tasks.stone_farm import StoneFarm
from drift.tasks.boss_defense import BossDefense
from tests import test_stone_farm as base
from tests import test_stone_boss_defense as runtime_base


class FirstBudgetTests(unittest.TestCase):
    setUpClass=classmethod(base.StoneFarmTests.setUpClass.__func__)
    setUp=base.StoneFarmTests.setUp
    tearDown=base.StoneFarmTests.tearDown
    act=base.StoneFarmTests.act
    begin=base.StoneFarmTests.begin

    def selected(self):
        frame=base.image('tests/fixtures/stone-first-budget/selected-map.png')
        self.begin()
        self.w.journal.update(verified_gate_stage=8,entered_verified_at=self.now,
            first_deadline=self.now+60,first_selected_rank=39,first_selected_point=[197,485],
            first_unselected=True,first_signature=self.w.signature(frame),
            attempts={'first-select':1},issued_inputs=[{'kind':'first-select'}])
        self.w._set('first_close',30)
        return frame

    def test_actual_selected_skull_is_relocated_without_global_scan(self):
        frame=self.selected();old=deepcopy(self.w.journal)
        self.assertEqual(self.w._map_task_target(frame)[:2],(197,485))
        with patch('drift.tasks.parallel_world.ParallelWorld._map_task_target',side_effect=AssertionError('global scan')):
            self.assertTrue(self.w._first_selected_map(frame.copy()))
        self.assertEqual(self.w.journal,old);self.assertEqual(self.actions,[])

    def test_saved_coordinate_without_current_skull_cannot_close_map(self):
        frame=self.selected();frame[445:525,157:237]=0
        self.assertFalse(self.w._first_selected_map(frame))
        frame=self.selected();self.w.journal['first_selected_point']=[400,485]
        self.assertFalse(self.w._first_selected_map(frame))
        self.assertEqual(self.actions,[])

    def test_fresh_world_rank_off_and_selection_receipt_still_required(self):
        frame=self.selected()
        for owner,name,value in ((self.w,'_parallel',False),(self.w,'_rank_value',38),
                (self.w,'_stage',9),(self.w,'_off_icon',False),(self.w.training,'boss_mode',False)):
            with self.subTest(name=name),patch.object(owner,name,return_value=value):
                self.assertFalse(self.w._first_selected_map(frame))
        self.w.journal['issued_inputs']=[]
        self.assertFalse(self.w._first_selected_map(frame));self.assertEqual(self.actions,[])

    def test_untrusted_or_out_of_map_point_is_rejected(self):
        frame=self.selected()
        for point in ([197],[-1,485],[197,1280],['197',485],[197.,485],{}):
            self.w.journal['first_selected_point']=point
            self.assertFalse(self.w._first_selected_map(frame))

    def test_expired_first_budget_preserves_every_journal_field(self):
        self.selected();self.w._set('first_verify',15)
        old=deepcopy(self.w.journal);raw=self.path.read_bytes()
        self.now=self.w.journal['phase_deadline']+1
        with self.assertRaisesRegex(RuntimeError,'first_verify'):self.w.check_deadline()
        self.assertEqual(self.w.journal,old);self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual(self.actions,[])


class FirstBudgetRuntimeTests(unittest.TestCase):
    def test_deadline_reached_inside_burst_prevents_a_second_contact(self):
        original=StoneFarm.__init__;workers=[]
        def initialize(worker,*args,**kw):
            original(worker,*args,**kw);workers.append(worker)
            worker.phase='first_verify'
            worker.journal.update(pending=True,phase='first_verify',deadline=time.time()+30,
                phase_deadline=time.time()+30,first_deadline=time.time()+30)
        def expire():workers[-1].journal['phase_deadline']=time.time()-1
        with patch.object(StoneFarm,'__init__',initialize), \
                patch.object(StoneFarm,'observe_first_cast',return_value=False):
            inputs,_=runtime_base.StoneBossDefenseTests().exercise(
                [runtime_base.frame('native-right-charge-low')],on_touch=expire,
                expected_error='阶段=first_verify')
        self.assertEqual(len(inputs),1)

    def test_expired_handoff_stops_before_real_combat_gets_any_input(self):
        original=StoneFarm.__init__
        def initialize(worker,*args,**kw):
            original(worker,*args,**kw)
            worker.phase='first_verify'
            worker.journal.update(pending=True,phase='first_verify',deadline=time.time()+30,
                phase_deadline=time.time()-1,first_deadline=time.time()-1)
        with patch.object(StoneFarm,'__init__',initialize),patch.object(BossDefense,'step') as battle:
            inputs,_=runtime_base.StoneBossDefenseTests().exercise(
                [runtime_base.frame('native-right-charge-low')],expected_error='阶段=first_verify')
        self.assertEqual(inputs,[]);battle.assert_not_called()

    def test_first_verification_bounds_burst_without_changing_other_fights(self):
        original_init=StoneFarm.__init__;original_burst=BossDefense.burst
        for first,expected in ((True,.5),(False,4.)):
            limits=[]
            def initialize(worker,*args,**kw):
                original_init(worker,*args,**kw)
                if first:worker.phase='first_verify'
            def burst(worker,*args,**kw):
                limits.append(kw.get('seconds',4.))
                return original_burst(worker,*args,**kw)
            with self.subTest(first=first),patch.object(StoneFarm,'__init__',initialize), \
                    patch.object(StoneFarm,'observe_first_cast',return_value=False), \
                    patch.object(BossDefense,'burst',burst):
                runtime_base.StoneBossDefenseTests().exercise([runtime_base.frame('native-right-charge-low')])
            self.assertTrue(limits);self.assertEqual(set(limits),{expected})


if __name__=='__main__':unittest.main()
