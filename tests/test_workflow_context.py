from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from drift.core.workflow import WorkflowContext,classify


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.now=100.;self.mono=10.
        self.ctx=WorkflowContext(self.temp.name,'serial',wall=lambda:self.now,monotonic=lambda:self.mono)

    def equipment(self):
        return NS(active=True,phase='upgrade',started_at=5.,phase_since=9.,index=0,holds=0,
                  passes=0,cycles=0,pass_holds=0,page='top',stable=3,interrupted=False)

    def test_restart_keeps_remaining_time_and_counters(self):
        task=self.equipment();self.ctx.capture({'equipment_upgrade':task},stage='main')
        self.now=120;self.mono=2
        ctx=WorkflowContext(self.temp.name,'serial',wall=lambda:self.now,monotonic=lambda:self.mono)
        restored=self.equipment();restored.started_at=2;restored.stable=9
        self.assertEqual(ctx.restore('equipment_upgrade',restored),'resume_existing_window')
        self.assertEqual(self.mono-restored.started_at,25.)
        self.assertEqual(restored.stable,0)

    def test_reserved_hold_is_not_repeated_or_counted_as_full_cycle(self):
        task=self.equipment();self.ctx.capture({'equipment_upgrade':task},stage='main')
        self.ctx.reserve('equipment_upgrade','adb',('shell','input','swipe','1','2','1','2','12000'),{'equipment_upgrade':task},{})
        restored=self.equipment();self.ctx.restore('equipment_upgrade',restored)
        self.assertEqual((restored.index,restored.holds),(1,1));self.assertTrue(restored.interrupted)

    def test_expired_volatile_work_is_observed_without_renewing(self):
        task=self.equipment();self.ctx.capture({'equipment_upgrade':task},stage='main');self.now=300
        fresh=self.equipment();fresh.started_at=10
        self.assertEqual(self.ctx.restore('equipment_upgrade',fresh),'observe_only')
        self.assertEqual(fresh.started_at,10)
        self.assertIn('equipment_upgrade',self.ctx.blocked)

    def test_unchanged_observation_does_not_rewrite_every_frame(self):
        task=self.equipment();tasks={'equipment_upgrade':task}
        self.ctx.capture(tasks,stage='main');old=self.ctx.path.read_bytes()
        self.now+=1;self.mono+=1
        self.ctx.capture(tasks,stage='main')
        self.assertEqual(self.ctx.path.read_bytes(),old)

    def test_generic_resume_requires_three_new_frames_without_deadline_extension(self):
        budget={'stage_deadline_epoch':130.,'stage_started_at_epoch':95.,'consecutive':2,'stage_trigger':'entry'}
        self.ctx.capture({},stage='reward_return',budget=budget)
        ctx=WorkflowContext(self.temp.name,'serial',wall=lambda:self.now,monotonic=lambda:self.mono)
        loop=NS(stage='main');rules={'return':({'id':'return','stage':'reward_return','action':'observe','threshold':.95},object(),(0,0))}
        module=NS(appear=lambda *a,**kw:True)
        for mono,answer in ((10,False),(11,False),(13,True)):
            self.mono=mono;self.now=mono+100
            self.assertEqual(ctx.reconcile_stage(loop,rules,module,object(),str(mono)),answer)
        self.assertEqual(loop.stage,'reward_return')
        self.assertEqual(loop.stage_deadline,30.)
        self.assertEqual(loop.consecutive,2)

    def test_uncertain_same_step_purchase_cannot_resume_a_click(self):
        budget={'stage_deadline_epoch':150.}
        self.ctx.capture({},stage='shop',budget=budget)
        self.ctx.reserve('rule:buy','adb',(),{},budget)
        ctx=WorkflowContext(self.temp.name,'serial',wall=lambda:self.now,monotonic=lambda:self.mono)
        rules={'buy':({'id':'buy','stage':'shop','action':'click','threshold':.95},object(),None)}
        module=NS(appear=lambda *a,**kw:True);loop=NS(stage='main')
        for i in range(5):
            self.mono+=2;self.now+=2
            self.assertFalse(ctx.reconcile_stage(loop,rules,module,object(),str(i)))
        self.assertEqual(loop.stage,'main')

    def test_authority_record_is_never_replaced_by_context_snapshot(self):
        source={'version':1,'pending':True,'deadline':150,'attempts':{'confirm':1},'phase':'verify'}
        task=NS(active=True,phase='verify',journal=source)
        self.ctx.capture({'guild_daily':task},stage='main');before=deepcopy(source)
        self.assertEqual(self.ctx.restore('guild_daily',task),'resume_existing_window')
        self.assertEqual(source,before)

    def test_game_launch_and_title_reservations_survive_restart_without_new_120_seconds(self):
        from collections import deque
        task=NS(started_at=None,phase='idle',login_clicked=False,restarts=deque())
        self.ctx.capture({'game_recovery':task},stage='main')
        self.ctx.reserve('game_recovery','launch',(),{'game_recovery':task},{})
        self.now=110;self.mono=20
        self.ctx.reserve('game_recovery','adb',(),{'game_recovery':task},{})
        restored=NS(started_at=None,phase='idle',login_clicked=False,restarts=deque())
        self.now=150;self.mono=1
        context=WorkflowContext(self.temp.name,'serial',wall=lambda:self.now,monotonic=lambda:self.mono)
        context.restore('game_recovery',restored)
        self.assertEqual(self.mono-restored.started_at,50)
        self.assertTrue(restored.login_clicked);self.assertEqual(len(restored.restarts),1)
        self.assertEqual(context.data['tasks']['game_recovery']['deadline'],220)


if __name__=='__main__':unittest.main()
