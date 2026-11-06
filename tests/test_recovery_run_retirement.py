"""New-run login recovery preserves old evidence and cross-run launch limits."""
from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from drift.core.lifecycle import UiOwnership, InputOwnershipError
from drift.core.workflow import WorkflowContext
from drift.tasks.game_recovery import GameRecovery, COMPONENT
from tests.test_game_recovery import Device, ROOT


class RecoveryRunRetirementTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.now=1000.;self.mono=100.
        self.events=[]
        self.ctx=WorkflowContext(self.temp.name,'serial',self.events.append,
            wall=lambda:self.now,monotonic=lambda:self.mono)
        self.old={'pending':True,'phase':'loading','cycle_id':'old-login','deadline':950.,
                  'authority_journal':False,'values':{'phase':'loading',
                    'started_at':{'epoch':830.,'clock':'monotonic'},'login_clicked':False,
                    'restarts_epoch':[840.]},
                  'spent':{'launch:840':{'at':840.,'state':'reserved'}}}
        self.ctx.data['tasks']={'game_recovery':deepcopy(self.old),
            'guild_daily':{'pending':True,'authority_journal':True,'deadline':910.,'spent':{'submit':3}}}
        self.ctx.data['pending']=True
        self.ctx.save()
        self.raw=self.ctx.path.read_bytes()

    def test_start_archives_exact_old_record_and_keeps_other_tasks(self):
        other=deepcopy(self.ctx.data['tasks']['guild_daily'])
        self.assertTrue(self.ctx.retire_expired_recovery_on_start())
        archive=Path(self.temp.name)/'logs/workflow-context-retired'/(hashlib.sha256(self.raw).hexdigest()+'.json')
        self.assertEqual(archive.read_bytes(),self.raw)
        self.assertEqual(self.ctx.data['tasks']['guild_daily'],other)
        self.assertEqual(self.ctx.data['tasks']['game_recovery']['deadline'],950.)
        self.assertEqual(self.ctx.data['tasks']['game_recovery']['spent'],self.old['spent'])
        self.assertFalse(self.ctx.data['tasks']['game_recovery']['pending'])
        self.assertEqual(self.ctx.data['recovery_retirements'][-1]['spent'],self.old['spent'])

    def test_unexpired_or_invalid_deadline_is_not_retired(self):
        for deadline in (1001.,None,'950'):
            with self.subTest(deadline=deadline):
                self.ctx._recovery_start_processed=False
                self.ctx.data['tasks']['game_recovery']['deadline']=deadline
                before=deepcopy(self.ctx.data)
                self.assertFalse(self.ctx.retire_expired_recovery_on_start())
                self.assertEqual(self.ctx.data,before)

    def test_hook_only_runs_once_per_worker(self):
        self.assertTrue(self.ctx.retire_expired_recovery_on_start())
        self.ctx.data['tasks']['game_recovery']=deepcopy(self.old)
        before=deepcopy(self.ctx.data)
        self.assertFalse(self.ctx.retire_expired_recovery_on_start())
        self.assertEqual(self.ctx.data,before)

    def test_ordinary_restore_still_blocks_expired_input(self):
        task=NS(started_at=None,restarts=deque(),phase='idle')
        self.assertEqual(self.ctx.restore('game_recovery',task),'observe_only')
        ui=UiOwnership(lambda:('main',{'game_recovery':task},False))
        ui.checkpoint=self.ctx
        with ui.scope('game_recovery'),self.assertRaises(InputOwnershipError):ui.authorize('launch',())
        self.assertEqual(self.ctx.path.read_bytes(),self.raw)

    def test_reserved_launches_are_carried_and_deduplicated(self):
        saved=self.ctx.data['tasks']['game_recovery']
        saved['spent']['launch:900']={'at':900.,'state':'reserved'}
        self.ctx.save()
        self.ctx.retire_expired_recovery_on_start()
        task=NS(started_at=None,restarts=deque(),phase='idle')
        self.ctx.restore('game_recovery',task)
        self.assertEqual(list(task.restarts),[-60.,0.])

    def test_archive_failure_does_not_retire(self):
        before=deepcopy(self.ctx.data)
        with patch.object(self.ctx,'_archive',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):self.ctx.retire_expired_recovery_on_start()
        self.assertEqual(self.ctx.data,before)
        self.assertEqual(self.ctx.path.read_bytes(),self.raw)

    def test_fresh_title_can_login_once_after_retirement(self):
        self.ctx.retire_expired_recovery_on_start()
        device=Device();device.screen='login';device.state.update(running=True,foreground=COMPONENT)
        task=GameRecovery(ROOT,device,self.events.append)
        self.assertEqual(self.ctx.restore('game_recovery',task),'new')
        ui=UiOwnership(lambda:('main',{'game_recovery':task},False));ui.checkpoint=self.ctx
        click=device.click
        def guarded(button):
            token=ui.authorize('adb',('shell','input','tap','360','914'))
            click(button);ui.issued(token)
        device.click=guarded
        with patch('drift.tasks.game_recovery.time.monotonic',side_effect=lambda:self.mono):
            for i in range(4):
                self.now+=1;self.mono+=1
                self.ctx.capture({'game_recovery':task},stage='main')
                with ui.scope('game_recovery'):task.step(device.screenshot(),False,False)
            self.assertEqual(device.actions,[('login',(360,914))])
            current=self.ctx.data['tasks']['game_recovery']
            self.assertNotEqual(current['cycle_id'],'old-login')
            self.assertEqual(current['deadline'],1123.)
            self.assertTrue(task.login_clicked)
            for i in range(3):
                with ui.scope('game_recovery'):result=task.step(device.screenshot(),True,True)
            self.assertEqual(result,'ready')
        self.assertEqual(self.ctx.data['recovery_retirements'][-1]['deadline'],950.)

    def test_new_run_does_not_bypass_two_launch_limit(self):
        self.ctx.data['tasks']['game_recovery']['spent']['launch:900']={'at':900.,'state':'reserved'}
        self.ctx.save();self.ctx.retire_expired_recovery_on_start()
        device=Device();task=GameRecovery(ROOT,device,self.events.append)
        self.ctx.restore('game_recovery',task)
        with patch('drift.tasks.game_recovery.time.monotonic',return_value=self.mono):
            task.step(device.screenshot(),False,False);task.step(device.screenshot(),False,False)
            with self.assertRaisesRegex(RuntimeError,'30 分钟内已恢复两次'):
                task.step(device.screenshot(),False,False)
        self.assertEqual(device.actions,[])


if __name__=='__main__':unittest.main()
