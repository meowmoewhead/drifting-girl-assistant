import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

from drift.core.lifecycle import UiOwnership,InputOwnershipError
from drift.device.adb import AdbDevice


class InputAuthorityTests(unittest.TestCase):
    def device(self,ui):
        d=AdbDevice.__new__(AdbDevice);d.config={'adb_path':'adb','serial':'x'}
        d.stop=threading.Event();d.input_guard=ui.authorize;d.input_issued=ui.issued
        d.screenshot_transport='adb';d.native_capture=None
        return d

    def backend(self):return patch('drift.device.adb.subprocess.run',return_value=NS(stdout=b'ok',stderr=b'',returncode=0))

    def test_unowned_command_cannot_reach_adb(self):
        ui=UiOwnership();device=self.device(ui)
        with self.backend() as send:
            with self.assertRaises(InputOwnershipError):device.command('shell','input','tap','1','2')
            send.assert_not_called()

    def test_task_cannot_click_another_tasks_panel(self):
        task=NS(active=True,phase='feed',journal={'attempts':{}})
        ui=UiOwnership(lambda:('main',{'restaurant_upgrade':task},False));device=self.device(ui)
        with self.backend() as send,ui.scope('cruise_setup'):
            with self.assertRaises(InputOwnershipError):device.command('shell','input','swipe','1','2','1','2','80')
            send.assert_not_called()

    def test_original_owner_and_its_nested_input_are_allowed(self):
        task=NS(active=True,phase='feed',journal={'attempts':{}})
        ui=UiOwnership(lambda:('main',{'restaurant_upgrade':task},False));device=self.device(ui)
        with self.backend() as send,ui.scope('restaurant_upgrade'):
            self.assertEqual(device.command('shell','input','tap','1','2'),b'ok')
            self.assertEqual(send.call_count,1)

    def test_generic_rule_must_belong_to_current_stage(self):
        ui=UiOwnership(lambda:('mailbox_return',{},False));device=self.device(ui)
        ui.select_rule({'id':'wrong','stage':'main','action':'click'})
        with self.backend() as send:
            with self.assertRaises(InputOwnershipError):device.command('shell','input','tap','1','2')
            ui.select_rule({'id':'close','stage':'mailbox_return','action':'click'})
            device.command('shell','input','tap','1','2');self.assertEqual(send.call_count,1)

    def test_popup_and_game_recovery_preempt_without_granting_other_inputs(self):
        task=NS(active=True,phase='feed',journal={})
        ui=UiOwnership(lambda:('popup_return',{'restaurant_upgrade':task},True));device=self.device(ui)
        with self.backend() as send:
            ui.select_rule({'id':'popup','stage':'*','interrupt':True})
            device.command('shell','input','tap','1','2')
            with ui.scope('restaurant_upgrade'):
                with self.assertRaises(InputOwnershipError):device.command('shell','input','tap','1','2')
            with ui.scope('game_recovery'):device.command('shell','am','start','-n','game/activity')
            self.assertEqual(send.call_count,2)

    def test_runtime_rule_callback_scope_matches_only_its_selected_rule(self):
        ui=UiOwnership(lambda:('mailbox_return',{},False));device=self.device(ui)
        ui.select_rule({'id':'close','stage':'mailbox_return','action':'click'})
        with self.backend() as send:
            with ui.scope('rule_close'):
                device.command('shell','input','tap','1','2')
            with ui.scope('rule_other'):
                with self.assertRaises(InputOwnershipError):device.command('shell','input','tap','1','2')
            with ui.scope('rule_scan'):
                with self.assertRaises(InputOwnershipError):device.command('shell','input','tap','1','2')
            self.assertEqual(send.call_count,1)

    def test_rule_omitting_stage_keeps_existing_main_only_default(self):
        ui=UiOwnership();device=self.device(ui)
        ui.select_rule({'id':'driftinggirl_relief_parachute','action':'click'})
        with self.backend() as send,ui.scope('rule_driftinggirl_relief_parachute'):
            device.command('shell','input','tap','1','2')
            ui.snapshot=lambda:('mailbox_return',{},False)
            with self.assertRaises(InputOwnershipError):device.command('shell','input','tap','1','2')
            self.assertEqual(send.call_count,1)

    def test_native_down_is_denied_before_sdk_and_release_code_is_not_replaced(self):
        ui=UiOwnership();device=self.device(ui);device.screenshot_transport='mumu-native'
        device.native_capture=NS(touch=Mock())
        with self.assertRaises(InputOwnershipError):device.touch_fast(1,2,0,lambda:False)
        device.native_capture.touch.assert_not_called()
        device.native_capture.touch.return_value={'hold_ms':40}
        with ui.scope('battle'):
            self.assertEqual(device.touch_fast(1,2,0,lambda:False),{'hold_ms':40})
        self.assertEqual(device.native_capture.touch.call_count,1)

    def test_actual_rule_dispatch_keeps_authority_in_engine_callback_thread(self):
        from drift.tasks.frame_runtime import FrameRuntime
        ui=UiOwnership(lambda:('mailbox_return',{},False));device=self.device(ui)
        ui.select_rule({'id':'close','stage':'mailbox_return','action':'click'})
        runtime=FrameRuntime(device,device.stop);runtime.ui=ui
        with ThreadPoolExecutor(max_workers=1) as pool, self.backend() as send:
            runtime.engine=NS(in_action=lambda:False,call=lambda name,fn:pool.submit(fn).result())
            self.assertEqual(runtime.call('rule_close',device.command,'shell','input','tap','1','2'),b'ok')
            with self.assertRaises(InputOwnershipError):
                runtime.call('rule_other',device.command,'shell','input','tap','1','2')
            self.assertEqual(send.call_count,1)

    def test_disk_reservation_cannot_extend_original_rule_deadline(self):
        import tempfile
        from drift.core.workflow import WorkflowContext
        with tempfile.TemporaryDirectory() as root:
            ui=UiOwnership();device=self.device(ui)
            ui.checkpoint=WorkflowContext(root,'x',wall=lambda:11,monotonic=lambda:1)
            ui.budget=lambda:{'stage_deadline_epoch':10}
            ui.select_rule({'id':'claim','stage':'main','action':'click'})
            with self.backend() as send,ui.scope('rule_claim'):
                with self.assertRaisesRegex(InputOwnershipError,'到期'):
                    device.command('shell','input','tap','1','2')
                send.assert_not_called()
            self.assertEqual(ui.checkpoint.data['last_input']['state'],'reserved')

    def test_read_only_probe_and_stop_request_cannot_issue_input(self):
        ui=UiOwnership();device=self.device(ui)
        with self.backend() as send:
            device.command('shell','getprop','sys.boot_completed')
            with ui.scope('observe'):
                with self.assertRaises(InputOwnershipError):device.command('shell','input','tap','1','2')
            device.stop.set()
            with ui.scope('battle'):
                with self.assertRaises(InterruptedError):device.command('shell','input','tap','1','2')
            self.assertEqual(send.call_count,1)


if __name__=='__main__':unittest.main()
