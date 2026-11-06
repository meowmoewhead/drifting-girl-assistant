"""2026-10-26 公会补跑不能把滞留地点覆盖成农场起点。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drift.tasks.guild_daily import GuildDaily


SETTINGS = dict(pearl=True,material=False,amber=True,gem=True,boss=False,
                sale=False,upgrade=False,priority='saved-priority')
FARM = dict(parallel=True,stage=8,rank=22,route=True,fishing=True,settings=SETTINGS)
GUILD = dict(parallel=False,stage=None,rank=None,route=False,fishing=False,at_guild=True)


class GuildReturnHomeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'guild_daily_state.json'
        self.now=100.
        self.inputs=[]
        self.c=self.controller()

    def controller(self,*,collection_only=False):
        with patch('drift.tasks.guild_daily.StoneFarm'), \
                patch('drift.tasks.guild_daily.BackpackSale'), \
                patch('drift.tasks.guild_daily.GuildVisual'):
            c=GuildDaily(ROOT,Mock(),lambda _:False,Mock(),'test-device',
                        state_path=self.path,clock=lambda:self.now,collection_only=collection_only)
        c.signature=Mock(return_value=('world','rank','location'))
        c.settings=Mock(return_value=None)
        c.origin_screen=Mock(return_value=deepcopy(GUILD))
        c.visual.panel.return_value=False
        c.visual.task_page.return_value=False
        c.visual.point.return_value=None
        c.view._parallel.return_value=False
        c.view._stage.return_value=8
        c.view._rank_value.return_value=22
        c.view._sailing_menu_rank.side_effect=lambda frame:c.view._rank_value(frame)
        c.view._off_icon.return_value=True
        c.view._idle_fishing.return_value=True
        c.view._map.return_value=False
        c.view._battle.return_value=False
        c.view._fixed.return_value=None
        c.view._original.return_value=True
        c.view._npc_ready.return_value=True
        c.view._gate_active.return_value=True
        c.view.training.boss_mode.return_value=False
        c.fishing._main_visible.return_value=True
        c.fishing.at_port.return_value=False
        c.fishing.blue_fill.return_value=False
        c.fishing.real_battle.return_value=False
        c.sailing.running.return_value=False
        c.sailing.menu.return_value=False
        c.bag.moving.return_value=False
        c.stopped_map=Mock(return_value=False)
        c.map_target=Mock(return_value=(100,200))

        def action(request):
            if c.validate(request,self.frame) and c.reserve(request):
                self.inputs.append(request['kind'])
                return True
            return False
        c.action=action
        return c

    def tick(self,*,due=False,frame=None,seconds=2):
        self.now+=seconds
        self.frame=np.zeros((4,4,3),np.uint8) if frame is None else frame
        return self.c.step(self.frame,due=due,period='2026-10-26')

    def confirm(self,*,due=False,frame=None):
        for _ in range(3):self.tick(due=due,frame=frame)

    def saved(self):
        return json.loads(self.path.read_text(encoding='utf-8'))

    def activate(self,phase='close',*,target=None):
        c=self.c
        target=deepcopy(FARM if target is None else target)
        c.journal=dict(version=1,serial='test-device',pending=True,phase=phase,origin=deepcopy(GUILD),target=target,
                         farm_origin=deepcopy(target),farm_restored_at=None,
                         deadline=self.now+1000,phase_deadline=self.now+180,
                         attempts={},categories=[0,1],submitted=9)
        c.phase=phase
        c.key=c.since=c.last_seen=c.last_frame=None;c.seen=0;c.next_at=0
        c._save()

    def home(self,*,running=True,fishing=True):
        c=self.c
        c.view._parallel.return_value=True
        c.view._off_icon.return_value=not running
        c.view._idle_fishing.return_value=not fishing
        c.sailing.running.return_value=running
        c.sailing.menu.return_value=False
        c.fishing.real_battle.return_value=fishing
        c.settings.return_value=None

    def menu(self,settings=None):
        self.home(running=False,fishing=False)
        self.c.sailing.menu.return_value=True
        self.c.settings.return_value=deepcopy(SETTINGS if settings is None else settings)

    def test_parallel_start_and_snapshot_persist_before_departure(self):
        c=self.c
        c.journal.update(farm_origin=dict(FARM,rank=18),farm_restored_at=50.)
        origin={k:v for k,v in FARM.items() if k!='settings'}
        c.origin_screen.return_value=deepcopy(origin)
        self.home()
        self.confirm(due=True)
        self.assertEqual(self.inputs,['stop'])
        self.assertEqual(self.saved()['farm_origin'],dict(origin,settings=None))
        self.assertIsNone(self.saved()['farm_restored_at'])
        self.home(running=False,fishing=False)
        self.confirm()
        self.assertEqual(c.phase,'snapshot_open')
        self.tick()
        self.assertEqual(self.inputs[-1],'snapshot-open')
        self.menu()
        self.confirm()
        self.assertEqual(c.phase,'stop_fishing')
        self.assertEqual(self.saved()['farm_origin'],FARM)
        self.assertEqual(self.saved()['target'],FARM)
        c.journal['target']['settings']['priority']='changed-in-memory'
        self.assertEqual(c.journal['farm_origin'],FARM)
        self.c=self.controller()
        self.assertEqual(self.c.journal['farm_origin'],FARM)

    def test_finish_abandon_block_and_retirement_keep_unrestored_farm(self):
        for operation in ('finish-success','finish-incomplete','abandon','blocked','retired'):
            with self.subTest(operation=operation):
                self.activate()
                c=self.c
                if operation.startswith('finish'):c._finish(operation=='finish-success')
                elif operation=='abandon':c.abandon()
                else:
                    self.now=c.journal['phase_deadline']+1
                    with self.assertRaises(RuntimeError):self.tick()
                    if operation=='retired':self.assertTrue(c.retire_expired_on_start())
                self.assertEqual(self.saved()['farm_origin'],FARM)
                self.assertIsNone(self.saved()['farm_restored_at'])
                self.c=self.controller()
                self.assertEqual(self.c.journal['farm_origin'],FARM)

    def test_retry_from_guild_uses_saved_target_after_restart(self):
        self.activate('guild_travel')
        self.c.abandon()
        self.c=self.controller()
        self.confirm(due=True)
        self.assertEqual(self.c.phase,'guild_at')
        self.assertEqual(self.saved()['origin'],GUILD)
        self.assertEqual(self.saved()['target'],FARM)
        self.assertEqual(self.saved()['farm_origin'],FARM)
        self.c.journal['categories']=[0,1]
        self.c._set('close')
        self.confirm()
        self.assertEqual(self.c.phase,'map_open')
        self.assertEqual(self.c.journal['map_goal'],'observatory')
        self.assertEqual(self.c.journal['map_after'],'gate')
        self.assertIsNone(self.c.done)

    def test_full_return_chain_restarts_saved_route_and_confirms_restoration(self):
        self.activate()
        c=self.c
        deadline=c.journal['deadline']
        self.confirm()
        self.assertEqual(c.phase,'map_open')
        c.stopped_map.return_value=True
        c.view._map.return_value=True
        self.tick()
        self.assertEqual(c.phase,'map_select')
        self.tick()
        self.assertEqual(c.phase,'map_close')
        c.bag._find.assert_not_called()
        self.confirm()
        c.view._map.return_value=False
        c.stopped_map.return_value=False
        self.confirm()
        self.assertEqual(c.phase,'travel')
        self.confirm()
        self.assertEqual(c.phase,'gate')
        self.confirm()
        self.assertEqual(c.phase,'enter')
        self.tick()
        c.view._fixed.return_value=(208,787)
        self.tick()
        self.assertEqual(c.phase,'arrive')
        self.home(running=False,fishing=False)
        self.confirm()
        self.assertEqual(c.phase,'resume')
        self.tick()
        self.assertEqual(c.phase,'resume_menu')
        self.menu()
        self.confirm()
        self.assertEqual(c.phase,'resume_verify')
        self.assertIsNone(self.saved()['farm_restored_at'])
        self.assertIsNone(c.done)
        self.c=self.controller()
        self.home()
        self.confirm()
        self.assertTrue(self.c.done)
        self.assertFalse(self.c.active)
        self.assertEqual(self.saved()['farm_origin'],FARM)
        self.assertEqual(self.saved()['farm_restored_at'],self.now)
        self.assertEqual(self.c.journal['deadline'],deadline)
        self.assertEqual(self.inputs,['map-select','map-close','enter','enter-confirm',
                                      'route-open','route-start'])

    def test_observatory_departure_does_not_replace_saved_settings(self):
        c=self.c
        c.journal['farm_origin']=deepcopy(FARM)
        c.origin_screen.return_value=dict(GUILD,at_guild=False)
        self.confirm(due=True)
        self.confirm()
        self.assertEqual(c.phase,'snapshot_open')
        self.tick()
        c.sailing.menu.return_value=True
        c.settings.return_value=dict(SETTINGS,priority='observatory-options')
        self.confirm()
        self.assertEqual(c.phase,'stop_fishing')
        self.assertEqual(self.saved()['target'],FARM)
        self.assertEqual(self.saved()['farm_origin'],FARM)
        self.assertNotEqual(self.saved()['settings'],SETTINGS)

    def test_unknown_origin_waits_without_reserving_old_farm(self):
        c=self.c
        c.journal['farm_origin']=deepcopy(FARM)
        c.origin_screen.return_value=None
        self.confirm(due=True)
        self.assertFalse(c.active)
        self.assertNotIn('origin',c.journal)
        self.assertNotIn('target',c.journal)
        self.assertEqual(self.inputs,[])
        c.origin_screen.return_value=deepcopy(GUILD)
        self.confirm(due=True)
        self.assertEqual(c.journal['origin'],GUILD)
        self.assertEqual(c.phase,'guild_at')

    def test_unowned_open_panel_does_not_create_a_new_round_trip(self):
        c=self.c
        c.journal['farm_origin']=deepcopy(FARM)
        c.origin_screen.return_value=None
        c.visual.panel.return_value=True
        self.confirm(due=True)
        self.assertEqual(c.phase,'idle')
        c.visual.task_page.return_value=True
        self.tick()
        self.assertFalse(c.active)
        self.assertEqual(c.journal['farm_origin'],FARM)
        self.assertEqual(self.inputs,[])

    def test_missing_or_invalid_farm_never_invents_a_destination(self):
        for farm in (None,dict(FARM,rank=40),dict(FARM,route=None),dict(FARM,stage=None)):
            with self.subTest(farm=farm):
                self.c.journal['farm_origin']=deepcopy(farm)
                self.c.origin_screen.return_value=None
                self.confirm(due=True)
                self.assertFalse(self.c.active)
                self.assertEqual(self.inputs,[])

    def test_existing_high_rank_and_final_stage_guards_are_not_bypassed(self):
        c=self.c
        del c.origin_screen
        c.journal['farm_origin']=deepcopy(FARM)
        self.home()
        for stage,rank in ((8,40),(9,39)):
            with self.subTest(stage=stage,rank=rank):
                c.view.screen.return_value=dict(stage=stage,rank=rank,route=True,menu=False,completion=False)
                self.assertIsNone(c.origin_screen(np.zeros((4,4,3),np.uint8)))
                self.confirm(due=True)
                self.assertEqual(c.journal['farm_origin'],FARM)
                self.assertEqual(self.inputs,[])

    def test_same_origin_compares_target_world_stage_and_rank(self):
        self.activate()
        self.home()
        frame=np.zeros((4,4,3),np.uint8)
        self.assertTrue(self.c.same_origin(frame))
        for probe,value in ((self.c.view._parallel,False),(self.c.view._stage,9),
                            (self.c.view._rank_value,23)):
            previous=probe.return_value
            probe.return_value=value
            self.assertFalse(self.c.same_origin(frame))
            probe.return_value=previous

    def test_close_already_home_finishes_only_after_fresh_matching_frames(self):
        self.activate()
        self.home()
        self.c.journal['resume_settings']=deepcopy(SETTINGS)
        self.c._map_begin=Mock()
        frame=np.zeros((4,4,3),np.uint8)
        self.confirm(frame=frame)
        self.assertIsNone(self.c.done)
        self.tick();self.tick()
        self.assertTrue(self.c.done)
        self.assertIsNotNone(self.saved()['farm_restored_at'])
        self.c._map_begin.assert_not_called()
        self.assertEqual(self.inputs,[])

    def test_close_already_home_with_unverified_options_rechecks_route(self):
        self.activate()
        self.home()
        self.confirm()
        self.assertEqual(self.c.phase,'resume')
        self.assertIsNone(self.c.done)
        self.tick()
        self.assertEqual(self.inputs,['stop'])
        self.assertIsNone(self.saved()['farm_restored_at'])

    def test_settings_mismatch_resets_confirmation_and_never_starts_route(self):
        self.activate('resume_menu')
        self.menu(dict(SETTINGS,priority='wrong-order'))
        self.confirm()
        self.assertEqual(self.inputs,[])
        self.assertIn('不一致',self.c.hold)
        self.menu()
        self.tick()
        self.menu(dict(SETTINGS,boss=True))
        self.tick()
        self.menu()
        self.tick();self.tick()
        self.assertEqual(self.inputs,[])
        self.tick()
        self.assertEqual(self.inputs,['route-start'])
        self.assertEqual(self.c.journal['farm_origin'],FARM)
        self.assertIsNone(self.c.done)

    def test_fresh_route_input_guard_rejects_changed_settings_and_world(self):
        for change in ('settings','world'):
            with self.subTest(change=change):
                self.activate('resume_menu')
                self.c.key=self.c.last_frame=None
                self.menu()
                original_action=self.c.action
                def changed(request):
                    if change=='settings':self.c.settings.return_value=dict(SETTINGS,sale=True)
                    else:self.c.view._stage.return_value=9
                    return original_action(request)
                self.c.action=changed
                self.confirm()
                self.assertEqual(self.inputs,[])
                self.assertNotIn('route-start',self.c.journal['attempts'])
                self.assertNotIn('resume_settings',self.c.journal)
                self.c.action=original_action
                self.c.view._stage.return_value=8

    def test_incomplete_farm_snapshot_cannot_borrow_normal_world_settings(self):
        self.activate('resume_menu',target=dict(FARM,settings=None))
        self.c.journal['settings']=deepcopy(SETTINGS)
        self.menu()
        self.confirm()
        self.assertEqual(self.inputs,[])
        self.assertIsNone(self.c.target_settings())
        self.assertIsNone(self.saved()['farm_restored_at'])

    def test_visible_settings_override_earlier_matching_proof(self):
        self.activate('resume_menu')
        self.c.journal['resume_settings']=deepcopy(SETTINGS)
        self.menu(dict(SETTINGS,upgrade=True))
        self.assertFalse(self.c.restored_settings(np.zeros((4,4,3),np.uint8)))
        self.confirm()
        self.assertEqual(self.inputs,[])
        self.assertIsNone(self.saved()['farm_restored_at'])

    def test_resume_uses_farm_options_instead_of_departure_snapshot(self):
        self.activate('resume_menu')
        self.c.journal['settings']=dict(SETTINGS,priority='normal-world-options')
        self.menu()
        self.confirm()
        self.assertEqual(self.inputs,['route-start'])
        self.assertEqual(self.saved()['resume_settings'],SETTINGS)
        deadline=self.c.journal['phase_deadline']
        self.now=deadline+1
        with self.assertRaises(RuntimeError):self.tick()
        self.assertEqual(self.saved()['phase_deadline'],deadline)
        self.assertEqual(self.saved()['attempts']['route-start'],1)
        self.assertEqual(self.saved()['farm_origin'],FARM)
        self.assertIsNone(self.saved()['farm_restored_at'])

    def test_stopped_farm_restores_idle_or_fishing_without_starting_route(self):
        for fishing in (False,True):
            with self.subTest(fishing=fishing):
                self.inputs.clear()
                self.activate('resume',target=dict(FARM,route=False,fishing=fishing))
                self.home(running=False,fishing=False)
                self.tick()
                self.assertEqual(self.c.phase,'resume_menu')
                self.menu()
                self.confirm()
                self.assertEqual(self.c.phase,'resume_verify')
                self.home(running=False,fishing=fishing)
                self.confirm()
                self.assertTrue(self.c.done)
                self.assertEqual(self.inputs,['route-open','snapshot-close'])
                self.assertFalse(self.saved()['farm_origin']['route'])

    def test_wrong_world_rank_or_route_does_not_credit_restoration(self):
        self.activate('resume_verify')
        self.c.journal['resume_settings']=deepcopy(SETTINGS)
        self.home()
        self.tick();self.tick()
        self.c.view._rank_value.return_value=23
        self.tick()
        self.c.view._rank_value.return_value=22
        self.tick();self.tick()
        self.assertIsNone(self.c.done)
        self.c.sailing.running.return_value=None
        self.tick()
        self.c.sailing.running.return_value=True
        self.tick();self.tick()
        self.assertIsNone(self.c.done)
        self.tick()
        self.assertTrue(self.c.done)

    def test_cancellation_uses_farm_target_even_when_departure_was_normal(self):
        self.activate('guild_cancel')
        self.confirm()
        self.assertEqual(self.c.phase,'gate')
        self.assertIsNone(self.c.done)

    def test_incomplete_collection_can_still_confirm_home_without_claiming_daily(self):
        self.activate('resume_verify')
        self.c.journal.update(categories=[0],resume_settings=deepcopy(SETTINGS))
        self.home()
        self.confirm()
        self.assertFalse(self.c.done)
        self.assertIsNotNone(self.saved()['farm_restored_at'])
        self.assertEqual(self.saved()['farm_origin'],FARM)

    def test_collection_only_preserves_farm_and_its_restoration_marker(self):
        self.c=self.controller(collection_only=True)
        self.c.journal.update(farm_origin=deepcopy(FARM),farm_restored_at=50.)
        self.c.visual.task_page.return_value=True
        self.confirm(due=True)
        self.c.journal['categories']=[0,1]
        self.c._set('close')
        self.confirm()
        self.assertTrue(self.c.done)
        self.assertEqual(self.saved()['farm_origin'],FARM)
        self.assertEqual(self.saved()['farm_restored_at'],50.)

    # F1: a retry at the stopped farm must retain the activity still owed.
    def test_failed_retry_keeps_original_farm_activity(self):
        for operation in ('abandon','retire'):
            for restart in (False,True):
                with self.subTest(operation=operation,restart=restart):
                    self.inputs.clear()
                    self.activate('stop_fishing')
                    if operation=='abandon':self.c.abandon()
                    else:
                        self.now=self.c.journal['phase_deadline']+1
                        with self.assertRaises(RuntimeError):self.tick()
                        self.assertTrue(self.c.retire_expired_on_start())
                    if restart:self.c=self.controller()
                    self.home(running=False,fishing=False)
                    self.c.origin_screen.return_value=dict(FARM,route=False,fishing=False,settings=None)
                    self.confirm(due=True)
                    self.assertTrue(self.c.active)
                    self.assertEqual(self.saved()['farm_origin'],FARM)
                    self.assertEqual(self.saved()['target'],FARM)
                    self.assertTrue(self.saved()['resume_pending'])
                    self.assertFalse(self.saved()['capture_target_settings'])
                    self.confirm()
                    self.assertEqual(self.c.phase,'snapshot_open')
                    self.tick()
                    self.menu(dict(SETTINGS,priority='temporary-options'))
                    self.confirm()
                    self.assertEqual(self.c.phase,'stop_fishing')
                    self.assertEqual(self.saved()['farm_origin'],FARM)
                    self.assertEqual(self.saved()['target'],FARM)
                    self.c=self.controller()
                    self.assertEqual(self.c.journal['farm_origin'],FARM)

    # F2: issued inputs need bounded recovery from the next observed screen.
    def test_arrive_retries_unapplied_enter_confirmation_after_restart(self):
        self.activate('enter')
        self.c.view._fixed.return_value=(208,787)
        self.tick()
        self.assertEqual(self.c.phase,'arrive')
        self.assertEqual(self.inputs,['enter-confirm'])
        deadlines={key:self.saved()[key] for key in ('deadline','phase_deadline')}
        self.c=self.controller()
        self.c.view._fixed.return_value=(208,787)
        self.confirm(frame=np.zeros((4,4,3),np.uint8))
        self.assertEqual(self.inputs,['enter-confirm'])
        self.tick();self.tick()
        self.assertEqual(self.inputs,['enter-confirm','enter-confirm'])
        self.assertEqual(self.saved()['attempts']['enter-confirm'],2)
        self.assertEqual({key:self.saved()[key] for key in deadlines},deadlines)
        self.c.view._fixed.return_value=None
        self.home(running=False,fishing=False)
        self.confirm()
        self.assertEqual(self.c.phase,'resume')

    def test_arrive_confirmation_retries_stop_at_saved_limit(self):
        self.activate('arrive')
        self.c.journal['attempts']['enter-confirm']=1
        self.c._save()
        deadlines={key:self.saved()[key] for key in ('deadline','phase_deadline')}
        for expected in (2,3):
            self.c=self.controller()
            self.c.view._fixed.return_value=(208,787)
            self.confirm()
            self.assertEqual(self.saved()['attempts']['enter-confirm'],expected)
        self.confirm();self.confirm()
        self.assertEqual(self.inputs,['enter-confirm','enter-confirm'])
        self.assertEqual(self.saved()['attempts']['enter-confirm'],3)
        self.assertEqual({key:self.saved()[key] for key in deadlines},deadlines)
        self.now=deadlines['phase_deadline']
        with self.assertRaises(RuntimeError):self.tick()
        self.assertEqual(self.saved()['farm_origin'],FARM)
        self.assertEqual(self.saved()['attempts']['enter-confirm'],3)

    def test_failed_arrival_unknown_dialog_does_not_renew_spent_inputs(self):
        self.activate('arrive')
        self.c.journal['attempts']['enter-confirm']=3
        self.now=self.c.journal['phase_deadline']
        with self.assertRaises(RuntimeError):self.tick()
        self.c.abandon()
        self.c=self.controller()
        self.c.origin_screen.return_value=None
        self.c.view._fixed.return_value=(208,787)
        self.confirm(due=True)
        deadlines={key:self.saved()[key] for key in ('deadline','phase_deadline')}
        self.confirm()
        self.assertEqual(self.c.phase,'idle')
        self.assertEqual(self.inputs,[])
        self.assertEqual({key:self.saved()[key] for key in deadlines},deadlines)
        self.assertEqual(self.saved()['attempts']['enter-confirm'],3)
        self.assertEqual(self.saved()['farm_origin'],FARM)
        self.assertEqual({key:self.saved()[key] for key in deadlines},deadlines)

    def test_resume_menu_retries_unopened_menu_after_restart(self):
        self.activate('resume')
        self.home(running=False,fishing=False)
        self.tick()
        self.assertEqual(self.c.phase,'resume_menu')
        self.assertEqual(self.inputs,['route-open'])
        deadlines={key:self.saved()[key] for key in ('deadline','phase_deadline')}
        self.c=self.controller()
        self.home(running=False,fishing=False)
        self.confirm(frame=np.zeros((4,4,3),np.uint8))
        self.assertEqual(self.inputs,['route-open'])
        self.tick();self.tick()
        self.assertEqual(self.inputs,['route-open','route-open'])
        self.assertEqual(self.saved()['attempts']['route-open'],2)
        self.assertEqual({key:self.saved()[key] for key in deadlines},deadlines)
        self.menu()
        self.confirm()
        self.assertEqual(self.inputs[-1],'route-start')
        self.home()
        self.confirm()
        self.assertTrue(self.c.done)
        self.assertFalse(self.saved()['resume_pending'])
        self.assertEqual(self.saved()['deadline'],deadlines['deadline'])

    def test_resume_menu_exhaustion_falls_back_without_extending_deadlines(self):
        self.activate('resume_menu')
        self.c.journal['attempts']['route-open']=2
        self.c._save()
        deadlines={key:self.saved()[key] for key in ('deadline','phase_deadline')}
        self.home(running=False,fishing=False)
        self.confirm()
        self.assertEqual(self.saved()['attempts']['route-open'],3)
        self.c=self.controller()
        self.home(running=False,fishing=False)
        self.confirm()
        self.assertEqual(self.c.phase,'resume')
        self.confirm()
        self.assertEqual(self.inputs,['route-open'])
        self.assertEqual(self.saved()['attempts']['route-open'],3)
        self.assertEqual({key:self.saved()[key] for key in deadlines},deadlines)
        self.c.journal['resume_settings']=deepcopy(SETTINGS)
        self.home()
        self.confirm()
        self.assertTrue(self.c.done)

    def test_return_retries_reset_confirmation_and_recheck_fresh_frame(self):
        for phase,kind in (('arrive','enter-confirm'),('resume_menu','route-open')):
            with self.subTest(phase=phase):
                self.c=self.controller()
                self.inputs.clear()
                self.activate(phase)
                self.c.journal['attempts'][kind]=1
                if phase=='resume_menu':self.home(running=False,fishing=False)
                probe=self.c.view._fixed if phase=='arrive' else self.c.fishing._main_visible
                visible=(208,787) if phase=='arrive' else True
                hidden=None if phase=='arrive' else False
                probe.return_value=visible
                self.tick();self.tick()
                probe.return_value=hidden
                self.tick()
                probe.return_value=visible
                self.tick();self.tick()
                self.assertEqual(self.inputs,[])
                action=self.c.action
                def changed(request):
                    probe.return_value=hidden
                    return action(request)
                self.c.action=changed
                self.tick()
                self.assertEqual(self.inputs,[])
                self.assertEqual(self.c.journal['attempts'][kind],1)
                self.c.action=action
                probe.return_value=visible
                self.confirm()
                self.assertEqual(self.inputs,[kind])

    def test_return_retries_obey_phase_and_overall_deadlines(self):
        for phase,kind in (('arrive','enter-confirm'),('resume_menu','route-open')):
            for limit in ('deadline','phase_deadline'):
                with self.subTest(phase=phase,limit=limit):
                    self.c=self.controller()
                    self.inputs.clear()
                    self.activate(phase)
                    self.c.journal['attempts'][kind]=1
                    self.c.journal[limit]=self.now+5
                    self.c._save()
                    deadlines={key:self.saved()[key] for key in ('deadline','phase_deadline')}
                    if phase=='arrive':self.c.view._fixed.return_value=(208,787)
                    else:self.home(running=False,fishing=False)
                    self.tick();self.tick()
                    with self.assertRaises(RuntimeError):self.tick()
                    self.assertEqual(self.inputs,[])
                    self.assertEqual(self.saved()['attempts'][kind],1)
                    self.assertEqual({key:self.saved()[key] for key in deadlines},deadlines)

    def test_resume_menu_waits_for_recognizable_matching_main_screen(self):
        for change in ('unknown-route','hidden-main','world','stage','rank'):
            with self.subTest(change=change):
                self.c=self.controller()
                self.activate('resume_menu')
                self.c.journal['attempts']['route-open']=1
                self.home(running=False,fishing=False)
                if change=='unknown-route':self.c.sailing.running.return_value=None
                elif change=='hidden-main':self.c.fishing._main_visible.return_value=False
                elif change=='world':self.c.view._parallel.return_value=False
                elif change=='stage':self.c.view._stage.return_value=9
                else:self.c.view._rank_value.return_value=23
                self.confirm()
                self.assertEqual(self.c.phase,'resume_menu')
                self.assertEqual(self.inputs,[])
                self.assertEqual(self.c.journal['attempts']['route-open'],1)

    # F3: only a valid farm or the separate collection-only entry may start.
    def test_normal_world_without_valid_farm_waits_without_ownership_or_input(self):
        for farm in (None,dict(FARM,rank=40),dict(FARM,route=None),dict(FARM,stage=None)):
            for at_guild in (False,True):
                with self.subTest(farm=farm,at_guild=at_guild):
                    self.c=self.controller()
                    self.c.phase='idle'
                    self.c.journal=dict(pending=False,phase='idle',farm_origin=deepcopy(farm))
                    self.c.origin_screen.return_value=dict(GUILD,at_guild=at_guild)
                    self.c.visual.point.side_effect=lambda frame,name: (679,1013) if name in ('g-icon','guild-location') else None
                    self.inputs.clear()
                    results=[self.tick(due=True) for _ in range(6)]
                    self.assertFalse(any(results))
                    self.assertFalse(self.c.active)
                    self.assertEqual(self.c.phase,'idle')
                    self.assertEqual(self.inputs,[])

    def test_first_parallel_departure_creates_farm_snapshot(self):
        origin={key:value for key,value in FARM.items() if key!='settings'}
        self.c.origin_screen.return_value=deepcopy(origin)
        self.home()
        self.confirm(due=True)
        self.assertTrue(self.c.active)
        self.assertEqual(self.inputs,['stop'])
        self.assertEqual(self.saved()['farm_origin'],dict(origin,settings=None))
        self.assertEqual(self.saved()['target'],self.saved()['farm_origin'])
        self.assertTrue(self.saved()['resume_pending'])

    def test_collection_only_without_farm_still_collects(self):
        self.c=self.controller(collection_only=True)
        self.c.visual.task_page.return_value=True
        self.confirm(due=True)
        self.assertTrue(self.c.active)
        self.assertEqual(self.c.phase,'collect')
        self.assertIsNone(self.c.journal.get('farm_origin'))
        self.c.journal['categories']=[0,1]
        self.c._set('close')
        self.confirm()
        self.assertTrue(self.c.done)

    # F4: an explicit departure refusal cannot borrow the historical target.
    def test_rejected_departure_never_claims_ownership_with_saved_farm(self):
        for stage,rank,boss,completion in ((8,40,False,False),(9,39,False,False),
                                           (8,22,True,False),(8,22,False,True)):
            with self.subTest(stage=stage,rank=rank,boss=boss,completion=completion):
                self.c=self.controller()
                self.c.phase='idle'
                self.c.journal=dict(pending=False,phase='idle',farm_origin=deepcopy(FARM),farm_restored_at=None)
                del self.c.origin_screen
                self.home()
                self.c.view.screen.return_value=dict(stage=stage,rank=rank,route=True,menu=False,completion=completion)
                self.c.view.training.boss_mode.return_value=boss
                self.inputs.clear()
                results=[self.tick(due=True) for _ in range(6)]
                self.assertFalse(any(results))
                self.assertFalse(self.c.active)
                self.assertEqual(self.c.phase,'idle')
                self.assertEqual(self.c.journal['farm_origin'],FARM)
                self.assertTrue(self.c.reject_reason)
                self.assertEqual(self.inputs,[])

    def test_departure_refusal_clears_start_votes_and_recovers(self):
        del self.c.origin_screen
        self.c.journal.update(farm_origin=deepcopy(FARM),farm_restored_at=50.)
        self.home()
        current=dict(stage=8,rank=22,route=True,menu=False,completion=False)
        self.c.view.screen.return_value=current
        self.assertFalse(self.tick(due=True));self.assertFalse(self.tick(due=True))
        self.c.view.screen.return_value=dict(current,rank=40)
        self.assertFalse(self.tick(due=True))
        self.assertFalse(self.c.active)
        self.c.view.screen.return_value=current
        self.assertFalse(self.tick(due=True));self.assertFalse(self.tick(due=True))
        self.assertTrue(self.tick(due=True))
        self.assertIsNone(self.c.reject_reason)
        self.assertEqual(self.inputs,['stop'])

    # F5: an interrupted settings capture remains completable at the saved farm.
    def test_unfinished_settings_snapshot_is_completed_on_retry(self):
        for scenario in ('retry','restart','retire','normal-world','stage','rank',
                         'interrupted-world','interrupted-settings'):
            with self.subTest(scenario=scenario):
                self.c=self.controller()
                c=self.c
                c.phase='idle'
                c.journal=dict(version=1,serial='test-device',pending=False,phase='idle')
                self.inputs.clear()
                origin={key:value for key,value in FARM.items() if key!='settings'}
                c.origin_screen.return_value=deepcopy(origin)
                self.home()
                self.confirm(due=True)
                self.assertEqual(self.inputs,['stop'])
                incomplete=dict(origin,settings=None)
                self.assertEqual(self.saved()['farm_origin'],incomplete)
                self.home(running=False,fishing=False)
                self.confirm()
                self.assertEqual(c.phase,'snapshot_open')
                if scenario=='retire':
                    self.now=c.journal['phase_deadline']
                    with self.assertRaises(RuntimeError):self.tick()
                    self.assertTrue(c.retire_expired_on_start())
                else:c.abandon()
                if scenario in ('restart','retire'):self.c=self.controller()
                c=self.c
                self.home(running=False,fishing=False)
                retry_origin=dict(origin,route=False,fishing=False)
                if scenario=='normal-world':
                    retry_origin=dict(GUILD,at_guild=False)
                    c.view._parallel.return_value=False
                elif scenario=='stage':
                    retry_origin['stage']=9;c.view._stage.return_value=9
                elif scenario=='rank':
                    retry_origin['rank']=23;c.view._rank_value.return_value=23
                c.origin_screen.return_value=deepcopy(retry_origin)
                self.confirm(due=True)
                self.assertEqual(self.saved()['origin'],retry_origin)
                self.assertEqual(self.saved()['farm_origin'],incomplete)
                self.assertEqual(self.saved()['target'],incomplete)
                self.assertTrue(self.saved()['resume_pending'])
                self.confirm()
                self.assertEqual(c.phase,'snapshot_open')
                self.tick()
                self.assertEqual(c.phase,'snapshot')
                c.sailing.menu.return_value=True
                c.settings.return_value=deepcopy(SETTINGS)
                self.tick();self.tick()
                self.assertEqual(c.phase,'snapshot')
                self.assertIsNone(self.saved()['target']['settings'])
                if scenario.startswith('interrupted-'):
                    if scenario=='interrupted-world':c.view._parallel.return_value=False
                    else:c.settings.return_value=None
                    self.tick()
                    c.view._parallel.return_value=True
                    c.settings.return_value=deepcopy(SETTINGS)
                    self.tick();self.tick()
                    self.assertEqual(c.phase,'snapshot')
                    self.assertIsNone(self.saved()['farm_origin']['settings'])
                self.tick()
                self.assertEqual(c.phase,'stop_fishing')
                self.assertEqual(self.saved()['settings'],SETTINGS)
                if scenario in ('normal-world','stage','rank'):
                    self.assertEqual(self.saved()['farm_origin'],incomplete)
                    self.assertEqual(self.saved()['target'],incomplete)
                    self.assertTrue(self.saved()['resume_pending'])
                    self.assertIsNone(self.saved()['farm_restored_at'])
                    continue
                self.assertEqual(self.saved()['farm_origin'],FARM)
                self.assertEqual(self.saved()['target'],FARM)
                self.assertIsNot(c.journal['target']['settings'],c.journal['farm_origin']['settings'])
                self.assertIsNot(c.journal['target']['settings'],c.journal['settings'])
                self.assertTrue(self.saved()['resume_pending'])
                # Resume from the completed, persisted snapshot at the return boundary.
                self.c=self.controller()
                c=self.c
                self.assertEqual(c.journal['farm_origin'],FARM)
                deadline=c.journal['deadline']
                c.journal['categories']=[0,1]
                c._set('close')
                self.home(running=False,fishing=False)
                self.confirm()
                self.assertEqual(c.phase,'resume')
                self.tick()
                self.assertEqual(c.phase,'resume_menu')
                self.menu()
                self.confirm()
                self.assertEqual(c.phase,'resume_verify')
                self.assertTrue(self.saved()['resume_pending'])
                self.home()
                self.tick();self.tick()
                self.assertTrue(c.active)
                self.assertIsNone(self.saved()['farm_restored_at'])
                self.tick()
                self.assertTrue(c.done)
                self.assertFalse(c.active)
                self.assertFalse(self.saved()['resume_pending'])
                self.assertEqual(self.saved()['farm_restored_at'],self.now)
                self.assertEqual(self.saved()['farm_origin'],FARM)
                self.assertEqual(self.saved()['target'],FARM)
                self.assertEqual(self.saved()['deadline'],deadline)
                self.assertEqual(self.inputs,['stop','snapshot-open','snapshot-close',
                                              'route-open','route-start'])

    # F6: falling back to resume cannot renew the pending menu work's budget.
    def test_resume_menu_reentry_keeps_original_phase_deadline(self):
        for restart in (False,True):
            with self.subTest(restart=restart):
                self.c=self.controller()
                self.inputs.clear()
                self.activate('resume')
                self.c.journal.update(resume_pending=True,attempts={'stop':1})
                self.home(running=False,fishing=False)
                self.tick()
                self.assertEqual(self.c.phase,'resume_menu')
                self.assertEqual(self.inputs,['route-open'])
                deadlines={key:self.saved()[key] for key in ('deadline','phase_deadline')}
                self.assertEqual(deadlines['phase_deadline'],self.now+120)
                self.now=deadlines['phase_deadline']-18
                self.home()
                self.assertFalse(self.c.restored_settings(np.zeros((4,4,3),np.uint8)))
                self.tick();self.tick()
                self.assertEqual(self.c.phase,'resume_menu')
                self.assertEqual(self.inputs,['route-open'])
                self.tick()
                self.assertEqual(self.c.phase,'resume')
                self.assertEqual({key:self.saved()[key] for key in deadlines},deadlines)
                if restart:self.c=self.controller()
                self.home()
                self.tick()
                self.assertEqual(self.inputs,['route-open','stop'])
                self.home(running=False,fishing=False)
                self.tick()
                self.assertEqual(self.c.phase,'resume_menu')
                self.assertEqual(self.inputs,['route-open','stop','route-open'])
                self.assertEqual(self.saved()['attempts'],{'stop':2,'route-open':2})
                self.assertEqual({key:self.saved()[key] for key in deadlines},deadlines)
                self.c=self.controller()
                self.home(running=False,fishing=False)
                self.now=deadlines['phase_deadline']
                with self.assertRaises(RuntimeError):self.tick(seconds=0)
                self.assertEqual(self.c.phase,'blocked')
                self.assertEqual({key:self.saved()[key] for key in deadlines},deadlines)
                self.assertEqual(self.inputs,['route-open','stop','route-open'])
                self.assertEqual(self.saved()['attempts'],{'stop':2,'route-open':2})
                self.assertEqual(self.saved()['farm_origin'],FARM)
                self.assertTrue(self.saved()['resume_pending'])
                self.assertIsNone(self.saved()['farm_restored_at'])


    def stale_current_trip(self):
        self.activate('arrive')
        c=self.c
        c.journal.update(period='2026-10-26',pending=False,phase='idle',resume_pending=True,
            origin=dict(parallel=True,stage=10,rank=5,route=True,fishing=True),
            settings=deepcopy(SETTINGS),categories=[0],submitted=7,failed_at=90.,phase_deadline=95.,
            last_input=dict(kind='enter-confirm',state='issued_only'),
            attempts={k:1 for k in ('snapshot-open','snapshot-close','return-confirm','enter-confirm')})
        c.phase='idle';c._save()
        self.home(running=False,fishing=False)
        c.view._stage.return_value=10;c.view._rank_value.return_value=5

    def test_new_parallel_stage_replaces_unrestored_old_stage_target(self):
        self.c.journal.update(farm_origin=deepcopy(FARM),resume_pending=True)
        current=dict(parallel=True,stage=10,rank=5,route=True,fishing=True)
        self.c.origin_screen.return_value=deepcopy(current)
        self.home();self.confirm(due=True)
        self.assertEqual(self.saved()['target'],dict(current,settings=None))
        self.assertTrue(self.saved()['capture_target_settings'])

    def test_stale_target_read_only_reconciliation_keeps_old_budget_and_archive(self):
        import hashlib
        self.stale_current_trip();old=deepcopy(self.saved());raw=self.path.read_bytes()
        self.confirm(due=False)
        fixed=self.saved()
        self.assertEqual(fixed['target'],dict(old['origin'],settings=SETTINGS))
        for key in ('origin','deadline','phase_deadline','started_at','attempts','submitted','categories','failure'):
            self.assertEqual(fixed.get(key),old.get(key),key)
        digest=hashlib.sha256(raw).hexdigest()
        self.assertEqual((self.path.parent/'logs/guild-daily-retired'/(digest+'.json')).read_bytes(),raw)
        self.assertEqual(self.inputs,[])
        self.c=self.controller();self.home(running=False,fishing=False)
        self.c.view._stage.return_value=10;self.c.view._rank_value.return_value=5
        self.menu();self.confirm(due=False)
        self.assertEqual(self.saved()['resume_settings'],SETTINGS)
        self.home();self.c.view._rank_value.return_value=6
        self.confirm(due=False)
        self.assertFalse(self.saved()['resume_pending'])
        self.assertEqual(self.saved()['attempts'],old['attempts'])
        self.assertEqual(self.saved()['phase_deadline'],old['phase_deadline'])
        self.assertEqual(self.inputs,[])

    def test_stale_return_needs_new_frames_current_stage_settings_and_actual_fishing(self):
        self.stale_current_trip();old=self.path.read_bytes()
        frame=np.zeros((4,4,3),np.uint8)
        for _ in range(4):self.tick(due=True,frame=frame)
        self.assertEqual(self.path.read_bytes(),old)
        self.c.view._stage.return_value=9;self.confirm(due=True)
        self.assertEqual(self.path.read_bytes(),old)
        self.c.view._stage.return_value=10;self.confirm(due=False)
        self.menu(dict(SETTINGS,boss=True));self.confirm(due=False)
        self.assertNotIn('resume_settings',self.saved())
        self.home();self.confirm(due=True)
        self.assertTrue(self.saved()['resume_pending'])
        self.menu();self.confirm(due=False)
        self.home(fishing=False);self.confirm(due=False)
        self.assertTrue(self.saved()['resume_pending'])
        self.assertEqual(self.inputs,[])

    def test_unknown_old_snapshot_never_reconciles_or_replays_when_not_due(self):
        self.stale_current_trip();self.c.journal['settings'].pop('sale');self.c._save()
        old=self.path.read_bytes();self.confirm(due=False)
        self.assertEqual(self.path.read_bytes(),old)
        self.assertEqual(self.inputs,[])

    def test_matching_brief_menu_only_records_settings_not_restoration(self):
        self.stale_current_trip();self.confirm(due=False)
        self.menu();self.tick(due=False,seconds=.2)
        self.assertEqual(self.saved()['resume_settings'],SETTINGS)
        self.assertTrue(self.saved()['resume_pending'])
        self.assertIsNone(self.saved()['farm_restored_at'])
        self.assertEqual(self.inputs,[])
        self.home();self.tick(due=False,seconds=.2)
        self.assertTrue(self.saved()['resume_pending'])
        self.confirm(due=False)
        self.assertFalse(self.saved()['resume_pending'])

    def test_new_active_rank_replaces_old_unrestored_rank_target(self):
        self.c.journal.update(farm_origin=deepcopy(FARM),resume_pending=True)
        current=dict(parallel=True,stage=FARM['stage'],rank=FARM['rank']+1,route=True,fishing=True)
        self.c.origin_screen.return_value=deepcopy(current)
        self.home();self.confirm(due=True)
        self.assertEqual(self.saved()['target'],dict(current,settings=None))


if __name__ == '__main__':
    unittest.main()
