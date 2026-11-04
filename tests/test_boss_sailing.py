"""Early BOSS route ownership, fresh guards, and the RANK35 handoff."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from boss_training import BossTraining, training_options
from digit_ocr import NumberReading
from parallel_world import ParallelWorld
from training_sailing import TARGETS, TrainingSailing
from app import Clicker

ROOT = Path(__file__).resolve().parents[1]


def image(name):
    return np.asarray(Image.open(ROOT/'tests/fixtures/training-sailing'/f'{name}.png').convert('RGB')).copy()


def menu(selected):
    result = image('menu-three')
    # Use native selected/unselected resource tiles, including the icon's
    # changed illumination. The BOSS toggle varies only its checked corners.
    for name,(x,_,_) in TARGETS.items():
        if name in ('pearl','amber','gem'):
            source = image('menu-three' if name in selected else
                           'menu-no-pearl' if name == 'pearl' else 'menu-one')
            result[428:486,x-27:x+27] = source[428:486,x-27:x+27]
            continue
        for y in (428,477):
            for X in (x-26,x+17):
                result[y:y+8,X:X+8] = 254 if name in selected else 193
    return result


class Reader:
    value = 34
    def read(self, _):
        return NumberReading(str(self.value),self.value,.999)


class BossSailingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'training.json'
        self.reader = Reader()
        self.actions,self.events = [],[]
        self.world = ParallelWorld(ROOT,Mock(),Mock(),self.events.append,self.reader,
            training={'boss_auto_sailing_until_rank':35},training_state_path=self.path,
            sailing_action=self.click)
        self.world.phase,self.world.rank = 'play',34
        self.sailing,self.training = self.world.sailing,self.world.training
        self.current = image('inactive')

    def click(self,action,x,y,label):
        if not self.sailing.validate(action,self.current):
            return False
        self.sailing.before_click(action)
        self.actions.append((action,x,y,label))
        return True

    def step(self,frame,wanted=True,profile='boss',rank=34):
        self.current = frame
        self.sailing.next_at = 0
        return self.sailing.step(frame,wanted,profile=profile,rank=rank)

    def owned(self,verified=True):
        self.training.set_sailing(True,verified,profile='boss',rank=34)
        self.sailing.profile,self.sailing.route_rank = 'boss',34

    def test_cutoff_defaults_off_and_rejects_unknown_or_unsafe_values(self):
        self.assertEqual(training_options()['boss_auto_sailing_until_rank'],0)
        for value in (-1,40,True,35.5,'35'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                training_options({'boss_auto_sailing_until_rank':value})
        self.assertEqual(training_options({'boss_auto_sailing_until_rank':35})['boss_auto_sailing_until_rank'],35)

    def test_only_boss_tiles_and_extras_off_can_start_then_two_arrows_commit(self):
        self.step(image('inactive'))
        self.assertEqual(self.actions[-1][0],'open')
        selected = {'pearl','amber','gem'}
        for name in ('pearl','amber','gem','boss'):
            self.step(menu(selected))
            self.assertEqual(self.actions[-1][0],'target:'+name)
            selected.symmetric_difference_update({name})
        configured = menu({'boss'})
        self.assertTrue(self.sailing.configured(configured))
        self.assertFalse(self.sailing.configured(image('menu-three')))
        self.step(configured);self.step(configured)
        self.assertEqual(self.actions[-1][0],'start')
        self.assertFalse(self.training.state['sailing_verified'])
        self.step(image('active'))
        self.assertTrue(self.training.state['sailing_enabled'])
        self.assertFalse(self.training.state['sailing_verified'])
        self.assertFalse(self.step(image('active')))
        self.assertTrue(self.training.state['sailing_verified'])
        self.assertEqual(self.sailing.status()['targets'],['BOSS'])
        self.assertEqual(self.training.state['mode'],'idle')

    def test_unknown_extras_or_dimmed_boss_menu_never_authorize_start(self):
        self.sailing.profile = 'boss'
        native = menu({'boss'})
        for frame in ((native*.7).astype('uint8'),native.copy()):
            if frame.max() == 255:
                frame[636:658,566:586] = 25
            self.assertFalse(self.sailing.validate('start',frame))

    def test_ranks0_to34_only_and_existing_farming_always_has_priority(self):
        for rank in (0,1,30,34):
            self.world.rank = rank
            self.assertTrue(self.world._boss_sailing_wanted())
        for rank in (35,36,38,39,40,None):
            self.world.rank = rank
            self.assertFalse(self.world._boss_sailing_wanted())
        self.world.rank = 34
        self.training.begin(34,650,868)
        self.assertFalse(self.world._boss_sailing_wanted())
        self.assertTrue(self.world._sailing_wanted())

    def test_early_verified_route_does_not_run_training_or_manual_map_navigation(self):
        self.owned()
        self.training.evaluate = Mock(side_effect=AssertionError('early route must not start training'))
        self.assertFalse(self.world.step(image('active')))
        self.world.click.assert_not_called()
        self.assertEqual(self.actions,[])

    def test_rank35_stops_and_requires_two_off_frames_before_existing_training(self):
        self.owned()
        self.world.rank = self.reader.value = 35
        self.training.evaluate = Mock(return_value=None)
        self.current = image('active')
        self.assertTrue(self.world.step(self.current))
        self.assertEqual(self.actions[-1][0],'stop')
        self.training.evaluate.assert_not_called()
        self.current = image('stopped');self.sailing.next_at = 0
        self.assertTrue(self.world.step(self.current))
        self.assertTrue(self.training.state['sailing_enabled'])
        self.training.evaluate.assert_not_called()
        self.sailing.next_at = 0
        self.world.step(self.current.copy())
        self.assertFalse(self.training.state['sailing_enabled'])
        self.training.evaluate.assert_called_once()
        self.assertEqual([a[0] for a in self.actions],['stop'])

    def test_idle_verified_boss_route_reloads_without_repeated_start_or_stop(self):
        self.owned()
        reloaded = BossTraining(ROOT,self.reader,self.events.append,state_path=self.path)
        self.assertTrue(reloaded.state['sailing_enabled'])
        self.assertTrue(reloaded.state['sailing_verified'])
        self.assertEqual(reloaded.state['sailing_profile'],'boss')
        self.sailing.training = reloaded
        self.assertFalse(self.step(image('active')))
        self.assertEqual(self.actions,[])

    def test_unconfirmed_boss_start_reloads_but_must_stop_and_reconfigure(self):
        self.owned(False)
        reloaded = BossTraining(ROOT,self.reader,self.events.append,state_path=self.path)
        self.sailing.training = reloaded
        self.step(image('active'))
        self.assertEqual(self.actions[-1][0],'stop')
        self.assertFalse(reloaded.state['sailing_verified'])

    def test_missing_idle_ownership_rank_fails_closed(self):
        self.path.write_text(json.dumps({'version':1,'mode':'idle','sailing_enabled':True,
            'sailing_verified':True,'sailing_profile':'boss'}))
        with self.assertRaisesRegex(RuntimeError,'BOSS 自动航行保存记录无效'):
            BossTraining(ROOT,self.reader,self.events.append,state_path=self.path)

    def test_switch_to_resources_first_stops_the_verified_boss_route(self):
        self.owned()
        self.step(image('active'),profile='resources')
        self.assertEqual(self.actions[-1][0],'stop')
        self.assertEqual(self.sailing.profile,'boss')
        self.step(image('stopped'),profile='resources')
        self.step(image('stopped'),profile='resources')
        self.assertFalse(self.training.state['sailing_enabled'])
        self.assertEqual(self.sailing.profile,'boss')
        self.training.begin(34,650,868)
        self.step(image('stopped'),profile='resources')
        self.assertEqual(self.sailing.profile,'resources')
        self.assertEqual(self.actions[-1][0],'open')
        self.assertEqual(self.sailing.status()['targets'],['珍珠','琥珀','蓝色宝石'])

    def test_bag_pause_retains_boss_profile_and_resume_revalidates_goals(self):
        self.owned()
        self.step(image('active'),False)
        self.step(image('stopped'),False);self.step(image('stopped'),False)
        self.assertEqual(self.training.state['sailing_profile'],'boss')
        self.assertTrue(self.world._boss_sailing_wanted())
        self.step(image('stopped'))
        self.assertEqual(self.actions[-1][0],'open')
        self.assertFalse(self.sailing.validate('start',image('menu-three')))

    def test_new_rank_world_or_running_arrow_rejects_new_target_inputs(self):
        self.sailing.profile = 'boss'
        frame = menu({'boss'})
        self.assertTrue(self.sailing.validate('start',frame))
        self.reader.value = 35
        self.assertFalse(self.sailing.validate('start',frame))
        self.reader.value = 34
        with patch.object(self.world,'_parallel',return_value=False):
            self.assertFalse(self.sailing.validate('start',frame))
        self.assertFalse(self.sailing.validate('open',image('active')))

    def test_rank_vote_shared_with_step_is_not_counted_as_another_new_frame(self):
        self.owned()
        self.reader.value = 35
        a = image('active')
        self.world.observe_boss_sailing_rank(a,ready=True)
        self.assertTrue(self.world.boss_sailing_stop_requested())
        self.world.step(a)
        self.assertEqual(self.world.rank,34)
        self.assertEqual(len(self.world.rank_readings),1)
        self.world.observe_boss_sailing_rank(a.copy(),ready=True)
        self.assertEqual(self.world.rank,34)
        self.world.observe_boss_sailing_rank(a.copy(),ready=True)
        self.assertEqual(self.world.rank,35)

    def test_cutoff_prevents_new_long_weapon_hold_and_requires_new_frame_to_close(self):
        self.owned()
        a = image('active')
        self.world.latest_rank_frame = a
        self.assertTrue(self.world.boss_sailing_hold_allowed(a))
        self.reader.value = 35
        self.assertFalse(self.world.boss_sailing_hold_allowed(a))
        self.world.rank = 35
        self.assertTrue(self.world.boss_sailing_close_guard(a))
        with patch.object(self.world,'_parallel',return_value=False):
            self.assertFalse(self.world.boss_sailing_close_guard(a))

    def test_disabled_cutoff_drains_saved_route_before_allowing_other_navigation(self):
        self.owned()
        self.training.options['boss_auto_sailing_until_rank'] = 0
        self.assertTrue(self.world.boss_sailing_stop_requested())
        self.current = image('active')
        self.world.step(self.current)
        self.assertEqual(self.actions[-1][0],'stop')
        self.world.click.assert_not_called()

    def test_early_boss_profile_never_invokes_resource_meter_drag_or_long_hold(self):
        self.owned()
        self.sailing.fishing = Mock()
        self.sailing._wake = Mock(side_effect=AssertionError('resource recovery on task BOSS'))
        self.sailing._adopt_new_start = Mock(side_effect=AssertionError('resource start on task BOSS'))
        self.step(image('active'))
        self.assertEqual(self.actions,[])

    def test_rejected_new_inputs_preserve_the_original40second_limit(self):
        self.sailing.action = Mock(return_value=False)
        self.step(image('inactive'))
        self.sailing.since -= 41
        with self.assertRaisesRegex(RuntimeError,'保留练级记录'):
            self.step(image('inactive'))
        self.assertEqual(self.actions,[])

    def test_unrecognized_frames_reset_cutoff_votes_and_cannot_close_equipment(self):
        self.owned()
        self.reader.value = 35
        a = image('active')
        self.world.observe_boss_sailing_rank(a,ready=True)
        self.world.observe_boss_sailing_rank(a.copy(),ready=False)
        self.assertEqual(len(self.world.rank_readings),0)
        self.world.rank = 35
        with patch.object(self.world,'_stage_strokes',return_value=None):
            self.assertFalse(self.world.boss_sailing_close_guard(a))
            self.assertFalse(self.world.boss_sailing_hold_allowed(a))

    def test_world_change_does_not_authorize_original_world_map_navigation(self):
        self.owned()
        with patch.object(self.world,'_parallel',return_value=False):
            self.assertTrue(self.world.step(image('active')))
        self.assertTrue(self.training.state['sailing_enabled'])
        self.assertEqual(self.actions,[])
        self.world.click.assert_not_called()

    def test_persistent_empty_sleeping_dial_stops_boss_route_before_task_recovery(self):
        self.owned()
        self.world.fishing = Mock()
        self.world.fishing.at_port.return_value = False
        self.world.fishing.idle_icon.return_value = 'sleep'
        self.world.fishing.blue_fill.return_value = False
        self.world.fishing.busy.return_value = False
        self.world.defense = Mock()
        self.world.defense._match.return_value = False
        self.world.latest_rank_value = 34
        a = image('active')
        with patch.object(self.training,'boss_mode',return_value=True),patch.object(self.world,'_fixed',return_value=False):
            self.world._observe_boss_route_idle(a)
            self.world.boss_sailing_idle_since -= 4
            self.world._observe_boss_route_idle(a.copy())
            self.assertTrue(self.world._boss_sailing_wanted())
            self.world._observe_boss_route_idle(a.copy())
        self.assertFalse(self.world._boss_sailing_wanted())
        self.assertEqual(self.training.state['boss_sailing_sleep_rank'],34)
        self.assertEqual(self.actions,[])
        reloaded = BossTraining(ROOT,self.reader,self.events.append,state_path=self.path)
        self.assertEqual(reloaded.state['boss_sailing_sleep_rank'],34)

    def test_existing_fill_prevents_the_boss_route_sleep_handoff(self):
        self.owned()
        self.world.fishing = Mock()
        self.world.fishing.at_port.return_value = False
        self.world.fishing.idle_icon.return_value = 'sleep'
        self.world.fishing.blue_fill.return_value = True
        self.world.latest_rank_value = 34
        with patch.object(self.training,'boss_mode',return_value=True):
            for _ in range(4):self.world._observe_boss_route_idle(image('active'))
        self.assertNotIn('boss_sailing_sleep_rank',self.training.state)
        self.assertEqual(self.actions,[])

    def handoff_world(self):
        actions = []
        reader = Reader();reader.value = 35
        world = ParallelWorld(ROOT,Mock(),Mock(),self.events.append,reader,
            training={'boss_auto_sailing_until_rank':35},training_state_path=self.path,
            fishing_action=Mock(),sailing_action=Mock(),defense_action=Mock(),
            portal_action=lambda action: actions.append(action) or True)
        world.phase,world.rank = 'play',35
        port = image('handoff-port-rank35')
        world.latest_rank_value,world.latest_rank_frame = 35,port
        return world,port,actions

    def test_actual_stopped_port_at_cutoff_opens_current_task_without_next_arrow(self):
        world,port,actions = self.handoff_world()
        self.assertTrue(world.fishing.at_port(port))
        self.assertFalse(world._fixed(port,'next-arrow',(307,64,398,127),.88))
        self.assertTrue(world._boss_handoff_ready(port))
        world.step(port)
        self.assertEqual(world.phase,'boss_handoff_map')
        self.assertEqual(actions,[])
        world.next_action_at = 0
        world.step(port.copy())
        self.assertEqual([a['kind'] for a in actions],['handoff-map'])
        self.assertTrue(world.validate_portal(actions[0],port.copy()))
        self.assertEqual(world.training.state['boss_sailing_handoff_rank'],35)
        reloaded = BossTraining(ROOT,world.reader,self.events.append,state_path=self.path)
        self.assertEqual(reloaded.state['boss_sailing_handoff_rank'],35)

    def test_handoff_map_rejects_new_rank_running_route_unknown_stage_and_world(self):
        world,port,actions = self.handoff_world()
        world._boss_handoff_begin(port)
        world.step(port.copy())
        action = actions[0]
        world.reader.value = 36
        self.assertFalse(world.validate_portal(action,port.copy()))
        world.reader.value = 35
        changed = port.copy();changed[45:72,37:69] = 0
        self.assertFalse(world.validate_portal(action,changed))
        self.assertFalse(world.validate_portal(action,(port*.4).astype(np.uint8)))
        with patch.object(world.sailing,'running',return_value=True):
            self.assertFalse(world.validate_portal(action,port))
        with patch.object(world,'_parallel',return_value=False):
            self.assertFalse(world.validate_portal(action,port))
        world.training.state['sailing_enabled'] = True
        self.assertFalse(world.validate_portal(action,port))
        self.assertEqual(len(actions),1)

    def test_handoff_map_keeps_three_attempts30second_and_total60second_limits(self):
        world,port,actions = self.handoff_world()
        world._boss_handoff_begin(port)
        since = world.phase_since
        for _ in range(5):
            world.next_action_at = 0
            world.step(port.copy())
        self.assertEqual(len(actions),3)
        self.assertEqual(world.phase_since,since)
        world.phase_since -= 31
        with self.assertRaisesRegex(RuntimeError,'30 秒'):
            world.step(port.copy())
        world.phase_since = since
        world.boss_handoff_since -= 61
        with self.assertRaisesRegex(RuntimeError,'60 秒'):
            world.step(port.copy())

    def test_handoff_requires_three_stopped_maps_then_real_skull_selection_and_close(self):
        world,port,actions = self.handoff_world()
        world._boss_handoff_begin(port)
        frame = np.asarray(Image.open(ROOT/'tests/fixtures/parallel/parallel-map-rank2.png').convert('RGB')).copy()
        # Supply the observed cutoff header to this older real map replay;
        # actual map symbols/legend/skull and their thresholds remain intact.
        frame[:80] = port[:80]
        self.assertTrue(world._boss_handoff_ready(frame,map_ready=True))
        for _ in range(2):
            world.step(frame.copy())
            self.assertEqual(world.phase,'boss_handoff_map')
        world.step(frame.copy())
        self.assertEqual(world.phase,'find_boss')
        world.next_action_at = 0
        world.step(frame.copy())
        self.assertEqual(actions[-1]['kind'],'handoff-select')
        self.assertEqual(world.target_rank,35)
        world.next_action_at = 0
        world.step(frame.copy())
        self.assertEqual(actions[-1]['kind'],'handoff-close')
        world.next_action_at = 0
        world.step(port.copy())
        self.assertEqual(world.phase,'play')
        self.assertNotIn('boss_sailing_handoff_rank',world.training.state)
        self.assertIsNone(world.boss_handoff_since)

    def test_cold_port_relocation_is_off_when_cutoff_disabled_or_farming(self):
        for changes in ({'boss_auto_sailing_until_rank':0},{'mode':'farming'}):
            world,port,actions = self.handoff_world()
            if 'mode' in changes:
                world.training.begin(35,778,1112)
            else:
                world.training.options.update(changes)
            world.training.evaluate = Mock(return_value=None)
            world.sailing.step = Mock(return_value=False)
            world.step(port)
            self.assertNotEqual(world.phase,'boss_handoff_map')
            self.assertEqual(actions,[])

    def test_handoff_new_map_can_resume_a_saved_marker_after_restart(self):
        world,port,actions = self.handoff_world()
        world.training.state['boss_sailing_handoff_rank'] = 35
        world.training._save()
        frame = np.asarray(Image.open(ROOT/'tests/fixtures/parallel/parallel-map-rank2.png').convert('RGB')).copy()
        frame[:80] = port[:80]
        world.step(frame)
        self.assertEqual(world.phase,'boss_handoff_map')
        self.assertEqual(actions,[])
        self.assertEqual(world.training.state['boss_sailing_handoff_rank'],35)

    def test_handoff_map_never_drags_or_cancels_existing_fill_and_battle_blocks_open(self):
        world,port,actions = self.handoff_world()
        with patch.object(world.fishing,'blue_fill',return_value=True):
            self.assertTrue(world._boss_handoff_ready(port))
        with patch.object(world.defense,'_match',return_value=True):
            self.assertFalse(world._boss_handoff_ready(port))
            self.assertTrue(world._boss_handoff_ready(port,allow_battle=True))
        self.assertEqual(actions,[])
        world.rank = world.reader.value = 40
        self.assertFalse(world._boss_handoff_ready(port))

    def test_invalid_handoff_record_fails_closed(self):
        self.path.write_text(json.dumps({'version':1,'mode':'idle','boss_sailing_handoff_rank':True}))
        with self.assertRaisesRegex(RuntimeError,'交接记录无效'):
            BossTraining(ROOT,self.reader,self.events.append,state_path=self.path)


class BossSailingWorkerTests(unittest.TestCase):
    def test_worker_stops_boss_route_before_optional_battle_damage_at_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT/'assets',root/'assets')
            config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            rule = deepcopy(next(r for r in config['rules'] if r['id']=='driftinggirl_parallel_world'))
            rule.update(enabled=True,training=training_options({'boss_auto_sailing_until_rank':35,'battle_tap':True}))
            config.update(rules=[rule],poll_interval=.1)
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            reader = Reader();reader.value = 35
            training = BossTraining(ROOT,reader,lambda _:None,state_path=root/'training_state.json')
            training.set_sailing(True,True,profile='boss',rank=34)
            taps,defense_ownership = [],[]

            class Device:
                def __init__(self,_,stop):
                    self.stop,self.io_lock = stop,threading.RLock()
                    self.screen,self.index = 'active',0
                def screenshot(self):
                    self.index += 1
                    if self.index >= 18:self.stop.set()
                    return image(self.screen)
                def click(self,button):
                    taps.append((button.button[0]+5,button.button[1]+5))
                    self.screen = 'stopped'
                def stuck_record_add(self,_):pass

            app = Clicker(root,Device)
            app.digit_reader = reader
            def create_world(*args,**kwargs):
                world = ParallelWorld(*args,**kwargs)
                world.phase,world.rank = 'play',35
                def observe_rank(frame):
                    world.latest_rank_value,world.latest_rank_frame = 35,frame
                    return 35
                world._rank = observe_rank
                original = world.sailing.step
                def immediate(frame,wanted,**options):
                    world.sailing.next_at = 0
                    return original(frame,wanted,**options)
                world.sailing.step = immediate
                def damage(*_):
                    defense_ownership.append(world.training.state.get('sailing_enabled',False))
                    app.stop_event.set()
                    return False
                world.defense.step = damage
                world.training.evaluate = Mock(return_value=None)
                return world
            try:
                with patch('app.ParallelWorld',side_effect=create_world):
                    app.start(preview=False)
                    app.thread.join(8)
                self.assertFalse(app.running())
                self.assertIsNone(app.state['last_error'])
                self.assertEqual(taps,[(233,45)])
                self.assertEqual(defense_ownership,[False])
                saved = json.loads((root/'training_state.json').read_text())
                self.assertFalse(saved['sailing_enabled'])
                self.assertEqual(saved['mode'],'idle')
            finally:
                app.stop()
                if app.thread:app.thread.join(3)
                for handler in list(app.log.handlers):
                    handler.close();app.log.removeHandler(handler)


if __name__ == '__main__':
    unittest.main()
