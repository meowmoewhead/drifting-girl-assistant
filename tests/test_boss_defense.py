from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
import hashlib
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import Clicker
from boss_training import BossTraining, training_options
from boss_defense import BossDefense
from digit_ocr import NumberReading
from parallel_world import ParallelWorld
from parallel_skills import ParallelSkills
from equipment_upgrade import EquipmentUpgrade


ROOT = Path(__file__).resolve().parents[1]


def image(path):
    return np.asarray(Image.open(ROOT/path).convert('RGB')).copy()


def guarded_battle():
    # Compose control anchors for guard tests; this is not a real Sea God win.
    frame = image('tests/fixtures/training/battle.png')
    frame[12:59,68:135] = image('assets/boss-defense/rank39.png')
    source = image('assets/navigation/parallel-boss-battle.png')
    frame[38:75,254:289] = source[38:75,254:289]
    frame[174:194,195:227] = image('assets/boss-defense/sea-god-name.png')
    frame[460:520,480:540] = (245,220,165)
    frame[520:610,485:535] = (35,20,8)
    return frame


class Reader:
    def read(self,_):
        return NumberReading('39',39,.999)


class BossDefenseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.training = BossTraining(ROOT,Reader(),lambda _:None,
                                     state_path=Path(self.temp.name)/'state.json')
        # 这个类验证的是"只点充能武器"的老语义,所以保持 game 配置里的默认
        # battle_tap=False。打 40 BOSS 时两种攻击会被强制打开(见
        # drift/tasks/final_boss_attack.py),那部分行为在
        # tests/test_final_boss_attack.py 里单独验证。
        self.actions = []
        self.defense = BossDefense(ROOT,self.training,lambda t:self.actions.append(t) or 3,lambda _:None)

    def tearDown(self):
        self.temp.cleanup()

    def test_battle_is_owned_and_glowing_weapon_receives_repeated_taps(self):
        frame = guarded_battle()
        self.assertTrue(self.defense.battle(frame,39))
        self.assertTrue(self.defense.step(frame,39))
        self.assertEqual(self.actions[0]['side'],'right')
        self.assertEqual(self.defense.taps,3)
        frame[245:690,410:614] = 0
        self.assertTrue(self.defense.step(frame,39))
        self.assertEqual(len(self.actions),1)

    def test_native_other_fish_waiting_menus_and_wrong_rank_never_trigger(self):
        for path in ('tests/fixtures/training/battle.png','tests/fixtures/fishing-meter/low-meter-waiting.png',
                     'tests/fixtures/fishing-meter/low-meter-bite.png',
                     'tests/fixtures/parallel-skills/parallel-growth-confirm.png',
                     'tests/fixtures/equipment/panel-top.png'):
            candidate = ROOT/path
            if candidate.is_file():
                self.assertFalse(self.defense.step(image(path),39),path)
        frame = guarded_battle()
        self.assertFalse(self.defense.step(frame,38))
        self.assertFalse(self.defense.step(frame,40))
        frame[46:72,13:39] = 0
        self.assertFalse(self.defense.step(frame,39))
        self.assertEqual(self.actions,[])

    def test_rank_in_capture_and_all_battle_anchors_are_required(self):
        original = guarded_battle()
        for x,y,X,Y in ((68,12,135,59),(254,38,289,75),(307,962,415,1057),(188,125,566,165)):
            frame = original.copy();frame[y:Y,x:X] = 0
            self.assertFalse(self.defense.battle(frame,39))
        self.assertFalse(self.defense.battle(original//2,39))

    def test_fresh_weapon_change_prevents_click_and_side_ui_is_excluded(self):
        frame = guarded_battle();expected = self.defense.locate(frame,39)
        self.assertIsNotNone(self.defense.validate(frame,39,expected))
        frame[245:690,410:614] = 0
        frame[600:665,650:705] = (255,230,40)
        self.assertIsNone(self.defense.validate(frame,39,expected))
        self.assertIsNone(self.defense.locate(frame,39))

    def test_both_switches_are_boolean_and_combat_can_be_disabled(self):
        for key in ('boss_interrupt','meter_auto_adjust'):
            self.assertTrue(training_options()[key])
            for bad in (1,'true',None):
                with self.assertRaises(ValueError):training_options({key:bad})
        self.training.options['boss_interrupt'] = False
        self.assertFalse(self.defense.step(guarded_battle(),39))
        self.assertEqual(self.actions,[])

    def test_native_raised_weapon_cores_and_sword_animation(self):
        for name,side,box in (('raised-left','left',(130,500,215,590)),
                              ('raised-right','right',(475,445,535,530))):
            with self.subTest(name=name):
                frame=image('tests/fixtures/boss-defense/'+name+'.png')
                self.assertTrue(self.defense.battle(frame,39))
                target=self.defense.locate(frame,39)
                self.assertIsNotNone(target)
                self.assertEqual(target['side'],side)
                x,y,X,Y=box
                self.assertTrue(x<=target['x']<=X and y<=target['y']<=Y)
                self.assertIsNotNone(self.defense.validate(frame,39,target))

    def test_native_ship_fire_lowered_weapons_and_attack_flash_are_not_targets(self):
        for name in ('ship-fire','lowered-weapon','attack-flash',
                     'native-lamp-reflection','native-lamp-reflection-wide',
                     'native-reward-gears-right','native-reward-gears-chain-right',
                     'native-right-attack-arc','native-right-no-visible-arm',
                     'native-right-firing-elongated'):
            with self.subTest(name=name):
                frame=image('tests/fixtures/boss-defense/'+name+'.png')
                self.assertTrue(self.defense.battle(frame,39))
                self.assertIsNone(self.defense.locate(frame,39))
        frame=image('tests/fixtures/boss-defense/raised-right.png')
        frame[1137:1192,208:269]=0
        self.assertFalse(self.defense.battle(frame,39))

    def test_native_pulsing_blades_and_pale_weapon_cores_are_identified(self):
        for name,side,box in (
                ('native-pulse-right','right',(484,477,557,542)),
                ('native-pulse-left','left',(129,528,206,620)),
                ('native-white-left','left',(136,541,209,633))):
            with self.subTest(name=name):
                frame=image('tests/fixtures/boss-defense/'+name+'.png')
                self.assertTrue(self.defense.battle(frame,39))
                target=self.defense.locate(frame,39)
                self.assertIsNotNone(target)
                self.assertEqual(target['side'],side)
                x,y,X,Y=box
                self.assertTrue(x<=target['x']<=X and y<=target['y']<=Y,target)
                self.assertGreaterEqual(target['core_radius'],12)

    def test_native_right_tip_is_complete_and_attached_despite_touch_bubbles(self):
        for name in ('native-right-whole-cap','native-right-bubbles'):
            with self.subTest(name=name):
                frame=image('tests/fixtures/boss-defense/'+name+'.png')
                self.assertTrue(self.defense.battle(frame,39))
                target=self.defense.locate(frame,39)
                self.assertEqual(target['side'],'right')
                self.assertLess(target['box'][0],460)
                self.assertTrue(455<=target['x']<=525 and 440<=target['y']<=530,target)
                self.assertGreaterEqual(target['core_radius'],12)

    def test_right_pale_object_without_a_dark_attached_arm_is_not_a_target(self):
        frame=guarded_battle()
        frame[245:690,410:614]=0
        frame[460:520,480:540]=(245,220,165)
        self.assertTrue(self.defense.battle(frame,39))
        self.assertIsNone(self.defense.locate(frame,39))

    def test_native_early_right_charge_keeps_its_visible_shoulder(self):
        frame=image('tests/fixtures/boss-defense/native-right-early-charge.png')
        self.assertTrue(self.defense.battle(frame,39))
        target=self.defense.locate(frame,39)
        self.assertIsNotNone(target)
        self.assertEqual(target['side'],'right')
        self.assertTrue(490<=target['x']<=550 and 550<=target['y']<=620,target)
        self.assertGreaterEqual(target['core_radius'],12)
        for box in ((276,881,310,913),(208,1137,269,1192),
                    (539,1137,600,1186),(307,962,415,1057)):
            x,y,X,Y=box
            missing=frame.copy();missing[y:Y,x:X]=0
            self.assertIsNone(self.defense.locate(missing,39))

    def test_native_right_charge_starts_below_the_left_arm_limit(self):
        for name in ('native-right-charge-lowest','native-right-charge-low',
                     'native-right-charge-rising'):
            with self.subTest(name=name):
                frame=image('tests/fixtures/boss-defense/'+name+'.png')
                self.assertTrue(self.defense.battle(frame,39))
                target=self.defense.locate(frame,39)
                self.assertIsNotNone(target)
                self.assertEqual(target['side'],'right')
                self.assertTrue(495<=target['x']<=560 and 595<=target['y']<=665,target)
                self.assertGreaterEqual(target['core_radius'],12)
                self.assertIsNone(self.defense.locate(frame,40))
                self.assertIsNone(self.defense.locate(frame//2,39))

    def test_thin_reflections_do_not_authorize_input_even_with_battle_guards(self):
        frame=guarded_battle()
        frame[245:690,410:614]=0
        # Similar brightness on a narrow background lamp is not a wide cap.
        frame[480:540,560:598]=(255,245,230)
        self.assertTrue(self.defense.battle(frame,39))
        self.assertIsNone(self.defense.locate(frame,39))

    def test_native_right_charge_survives_touch_holes_and_a_joined_combat_arc(self):
        # Exact native after-images from the video containing a failed right
        # counter. These are still raised/charged, not proof of cancellation.
        for name in ('native-right-touch-raised','native-right-touch-leaning',
                     'native-right-touch-low','native-right-touch-low-second',
                     'native-right-touch-rising','native-right-touch-arc-joined',
                     'native-right-touch-top'):
            with self.subTest(name=name):
                frame=image('tests/fixtures/boss-defense/'+name+'.png')
                target=self.defense.locate(frame,39)
                self.assertIsNotNone(target)
                self.assertEqual(target['side'],'right')
                self.assertTrue(450<=target['x']<=550 and 450<=target['y']<=660,target)
                self.assertGreaterEqual(target['core_radius'],12)
                # The current visible gold cap, not a grey touch-dot centre.
                r,g,b=map(int,frame[target['y'],target['x']])
                self.assertGreaterEqual(r-b,25)
                self.assertGreater(g,b)

    def test_native_right_touch_effects_alone_do_not_extend_the_charge(self):
        for name in ('native-right-touch-insufficient-core',
                     'native-right-touch-dropped','native-right-charge-ended',
                     'native-right-touch-darkening'):
            with self.subTest(name=name):
                frame=image('tests/fixtures/boss-defense/'+name+'.png')
                self.assertTrue(self.defense.battle(frame,39))
                self.assertIsNone(self.defense.locate(frame,39))

    def test_right_touch_hole_recovery_keeps_every_fresh_battle_guard(self):
        source=image('tests/fixtures/boss-defense/native-right-touch-raised.png')
        self.assertIsNotNone(self.defense.locate(source,39))
        for box in ((68,12,135,59),(254,38,289,75),(188,125,566,165),
                    (195,174,227,194),(276,881,310,913),(208,1137,269,1192),
                    (539,1137,600,1186),(307,962,415,1057)):
            with self.subTest(box=box):
                x,y,X,Y=box
                fresh=source.copy();fresh[y:Y,x:X]=0
                self.assertIsNone(self.defense.locate(fresh,39))
        for rank in (38,40,None):
            self.assertIsNone(self.defense.locate(source,rank))
        self.assertIsNone(self.defense.locate(source//2,39))

    def test_right_touch_holes_take_priority_over_neutral_damage_each_new_frame(self):
        names=('native-right-touch-low','native-right-touch-arc-joined',
               'native-right-touch-raised')
        frames=[image('tests/fixtures/boss-defense/'+name+'.png') for name in names]
        self.training.options['battle_tap']=True
        index=[0];hits=[]
        def capture():
            frame=frames[index[0]];index[0]+=1
            return frame
        self.assertEqual(self.defense.burst(capture,hits.append,39,lambda:False,max_frames=3),3)
        self.assertEqual(index[0],3)
        self.assertTrue(all(t['side']=='right' and t.get('kind')!='damage' for t in hits))
        self.assertGreater(len({(t['x'],t['y']) for t in hits}),2)
        self.assertEqual(self.defense.last_burst['damage_taps'],0)

    def test_native_occlusion_still_waits_without_relaxing_other_guards(self):
        obscured=image('tests/fixtures/boss-defense/native-footer-occluded.png')
        self.assertTrue(self.defense.encounter(obscured,39))
        self.assertFalse(self.defense.battle(obscured,39))
        self.assertIsNone(self.defense.locate(obscured,39))
        original=image('tests/fixtures/boss-defense/native-pulse-right.png')
        for x,y,X,Y in ((276,881,310,913),(208,1137,269,1192),
                       (539,1137,600,1186),(307,962,415,1057)):
            frame=original.copy();frame[y:Y,x:X]=0
            self.assertIsNone(self.defense.locate(frame,39))
        for rank in (38,40,None):
            self.assertIsNone(self.defense.locate(original,rank))
        self.assertIsNone(self.defense.locate(original//2,39))

    def test_task_skull_during_an_ordinary_fish_does_not_acquire_sea_god(self):
        frame=image('tests/fixtures/boss-defense/task-ordinary-fish.png')
        frame[320:380,480:525]=(245,220,165)
        self.assertFalse(self.defense.owns(frame,39))
        self.assertFalse(self.defense.step(frame,39))
        self.assertEqual(self.actions,[])

    def test_native_burst_waits_and_reacquires_the_current_weapon_from_each_capture(self):
        idle=image('tests/fixtures/boss-defense/ship-fire.png')
        charging=image('tests/fixtures/boss-defense/raised-left.png')
        right=image('tests/fixtures/boss-defense/raised-right.png')
        frames=iter((idle,charging,right,idle))
        hits=[]
        self.assertEqual(self.defense.burst(lambda:next(frames),hits.append,39,lambda:False,max_frames=4),2)
        self.assertTrue(130<=hits[0]['x']<=215 and 500<=hits[0]['y']<=590)
        self.assertTrue(475<=hits[1]['x']<=535 and 445<=hits[1]['y']<=530)
        self.assertEqual(self.defense.last_side,'right')

    def test_high_frequency_burst_reacquires_after_an_obscured_glow_and_never_taps_idle(self):
        left=image('tests/fixtures/boss-defense/raised-left.png')
        idle=image('tests/fixtures/boss-defense/ship-fire.png')
        right=image('tests/fixtures/boss-defense/raised-right.png')
        ordinary=image('tests/fixtures/boss-defense/task-ordinary-fish.png')
        frames=iter((left,idle,right,ordinary))
        hits=[]
        self.assertEqual(self.defense.burst(lambda:next(frames),hits.append,39,lambda:False),2)
        self.assertEqual([h['side'] for h in hits],['left','right'])

    def test_high_frequency_burst_can_exceed_three_taps_but_is_bounded_and_cancellable(self):
        charging=image('tests/fixtures/boss-defense/raised-right.png')
        hits=[];captures=[]
        def capture():
            captures.append(1)
            return charging.copy()
        self.assertEqual(self.defense.burst(capture,hits.append,39,lambda:False),12)
        self.assertEqual(len(captures),12)
        self.assertEqual(self.defense.last_burst['taps'],12)
        self.assertEqual(len(self.defense.last_burst['tap_gaps_ms']),11)
        hits.clear();captures.clear()
        self.assertEqual(self.defense.burst(capture,hits.append,39,lambda:len(captures)>=6),5)
        self.assertEqual(len(captures),6)
        self.assertEqual(len(hits),5)

    def test_refused_stale_contact_is_not_counted_or_reused(self):
        charging=image('tests/fixtures/boss-defense/raised-right.png')
        frames=[];contacts=[]
        def hit(target):
            contacts.append(target)
            return False if len(contacts)==1 else None
        taps=self.defense.burst(lambda:frames.append(1) or charging,hit,39,lambda:False,
            max_frames=3,max_taps=2)
        self.assertEqual((taps,len(frames),len(contacts)),(2,3,3))
        self.assertEqual(len(self.defense.last_burst['tap_gaps_ms']),1)

    def test_high_frequency_burst_checks_time_again_after_each_capture(self):
        charging=image('tests/fixtures/boss-defense/raised-right.png')
        clock=[0.];hits=[]
        def capture():
            clock[0]+=1
            return charging.copy()
        with patch('boss_defense.time.monotonic',side_effect=lambda:clock[0]):
            self.assertEqual(self.defense.burst(capture,hits.append,39,lambda:False),3)
        self.assertEqual(len(hits),3)
        self.assertEqual(self.defense.last_burst['tap_gaps_ms'],[1000.,1000.])

    def test_burst_aborts_on_world_change_stop_or_finished_battle_and_is_bounded(self):
        idle=image('tests/fixtures/boss-defense/ship-fire.png')
        charging=image('tests/fixtures/boss-defense/raised-right.png')
        ordinary=image('tests/fixtures/boss-defense/task-ordinary-fish.png')
        for changed in (ordinary,charging//2):
            hits=[];frames=iter((idle,changed,charging))
            self.assertEqual(self.defense.burst(lambda:next(frames),hits.append,39,lambda:False),0)
            self.assertEqual(hits,[])
        stopped=[];hits=[]
        def capture_then_stop():
            stopped.append(True)
            return charging
        self.assertEqual(self.defense.burst(capture_then_stop,hits.append,39,lambda:bool(stopped)),0)
        captures=[]
        self.assertEqual(self.defense.burst(lambda:captures.append(1) or idle,hits.append,39,
                                           lambda:False,max_frames=3),0)
        self.assertEqual(len(captures),3)
        self.assertEqual(hits,[])

    def test_encounter_is_held_while_sword_guard_is_obscured_but_does_not_click(self):
        frame=image('tests/fixtures/boss-defense/raised-right.png')
        frame[962:1057,307:415]=0
        self.assertTrue(self.defense.owns(frame,39))
        self.assertFalse(self.defense.battle(frame,39))
        self.assertTrue(self.defense.step(frame,39))
        self.assertEqual(self.actions,[])


class BattleDamageTests(unittest.TestCase):
    def setUp(self):
        BossDefenseTests.setUp(self)
        self.training.options['battle_tap'] = True

    def tearDown(self):
        BossDefenseTests.tearDown(self)

    def test_damage_is_opt_in_boolean_and_does_not_require_counterattack(self):
        self.assertFalse(training_options()['battle_tap'])
        for bad in (1,'true',None):
            with self.assertRaises(ValueError):training_options({'battle_tap':bad})
        idle=image('tests/fixtures/boss-defense/ship-fire.png')
        self.assertTrue(self.defense.damage_battle(idle,39))
        self.training.options['boss_interrupt']=False
        self.assertFalse(self.defense.owns(idle,39))
        self.assertTrue(self.defense.damage_battle(idle,39))
        self.training.options['battle_tap']=False
        self.assertFalse(self.defense.damage_battle(idle,39))

    def test_damage_requires_new_meter_timer_world_rank_and_complete_battle_footer(self):
        original=image('tests/fixtures/boss-defense/ship-fire.png')
        for x,y,X,Y in ((153,205,192,245),(188,125,610,165),(8,40,46,80),
                       (65,9,139,63),(276,881,310,913),(208,1137,269,1192),
                       (539,1137,600,1186),(307,962,415,1057)):
            frame=original.copy();frame[y:Y,x:X]=0
            self.assertFalse(self.defense.damage_battle(frame,39),(x,y,X,Y))
        for frame in (original//2,original[:720],
                      image('tests/fixtures/training/failed.png'),
                      image('tests/fixtures/training/success.png'),
                      image('tests/fixtures/training/waiting.png'),
                      image('tests/fixtures/parallel-skills/parallel-growth-confirm.png'),
                      image('tests/fixtures/boss-defense/equipment-occlusion.png')):
            self.assertFalse(self.defense.damage_battle(frame,39))
        self.assertFalse(self.defense.damage_battle(original,40))
        self.assertFalse(self.defense.damage_battle(original,None))

    def test_normal_fish_needs_a_confirmed_current_rank_and_never_uses_sea_god_guard(self):
        ordinary=image('tests/fixtures/training/battle.png')
        self.assertFalse(self.defense.damage_battle(ordinary,34))
        self.defense.confirm_rank(ordinary,34)
        self.assertTrue(self.defense.damage_battle(ordinary,34))
        self.assertFalse(self.defense.owns(ordinary,34))
        self.assertFalse(self.defense.damage_battle(ordinary,35))
        changed=ordinary.copy();changed[12:50,72:130]=0
        self.assertFalse(self.defense.damage_battle(changed,34))
        self.defense.confirm_rank(ordinary,40)
        self.assertFalse(self.defense.damage_battle(ordinary,34))

    def test_charging_weapon_preempts_damage_taps_on_each_fresh_frame(self):
        idle=image('tests/fixtures/boss-defense/ship-fire.png')
        left=image('tests/fixtures/boss-defense/raised-left.png')
        right=image('tests/fixtures/boss-defense/raised-right.png')
        frames=iter((idle,left,right,idle));clock=[100.];hits=[]
        def capture():
            clock[0]+=.2
            return next(frames).copy()
        with patch('boss_defense.time.monotonic',side_effect=lambda:clock[0]):
            taps=self.defense.burst(capture,hits.append,39,lambda:False,max_frames=4)
        self.assertEqual(taps,4)
        self.assertEqual([h['side'] for h in hits],['screen','left','right','screen'])
        self.assertEqual(self.defense.last_burst['defense_taps'],2)
        self.assertEqual(self.defense.last_burst['damage_taps'],2)
        self.assertEqual(self.defense.last_side,'right')
        self.assertEqual((hits[0]['x'],hits[0]['y']),(360,330))

    def test_damage_rate_limit_never_delays_a_fresh_charging_weapon(self):
        idle=image('tests/fixtures/boss-defense/ship-fire.png')
        right=image('tests/fixtures/boss-defense/raised-right.png')
        frames=iter((idle,idle,right));clock=[100.];hits=[]
        def capture():
            clock[0]+=.05
            return next(frames).copy()
        with patch('boss_defense.time.monotonic',side_effect=lambda:clock[0]):
            self.assertEqual(self.defense.burst(capture,hits.append,39,lambda:False,max_frames=3),2)
        self.assertEqual([h['side'] for h in hits],['screen','right'])

    def test_damage_aborts_on_stop_rank_location_or_target_change(self):
        idle=image('tests/fixtures/boss-defense/ship-fire.png')
        for box in ((72,12,130,50),(286,12,434,34),(194,174,230,194)):
            frame=idle.copy();x,y,X,Y=box;frame[y:Y,x:X]=0
            frames=iter((idle,frame));hits=[];clock=[100.]
            self.defense.next_damage_at=0
            def capture():
                clock[0]+=.2
                return next(frames).copy()
            with patch('boss_defense.time.monotonic',side_effect=lambda:clock[0]):
                self.assertEqual(self.defense.burst(capture,hits.append,39,lambda:False,max_frames=2),1)
        hits=[]
        self.assertEqual(self.defense.burst(lambda:idle.copy(),hits.append,39,lambda:True),0)
        self.assertEqual(hits,[])

    def test_damage_issued_count_is_separate_from_attack_cancellation_attempts(self):
        idle=image('tests/fixtures/boss-defense/ship-fire.png')
        clock=[100.];hits=[]
        def capture():
            clock[0]+=.2
            return idle.copy()
        self.defense.scan=True
        self.defense.action=lambda _:self.defense.burst(capture,hits.append,39,lambda:False,max_taps=3)
        with patch('boss_defense.time.monotonic',side_effect=lambda:clock[0]):
            self.assertTrue(self.defense.step(idle,39))
        self.assertEqual(self.defense.taps,0)
        self.assertEqual(self.defense.damage_taps,3)
        self.assertEqual(len(hits),3)


class BossDefenseWorkerTests(unittest.TestCase):
    def test_slow_new_fallback_frame_is_refused_then_a_new_frame_is_acquired(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shutil.copytree(ROOT/'assets',root/'assets')
            config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            rule=deepcopy(next(r for r in config['rules'] if r['id']=='driftinggirl_parallel_world'))
            rule['enabled']=True;config['rules']=[rule]
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            battle=guarded_battle();captures=[];taps=[]
            class Device:
                def __init__(self,_,stop):
                    self.stop,self.io_lock=stop,threading.RLock();self.image=None
                def screenshot(self):
                    self.image=battle.copy();return self.image
                def screenshot_fast(self):
                    captures.append(1)
                    # Capture one is the outer observation; delay the fresh
                    # input guard so this still exercises stale-frame refusal.
                    if len(captures)==2:time.sleep(.14)
                    return self.screenshot()
                def command(self,*args):taps.append(args)
                def stuck_record_add(self,_):pass
            def make_world(*args,**kwargs):
                w=ParallelWorld(*args,**kwargs);w.rank=39;w.phase='play';return w
            app=Clicker(root,Device);app.digit_reader=Reader()
            try:
                with patch('app.ParallelWorld',side_effect=make_world):
                    app.start(preview=False,max_clicks=1,timeout=3);app.thread.join(5)
                self.assertFalse(app.running());self.assertIsNone(app.state['last_error'])
                self.assertEqual(app.state['stop_reason'],'click_limit')
                self.assertEqual((len(captures),len(taps),app.state['clicks']),(3,1,1))
                records=[json.loads(p.read_text(encoding='utf-8'))
                         for p in (root/'logs/boss-input').glob('input-*.json')]
                self.assertEqual(len(records),1)
                self.assertLessEqual(records[0]['capture_to_command_ms'],120)
            finally:
                app.stop()
                if app.thread:app.thread.join(2)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

    def test_damage_worker_uses_fresh_frames_actual_rank_and_the_global_click_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shutil.copytree(ROOT/'assets',root/'assets')
            config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            rule=deepcopy(next(r for r in config['rules'] if r['id']=='driftinggirl_parallel_world'))
            rule['enabled']=True;rule['training']['battle_tap']=True
            config['rules']=[rule];config['poll_interval']=.1
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            ordinary=image('tests/fixtures/training/battle.png');taps=[];captures=[]
            class Device:
                def __init__(self,_,stop):
                    self.stop,self.io_lock=stop,threading.RLock();self.image=None
                def screenshot(self):
                    self.image=ordinary.copy();return self.image
                def screenshot_fast(self):
                    captures.append(1);return self.screenshot()
                def command(self,*args):taps.append(args)
                def stuck_record_add(self,_):pass
            def make_world(*args,**kwargs):
                w=ParallelWorld(*args,**kwargs);w.rank=34;w.phase='play'
                w.defense.confirm_rank(ordinary,34);return w
            app=Clicker(root,Device);app.digit_reader=Reader()
            try:
                with patch('app.ParallelWorld',side_effect=make_world):
                    app.start(preview=False,max_clicks=3,timeout=3);app.thread.join(5)
                self.assertFalse(app.running());self.assertIsNone(app.state['last_error'])
                self.assertEqual(app.state['stop_reason'],'click_limit')
                self.assertEqual((app.state['clicks'],len(taps)),(3,3))
                self.assertTrue(all(r[3:7]==('360','330','360','330') for r in taps))
                self.assertGreaterEqual(len(captures),3)
                records=[json.loads(p.read_text(encoding='utf-8'))
                         for p in (root/'logs/boss-input').glob('input-*.json')]
                self.assertEqual(len(records),3)
                self.assertEqual(len({r['frame_id'] for r in records}),3)
                self.assertTrue(all(r['rank']==34 and r['kind']=='damage' and r['result']=='issued_only'
                                    for r in records))
                self.assertEqual(app.state['parallel']['defense']['taps'],0)
                self.assertEqual(app.state['parallel']['defense']['damage_taps'],3)
            finally:
                app.stop()
                if app.thread:app.thread.join(2)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

    def test_high_frequency_worker_uses_fast_capture_and_honors_the_tap_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            shutil.copytree(ROOT/'assets',root/'assets')
            config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            config['rules']=[deepcopy(r) for r in config['rules'] if r['id']=='driftinggirl_parallel_world']
            config['rules'][0]['enabled']=True
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            battle=guarded_battle();taps=[];fast_captures=[]
            class Device:
                def __init__(self,_,stop):
                    self.stop,self.io_lock=stop,threading.RLock();self.image=None
                def screenshot(self):
                    self.image=battle.copy();return self.image
                def screenshot_fast(self):
                    fast_captures.append(1);return self.screenshot()
                def command(self,*args):taps.append(args)
                def stuck_record_add(self,_):pass
            def make_world(*args,**kwargs):
                w=ParallelWorld(*args,**kwargs);w.rank=39;w.phase='play';return w
            app=Clicker(root,Device);app.digit_reader=Reader()
            try:
                with patch('app.ParallelWorld',side_effect=make_world):
                    app.start(preview=False,max_clicks=3);app.thread.join(5)
                self.assertFalse(app.running())
                self.assertIsNone(app.state['last_error'])
                self.assertEqual(app.state['stop_reason'],'click_limit')
                self.assertEqual(app.state['clicks'],3)
                self.assertEqual(len(taps),3)
                self.assertEqual(len(fast_captures),4)  # Outer observation plus three input guards.
                self.assertTrue(all(args[0:3]==('shell','input','swipe') and args[-1]=='80'
                                    and args[3:5]==args[5:7] for args in taps))
                records=[json.loads(p.read_text(encoding='utf-8'))
                         for p in (root/'logs/boss-input').glob('input-*.json')]
                self.assertEqual(len(records),3,list(app.events))
                self.assertEqual(len({r['frame_id'] for r in records}),3)
                self.assertTrue(all(r['result']=='issued_only' and r['hold_ms']==80 for r in records))
                for r in records:
                    before=np.asarray(Image.open(root/'logs/boss-input'/r['before_file']).convert('RGB'))
                    np.testing.assert_array_equal(before,battle)
            finally:
                app.stop()
                if app.thread:app.thread.join(2)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

    def test_native_equipment_panel_is_closed_before_defense_with_a_fresh_guard(self):
        for changed in (False,True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                shutil.copytree(ROOT/'assets',root/'assets')
                config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
                keys={'driftinggirl_parallel_world','driftinggirl_equipment_upgrade'}
                config['rules']=[deepcopy(r) for r in config['rules'] if r['id'] in keys]
                for rule in config['rules']:rule['enabled']=True
                config['poll_interval']=.1
                (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
                panel=image('tests/fixtures/boss-defense/equipment-occlusion.png')
                raised=image('tests/fixtures/boss-defense/raised-right.png')
                taps,menus,created=[],[],[]
                class Device:
                    def __init__(self,_,stop):
                        self.stop,self.io_lock=stop,threading.RLock()
                        self.index=0;self.image=None;self.panel_open=True
                    def screenshot(self):
                        self.index+=1
                        self.image=(panel if self.panel_open else raised).copy()
                        if changed and self.index>1:self.image=(self.image*.4).astype(np.uint8)
                        if self.index>=15:self.stop.set()
                        return self.image
                    def command(self,*args):taps.append(args)
                    def click(self,button):
                        menus.append(button.button);self.panel_open=False
                    def stuck_record_add(self,_):pass
                def make_world(*args,**kwargs):
                    w=ParallelWorld(*args,**kwargs);w.rank=39;w.phase='play';return w
                def make_equipment(*args,**kwargs):
                    u=EquipmentUpgrade(*args,**kwargs)
                    u.holds=2;u.passes=1;u._set('to_bottom');created.append(u)
                    return u
                app=Clicker(root,Device);app.digit_reader=Reader()
                try:
                    with patch('app.ParallelWorld',side_effect=make_world),\
                         patch('app.EquipmentUpgrade',side_effect=make_equipment):
                        app.start(preview=False);app.thread.join(5)
                    self.assertFalse(app.running())
                    self.assertIsNone(app.state['last_error'])
                    self.assertEqual(menus,[] if changed else [(355,1161,365,1171)])
                    self.assertTrue(not taps if changed else taps)
                    self.assertEqual(created[0].cycles,0)
                    self.assertEqual((created[0].holds,created[0].passes),(2,1))
                    if not changed:self.assertEqual(created[0].phase,'close')
                finally:
                    app.stop()
                    if app.thread:app.thread.join(2)
                    for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

    def test_skill_ui_deadline_is_suspended_without_extending_the_paid_boost(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            shutil.copytree(ROOT/'assets',root/'assets')
            config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            keys={'driftinggirl_parallel_world','driftinggirl_parallel_skills'}
            config['rules']=[deepcopy(r) for r in config['rules'] if r['id'] in keys]
            for rule in config['rules']:rule['enabled']=True
            config['poll_interval']=.1
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            journal=root/'parallel_skill_state.json'
            journal.write_text(json.dumps({'pending':False,'confirmed':True,'growth_until':1900000000}),encoding='utf-8')
            original=hashlib.sha256(journal.read_bytes()).hexdigest()
            clock=[100.];created=[]
            boss=image('tests/fixtures/boss-defense/ship-fire.png')
            panel=image('tests/fixtures/parallel-skills/parallel-growth-confirm.png')
            main=image('tests/fixtures/parallel-skills/main.png')
            class Device:
                def __init__(self,_,stop):
                    self.stop,self.io_lock=stop,threading.RLock()
                    self.index=0;self.image=None
                def screenshot(self):
                    self.index+=1
                    clock[0]=100. if self.index==1 else 158.+self.index
                    self.image=(boss if self.index<=2 else panel if self.index<=7 else main).copy()
                    if self.index>=12:self.stop.set()
                    return self.image
                def command(self,*args):pass
                def click(self,*args):pass
                def stuck_record_add(self,_):pass
            def make_skills(*args,**kwargs):
                u=ParallelSkills(*args,**kwargs);u._set('configure')
                u.effect=lambda _:690;u.points=lambda _:12000
                created.append(u);return u
            def make_world(*args,**kwargs):
                u=ParallelWorld(*args,**kwargs);u.rank=39;u.phase='play';return u
            app=Clicker(root,Device);app.digit_reader=Reader()
            try:
                with patch('app.time.monotonic',side_effect=lambda:clock[0]),\
                     patch('app.ParallelSkills',side_effect=make_skills),\
                     patch('app.ParallelWorld',side_effect=make_world):
                    app.start(preview=False);app.thread.join(5)
                self.assertFalse(app.running())
                self.assertIsNone(app.state['last_error'])
                self.assertEqual(created[0].phase,'idle')
                self.assertEqual(created[0].cycles,1)
                self.assertEqual(hashlib.sha256(journal.read_bytes()).hexdigest(),original)
            finally:
                app.stop()
                if app.thread:app.thread.join(2)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

    def test_worker_preempts_equipment_and_rechecks_before_every_tap(self):
        for changed in (False,True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                shutil.copytree(ROOT/'assets',root/'assets')
                config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
                keys = {'driftinggirl_parallel_world','driftinggirl_equipment_upgrade'}
                rules = [deepcopy(r) for r in config['rules'] if r['id'] in keys]
                for rule in rules:rule['enabled'] = True
                config.update(rules=rules,poll_interval=.1)
                (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
                taps,menus = [],[]
                battle = guarded_battle()
                class Device:
                    def __init__(self,_,stop):
                        self.stop,self.io_lock = stop,threading.RLock()
                        self.image=None;self.index=0
                    def screenshot(self):
                        self.index += 1
                        self.image=battle.copy()
                        if changed and self.index>1:self.image[245:690,410:614]=0
                        if self.index >= 8:self.stop.set()
                        return self.image
                    def command(self,*args):taps.append(args)
                    def click(self,button):menus.append(button)
                    def stuck_record_add(self,_):pass
                def world(*args,**kwargs):
                    w=ParallelWorld(*args,**kwargs);w.rank=39;w.phase='play'
                    return w
                app=Clicker(root,Device);app.digit_reader=Reader()
                try:
                    with patch('app.ParallelWorld',side_effect=world):
                        app.start(preview=False);app.thread.join(5)
                    self.assertFalse(app.running())
                    self.assertIsNone(app.state['last_error'])
                    self.assertEqual(menus,[])
                    # 武器变暗后不得再点旧坐标;此时"点击攻击"在打 40 BOSS 时是生效的
                    # (见 drift/tasks/final_boss_attack.py),所以允许点屏幕空白补伤害。
                    if changed:
                        self.assertTrue(all(t[3:7]==('360','330','360','330') for t in taps),
                                        f'变暗后只应点屏幕空白,实际 {taps}')
                    else:
                        self.assertTrue(taps)
                finally:
                    app.stop()
                    if app.thread:app.thread.join(2)
                    for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)


if __name__ == '__main__':unittest.main()
