from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import cv2
import numpy as np

from drift.perception.digit_ocr import DigitReader
from drift.perception.restaurant import RestaurantVisual
from drift.tasks.restaurant_upgrade import RestaurantUpgrade, unpack
from drift.tasks.schedule import TaskSchedule
from drift.tasks.daily import BEIJING
from tools.configure_restaurant_upgrade import add, RULE

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'diagnostics/2026-10-25/restaurant-upgrade'

class RestaurantFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.visual=RestaurantVisual(ROOT,DigitReader(ROOT))

    def frame(self,name):
        image=cv2.imread(str(EVIDENCE/name))
        if image is None:self.skipTest('Local real-frame fixture unavailable')
        return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    def test_crafting_is_never_enhancement(self):
        f=self.frame('after-9ec8fbcdeb074423b6444a7db7b1ca66.png')
        self.assertFalse(self.visual.detail(f));self.assertFalse(self.visual.atlas(f))
        self.assertIsNone(self.visual.detail_values(f))
        self.assertEqual(self.visual.cards(f),[])

    def test_four_counters_before_and_after_real_upgrade(self):
        a=self.visual.detail_values(self.frame('after-bd70f3da006d42249464564edacda13e.png'))
        b=self.visual.detail_values(self.frame('after-2003feb758f84178b1c247333c99d29d.png'))
        self.assertEqual(a['identity'],b['identity'])
        self.assertEqual((a['level'],b['level']),(1,2))
        self.assertEqual(a['counters'],[(12,12)]*4)
        self.assertEqual(b['counters'],[(0,12)]*4)

    def test_all_four_available_icons_recognized(self):
        f=self.frame('after-2003feb758f84178b1c247333c99d29d.png');v=self.visual.detail_values(f)
        self.assertEqual([bool(self.visual.ingredient_ready(f,i,v)) for i in range(4)],[True]*4)

    def test_partial_counter_cannot_confirm_upgrade(self):
        f=self.frame('after-0b03b7398f48447a8667c381e7aa0f07.png');v=self.visual.detail_values(f)
        self.assertEqual(v['counters'][-1],(10,12));self.assertFalse(self.visual.upgrade_ready(f,v))
        self.assertFalse(self.visual.ingredient_ready(f,0,v))
        self.assertTrue(self.visual.ingredient_ready(f,3,v))

    def test_full_requires_bright_green_confirmation(self):
        f=self.frame('after-bd70f3da006d42249464564edacda13e.png');v=self.visual.detail_values(f)
        self.assertTrue(self.visual.upgrade_ready(f,v))
        f=f.copy();f[735:773,574:637]=0
        self.assertFalse(self.visual.upgrade_ready(f,v))

    def test_absent_available_arrow_prevents_material_hold(self):
        f=self.frame('after-2003feb758f84178b1c247333c99d29d.png');v=self.visual.detail_values(f)
        f=f.copy();f[596:619,460:501]=0
        self.assertFalse(self.visual.ingredient_ready(f,3,v))

    def test_max_level_and_partial_view_cards(self):
        f=self.frame('after-e86acac3e59b497da51a8bc936331436.png')
        cards=self.visual.cards(f)
        self.assertEqual([c['level'] for c in cards],[20,20,20,1])
        self.assertFalse(self.visual.art_matches(cards[0]['art'],cards[1]['art']))

    def test_fully_visible_final_card_at_right_edge_is_not_skipped(self):
        f=self.frame('after-e86acac3e59b497da51a8bc936331436.png')
        shifted=f.copy();shifted[330:490,107:151]=[225,205,160]
        shifted[330:490,151:623]=f[330:490,107:579]
        self.assertEqual([c['level'] for c in self.visual.cards(shifted)],[20,20,20,1])

    def test_beige_level_two_card_is_not_hidden_by_green_max_level_template(self):
        f=self.frame('controller-verification/frame-4e8bd945e21849ddbf4836e1f08932f5.png')
        self.assertEqual([c['level'] for c in self.visual.cards(f)],[20,20,20,2])

    def test_eight_material_counter_with_one_low_confidence_threshold(self):
        f=self.frame('controller-fixed-verification/frame-bdb5795557d34ac2abd6fbf403b23910.png')
        v=self.visual.detail_values(f)
        self.assertEqual(v['level'],2);self.assertEqual(v['counters'],[(0,8)]*4)
        self.assertEqual([bool(self.visual.ingredient_ready(f,i,v)) for i in range(4)],[True]*4)

    def test_bobbing_arrow_not_rectangle_or_ingredient_glint(self):
        f=self.frame('controller-fixed-verification/frame-bdb5795557d34ac2abd6fbf403b23910.png')
        f=f.copy();f[580:625,122:162]=0
        f[595:606,129:156]=[210,170,60]
        self.assertFalse(self.visual.available_arrow(f,0))

    def test_conflicting_complete_counter_decodes_are_rejected(self):
        f=self.frame('controller-fixed-verification/frame-bdb5795557d34ac2abd6fbf403b23910.png')
        reader=Mock();v=RestaurantVisual(ROOT,reader)
        reader.read_many.return_value=[SimpleNamespace(value=2,confidence=.99)]*2+[
            SimpleNamespace(text=t,confidence=.99) for t in ['0/8','1/8','0/8']+['0/8']*9]
        self.assertIsNone(v.detail_values(f))

    def test_dark_overlay_and_unknown_frame_refused(self):
        f=self.frame('after-2003feb758f84178b1c247333c99d29d.png')
        self.assertFalse(self.visual.detail((f*.4).astype('uint8')))
        self.assertIsNone(self.visual.detail_values(np.zeros_like(f)))

    def test_fixed_restaurant_arrival_survives_bouncing_entry(self):
        folder=EVIDENCE/'formal-run-0828/idle-probe'
        if not (folder/'history.jsonl').exists():self.skipTest('Real animation sequence unavailable')
        with tempfile.TemporaryDirectory() as d:
            task=RestaurantUpgrade(ROOT,None,Mock(),Mock(),'emulator-5554',state_path=Path(d)/'state.json')
            confirmed=False;missing_icon=False
            for line in (folder/'history.jsonl').read_text(encoding='utf-8').splitlines():
                row=json.loads(line)
                f=cv2.cvtColor(cv2.imread(str(folder/('native-'+row['frame']+'.png'))),cv2.COLOR_BGR2RGB)
                origin=task.origin_screen(f)
                self.assertTrue(origin['at_restaurant'])
                missing_icon=missing_icon or task.visual.point(f,'entry') is None
                task.clock=lambda now=row['at']:now
                confirmed=task._confirmed(tuple((k,str(v)) for k,v in origin.items()),f,3) or confirmed
            self.assertTrue(missing_icon);self.assertTrue(confirmed)
            task.action.assert_not_called()

class RestaurantStartupTests(unittest.TestCase):
    def controller(self,d,*,running=True,moving=True):
        c=RestaurantUpgrade(ROOT,None,Mock(),Mock(),'x',
            state_path=Path(d)/'state.json',clock=lambda:100)
        c.root=Path(d)  # Keep state archives inside the temporary test directory.
        c.view=Mock();c.view._parallel.return_value=True;c.view._battle.return_value=False
        c.view.screen.return_value=dict(completion=False,rank=20)
        c.view.training.boss_mode.return_value=False;c.view._off_icon.return_value=True
        c.fishing=Mock();c.fishing._main_visible.return_value=True
        c.sailing=Mock();c.sailing.running.return_value=running;c.sailing.menu.return_value=False
        c.sailing.selected.return_value=True
        c.sailing.sale_selected.return_value=True;c.sailing.upgrade_selected.return_value=True
        c.bag=Mock();c.bag.moving.return_value=moving;c.bag._scan_map.return_value=False
        c.visual=Mock();c.visual.find.return_value=None
        c.visual.detail.return_value=False;c.visual.panel.return_value=False
        c.guild=Mock();c.guild.point.return_value=None
        def action(a):
            # Mirror the runtime's fresh validation, reservation and receipt only.
            if not c.validate(a,c.test_frame.copy()) or not c.reserve(a):return False
            c.input_issued(a,dict(input_id='test-input'))
            return True
        c.action=Mock(side_effect=action)
        return c

    def frame(self,place=0,world=0):
        frame=np.zeros((1280,720,3),dtype=np.uint8)
        rng=np.random.default_rng(100+world)
        for x,y,X,Y in ((71,9,136,59),(13,45,68,72)):
            frame[y:Y,x:X]=rng.integers(0,256,(Y-y,X-x,3),dtype=np.uint8)
        frame[12:36,286:434]=np.random.default_rng(place).integers(0,256,(24,148,3),dtype=np.uint8)
        return frame

    def step(self,c,now,frame=None,*,due=True):
        c.clock=lambda:now;c.test_frame=self.frame() if frame is None else frame
        return c.step(c.test_frame,due=due)

    def start(self,c):
        for now in (100,101,103):self.step(c,now)
        self.assertTrue(c.active);self.assertEqual(c.phase,'stop')

    def finish_stop_menu(self,c,now):
        c.sailing.menu.return_value=True;c.sailing.running.return_value=None
        c.fishing._main_visible.return_value=False;c.view._off_icon.return_value=False
        self.step(c,now,due=False);self.step(c,now+1,due=False)
        self.assertEqual([a.args[0]['kind'] for a in c.action.call_args_list][-2:],
                         ['stop-toggle','stop-close'])
        c.sailing.menu.return_value=False;c.fishing._main_visible.return_value=True
        c.view._off_icon.return_value=True

    def test_start_screen_accepts_rank_39_stage_9(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d,running=False,moving=False)
            c.view.screen.return_value=dict(completion=False,rank=39,stage=9)
            frame=self.frame()
            startup=c.start_screen(frame)
            self.assertIsNotNone(startup)
            self.assertIs(startup['parallel'],True);self.assertIs(startup['route'],False)
            self.assertIsNotNone(c.origin_screen(frame))
            self.start(c)
            c.action.assert_not_called()

    def test_start_screen_accepts_rank_40(self):
        # RANK never blocks the cookbook: a max-RANK world still starts it.
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d,running=False,moving=False)
            c.view.screen.return_value=dict(completion=False,rank=40,stage=9)
            frame=self.frame()
            startup=c.start_screen(frame)
            self.assertIsNotNone(startup)
            self.assertEqual(sorted(startup),['header','parallel','route'])
            self.assertIsNotNone(c.origin_screen(frame))
            self.start(c)
            c.action.assert_not_called()

    def test_start_screen_rejects_completion_at_rank_39(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d,running=False,moving=False)
            c.view.screen.return_value=dict(completion=True,rank=39,stage=9)
            self.assertIsNone(c.start_screen(self.frame()))

    def test_start_screen_rejects_boss_mode_at_rank_39(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d,running=False,moving=False)
            c.view.screen.return_value=dict(completion=False,rank=39,stage=9)
            c.view.training.boss_mode.return_value=True
            self.assertIsNone(c.start_screen(self.frame()))

    def test_moving_cruise_enters_stop_after_fresh_confirmation(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d)
            for now,expected in ((100,False),(101,False),(103,True)):
                self.assertEqual(self.step(c,now,self.frame(place=now)),expected)
            self.assertTrue(c.active);self.assertEqual(c.phase,'stop')
            self.assertNotIn('place',c.journal.get('origin',{}))
            c.action.assert_not_called()
            self.assertTrue(self.step(c,104,due=False))
            self.assertEqual(c.journal['attempts'],{'stop':1})
            self.assertEqual(c.journal['last_input']['state'],'issued_only')
            self.assertEqual(c.phase,'stop')

    def test_origin_is_captured_from_stable_stopped_frames(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d,moving=False);self.start(c)
            c.sailing.running.return_value=False
            stopped=self.frame(place=7)
            for now in (104,105,107):self.step(c,now,stopped.copy(),due=False)
            self.assertEqual(c.phase,'settings_open')
            origin=c.journal['origin']
            np.testing.assert_array_equal(unpack(origin['place']),stopped[12:36,286:434])
            self.assertIs(origin['route'],False)
            self.assertFalse(origin['at_restaurant'])
            saved=json.loads(c.path.read_text(encoding='utf-8'))
            self.assertEqual(saved['origin'],origin)

    def test_stop_receipts_do_not_confirm_arrival_or_renew_input_budget(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);self.start(c)
            deadline=c.journal['phase_deadline']
            for now in (104,105,106,107,108):self.step(c,now,due=False)
            self.assertEqual(c.phase,'stop');self.assertNotIn('origin',c.journal)
            self.assertEqual(c.journal['attempts'],{'stop':1})
            self.assertEqual(c.journal['phase_deadline'],deadline)
            self.assertEqual(c.journal['submitted'],0);self.assertEqual(c.journal['upgraded'],0)
            self.finish_stop_menu(c,109)
            c.sailing.running.return_value=False
            for now in (111,112,113):self.step(c,now,due=False)
            self.assertEqual(c.phase,'stop');self.assertNotIn('origin',c.journal)
            c.bag.moving.return_value=False
            for now,place in ((114,1),(115,1),(117,2),(118,2)):
                self.step(c,now,self.frame(place=place),due=False)
                self.assertEqual(c.phase,'stop')
            self.step(c,120,self.frame(place=2),due=False)
            self.assertEqual(c.phase,'settings_open')
            self.assertEqual(c.journal['attempts'],{'stop':1,'stop-toggle':1,'stop-close':1})
            self.assertEqual(c.journal['origin_deadline'],123)

    def test_cached_stopped_frame_cannot_confirm_origin(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);self.start(c)
            c.sailing.running.return_value=False;c.bag.moving.return_value=False
            stopped=self.frame(place=7)
            for now in (104,105,107):self.step(c,now,stopped,due=False)
            self.assertEqual(c.phase,'stop');self.assertNotIn('origin',c.journal)
            for now in (108,110):self.step(c,now,stopped.copy(),due=False)
            self.assertEqual(c.phase,'settings_open')

    def test_unknown_route_interrupts_stopped_confirmation(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);self.start(c);c.bag.moving.return_value=False
            c.sailing.running.return_value=False
            for now in (104,105):self.step(c,now,due=False)
            c.sailing.running.return_value=None;self.step(c,106,due=False)
            c.sailing.running.return_value=False
            for now in (107,108):
                self.step(c,now,due=False);self.assertEqual(c.phase,'stop')
            self.step(c,110,due=False);self.assertEqual(c.phase,'settings_open')

    def test_unknown_route_starts_then_stops_after_fresh_evidence(self):
        # 2026-10-29 线上：航巡读数未知（主界面某一路守卫没读出来）被当成
        # 「船在港口」，任务永远不启动。仍可接管，但本次修正后仅明确
        # running=True 才允许开上下文菜单，未知读数不能授权切换航巡。
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d,running=None)
            for now,expected in ((100,False),(101,False),(103,True)):
                self.assertEqual(self.step(c,now,self.frame(place=now)),expected)
            self.assertTrue(c.active);self.assertEqual(c.phase,'stop')
            self.assertNotIn('origin',c.journal)  # 航行中不伪造原画面
            c.action.assert_not_called()          # 未知航巡绝不当场盲点
            for now in (104,105):
                self.step(c,now,due=False);self.assertEqual(c.phase,'stop')
            c.action.assert_not_called()
            self.step(c,107,due=False)
            c.action.assert_not_called()
            c.sailing.running.return_value=True;self.step(c,108,due=False)
            self.assertEqual([a.args[0]['kind'] for a in c.action.call_args_list],['stop'])
            self.assertEqual(c.journal['attempts'],{'stop':1})

    def test_unknown_route_after_stop_captures_origin_from_stable_screen(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d,running=None)
            self.start(c)
            for now in (104,105):self.step(c,now,due=False)
            c.action.assert_not_called()
            c.sailing.running.return_value=True;self.step(c,107,due=False)
            self.assertEqual(c.journal['attempts'],{'stop':1})
            self.finish_stop_menu(c,108)
            c.bag.moving.return_value=False  # 停航巡后顶部「移动中」消失
            stopped=self.frame(place=7)
            for now in (110,111,113):self.step(c,now,stopped.copy(),due=False)
            self.assertEqual(c.phase,'settings_open')
            origin=c.journal['origin']
            self.assertIsNone(origin['route'])
            np.testing.assert_array_equal(unpack(origin['place']),stopped[12:36,286:434])
            self.assertIs(c.journal['resume_route'],True)  # 我们点了停航巡，必须恢复

    def test_unknown_route_without_inactive_icon_never_confirms_origin(self):
        # route is None 只放宽「是否可接管」，不放宽「画面确实停着」：
        # 没有 AUTO 标签/非激活面板/无激活箭头时，取到原画面也不认。
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d,running=None)
            self.start(c)
            c.sailing.running.return_value=True;self.step(c,104,due=False)
            self.finish_stop_menu(c,105)
            c.view._off_icon.return_value=False
            c.bag.moving.return_value=False
            for now in (108,109,111,112,113,115):
                self.step(c,now,due=False);self.assertEqual(c.phase,'stop')
            self.assertNotIn('origin',c.journal)
            self.assertEqual(c.journal['attempts'],{'stop':1,'stop-toggle':1,'stop-close':1})

    def test_stalled_origin_confirmation_abandons_without_unknown_restore_input(self):
        # 线上事故：点了停航巡却一直取不到 origin，于是每轮空耗在 stop 上。
        # 整段 20 秒拿不到原画面就归档退出，未知画面不再转恢复后盲开菜单。
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);self.start(c)
            self.step(c,104,due=False);self.finish_stop_menu(c,105)
            self.assertEqual(c.journal['attempts'],{'stop':1,'stop-toggle':1,'stop-close':1})
            tried=len(c.action.call_args_list)
            c.sailing.running.return_value=None   # 停航巡后读数未知
            c.bag.moving.return_value=True         # 「移动中」一直不消失
            c.fishing._main_visible.return_value=False
            self.step(c,109,due=False)
            self.assertEqual(c.phase,'stop')
            self.assertEqual(c.journal['origin_deadline'],123)
            for now in (110,120,122):self.step(c,now,due=False)
            self.assertEqual(c.phase,'stop')
            raw=c.path.read_bytes();self.step(c,123,due=False)
            self.assertFalse(c.active);self.assertIs(c.done,False)
            self.assertEqual(c.journal['retirement']['reason'],'停航后未能确认原画面，停止本轮输入')
            self.assertEqual(Path(c.journal['retirement']['archive']).read_bytes(),raw)
            self.assertEqual(c.journal['attempts'],{'stop':1,'stop-toggle':1,'stop-close':1})
            self.step(c,190,due=False)
            self.assertEqual(len(c.action.call_args_list),tried)  # 超时退出后不再发输入

    def test_unknown_route_interrupt_restores_cruise_and_confirms_fishing(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d,running=None)
            self.start(c)
            for now in (104,105):self.step(c,now,due=False)
            c.sailing.running.return_value=True;self.step(c,106,due=False)
            self.finish_stop_menu(c,107)
            c.bag.moving.return_value=False
            stopped=self.frame(place=7)
            for now in (110,111,113):self.step(c,now,stopped.copy(),due=False)
            self.assertEqual(c.phase,'settings_open')
            self.assertIs(c.journal['resume_route'],True)
            c._set('restore')
            c.sailing.running.return_value=None  # 菜单关掉瞬间读数未知
            self.step(c,150,stopped.copy(),due=False)
            self.assertEqual(c.phase,'restore_menu')
            self.assertEqual(c.action.call_args.args[0]['kind'],'restore-open')
            c.sailing.menu.return_value=True
            c.sailing.running.return_value=False
            c.journal['settings']={'sale':True};c.settings=Mock(return_value={'sale':True})
            for now in (151,152,154):self.step(c,now,stopped.copy(),due=False)
            self.assertEqual(c.phase,'restore_verify')
            c.sailing.menu.return_value=False
            c.sailing.running.return_value=True
            c.fishing.blue_fill.return_value=True
            for now in (155,156,158):self.step(c,now,stopped.copy(),due=False)
            self.assertIs(c.done,True);self.assertEqual(c.phase,'idle');self.assertFalse(c.active)
            self.assertEqual([a.args[0]['kind'] for a in c.action.call_args_list],
                             ['stop','stop-toggle','stop-close','restore-open','restore-start'])

    def test_stopped_route_with_moving_banner_waits_without_pending_or_input(self):
        # 船已停（route False）但顶部还有「移动中」：原画面不可信，此时
        # 不写 pending、不发任何输入，等画面稳定后下一帧再评估。
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d,running=False,moving=True)
            for now in (100,101,103,105,107):
                self.assertFalse(self.step(c,now))
                self.assertFalse(c.active);self.assertEqual(c.phase,'idle')
            c.action.assert_not_called()
            self.assertFalse(c.path.exists())

    def test_unknown_route_still_rejects_battle_cards_and_unknown_rank(self):
        cases={
            'battle':lambda c:setattr(c.view._battle,'return_value',True),
            'completion':lambda c:setattr(c.view.screen,'return_value',dict(completion=True,rank=20)),
            'boss':lambda c:setattr(c.view.training.boss_mode,'return_value',True),
            'unknown_rank':lambda c:setattr(c.view.screen,'return_value',None),
            'not_main':lambda c:setattr(c.fishing._main_visible,'return_value',False),
        }
        for name,change in cases.items():
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
                c=self.controller(d,running=None);change(c)
                for now in (100,101,103,105):self.assertFalse(self.step(c,now))
                self.assertFalse(c.active);c.action.assert_not_called()

    def test_start_still_requires_due_enabled_and_safe_known_scene(self):
        cases={
            'not_due':lambda c:None,
            'disabled':lambda c:setattr(c,'enabled',lambda:False),
            'not_main':lambda c:setattr(c.fishing._main_visible,'return_value',False),
            'battle':lambda c:setattr(c.view._battle,'return_value',True),
            # 2026-10-29: an unknown route reading no longer blocks the start;
            # see test_unknown_route_starts_then_stops_after_fresh_evidence.
            'unknown_rank':lambda c:setattr(c.view.screen,'return_value',None),
            # RANK is deliberately absent here: it never blocks the cookbook.
            'completion':lambda c:setattr(c.view.screen,'return_value',dict(completion=True,rank=20)),
            'boss':lambda c:setattr(c.view.training.boss_mode,'return_value',True),
            'manual_travel':lambda c:setattr(c.sailing.running,'return_value',False),
        }
        for name,change in cases.items():
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
                c=self.controller(d);change(c)
                for now in (100,101,103,105):
                    self.assertFalse(self.step(c,now,due=name!='not_due'))
                self.assertFalse(c.active);c.action.assert_not_called()

    def test_stop_revalidates_fresh_scene_and_rejects_unrelated_actions(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);self.start(c)
            stop=dict(kind='stop',phase='stop',x=233,y=45)
            self.assertTrue(c.validate(stop,self.frame()))
            self.assertFalse(c.validate(stop,self.frame(world=1)))
            c.view._battle.return_value=True
            self.assertFalse(c.validate(stop,self.frame()))
            c.view._battle.return_value=False
            c.sailing.running.return_value=False;c.bag.moving.return_value=False
            self.assertFalse(c.validate(stop,self.frame()))
            self.assertFalse(c.validate(dict(kind='settings-open',phase='stop',x=233,y=45),self.frame()))
            self.assertFalse(c.map_ready(self.frame()))
            for now in (104,105,107):self.step(c,now,self.frame(world=1),due=False)
            self.assertEqual(c.phase,'stop');self.assertNotIn('origin',c.journal)
            c.action.assert_not_called()

    def test_stopped_route_keeps_original_navigation_and_restaurant_entry(self):
        for at_restaurant in (False,True):
            with self.subTest(at_restaurant=at_restaurant),tempfile.TemporaryDirectory() as d:
                c=self.controller(d,running=False,moving=False)
                c.visual.find.return_value=(500,30) if at_restaurant else None
                self.start(c)
                for now in ((104,) if at_restaurant else (104,105,107)):
                    self.step(c,now,due=False)
                self.assertEqual(c.phase,'open' if at_restaurant else 'map_open')
                self.assertIs(c.journal['origin']['route'],False)
                self.assertFalse(c.route_was_running());c.action.assert_not_called()

    def test_restart_during_stop_preserves_budget_then_captures_origin(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);self.start(c);self.step(c,104,due=False)
            self.finish_stop_menu(c,105)
            saved=deepcopy(c.journal)
            c=self.controller(d,running=False,moving=False)
            self.assertEqual(c.journal,saved);self.assertEqual(c.phase,'stop')
            for now in (107,108,110):self.step(c,now,self.frame(place=7),due=False)
            self.assertEqual(c.phase,'settings_open')
            self.assertEqual(c.journal['attempts'],{'stop':1,'stop-toggle':1,'stop-close':1})
            self.assertEqual(c.journal['deadline'],saved['deadline'])
            self.assertIs(c.journal['origin']['route'],False);self.assertTrue(c.route_was_running())

    def test_legacy_stop_journal_preserves_original_cruise_restore_intent(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d,moving=False);self.start(c)
            c.journal['origin']=c.origin_screen(self.frame())
            c.journal.pop('startup');c.journal.pop('resume_route');c._save()
            c=self.controller(d,running=False,moving=False)
            for now in (104,105,107):self.step(c,now,self.frame(place=7),due=False)
            self.assertEqual(c.phase,'settings_open')
            self.assertIs(c.journal['origin']['route'],False);self.assertTrue(c.route_was_running())
            c._set('close');c.visual.find.return_value=(500,30)
            for now in (108,109,111):self.step(c,now,self.frame(place=9),due=False)
            self.assertEqual(c.phase,'restore')

    def test_expired_startup_retires_journal_without_any_input(self):
        # 2026-10-28 事故：过期残留 journal 会续开 cleanup，_begin_cleanup 立刻抛
        # 「安全返回超时」，而 step_restaurant 没有兜底，于是每次启动 8 秒内打死 runner。
        # 新契约：过期记录在启动时归档为 retired，不再续开 cleanup、不点任何东西。
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);self.start(c);self.step(c,104,due=False)
            deadline=c.journal['deadline'];phase_deadline=c.journal['phase_deadline']
            submitted=c.journal['submitted']
            c=self.controller(d,running=False,moving=False);c.clock=lambda:149
            c.retire_expired_on_start()
            self.assertEqual(c.phase,'idle');self.assertFalse(c.active);self.assertIs(c.done,False)
            self.assertEqual(c.journal['last_failure']['reason'],'启动时发现上次执行已过期')
            self.assertEqual(c.journal['last_failure']['phase_deadline'],phase_deadline)
            self.assertEqual(c.journal['retirement']['previous_phase'],'stop')
            self.assertEqual(c.journal['deadline'],deadline)
            self.assertEqual(c.journal['attempts'],{'stop':1})
            self.assertEqual(c.journal['submitted'],submitted)
            c.action.assert_not_called()
            for now in (149,150,152):self.step(c,now,due=False)
            self.assertEqual(c.phase,'idle')
            self.assertEqual(c.journal['submitted'],submitted)
            c.action.assert_not_called()


class RestaurantEntryMenuTests(unittest.TestCase):
    """720x1280 stub frames; exercise the runtime input gate without any device."""
    controller=RestaurantStartupTests.controller
    frame=RestaurantStartupTests.frame
    step=RestaurantStartupTests.step
    start=RestaurantStartupTests.start

    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        self.c=self.controller(directory.name)
        self.c.view._parallel.return_value=False
        self.c.view._observatory.return_value=False
        self.issued=[]
        original=self.c.action.side_effect
        def issue(a):
            if not self.c.allowed() or not original(a):return False
            self.issued.append(a.copy());return True
        self.c.action.side_effect=issue

    def points(self):return [(a['x'],a['y']) for a in self.issued]

    def menu(self):
        self.c.sailing.menu.return_value=True
        self.c.sailing.running.return_value=None
        self.c.fishing._main_visible.return_value=False
        self.c.view._off_icon.return_value=False

    def stopped(self):
        self.c.sailing.menu.return_value=False
        self.c.sailing.running.return_value=False
        self.c.fishing._main_visible.return_value=True
        self.c.view._off_icon.return_value=True
        self.c.bag.moving.return_value=False

    def pending(self,phase='stop',**extra):
        # Seed a pending journal so timeout tests do not depend on the entry fix.
        c=self.c;c.phase=phase;c.view._observatory.return_value=True
        c.journal.update(pending=True,phase=phase,deadline=1000,phase_deadline=200,
            startup=dict(parallel=False,header=c.header(self.frame()),route=True),
            resume_route=True,attempts={},visited=[],categories=[],skipped=[],
            submitted=0,upgraded=0,origin_deadline=120,**extra)
        c._save();return c.path.read_bytes()

    def assert_failed_archive(self,raw):
        c=self.c
        self.assertFalse(c.active);self.assertIs(c.done,False)
        self.assertFalse(c.journal['success'])
        self.assertEqual(Path(c.journal['retirement']['archive']).read_bytes(),raw)
        self.assertIn('last_failure',c.journal)

    def test_entry_from_open_sea_accepts_running_main(self):
        c=self.c
        self.assertFalse(c.at_restaurant(self.frame()))
        self.assertIsNone(c.guild.point(self.frame(),'g-icon'))
        self.assertFalse(c.view._observatory(self.frame()))
        startup=c.start_screen(self.frame())
        self.assertIsNotNone(startup)
        self.assertIs(startup['parallel'],False);self.assertIs(startup['route'],True)

    def test_parallel_completion_and_boss_guards_remain(self):
        c=self.c;c.view._parallel.return_value=True
        for completion,boss in ((True,False),(False,True),(True,True)):
            with self.subTest(completion=completion,boss=boss):
                c.view.screen.return_value=dict(completion=completion,rank=39)
                c.view.training.boss_mode.return_value=boss
                self.assertIsNone(c.start_screen(self.frame()))
        c.action.assert_not_called()

    def test_stop_orders_open_toggle_close_before_origin(self):
        c=self.c;self.pending()
        self.step(c,101,due=False)
        self.assertEqual(self.points(),[(233,45)])
        self.menu();self.step(c,102,due=False)
        self.assertEqual(self.points(),[(233,45),(360,880)])
        self.assertNotIn('origin',c.journal)
        self.step(c,103,due=False)
        self.assertEqual(self.points(),[(233,45),(360,880),(600,363)])
        self.assertNotIn('origin',c.journal)
        self.stopped()
        for now in (104,105):
            self.step(c,now,due=False);self.assertNotIn('origin',c.journal)
        self.step(c,107,due=False)
        self.assertIn('origin',c.journal);self.assertEqual(c.phase,'settings_open')
        self.assertTrue(c.journal['resume_route'])
        self.assertEqual(len(self.points()),3)

    def test_already_stopped_skips_menu_and_toggle(self):
        c=self.c;c.view._observatory.return_value=True;self.stopped();self.start(c)
        for now in (104,105,107):self.step(c,now,due=False)
        self.assertEqual(c.phase,'map_open');self.assertFalse(c.route_was_running())
        c.action.assert_not_called();self.assertEqual(self.points(),[])

    def test_stop_expiry_closes_menu_once_and_still_abandons(self):
        c=self.c;raw=self.pending();self.menu()
        self.step(c,120,due=False)
        self.assertEqual(self.points(),[(600,363)])
        self.assertNotIn('origin',c.journal)
        self.stopped();self.step(c,121,due=False)
        self.assert_failed_archive(raw)
        self.assertEqual(self.points(),[(600,363)])

    def test_missing_menu_abandons_at_five_seconds_without_more_input(self):
        c=self.c;c.view._observatory.return_value=True;self.start(c);deadline=109
        self.step(c,104,due=False)
        self.assertEqual(self.points(),[(233,45)])
        for now in (105,107,deadline-1):self.step(c,now,due=False)
        raw=c.path.read_bytes()
        self.step(c,deadline,due=False)
        self.assert_failed_archive(raw)
        self.assertEqual(self.points(),[(233,45)])
        self.assertIn('未打开航巡菜单',c.journal['retirement']['reason'])

    def test_unreadable_or_other_world_still_expires_without_input(self):
        for other_world in (False,True):
            with self.subTest(other_world=other_world):
                c=self.c;self.pending();self.step(c,101,due=False)
                c.fishing._main_visible.return_value=False
                c.sailing.running.return_value=None
                f=self.frame(world=1 if other_world else 0)
                raw=c.path.read_bytes();self.step(c,120,f,due=False)
                self.assert_failed_archive(raw)
                self.assertEqual(self.points(),[(233,45)])
                self.issued.clear();c.action.reset_mock()
                c.fishing._main_visible.return_value=True
                c.sailing.running.return_value=True

    def test_unknown_route_never_opens_or_toggles(self):
        c=self.c;raw=self.pending();c.sailing.running.return_value=None
        c.view._off_icon.return_value=False
        for now in (101,104,108,119,120):self.step(c,now,due=False)
        self.assert_failed_archive(raw);c.action.assert_not_called()

    def test_restore_expiry_closes_even_with_expired_input_deadlines(self):
        c=self.c;self.pending('restore_menu',restore_deadline=110)
        c.journal.update(deadline=110,phase_deadline=110);c._save()
        raw=c.path.read_bytes();self.menu();self.step(c,110,due=False)
        self.assertEqual(self.points(),[(600,363)])
        self.stopped();self.step(c,111,due=False)
        self.assert_failed_archive(raw)
        self.assertEqual(c.journal['retirement']['reason'],'航巡恢复确认超时')

    def test_abandon_closes_without_readable_settings_and_preserves_failure(self):
        c=self.c;self.pending('restore_menu')
        failure=dict(reason='原失败',phase='detail',at=90)
        c.journal['failure']=failure.copy();c._save();raw=c.path.read_bytes()
        self.menu();c.settings=Mock(return_value=None)
        c.test_frame=c._observed_frame=self.frame()
        c.abandon('再次失败')
        self.assertEqual(self.points(),[(600,363)])
        self.stopped();self.step(c,101,due=False)
        self.assert_failed_archive(raw)
        self.assertEqual(c.journal['last_failure'],failure)

    def test_menu_close_confirmation_is_bounded_and_never_retries(self):
        c=self.c;raw=self.pending();self.menu()
        self.step(c,120,due=False)
        self.assertEqual(self.points(),[(600,363)])
        for now in (120.5,121,122,123):self.step(c,now,due=False)
        self.assert_failed_archive(raw)
        for now in (124,125,130):self.step(c,now,due=False)
        self.assertEqual(self.points(),[(600,363)])

    def test_abort_close_revalidates_menu_before_input(self):
        c=self.c;raw=self.pending();self.menu()
        # The menu is covered between step() and runtime's fresh-frame gate.
        original=c.action.side_effect
        def covered(a):
            c.sailing.menu.return_value=False
            return original(a)
        c.action.side_effect=covered
        self.step(c,120,due=False)
        self.assert_failed_archive(raw);self.assertEqual(self.points(),[])
        self.assertEqual(c.action.call_count,1)

    def test_stop_toggle_revalidates_phase_menu_and_coordinates(self):
        c=self.c;self.pending();self.step(c,101,due=False);self.menu()
        toggle=dict(kind='stop-toggle',phase='stop',x=360,y=880)
        self.assertTrue(c.validate(toggle,self.frame()))
        self.assertFalse(c.validate(dict(toggle,x=359),self.frame()))
        self.assertFalse(c.validate(dict(toggle,phase='restore_menu'),self.frame()))
        self.assertFalse(c.validate(toggle,self.frame(world=1)))
        c.sailing.menu.return_value=False
        self.assertFalse(c.validate(toggle,self.frame()))
        c.sailing.menu.return_value=True;c.sailing.running.return_value=False
        self.assertFalse(c.validate(toggle,self.frame()))

    def test_archive_failure_prevents_close_and_preserves_original_record(self):
        c=self.c;raw=self.pending();self.menu();c.test_frame=c._observed_frame=self.frame()
        with patch('drift.tasks.restaurant_upgrade.archive_bytes',side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError,'disk full'):
                c.abandon('磁盘故障')
        self.assertEqual(c.path.read_bytes(),raw);self.assertTrue(c.active)
        c.action.assert_not_called()

    def test_menu_reports_stopped_after_open_so_only_close_is_issued(self):
        c=self.c;self.pending();self.step(c,101,due=False)
        self.menu();c.sailing.running.return_value=False
        self.step(c,102,due=False)
        self.assertEqual(self.points(),[(233,45),(600,363)])
        self.stopped()
        for now in (103,104,106):self.step(c,now,due=False)
        self.assertEqual(c.phase,'settings_open')
        self.assertNotIn('stop-toggle',c.journal['attempts'])

    def test_toggle_that_dismisses_menu_never_clicks_close_on_main(self):
        c=self.c;self.pending();self.step(c,101,due=False)
        self.menu();self.step(c,102,due=False);self.stopped()
        for now in (103,104,106):self.step(c,now,due=False)
        self.assertEqual(c.phase,'settings_open')
        self.assertEqual(self.points(),[(233,45),(360,880)])

    def test_restart_after_toggle_closes_without_repeating_toggle(self):
        c=self.c;self.pending();self.step(c,101,due=False)
        self.menu();self.step(c,102,due=False)
        saved=deepcopy(c.journal)
        c=self.controller(c.root,running=None);c.view._parallel.return_value=False
        c.sailing.menu.return_value=True;c.fishing._main_visible.return_value=False
        self.step(c,103,due=False)
        self.assertEqual([a.args[0]['kind'] for a in c.action.call_args_list],['stop-close'])
        self.assertEqual(c.journal['attempts'],{'stop':1,'stop-toggle':1,'stop-close':1})
        self.assertEqual(c.journal['origin_deadline'],saved['origin_deadline'])

    def test_failed_cleanup_entry_closes_menu_without_readable_settings(self):
        c=self.c;self.pending('detail');c.journal['phase_deadline']=110;c._save()
        raw=c.path.read_bytes();self.menu();c.settings=Mock(return_value=None)
        self.step(c,110,due=False)
        self.assertEqual(self.points(),[(600,363)])
        self.stopped();self.step(c,111,due=False);self.assert_failed_archive(raw)
        self.assertEqual(c.journal['retirement']['previous_phase'],'detail')

    def test_input_budget_exhaustion_closes_menu_before_retiring(self):
        c=self.c;self.pending('restore_menu')
        c.journal['attempts']={'restore-start':3};c._save();raw=c.path.read_bytes()
        self.menu();c.test_frame=self.frame()
        self.assertFalse(c._act('restore-start',(360,873),c.test_frame))
        self.assertEqual(self.points(),[(600,363)])
        self.stopped();self.step(c,101,due=False);self.assert_failed_archive(raw)
        self.assertEqual(c.journal['attempts']['restore-start'],3)

    def test_repeated_abandon_keeps_original_archive_and_close_deadline(self):
        c=self.c;raw=self.pending();self.menu();self.step(c,120,due=False)
        deadline=c.journal['menu_close']['deadline']
        retirement=c.journal['retirement'].copy()
        c.clock=lambda:122;c.abandon('重复放弃',frame=self.frame())
        self.assertEqual(c.journal['menu_close']['deadline'],deadline)
        self.assertEqual(c.journal['retirement'],retirement)
        self.step(c,123,due=False);self.assert_failed_archive(raw)
        self.assertEqual(self.points(),[(600,363)])

    def test_restart_during_menu_retirement_never_replays_close(self):
        c=self.c;raw=self.pending();self.menu();self.step(c,120,due=False)
        deadline=c.journal['menu_close']['deadline']
        c=self.controller(c.root,running=None)
        c.sailing.menu.return_value=True;c.fishing._main_visible.return_value=False
        self.step(c,122,due=False);self.assertTrue(c.active)
        self.assertEqual(c.journal['menu_close']['deadline'],deadline)
        self.step(c,123,due=False)
        self.c=c;self.assert_failed_archive(raw);c.action.assert_not_called()

    def test_startup_expired_journal_with_menu_uses_bounded_close(self):
        c=self.c;raw=self.pending();c._check_startup=True;self.menu()
        self.step(c,201,due=False)
        self.assertEqual(self.points(),[(600,363)])
        self.stopped();self.step(c,202,due=False);self.assert_failed_archive(raw)
        self.assertEqual(c.journal['retirement']['reason'],'启动时发现上次执行已过期')

    def test_factory_expiry_without_frame_checks_first_frame_once(self):
        c=self.c;raw=self.pending();c.clock=lambda:201
        self.assertTrue(c.retire_expired_on_start());self.assert_failed_archive(raw)
        self.menu();self.step(c,202,due=False)
        self.assertEqual(self.points(),[(600,363)])
        self.assertEqual(c.journal['menu_close']['deadline'],204)
        self.stopped();self.step(c,203,due=False);self.assert_failed_archive(raw)

    def test_retirement_window_permits_only_menu_close_and_never_extends_work_deadlines(self):
        c=self.c;self.pending('restore_menu',restore_deadline=110)
        c.journal.update(deadline=110,phase_deadline=110);c._save()
        self.menu();self.step(c,110,due=False)
        self.assertEqual(c.journal['deadline'],110)
        self.assertEqual(c.journal['phase_deadline'],110)
        for kind,point in (('stop',(233,45)),('restore-start',(360,873)),
                           ('feed:fish:0:0',(222,647)),('menu-leak-close',(600,362))):
            request=dict(kind=kind,phase=c.phase,x=point[0],y=point[1])
            self.assertFalse(c.validate(request,self.frame()))
            if kind!='menu-leak-close':self.assertFalse(c.reserve(request))
        self.assertEqual(self.points(),[(600,363)])


class RestaurantStopContextTests(unittest.TestCase):
    """Context entry and bounded recovery, through fresh validation and receipts."""
    controller=RestaurantStartupTests.controller
    frame=RestaurantStartupTests.frame
    step=RestaurantStartupTests.step
    setUp=RestaurantEntryMenuTests.setUp
    pending=RestaurantEntryMenuTests.pending
    menu=RestaurantEntryMenuTests.menu
    stopped=RestaurantEntryMenuTests.stopped
    points=RestaurantEntryMenuTests.points
    assert_failed_archive=RestaurantEntryMenuTests.assert_failed_archive

    def map_screen(self):
        self.c.fishing._main_visible.return_value=False
        self.c.bag._scan_map.return_value=True
        self.c.view._off_icon.return_value=True

    def panel_screen(self):
        self.c.fishing._main_visible.return_value=False
        self.c.visual.panel.return_value=True
        self.c.visual.point.return_value=(650,100)
        self.c.view._off_icon.return_value=True

    def test_restaurant_start_skips_context_input_and_keeps_resume_intent(self):
        c=self.c;self.pending()
        c.visual.find.return_value=(500,30);c.bag.moving.return_value=False
        c.view._off_icon.return_value=False  # No stop evidence is required here.
        frame=self.frame(place=7)
        self.assertTrue(c.at_restaurant(frame));self.assertIs(c.sailing.running(frame),True)
        self.step(c,101,frame,due=False)
        c.action.assert_not_called();self.assertEqual(c.phase,'open')
        origin=c.journal['origin']
        self.assertEqual(set(origin),{'parallel','header','route','at_restaurant','place'})
        self.assertIs(origin['at_restaurant'],True);self.assertIs(origin['route'],False)
        self.assertIs(origin['parallel'],False);self.assertEqual(origin['header'],c.header(frame))
        np.testing.assert_array_equal(unpack(origin['place']),frame[12:36,286:434])
        saved=json.loads(c.path.read_text(encoding='utf-8'))
        self.assertIs(saved['resume_route'],True);self.assertEqual(saved['origin'],origin)

    def test_restaurant_start_preserves_existing_origin_and_resume_intent(self):
        c=self.c;self.pending();c.bag.moving.return_value=False
        origin=c.origin_screen(self.frame(place=8))
        c.journal['origin']=deepcopy(origin);c.journal.pop('resume_route')
        c.at_restaurant=Mock(return_value=True)
        self.step(c,101,due=False)
        self.assertEqual(c.phase,'open');self.assertEqual(c.journal['origin'],origin)
        self.assertIs(c.journal['resume_route'],True);c.action.assert_not_called()

    def test_initial_map_uses_only_bounded_map_close(self):
        c=self.c;self.pending();self.map_screen()
        self.step(c,101,due=False)
        self.assertEqual([a['kind'] for a in self.issued],['cleanup-map-close'])
        self.assertEqual(self.points(),[(668,1187)]);self.assertNotIn('stop',c.journal['attempts'])
        self.assertNotIn('origin',c.journal);self.assertEqual(c.phase,'stop')
        c.bag._scan_map.return_value=False;self.stopped()
        c.visual.find.return_value=(500,30)
        self.step(c,102,due=False)
        self.assertEqual(c.phase,'open');self.assertEqual(len(self.issued),1)

    def test_initial_panel_uses_only_bounded_panel_close(self):
        c=self.c;self.pending();self.panel_screen()
        self.step(c,101,due=False)
        self.assertEqual([a['kind'] for a in self.issued],['cleanup-panel-close'])
        self.assertEqual(self.points(),[(650,100)]);self.assertNotIn('stop',c.journal['attempts'])
        c.visual.panel.return_value=False;self.stopped();c.visual.find.return_value=(500,30)
        self.step(c,102,due=False)
        self.assertEqual(c.phase,'open');self.assertEqual(len(self.issued),1)

    def test_entry_opening_restaurant_panel_enters_open(self):
        c=self.c;self.pending();self.step(c,101,due=False)
        self.panel_screen();c.at_restaurant=Mock(return_value=True)
        self.step(c,102,due=False)
        self.assertEqual(c.phase,'open');self.assertEqual(self.points(),[(233,45)])
        self.assertNotIn('retirement',c.journal);self.assertTrue(c.journal['resume_route'])
        self.assertTrue(c.journal['origin']['at_restaurant'])

    def test_entry_opening_map_closes_then_enters_open(self):
        c=self.c;self.pending();self.step(c,101,due=False)
        self.map_screen();self.step(c,102,due=False)
        self.assertEqual(self.points(),[(233,45),(668,1187)])
        self.assertEqual([a['kind'] for a in self.issued],['stop','cleanup-map-close'])
        self.assertEqual(c.phase,'stop');self.assertNotIn('origin',c.journal)
        c.bag._scan_map.return_value=False;self.stopped()
        self.assertFalse(c.at_restaurant(self.frame()))  # Entry result supplied destination context.
        self.step(c,103,due=False)
        self.assertEqual(c.phase,'open');self.assertTrue(c.journal['origin']['at_restaurant'])
        self.assertTrue(c.journal['resume_route']);self.assertNotIn('retirement',c.journal)
        self.assertEqual(len(self.issued),2)

    def test_missing_entry_result_abandons_after_five_seconds(self):
        c=self.c;self.pending();self.step(c,101,due=False)
        for now in (102,104,105.9):
            self.step(c,now,due=False);self.assertTrue(c.active)
        raw=c.path.read_bytes();self.step(c,106,due=False)
        self.assert_failed_archive(raw)
        self.assertEqual(c.journal['retirement']['reason'],'入口(233,45)未打开航巡菜单，画面语义不符')
        for now in (107,120,130):self.step(c,now,due=False)
        self.assertEqual(self.points(),[(233,45)]);self.assertEqual(c.action.call_count,1)
        self.assertEqual(c.journal['attempts'],{'stop':1})

    def test_normal_sea_entry_still_toggles_closes_and_captures_origin(self):
        c=self.c;self.pending();self.step(c,101,due=False)
        self.menu();self.step(c,102,due=False);self.step(c,103,due=False)
        self.assertNotIn('origin',c.journal);self.stopped()
        for now in (104,105,107):self.step(c,now,due=False)
        self.assertEqual(self.points(),[(233,45),(360,880),(600,363)])
        self.assertEqual(c.phase,'settings_open');self.assertIn('origin',c.journal)
        self.assertTrue(c.journal['resume_route'])

    def test_stop_revalidates_restaurant_context_before_input(self):
        c=self.c;self.pending();original=c.action.side_effect
        def arrive(a):
            c.at_restaurant=Mock(return_value=True)
            return original(a)
        c.action.side_effect=arrive
        self.step(c,101,due=False)
        self.assertEqual(self.points(),[]);self.assertNotIn('stop',c.journal['attempts'])
        self.step(c,102,due=False);self.assertEqual(c.phase,'open')

    def test_context_close_never_retries_and_expires_in_three_seconds(self):
        for screen,kind in ((self.map_screen,'cleanup-map-close'),
                            (self.panel_screen,'cleanup-panel-close')):
            with self.subTest(kind=kind):
                self.setUp()
                self.pending();screen();self.step(self.c,101,due=False)
                for now in (102,103):self.step(self.c,now,due=False)
                raw=self.c.path.read_bytes();self.step(self.c,104,due=False)
                self.assert_failed_archive(raw)
                self.assertEqual([a['kind'] for a in self.issued],[kind])
                self.assertEqual(self.c.action.call_count,1)

    def test_restart_does_not_repeat_context_close(self):
        c=self.c;self.pending();self.map_screen();self.step(c,101,due=False)
        saved=deepcopy(c.journal);c=self.c=self.controller(c.root)
        c.view._parallel.return_value=False;self.map_screen()
        self.step(c,102,due=False)
        self.assertTrue(c.active);self.assertEqual(c.journal['attempts'],saved['attempts'])
        self.step(c,104,due=False)
        self.assertFalse(c.active);c.action.assert_not_called()
        self.assertEqual(c.journal['attempts'],{'cleanup-map-close':1})

    def test_entry_deadline_survives_restart(self):
        c=self.c;self.pending();self.step(c,101,due=False)
        c=self.c=self.controller(c.root);c.view._parallel.return_value=False
        self.step(c,104,due=False);self.assertTrue(c.active)
        raw=c.path.read_bytes();self.step(c,106,due=False)
        self.assert_failed_archive(raw);c.action.assert_not_called()
        self.assertIn('未打开航巡菜单',c.journal['retirement']['reason'])
        self.assertEqual(c.journal['attempts'],{'stop':1})

    def test_unknown_entry_result_times_out_even_in_other_world(self):
        c=self.c;self.pending();self.step(c,101,due=False)
        raw=c.path.read_bytes();self.step(c,106,self.frame(world=1),due=False)
        self.assert_failed_archive(raw)
        self.assertIn('未打开航巡菜单',c.journal['retirement']['reason'])
        self.assertEqual(self.points(),[(233,45)])

    def test_context_close_revalidates_window_before_input(self):
        c=self.c;self.pending();self.map_screen();original=c.action.side_effect
        def vanished(a):
            c.bag._scan_map.return_value=False;c.fishing._main_visible.return_value=True
            return original(a)
        c.action.side_effect=vanished
        self.step(c,101,due=False)
        self.assertEqual(self.points(),[]);self.assertFalse(c.active)
        self.assertEqual(c.action.call_count,1)

    def test_map_close_does_not_authorize_other_world_destination(self):
        c=self.c;self.pending();self.step(c,101,due=False)
        self.map_screen();self.step(c,102,due=False)
        c.bag._scan_map.return_value=False;self.stopped()
        self.step(c,103,self.frame(world=1),due=False)
        self.assertEqual(c.phase,'stop');self.assertNotIn('origin',c.journal)
        self.step(c,105,self.frame(world=1),due=False)
        self.assertFalse(c.active);self.assertEqual(self.points(),[(233,45),(668,1187)])

    def test_late_entry_result_cannot_extend_five_second_budget(self):
        for result in ('restaurant','map','menu'):
            with self.subTest(result=result):
                self.setUp();c=self.c;self.pending();self.step(c,101,due=False)
                if result=='restaurant':c.at_restaurant=Mock(return_value=True)
                elif result=='map':self.map_screen()
                else:self.menu()
                raw=c.path.read_bytes();self.step(c,107,due=False)
                self.assertTrue('retirement' in c.journal)
                if result=='menu':
                    self.stopped();self.step(c,108,due=False)
                self.assert_failed_archive(raw)
                self.assertIn('未打开航巡菜单',c.journal['retirement']['reason'])
                expected=[(233,45),(600,363)] if result=='menu' else [(233,45)]
                self.assertEqual(self.points(),expected)

    def test_recognized_map_closes_even_if_main_probe_also_matches(self):
        c=self.c;self.pending();self.step(c,101,due=False);self.map_screen()
        c.fishing._main_visible.return_value=True
        self.step(c,102,due=False)
        self.assertEqual(self.points(),[(233,45),(668,1187)])
        self.assertNotIn('origin',c.journal)


class RestaurantJournalTests(unittest.TestCase):
    def controller(self,d,phase='detail'):
        c=RestaurantUpgrade.__new__(RestaurantUpgrade)
        c.root=Path(d);c.path=Path(d)/'restaurant_upgrade_state.json';c.clock=lambda:100
        c.phase=phase;c.journal=dict(version=1,serial='x',pending=True,phase=phase,deadline=1000,
            phase_deadline=200,attempts={},origin={'route':False},skipped=[],visited=[],upgraded=0,submitted=0,
            category=0,feed_counts={},blocked_ingredients=[])
        c.key=c.since=c.last_seen=c.last_frame=None;c.seen=0;c.next_at=0
        c.event=Mock();c.enabled=lambda:True;c.visual=Mock();c.view=Mock();c._save()
        return c

    def test_each_hold_has_bounded_feed_reservation_budget(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);a={'kind':'feed:a:1:0:0'}
            for _ in range(24):self.assertTrue(c.reserve(a))
            self.assertFalse(c.reserve(a))
            self.assertEqual(json.loads(c.path.read_text())['attempts'][a['kind']],24)

    def test_same_phase_never_extends_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);c._set('detail',600)
            self.assertEqual(c.journal['phase_deadline'],200)

    def test_expired_start_archives_and_retires_without_cleanup(self):
        # 过期记录只归档退休（保留原 deadline/attempts 作为证据），不再续开 cleanup。
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);c.journal['attempts']={'feed:a:1:0:0':1};c._save()
            old=c.path.read_bytes();c.clock=lambda:1001;c.retire_expired_on_start()
            self.assertEqual(c.journal['deadline'],1000)
            self.assertEqual(c.journal['attempts'],{'feed:a:1:0:0':1})
            self.assertEqual(c.phase,'idle');self.assertFalse(c.active)
            self.assertNotIn('cleanup_deadline',c.journal)
            self.assertIn(old,[p.read_bytes() for p in (Path(d)/'logs/restaurant-history').glob('*.json')])
            c.same_world=lambda _:True
            self.assertFalse(c.validate({'kind':'feed:new:1:1:0','phase':'cleanup','x':255,'y':647},object()))

    def test_cleanup_window_is_not_restarted(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);self.assertTrue(c._begin_cleanup('test'))
            deadline=c.journal['cleanup_deadline']
            c.clock=lambda:deadline+1
            # 第二次申请不再抛异常（抛异常会打死 runner），改为归档退休并返回 False。
            self.assertFalse(c._begin_cleanup('again'))
            self.assertEqual(c.journal['cleanup_deadline'],deadline)
            self.assertEqual(c.phase,'idle');self.assertFalse(c.active)
            self.assertEqual(c.journal['last_failure']['reason'],'test')
            self.assertEqual(c.journal['retirement']['reason'],'again')

    def test_cached_frame_is_not_fresh_confirmation(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);f=object()
            for now in (100,101,104):
                c.clock=lambda now=now:now;self.assertFalse(c._confirmed('x',f,3))

    def test_new_counter_before_input_invalidates_old_proposal(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);c.same_world=lambda _:True;c.view._off_icon.return_value=True
            before=dict(identity='a',level=1,counters=[(0,8)]*4)
            c.visual.detail_values.return_value={**before,'counters':[(1,8)]+[(0,8)]*3}
            a=dict(kind='feed:a:1:0:0',phase='detail',x=142,y=647,hold_ms=3000,ingredient=0,before=before)
            self.assertFalse(c.validate(a,object()))

    def test_other_world_refused_even_when_material_ready(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);c.same_world=lambda _:False
            self.assertFalse(c.validate(dict(kind='feed:a:1:0:0',phase='detail',x=142,y=647),object()))

    def test_previous_level_attempt_does_not_block_next_level_feed(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);c.same_world=lambda _:True;c.view._off_icon.return_value=True
            c.journal['attempts']={'upgrade:a:1':1};c.journal['card']={'level':2}
            v=dict(identity='a',level=2,counters=[(0,8)]*4)
            c.visual.detail_values.return_value=v;c._confirmed=lambda *a:True
            c.visual.upgrade_ready.return_value=False;c.visual.ingredient_ready.return_value=True
            c.action=Mock(return_value=True)
            self.assertTrue(c.validate(dict(kind='feed:a:2:0:0',phase='detail',x=142,y=647,before=v,ingredient=0,hold_ms=3000),object()))
            c.step(object(),due=False)
            self.assertEqual(c.phase,'feed_wait')
            self.assertEqual(c.journal['attempts'],{'upgrade:a:1':1})

    def test_input_receipt_is_not_upgrade_proof(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d,'upgrade_wait');before=dict(identity='a',level=1,counters=[(8,8)]*4)
            c.journal['awaiting']=before;c.same_world=lambda _:True;c.visual.detail_values.return_value=before
            c.step(object(),due=False)
            self.assertEqual(c.journal['upgraded'],0);self.assertEqual(c.phase,'upgrade_wait')

    def test_real_level_increment_requires_three_new_frames_over_three_seconds(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d,'upgrade_wait');c.journal['awaiting']=dict(identity='a',level=1,counters=[(8,8)]*4)
            c.same_world=lambda _:True;c.visual.detail_values.return_value=dict(identity='a',level=2,counters=[(0,8)]*4)
            for now,expected in ((100,0),(101,0),(103,1)):
                c.clock=lambda now=now:now;c.step(object(),due=False);self.assertEqual(c.journal['upgraded'],expected)
            self.assertEqual(c.phase,'detail')

class RestaurantScheduleTests(unittest.TestCase):
    def test_failed_run_also_waits_three_days(self):
        with tempfile.TemporaryDirectory() as d:
            now=datetime(2026,10,25,9,41,23,tzinfo=BEIJING)
            s=TaskSchedule(d,'x',lambda:now)
            s.begin('restaurant_upgrade',259200)
            s.finish('restaurant_upgrade',259200,success=False)
            self.assertEqual(s.next_run('restaurant_upgrade'),now+timedelta(days=3))
            self.assertEqual(s.entries['restaurant_upgrade']['reason'],'retry')

    def test_three_day_schedule_survives_restart(self):
        with tempfile.TemporaryDirectory() as d:
            now=datetime(2026,10,25,4,tzinfo=BEIJING);s=TaskSchedule(d,'x',lambda:now)
            self.assertEqual(RULE['interval'],259200)
            s.begin('restaurant_upgrade',259200);s.finish('restaurant_upgrade',259200)
            reload=TaskSchedule(d,'x',lambda:now)
            self.assertEqual(reload.next_run('restaurant_upgrade'),now+timedelta(days=3))
            self.assertFalse(reload.ready('restaurant_upgrade'))

    def test_config_append_is_idempotent_and_preserves_online_changes(self):
        original=dict(serial='x',rules=[dict(id='another',enabled=False,interval=123)],poll_interval=.7)
        before=deepcopy(original);updated=add(original)
        self.assertEqual(original,before);self.assertEqual(updated['rules'][:-1],before['rules'])
        updated['rules'][-1]['interval']=7200;updated['rules'][-1]['enabled']=False
        self.assertEqual(add(updated),updated)

if __name__=='__main__':unittest.main()
