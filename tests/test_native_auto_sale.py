"""Real native menus and input-budget guards for mandatory auto-sale."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2

from digit_ocr import DigitReader
from drift.tasks import native_auto_sale as sale
from drift.tasks.meter_calibration import MeterCalibration
from drift.tasks.stone_farm import StoneFarm

ROOT = Path(__file__).resolve().parents[1]


class NativeAutoSaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)
        cls.off = cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/native-auto-sale/sale-off.png')),cv2.COLOR_BGR2RGB)
        cls.on = cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/native-auto-sale/sale-on.png')),cv2.COLOR_BGR2RGB)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.actions = []
        self.w = StoneFarm(ROOT,self.reader,lambda a:self.actions.append(a) or True,
                           lambda _:None,state_path=Path(self.temp.name)/'stone.json')
        self.w.journal.update(pending=True,reason='lower',rank=39,to_stage=9,entry_rank=31,
            from_stage=9,phase='route_menu',deadline=1090,phase_deadline=1045,
            return_deadline=1030,attempts={},inputs=[])
        self.w.phase = 'route_menu'
        self.w.off_verified = True
        self.w.journal['native_extras'] = self.w.settings_key(self.off)[1]
        self.clock = patch('time.time',return_value=1000)
        self.clock.start()

    def tearDown(self):
        self.clock.stop()
        self.temp.cleanup()

    def action(self, kind, frame, xy):
        return dict(kind=kind,phase='route_menu',signature=self.w.signature(frame),x=xy[0],y=xy[1])

    def arm(self):
        return sale.arm(self.w,self.off,self.w.settings_key(self.off))

    def test_real_on_off_and_unknown_have_distinct_results(self):
        self.assertIs(self.w.sailing.sale_selected(self.off),False)
        self.assertIs(self.w.sailing.sale_selected(self.on),True)
        hidden=self.on.copy();hidden[625:670,550:605]=255
        self.assertIsNone(self.w.sailing.sale_selected(hidden))
        self.assertIsNone(self.w.sailing.sale_selected(self.on//2))

    def test_fingerprint_ignores_only_sale_and_coupon_count(self):
        self.assertEqual(sale.preserved_key(self.w.sailing,self.off),sale.preserved_key(self.w.sailing,self.on))
        f=self.on.copy();f[680:715,450:530]=255
        self.assertEqual(sale.preserved_key(self.w.sailing,f),sale.preserved_key(self.w.sailing,self.on))
        for box in ((557,741,595,781),(260,520,415,574),(434,520,594,574)):
            changed=self.on.copy();x,y,X,Y=box;changed[y:Y,x:X]=255
            self.assertNotEqual(sale.preserved_key(self.w.sailing,changed),sale.preserved_key(self.w.sailing,self.on))

    def test_off_menu_cannot_start_even_when_all_goals_match(self):
        self.w.menu_key=self.w.settings_key(self.off)
        self.assertFalse(self.w.validate_portal(self.action('route-start',self.off,(360,873)),self.off))

    def test_toggle_requires_stopped_current_menu_and_reviewed_off_box(self):
        self.assertTrue(self.arm())
        a=self.action(sale.ACTION,self.off,(576,647))
        self.assertTrue(self.w.validate_portal(a,self.off))
        for f in (self.on,self.off//2):
            self.assertFalse(self.w.validate_portal(a,f))
        self.w.off_verified=False
        self.assertFalse(self.w.validate_portal(a,self.off))

    def test_changed_rank_or_world_rejects_toggle(self):
        self.arm();a=self.action(sale.ACTION,self.off,(576,647))
        for box in ((40,0,170,65),(8,45,75,74)):
            f=self.off.copy();x,y,X,Y=box;f[y:Y,x:X]=255
            self.assertFalse(self.w.validate_portal(a,f))

    def test_toggle_does_not_allow_different_coordinates(self):
        self.arm()
        self.assertFalse(self.w.validate_portal(self.action(sale.ACTION,self.off,(576,760)),self.off))

    def test_verified_on_preserves_other_options_and_deadlines(self):
        self.arm();before=deepcopy(self.w.journal)
        self.assertFalse(sale.verified_on(self.w,self.on,self.w.settings_key(self.on),{}))
        self.w.journal['attempts'][sale.ACTION]=1
        self.assertTrue(sale.verified_on(self.w,self.on,self.w.settings_key(self.on),self.w.journal['attempts']))
        self.assertEqual(self.w.journal['native_extras'],self.w.settings_key(self.on)[1])
        for key in ('deadline','phase_deadline','return_deadline'):
            self.assertEqual(self.w.journal[key],before[key])
        self.w.menu_key=self.w.settings_key(self.on)
        self.assertTrue(self.w.validate_portal(self.action('route-start',self.on,(360,873)),self.on))

    def test_upgrade_or_priority_change_cannot_be_adopted_with_sale(self):
        self.arm();self.w.journal['attempts'][sale.ACTION]=1
        for box in ((557,741,595,781),(260,520,415,574)):
            f=self.on.copy();x,y,X,Y=box;f[y:Y,x:X]=255
            self.assertFalse(sale.verified_on(self.w,f,self.w.settings_key(f),self.w.journal['attempts']))
        self.assertFalse(self.w.journal['auto_sale_guard']['verified'])

    def test_changed_target_cannot_be_adopted_with_sale(self):
        self.arm();settings=self.w.settings_key(self.on)
        self.assertFalse(sale.verified_on(self.w,self.on,((False,False,True,True,True),settings[1]),{sale.ACTION:1}))

    def test_repeated_or_expired_input_never_resets_budget(self):
        self.arm();a=self.action(sale.ACTION,self.off,(576,647))
        self.assertTrue(self.w.reserve_input(a))
        with self.assertRaises(RuntimeError):self.w.reserve_input(a)
        self.w.journal['phase_deadline']=999
        self.assertFalse(self.w.validate_portal(a,self.off))
        self.assertEqual(self.w.journal['attempts'][sale.ACTION],1)

    def test_saved_pending_toggle_can_confirm_without_repeating_input(self):
        self.arm();self.w.journal['attempts'][sale.ACTION]=1
        raw=json.dumps(self.w.journal)
        self.w.journal=json.loads(raw)
        self.assertTrue(sale.verified_on(self.w,self.on,self.w.settings_key(self.on),self.w.journal['attempts']))
        self.assertFalse(sale.arm(self.w,self.off,self.w.settings_key(self.off)))
        self.assertEqual(self.actions,[])

    def test_stone_route_steps_enable_sale_then_start_once(self):
        def issue(action):
            frame = self.off if action['kind'] == sale.ACTION else self.on
            self.assertTrue(self.w.validate_portal(action,frame))
            self.assertTrue(self.w.reserve_input(action))
            self.actions.append(action['kind'])
            return True
        self.w.action=issue
        for i,f in enumerate([self.off]*4+[self.on]*4):
            with patch('time.monotonic',return_value=100+i*1.6):
                self.w.next_at=0
                self.w.step(f)
        self.assertEqual(self.actions,[sale.ACTION,'route-start'])
        self.assertEqual(self.w.journal['attempts'][sale.ACTION],1)
        self.assertEqual(self.w.journal['deadline'],1090)
        self.assertEqual(self.w.journal['auto_sale_guard']['targets'],[True,False,True,True,True])

    def meter(self):
        c=object.__new__(MeterCalibration)
        c.sailing=self.w.sailing;c.journal=dict(pending=True,deadline=1090,sailing_attempts={})
        c.phase='menu';c.off_verified=True;c.menu_key=c.settings_key(self.off)
        c._save=lambda:None;c.enabled=lambda:True
        c.screen=lambda f:dict(menu=c.sailing.menu(f),route=None,rank=31,stage=9)
        return c

    def test_meter_start_requires_positive_sale_check(self):
        c=self.meter()
        self.assertFalse(c.validate_sailing('start',self.off))
        c.menu_key=c.settings_key(self.on)
        self.assertTrue(c.validate_sailing('start',self.on))

    def test_meter_toggle_has_same_single_attempt_and_deadline(self):
        c=self.meter();self.assertTrue(sale.arm(c,self.off,c.settings_key(self.off)))
        self.assertTrue(c.validate_sailing(sale.ACTION,self.off))
        self.assertTrue(c.reserve_sailing_input(sale.ACTION))
        with self.assertRaises(RuntimeError):c.reserve_sailing_input(sale.ACTION)
        c.journal['deadline']=999
        self.assertFalse(c.validate_sailing(sale.ACTION,self.off))
        self.assertEqual(c.journal['sailing_attempts'][sale.ACTION],1)


if __name__ == '__main__':unittest.main()
