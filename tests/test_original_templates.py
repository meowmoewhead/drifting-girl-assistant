"""Original artwork must preserve controller guards and crop coordinates."""
from pathlib import Path
import json
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from drift.perception.original_templates import OriginalTemplates
from drift.tasks.equipment_upgrade import EquipmentUpgrade
from drift.tasks.training_sailing import TrainingSailing
from drift.perception.restaurant import RestaurantVisual

ROOT = Path(__file__).resolve().parents[1]


def frame(path):
    return cv2.cvtColor(cv2.imread(str(ROOT/path)), cv2.COLOR_BGR2RGB)


class OriginalTemplateTests(unittest.TestCase):
    def test_all_reviewed_crops_and_uniform_negatives(self):
        manifest = json.loads((ROOT/'assets/original-templates/manifest.json').read_text('utf-8'))
        for entry in manifest['templates']:
            with self.subTest(key=entry['group']+'/'+entry['name']):
                bank = OriginalTemplates(ROOT, entry['group'])
                legacy = frame('assets/'+entry['group']+'/'+entry['name']+'.png')
                h,w=legacy.shape[:2];f=np.zeros((1280,720,3),np.uint8)
                f[100:100+h,100:100+w]=legacy
                # Actual callers retain their existing thresholds. This test
                # checks the source crop alignment at its reviewed floor.
                threshold=.90 if entry['name']=='map-restaurant' else .94
                hit=bank.find(f,entry['name'],(98,98,102+w,102+h),threshold)
                self.assertIsNotNone(hit)
                self.assertEqual(hit[1],(100,100,w,h))
                for value in (0,120,255):
                    f[:]=value
                    self.assertIsNone(bank.find(f,entry['name'],(98,98,102+w,102+h),threshold))

    def test_transparent_background_does_not_choose_click_geometry(self):
        bank=OriginalTemplates(ROOT,'equipment')
        rgb,mask=bank.entries['rod-icon'];h,w=mask.shape
        f=np.full((1280,720,3),70,np.uint8)
        f[200:200+h,200:200+w]=rgb
        hit=bank.find(f,'rod-icon',(197,197,203+w,203+h),.94)
        self.assertEqual(hit[1],(200,200,w,h))
        crop=f[200:200+h,200:200+w];crop[mask==0]=(30,120,225)
        changed=bank.find(f,'rod-icon',(197,197,203+w,203+h),.94)
        self.assertEqual(changed[1],hit[1])
        self.assertAlmostEqual(changed[2],hit[2],places=4)

    def test_unreviewed_or_replaced_legacy_asset_uses_existing_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            shutil.copytree(ROOT/'assets/original-templates',root/'assets/original-templates')
            shutil.copytree(ROOT/'assets/equipment',root/'assets/equipment')
            path=root/'assets/equipment/rod-icon.png'
            path.write_bytes(path.read_bytes()+b'changed')
            bank=OriginalTemplates(root,'equipment')
            self.assertFalse(bank.has('rod-icon'))
            self.assertTrue(bank.has('cannon-icon'))
            self.assertFalse(bank.has('header-attack'))

    def test_corrupt_original_is_not_silently_loaded(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            shutil.copytree(ROOT/'assets/original-templates',root/'assets/original-templates')
            shutil.copytree(ROOT/'assets/equipment',root/'assets/equipment')
            path=root/'assets/original-templates/equipment--rod-icon.png'
            path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'integrity'):
                OriginalTemplates(root,'equipment')

    def test_unknown_shape_and_out_of_bounds_are_rejected(self):
        bank=OriginalTemplates(ROOT,'equipment')
        self.assertIsNone(bank.find(np.zeros((720,1280,3),np.uint8),'rod-icon',(0,0,90,90),.9))
        self.assertIsNone(bank.find(np.zeros((1280,720,3),np.uint8),'rod-icon',(-1,0,90,90),.9))

    def test_real_weapon_panels_preserve_money_title_and_dim_guards(self):
        e=EquipmentUpgrade(ROOT,None,None,None,lambda *_:None)
        for name,page in [('top','top'),('top-upgraded','top'),('bottom','bottom'),('bottom-upgraded','bottom')]:
            f=frame('tests/fixtures/equipment/'+name+'.png')
            self.assertTrue(e.page_visible(f,page),name)
            self.assertFalse(e.page_visible((f*.4).astype(np.uint8),page),name)
            f[797:832,27:63]=0
            self.assertFalse(e.panel(f),name)

    def test_selected_and_unselected_sailing_targets_are_preserved(self):
        s=TrainingSailing(ROOT,type('T',(),{'state':{}})(),None,lambda *_:None)
        for name in ('menu-one','menu-two','menu-three','menu-no-pearl'):
            f=frame('tests/fixtures/training-sailing/'+name+'.png')
            self.assertTrue(s.menu(f))
            for key in ('pearl','amber','gem','material','boss'):
                self.assertIsNotNone(s.selected(f,key),(name,key))
            self.assertFalse(s.menu((f*.6).astype(np.uint8)))

    def test_real_restaurant_map_negatives(self):
        r=RestaurantVisual(ROOT,None)
        atlas=frame('tests/fixtures/restaurant-map/negative-03989.png')
        # This is a map-detection negative, but a real noodle atlas positive.
        self.assertTrue(r.atlas(atlas))
        self.assertIsNone(r.find(atlas,'map-restaurant'))
        self.assertFalse(r.detail(atlas))
        world_map=frame('tests/fixtures/restaurant-map/s04.png')
        self.assertFalse(r.atlas(world_map))
        self.assertFalse(r.detail(world_map))
        self.assertIsNotNone(r.find(world_map,'map-restaurant'))


if __name__=='__main__':unittest.main()
