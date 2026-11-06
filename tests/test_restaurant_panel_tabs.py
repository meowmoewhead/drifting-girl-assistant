"""The container header must work for materials and cooked-food storage tabs."""
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from drift.perception.restaurant import RestaurantVisual

ROOT=Path(__file__).resolve().parents[1]


class RestaurantPanelTabsTests(unittest.TestCase):
    def setUp(self):
        self.visual=RestaurantVisual(ROOT,None)

    def read(self,name):
        return np.asarray(Image.open(ROOT/name).convert('RGB'))

    def test_storage_tab_is_a_restaurant_container_but_never_enhancement(self):
        frame=self.read('tests/fixtures/restaurant-list/storage-tab.png')
        self.assertTrue(self.visual.panel(frame))
        self.assertFalse(self.visual.atlas(frame))
        self.assertFalse(self.visual.detail(frame))
        self.assertIsNone(self.visual.detail_values(frame))
        self.assertEqual(self.visual.cards(frame),[])

    def test_original_materials_atlas_remains_recognized(self):
        frame=self.read('tests/fixtures/restaurant-list/settled.png')
        self.assertTrue(self.visual.panel(frame))
        self.assertTrue(self.visual.atlas(frame))

    def test_header_book_and_cross_are_all_still_required(self):
        original=self.read('tests/fixtures/restaurant-list/storage-tab.png')
        for area in [(35,911,165,947),(318,81,397,161),(592,145,683,220)]:
            frame=original.copy();x,y,X,Y=area;frame[y:Y,x:X]=0
            self.assertFalse(self.visual.panel(frame),area)

    def test_main_screen_and_unrelated_purchase_dialog_are_rejected(self):
        for name in ['assets/event-potion/main-star.png','assets/event-potion/confirm-ghost.png']:
            self.assertFalse(self.visual.panel(self.read(name)))


if __name__=='__main__':unittest.main()
