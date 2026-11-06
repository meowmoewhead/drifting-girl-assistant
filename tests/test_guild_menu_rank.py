"""Guild restoration reads an actual dimmed native menu through its menu decoder."""
from pathlib import Path
from copy import deepcopy
import tempfile
import unittest
import numpy as np
from PIL import Image
from digit_ocr import DigitReader
from drift.tasks.guild_daily import GuildDaily

ROOT=Path(__file__).resolve().parents[1]

class GuildMenuRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.reader=DigitReader(ROOT)

    def test_actual_menu_restoration_and_read_only_receipt(self):
        for index in range(3):
            with self.subTest(index=index),tempfile.TemporaryDirectory() as folder:
                c=GuildDaily(ROOT,self.reader,lambda _:self.fail('Unexpected input'),lambda _:None,
                    'test-device',state_path=Path(folder)/'guild_daily_state.json',clock=lambda:100.)
                f=np.array(Image.open(ROOT/f'tests/fixtures/cruise-menu-rank5/menu-{index}.png').convert('RGB'))
                settings=c.settings(f)
                self.assertIsNotNone(settings)
                origin=dict(parallel=True,stage=10,rank=5,route=True,fishing=True)
                target=dict(origin,settings=settings)
                c.journal.update(period='test',resume_pending=True,origin=origin,
                    target=target,farm_origin=deepcopy(target),settings=settings,
                    return_target_reconciliation=dict(read_only=True),deadline=1.,phase_deadline=1.,
                    attempts={'return-confirm':1,'enter-confirm':1},categories=[0],submitted=7)
                c._save()
                self.assertTrue(c.same_origin(f,menu=True))
                self.assertTrue(c._observe_stale_return(f,'test'))
                self.assertEqual(c.journal['resume_settings'],settings)
                self.assertTrue(c.journal['resume_pending'])
                self.assertEqual(c.journal['deadline'],1.)
                self.assertEqual(c.journal['attempts'],{'return-confirm':1,'enter-confirm':1})
                c.journal['target']['rank']=6
                self.assertFalse(c.same_origin(f,menu=True))

if __name__=='__main__':unittest.main()
