from copy import deepcopy
import unittest
from unittest.mock import Mock
import numpy as np

from tests import test_guild_return_home as fixtures


class GuildDepartureTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.GuildReturnHomeTests('test_parallel_start_and_snapshot_persist_before_departure')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.f=fixture;self.c=fixture.c
        self.c.journal.update(period='2026-11-04',farm_origin=deepcopy(fixtures.FARM),
            farm_restored_at=None,resume_pending=True)
        self.c._observe_stale_return=lambda *a:False

    def test_unknown_departure_does_not_reserve_old_rank_as_new_target(self):
        self.c.origin_screen=Mock(return_value=None);self.c.reject_reason=None
        before=deepcopy(self.c.journal)
        for _ in range(4):self.f.tick(due=True)
        self.assertFalse(self.c.active)
        self.assertEqual(self.c.journal,before)
        self.assertEqual(self.f.inputs,[])

    def test_next_confirmed_real_departure_replaces_old_rank(self):
        old=deepcopy(fixtures.FARM);old.update(stage=10,rank=37)
        actual={**old,'rank':11}
        self.c.journal['farm_origin']=old
        self.c.origin_screen=Mock(return_value=None);self.c.reject_reason=None
        self.f.tick(due=True)
        self.c.origin_screen=Mock(return_value=actual)
        self.c.same_origin=lambda *a,**kw:False
        for _ in range(3):self.f.tick(due=True)
        self.assertEqual(self.c.journal['origin']['rank'],11)
        self.assertEqual(self.c.journal['target']['rank'],11)
        self.assertTrue(self.c.journal['capture_target_settings'])
        self.assertEqual(self.f.inputs,[])


if __name__=='__main__':unittest.main()
