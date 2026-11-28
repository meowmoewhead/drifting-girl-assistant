"""A real lowered Sea God battle must not starve its own cast observation."""
from copy import deepcopy
import hashlib
import json
import time
import unittest
from unittest.mock import patch

from drift.tasks.boss_defense import BossDefense
from drift.tasks.stone_farm import StoneFarm
from tests import test_stone_farm as base
from tests import test_stone_boss_defense as defense_tests


class FirstCombatRuntimeTests(unittest.TestCase):
    def test_confirming_real_cast_does_not_spend_remaining_time_on_combat_burst(self):
        original_init = StoneFarm.__init__
        calls = []
        def initialize(worker, *args, **kwargs):
            original_init(worker, *args, **kwargs)
            worker.phase = worker.journal['phase'] = 'first_verify'
        def observe(worker, image):
            calls.append('observe')
            worker.seen, worker.first_result_rank = 1, 39
            return False
        def battle(worker, *args):
            calls.append('battle')
            return True
        with patch.object(StoneFarm, '__init__', initialize), \
             patch.object(StoneFarm, 'observe_first_cast', observe), \
             patch.object(StoneFarm, 'input_allowed', return_value=True), \
             patch.object(BossDefense, 'step', battle):
            defense_tests.StoneBossDefenseTests().exercise([defense_tests.frame('native-right-charge-low')])
        self.assertIn('observe', calls)
        self.assertNotIn('battle', calls)

    def test_worker_observes_first_cast_before_combat_can_continue_the_loop(self):
        original_init = StoneFarm.__init__
        calls = []
        def initialize(worker,*args,**kwargs):
            original_init(worker,*args,**kwargs)
            worker.phase=worker.journal['phase']='first_verify'
        def observe(worker,image):
            calls.append('observe')
            return False
        def battle(worker,*args):
            calls.append('battle')
            return True
        with patch.object(StoneFarm,'__init__',initialize), \
             patch.object(StoneFarm,'observe_first_cast',observe), \
             patch.object(BossDefense,'step',battle):
            defense_tests.StoneBossDefenseTests().exercise([defense_tests.frame('native-right-charge-low')])
        self.assertGreaterEqual(calls.count('observe'),1)
        self.assertEqual(calls[0:2],['observe','battle'])
        self.assertTrue(all(calls[i:i+2]==['observe','battle'] for i in range(0,len(calls),2)))
