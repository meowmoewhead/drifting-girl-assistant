"""One approved stage-9 lowering can abandon one guarded Sea God battle."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from digit_ocr import DigitReader
from drift.tasks.stone_farm import StoneFarm
from tests import test_stone_farm as helpers

ROOT, image = helpers.ROOT, helpers.image


class LowerBossAbandonTests(unittest.TestCase):
    setUp = helpers.StoneFarmTests.setUp
    tearDown = helpers.StoneFarmTests.tearDown
    act = helpers.StoneFarmTests.act
    tick = helpers.StoneFarmTests.tick
    def begin(self, **options):
        # Replay the older saved approval format. New automatic handoffs use
        # the lower-right continuous-stop button, tested separately.
        helpers.StoneFarmTests.begin(self, **options)
        self.w.journal.pop('continuous_cast_stop',None)
    action = helpers.StoneFarmTests.action

    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)
        cls.main = image('tests/fixtures/stone-farm/lower-seagod-battle.png')

    def prepare(self):
        self.begin()
        self.w.journal['allow_boss_abandon'] = True
        self.w._set('stop_fishing',180)
        self.w.off_verified = True
        self.w.boss_abandon_key = self.w.boss_abandon_context(self.main)
        action = self.action('lower-boss-abandon',(611,184))
        action['abandon_context'] = self.w.boss_abandon_key
        return action

    def test_real_seagod_button_and_complete_current_context(self):
        action = self.prepare()
        self.assertEqual(self.w.screen(self.main),
            dict(rank=39,stage=9,menu=False,completion=False,route=False))
        self.assertIsNotNone(self.w.boss_abandon_key)
        self.assertEqual(self.w.boss_abandon_key[1],(611,184))
        self.assertTrue(self.w.validate_portal(action,self.main))
        self.assertFalse(self.w.validate_portal(dict(action,x=690,y=1090),self.main))
        self.assertFalse(self.w.validate_portal(action,self.main[:900]))

    def test_missing_real_battle_guards_or_red_x_never_authorize(self):
        action = self.prepare()
        for name,box in (
            ('world',(8,40,46,80)),('rank',(65,9,139,63)),
            ('name',(187,168,280,199)),('skull',(248,32,295,81)),
            ('timer',(500,125,556,165)),('swords',(307,962,415,1057)),
            ('star',(208,1137,269,1192)),('diamond',(539,1137,600,1186)),
            ('auto',(276,881,310,913)),('red-x',(584,158,642,210)),
            ('off',(194,4,246,42)),('stage',(40,45,68,71)),
            ('place',(286,12,434,34))):
            with self.subTest(name=name):
                changed=self.main.copy();x,y,X,Y=box;changed[y:Y,x:X]=0
                self.assertFalse(self.w.validate_portal(action,changed))
        self.assertFalse(self.w.validate_portal(action,self.main//2))
        with patch.object(self.w.sailing,'running',return_value=True):
            self.assertFalse(self.w.validate_portal(action,self.main))

    def test_wrong_phase_scope_consent_or_unproved_off_never_authorize(self):
        action=self.prepare()
        saved=deepcopy(self.w.journal)
        for name,value in (('allow_boss_abandon',False),('from_stage',8),('to_stage',9),
                           ('rank',40),('entry_rank',0),('reason','cycle')):
            with self.subTest(name=name):
                self.w.journal={**saved,name:value}
                self.assertFalse(self.w.validate_portal(action,self.main))
        self.w.journal=saved
        self.w.off_verified=False
        self.assertFalse(self.w.validate_portal(action,self.main))
        self.w.off_verified=True;self.w.boss_abandon_key=None
        self.assertFalse(self.w.validate_portal(action,self.main))
        self.w.boss_abandon_key=action['abandon_context'];self.w.phase='return'
        self.assertFalse(self.w.validate_portal(dict(action,phase='return'),self.main))

    def test_deadline_is_not_extended_to_abandon(self):
        action=self.prepare()
        self.now=self.w.journal['return_deadline']+1
        self.assertFalse(self.w.validate_portal(action,self.main))
        with self.assertRaises(RuntimeError):
            self.tick()
        self.assertEqual(self.actions,[])

    def test_three_seconds_real_battle_then_one_attempt_and_no_stop_credit(self):
        self.prepare();self.w.boss_abandon_key=None
        self.tick();self.tick()
        self.assertEqual(self.actions,[])
        self.tick()
        self.assertEqual(len(self.actions),1)
        action=self.actions[0]
        self.assertEqual(action['kind'],'lower-boss-abandon')
        self.assertFalse(self.w.fishing_stopped)
        self.assertEqual(self.w.phase,'stop_fishing')
        self.assertTrue(self.w.reserve_input(action))
        for _ in range(4):self.tick()
        self.assertEqual(len(self.actions),1)
        self.assertFalse(self.w.fishing_stopped)
        self.assertEqual(self.w.phase,'stop_fishing')

    def test_unknown_frame_breaks_battle_consensus_and_green_arrow_requires_stop(self):
        self.prepare();self.w.boss_abandon_key=None
        self.tick();self.tick()
        self.tick(self.main//2)
        self.tick();self.tick()
        self.assertEqual(self.actions,[])
        with patch.object(self.w,'_expected',return_value=dict(stage=9,rank=39,menu=False,route=True)):
            self.tick()
        self.assertEqual(self.w.phase,'stop')
        self.assertFalse(self.w.off_verified)
        self.assertIsNone(self.w.boss_abandon_key)

    def test_abandon_attempt_and_new_off_proof_survive_restart(self):
        action=self.prepare()
        self.w.reserve_input(action)
        resumed=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        self.assertEqual(resumed.journal['attempts']['lower-boss-abandon'],1)
        self.assertFalse(resumed.off_verified)
        self.assertIsNone(resumed.boss_abandon_key)
        self.assertFalse(resumed.validate_portal(action,self.main))
        with self.assertRaises(RuntimeError):resumed.reserve_input(action)

    def test_abandon_not_granted_to_meter_or_ordinary_cancel(self):
        self.prepare()
        self.assertIsNone(self.w.cancel_context(self.main))
        self.assertFalse(self.w.fishing.cancellable_cast(self.main))
        self.w.phase='fishing_prepare'
        self.assertFalse(self.w.meter_frame_allowed(self.main))

    def test_consent_does_not_carry_to_next_automatic_handoff_or_cycle(self):
        self.prepare()
        self.begin()
        self.assertNotIn('allow_boss_abandon',self.w.journal)
        self.assertFalse(self.w.boss_abandon_handoff)
        self.begin(cycle=True)
        self.assertFalse(self.w.boss_abandon_handoff)

    def expired(self):
        self.begin();self.w._set('stop_fishing',180)
        self.w.journal['return_deadline']=self.now-1;self.w._save()
        (self.path.parent/'config.json').write_text(json.dumps(
            dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])))
        raw=self.path.read_bytes()
        return raw,hashlib.sha256(raw).hexdigest()

    def test_hash_bound_approval_archives_exact_old_bytes_and_only_then_grants(self):
        raw,digest=self.expired()
        request=StoneFarm.request_retry(self.path.parent,digest,allow_boss_abandon=True)
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertTrue(request['allow_boss_abandon'])
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        self.assertTrue(self.w.boss_abandon_handoff)
        self.tick();self.tick()
        self.assertEqual(self.path.read_bytes(),raw)
        self.tick()
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertEqual(self.w.journal['previous_failed_sha256'],digest)
        self.assertTrue(self.w.journal['allow_boss_abandon'])
        self.assertEqual(self.actions,[])
        self.assertFalse(self.w.off_verified)

    def test_default_retry_does_not_grant_boss_permission(self):
        _,digest=self.expired()
        request=StoneFarm.request_retry(self.path.parent,digest)
        self.assertNotIn('allow_boss_abandon',request)
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        self.assertFalse(self.w.boss_abandon_handoff)

    def test_approval_cannot_be_renewed_after_attempt_or_used_for_rank40(self):
        self.expired()
        original=deepcopy(self.w.journal)
        for updates in (dict(reason='cycle',rank=40,from_stage=8,to_stage=9,entry_rank=0,rank40_verified=True),
                        dict(attempts={'lower-boss-abandon':1}),dict(phase='return')):
            with self.subTest(updates=updates):
                self.w.journal={**original,**updates};self.w._save()
                digest=hashlib.sha256(self.path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    StoneFarm.request_retry(self.path.parent,digest,allow_boss_abandon=True)


if __name__ == '__main__':unittest.main()
