"""An explicitly approved extra menu contact retains the first native attempt."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

from digit_ocr import DigitReader
from drift.tasks.stone_farm import StoneFarm
from tests import test_stone_boss_abandon as helpers

ROOT, image = helpers.ROOT, helpers.image


class LowerBossMenuTests(unittest.TestCase):
    setUp = helpers.LowerBossAbandonTests.setUp
    tearDown = helpers.LowerBossAbandonTests.tearDown
    act = helpers.LowerBossAbandonTests.act
    tick = helpers.LowerBossAbandonTests.tick
    begin = helpers.LowerBossAbandonTests.begin
    action = helpers.LowerBossAbandonTests.action
    prepare = helpers.LowerBossAbandonTests.prepare

    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)
        cls.main = image('tests/fixtures/stone-farm/lower-seagod-battle.png')

    def prior_contact(self):
        action = self.prepare()
        self.w.reserve_input(action)
        self.receipt = dict(input_id='a'*32,frame_id='b'*32,kind='lower-boss-abandon',
            controller='stone_farm',rank=39,completed_rank_40=False,
            x=611,y=184,native_xy=[611,184],input_transport='mumu-native',hold_ms=40,
            input_begin_time=1020.,touch_down_time=1020.001,touch_up_time=1020.041,
            capture_age_ms=28.25)
        self.w.input_issued(action,self.receipt)
        self.now=1200.
        self.w._save()
        (self.path.parent/'config.json').write_text(json.dumps(
            dict(rules=[dict(id='driftinggirl_stone_farm',enabled=True)])),encoding='utf-8')
        self.receipt_path=self.path.parent/'logs/portal-input'/('input-'+'a'*32+'.json')
        self.receipt_path.parent.mkdir(parents=True)
        self.write_receipt(self.receipt)
        return self.failed()

    def write_receipt(self, data):
        self.receipt_path.write_text(json.dumps(data),encoding='utf-8')

    def failed(self):
        raw=self.path.read_bytes()
        return raw,hashlib.sha256(raw).hexdigest()

    def approve(self):
        return StoneFarm.request_retry(self.path.parent,self.failed()[1],allow_boss_abandon_menu=True)

    def resume(self):
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        self.tick();self.tick();self.tick()

    def test_new_permission_is_explicit_and_default_retry_cannot_erase_first_attempt(self):
        raw,digest=self.prior_contact()
        for options in ({},{'allow_boss_abandon':True}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                StoneFarm.request_retry(self.path.parent,digest,**options)
        self.assertEqual(self.path.read_bytes(),raw)
        approval=self.approve()
        self.assertEqual(approval['mode'],'boss_abandon_menu_retry')
        self.assertTrue(approval['allow_boss_abandon_menu'])
        self.assertEqual(self.path.read_bytes(),raw)

    def test_missing_unreleased_stale_or_unrelated_receipt_is_rejected(self):
        self.prior_contact()
        bad = [None,[],{**self.receipt,'touch_up_time':None},
            {**self.receipt,'touch_up_time':1019.}, {**self.receipt,'capture_age_ms':121},
            {**self.receipt,'frame_id':'c'*32}, {**self.receipt,'input_transport':'adb'},
            {**self.receipt,'uncertain_down':True}, {**self.receipt,'x':600},
            {**self.receipt,'touch_down_time':1019.99}, {**self.receipt,'rank':40}]
        for data in bad:
            with self.subTest(data=data):
                if data is None:self.receipt_path.unlink(missing_ok=True)
                else:self.write_receipt(data)
                with self.assertRaises(ValueError):self.approve()
        self.assertEqual(self.actions,[])

    def test_wrong_scope_or_both_permissions_do_not_grant_extra_contact(self):
        self.prior_contact()
        saved=deepcopy(self.w.journal)
        for updates in (dict(rank=40),dict(from_stage=8),dict(to_stage=9),
                        dict(allow_boss_abandon=False),dict(phase='return'),
                        dict(boss_abandon_menu_retry=True),dict(attempts={})):
            with self.subTest(updates=updates):
                self.w.journal={**saved,**updates};self.w._save()
                with self.assertRaises(ValueError):self.approve()
        self.w.journal=saved;self.w._save()
        with self.assertRaises(ValueError):
            StoneFarm.request_retry(self.path.parent,self.failed()[1],
                                    allow_boss_abandon=True,allow_boss_abandon_menu=True)

    def test_new_observation_archives_original_and_retains_all_original_inputs(self):
        raw,digest=self.prior_contact()
        prior=deepcopy(self.w.journal)
        approval=self.approve()
        self.resume()
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        for key in ('attempts','inputs','issued_inputs','started_at','cycles','handoffs'):
            self.assertEqual(self.w.journal[key],prior[key],key)
        self.assertEqual(self.w.journal['previous_failed_sha256'],digest)
        self.assertEqual(self.w.journal['native_abandon_receipt_sha256'],approval['native_abandon_receipt_sha256'])
        self.assertTrue(self.w.journal['boss_abandon_menu_retry'])
        self.assertEqual(self.w.phase,'stop')
        self.assertFalse(self.w.off_verified)
        self.assertEqual(self.actions,[])

    def test_receipt_change_after_approval_prevents_consuming_new_input(self):
        raw,_=self.prior_contact();self.approve()
        self.write_receipt({**self.receipt,'capture_age_ms':29})
        self.w=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        self.tick();self.tick()
        with self.assertRaisesRegex(RuntimeError,'回执已变化'):self.tick()
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual(self.actions,[])

    def test_menu_input_requires_new_off_battle_proof_and_cannot_repeat(self):
        self.prior_contact();self.approve();self.resume()
        for _ in range(4):self.tick()
        self.assertEqual(self.actions,[])
        self.tick()
        self.assertEqual(len(self.actions),1)
        action=self.actions[0]
        self.assertEqual(action['kind'],'lower-boss-abandon-menu')
        self.assertEqual((action['x'],action['y']),(611,184))
        self.assertFalse(self.w.fishing_stopped)
        self.assertEqual(self.w.phase,'stop_fishing')
        self.w.reserve_input(action)
        for _ in range(4):self.tick()
        self.assertEqual(len(self.actions),1)
        self.assertEqual(self.w.journal['attempts'],
                         {'lower-boss-abandon':1,'lower-boss-abandon-menu':1})
        resumed=StoneFarm(ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        self.assertFalse(resumed.off_verified)
        self.assertEqual(resumed.journal['attempts'],self.w.journal['attempts'])
        self.assertFalse(resumed.validate_portal(action,self.main))
        self.now=self.w.journal['return_deadline']+1
        for options in ({},{'allow_boss_abandon_menu':True},{'allow_boss_abandon':True}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                StoneFarm.request_retry(self.path.parent,self.failed()[1],**options)

    def test_menu_contact_keeps_scope_coordinates_stop_deadline_and_meter_guards(self):
        self.prior_contact();self.approve();self.resume()
        self.w._set('stop_fishing',180);self.w.off_verified=True
        context=self.w.boss_abandon_context(self.main);self.w.boss_abandon_key=context
        action=self.action('lower-boss-abandon-menu',(611,184));action['abandon_context']=context
        self.assertTrue(self.w.validate_portal(action,self.main))
        self.assertFalse(self.w.validate_portal({**action,'kind':'lower-boss-abandon'},self.main))
        self.assertFalse(self.w.validate_portal({**action,'x':610},self.main))
        changed=self.main.copy();changed[158:210,584:642]=0
        self.assertFalse(self.w.validate_portal(action,changed))
        self.assertFalse(self.w.validate_portal(action,self.main//2))
        with patch.object(self.w.sailing,'running',return_value=True):
            self.assertFalse(self.w.validate_portal(action,self.main))
        self.assertIsNone(self.w.cancel_context(self.main))
        self.w.phase='fishing_prepare'
        self.assertFalse(self.w.meter_frame_allowed(self.main))
        self.w.phase='stop_fishing';self.now=self.w.journal['return_deadline']+1
        self.assertFalse(self.w.validate_portal(action,self.main))

    def test_extra_permission_does_not_carry_into_automatic_next_handoff(self):
        self.prior_contact();self.approve();self.resume();self.begin()
        self.assertNotIn('boss_abandon_menu_retry',self.w.journal)
        self.assertFalse(self.w.boss_abandon_handoff)
        self.assertEqual(self.w.journal['attempts'],{})


if __name__ == '__main__':unittest.main()
