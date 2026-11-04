"""Real 1.16k teleport quote and a bounded continuation of a failed trip."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
import cv2
from drift.tasks.guild_daily import GuildDaily
from drift.perception.digit_ocr import DigitReader,NumberReading

ROOT=Path(__file__).resolve().parents[1]
def frame():
    return cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/guild-teleport-k/quote.png')),cv2.COLOR_BGR2RGB)

class TeleportKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.reader=DigitReader(ROOT)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.now=1000.;self.actions=[]
        self.c=GuildDaily(ROOT,self.reader,lambda a:self.actions.append(a) or False,lambda a:None,
            'serial',clock=lambda:self.now,state_path=Path(self.temp.name)/'guild_daily_state.json')
        self.f=frame()
    def prior(self):
        settings=dict(pearl=True,material=False,amber=True,gem=True,boss=True,sale=True,upgrade=True,priority='rare')
        origin=dict(parallel=True,stage=9,rank=38,route=True,fishing=True)
        return dict(version=1,serial='serial',period='2026-10-30',pending=False,phase='idle',
            failure='公会往返guild_prompt阶段超过原有时限',started_at=100.,deadline=1900.,phase_deadline=400.,
            last_input=dict(kind='direct',state='issued_only',input_id='owned'),
            origin=origin,target=dict(origin,rank=35,settings=settings),settings=settings,
            attempts={'stop':1,'cancel':2,'return-request':1,'return-confirm':1,'direct':1})
    def archive(self,record):
        raw=(json.dumps(record,ensure_ascii=False)+'\n').encode();digest=hashlib.sha256(raw).hexdigest()
        path=Path(self.temp.name)/'logs/guild-daily-retired'/(digest+'.json')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        return path,raw
    def active(self):
        self.c.journal=dict(version=1,serial='serial',period='2026-10-30',pending=True,phase='stop',
            origin=None,target=self.prior()['target'],farm_origin=self.prior()['target'],
            attempts={},deadline=2800.,phase_deadline=1180.)
        self.c.phase='stop';self.c._save()
    def test_real_k_balance_uses_conservative_floor(self):
        self.assertEqual(self.c.teleport_quote(self.f),dict(balance=1150,cost=10))
        self.assertFalse(self.c.visual.teleport_prompt(self.f//2))
        self.assertIsNone(self.c.teleport_quote(self.f//2))
    def test_disagreeing_low_confidence_or_malformed_balances_never_confirm(self):
        good=lambda s,c=.99:NumberReading(s,None,c)
        for values in ((good('1.16k'),good('1.17k')),(good('1.16k',.5),good('1.16k')),
                       (good('1.16'),good('1.16')),(good('1.16m'),good('1.16m'))):
            with patch.object(self.c.reader,'read_many',return_value=values):
                self.assertIsNone(self.c.teleport_quote(self.f))

    def test_real_distance_price_is_read_and_rechecked_instead_of_assumed_ten(self):
        f=cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/guild-teleport-k/quote-50.png')),cv2.COLOR_BGR2RGB)
        self.assertEqual(self.c.teleport_quote(f),dict(balance=1150,cost=50))
        self.active();self.c.phase='guild_prompt';self.c.journal['phase']='guild_prompt'
        self.c.journal['teleport_quote']=dict(balance=1150,cost=10)
        p=self.c.visual.point(f,'teleport-continue')
        a=dict(kind='teleport-confirm',phase='guild_prompt',x=p[0],y=p[1])
        self.assertFalse(self.c.validate(a,f))
        self.c.journal['teleport_quote']=dict(balance=1150,cost=50)
        self.assertTrue(self.c.validate(a,f))
        good=lambda s,c=.99:NumberReading(s,None,c)
        for price in ((good('50'),good('10')),(good('50',.5),good('50')),(good('5.0'),good('5.0'))):
            with patch.object(self.c.reader,'read_many',return_value=[good('1.16k'),good('1.16k'),*price]):
                self.assertIsNone(self.c.teleport_quote(f))
    def test_new_retry_adopts_prompt_without_renewing_its_deadlines_or_spent_inputs(self):
        old=self.prior();path,raw=self.archive(old);self.active()
        for now in (1000,1002,1004):
            self.now=now;self.assertTrue(self.c._adopt_teleport_prompt(self.f.copy(),'2026-10-30'))
        self.assertEqual(self.c.phase,'guild_prompt')
        self.assertEqual(self.c.journal['deadline'],2800.)
        self.assertLessEqual(self.c.journal['phase_deadline'],1180.)
        self.assertEqual(self.c.journal['attempts'],old['attempts'])
        self.assertEqual(self.c.journal['target']['rank'],38)
        self.assertEqual(path.read_bytes(),raw)
        self.assertEqual(self.actions,[])
    def test_used_confirmation_different_period_or_serial_cannot_adopt(self):
        for change in ({'attempts':{'teleport-confirm':1}},{'period':'2026-10-29'},{'serial':'another'}):
            old=self.prior();old.update(change);path,_=self.archive(old);self.active()
            for now in (1000,1002,1004):
                self.now=now;self.assertIsNone(self.c._adopt_teleport_prompt(self.f.copy(),'2026-10-30'))
            self.assertEqual(self.c.phase,'stop');path.unlink()
    def test_expired_current_window_never_uses_the_checkpoint_to_input(self):
        self.archive(self.prior());self.active();self.now=1181
        before=deepcopy(self.c.journal)
        with self.assertRaises(RuntimeError):self.c.step(self.f,due=False,period='2026-10-30')
        self.assertEqual(self.c.journal['attempts'],before['attempts'])
        self.assertEqual(self.c.journal['deadline'],before['deadline'])
        self.assertEqual(self.actions,[])
    def test_confirmation_reservation_and_fresh_quote_remain_required(self):
        self.archive(self.prior());self.active()
        for now in (1000,1002,1004):
            self.now=now;self.c._adopt_teleport_prompt(self.f.copy(),'2026-10-30')
        self.c.journal['teleport_quote']=self.c.teleport_quote(self.f)
        point=self.c.visual.point(self.f,'teleport-continue')
        action=dict(kind='teleport-confirm',phase='guild_prompt',x=point[0],y=point[1])
        self.assertTrue(self.c.validate(action,self.f))
        self.assertFalse(self.c.validate(action,self.f//2))
        with patch.object(self.c,'teleport_quote',return_value=dict(balance=0,cost=10)):
            self.assertFalse(self.c.validate(action,self.f))
        self.assertTrue(self.c.reserve(action));self.assertFalse(self.c.reserve(action))
        self.assertEqual(self.c.journal['attempts']['cancel'],2)

    def test_matching_archived_departure_recovers_actual_home_without_prompt(self):
        old=self.prior();path,raw=self.archive(old)
        target=self.c._departure_target('2026-10-30',deepcopy(old['target']))
        self.assertEqual(target['rank'],38)
        self.assertEqual(target['settings'],old['settings'])
        self.assertEqual(path.read_bytes(),raw)
        self.assertEqual(self.c._departure_target('2026-10-29',old['target']),old['target'])
        other=dict(old['target'],rank=34)
        self.assertEqual(self.c._departure_target('2026-10-30',other),other)
        path.write_bytes(raw+b' ')
        self.assertEqual(self.c._departure_target('2026-10-30',old['target']),old['target'])

    def test_retired_collection_retry_preserves_inputs_and_only_returns_home(self):
        old=self.prior();self.archive(old)
        spent={'stop':1,'snapshot-open':1,'snapshot-close':1,'teleport-confirm':1,
               'submit:0:claimed':1,'scroll:0':2}
        self.c.journal=dict(version=1,serial='serial',period='2026-10-30',phase='idle',pending=False,
            farm_origin=deepcopy(old['target']),resume_pending=True,farm_restored_at=None,
            target=deepcopy(old['target']),attempts=spent.copy(),categories=[],submitted=4,
            retirement=dict(previous_phase='collect',sha256='old-bytes',archive='immutable'))
        previous=deepcopy(self.c.journal)
        self.c.origin_screen=Mock(return_value=dict(parallel=False,route=False,at_guild=False))
        self.c.same_origin=Mock(return_value=True)
        self.c._confirmed=Mock(return_value=True)
        self.c.sailing.running=Mock(return_value=False)
        self.assertTrue(self.c.step(self.f,due=True,period='2026-10-30'))
        self.assertTrue(self.c.journal['return_only'])
        self.assertEqual(self.c.return_target()['rank'],38)
        self.assertEqual(self.c.journal['attempts'],spent)
        self.assertEqual(self.c.journal['submitted'],4)
        self.assertEqual(self.c.journal['retry_of'],previous['retirement'])
        self.assertEqual(self.actions,[])
        for kind in ('direct','teleport-confirm','submit:1:new','tab-tasks','g-icon'):
            self.assertFalse(self.c.validate(dict(kind=kind),self.f))
        self.c.phase='stop_fishing';self.c.journal['phase']='stop_fishing'
        self.c.view._idle_fishing=Mock(return_value=True)
        self.c.view._off_icon=Mock(return_value=True)
        self.c.same_origin=Mock(return_value=False)
        self.c.step(self.f,due=False,period='2026-10-30')
        self.assertEqual(self.c.phase,'map_open')
        self.assertEqual(self.c.journal['map_goal'],'observatory')
        self.assertEqual(self.c.journal['attempts'],spent)
        self.assertEqual(self.actions,[])

    def test_return_only_adopts_real_leftover_cruise_menu_without_input(self):
        f=cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/guild-teleport-k/leftover-cruise.png')),cv2.COLOR_BGR2RGB)
        self.active();self.c.journal['return_only']=True
        self.c.journal['target']=dict(self.prior()['target'],rank=38)
        original_deadline=self.c.journal['deadline'];old_phase_deadline=self.c.journal['phase_deadline']
        self.assertTrue(self.c.sailing.menu(f));self.assertTrue(self.c.view._off_icon(f))
        for now in (1000,1002,1004):
            self.now=now;self.c.step(f.copy(),due=False,period='2026-10-30')
        self.assertEqual(self.c.phase,'snapshot')
        self.assertEqual(self.c.journal['deadline'],original_deadline)
        self.assertLessEqual(self.c.journal['phase_deadline'],old_phase_deadline)
        self.assertEqual(self.actions,[])

    def test_leftover_cruise_menu_requires_return_only_and_confirmed_options(self):
        f=cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/guild-teleport-k/leftover-cruise.png')),cv2.COLOR_BGR2RGB)
        self.active()
        for now in (1000,1002,1004):
            self.now=now;self.c.step(f.copy(),due=False,period='2026-10-30')
        self.assertEqual(self.c.phase,'stop')
        self.c.journal['return_only']=True
        with patch.object(self.c,'settings',return_value=None):
            for now in (1006,1008,1010):
                self.now=now;self.c.step(f.copy(),due=False,period='2026-10-30')
        self.assertEqual(self.c.phase,'stop')
        self.assertEqual(self.actions,[])

    def test_return_map_adoption_keeps_prior_scans_and_original_new_retry_deadline(self):
        f=cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/guild-teleport-k/leftover-map.png')),cv2.COLOR_BGR2RGB)
        self.active();self.c.journal['return_only']=True
        self.c.journal['attempts']['map-scan:observatory']=3
        self.assertTrue(self.c.stopped_map(f));self.assertFalse(self.c.bag.moving(f))
        for now in (1000,1002,1004):
            self.now=now;self.c.step(f.copy(),due=False,period='2026-10-30')
        self.assertEqual(self.c.phase,'map_select')
        self.assertEqual(self.c.journal['map_scans'],3)
        self.assertEqual(self.c.journal['attempts']['map-scan:observatory'],3)
        self.assertEqual(self.c.journal['deadline'],2800.)
        self.assertLessEqual(self.c.journal['phase_deadline'],1180.)
        self.assertEqual(self.actions,[])
        for _ in range(13):self.assertTrue(self.c.reserve(dict(kind='map-scan:observatory')))
        self.assertFalse(self.c.reserve(dict(kind='map-scan:observatory')))
        self.assertEqual(self.c.journal['attempts']['map-scan:observatory'],16)
        self.assertEqual(self.c.journal['deadline'],2800.)

    def test_return_map_still_waits_while_moving_or_without_a_confirmed_map(self):
        f=cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/guild-teleport-k/leftover-map.png')),cv2.COLOR_BGR2RGB)
        for moving,stopped in ((True,True),(False,False)):
            self.active();self.c.journal['return_only']=True
            with patch.object(self.c,'stopped_map',return_value=stopped),patch.object(self.c.bag,'moving',return_value=moving):
                for now in (1000,1002,1004):
                    self.now=now;self.c.step(f.copy(),due=False,period='2026-10-30')
            self.assertEqual(self.c.phase,'stop')
            self.assertEqual(self.actions,[])

    def test_observatory_search_continues_west_after_six_within_same_limit(self):
        self.active();self.c.phase='map_select';self.c.journal['phase']='map_select'
        self.c.journal.update(map_goal='observatory',map_scans=6)
        self.c.stopped_map=Mock(return_value=True);self.c.map_target=Mock(return_value=None)
        self.c.step(self.f,due=False,period='2026-10-30')
        self.assertEqual(self.actions[-1]['x'],200)
        self.assertEqual(self.actions[-1]['swipe'],[520,650,400])
        self.assertEqual(self.c.journal['map_scans'],6)
        self.actions.clear();self.c.journal['map_goal']='nearby'
        self.c.step(self.f,due=False,period='2026-10-30')
        self.assertEqual(self.actions[-1]['x'],520)
        self.assertEqual(self.c.input_limit('map-scan:observatory'),16)

    def test_observatory_arrival_wait_keeps_the_existing_total_deadline(self):
        self.active();self.c.phase='map_select';self.c.journal['phase']='map_select'
        self.c.journal.update(map_goal='observatory',map_scans=16,deadline=1500.)
        self.c.stopped_map=Mock(return_value=True);self.c.map_target=Mock(return_value=(300,700))
        self.c.action=Mock(return_value=True)
        self.c.step(self.f,due=False,period='2026-10-30')
        self.assertEqual(self.c.phase,'map_close')
        self.assertEqual(self.c.journal['phase_deadline'],1500.)
        self.assertEqual(self.c.journal['deadline'],1500.)
        self.assertEqual(self.c.journal['map_scans'],16)

    def test_return_only_at_observatory_skips_departure_settings_and_map_reopen(self):
        self.active();self.c.journal['return_only']=True
        self.c.journal['origin']=dict(parallel=False,route=False,at_guild=False)
        self.c.journal['attempts'].update({'snapshot-open':2,'snapshot-close':2,'map-scan:observatory':16})
        self.c.same_origin=Mock(return_value=True);self.c._confirmed=Mock(return_value=True)
        self.c.sailing.running=Mock(return_value=False)
        self.c.step(self.f,due=False,period='2026-10-30')
        self.assertEqual(self.c.phase,'stop_fishing')
        self.c.view._idle_fishing=Mock(return_value=True);self.c.view._off_icon=Mock(return_value=True)
        self.c.same_origin=Mock(return_value=False);self.c.view._npc_ready=Mock(return_value=True)
        self.c.step(self.f.copy(),due=False,period='2026-10-30')
        self.assertEqual(self.c.phase,'gate')
        self.assertEqual(self.c.journal['attempts']['snapshot-open'],2)
        self.assertEqual(self.c.journal['attempts']['map-scan:observatory'],16)
        self.assertEqual(self.actions,[])

    def test_remaining_collection_at_actual_guild_preserves_spent_counts_and_blocks_repayment(self):
        old=self.prior();self.archive(old)
        spent={'g-icon':2,'teleport-confirm':1,'submit:0:claimed':1}
        self.c.journal=dict(version=1,serial='serial',period='2026-10-30',phase='idle',pending=False,
            farm_origin=deepcopy(old['target']),resume_pending=True,farm_restored_at=None,
            target=deepcopy(old['target']),attempts=spent.copy(),categories=[],submitted=4,
            retirement=dict(previous_phase='collect',sha256='original'))
        self.c.origin_screen=Mock(return_value=dict(parallel=False,route=False,at_guild=True))
        self.c._confirmed=Mock(return_value=True);self.c.normal_idle=Mock(return_value=False)
        self.c.step(self.f,due=True,period='2026-10-30')
        self.assertEqual(self.c.phase,'guild_at');self.assertTrue(self.c.journal['collect_here'])
        self.assertEqual(self.c.journal['attempts'],spent)
        self.assertEqual(self.c.return_target()['rank'],38)
        for kind in ('direct','teleport-confirm'):
            self.assertFalse(self.c.validate(dict(kind=kind),self.f))
        self.assertEqual(self.actions,[])

    def test_real_return_success_rejects_preparation_dark_frames_and_green_rectangle(self):
        def load(name):return cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/guild-teleport-k'/name)),cv2.COLOR_BGR2RGB)
        for name in ('return-success.png','return-success-heldout.png'):
            f=load(name);self.assertTrue(self.c.visual.restore_success(f));self.assertFalse(self.c.visual.restore_success(f//2))
        f=load('return-preparation.png');self.assertFalse(self.c.visual.restore_success(f))
        f[985:1040,310:410]=(100,200,40);self.assertFalse(self.c.visual.restore_success(f))

    def finished_handin(self):
        target=dict(self.prior()['target'],rank=38)
        self.c.journal=dict(version=1,serial='serial',period='2026-10-30',pending=False,phase='idle',
            target=target,farm_origin=deepcopy(target),resume_pending=True,categories=[0,1],submitted=9,
            resume_settings=deepcopy(target['settings']),deadline=900.,phase_deadline=800.,
            attempts={'cast':1,'route-start':1,'map-scan:observatory':16})
        self.c._save();self.c.same_origin=Mock(return_value=True)
        self.c.restored_settings=Mock(return_value=True);self.c.sailing.running=Mock(return_value=True)
        self.c.fishing.blue_fill=Mock(return_value=False);self.c.fishing.real_battle=Mock(return_value=False)
        return self.c.path.read_bytes()

    def test_read_only_finish_after_timeout_preserves_old_deadlines_and_spent_inputs(self):
        old=self.finished_handin();spent=deepcopy(self.c.journal['attempts'])
        f=cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/guild-teleport-k/return-success.png')),cv2.COLOR_BGR2RGB)
        for now in (1000,1002,1004):
            self.now=now;self.c.step(f.copy(),due=True,period='2026-10-30')
        self.assertTrue(self.c.done);self.assertFalse(self.c.journal['resume_pending'])
        self.assertEqual(self.c.journal['deadline'],900.);self.assertEqual(self.c.journal['phase_deadline'],800.)
        self.assertEqual(self.c.journal['attempts'],spent);self.assertEqual(self.actions,[])
        digest=hashlib.sha256(old).hexdigest()
        self.assertEqual((self.c.path.parent/'logs/guild-daily-retired'/(digest+'.json')).read_bytes(),old)

    def test_running_auto_or_wrong_rank_never_finishes_or_replays_the_failed_return(self):
        self.finished_handin()
        f=cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/guild-teleport-k/return-preparation.png')),cv2.COLOR_BGR2RGB)
        for now in (1000,1002,1004):
            self.now=now;self.c.step(f.copy(),due=True,period='2026-10-30')
        self.assertIsNone(self.c.done);self.assertEqual(self.actions,[])
        self.c.same_origin.return_value=False
        f=cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/guild-teleport-k/return-success.png')),cv2.COLOR_BGR2RGB)
        self.c.step(f,due=True,period='2026-10-30')
        self.assertIsNone(self.c.done);self.assertEqual(self.actions,[])

if __name__=='__main__':unittest.main()
