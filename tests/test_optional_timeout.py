from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import shutil
import tempfile
from unittest.mock import patch
import numpy as np
from PIL import Image

from drift.runtime import Clicker
from drift.core.lifecycle import UiOwnership
from drift.tasks.optional_timeout import OptionalCleanup, apply_policy

ROOT=Path(__file__).resolve().parents[1]

class OptionalTimeoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/'config.json').read_text('utf-8-sig'))
        for r in cls.config['rules']:
            if r['id'].startswith('driftinggirl_achievements_'):r['enabled']=True
        cls.matchers={r['id']:Clicker.load_rule(SimpleNamespace(root=ROOT,config=cls.config),deepcopy(r))
                      for r in cls.config['rules'] if r['id'].startswith('driftinggirl_achievements_')}
        cls.battle=np.asarray(Image.open(ROOT/'tests/fixtures/crown-shifted-timer/battle.png').convert('RGB'))
        cls.menu=np.asarray(Image.open(ROOT/'tests/fixtures/crown-shifted-timer/menu.png').convert('RGB'))

    def test_only_optional_rewards_default_to_skip_and_saved_false_wins(self):
        for id in ('daily_open','achievements_open','mailbox_claim_all','event_potion_confirm'):
            r=dict(id='driftinggirl_'+id,next_stage='daily_tab')
            self.assertTrue(apply_policy(r)['skip_on_timeout']);self.assertNotIn('skip_on_timeout',r)
            self.assertFalse(apply_policy(dict(r,skip_on_timeout=False))['skip_on_timeout'])
        for id in ('guild_potion_buy','stone_farm','raid_start','event_potion_close','event_potion_return',
                   'mailbox_close','mailbox_limit_close','achievements_close','achievements_fold'):
            self.assertNotIn('skip_on_timeout',apply_policy(dict(id='driftinggirl_'+id,next_stage='x')))
        self.assertNotIn('skip_on_timeout',apply_policy(dict(id='driftinggirl_mailbox_open',next_stage='x',purchase_condition={'cost':1})))

    def test_cleanup_never_includes_claim_purchase_or_reopen(self):
        cleanup=OptionalCleanup('achievements_claim',0)
        selected=cleanup.rules(list(self.matchers.values()))
        self.assertTrue(selected)
        self.assertTrue(all('claim' not in r['id'] and not r['id'].endswith('_open') for r,_,_ in selected))
        self.assertEqual(cleanup.deadline,40)

    def test_real_battle_returns_only_after_three_new_frames_and_three_seconds(self):
        cleanup=OptionalCleanup('achievements_tab',0)
        self.assertFalse(cleanup.observe_return(self.battle.copy(),self.matchers,1))
        self.assertFalse(cleanup.observe_return(self.battle.copy(),self.matchers,2))
        self.assertTrue(cleanup.observe_return(self.battle.copy(),self.matchers,4.1))

    def test_dark_menu_unknown_changed_rank_and_cached_frame_cannot_prove_return(self):
        for f in (self.menu,self.battle//2,np.zeros_like(self.battle),self.battle[:900]):
            c=OptionalCleanup('achievements_tab',0)
            for t in (1,3,5):self.assertFalse(c.observe_return(f.copy(),self.matchers,t))
        c=OptionalCleanup('achievements_tab',0);f=self.battle.copy()
        c.observe_return(f,self.matchers,1);self.assertFalse(c.observe_return(f,self.matchers,5))
        altered=f.copy();altered[20:25,90:95]=255-altered[20:25,90:95]
        self.assertFalse(c.observe_return(altered,self.matchers,6))
        self.assertFalse(c.observe_return(f.copy(),self.matchers,41))

    def test_cleanup_does_not_support_a_purchase_workflow(self):
        with self.assertRaises(ValueError):OptionalCleanup('guild_potion_confirm',0)

    def test_only_guild_entrance_has_read_only_cleanup(self):
        entry = dict(id='driftinggirl_guild_open',next_stage='guild_special')
        self.assertTrue(apply_policy(entry)['skip_on_timeout'])
        self.assertFalse(apply_policy(dict(entry,skip_on_timeout=False))['skip_on_timeout'])
        self.assertNotIn('skip_on_timeout', apply_policy(dict(entry,id='driftinggirl_guild_special')))
        cleanup = OptionalCleanup('guild_special', 30, original_deadline=30)
        self.assertEqual(cleanup.prepare(list(self.matchers.values()),None), [])
        self.assertEqual(cleanup.rules(list(self.matchers.values())), [])

    def test_cleanup_has_distinct_close_rule_and_preserves_normal_absence_guard(self):
        original=deepcopy(self.config)
        c=OptionalCleanup('achievements_tab',10,original_deadline=9)
        actions=c.prepare(list(self.matchers.values()),lambda r:Clicker.load_rule(
            SimpleNamespace(root=ROOT,config=self.config),r))
        close=next(x for x in actions if x[0].get('cleanup_origin')=='driftinggirl_achievements_close')
        self.assertNotIn('absent_rule',close[0])
        self.assertEqual(self.config,original)
        self.assertEqual(self.matchers['driftinggirl_achievements_close'][0]['absent_rule'],
                         'driftinggirl_achievements_claim')
        ready=np.asarray(Image.open(ROOT/'assets/achievements-list-native.png').convert('RGB'))
        self.assertTrue(close[1].match(ready,offset=close[2],similarity=close[0]['threshold']))
        self.assertEqual(c.deadline,50)
        c.issued(close[0]['id']);c.issued(close[0]['id'])
        self.assertFalse(c.allowed(close[0]['id']))
        self.assertEqual(c.original_deadline,9)

    def test_mailbox_cleanup_matches_available_mail_but_never_claims(self):
        source=deepcopy(self.config)
        selected=[]
        for r in source['rules']:
            if r['id'].startswith('driftinggirl_mailbox_'):
                r['enabled']=True
                selected.append(Clicker.load_rule(SimpleNamespace(root=ROOT,config=source),r))
        c=OptionalCleanup('mailbox',0)
        actions=c.prepare(selected,lambda r:Clicker.load_rule(SimpleNamespace(root=ROOT,config=source),r))
        self.assertEqual(len(actions),1)
        rule,button,offset=actions[0]
        frame=np.asarray(Image.open(ROOT/'assets/mailbox-native.png').convert('RGB'))
        self.assertTrue(button.match(frame,offset=offset,similarity=rule['threshold']))
        self.assertEqual(rule['click_source_area'],[612,170,690,230])
        self.assertFalse(c.ready(rule['id'],button,frame.copy(),1))
        self.assertFalse(c.ready(rule['id'],button,frame.copy(),2))
        self.assertTrue(c.ready(rule['id'],button,frame.copy(),4))
        c.uncertain=True
        self.assertFalse(c.allowed(rule['id']))

    def test_worker_skips_real_entry_timeout_and_observes_main_without_claiming(self):
        config=deepcopy(self.config);config['poll_interval']=.1
        config['rules']=[r for r in config['rules'] if r['id'].startswith('driftinggirl_achievements_')]
        for r in config['rules']:
            r.update(enabled=True,next_delay=0)
            if r['id']=='driftinggirl_achievements_resume':r.update(stable_frames=1,stage_timeout=2)
        clock=[0.];devices=[];menu=self.menu;battle=self.battle
        class Device:
            def __init__(self,config,stop):self.stop=stop;self.reads=0;self.image=None;self.taps=[];devices.append(self)
            def screenshot(self):
                clock[0]+=1;self.reads+=1
                if self.reads>30:raise InterruptedError('bounded replay')
                self.image=(menu if self.reads==1 else np.zeros_like(menu) if self.reads<8 else battle).copy()
                return self.image
            def click(self,b):self.taps.append(b.name);raise AssertionError('Skipping must not claim or reopen')
            def stuck_record_add(self,_):pass
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            assets={r['asset'] for r in config['rules']}|{a for r in config['rules'] for a in r.get('asset_variants',[])}
            for name in assets:
                (root/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,root/name)
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            app=Clicker(root,Device)
            try:
                with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])):
                    app.start(preview=False,stop_on_main=True,timeout=60);app.thread.join(15)
                self.assertFalse(app.running());self.assertIsNone(app.state['last_error'])
                self.assertEqual(app.state['stop_reason'],'cycle_complete')
                self.assertEqual(devices[0].taps,[])
                self.assertTrue(any('跳过本次' in e['message'] for e in app.events))
                self.assertFalse(any(x.get('completed') for x in app.daily_schedule.status().values()))
            finally:
                app.stop()
                if app.thread:app.thread.join(2)
                for h in list(app.log.handlers):h.close();app.log.removeHandler(h)

    def test_deadline_at_adb_dispatch_retires_the_optional_step(self):
        config=deepcopy(self.config);config['poll_interval']=.1
        config['rules']=[r for r in config['rules'] if r['id'].startswith('driftinggirl_achievements_')]
        for rule in config['rules']:
            rule.update(enabled=True,next_delay=0)
        clock=[0.]
        attempted=[]
        inputs=[]
        menu,battle=self.menu,self.battle
        original=UiOwnership.authorize

        def authorize(ui,kind,args):
            if ui.current_actor()=='rule:driftinggirl_achievements_crown':
                deadline=ui.budget()['stage_deadline_epoch']
                attempted.append(deadline)
                clock[0]=deadline-ui.checkpoint.clock_offset+.1
                ui.checkpoint.wall=lambda:ui.checkpoint.clock_offset+clock[0]
            return original(ui,kind,args)

        class Device:
            def __init__(self,config,stop):self.image=None;self.count=0
            def screenshot(self):
                clock[0]+=1
                self.count+=1
                if self.count>50:raise InterruptedError('bounded replay')
                self.image=(battle if attempted else menu).copy()
                return self.image
            def click(self,button):
                token=self.input_guard('adb',('shell','input','tap'))
                inputs.append(button.name)
                self.input_issued(token)
            def stuck_record_add(self,button):pass

        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            files={p for r in config['rules'] for p in [r['asset'],*r.get('asset_variants',[])]}
            for relative in files:
                target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/relative,target)
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            app=Clicker(root,Device)
            try:
                with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])), \
                     patch.object(UiOwnership,'authorize',authorize):
                    app.start(preview=False,stop_on_main=True,timeout=120)
                    app.thread.join(15)
                self.assertFalse(app.running())
                self.assertIsNone(app.state['last_error'])
                self.assertEqual(app.state['stop_reason'],'cycle_complete')
                self.assertEqual(len(attempted),1)
                self.assertEqual(inputs,[])
                records=[json.loads(p.read_bytes()) for p in (root/'logs/workflow-context-retired').glob('*.json')]
                self.assertTrue(any(r.get('last_input',{}).get('actor')=='rule:driftinggirl_achievements_crown'
                                    and r['last_input']['state']=='reserved'
                                    and r['stage']['budget']['stage_deadline_epoch']==attempted[0] for r in records))
            finally:
                app.stop()
                if app.thread:app.thread.join(2)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)
