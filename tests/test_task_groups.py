"""Group navigation must preserve task identities, scheduling and exclusive portal policy."""
from copy import deepcopy
import json
from http.server import ThreadingHTTPServer
import re
import threading
import unittest
from unittest.mock import patch
import urllib.request
import urllib.error

from tests import test_features
from tests import test_clicker as helpers
from app import make_handler
from drift.tasks.features import TASK_GROUPS, FEATURES, feature_states, execution_mode


class TaskGroupTests(unittest.TestCase):
    setUp = helpers.FrameworkTests.setUp
    tearDown = helpers.FrameworkTests.tearDown
    rule = helpers.FrameworkTests.rule
    feature = test_features.FeatureTests.feature

    def setup_modes(self):
        self.feature('stone_farm')
        self.feature('parallel_world')[0]['training']={'final_boss_retry_level_gap':0}
        self.feature('equipment_upgrade')[0]['interval']=900
        self.feature('parallel_skills')
        self.app.config['return_parallel_enabled']=True
        self.app.persist()

    def test_eight_groups_cover_every_feature_once_and_weapon_belongs_to_stone(self):
        ids=[f for g in TASK_GROUPS for f in g['features']]
        self.assertEqual(len(TASK_GROUPS),8)
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{f['id'] for f in FEATURES})
        stone=next(g for g in TASK_GROUPS if g['id']=='stone')
        self.assertIn('equipment_upgrade',stone['features'])
        restaurant=next(g for g in TASK_GROUPS if g['id']=='restaurant')
        self.assertEqual(restaurant['features'],('restaurant_upgrade',))
        self.assertEqual(execution_mode('equipment_upgrade'),'timed')
        self.assertEqual(execution_mode('meter_calibration'),'realtime')
        self.assertEqual(execution_mode('pass_daily'),'daily')

    def test_existing_dual_flags_are_reported_without_migrating_config(self):
        self.setup_modes();before=deepcopy(self.app.config)
        features={f['id']:f for f in feature_states(self.app.config)}
        self.assertTrue(features['parallel_world']['enabled'])
        self.assertFalse(features['parallel_world']['effective_enabled'])
        self.assertTrue(features['stone_farm']['effective_enabled'])
        self.assertEqual(self.app.config,before)

    def test_mode_save_is_atomic_and_preserves_all_other_settings_and_records(self):
        self.setup_modes();before=deepcopy(self.app.config)
        paths=['training_state.json','stone_farm_state.json','task_schedule.json','parallel_skill_state.json']
        for name in paths:(self.root/name).write_bytes(b'{"pending":true,"attempts":2,"deadline":1}')
        for mode in ('legacy','off','stone'):
            self.app.change_portal_mode({'mode':mode})
            saved=json.loads((self.root/'config.json').read_text(encoding='utf-8'))
            expected=deepcopy(before)
            for r in expected['rules']:
                if r['id'] in ('driftinggirl_stone_farm','driftinggirl_parallel_world'):
                    r['enabled']=r['id']=={'legacy':'driftinggirl_parallel_world','stone':'driftinggirl_stone_farm','off':None}[mode]
            self.assertEqual(saved,expected)
            for name in paths:self.assertEqual((self.root/name).read_bytes(),b'{"pending":true,"attempts":2,"deadline":1}')

    def test_rejected_or_failed_mode_save_changes_nothing(self):
        self.setup_modes();before=deepcopy(self.app.config)
        for data in ({'mode':'bad'},{'mode':'stone','enabled':True},{}, {'mode':None}):
            with self.assertRaises(ValueError):self.app.change_portal_mode(data)
            self.assertEqual(self.app.config,before)
        with patch.object(self.app,'persist',side_effect=OSError('full')):
            with self.assertRaises(OSError):self.app.change_portal_mode({'mode':'off'})
        self.assertEqual(self.app.config,before)
        with patch.object(self.app,'running',return_value=True):
            with self.assertRaises(ValueError):self.app.change_portal_mode({'mode':'legacy'})
        self.assertEqual(self.app.config,before)

    def test_incomplete_mode_never_disables_the_other_half(self):
        self.feature('stone_farm');before=deepcopy(self.app.config)
        with self.assertRaises(ValueError):self.app.change_portal_mode({'mode':'legacy'})
        self.assertEqual(self.app.config,before)

    def test_http_group_inventory_and_authenticated_mode_change(self):
        self.setup_modes()
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(self.app))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(base+'/') as r:html=r.read().decode()
            token=re.search(r"const TOKEN='([0-9a-f]+)'",html).group(1)
            with urllib.request.urlopen(base+'/api/status') as r:state=json.load(r)
            self.assertEqual(len(state['task_groups']),8)
            payload=b'{"mode":"stone"}'
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(base+'/api/portal-mode',data=payload))
            self.assertEqual(error.exception.code,403);error.exception.close()
            request=urllib.request.Request(base+'/api/portal-mode',data=payload,headers={'X-Session-Token':token})
            with urllib.request.urlopen(request) as r:self.assertEqual(json.load(r)['result'],{'mode':'stone'})
            self.assertFalse(next(r for r in self.app.config['rules'] if r['id']=='driftinggirl_parallel_world')['enabled'])
        finally:server.shutdown();server.server_close();thread.join(2)


if __name__=='__main__':unittest.main()
