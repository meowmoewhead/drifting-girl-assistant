"""A popup-consumed first-map input keeps all old budgets during explicit repair."""
from copy import deepcopy
import hashlib,json
import unittest

from drift.tasks.stone_farm import StoneFarm
from tests import test_stone_farm as base

class PopupMapRestoreTests(unittest.TestCase):
    setUpClass=classmethod(base.StoneFarmTests.setUpClass.__func__)
    setUp=base.StoneFarmTests.setUp
    tearDown=base.StoneFarmTests.tearDown
    act=base.StoneFarmTests.act
    tick=base.StoneFarmTests.tick
    begin=base.StoneFarmTests.begin

    def failed(self):
        self.begin()
        frame=base.image('tests/fixtures/task-popup-idle/returned.png')
        self.w.journal.update(verified_gate_stage=8,entered_verified_at=self.now-10,
            first_deadline=self.now-1,phase='first_map',phase_deadline=self.now-1,
            first_recovery=True,first_unselected=True,first_signature=self.w.signature(frame),
            attempts={'lower':1,'enter-confirm':1,'first-map':1},
            issued_inputs=[dict(kind=k) for k in ('lower','enter-confirm','first-map')])
        self.w.phase='first_map';self.w._save()
        (self.path.parent/'config.json').write_text(json.dumps(dict(rules=[
            dict(id='driftinggirl_stone_farm',enabled=True)])),encoding='utf-8')
        return frame

    def test_explicit_repair_archives_old_record_and_preserves_used_map_input(self):
        frame=self.failed();raw=self.path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        request=StoneFarm.request_retry(self.path.parent,digest)
        self.assertEqual(request['mode'],'first_map_restore')
        self.assertTrue(request['map_open_interrupted'])
        self.assertEqual(self.path.read_bytes(),raw)
        self.w=StoneFarm(base.ROOT,self.reader,self.act,lambda _:None,state_path=self.path)
        for i in range(3):
            self.tick(frame)
            if i<2:self.assertEqual(self.actions,[])
        self.assertEqual((self.path.parent/'logs/stone-farm-failed'/f'{digest}.json').read_bytes(),raw)
        self.assertEqual(self.w.journal['attempts'],json.loads(raw)['attempts'])
        self.assertEqual(self.w.journal['issued_inputs'],json.loads(raw)['issued_inputs'])
        self.assertEqual(self.w.journal['first_map_repair_runs'],1)
        self.assertEqual(self.w.phase,'first_map')
        self.assertEqual([a['kind'] for a in self.actions],['first-map'])
        self.assertEqual([(a['x'],a['y']) for a in self.actions],[(350,30)])
        self.assertAlmostEqual(self.w.journal['deadline']-self.w.journal['started_at'],180)
        self.assertAlmostEqual(self.w.journal['first_deadline']-self.w.journal['started_at'],60)
        self.assertLessEqual(self.w.journal['phase_deadline'],self.w.journal['first_deadline'])

    def test_no_receipt_exhausted_or_used_cast_never_get_a_new_budget(self):
        self.failed();original=deepcopy(self.w.journal)
        for changes in ({'issued_inputs':[]},{'attempts':{'first-map':3}},
                        {'attempts':{'first-map':1,'first-cast':1}},
                        {'attempts':{'first-map':1,'first-select':1}},
                        {'first_map_repair_runs':1},{'meter_attempts':{'start':1}},
                        {'first_unselected':False}):
            self.w.journal={**original,**changes};self.w._save();raw=self.path.read_bytes()
            with self.assertRaises(ValueError):
                StoneFarm.request_retry(self.path.parent,hashlib.sha256(raw).hexdigest())
            self.assertEqual(self.path.read_bytes(),raw)

if __name__=='__main__':unittest.main()
