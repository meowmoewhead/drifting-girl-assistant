from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from drift.core.persistence import atomic_write_json
from drift.tasks.entry_recovery import EntryRecoveryJournal,identity


class EntryRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path=self.root/'stone_farm_state.json';self.now=110
        self.base={'version':1,'pending':True,'phase':'arrive','reason':'cycle','rank':40,'entry_rank':0,
            'from_stage':9,'to_stage':10,'started_at':90,'entered_verified_at':100,
            'deadline':600,'phase_deadline':180,'attempts':{'enter-confirm':1},
            'issued_inputs':[{'kind':'enter-confirm','input_id':'entry-a'}]}
        atomic_write_json(self.path,self.base)
        self.c=NS(path=self.path,root=self.root,journal=deepcopy(self.base),phase='arrive')
        self.ledger=EntryRecoveryJournal(self.root,'serial',clock=lambda:self.now)

    def committed_new_phase(self):
        self.c.journal.update(phase='first_map',phase_deadline=150,first_deadline=160,
                              first_travel_deadline=500,first_cast_deadline=550,first_recovery=True)
        self.c.phase='first_map';self.ledger.capture(self.c)

    def test_failed_main_commit_resumes_only_precommitted_phase(self):
        self.committed_new_phase();old=self.path.read_bytes();self.now=120
        restarted=NS(path=self.path,journal=json.loads(old),phase='arrive')
        ledger=EntryRecoveryJournal(self.root,'serial',clock=lambda:self.now)
        self.assertEqual(ledger.reconcile(restarted),'resume_existing_phase')
        self.assertEqual(restarted.phase,'first_map')
        self.assertEqual(restarted.journal['deadline'],600)
        self.assertEqual(restarted.journal['phase_deadline'],150)
        self.assertEqual(restarted.journal['attempts'],{'enter-confirm':1})
        archive=list((self.root/'logs/entry-recovery-retired').glob('*-main.json'))
        self.assertEqual(archive[0].read_bytes(),old)

    def test_expired_ledger_never_gets_a_new_window(self):
        self.committed_new_phase();old=self.path.read_bytes();self.now=151
        restarted=NS(path=self.path,journal=json.loads(old),phase='arrive')
        self.assertEqual(self.ledger.reconcile(restarted),'observe_only')
        self.assertEqual(self.path.read_bytes(),old)
        self.assertEqual(self.ledger.saved['journal']['phase_deadline'],150)

    def test_an_uncertain_cast_reservation_remains_spent_after_main_failure(self):
        self.committed_new_phase()
        self.c.journal.update(phase='first_idle',phase_deadline=170)
        self.c.journal['attempts']['first-cast']=1
        self.c.journal['last_input']={'kind':'first-cast','state':'reserved'}
        self.ledger.capture(self.c);self.now=120
        restarted=NS(path=self.path,journal=json.loads(self.path.read_bytes()),phase='arrive')
        self.ledger.reconcile(restarted)
        self.assertEqual(restarted.journal['attempts']['first-cast'],1)
        self.assertEqual(restarted.journal['last_input']['state'],'reserved')

    def test_foreign_edit_and_another_entry_are_never_overwritten(self):
        for update in ({'unrelated_online_edit':1},{'entered_verified_at':101}):
            self.committed_new_phase();changed={**self.base,**update};atomic_write_json(self.path,changed)
            c=NS(path=self.path,journal=changed,phase='arrive');before=self.path.read_bytes()
            self.ledger.reconcile(c);self.assertEqual(self.path.read_bytes(),before)

    def test_rank_zero_alone_is_not_arrival_proof(self):
        for missing in ('entered_verified_at','issued_inputs'):
            c=deepcopy(self.base);c.pop(missing)
            self.assertIsNone(identity(c,'serial'))
        c={**self.base,'rank':39};self.assertIsNone(identity(c,'serial'))

    def test_next_entry_archives_every_old_reservation(self):
        self.committed_new_phase();old=self.ledger.raw
        self.c.journal.update(entered_verified_at=102,started_at=101)
        self.c.journal['issued_inputs']=[{'kind':'enter-confirm','input_id':'entry-b'}]
        self.ledger.capture(self.c)
        self.assertIn(old,[p.read_bytes() for p in (self.root/'logs/entry-recovery-retired').glob('*.json')])


if __name__=='__main__':unittest.main()
