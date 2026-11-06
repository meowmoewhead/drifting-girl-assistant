import json
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

from drift.core import persistence as p


class ReliableRecordTests(unittest.TestCase):
    def test_sharing_failure_retries_only_replace(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'state.json';path.write_bytes(b'old')
            replace = p.os.replace
            attempts = []
            def locked(source, target):
                attempts.append(source)
                if len(attempts) < 3:raise PermissionError('WinError 5')
                return replace(source, target)
            with patch.object(p.os, 'replace', side_effect=locked):
                p.atomic_write_json(path, {'attempts': {'cast': 1}, 'deadline': 10})
            self.assertEqual(len(set(attempts)), 1)
            self.assertEqual(json.loads(path.read_bytes())['attempts'], {'cast': 1})
            self.assertEqual(list(Path(d).glob('*.tmp')), [])

    def test_permanent_failure_keeps_old_record_and_is_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.json';path.write_bytes(b'old')
            with patch.object(p.os,'replace',side_effect=PermissionError('locked')), \
                    patch.object(p,'sharing_retry',side_effect=lambda f: f()):
                with self.assertRaises(PermissionError):p.atomic_write_json(path, {'attempts': 1})
            self.assertEqual(path.read_bytes(), b'old')

    def test_unrelated_io_errors_are_not_retried(self):
        action=unittest.mock.Mock(side_effect=OSError('disk full'))
        with self.assertRaises(OSError):p.sharing_retry(action)
        self.assertEqual(action.call_count, 1)

    def test_compare_and_swap_rejects_changed_record(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.json';path.write_bytes(b'online edit')
            with self.assertRaises(ValueError):p.atomic_write_bytes(path,b'retired',expected=b'old')
            self.assertEqual(path.read_bytes(),b'online edit')

    def test_compare_and_swap_does_not_trust_an_inaccessible_stat(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.json';path.write_bytes(b'preserved input receipt')
            with patch.object(Path,'exists',return_value=False):
                with self.assertRaises(ValueError):
                    p.atomic_write_bytes(path,b'new record',expected=None)
            self.assertEqual(path.read_bytes(),b'preserved input receipt')

    def test_concurrent_writers_leave_complete_json_and_no_shared_temp(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.json'
            def save(i):p.atomic_write_json(path,{'value':i,'payload':'x'*10000})
            with ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(save,range(20)))
            self.assertEqual(len(json.loads(path.read_bytes())['payload']),10000)
            self.assertEqual(len(list(Path(d).iterdir())),1)

    def test_input_cannot_follow_failed_reservation(self):
        from drift.tasks.guild_daily import GuildDaily
        c=GuildDaily.__new__(GuildDaily);c.journal={'attempts':{}}
        # Callers must observe the failed durable save before issuing anything.
        c._save=unittest.mock.Mock(side_effect=PermissionError('locked'))
        c.allowed=lambda:True;c.clock=lambda:10
        with self.assertRaises(PermissionError):c.reserve({'kind':'stop'})
        self.assertEqual(c.journal['attempts']['stop'],1)


if __name__=='__main__':unittest.main()
