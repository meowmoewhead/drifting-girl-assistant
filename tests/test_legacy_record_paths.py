from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock

from drift.core.persistence import ReliablePath,RecordUnavailable


class LegacyRecordPathTests(unittest.TestCase):
    def test_legacy_json_temp_names_are_unique_and_round_trip_exact_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            path=ReliablePath(d)/'state.json'
            first,second=path.with_suffix('.tmp'),path.with_suffix('.tmp')
            self.assertNotEqual(first,second)
            raw=b'\xef\xbb\xbf{"deadline":1,"attempts":{"cast":1}}\r\n'
            first.write_bytes(raw);first.replace(path)
            self.assertEqual(Path(path).read_bytes(),raw)
            self.assertFalse(first.exists())

    def test_non_json_suffix_and_plain_path_are_unmodified(self):
        path=ReliablePath('example.png')
        self.assertEqual(str(path.with_suffix('.json')),str(Path('example.json')))
        self.assertEqual(str(Path('state.json').with_suffix('.tmp')),'state.tmp')

    def test_locked_stat_is_not_mistaken_for_a_missing_record(self):
        path=ReliablePath('existing_state.json')
        with patch('drift.core.persistence.sharing_retry',side_effect=PermissionError('locked')):
            for check in (path.exists,path.is_file,path.is_dir):
                with self.assertRaises(RecordUnavailable):check()

    def test_persistent_read_failure_is_not_caught_as_missing_or_resettable_oserror(self):
        with tempfile.TemporaryDirectory() as d:
            path=ReliablePath(d)/'state.json';Path(path).write_bytes(b'old')
            with patch('drift.core.persistence.sharing_retry',side_effect=PermissionError('locked')):
                with self.assertRaises(RecordUnavailable):path.read_text(encoding='utf-8')
            self.assertEqual(Path(path).read_bytes(),b'old')

    def test_locked_legacy_replace_keeps_previous_record_and_cleans_own_temp(self):
        with tempfile.TemporaryDirectory() as d:
            path=ReliablePath(d)/'state.json';path.write_bytes(b'old')
            tmp=path.with_suffix('.tmp');tmp.write_bytes(b'new')
            with patch('drift.core.persistence.os.replace',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):tmp.replace(path)
            self.assertEqual(Path(path).read_bytes(),b'old');self.assertFalse(tmp.exists())


if __name__=='__main__':unittest.main()
