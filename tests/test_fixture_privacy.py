"""The public source package must keep fixture provenance machine independent."""
from pathlib import Path
import json,tempfile,unittest
from tools.verify_release import verify_fixture_metadata

class FixturePrivacyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.file=self.root/'tests/fixtures/example/manifest.json'
        self.file.parent.mkdir(parents=True)

    def write(self,value):
        self.file.write_text(json.dumps(value),encoding='utf-8')

    def test_source_filename_and_hash_are_allowed(self):
        self.write({'frame.png':{'source':'capture.png','sha256':'a'*64}})
        self.assertEqual(verify_fixture_metadata(self.root),1)

    def test_nested_absolute_path_is_rejected(self):
        self.write([{'capture':{'source':str(self.root.resolve()/'capture.png')}}])
        with self.assertRaisesRegex(AssertionError,'Absolute path in public fixture metadata'):
            verify_fixture_metadata(self.root)

    def test_source_url_is_not_a_machine_path(self):
        self.write({'source':'https://example.com/capture.png'})
        self.assertEqual(verify_fixture_metadata(self.root),1)

    def test_relative_fixture_path_is_allowed(self):
        self.write({'source':'captures/capture.png'})
        self.assertEqual(verify_fixture_metadata(self.root),1)
