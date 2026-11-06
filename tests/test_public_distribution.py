"""Publishing must never export the developer's live configuration."""
import json
from pathlib import Path
import unittest
from deploy import package_release as release
from deploy.verify_release import verify_source
from tests import test_package_release as fixtures

class PublicSourceTests(unittest.TestCase):
    def test_all_public_tasks_start_disabled_and_assets_are_complete(self):
        result=verify_source()
        self.assertEqual(result['default_enabled_features'],0)
        self.assertGreater(result['rules'],0)

class PublicArchiveTests(unittest.TestCase):
    setUp=fixtures.PackageReleaseTests.setUp
    write=fixtures.PackageReleaseTests.write
    archive=fixtures.PackageReleaseTests.archive
    def test_live_config_is_never_exported(self):
        self.write('config.json',b'{"adb_path":"D:/private-machine/adb.exe","rules":[{"enabled":true}]}')
        with self.archive() as archive:
            self.assertEqual(archive.read('mumu-clicker/config.json'),(self.root/'config/default.json').read_bytes())
            self.assertNotIn(b'private-machine',archive.read('mumu-clicker/config.json'))

    def test_no_local_config_is_required_for_public_packaging(self):
        (self.root/'config.json').unlink()
        with self.archive() as archive:
            self.assertIn('mumu-clicker/config.json',archive.namelist())

    def test_enabled_example_is_rejected(self):
        self.write('config/default.json',b'{"rules":[{"enabled":true}]}')
        with self.assertRaisesRegex(ValueError,'disabled'):
            release.package_release(self.root)

    def test_ghost_variant_is_shipped_if_it_is_a_runtime_asset(self):
        self.write('assets/event-potion/main-ghost.png',b'fixture')
        with self.archive() as archive:
            self.assertIn('mumu-clicker/assets/event-potion/main-ghost.png',archive.namelist())

if __name__=='__main__': unittest.main()
