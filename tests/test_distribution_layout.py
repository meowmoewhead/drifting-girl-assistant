"""New templates coexist with legacy saved state; update packages stay scoped."""
import json
from pathlib import Path
import tempfile
import unittest
from drift.core.paths import ensure_user_config
from drift.update.package import program_path


class DistributionLayoutTests(unittest.TestCase):
    def test_first_start_copies_disabled_template_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config').mkdir()
            content=b'{"rules":[{"enabled":false}],"serial":"emulator-5554"}'
            (root/'config/default.json').write_bytes(content)
            self.assertTrue(ensure_user_config(root))
            self.assertEqual((root/'config.json').read_bytes(),content)
            (root/'config.json').write_bytes(b'operator configuration')
            (root/'meter_calibration_state.json').write_bytes(b'preserve attempts and deadline')
            self.assertFalse(ensure_user_config(root))
            self.assertEqual((root/'config.json').read_bytes(),b'operator configuration')
            self.assertEqual((root/'meter_calibration_state.json').read_bytes(),b'preserve attempts and deadline')

    def test_enabled_template_is_not_used_on_first_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config').mkdir()
            (root/'config/default.json').write_text('{"rules":[{"enabled":true}]}')
            with self.assertRaises(ValueError):ensure_user_config(root)
            self.assertFalse((root/'config.json').exists())

    def test_legacy_template_still_supported_without_migrating_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.example.json').write_text('{"rules":[]}')
            self.assertTrue(ensure_user_config(root))
            self.assertEqual(json.loads((root/'config.json').read_text()),{'rules':[]})

    def test_updater_accepts_public_template_but_never_operator_files(self):
        for name in ('config/default.json','README.md','docs/STRUCTURE.md'):
            self.assertTrue(program_path(name),name)
        for name in ('config.json','config/user.json','meter_calibration_state.json',
                     'task_schedule.json','deploy/install.ps1','dev_tools/private.py','assets/custom/a.png'):
            self.assertFalse(program_path(name),name)


if __name__=='__main__':unittest.main()
