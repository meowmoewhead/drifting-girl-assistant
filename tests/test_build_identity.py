from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from drift.core.build_identity import generate,verify,literal_identity


class BuildIdentityTests(unittest.TestCase):
    def root(self,d):
        root=Path(d);(root/'drift').mkdir();(root/'web').mkdir()
        (root/'app.py').write_text('import drift\n',encoding='utf-8')
        (root/'drift/a.py').write_text('VALUE=1\n',encoding='utf-8')
        (root/'web/app.js').write_text('let value=1;',encoding='utf-8')
        return root

    def test_normal_build_generates_a_verifiable_literal_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.root(d);identity,manifest=generate(root)
            self.assertEqual(verify(root),identity)
            self.assertEqual(literal_identity(compile((root/'drift/build_info.py').read_text('utf-8'),'<metadata>','exec')),identity)
            self.assertIn('app.py',manifest)

    def test_hot_config_and_logs_do_not_change_source_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.root(d);identity,_=generate(root)
            (root/'config.json').write_text('{"enabled":false}')
            (root/'logs').mkdir();(root/'logs/new.log').write_text('changed')
            self.assertEqual(verify(root),identity)

    def test_source_change_is_rejected_until_rebuilt(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.root(d);old,_=generate(root)
            (root/'drift/a.py').write_text('VALUE=2\n',encoding='utf-8')
            with self.assertRaises(ValueError):verify(root)
            new,_=generate(root);self.assertNotEqual(old['source_sha256'],new['source_sha256'])

    def test_packaged_python_is_not_executed_to_read_identity(self):
        code=compile("import os\nBUILD={'schema':2}",'<unsafe>','exec')
        with self.assertRaises(ValueError):literal_identity(code)

    def test_normal_entry_uses_fixed_spec_instead_of_regenerating_it(self):
        root=Path(__file__).resolve().parents[1]
        text=(root/'deploy/build.ps1').read_text('utf-8-sig')
        self.assertIn('mumu-panel.spec',text)
        self.assertNotIn('$pyinstallerArgs += (Join-Path $taskRoot \'app.py\')',text)
        self.assertNotIn('Remove-Item $taskSpec',text)
        spec=(root/'deploy/mumu-panel.spec').read_text('utf-8-sig')
        self.assertIn("build_identity['generate'](BUILD_ROOT)",spec)
        self.assertIn('module_proxy',spec)

    def test_source_gate_excludes_generated_identity_but_checks_recipe(self):
        from dev_tools import test_gate
        with tempfile.TemporaryDirectory() as d:
            root=self.root(d);(root/'deploy').mkdir()
            recipe=root/'deploy/mumu-panel.spec';recipe.write_text('recipe 1')
            with patch.object(test_gate,'ROOT',root):
                before=test_gate.manifest();generate(root)
                self.assertEqual(test_gate.manifest(),before)
                recipe.write_text('recipe 2')
                self.assertNotEqual(test_gate.manifest(),before)


if __name__=='__main__':unittest.main()
