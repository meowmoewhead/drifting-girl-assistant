import builtins
from pathlib import Path
import symtable
import unittest


class RuntimeExtractionTests(unittest.TestCase):
    def test_no_nested_callback_loses_runtime_globals(self):
        root=Path(__file__).resolve().parents[1]
        for file in ('drift/runtime_inputs.py','drift/runtime_tasks.py'):
            table=symtable.symtable((root/file).read_text('utf-8-sig'),file,'exec')
            declared={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
            def check(scope):
                missing={s.get_name() for s in scope.get_symbols() if s.is_global()}-declared-set(dir(builtins))
                self.assertEqual(missing,set(),(file,scope.get_name(),missing))
                for child in scope.get_children():check(child)
            for child in table.get_children():check(child)


if __name__=='__main__':unittest.main()
