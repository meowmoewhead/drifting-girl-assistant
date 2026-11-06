"""All test modules in isolated processes; a source-bound build gate."""
import concurrent.futures
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'tests'))

def manifest():
    paths = list(ROOT.glob('*.py'))
    for name in ('drift', 'alas_core', 'models', 'deploy', 'dev_tools', 'tests', 'web', 'assets', 'licenses'):
        paths.extend(p for p in (ROOT / name).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix != '.pyc' and p.name != 'build_info.py')
    paths.extend(ROOT / name for name in ('config.json', 'requirements.txt', 'README.md',
        'THIRD_PARTY_NOTICES.md', 'VALIDATION.md', 'LICENSE', 'start-panel.cmd', 'start-panel.ps1'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths)) if p.exists()}

def peaks():
    class Counts(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]
    value = Counts(); value.cb = ctypes.sizeof(value)
    kernel = ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype = wintypes.HANDLE
    info = ctypes.WinDLL('psapi').GetProcessMemoryInfo
    info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counts), wintypes.DWORD]
    info.restype = wintypes.BOOL
    if not info(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
        raise ctypes.WinError()
    return dict(working_set=value.WorkingSetSize, peak_working_set=value.PeakWorkingSetSize,
                private_commit=value.PrivateUsage, peak_commit=value.PeakPagefileUsage)

def child(module, folder):
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(module))
    report = dict(module=module, tests=result.testsRun, failures=len(result.failures),
                  errors=len(result.errors), skipped=len(result.skipped), seconds=time.monotonic()-started,
                  failed_ids=[test.id() for test, _ in result.failures + result.errors], memory=peaks())
    (folder / (module + '.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0 if result.wasSuccessful() else 1

def run_module(module, folder):
    with (folder / (module + '.log')).open('w', encoding='utf-8') as log:
        try:
            completed = subprocess.run([sys.executable, '-X', 'utf8', str(__file__), '--child', module, str(folder)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600,
                env={**os.environ, 'PYTHONUTF8': '1'})
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = 124
    path = folder / (module + '.json')
    row = json.loads(path.read_text()) if path.exists() else dict(module=module, tests=0, failures=0, errors=1)
    row['exit'] = code
    return row


def verify(folder):
    report=json.loads((folder/'result.json').read_text())
    expected=json.loads((folder/'source-manifest.json').read_text())
    if (not report.get('full_suite') or report.get('exit') != 0
            or not report.get('source_unchanged') or report.get('failures') or report.get('errors')
            or set(report['modules']) != {'tests.'+p.stem for p in (ROOT/'tests').glob('test_*.py')}
            or manifest() != expected):
        raise RuntimeError('Full tests failed, are incomplete, or source changed; build refused')
    print(json.dumps(dict(verified=True, tests=report['tests'], modules=len(report['modules']))))

if __name__ == '__main__':
    if sys.argv[1] == '--verify':
        verify(Path(sys.argv[2]))
        raise SystemExit(0)
    if sys.argv[1] == '--child':
        raise SystemExit(child(sys.argv[2], Path(sys.argv[3])))
    label = sys.argv[1]
    folder = HERE / label
    folder.mkdir(parents=True, exist_ok=False)
    modules = sys.argv[2:] or ['tests.' + p.stem for p in sorted((ROOT / 'tests').glob('test_*.py'))]
    before = manifest()
    (folder / 'source-manifest.json').write_text(json.dumps(before, sort_keys=True, indent=2), encoding='utf-8')
    started = time.monotonic(); rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_module, name, folder) for name in modules]
        for future in concurrent.futures.as_completed(futures):
            row = future.result(); rows.append(row)
            (folder / 'progress.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            print(json.dumps(dict(completed=len(rows), total=len(modules), **row)), flush=True)
    after = manifest()
    report = dict(full_suite=len(sys.argv) == 2, modules=modules, rows=rows,
                  tests=sum(r['tests'] for r in rows), failures=sum(r['failures'] for r in rows),
                  errors=sum(r['errors'] for r in rows), source_unchanged=before == after,
                  seconds=time.monotonic()-started,
                  exit=0 if before == after and all(r['exit'] == 0 for r in rows) else 1)
    (folder / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('full_suite','tests','failures','errors','source_unchanged','seconds','exit')}))
    raise SystemExit(report['exit'])
