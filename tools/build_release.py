"""Build a Windows portable release from this source snapshot; no deployment."""
from datetime import datetime,timezone
from pathlib import Path
import argparse, json, os, re, shutil, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import package_release
from tools.verify_release import verify_source,verify_archive

def execute(args,log=None):
    print('RUN '+json.dumps([str(a) for a in args],ensure_ascii=False),flush=True)
    env=dict(os.environ)
    env.pop('PYTHONPATH',None)
    if log:
        with Path(log).open('w',encoding='utf-8') as f:
            subprocess.run([str(a) for a in args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    else:
        subprocess.run([str(a) for a in args],cwd=ROOT,env=env,check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-tests',action='store_true',help='Also run every regression module in isolated processes')
    args=parser.parse_args()
    started=time.time()
    build_id=datetime.now().strftime('%Y%m%d-%H%M%S')
    work=ROOT/'build'/build_id
    work.mkdir(parents=True,exist_ok=False)
    if not (ROOT/'config.json').exists(): shutil.copy2(ROOT/'config.example.json',ROOT/'config.json')
    preflight=verify_source()
    source_hash=package_release.source_sha256(ROOT)
    if args.full_tests:
        execute([sys.executable,'-B','-X','utf8',ROOT/'tools/test_gate.py',f'diagnostics/build-{build_id}'],work/'tests.log')
    else:
        execute([sys.executable,'-B','-X','utf8','-m','unittest','-v',
                 'tests.test_public_distribution','tests.test_package_release','tests.test_version',
                 'tests.test_task_registry','tests.test_fixture_privacy'],work/'tests.log')
    hidden=[]
    for path in ROOT.glob('*.py'):
        hidden += re.findall(r'module_proxy\(\s*__name__\s*,\s*["\']([^"\']+)["\']',path.read_text(encoding='utf-8-sig'))
    assert len(set(hidden))>=10
    options=[sys.executable,'-B','-m','PyInstaller','--noconfirm','--clean','--onefile','--name','mumu-panel',
             '--distpath',ROOT/'release','--workpath',work/'pyinstaller','--specpath',work/'spec',
             '--paths',ROOT,'--collect-submodules','drift','--collect-all','maa',
             '--collect-binaries','onnxruntime','--add-data',str(ROOT/'web')+';web']
    for name in ('MSVCP140','MSVCP140_1','scipy','matplotlib','pandas','tkinter','PyQt5','PySide6','IPython','pytest','imageio'):
        options += ['--exclude-module',name]
    for name in sorted(set(hidden)): options += ['--hidden-import',name]
    options.append(ROOT/'app.py')
    execute(options,work/'pyinstaller.log')
    exe=ROOT/'release/mumu-panel.exe'
    assert exe.is_file() and exe.stat().st_mtime>=started
    assert source_hash==package_release.source_sha256(ROOT),'Sources changed during build'
    execute([exe,'--version'],work/'version.log')
    execute([exe,'framework-check','--root',ROOT],work/'framework-check.log')
    archive=package_release.package_release(ROOT)
    verified=verify_archive(Path(archive['archive']))
    report=dict(built_at=datetime.now(timezone.utc).isoformat(),preflight=preflight,
                full_regression_suite=args.full_tests,source_sha256=source_hash,archive=archive,verification=verified,
                seconds=round(time.time()-started,2),build_log=str(work.relative_to(ROOT)))
    (ROOT/'release/build-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
