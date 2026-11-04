"""Offline public-source/archive checks. Does not connect to an emulator."""
from pathlib import Path
import argparse, ast, hashlib, json, re, sys, zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def verify_fixture_metadata(root=ROOT):
    """Published fixture provenance must not expose machine-specific directories."""
    checked=0
    def visit(value, location):
        if isinstance(value,dict):
            for key,item in value.items():visit(item,location+'/'+str(key))
        elif isinstance(value,list):
            for index,item in enumerate(value):visit(item,location+'/'+str(index))
        elif isinstance(value,str):
            absolute=bool(re.match(r'^[A-Za-z]:[\\/]',value)) or value.startswith(('\\\\','/'))
            assert not absolute, 'Absolute path in public fixture metadata: '+location
    for path in sorted((Path(root)/'tests/fixtures').rglob('*.json')):
        visit(json.loads(path.read_text(encoding='utf-8-sig')),path.relative_to(root).as_posix())
        checked+=1
    return checked

def verify_source(root=ROOT):
    root=Path(root)
    config=json.loads((root/'config.example.json').read_text(encoding='utf-8-sig'))
    assert config['adb_path']=='adb' and config['serial']=='emulator-5554'
    assert config.get('return_parallel_enabled') is False
    assert config['rules'] and all(r['enabled'] is False for r in config['rules'])
    from drift.tasks.features import FEATURES, RULE_FEATURE, feature_states
    features=feature_states(config)
    assert {r['id'] for r in config['rules']}==set(RULE_FEATURE)
    assert all(f['available'] and not f['enabled'] for f in features)
    assets=set()
    for rule in config['rules']:
        for name in (rule.get('asset'),*rule.get('asset_variants',[])):
            if name:
                path=(root/name).resolve()
                assert path.is_relative_to(root.resolve()) and path.is_file(),name
                assets.add(name)
    metadata=json.loads((root/'models/digit-model.json').read_text(encoding='utf-8'))
    model=root/'models'/metadata['file']
    assert hashlib.sha256(model.read_bytes()).hexdigest()==metadata['sha256']
    scripts=[*root.glob('*.py'),*root.glob('drift/**/*.py'),*root.glob('alas_core/**/*.py'),
             *root.glob('tools/*.py'),*root.glob('tests/*.py')]
    for script in scripts: ast.parse(script.read_text(encoding='utf-8-sig'),filename=str(script))
    for name in ('LICENSE','THIRD_PARTY_NOTICES.md','USER-GUIDE.md','README.md','VERSION'):
        assert (root/name).is_file()
    fixture_metadata=verify_fixture_metadata(root)
    return dict(rules=len(config['rules']),features=len(FEATURES),default_enabled_features=0,
                fixture_metadata_files=fixture_metadata,
                rule_assets=len(assets),parsed_python_files=len(scripts),model_sha256_verified=True)

def verify_archive(path, root=ROOT):
    from tools.package_release import source_sha256
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        names=archive.namelist()
        assert {n.split('/')[0] for n in names}=={'mumu-clicker'}
        assert all(not any(part in ('tests','diagnostics','captures','runtime','.git','.venv','backups')
                           for part in Path(n).parts) for n in names)
        assert archive.read('mumu-clicker/config.json')==(root/'config.example.json').read_bytes()
        assert archive.read('mumu-clicker/config.example.json')==(root/'config.example.json').read_bytes()
        meta=json.loads(archive.read('mumu-clicker/version.json'))
        assert meta['source_sha256']==source_sha256(root)
        assert meta['exe_sha256']==hashlib.sha256(archive.read('mumu-clicker/mumu-panel.exe')).hexdigest()
        config=json.loads(archive.read('mumu-clicker/config.json'))
        for r in config['rules']:
            for name in (r.get('asset'),*r.get('asset_variants',[])):
                if name: assert 'mumu-clicker/'+name in names,name
        assert len([n for n in names if n.startswith('mumu-clicker/logs/')])==1
        return dict(files=meta['files'],exe_sha256=meta['exe_sha256'],source_sha256=meta['source_sha256'],
                    public_config_verified=True,archive_crc_verified=True,all_rule_assets_included=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path)
    args=p.parse_args()
    print(json.dumps(verify_archive(args.archive) if args.archive else verify_source(),ensure_ascii=False,indent=2))
