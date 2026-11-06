# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from pathlib import Path
import runpy
import re

BUILD_ROOT = Path(SPECPATH).resolve().parent
build_identity = runpy.run_path(str(BUILD_ROOT / 'drift/core/build_identity.py'))
build_identity['generate'](BUILD_ROOT)

datas = [(str(BUILD_ROOT / 'web'), 'web')]
binaries = []
hiddenimports = ['drift.device.evidence', 'drift.device.native', 'drift.perception.digit_ocr', 'drift.perception.purchase_checks', 'drift.perception.reward_list', 'drift.tasks.backpack_sale', 'drift.tasks.boss_defense', 'drift.tasks.boss_training', 'drift.tasks.daily', 'drift.tasks.equipment_upgrade', 'drift.tasks.features', 'drift.tasks.fishing_meter', 'drift.tasks.frame_runtime', 'drift.tasks.game_recovery', 'drift.tasks.meter_calibration', 'drift.tasks.parallel_skills', 'drift.tasks.parallel_world', 'drift.tasks.schedule', 'drift.tasks.training_sailing']
for proxy in BUILD_ROOT.glob('*.py'):
    hiddenimports.extend(re.findall(r'''module_proxy\(\s*__name__\s*,\s*["']([^"']+)["']\s*\)''',proxy.read_text(encoding='utf-8-sig')))
hiddenimports = sorted(set(hiddenimports))
binaries += collect_dynamic_libs('onnxruntime')
hiddenimports += collect_submodules('drift')
tmp_ret = collect_all('maa')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    [str(BUILD_ROOT / 'desktop_entry.py')],
    pathex=[str(BUILD_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['MSVCP140', 'MSVCP140_1', 'scipy', 'matplotlib', 'pandas', 'PyQt5', 'PySide6', 'IPython', 'pytest', 'imageio'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='mumu-panel',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
