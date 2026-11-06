"""Dispatch new commands without changing the existing headless CLI."""
import argparse
import os
import sys
from pathlib import Path


def dispatch(legacy):
    command = sys.argv[1] if len(sys.argv) > 1 else 'desktop'
    if command == 'desktop':
        from drift.core.paths import ROOT
        parser = argparse.ArgumentParser(description='漂流助手桌面控制面板')
        parser.add_argument('--root', type=Path, default=ROOT)
        parser.add_argument('--port', type=int, default=8765)
        parser.add_argument('--connect', default=None, help='本机服务或 SSH 转发地址')
        args = parser.parse_args(sys.argv[2:] if len(sys.argv) > 1 else [])
        from drift.desktop.panel import run
        return run(args.root, args.port, args.connect)
    if command == 'apply-update':
        parser = argparse.ArgumentParser()
        parser.add_argument('--plan', type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        from drift.update.installer import apply_plan
        return apply_plan(args.plan)
    if command == 'serve':
        from drift.core.paths import ROOT
        root = Path(sys.argv[sys.argv.index('--root') + 1]) if '--root' in sys.argv else ROOT
        lock = root / '.updates/install.lock'
        if lock.exists():
            directory = Path(lock.read_text().strip())
            if os.environ.get('DRIFT_UPDATE_SESSION') != directory.name:
                raise RuntimeError('正在更新程序，暂不启动另一个后台服务')
    return legacy()
