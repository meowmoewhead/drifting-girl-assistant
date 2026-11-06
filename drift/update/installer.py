"""Separate-process Windows installer; never terminates the emulator or game."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, ProxyHandler, build_opener

from .package import Transaction, atomic_json, contained, sha256


def process_image(pid):
    if os.name != 'nt':
        return Path(f'/proc/{pid}/exe').resolve() if Path(f'/proc/{pid}').exists() else None
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        length = wintypes.DWORD(len(buffer))
        if kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(length)):
            return Path(buffer.value).resolve()
    finally:
        kernel.CloseHandle(handle)
    return None


def wait_exit(pid, image, deadline):
    while process_image(pid) == image:
        if time.monotonic() > deadline:
            raise TimeoutError('控制面板未在90秒内退出，未强制结束任何现有进程')
        time.sleep(.25)


def health(port, version, digest, timeout=75, process=None):
    opener = build_opener(ProxyHandler({}))
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if process is not None and process.poll() is not None:
            raise RuntimeError('新面板在启动验证前已退出')
        try:
            request = Request(f'http://127.0.0.1:{port}/api/desktop/session', headers={'X-Desktop-Client': 'drift-native-v1'})
            with opener.open(request, timeout=2) as response:
                data = json.load(response)
            if data.get('version') == version and data.get('exe_sha256') == digest:
                with opener.open(f'http://127.0.0.1:{port}/api/status', timeout=3) as response:
                    state = json.load(response)
                if not state.get('running') and not state.get('config_missing'):
                    return data
        except (OSError, ValueError):
            pass
        time.sleep(.5)
    raise TimeoutError('新版本启动验证未通过')


def launch_server(root, port):
    if os.name == 'nt':
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    env = {**os.environ, 'PYINSTALLER_RESET_ENVIRONMENT': '1'}
    lock = root / '.updates/install.lock'
    if lock.exists():
        env['DRIFT_UPDATE_SESSION'] = Path(lock.read_text().strip()).name
    with (logs / 'update-start.log').open('ab') as log:
        return subprocess.Popen([str(root / 'mumu-panel.exe'), 'serve', '--root', str(root), '--port', str(port)],
                                cwd=root, stdout=log, stderr=log, env=env,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


def apply_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text('utf-8'))
    root = Path(plan['root']).resolve()
    directory = Path(plan['directory']).resolve()
    if directory.parent != root / '.updates' or path != directory / 'install-plan.json':
        raise ValueError('安装计划不属于本机更新目录')
    contained(root, '.updates/' + directory.name + '/install-plan.json')
    if Path(sys.executable).resolve() != directory / 'update-helper.exe' or sha256(sys.executable) != plan['old_exe_sha256']:
        raise ValueError('安装助手身份校验失败')
    if not 1 <= int(plan['port']) <= 65535:
        raise ValueError('面板端口无效')
    result = {'version': plan['version'], 'started_at': time.time(), 'success': False}
    transaction = None
    new_process = None
    lock = root / '.updates/install.lock'
    if lock.read_text().strip() != str(directory):
        raise ValueError('安装锁不匹配')
    try:
        until = time.monotonic() + 90
        image = root / 'mumu-panel.exe'
        wait_exit(plan['parent_pid'], image, until)
        if plan.get('client_pid'):
            wait_exit(plan['client_pid'], image, until)
        if sha256(image) != plan['old_exe_sha256']:
            raise ValueError('安装期间现有程序发生变化，停止更新')
        transaction = Transaction(root, directory / 'payload', root / 'backups' / ('update-' + directory.name), plan['files'])
        transaction.apply()
        new_process = launch_server(root, plan['port'])
        health(plan['port'], plan['version'], plan['files']['mumu-panel.exe'], process=new_process)
        transaction.commit()
        result.update(success=True, message='更新完成，脚本保持停止；配置和全部任务记录保留。', backup=str(transaction.backup))
    except Exception as exc:
        result['error'] = str(exc)
        if new_process is not None and new_process.poll() is None:
            # Only the newly launched, idle panel process tree; no emulator PID.
            if os.name == 'nt':
                subprocess.run(['taskkill.exe', '/PID', str(new_process.pid), '/T', '/F'],
                               capture_output=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                new_process.terminate()
            new_process.wait(timeout=15)
        if transaction and transaction.journal['replaced'] and transaction.journal['phase'] != 'rolled_back':
            try:
                transaction.rollback()
            except Exception as rollback_error:
                result['rollback_error'] = str(rollback_error)
        if transaction:
            result['rolled_back'] = transaction.journal['phase'] == 'rolled_back'
            result['backup'] = str(transaction.backup)
        # Do not start another server if we never observed the old one exit.
        if not result.get('rollback_error') and process_image(plan['parent_pid']) != root / 'mumu-panel.exe':
            launch_server(root, plan['port'])
    finally:
        result['finished_at'] = time.time()
        atomic_json(directory / 'result.json', result)
        atomic_json(root / '.updates/last-result.json', result)
        if not result.get('rollback_error'):
            lock.unlink(missing_ok=True)
    if plan.get('client_pid') and not result.get('rollback_error'):
        subprocess.Popen([str(root / 'mumu-panel.exe'), 'desktop', '--root', str(root), '--port', str(plan['port'])],
                         cwd=root, env={**os.environ, 'PYINSTALLER_RESET_ENVIRONMENT': '1'},
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return result
