"""Attach native APIs while retaining the exact deployed task/API implementation."""
import inspect
from contextlib import nullcontext
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid

from drift.core.paths import read_version
from drift.update.manager import UpdateManager
from drift.update.package import MAX_ARCHIVE, checked_digest, sha256


def extend_handler(base_factory, app):
    base = base_factory(app)
    token = inspect.getclosurevars(base.do_POST).nonlocals.get('token')
    if not isinstance(token, str):
        raise RuntimeError('当前面板会话不支持桌面客户端')
    mutation_lock = threading.RLock()
    manager = None
    exe_digest = None

    def updater():
        nonlocal manager
        if manager is None:
            manager = UpdateManager(app.root)
        return manager

    class NativeHandler(base):
        def native_guard(self, write=False):
            if self.headers.get('Origin') or self.headers.get('Sec-Fetch-Site') or self.headers.get('X-Desktop-Client') != 'drift-native-v1':
                raise ValueError('只接受本机桌面客户端请求')
            if write and not secrets.compare_digest(self.headers.get('X-Session-Token', ''), token):
                raise PermissionError('控制面板已重启，请重连')

        def do_GET(self):
            nonlocal exe_digest
            path = urlsplit(self.path).path
            if not path.startswith('/api/desktop/'):
                return super().do_GET()
            try:
                self.native_guard()
                if path == '/api/desktop/session':
                    executable = Path(sys.executable).resolve()
                    if exe_digest is None:
                        exe_digest = sha256(executable) if getattr(sys, 'frozen', False) else None
                    return self.respond({'protocol': 1, 'token': token, 'pid': os.getpid(),
                                         'root': str(app.root), 'version': read_version(app.root),
                                         'exe_sha256': exe_digest, 'executable': str(executable),
                                         'frozen': bool(getattr(sys, 'frozen', False))})
                if path == '/api/desktop/update':
                    return self.respond(updater().status())
                return self.respond({'error': '不存在'}, status=404)
            except Exception as exc:
                return self.respond({'error': str(exc)}, status=400)

        def do_POST(self):
            path = urlsplit(self.path).path
            # A slow upload must not block Stop or other ordinary task controls.
            with (nullcontext() if path == '/api/desktop/update/import' else mutation_lock):
                if manager is not None and manager.installing:
                    return self.respond({'error': '正在安装更新，请等待面板重连'}, status=409)
                if not path.startswith('/api/desktop/'):
                    return super().do_POST()
                try:
                    self.native_guard(write=True)
                    length = int(self.headers.get('Content-Length', 0))
                    if path == '/api/desktop/update/import':
                        if not 0 < length <= MAX_ARCHIVE:
                            raise ValueError('离线更新包大小无效')
                        digest = checked_digest(self.headers.get('X-Package-SHA256', ''))
                        update = updater()
                        with update.lock:
                            if update.status()['busy']:
                                raise ValueError('已有更新任务进行中')
                            update.uploading = True
                        try:
                            directory = update.directory / uuid.uuid4().hex
                            directory.mkdir(parents=True)
                            archive = directory / 'package.zip'
                            remaining = length
                            self.connection.settimeout(20)
                            until = time.monotonic() + 900
                            with archive.open('xb') as f:
                                while remaining:
                                    block = self.rfile.read(min(256 * 1024, remaining))
                                    if not block or time.monotonic() > until:
                                        raise ValueError('离线更新包上传中断')
                                    f.write(block)
                                    remaining -= len(block)
                            with update.lock:
                                update.uploading = False
                                result = update.import_file(archive, digest)
                        finally:
                            update.uploading = False
                    else:
                        if not 0 < length <= 65536:
                            raise ValueError('请求体大小无效')
                        data = json.loads(self.rfile.read(length))
                        if path == '/api/desktop/update/check':
                            result = updater().check()
                        elif path == '/api/desktop/update/download':
                            result = updater().download()
                        elif path == '/api/desktop/update/settings':
                            result = updater().save_options(data)
                        elif path == '/api/desktop/update/install':
                            result = updater().install(app, self.server, data.get('client_pid'))
                        elif path == '/api/desktop/shutdown':
                            if app.running() or (getattr(app, 'thread', None) and app.thread.is_alive()):
                                raise ValueError('脚本仍运行，请先停止并等待收尾')
                            result = {'stopping': True}
                            threading.Thread(target=lambda: (time.sleep(.3), self.server.shutdown()), daemon=True).start()
                        else:
                            return self.respond({'error': '不存在'}, status=404)
                    return self.respond({'ok': True, 'result': result})
                except PermissionError as exc:
                    return self.respond({'error': str(exc)}, status=403)
                except Exception as exc:
                    return self.respond({'error': str(exc)}, status=400)
    return NativeHandler
