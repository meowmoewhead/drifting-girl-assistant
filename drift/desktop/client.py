"""Local JSON transport only; desktop rendering never loads an HTML page."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, ProxyHandler, build_opener


def validate_address(address):
    parts = urlsplit(address.rstrip('/'))
    if parts.scheme != 'http' or parts.hostname not in ('127.0.0.1', 'localhost', '::1') or parts.username or parts.password or parts.path or parts.query or parts.fragment or not parts.port:
        raise ValueError('仅连接本机地址，例如 http://127.0.0.1:8765；远程机器请使用已有 SSH 转发')
    return address.rstrip('/')


class Client:
    def __init__(self, address):
        self.address = validate_address(address)
        self.session = None
        self.opener = build_opener(ProxyHandler({}))

    def request(self, path, data=None, binary=False):
        headers = {'X-Desktop-Client': 'drift-native-v1'}
        body = None
        if data is not None:
            if self.session is None:
                self.connect()
            headers.update({'X-Session-Token': self.session['token'], 'Content-Type': 'application/json'})
            body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        try:
            with self.opener.open(Request(self.address + path, data=body, headers=headers), timeout=25) as response:
                value = response.read(16 * 1024 * 1024 + 1)
            if len(value) > 16 * 1024 * 1024:
                raise ValueError('面板响应过大')
            return value if binary else json.loads(value)
        except HTTPError as exc:
            if exc.code == 403:
                self.session = None
            try:
                message = json.loads(exc.read(65536)).get('error', str(exc))
            except ValueError:
                message = str(exc)
            raise ValueError(message) from None

    def connect(self):
        session = self.request('/api/desktop/session')
        if session.get('protocol') != 1:
            raise ValueError('当前服务尚未支持桌面面板，请安装新版程序')
        self.session = session
        return session

    def upload(self, path, digest):
        import http.client
        if self.session is None:
            self.connect()
        path = Path(path)
        parts = urlsplit(self.address)
        connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=30)
        try:
            with path.open('rb') as source:
                connection.request('POST', '/api/desktop/update/import', body=source,
                                   headers={'Content-Length': str(path.stat().st_size), 'Content-Type': 'application/zip',
                                            'X-Desktop-Client': 'drift-native-v1', 'X-Session-Token': self.session['token'],
                                            'X-Package-SHA256': digest})
                response = connection.getresponse()
                data = json.loads(response.read(65536))
                if response.status != 200:
                    raise ValueError(data.get('error', '离线包导入失败'))
                return data
        finally:
            connection.close()


def ensure_server(root, port):
    client = Client(f'http://127.0.0.1:{port}')
    try:
        client.connect()
        return client
    except HTTPError:
        raise ValueError('该端口已有旧面板，请先安装新版后再打开桌面面板')
    except ValueError:
        raise
    except URLError as exc:
        if not isinstance(exc.reason, ConnectionRefusedError) and getattr(exc.reason, 'winerror', None) != 10061:
            raise
    root = Path(root).resolve()
    command = ([sys.executable] if getattr(sys, 'frozen', False) else [sys.executable, str(root / 'app.py')])
    logdir = root / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    with (logdir / 'desktop-service.log').open('ab') as log:
        process = subprocess.Popen(command + ['serve', '--root', str(root), '--port', str(port)], cwd=root,
                                   stdout=log, stderr=log,
                                   env={**os.environ, 'PYINSTALLER_RESET_ENVIRONMENT': '1'},
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    until = time.monotonic() + 45
    while time.monotonic() < until:
        try:
            client.connect()
            return client
        except (OSError, ValueError):
            if process.poll() is not None:
                raise RuntimeError('面板服务退出，请查看 logs/desktop-service.log')
            time.sleep(.4)
    raise TimeoutError('面板服务尚未就绪，请查看 logs/desktop-service.log；不会重复启动')
