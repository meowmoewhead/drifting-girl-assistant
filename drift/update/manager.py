"""Bounded background release checks/downloads for the native desktop panel."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
from urllib.request import Request, ProxyHandler, HTTPSHandler, build_opener
import uuid

from drift.core.paths import read_version
from .package import (REPOSITORY, MAX_ARCHIVE, Version, atomic_json, checked_digest,
                      checksum_for, sha256, unpack)


class Network:
    def __init__(self, proxy=''):
        if proxy:
            parts = urlsplit(proxy)
            if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password or parts.path not in ('', '/') or parts.query or parts.fragment:
                raise ValueError('代理须为 http(s)://主机:端口，不支持内嵌账号密码')
        self.opener = build_opener(ProxyHandler({'http': proxy, 'https': proxy} if proxy else {}), HTTPSHandler())

    def open(self, url):
        parts = urlsplit(url)
        if parts.scheme != 'https' or parts.username or parts.password or parts.hostname not in ('api.github.com', 'github.com'):
            raise ValueError('更新地址必须来自本项目 GitHub 发行页')
        if parts.hostname == 'api.github.com' and not parts.path.startswith('/repos/' + REPOSITORY + '/'):
            raise ValueError('更新仓库不匹配')
        if parts.hostname == 'github.com' and not parts.path.startswith('/' + REPOSITORY + '/releases/download/'):
            raise ValueError('更新附件不属于本项目')
        response = self.opener.open(Request(url, headers={'User-Agent': 'DriftingGirl-Updater/1', 'Accept': 'application/vnd.github+json'}), timeout=20)
        final = urlsplit(response.url)
        if final.scheme != 'https' or not (final.hostname in ('github.com', 'api.github.com') or
                                          (final.hostname or '').endswith('.githubusercontent.com')):
            response.close()
            raise ValueError('更新下载跳转到了非 GitHub 地址')
        return response

    def small(self, url, limit=4 * 1024 * 1024):
        with self.open(url) as response:
            content = response.read(limit + 1)
        if len(content) > limit:
            raise ValueError('更新元数据过大')
        return content

    def download(self, url, path, size, progress):
        if not 0 < size <= MAX_ARCHIVE:
            raise ValueError('更新包大小超过限制')
        received = 0
        started = time.monotonic()
        with self.open(url) as response, Path(path).open('xb') as output:
            while block := response.read(256 * 1024):
                received += len(block)
                if received > size or time.monotonic() - started > 900:
                    raise ValueError('下载大小异常或超过15分钟，请重新下载')
                output.write(block)
                progress(received, size)
        if received != size:
            raise ValueError('更新包下载不完整')


def pick_release(releases, current, preview):
    candidates = []
    for release in releases:
        if release.get('draft') or (release.get('prerelease') and not preview):
            continue
        try:
            version = Version(release['tag_name'])
        except (KeyError, ValueError):
            continue
        if version <= Version(current) or (version.pre and not preview):
            continue
        name = f'mumu-clicker-{version.value}-win64.zip'
        assets = release.get('assets', [])
        packages = [a for a in assets if a.get('name') == name]
        sums = [a for a in assets if a.get('name') == 'SHA256SUMS.txt']
        if len(packages) != 1 or len(sums) != 1:
            continue
        candidates.append((version, {
            'version': version.value, 'name': name, 'prerelease': bool(release.get('prerelease')),
            'notes': str(release.get('body') or '此版本没有附带更新说明。')[:40000],
            'published_at': release.get('published_at'), 'url': packages[0]['browser_download_url'],
            'bytes': packages[0]['size'], 'checksums_url': sums[0]['browser_download_url'],
        }))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


class UpdateManager:
    def __init__(self, root, network_factory=Network):
        self.root = Path(root).resolve()
        self.version = read_version(self.root)
        self.directory = self.root / '.updates'
        self.network_factory = network_factory
        self.lock = threading.RLock()
        self.thread = None
        self.installing = False
        self.uploading = False
        self.staged = None
        self.options = {'preview': Version(self.version).pre is not None, 'proxy': ''}
        try:
            saved = json.loads((self.root / 'updater_settings.json').read_text('utf-8-sig'))
            self.options = self.validate_options(saved)
        except FileNotFoundError:
            pass
        except (ValueError, TypeError, KeyError):
            pass
        self.state = {'phase': 'idle', 'message': '点击检查更新。更新来源：本项目 GitHub Releases。',
                      'current_version': self.version, 'release': None, 'progress': 0, 'error': None}

    @staticmethod
    def validate_options(data):
        if set(data) != {'preview', 'proxy'} or type(data['preview']) is not bool or not isinstance(data['proxy'], str):
            raise ValueError('更新设置无效')
        proxy = data['proxy'].strip()
        Network(proxy)
        return {'preview': data['preview'], 'proxy': proxy}

    def status(self):
        with self.lock:
            result = {**self.state, 'options': dict(self.options), 'repository': REPOSITORY,
                      'installable': self.staged is not None and not self.installing,
                      'busy': bool(self.thread and self.thread.is_alive()) or self.installing or self.uploading}
        outcome = self.directory / 'last-result.json'
        if outcome.is_file():
            try:
                result['last_install'] = json.loads(outcome.read_text('utf-8'))
            except (OSError, ValueError):
                pass
        return result

    def set_state(self, **values):
        with self.lock:
            self.state.update(values)

    def save_options(self, data):
        values = self.validate_options(data)
        with self.lock:
            if self.status()['busy']:
                raise ValueError('更新检查或下载进行中，请稍后修改')
            atomic_json(self.root / 'updater_settings.json', values)
            self.options = values
        return self.status()

    def launch(self, function, phase):
        with self.lock:
            if self.status()['busy']:
                raise ValueError('已有更新操作正在执行')
            self.set_state(phase=phase, error=None, progress=0)
            def work():
                try:
                    function()
                except Exception as exc:
                    self.set_state(phase='error', error=str(exc), message='更新未完成，现有程序和任务记录未被下载操作修改。')
            self.thread = threading.Thread(target=work, name='release-updater', daemon=True)
            self.thread.start()
        return self.status()

    def check(self):
        def work():
            network = self.network_factory(self.options['proxy'])
            releases = []
            for page in range(1, 4):
                data = json.loads(network.small(f'https://api.github.com/repos/{REPOSITORY}/releases?per_page=30&page={page}'))
                if not isinstance(data, list):
                    raise ValueError('GitHub 没有返回有效的发行版本列表')
                releases.extend(data)
                if len(data) < 30:
                    break
            release = pick_release(releases, self.version, self.options['preview'])
            self.set_state(phase='available' if release else 'current', release=release,
                           message=f'发现新版本 {release["version"]}' if release else '当前更新通道没有更高版本。')
        return self.launch(work, 'checking')

    def download(self):
        release = self.state.get('release')
        if not release:
            raise ValueError('请先检查并找到新版本')
        def work():
            self.staged = None
            network = self.network_factory(self.options['proxy'])
            digest = checksum_for(network.small(release['checksums_url'], 128 * 1024).decode('utf-8-sig'), release['name'])
            directory = self.directory / uuid.uuid4().hex
            directory.mkdir(parents=True)
            archive = directory / 'package.zip'
            if shutil.disk_usage(self.root).free < release['bytes'] * 4 + 500 * 1024 * 1024:
                raise ValueError('磁盘空间不足，需留出下载、展开和备份空间')
            network.download(release['url'], archive, release['bytes'],
                             lambda got, total: self.set_state(progress=round(got / total * 100, 1), message=f'正在下载 {got // 1048576} / {total // 1048576} MB'))
            self.prepare(archive, digest, release['version'])
        return self.launch(work, 'downloading')

    def prepare(self, archive, digest, expected_version=None):
        self.staged = None
        self.set_state(phase='verifying', message='正在校验更新包…')
        details = unpack(archive, Path(archive).parent / 'payload', digest, self.version, expected_version)
        self.staged = {'directory': str(Path(archive).parent), **details}
        atomic_json(Path(archive).parent / 'staged.json', self.staged)
        self.set_state(phase='ready', message=f'{details["version"]} 已下载并校验。停止脚本后可安装；配置和记录保留。', progress=100)

    def import_file(self, archive, digest):
        return self.launch(lambda: self.prepare(archive, checked_digest(digest)), 'verifying')

    def install(self, app, server, client_pid=None):
        """Called under the HTTP mutation lock; never stop a running game task."""
        if app.running() or (getattr(app, 'thread', None) and app.thread.is_alive()):
            raise ValueError('请先停止脚本并等待收尾完成，再安装更新')
        if not getattr(sys, 'frozen', False) or Path(sys.executable).resolve() != self.root / 'mumu-panel.exe':
            raise ValueError('源码运行时只支持检查和下载；安装更新请使用发行版 EXE')
        if not self.staged or self.status()['busy']:
            raise ValueError('没有已校验的更新包')
        from .installer import process_image
        if client_pid is not None and (type(client_pid) is not int or client_pid == os.getpid() or
                                       process_image(client_pid) != Path(sys.executable).resolve()):
            raise ValueError('桌面窗口进程不属于当前安装')
        directory = Path(self.staged['directory'])
        helper = directory / 'update-helper.exe'
        shutil.copy2(sys.executable, helper)
        plan = {**self.staged, 'root': str(self.root), 'parent_pid': os.getpid(), 'client_pid': client_pid,
                'port': server.server_address[1], 'old_exe_sha256': sha256(sys.executable)}
        atomic_json(directory / 'install-plan.json', plan)
        try:
            fd = os.open(self.directory / 'install.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise ValueError('还有未完成的安装；请查看 .updates 中的上次结果和备份')
        with os.fdopen(fd, 'w') as f:
            f.write(str(directory))
        try:
            subprocess.Popen([str(helper), 'apply-update', '--plan', str(directory / 'install-plan.json')],
                             cwd=self.root, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception:
            (self.directory / 'install.lock').unlink(missing_ok=True)
            raise
        self.installing = True
        self.set_state(phase='installing', message='安装器正在等待控制面板退出；安装后脚本保持停止。')
        threading.Thread(target=lambda: (time.sleep(1), server.shutdown()), daemon=True).start()
        return {'installing': True, 'close_desktop': client_pid is not None}
