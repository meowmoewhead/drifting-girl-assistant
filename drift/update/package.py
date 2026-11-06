"""Validate release archives and transact only explicitly allowed program files."""
from __future__ import annotations

from functools import total_ordering
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import zipfile

REPOSITORY = 'meowmoewhead/drifting-girl-assistant'
MAX_ARCHIVE = 600 * 1024 * 1024
MAX_EXPANDED = 1600 * 1024 * 1024
ROOT_FILES = frozenset({
    'mumu-panel.exe', 'VERSION', 'version.json', 'USER-GUIDE.md', 'LICENSE',
    'THIRD_PARTY_NOTICES.md', '启动.cmd',
    'start-panel.ps1', 'README.md', 'config/default.json',
})
RESOURCE_DIRS = frozenset({'assets', 'models', 'web', 'licenses', 'docs'})


@total_ordering
class Version:
    def __init__(self, value):
        match = re.fullmatch(r'v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?(?:\+([0-9A-Za-z.-]+))?', str(value))
        if not match:
            raise ValueError('版本号无效')
        self.value = str(value).removeprefix('v')
        self.core = tuple(map(int, match.group(1, 2, 3)))
        self.pre = match.group(4)
        if self.pre and any(not p or (p.isdigit() and len(p) > 1 and p[0] == '0') for p in self.pre.split('.')):
            raise ValueError('预发布版本号无效')

    def key(self):
        return self.core, self.pre is None, tuple((0, int(p)) if p.isdigit() else (1, p) for p in (self.pre or '').split('.'))

    def __lt__(self, other):
        return self.key() < other.key()

    def __eq__(self, other):
        return isinstance(other, Version) and self.key() == other.key()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_digest(value):
    if not isinstance(value, str) or not re.fullmatch('[a-fA-F0-9]{64}', value):
        raise ValueError('缺少有效的 SHA256 校验值')
    return value.lower()


def validate_windows_executable(path):
    """Reject corrupt/16-bit/wrong-architecture payloads before stopping anything."""
    with Path(path).open('rb') as f:
        header = f.read(64)
        if len(header) < 64 or header[:2] != b'MZ':
            raise ValueError('更新程序不是有效的 Windows 64 位程序')
        offset = struct.unpack_from('<I', header, 0x3c)[0]
        if not 64 <= offset <= 1024 * 1024:
            raise ValueError('更新程序的 PE 头无效')
        f.seek(offset)
        pe = f.read(96)
        if len(pe) < 96 or pe[:4] != b'PE\0\0' or struct.unpack_from('<H', pe, 4)[0] != 0x8664 or struct.unpack_from('<H', pe, 24)[0] != 0x20b:
            raise ValueError('更新程序必须为 Windows x64 可执行文件')


def program_path(name):
    """No traversal, ADS, Windows device names, links or mutable record paths."""
    if not isinstance(name, str) or '\\' in name or ':' in name or '\x00' in name:
        return False
    parts = name.split('/')
    if any(not p or p in ('.', '..') or p.rstrip(' .') != p or
           re.fullmatch(r'(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?', p) for p in parts):
        return False
    return name in ROOT_FILES or (len(parts) > 1 and parts[0] in RESOURCE_DIRS
                                  and parts[:2] != ['assets', 'custom'])


def contained(root, relative):
    root = Path(root).resolve()
    path = root.joinpath(*PurePosixPath(relative).parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError('文件路径越过程序目录')
    for part in [path, *path.parents]:
        if part == root:
            break
        if part.is_symlink() or (hasattr(part, 'is_junction') and part.is_junction()):
            raise ValueError('更新路径包含链接或联接点')
    return path


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def checksum_for(text, name):
    values = []
    for line in text.splitlines():
        match = re.fullmatch(r'([A-Fa-f0-9]{64})\s+\*?(.+)', line.strip())
        if match and match[2] == name:
            values.append(match[1].lower())
    if len(values) != 1:
        raise ValueError('校验清单没有唯一对应的 Windows 更新包')
    return values[0]


def unpack(archive, destination, digest, current_version, expected_version=None):
    archive, destination = Path(archive), Path(destination)
    if archive.stat().st_size > MAX_ARCHIVE or sha256(archive) != checked_digest(digest):
        raise ValueError('更新包大小或 SHA256 校验失败，未修改安装目录')
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        if len(entries) > 20000 or sum(i.file_size for i in entries) > MAX_EXPANDED:
            raise ValueError('更新包展开大小超过限制')
        files = {}
        seen = set()
        for entry in entries:
            name = entry.filename
            if '\\' in name or name.startswith('/') or ':' in name or any(p in ('.', '..') for p in name.split('/')):
                raise ValueError('更新包包含非法路径')
            if stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
                raise ValueError('更新包包含链接或加密成员')
            if not name.startswith('mumu-clicker/'):
                raise ValueError('不是本项目的 Windows 便携包')
            relative = name[len('mumu-clicker/'):]
            if not relative or entry.is_dir():
                continue
            if relative.casefold() in seen:
                raise ValueError('更新包有重复文件名')
            seen.add(relative.casefold())
            if relative in ('config.json', 'logs/.gitkeep'):
                continue  # Never replace the operator's configuration or logs.
            if not program_path(relative):
                raise ValueError('更新包含不允许替换的文件：' + relative)
            files[relative] = entry
        if not {'mumu-panel.exe', 'VERSION', 'version.json'} <= set(files):
            raise ValueError('更新包缺少程序或版本清单')
        if files['version.json'].file_size > 4 * 1024 * 1024:
            raise ValueError('版本清单过大')
        if files['VERSION'].file_size > 1024:
            raise ValueError('版本号文件过大')
        manifest = json.loads(z.read(files['version.json']))
        version = Version(manifest['version'])
        if version <= Version(current_version):
            raise ValueError('更新包版本不高于当前版本，拒绝降级或覆盖同版本')
        if expected_version is not None and version != Version(expected_version):
            raise ValueError('发行版本与包内版本不一致')
        if z.read(files['VERSION']).decode('utf-8-sig').strip() != version.value:
            raise ValueError('VERSION 与版本清单不一致')
        payload = manifest.get('payload_sha256')
        if manifest.get('update_schema') != 1 or not isinstance(payload, dict):
            raise ValueError('此发行包没有安全更新清单，请使用新版发行包')
        if set(payload) != set(files) - {'version.json'}:
            raise ValueError('更新文件列表与清单不一致')
        destination.mkdir(parents=True, exist_ok=False)
        for relative, entry in files.items():
            target = contained(destination, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(entry) as source, target.open('xb') as out:
                shutil.copyfileobj(source, out, 1024 * 1024)
            if relative != 'version.json' and sha256(target) != checked_digest(payload[relative]):
                raise ValueError('包内文件校验失败：' + relative)
        if sha256(destination / 'mumu-panel.exe') != checked_digest(manifest['exe_sha256']):
            raise ValueError('程序校验失败')
        validate_windows_executable(destination / 'mumu-panel.exe')
    return {'version': version.value, 'files': {n: sha256(destination / n) for n in files},
            'archive_sha256': checked_digest(digest)}


class Transaction:
    """Persist backups before writes; a partial or failed install is reversible."""
    def __init__(self, root, payload, backup, files):
        self.root, self.payload, self.backup = map(lambda p: Path(p).resolve(), (root, payload, backup))
        self.files = files
        self.journal = {'replaced': [], 'before': {}, 'phase': 'new'}

    def save(self):
        atomic_json(self.backup / 'transaction.json', self.journal)

    def apply(self):
        for name, digest in self.files.items():
            if not program_path(name) or sha256(contained(self.payload, name)) != checked_digest(digest):
                raise ValueError('待安装文件发生变化')
            contained(self.root, name)
        self.backup.mkdir(parents=True, exist_ok=False)
        # All backups complete before the first destination write.
        for name in self.files:
            target = contained(self.root, name)
            old = contained(self.backup / 'files', name)
            self.journal['before'][name] = sha256(target) if target.is_file() else None
            if target.exists():
                if not target.is_file():
                    raise ValueError('安装目标不是文件：' + name)
                old.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, old)
        self.journal['phase'] = 'applying'
        self.save()
        try:
            for name in self.files:
                target = contained(self.root, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                self.journal['replaced'].append(name)
                self.save()  # Record intent before replacing, including newly created files.
                temp = target.with_name(target.name + '.update-tmp')
                try:
                    shutil.copy2(contained(self.payload, name), temp)
                    os.replace(temp, target)
                finally:
                    temp.unlink(missing_ok=True)
            self.journal['phase'] = 'checking'
            self.save()
        except Exception:
            self.rollback()
            raise

    def rollback(self):
        for name in reversed(self.journal['replaced']):
            target = contained(self.root, name)
            if self.journal['before'][name] is None:
                target.unlink(missing_ok=True)
            else:
                old = contained(self.backup / 'files', name)
                if sha256(old) != self.journal['before'][name]:
                    raise ValueError('回滚备份校验失败')
                shutil.copy2(old, target)
        self.journal['phase'] = 'rolled_back'
        self.save()

    def commit(self):
        self.journal['phase'] = 'complete'
        self.save()
