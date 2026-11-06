"""Bounded, durable record writes shared by tasks and configuration.

Only filesystem sharing failures are retried. A game action is never retried
here, and callers still have to reserve their input before sending it.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid
import stat

_locks = tuple(threading.RLock() for _ in range(64))
_UNSET = object()
RETRY_SECONDS = .8


@contextmanager
def record_lock(path):
    key = os.path.normcase(str(Path(path).resolve()))
    lock = _locks[hash(key) % len(_locks)]
    with lock:
        yield


def sharing_retry(operation, *, timeout=RETRY_SECONDS):
    """Do not turn disk-full, missing files or corrupt data into silent success."""
    deadline = time.monotonic() + timeout
    delay = .02
    while True:
        try:
            return operation()
        except PermissionError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(.16, delay * 2)


class RecordUnavailable(RuntimeError):
    """A locked authority record must not be mistaken for a missing record."""


def read_bytes(path):
    try:
        return sharing_retry(Path(path).read_bytes)
    except PermissionError as exc:
        raise RecordUnavailable('记录暂时无法读取，保留原文件并停止输入：'+Path(path).name) from exc


def read_text(path, encoding='utf-8-sig', errors='strict'):
    return read_bytes(path).decode(encoding or 'utf-8-sig', errors or 'strict')


def atomic_write_bytes(path, data, *, expected=_UNSET):
    """Unique same-directory temp + fsync + replace; failed writes keep old bytes.

    expected is a cooperative compare-and-swap for retirement operations. It is
    checked again before every replace attempt, including sharing retries.
    """
    path = Path(path)
    if not isinstance(data, bytes):
        raise TypeError('Record payload must be bytes')
    with record_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            fd, name = sharing_retry(lambda: tempfile.mkstemp(
                dir=path.parent, prefix='.' + path.name + '.', suffix='.tmp'))
            temporary = Path(name)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())

            def replace():
                if expected is not _UNSET:
                    try:
                        current = read_bytes(path)
                    except FileNotFoundError:
                        current = None
                    if current != expected:
                        raise ValueError('Record changed during update; original preserved')
                os.replace(temporary, path)

            sharing_retry(replace)
        finally:
            if temporary is not None:
                try:
                    sharing_retry(lambda: temporary.unlink(missing_ok=True))
                except OSError:
                    # Never mask a failed reservation with a temp-cleanup error.
                    pass


def atomic_write_text(path, text, *, encoding='utf-8', expected=_UNSET):
    atomic_write_bytes(path, text.encode(encoding), expected=expected)


def atomic_write_json(path, value, *, expected=_UNSET):
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                      expected=expected)


class ReliablePath(type(Path())):
    """Module-local compatibility for legacy journal code, not a global patch.

    Normal Path operations keep their meaning. Only JSON staging suffixes use
    unique names; legacy write/replace pairs then commit through the same
    durable writer. This covers older packaged methods without replacing their
    task logic. Imports opt in explicitly as ``ReliablePath as Path``.
    """
    def read_bytes(self):
        return read_bytes(self)

    def _record_stat(self,follow_symlinks=True):
        try:
            return sharing_retry(lambda:Path(self).stat(follow_symlinks=follow_symlinks))
        except PermissionError as exc:
            raise RecordUnavailable('记录路径暂时无法核对，不能当作文件不存在：'+self.name) from exc

    def exists(self,*,follow_symlinks=True):
        try:self._record_stat(follow_symlinks);return True
        except (FileNotFoundError,NotADirectoryError):return False

    def is_file(self,*,follow_symlinks=True):
        try:return stat.S_ISREG(self._record_stat(follow_symlinks).st_mode)
        except (FileNotFoundError,NotADirectoryError):return False

    def is_dir(self,*,follow_symlinks=True):
        try:return stat.S_ISDIR(self._record_stat(follow_symlinks).st_mode)
        except (FileNotFoundError,NotADirectoryError):return False

    def read_text(self, encoding=None, errors=None, newline=None):
        text=read_text(self,encoding or 'utf-8-sig',errors or 'strict')
        return text.replace('\r\n','\n').replace('\r','\n') if newline is None else text

    def write_bytes(self, data):
        payload=bytes(data)
        atomic_write_bytes(self,payload)
        return len(payload)

    def write_text(self, data, encoding=None, errors=None, newline=None):
        if not isinstance(data,str):raise TypeError('data must be str')
        if newline not in (None,'','\n'):
            data=data.replace('\n',newline)
        atomic_write_bytes(self,data.encode(encoding or 'utf-8',errors or 'strict'))
        return len(data)

    def with_suffix(self, suffix):
        if self.suffix=='.json' and suffix in ('.tmp','.json.tmp','.retire.tmp'):
            return self.parent/('.'+self.name+'.drift-tmp-'+uuid.uuid4().hex+'.tmp')
        return super().with_suffix(suffix)

    def replace(self, target):
        if '.drift-tmp-' not in self.name:
            sharing_retry(lambda:os.replace(self,target))
            return type(self)(target)
        try:
            atomic_write_bytes(target,read_bytes(self))
        finally:
            sharing_retry(lambda:Path(self).unlink(missing_ok=True))
        return type(self)(target)
