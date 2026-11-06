"""Byte-preserving retirement shared by independent task journals."""
import json
from pathlib import Path
from drift.core.persistence import atomic_write_text, atomic_write_bytes, read_bytes, record_lock


def read_record(path, label):
    path = Path(path)
    if not path.exists():
        return None, None
    raw = read_bytes(path)
    record = json.loads(raw.decode('utf-8-sig'))
    if (not isinstance(record, dict) or record.get('version') != 1
            or type(record.get('pending')) is not bool):
        raise ValueError(f'{label}记录格式无效，原文件保留')
    return raw, record


def archive_bytes(path, raw, label):
    path = Path(path)
    with record_lock(path):
        if path.exists():
            if read_bytes(path) != raw:
                raise RuntimeError(f'{label}记录归档冲突，未清除原记录')
        else:
            atomic_write_bytes(path,raw,expected=None)


def archive_and_replace(path, raw, updated, archive, digest, label, retry_name):
    """Keep the exact failed record/approval and reject concurrent changes."""
    archive_bytes(archive, raw, label)
    retry_path = path.with_name(retry_name)
    retry_raw = retry_path.read_bytes() if retry_path.exists() else None
    retry = json.loads(retry_raw.decode('utf-8-sig')) if retry_raw else None
    matching = isinstance(retry, dict) and retry.get('failed_sha256') == digest
    if matching:
        archive_bytes(archive.with_name(f'{digest}-retry.json'), retry_raw, label)
    if path.read_bytes() != raw:
        raise ValueError(f'归档期间{label}记录已变化，未清除')
    atomic_write_text(path, json.dumps(updated, ensure_ascii=False, indent=2)+'\n', expected=raw)
    if matching and retry_path.exists() and retry_path.read_bytes() == retry_raw:
        retry_path.unlink()
