"""Archive a meter execution without erasing its restoration debt or budgets."""
import hashlib
import math
from pathlib import Path
import time

from drift.core.journal import read_record as read_journal, archive_and_replace


def read_record(path):
    return read_journal(path, '标尺')


def expiry(record, now):
    if not record or not record['pending']:
        return False
    deadline = record.get('deadline')
    retry = record.get('restore_retry')
    if isinstance(retry, dict):
        deadline = retry.get('deadline')
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError('标尺截止时间无效，原文件保留')
    return now >= deadline


def status(root):
    try:
        raw, record = read_record(Path(root)/'meter_calibration_state.json')
        retirement = record.get('retirement') if record else None
        return dict(pending=bool(record and record['pending']),
                    sha256=hashlib.sha256(raw).hexdigest() if raw else None,
                    expired=expiry(record, time.time()), retirement=retirement,
                    recovery_required=bool(retirement and not retirement.get('released')))
    except (OSError, ValueError) as exc:
        return dict(pending=False, error=str(exc))


def retire(path, expected_sha256, *, source, reason='', interval=60):
    path = Path(path)
    raw, record = read_record(path)
    if raw is None or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('标尺记录已变化，请刷新后重试')
    if not record['pending']:
        return dict(retired=False, reason='没有待处理的标尺执行')
    now = time.time()
    expired = expiry(record, now)
    if source == 'startup_expired' and not expired:
        return dict(retired=False, reason='标尺执行尚未到期')
    if source not in ('startup_expired', 'manual_clear', 'workflow_limit'):
        raise ValueError('未知标尺归档来源')
    archive = path.parent/'logs/meter-calibration-retired'/f'{expected_sha256}.json'
    # Keep the original training/range cache, proof, attempts and deadlines.
    # Clearing execution ownership cannot authorize replay of that execution.
    retirement = dict(at=now, source=source, reason=reason, sha256=expected_sha256,
        archive=str(archive), expired=expired, released=False,
        had_inputs=any(record.get('attempts', {}).values())
                   or any(record.get('sailing_attempts', {}).values()))
    updated = dict(record, pending=False, next_check=now+interval, retirement=retirement)
    updated['failure_history'] = [*record.get('failure_history', []), dict(retirement)]
    archive_and_replace(path, raw, updated, archive, expected_sha256, '标尺', 'meter_calibration_retry.json')
    return dict(retired=True, sha256=expected_sha256, archive=str(archive))


