"""Retire ownership of a handoff without erasing its inputs or replaying it."""
import hashlib
import math
from pathlib import Path
import time
from drift.core.journal import read_record as read_journal, archive_and_replace

# A retired handoff blocks replay of its working contexts for at most this
# long; afterwards the same context may be re-confirmed and released. Aligned
# with RetryableStage(retry_seconds=1800) for the stone farm.
BACKOFF_SECONDS = 1800


def blocked_contexts(record):
    """Real working positions of the failed handoff; sentinels are not contexts.

    The cycle journal carries entry_rank=0 for "entry rank not observed";
    0 is not a legal in-game RANK (1..40), so [to_stage, 0] must never be
    blocked. Both retirement constructors share this rule so they cannot
    drift apart again.
    """
    contexts = []
    for stage_key, rank_key in (('from_stage', 'rank'), ('to_stage', 'entry_rank')):
        stage, rank = record.get(stage_key), record.get(rank_key)
        if type(stage) is int and type(rank) is int and stage >= 1 and 1 <= rank <= 40:
            contexts.append([stage, rank])
    return contexts


def read_record(root):
    path = Path(root) / 'stone_farm_state.json'
    raw, record = read_journal(path, '交接')
    return path, raw, record


def expiry(record, now):
    if not record or not record['pending']:
        return []
    fields = ['deadline', 'phase_deadline']
    if str(record.get('phase', '')).startswith('first_'):
        fields.append('first_observation_deadline' if record.get('phase') == 'first_verify'
                      and 'first_observation_deadline' in record else 'first_deadline')
    if record.get('phase') in ('stop_fishing', 'return', 'return_confirm'):
        fields.append('return_deadline')
    for key in fields:
        value = record.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('交接截止时间无效，原文件保留')
    return [key for key in fields if now >= record[key]]


def status(root):
    try:
        _, raw, record = read_record(root)
        return dict(pending=bool(record and record['pending']),
                    sha256=hashlib.sha256(raw).hexdigest() if raw else None,
                    phase=record.get('phase') if record else None,
                    expired=bool(expiry(record, time.time())),
                    retirement=record.get('retirement') if record else None)
    except (OSError, ValueError) as exc:
        return dict(pending=False, error=str(exc))


def retire(root, expected_sha256, *, source):
    path, raw, record = read_record(root)
    if raw is None or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('交接记录已变化，请刷新后重试')
    if not record['pending']:
        return dict(retired=False, reason='没有待处理交接')
    now = time.time()
    expired = expiry(record, now)
    if source == 'startup_expired' and not expired:
        return dict(retired=False, reason='交接尚未到期')
    if source not in ('startup_expired', 'manual_clear'):
        raise ValueError('未知归档来源')
    archive = path.parent / 'logs' / 'stone-farm-retired' / (expected_sha256 + '.json')
    updated = dict(record)
    updated.update(pending=False, phase='idle', retirement=dict(
        at=now, source=source, sha256=expected_sha256, archive=str(archive),
        previous_phase=record['phase'], expired_fields=expired,
        blocked_contexts=blocked_contexts(record), backoff_seconds=BACKOFF_SECONDS,
        released=False))
    archive_and_replace(path, raw, updated, archive, expected_sha256, '交接', 'stone_farm_retry.json')
    return dict(retired=True, sha256=expected_sha256, archive=str(archive), previous_phase=record['phase'])


