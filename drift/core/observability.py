"""Bounded, in-memory diagnostics; never writes workflow journals or grants input."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import threading
import time


def stop_request_context(source, peer=None):
    # Keep code locations only: no frame locals, arguments, headers or tokens.
    caller = sys._getframe(2)
    stack = []
    try:
        while caller is not None and len(stack) < 5:
            stack.append(dict(file=Path(caller.f_code.co_filename).name,
                              function=caller.f_code.co_name, line=caller.f_lineno))
            caller = caller.f_back
    finally:
        del caller
    return dict(at=time.time(), source=str(source)[:64], peer=peer,
                pid=os.getpid(), thread=threading.current_thread().name, callers=stack)


class CandidateObservation:
    """Counts controller checks, not independently verified native frames."""
    def __init__(self, event, clock=time.monotonic):
        self.event, self.clock = event, clock
        self.record = None
        self.started = self.last_check = self.last_report = None
        self.key = None

    def begin(self, started_at, *, rank, stage, player, value, high):
        if self.record is not None and self.record['outcome'] is None:
            return
        self.started = self.last_check = self.last_report = self.clock()
        self.key = None
        self.record = dict(started_at=started_at, rank=rank, stage=stage,
                           player=player, value=value, high=high, checks=0,
                           reasons={}, last_reason=None, detail=None,
                           context_changes=0, blocking_transitions=0, max_check_gap_seconds=0.,
                           outcome=None)
        self._emit('begin')

    def check(self, reason, *, detail=None, key=None):
        if self.record is None or self.record['outcome'] is not None:
            return
        now = self.clock()
        row = self.record
        if reason not in row['reasons'] and len(row['reasons']) >= 24:
            reason = 'other'
        row['max_check_gap_seconds'] = round(max(row['max_check_gap_seconds'], now-self.last_check), 3)
        self.last_check = now
        if key is not None:
            if self.key is not None and key != self.key:
                row['context_changes'] += 1
            self.key = key
        if reason.startswith('runtime_') and reason != row['last_reason']:
            row['blocking_transitions'] += 1
        row['checks'] += 1
        row['reasons'][reason] = row['reasons'].get(reason, 0)+1
        row['last_reason'] = reason
        row['detail'] = str(detail)[:160] if detail is not None else None
        if now-self.last_report >= 60:
            self.last_report = now
            self._emit('waiting')

    def finish(self, outcome):
        if self.record is None or self.record['outcome'] is not None:
            return
        self.record['outcome'] = outcome
        self.record['elapsed_seconds'] = round(max(0., self.clock()-self.started), 3)
        self._emit('end')
        self.key = None

    def snapshot(self):
        if self.record is None:
            return None
        row = deepcopy(self.record)
        if row['outcome'] is None:
            row['elapsed_seconds'] = round(max(0., self.clock()-self.started), 3)
        return row

    def _emit(self, kind):
        try:
            self.event('标尺候选观测：'+json.dumps(dict(event=kind, **self.snapshot()),
                                               ensure_ascii=False, separators=(',', ':')))
        except Exception as exc:
            # Observability must not become a new reason to stop fishing.
            self.record['logging_error'] = type(exc).__name__
