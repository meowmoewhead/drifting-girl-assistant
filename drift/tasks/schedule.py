"""ALAS-style persisted NextRun for scheduled tasks; event tasks stay reactive."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading

from drift.tasks.daily import BEIJING
from drift.tasks.features import (DAILY_CONDITIONS, FEATURES, FEATURE_BY_ID,
                                  REALTIME_CONDITIONS, execution_mode)


def aware_time(value):
    if not isinstance(value, str):
        raise ValueError('下次开启时间必须为带时区的日期时间')
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError('下次开启时间格式无效') from exc
    if moment.tzinfo is None:
        raise ValueError('下次开启时间必须包含时区')
    return moment.astimezone(BEIJING)


class TaskSchedule:
    def __init__(self, root, serial, clock=None):
        self.path = Path(root) / 'task_schedule.json'
        self.serial = serial
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.lock = threading.RLock()
        self.entries = {}
        self.active = set()
        self.warning = None
        try:
            saved = json.loads(self.path.read_text(encoding='utf-8-sig'))
            if not isinstance(saved, dict) or saved.get('version') != 1 or not isinstance(saved.get('tasks'), dict):
                raise ValueError('Invalid task schedule')
            if saved.get('serial') == serial:
                entries = {}
                for feature, entry in saved['tasks'].items():
                    if (feature not in FEATURE_BY_ID
                            or execution_mode(feature) == 'realtime' and feature != 'stone_farm'):
                        continue
                    if not isinstance(entry, dict):
                        raise ValueError('Invalid task entry')
                    if (execution_mode(feature) == 'realtime'
                            and not (feature == 'stone_farm' and entry.get('reason') == 'retry')):
                        continue
                    aware_time(entry['next_run'])
                    entries[feature] = entry
                self.entries = entries
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, KeyError):
            self.warning = '任务时间记录无法读取，将重新核对任务；每日完成和购买记录仍独立保留。'

    def now(self):
        moment = self.clock()
        if moment.tzinfo is None:
            raise ValueError('Task schedule clock needs a timezone')
        return moment.astimezone(BEIJING)

    def _write(self, feature, entry):
        updated = deepcopy(self.entries)
        updated[feature] = entry
        payload = {'version': 1, 'serial': self.serial, 'tasks': updated}
        from drift.core.persistence import atomic_write_text
        atomic_write_text(self.path, json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
        self.entries = updated

    def next_run(self, feature, daily=None):
        with self.lock:
            entry = self.entries.get(feature)
            if execution_mode(feature) == 'realtime':
                # A classified failed handoff may back off without turning
                # ordinary realtime triggers into periodically scheduled work.
                return (aware_time(entry['next_run']) if feature == 'stone_farm'
                        and entry and entry.get('reason') == 'retry' else None)
            if execution_mode(feature) == 'daily':
                # Older daily completion proved only the free card. Recheck the
                # bonus once without deleting that record or a chosen NextRun.
                if (feature == 'daily_reward' and daily and daily.get('bonus_pending')
                        and daily.get('free_completed') and entry
                        and entry.get('reason') == 'complete'):
                    return None
                if daily and daily['completed']:
                    reset = aware_time(daily.get('next_eligible',daily['next_reset']))
                    # A manual postponement may be later than reset, never earlier.
                    return max(reset, aware_time(entry['next_run'])) if entry and entry.get('reason') == 'manual' else reset
                # A missed game-day reset releases yesterday's retry delay.
                # Explicitly chosen future times remain valid across that boundary.
                if entry and daily and entry.get('period') != daily['period'] and entry.get('reason') != 'manual':
                    entry = None
            when=aware_time(entry['next_run']) if entry else None
            if daily and daily.get('not_before'):
                minimum=aware_time(daily['not_before'])
                return max(when,minimum) if when else minimum
            return when

    def ready(self, feature, daily=None):
        when = self.next_run(feature, daily)
        return when is None or when <= self.now()

    def refresh_intervals(self, previous_rules, rules):
        """Rebase completed timed-task cooldowns; never renew reserved retries.

        Manual dates, daily reset dates, active attempts, and failed retries all
        keep their recorded times. Only a completed timed task's interval may
        be recomputed from its original completion instant.
        """
        previous = {r['id']: r for r in previous_rules}
        current = {r['id']: r for r in rules}
        with self.lock:
            for feature in FEATURES:
                key = feature['id']
                if execution_mode(key) != 'timed' or key in self.active:
                    continue
                entry = self.entries.get(key)
                if not entry or entry.get('reason') != 'complete' or not entry.get('last_finished'):
                    continue
                old = next((previous[r] for r in feature['rules'] if r in previous), None)
                new = next((current[r] for r in feature['rules'] if r in current), None)
                if new is None or old is not None and old.get('interval') == new.get('interval'):
                    continue
                moment = aware_time(entry['last_finished']) + timedelta(seconds=new['interval'])
                self._write(key, {**entry, 'next_run': moment.isoformat()})

    def set_next(self, feature, value, daily=None):
        if feature not in FEATURE_BY_ID or execution_mode(feature) == 'realtime':
            raise ValueError('实时触发任务不设置下次开启时间')
        moment = self.now() if value is None else aware_time(value)
        with self.lock:
            if feature in self.active:
                raise ValueError('正在进行的任务完成后才能修改下次开启时间')
            entry = {**self.entries.get(feature, {}), 'next_run': moment.isoformat(),
                     'reason': 'manual', 'outcome': None, 'detail': ''}
            if daily:
                entry['period'] = daily['period']
            self._write(feature, entry)

    def begin(self, feature, interval, daily=None):
        if execution_mode(feature) == 'realtime':
            return
        with self.lock:
            now = self.now()
            entry = {**self.entries.get(feature, {}), 'last_started': now.isoformat(),
                     'next_run': (now + timedelta(seconds=interval)).isoformat(), 'reason': 'attempt'}
            if daily:
                entry['period'] = daily['period']
            # Reserve a retry before the workflow proceeds, including across a restart.
            self._write(feature, entry)
            self.active.add(feature)

    def finish(self, feature, interval, daily=None, success=True, retry_seconds=None, *, outcome=None, detail='', controller=None):
        if execution_mode(feature)=='realtime' and feature!='stone_farm':
            self.active.discard(feature)
            return
        from drift.core.outcome import result, controller_result, Outcome
        finished=(controller_result(controller,success,retry_seconds=retry_seconds,detail=detail)
                  if controller is not None and outcome is None else
                  result(success,retry_seconds=retry_seconds,outcome=outcome,detail=detail))
        success=finished.kind is Outcome.COMPLETE or finished.kind is Outcome.SKIPPED and finished.checked
        with self.lock:
            now = self.now()
            when = (aware_time(daily.get('next_eligible',daily['next_reset'])) if success and daily and daily['completed'] else
                    now + timedelta(seconds=finished.delay(interval)))
            entry = {**self.entries.get(feature, {}), 'last_finished': now.isoformat(),
                     'next_run': when.isoformat(), 'reason': 'complete' if success else 'retry',
                     'outcome': finished.kind.value, 'detail': finished.detail}
            previous=self.entries.get(feature,{})
            if previous.get('reason')=='manual' and feature not in self.active:
                entry['next_run'],entry['reason']=previous['next_run'],'manual'
            if daily:
                entry['period'] = daily['period']
            self._write(feature, entry)
            self.active.discard(feature)

    def release(self):
        # Stopping never erases the reserved retry or marks unfinished work complete.
        with self.lock:
            self.active.clear()

    def step_controller(self, feature, controller, frame, invoke):
        """Gate an idle controller only; open panels and active returns always finish."""
        was_active, cycles = controller.active, controller.cycles
        recovering = controller.panel(frame)
        if feature == 'parallel_skills':
            recovering = (recovering or controller.modal(frame) or
                          controller.owns_purchase_prompt() and controller.purchase_prompt(frame))
        if not was_active:
            when = self.next_run(feature)
            remaining = max(0, (when - self.now()).total_seconds()) if when else 0
            if remaining and not recovering:
                import time
                controller.next_check_at = time.monotonic() + remaining
                return False
            # NextRun is authoritative. Local timers still control steps within a cycle.
            controller.next_check_at = 0
        owned = invoke(controller.step, frame)
        if not was_active and controller.active:
            self.begin(feature, controller.interval)
        elif was_active and not controller.active:
            import time
            retry = max(0, controller.next_check_at - time.monotonic())
            self.finish(feature, controller.interval, success=controller.cycles > cycles,
                        retry_seconds=retry if controller.cycles == cycles else None,controller=controller)
        return owned

    def status(self, config, daily_states, running=False):
        with self.lock:
            now = self.now()
            rules = {rule['id']: rule for rule in config['rules']}
            result = {}
            for feature in FEATURES:
                key = feature['id']
                enabled = all(rules.get(rule, {}).get('enabled', False) for rule in feature['rules'])
                mode = execution_mode(key)
                daily = daily_states.get(key)
                when = self.next_run(key, daily)
                remaining = max(0, (when - now).total_seconds()) if when else 0
                status = ('disabled' if not enabled else 'running' if key in self.active and running else
                          'waiting' if remaining > 0 else 'realtime' if mode == 'realtime' else 'pending')
                result[key] = {'mode': mode, 'status': status, 'next_run': when.isoformat() if when else None,
                               'seconds_until': round(remaining, 1) if mode != 'realtime' or when else None,
                               'timezone': 'Asia/Shanghai',
                               # 实时功能用画面条件,每日功能用刷新说明(两者都取自 features.py)
                               'condition': REALTIME_CONDITIONS.get(key) or DAILY_CONDITIONS.get(key),
                               'reason': self.entries.get(key, {}).get('reason'),
                               'outcome': self.entries.get(key, {}).get('outcome'),
                               'detail': self.entries.get(key, {}).get('detail')}
            return result

# Keep legacy task logic; delegate every record read/write to shared storage.
from drift.core.persistence import ReliablePath as Path

def refresh_intervals(self, previous_rules, rules):
    """Rebase completed timed-task cooldowns; never renew reserved retries.

    Manual dates, daily reset dates, active attempts, and failed retries all
    keep their recorded times. Only a completed timed task's interval may
    be recomputed from its original completion instant.
    """
    previous = {r['id']: r for r in previous_rules}
    current = {r['id']: r for r in rules}
    with self.lock:
        for feature in FEATURES:
            key = feature['id']
            if execution_mode(key) != 'timed' or key in self.active:
                continue
            entry = self.entries.get(key)
            if not entry or entry.get('reason') != 'complete' or not entry.get('last_finished'):
                continue
            old = next((previous[r] for r in feature['rules'] if r in previous), None)
            new = next((current[r] for r in feature['rules'] if r in current), None)
            if new is None or old is not None and old.get('interval') == new.get('interval'):
                continue
            moment = aware_time(entry['last_finished']) + timedelta(seconds=new['interval'])
            self._write(key, {**entry, 'next_run': moment.isoformat()})

TaskSchedule.refresh_intervals = refresh_intervals

# Shared reliability layer; retain all other deployed methods.
def _write(self, feature, entry):
    updated = deepcopy(self.entries)
    updated[feature] = entry
    payload = {'version': 1, 'serial': self.serial, 'tasks': updated}
    from drift.core.persistence import atomic_write_text
    atomic_write_text(self.path, json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    self.entries = updated

TaskSchedule._write = _write

def finish(self, feature, interval, daily=None, success=True, retry_seconds=None, *, outcome=None, detail=''):
    from drift.core.outcome import result, Outcome
    finished=result(success,retry_seconds=retry_seconds,outcome=outcome,detail=detail)
    success=finished.kind is Outcome.COMPLETE
    with self.lock:
        now = self.now()
        when = (aware_time(daily.get('next_eligible',daily['next_reset'])) if success and daily and daily['completed'] else
                now + timedelta(seconds=finished.delay(interval)))
        entry = {**self.entries.get(feature, {}), 'last_finished': now.isoformat(),
                 'next_run': when.isoformat(), 'reason': 'complete' if success or finished.kind is Outcome.SKIPPED else 'retry',
                 'outcome': finished.kind.value, 'detail': finished.detail}
        if daily:
            entry['period'] = daily['period']
        self._write(feature, entry)
        self.active.discard(feature)

TaskSchedule.finish = finish

def status(self, config, daily_states, running=False):
    with self.lock:
        now = self.now()
        rules = {rule['id']: rule for rule in config['rules']}
        result = {}
        for feature in FEATURES:
            key = feature['id']
            enabled = all(rules.get(rule, {}).get('enabled', False) for rule in feature['rules'])
            mode = execution_mode(key)
            daily = daily_states.get(key)
            when = self.next_run(key, daily)
            remaining = max(0, (when - now).total_seconds()) if when else 0
            status = ('disabled' if not enabled else 'running' if key in self.active and running else
                      'waiting' if remaining > 0 else 'realtime' if mode == 'realtime' else 'pending')
            result[key] = {'mode': mode, 'status': status, 'next_run': when.isoformat() if when else None,
                           'seconds_until': round(remaining, 1) if mode != 'realtime' or when else None,
                           'timezone': 'Asia/Shanghai',
                           # 实时功能用画面条件,每日功能用刷新说明(两者都取自 features.py)
                           'condition': REALTIME_CONDITIONS.get(key) or DAILY_CONDITIONS.get(key),
                           'reason': self.entries.get(key, {}).get('reason'),
                           'outcome': self.entries.get(key, {}).get('outcome'),
                           'detail': self.entries.get(key, {}).get('detail')}
        return result

TaskSchedule.status = status



def finish(self, feature, interval, daily=None, success=True, retry_seconds=None, *, outcome=None, detail='', controller=None):
    if execution_mode(feature)=='realtime' and feature!='stone_farm':
        self.active.discard(feature)
        return
    from drift.core.outcome import result, controller_result, Outcome
    finished=(controller_result(controller,success,retry_seconds=retry_seconds,detail=detail)
              if controller is not None and outcome is None else
              result(success,retry_seconds=retry_seconds,outcome=outcome,detail=detail))
    success=finished.kind is Outcome.COMPLETE or finished.kind is Outcome.SKIPPED and finished.checked
    with self.lock:
        now = self.now()
        when = (aware_time(daily.get('next_eligible',daily['next_reset'])) if success and daily and daily['completed'] else
                now + timedelta(seconds=finished.delay(interval)))
        entry = {**self.entries.get(feature, {}), 'last_finished': now.isoformat(),
                 'next_run': when.isoformat(), 'reason': 'complete' if success else 'retry',
                 'outcome': finished.kind.value, 'detail': finished.detail}
        previous=self.entries.get(feature,{})
        if previous.get('reason')=='manual' and feature not in self.active:
            entry['next_run'],entry['reason']=previous['next_run'],'manual'
        if daily:
            entry['period'] = daily['period']
        self._write(feature, entry)
        self.active.discard(feature)

TaskSchedule.finish = finish


def set_next(self, feature, value, daily=None):
    if feature not in FEATURE_BY_ID or execution_mode(feature) == 'realtime':
        raise ValueError('实时触发任务不设置下次开启时间')
    moment = self.now() if value is None else aware_time(value)
    with self.lock:
        if feature in self.active:
            raise ValueError('正在进行的任务完成后才能修改下次开启时间')
        entry = {**self.entries.get(feature, {}), 'next_run': moment.isoformat(),
                 'reason': 'manual', 'outcome': None, 'detail': ''}
        if daily:
            entry['period'] = daily['period']
        self._write(feature, entry)

TaskSchedule.set_next = set_next


def step_controller(self, feature, controller, frame, invoke):
    """Gate an idle controller only; open panels and active returns always finish."""
    was_active, cycles = controller.active, controller.cycles
    recovering = controller.panel(frame)
    if feature == 'parallel_skills':
        recovering = (recovering or controller.modal(frame) or
                      controller.owns_purchase_prompt() and controller.purchase_prompt(frame))
    if not was_active:
        when = self.next_run(feature)
        remaining = max(0, (when - self.now()).total_seconds()) if when else 0
        if remaining and not recovering:
            import time
            controller.next_check_at = time.monotonic() + remaining
            return False
        # NextRun is authoritative. Local timers still control steps within a cycle.
        controller.next_check_at = 0
    owned = invoke(controller.step, frame)
    if not was_active and controller.active:
        self.begin(feature, controller.interval)
    elif was_active and not controller.active:
        import time
        retry = max(0, controller.next_check_at - time.monotonic())
        self.finish(feature, controller.interval, success=controller.cycles > cycles,
                    retry_seconds=retry if controller.cycles == cycles else None,controller=controller)
    return owned

TaskSchedule.step_controller = step_controller


# Module-local compatibility for every authoritative record read/write.
from drift.core.persistence import ReliablePath as Path
