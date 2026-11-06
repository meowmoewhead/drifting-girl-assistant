"""One HTTP-only run check; scheduling and panel process recovery are separate.

No device imports or inputs. The only mutation on the panel is /api/start after
all hard-stop guards pass. Restart requests are persisted before POST because a
lost response does not prove that the panel rejected the start.

Lives under ``drift`` so the packaged ``mumu-panel.exe watchdog`` can run it on
the runtime host, where no standalone Python exists. ``dev_tools/run_watchdog.py`` stays as a
compatibility re-export for the test suite.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

from drift.core.paths import ROOT


COOLDOWN = 180
BUDGET = 5
WINDOW = 3600
BASE_URL = 'http://127.0.0.1:8765'
NORMAL_STOPS = {'cycle_complete', 'observation_complete', 'click_limit', 'game_stuck'}


def default_log_dir():
    """Program directory in a frozen build, repository root from source."""
    return ROOT / 'logs'


def should_restart(status, now, history):
    """Return a decision without I/O or mutation; history contains epoch seconds."""
    if status.get('running') is not False:
        return False, 'running_or_unknown'
    if not status.get('last_error'):
        return False, 'no_error'
    if status.get('last_stop_request'):
        return False, 'user_stop'
    if status.get('stop_reason') in NORMAL_STOPS:
        return False, 'stop_reason:' + status['stop_reason']
    if history and now - max(history) < COOLDOWN:
        return False, 'cooldown'
    if sum(now - stamp < WINDOW for stamp in history) >= BUDGET:
        return False, 'budget_exhausted'
    return True, 'hard_stop:' + str(status['last_error'])


def read_status(base_url):
    with urlopen(Request(base_url + '/'), timeout=10) as response:
        html = response.read().decode('utf-8')
    match = re.search(r"TOKEN='([0-9a-f]+)'", html)
    if match is None:
        raise ValueError('panel session token missing')
    token = match.group(1)
    request = Request(base_url + '/api/status', headers={'X-Session-Token': token})
    with urlopen(request, timeout=10) as response:
        status = json.load(response)
    if not isinstance(status, dict):
        raise ValueError('invalid panel status')
    return token, status


def start_run(base_url, token):
    request = Request(base_url + '/api/start', data=b'{"preview": false}',
                      headers={'X-Session-Token': token, 'Content-Type': 'application/json'},
                      method='POST')
    with urlopen(request, timeout=10) as response:
        result = json.load(response)
    if not isinstance(result, dict) or result.get('ok') is not True:
        raise ValueError('panel did not acknowledge start')


@contextmanager
def history_lock(path):
    """OS-owned lock releases even if a scheduled invocation is terminated."""
    with path.open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == 'nt':
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def read_history(path):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}
    if (not isinstance(data, dict) or any(
            not isinstance(stamps, list) or any(
                isinstance(stamp, bool) or not isinstance(stamp, (int, float))
                or not math.isfinite(stamp) for stamp in stamps)
            for stamps in data.values())):
        raise ValueError('invalid restart history')
    return data


def save_history(path, data):
    from drift.core.persistence import atomic_write_text
    atomic_write_text(path, json.dumps(data, ensure_ascii=False))


def main(argv=None, *, log_dir=None, fetcher=None, poster=None, clock=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='print/log only; do not POST or consume budget')
    parser.add_argument('--base-url', default=BASE_URL)
    args = parser.parse_args(argv)
    base_url = args.base_url.rstrip('/')
    log_dir = Path(log_dir) if log_dir is not None else default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    fetcher = fetcher or read_status
    poster = poster or start_run
    now = (clock or time.time)()
    history_path = log_dir / 'run-watchdog-history.json'
    restart, reason, exit_code = False, 'history_unavailable', 0
    try:
        with history_lock(log_dir / 'run-watchdog.lock'):
            try:
                token, status = fetcher(base_url)
            except (OSError, ValueError, HTTPException) as error:
                reason, exit_code = 'panel_unavailable:' + str(error), 2
            else:
                history = read_history(history_path)
                restart, reason = should_restart(status, now, history.get(base_url, []))
                if restart and not args.dry_run:
                    history[base_url] = [stamp for stamp in history.get(base_url, []) if now - stamp < WINDOW]
                    history[base_url].append(now)
                    save_history(history_path, history)
                    try:
                        poster(base_url, token)
                    except (OSError, ValueError, HTTPException) as error:
                        reason += ';panel_unavailable:' + str(error)
                        exit_code = 2
    except (OSError, ValueError) as error:
        # Busy lock, damaged history, or an unwritable budget must not grant
        # another restart. Other tasks/panels retain their recorded budgets.
        restart, reason = False, 'history_unavailable:' + str(error)
    decision = '重启' if restart else '不动'
    line = f"{datetime.fromtimestamp(now).isoformat(timespec='seconds')} 决策={decision} 原因={reason}"
    if args.dry_run:
        line += ' dry_run=true'
    line = ' '.join(line.splitlines())
    with (log_dir / 'run-watchdog.log').open('a', encoding='utf-8') as output:
        output.write(line + '\n')
    print(line)
    return exit_code


from drift.core.persistence import ReliablePath as Path

if __name__ == '__main__':
    raise SystemExit(main())

# Keep legacy task logic; delegate every record read/write to shared storage.

def save_history(path, data):
    from drift.core.persistence import atomic_write_text
    atomic_write_text(path, json.dumps(data, ensure_ascii=False))



# Module-local compatibility for every authoritative record read/write.
from drift.core.persistence import ReliablePath as Path
