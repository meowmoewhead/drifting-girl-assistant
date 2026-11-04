"""Compatibility shim: the implementation lives in drift.web.watchdog.

The packaged ``mumu-panel.exe watchdog`` is the supported entry point (the device
has no standalone Python). Tests keep importing ``tools.run_watchdog``.
"""
from drift.web.watchdog import (  # noqa: F401  re-exported on purpose
    BASE_URL, BUDGET, COOLDOWN, NORMAL_STOPS, WINDOW, default_log_dir, history_lock,
    main, read_history, read_status, save_history, should_restart, start_run)

__all__ = ['BASE_URL', 'BUDGET', 'COOLDOWN', 'NORMAL_STOPS', 'WINDOW', 'default_log_dir',
           'history_lock', 'main', 'read_history', 'read_status', 'save_history',
           'should_restart', 'start_run']


if __name__ == '__main__':
    raise SystemExit(main())
