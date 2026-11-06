"""Windowed executable bootstrap; legacy CLI output still supports redirection."""
import io
import os
from pathlib import Path
import sys


def redirected_stream(which, handle_id):
    if os.name == 'nt':
        try:
            import ctypes
            import msvcrt
            kernel = ctypes.WinDLL('kernel32')
            kernel.GetStdHandle.restype = ctypes.c_void_p
            handle = kernel.GetStdHandle(handle_id)
            if handle and handle != ctypes.c_void_p(-1).value:
                descriptor = msvcrt.open_osfhandle(handle, os.O_WRONLY)
                return io.TextIOWrapper(os.fdopen(os.dup(descriptor), 'wb'), encoding='utf-8', line_buffering=True)
        except (OSError, ValueError):
            pass
    root = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    try:
        (root / 'logs').mkdir(exist_ok=True)
        return (root / 'logs' / ('panel-' + which + '.log')).open('a', encoding='utf-8', buffering=1)
    except OSError:
        return open(os.devnull, 'w', encoding='utf-8')


if sys.stdout is None:
    sys.stdout = redirected_stream('output', -11)
if sys.stderr is None:
    sys.stderr = redirected_stream('errors', -12)
if sys.stdin is None:
    sys.stdin = open(os.devnull)

from drift.web.cli import main

if __name__ == '__main__':
    main()
