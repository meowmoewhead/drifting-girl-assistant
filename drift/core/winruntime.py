"""Windows:优先使用系统自带的 MSVC C++ 运行库,避开打进包里的旧版本。

问题
----
PyInstaller 会把构建机 PATH 上先找到的 ``msvcp140.dll`` 打进包。实测构建机的
``C:\\ProgramData\\Miniconda3`` 排在 System32 之前,PyInstaller 收进去的是
**14.16.27012.6**(2017 版);而目标机 System32 是 14.5x,MuMu 自带 14.38。
一个进程里同时驻留三份 C++ 运行库,跨 CRT 释放对象时会以
``0xc0000005``(MSVCP140.dll +0x192a0)崩溃 —— 104 上 16:30 与 18:35 两次崩溃
的错误模块、偏移、异常码完全相同,且两次都发生在按钮点击之后的几秒内。

做法
----
在导入 numpy / cv2 / onnxruntime **之前**,按绝对路径把 System32 的同名 DLL 先载入。
Windows 对已载入的同名模块会复用,因此后续按名字加载时拿到的是系统那一份,
进程里只留一个 CRT 版本。失败时静默跳过(非 Windows、或缺文件的机器不受影响)。
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

_PRELOADED = False

# System32 里的这些名字按依赖顺序加载:msvcp 依赖 vcruntime。
_LIBRARIES = (
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
)


def prefer_system_msvc_runtime(quiet=False, log_path=None):
    """预加载系统 MSVC 运行库;重复调用只会执行一次。返回已加载的文件名列表。

    ``log_path`` 给定时,把结果追加写进该文件(打包运行时 stderr 可能被父进程丢弃,
    而这条记录是"进程用的是系统运行库"的唯一证据)。
    """
    global _PRELOADED
    if _PRELOADED or os.name != "nt":
        return []
    _PRELOADED = True
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    loaded = []
    try:
        import ctypes
    except ImportError:  # pragma: no cover - CPython 一定有 ctypes
        return []
    for name in _LIBRARIES:
        path = system32 / name
        if not path.exists():
            continue
        try:
            ctypes.WinDLL(str(path))
            loaded.append(name)
        except OSError:
            continue
    message = f"[runtime] 预加载系统 MSVC 运行库: {', '.join(loaded) or '无'}"
    if loaded and not quiet:
        sys.stderr.write(message + "\n")
    if log_path is not None:
        try:
            from datetime import datetime

            target = Path(log_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}\n")
        except OSError:
            pass
    return loaded


