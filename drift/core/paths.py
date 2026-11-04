"""运行期路径与上游版本标记。

打包(frozen)与源码运行两种形态都要能找到 web/ 与 assets/,所以这里集中判定一次。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 源码运行时是项目根目录;PyInstaller onefile 运行时是 exe 所在目录。
ROOT = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[2])
# --add-data 解包出来的临时目录(打包时 web/ 在这里)。
BUNDLE = Path(getattr(sys, "_MEIPASS", ROOT))
# vendor/alas 的提交号,面板与诊断都会展示,便于确认借用的上游版本。
UPSTREAM = "77f4d01fcd2b0acab05a4d89260e8a0a9cd03d21"


def read_version(root: Path = ROOT) -> str:
    """从程序目录读取唯一版本来源;源码检出或缺失文件时仍可启动。"""
    try:
        return (Path(root) / "VERSION").read_text(encoding="utf-8-sig").strip() or "0.0.0+dev"
    except (OSError, UnicodeError):
        return "0.0.0+dev"


VERSION = read_version()
