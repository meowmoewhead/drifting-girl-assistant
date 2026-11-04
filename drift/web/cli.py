"""命令行入口:``serve`` / ``diagnose`` / ``click-once`` 三个子命令。

沿用原 ``app.py`` 的命令和默认值,增加版本显示与缺失配置时的空闲面板。
``ROOT`` / ``VERSION`` / ``UPSTREAM`` 取自 :mod:`drift.core.paths`,``number()`` 取自
:mod:`drift.core.util`。

``Clicker`` 在 :func:`main` 内部延迟导入(只来自 :mod:`drift.runtime`):``app.py`` 会
``from drift.web.cli import main``,若在模块顶部导入装配层就会形成循环,也会让
``drift.web`` 在导入期就拖上任务层。本模块**不** import ``app``,因此不依赖任何
运行期 ``app`` 属性绑定;``app.py`` 那边原路径的 ``app.main`` 仍照旧可用。

由 ``app.py`` 转发:``app.main`` 即本模块的 :func:`main`;
``python app.py serve``、``mumu-panel.exe serve`` 等原入口不变。

补丁语义说明:``main`` 内的 ``time`` / ``json`` / ``UPSTREAM`` 是本模块的全局名,
``patch('drift.web.cli.time')`` 生效;``patch('app.time')`` 只影响 ``app`` 自己的全局,
不再影响 ``main``(原实现二者同为一个模块字典)。现有测试不覆盖 ``main``,故无影响。
"""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import time

from drift.core.paths import ROOT, UPSTREAM, VERSION
from drift.core.util import number
from drift.core.winruntime import prefer_system_msvc_runtime
from drift.web.server import CONFIG_MISSING_MESSAGE, MissingConfigApp, make_handler

__all__ = ["main"]


def main():
    # ``watchdog`` 必须在 argparse 之前分流:它自带 --dry-run / --base-url,与面板参数
    # 同名冲突；计划任务只把这个子命令交给打包后的 exe（运行设备无需独立 Python）。
    if sys.argv[1:2] == ["watchdog"]:
        from drift.web.watchdog import main as watchdog_main
        raise SystemExit(watchdog_main(sys.argv[2:]))
    parser = argparse.ArgumentParser(description="MaaFramework MuMu helper with ALAS NextRun")
    parser.add_argument("--version", action="version",
                        version=f"mumu-clicker {VERSION} (upstream {UPSTREAM[:12]})")
    parser.add_argument("command", choices=["serve", "diagnose", "click-once", "framework-check", "watchdog"], nargs="?", default="serve")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rule", help="click-once 的单个目标 ID")
    parser.add_argument("--timeout", type=float, default=20, help="click-once 最长等待秒数")
    args = parser.parse_args()
    # 必须在 numpy / cv2 / onnxruntime 之前执行;--version 不加载运行期或写日志。
    prefer_system_msvc_runtime(log_path=args.root / "logs" / "runtime.log")
    if args.command == 'framework-check':
        from drift.core.maa_engine import framework_check
        print(json.dumps(framework_check(),ensure_ascii=False),flush=True)
        return
    if args.command == "serve" and not (args.root / "config.json").exists():
        print(CONFIG_MISSING_MESSAGE, flush=True)
        app = MissingConfigApp(args.root)
    else:
        # 延迟导入装配层:缺少配置时也不构建设备或启动任何任务。
        from drift.runtime import Clicker
        app = Clicker(args.root)
    if args.command == "click-once":
        timeout = number(args.timeout, 1, 60)
        selected = next((r for r in app.config["rules"] if r["id"] == args.rule), None)
        if selected is None or selected.get("stage", "main") != "main":
            parser.error("click-once 必须指定一个主界面目标 --rule")
        # Isolate the requested action in memory; never rewrite enabled states on disk.
        app.config["rules"] = [dict(selected, enabled=True, stop_after=True,
                                    stop_message="单项目标已点击一次，验证运行已停止。")]
        try:
            app.start(preview=False)
            app.thread.join(timeout)
        finally:
            app.stop(source='click_once_finally')
            if app.thread:
                app.thread.join(22)
        result = app.status()
        if result["frame_id"]:
            path = app.root / "captures" / "click-once-before.png"
            path.write_bytes(app.frame_png(result["frame_id"]))
        (app.root / "captures" / "click-once.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result["clicks"] != 1 or result["last_error"] or app.running():
            raise SystemExit(1)
        return
    if args.command == "diagnose":
        started = time.perf_counter()
        state = app.device.command("get-state").decode().strip()
        frame = app.capture()
        path = app.root / "captures" / "latest.png"
        path.write_bytes(app.frame_png(frame["id"]))
        result = {"adb_state": state, "serial": app.config["serial"], **frame,
                  "elapsed_seconds": round(time.perf_counter() - started, 3),
                  "image": str(path), "alas_commit": UPSTREAM}
        (app.root / "captures" / "diagnostic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    print(f"MuMu ALAS panel: http://127.0.0.1:{server.server_address[1]} "
          f"| mumu-clicker {VERSION} (upstream {UPSTREAM[:12]})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop(source='server_finally')
        if app.thread:
            app.thread.join(timeout=22)
        server.server_close()
