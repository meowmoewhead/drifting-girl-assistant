"""Read-only ADB instance discovery, following ALAS's explicit serial selection.

Only the chosen ADB binary is used. No server restart, game launch, port sweep,
emulator launch, or input is needed to list and check an instance.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from PIL import Image
import numpy as np


ADB_NAMES = {"adb", "adb.exe", "adb_server.exe", "nox_adb.exe", "hd-adb.exe"}
SERIAL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:\[\]%-]{0,127}\Z")
MAX_DEVICES = 16


def run_adb(argv, *, timeout=5):
    try:
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True,
                                timeout=timeout,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as exc:
        raise ValueError("设备检查超时，请确认模拟器已启动") from exc
    except OSError as exc:
        raise ValueError("无法运行 ADB，请检查程序路径") from exc
    error = result.stderr.decode("utf-8", "replace").strip()
    if result.returncode or any(line.lower().startswith(("error:", "adb: error:"))
                                for line in error.splitlines()):
        raise ValueError(error[:300] or "设备连接检查失败")
    return result.stdout


def _installed_adb():
    """Bounded install-directory lookup; never recursively scan users' disks."""
    if os.name != "nt":
        return
    import winreg
    key = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
            try:
                with winreg.OpenKey(hive, key, 0, winreg.KEY_READ | view) as parent:
                    for index in range(min(winreg.QueryInfoKey(parent)[0], 1024)):
                        try:
                            with winreg.OpenKey(parent, winreg.EnumKey(parent, index)) as child:
                                name = winreg.QueryValueEx(child, "DisplayName")[0]
                                if not any(part in str(name).lower() for part in
                                           ("mumu", "ldplayer", "雷电", "bluestacks", "nox", "逍遥")):
                                    continue
                                folder = Path(winreg.QueryValueEx(child, "InstallLocation")[0])
                                for relative in ("nx_main/adb.exe", "adb.exe", "HD-Adb.exe", "nox_adb.exe"):
                                    yield folder / relative
                        except OSError:
                            continue
            except OSError:
                continue


def find_adb(configured=""):
    requested = str(configured or "").strip().strip('"')
    if requested and Path(requested).is_file():
        if Path(requested).name.lower() not in ADB_NAMES:
            raise ValueError("请选择 adb.exe 或模拟器自带的 ADB 程序")
        return str(Path(requested).resolve())
    # Like ALAS, prefer a known installation's ADB over running several servers.
    candidates = [shutil.which("adb"), *_installed_adb()]
    for item in candidates:
        if item and Path(item).is_file() and Path(item).name.lower() in ADB_NAMES:
            return str(Path(item).resolve())
    raise ValueError("没有找到 ADB。请填写模拟器安装目录内的 adb.exe 路径后重新检测")


def parse_devices(raw):
    devices = []
    seen = set()
    for line in raw.decode("utf-8", "replace").splitlines():
        fields = line.split()
        if len(fields) < 2 or not SERIAL.fullmatch(fields[0]):
            continue
        serial, state = fields[:2]
        if state not in ("device", "offline", "unauthorized", "recovery", "sideload", "bootloader") or serial in seen:
            continue
        seen.add(serial)
        attrs = dict(item.split(":", 1) for item in fields[2:] if ":" in item)
        devices.append(dict(serial=serial, state=state, model=attrs.get("model", "").replace("_", " ")))
        if len(devices) >= MAX_DEVICES:
            break
    return devices


def parse_size(raw):
    matches = re.findall(r"(?:Physical|Override) size:\s*(\d+)x(\d+)", raw.decode("utf-8", "replace"))
    if not matches:
        raise ValueError("没有读到画面尺寸")
    size = list(map(int, matches[-1]))
    if not all(1 <= n <= 4096 for n in size):
        raise ValueError("画面尺寸无效")
    return size


def mumu_names(adb, runner=run_adb):
    manager = Path(adb).with_name("MuMuManager.exe")
    if not manager.is_file():
        return {}
    try:
        rows = json.loads(runner([str(manager), "info", "-v", "all"], timeout=3).decode("utf-8-sig"))
        rows = rows.values() if isinstance(rows, dict) else rows
        names = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            host, port = row.get("adb_host", "127.0.0.1"), row.get("adb_port")
            if host not in ("localhost", "127.0.0.1") or not str(port).isdigit():
                continue
            if not 1 <= int(port) <= 65535:
                continue
            for prefix in ("localhost", "127.0.0.1"):
                names[f"{prefix}:{port}"] = str(row.get("name") or "MuMu")[:100]
        return names
    except (ValueError, TypeError, OSError, AttributeError):
        # An unavailable manager (including a different Windows desktop session)
        # must never hide a real, online ADB device or guess its instance index.
        return {}


def discover(config, *, adb_path=None, runner=run_adb, finder=find_adb):
    adb = finder(config.get("adb_path", "") if adb_path is None else adb_path)
    devices = parse_devices(runner([adb, "devices", "-l"], timeout=5))
    names = mumu_names(adb, runner)
    expected = list(config["resolution"])

    def inspect(row):
        row = dict(row, resolution=None, compatible=False, selected=row["serial"] == config["serial"])
        row["name"] = names.get(row["serial"]) or row["model"] or row["serial"]
        row["reason"] = {"offline": "未连接", "unauthorized": "请先在模拟器中允许调试"}.get(row["state"], "设备尚未就绪")
        if row["state"] == "device":
            try:
                row["resolution"] = parse_size(runner([adb, "-s", row["serial"], "shell", "wm", "size"], timeout=2))
                row["compatible"] = row["resolution"] == expected
                row["reason"] = "可连接" if row["compatible"] else "请将模拟器画面设为 " + "×".join(map(str, expected))
            except ValueError as exc:
                row["reason"] = str(exc)
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
        devices = list(pool.map(inspect, devices))
    return dict(adb_path=adb, devices=devices, expected_resolution=expected,
                current_serial=config["serial"], automatic_available=sum(d["compatible"] for d in devices) == 1)


def choose(result, serial):
    if serial == "auto":
        available = [d for d in result["devices"] if d["compatible"]]
        if len(available) != 1:
            raise ValueError("检测到多个可连接实例，请选择一个" if available else "没有可连接实例，请先启动模拟器")
        return available[0]
    if not isinstance(serial, str) or not SERIAL.fullmatch(serial):
        raise ValueError("模拟器实例编号无效")
    row = next((d for d in result["devices"] if d["serial"] == serial), None)
    if row is None or not row["compatible"]:
        raise ValueError(row["reason"] if row else "所选实例已断开，请重新检测")
    return row


def verify_frame(adb, serial, resolution, runner=run_adb):
    """Verify a NEW screenshot before persisting the explicit device choice."""
    raw = runner([adb, "-s", serial, "exec-out", "screencap", "-p"], timeout=5)
    try:
        with Image.open(BytesIO(raw)) as image:
            if list(image.size) != list(resolution):
                raise ValueError("模拟器尺寸已变化，请重新检测")
            image.load()
            return np.asarray(image.convert("RGB")).copy()
    except (OSError, SyntaxError) as exc:
        raise ValueError("没有取得有效的新截图，未更换连接") from exc
