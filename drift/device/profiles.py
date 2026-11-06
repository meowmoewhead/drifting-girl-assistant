"""Preserve per-instance journals when explicitly changing the selected device.

The active instance keeps the existing root paths, so task implementations and
their original deadlines/input budgets do not change. Other instances' exact
bytes are parked in a private profile. A small write-ahead record permits
rollback if the panel exits before config.json is committed.
"""
from contextlib import contextmanager
import base64
import hashlib
import json
from pathlib import Path
import re


RECORD = re.compile(r"(?:[a-z][a-z0-9_]*_(?:state|retry)|task_schedule)\.json\Z")


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _snapshot(root):
    return {p.name: base64.b64encode(p.read_bytes()).decode("ascii")
            for p in root.iterdir() if p.is_file() and RECORD.fullmatch(p.name)}


def _restore(root, records):
    # Validate the entire snapshot before changing a file. Paths from a saved
    # profile can never escape the known root or name arbitrary config/assets.
    if not isinstance(records, dict) or len(records) > 256:
        raise ValueError("模拟器任务记录无效")
    decoded = {}
    for name, raw in records.items():
        if not isinstance(name, str) or not RECORD.fullmatch(name) or not isinstance(raw, str):
            raise ValueError("模拟器任务记录文件名无效")
        decoded[name] = base64.b64decode(raw, validate=True)
        if len(decoded[name]) > 8 * 1024 * 1024:
            raise ValueError("模拟器任务记录过大")
        if (root / name).is_symlink():
            raise ValueError("模拟器任务记录不能是链接")
    for name, raw in decoded.items():
        tmp = (root / name).with_suffix(".json.device-tmp")
        tmp.write_bytes(raw)
        tmp.replace(root / name)
    for path in root.iterdir():
        if RECORD.fullmatch(path.name) and path.is_file() and path.name not in decoded:
            path.unlink()


def _folder(root):
    return Path(root) / "device-profiles"


def _profile(root, serial):
    return _folder(root) / (hashlib.sha256(serial.encode("utf-8")).hexdigest() + ".json")


def recover_switch(root):
    root = Path(root)
    marker = _folder(root) / "pending-switch.json"
    if not marker.is_file():
        return
    saved = json.loads(marker.read_text(encoding="utf-8"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8-sig"))
    if saved.get("version") != 1 or config["serial"] not in (saved.get("old_serial"), saved.get("new_serial")):
        raise ValueError("上次模拟器切换未完成，请检查连接配置；原记录仍保留")
    if config["serial"] == saved["old_serial"]:
        _restore(root, saved["before"])
    # New config is committed only AFTER all destination records were restored.
    marker.unlink()


@contextmanager
def switch_records(root, old_serial, new_serial):
    root = Path(root)
    if old_serial == new_serial:
        yield
        return
    if (_folder(root) / "pending-switch.json").exists():
        raise ValueError("上次模拟器切换尚未恢复，请重新打开面板")
    before = _snapshot(root)
    destination = _profile(root, new_serial)
    after = {}
    if destination.is_file():
        saved = json.loads(destination.read_text(encoding="utf-8"))
        if saved.get("version") != 1 or saved.get("serial") != new_serial:
            raise ValueError("目标模拟器任务记录与实例不一致")
        after = saved["records"]
    _write(_profile(root, old_serial), dict(version=1, serial=old_serial, records=before))
    marker = _folder(root) / "pending-switch.json"
    _write(marker, dict(version=1, old_serial=old_serial, new_serial=new_serial, before=before))
    try:
        _restore(root, after)
        yield
    except BaseException:
        _restore(root, before)
        marker.unlink()
        raise
    else:
        marker.unlink()

def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    from drift.core.persistence import atomic_write_text
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))



# Module-local compatibility for every authoritative record read/write.
from drift.core.persistence import ReliablePath as Path
