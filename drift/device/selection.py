"""Panel-facing device selection; changing targets never starts the worker."""
from copy import deepcopy

from drift.device import discovery
from drift.device.profiles import switch_records
from drift.perception.daily_bonus import supported_bonus
from drift.tasks.daily import DailySchedule
from drift.tasks.schedule import TaskSchedule


def list_devices(app, data):
    if not isinstance(data, dict) or set(data) - {"adb_path"}:
        raise ValueError("不支持的设备检测参数")
    if "adb_path" in data and not isinstance(data["adb_path"], str):
        raise ValueError("ADB 路径必须为文字")
    with app.lock:
        config = deepcopy(app.config)
    return discovery.discover(config, adb_path=data.get("adb_path"))


def select_device(app, data):
    if not isinstance(data, dict) or "serial" not in data or set(data) - {"serial", "adb_path"}:
        raise ValueError("请选择要连接的模拟器")
    if "adb_path" in data and not isinstance(data["adb_path"], str):
        raise ValueError("ADB 路径必须为文字")
    with app.lock:
        app.idle_only()
        device = app.device
        native = getattr(device, "native_capture", None)
        if native is not None and native.capture_pending:
            raise ValueError("上一张截图尚未结束，请稍后再切换")
        io = getattr(device, "io_lock", None)
        if io is not None and not io.acquire(blocking=False):
            raise ValueError("设备正在截图，请稍后再切换")
        try:
            config = deepcopy(app.config)
            result = discovery.discover(config, adb_path=data.get("adb_path"))
            selected = discovery.choose(result, data["serial"])
            image = discovery.verify_frame(result["adb_path"], selected["serial"], config["resolution"])
            unchanged = selected["serial"] == config["serial"] and result["adb_path"] == config["adb_path"]
            if not unchanged:
                updated = dict(config, serial=selected["serial"], adb_path=result["adb_path"])
                replacement = app.device_factory(updated, app.stop_event)
                with switch_records(app.root, config["serial"], updated["serial"]):
                    daily = DailySchedule(app.root, updated["serial"])
                    daily.require_bonus = any(supported_bonus(r) and r["enabled"] for r in updated["rules"])
                    schedule = TaskSchedule(app.root, updated["serial"])
                    if daily.warning or schedule.warning:
                        raise ValueError(daily.warning or schedule.warning)
                    app.commit_config(updated)
                app.device = replacement
                app.daily_schedule, app.task_schedule = daily, schedule
                app.runtime = None
                app.frames.clear()
                app.state.update(mode="idle", stage="main", frame_id=None, last_error=None,
                                 frames=0, matches=0, clicks=0, recoveries=0,
                                 last_stop_request=None, stop_reason=None,
                                 numbers={}, warnings=[], parallel=None, equipment=None,
                                 backpack=None, skills=None, meter=None, stone=None, defense=None,
                                 pass_daily=None, guild_daily=None, restaurant=None, return_parallel=None,
                                 active_rule_ids=[], last_match=None, scoped=False)
            frame = app.frame_info(app.remember(image))
            app.event("已连接模拟器：" + selected["name"] + "（" + selected["serial"] + "）；点击开始后运行")
            return dict(serial=selected["serial"], name=selected["name"], frame=frame, changed=not unchanged)
        finally:
            if io is not None:
                io.release()
