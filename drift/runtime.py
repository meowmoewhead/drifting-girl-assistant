"""运行期装配:``Clicker``(应用容器 + 主循环)。
本文件是 ``app.py`` 中 ``Clicker`` 的机械搬运加 import 重写,行为逐字等价;
后续会把 ``run()`` 内各功能流程拆到 ``drift/tasks/`` 下的任务类里。
**注意**:本模块顶层的 ``time`` / ``subprocess`` / 各控制器类名必须保持模块级绑定 ——
测试大量使用 ``patch('app.time')``、``patch('app.ParallelWorld')`` 这类按模块路径替换的补丁,
而 ``app`` 与本模块共享这些名字(见 ``app.py`` 的转发说明),补丁必须对两边同时生效。
"""
from __future__ import annotations
from collections import OrderedDict, deque
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid
import cv2
import numpy as np
from PIL import Image
from alas_core.base import ModuleBase
from alas_core.button import Button
from features import FEATURE_BY_ID, RULE_FEATURE, RECOVERY_RULE_IDS, feature_states
from daily_schedule import DailySchedule, DAILY_ENTRIES, DAILY_RETURNS
from task_schedule import TaskSchedule
from meter_calibration import MeterCalibration
from drift.tasks.cruise_watchdog import CruiseWatchdog, CruiseView, fresh_cruise_input
from drift.tasks.cruise_gate import cruise_blockers as cruise_gate_blockers
from drift.tasks.raid_cast import RaidCastAssist
from drift.tasks.cruise_setup import CruiseSetup
from drift.tasks.cruise_options import from_rule as cruise_options, normalize as normalize_cruise
from drift.tasks.cruise_settings import CruiseSettings, SettingsView
from drift.tasks.stone_farm import StoneFarm
from drift.tasks import handoff_journal, meter_journal
from drift.tasks.meter_limits import MeterWorkflowLimit
from drift.tasks.optional_timeout import OptionalCleanup, CLOSES as OPTIONAL_CLOSES, apply_policy as optional_timeout_policy
from parallel_world import ParallelWorld
from equipment_upgrade import EquipmentUpgrade
from parallel_skills import ParallelSkills
from drift.tasks.boss_training import training_options
from drift.device.evidence import BossInputEvidence
from native_capture import MuMuCapture, NativeUnavailable
from frame_runtime import FrameRuntime
from backpack_sale import BackpackSale
from drift.tasks.game_recovery import GameRecovery, PACKAGE, game_state
from drift.ui.button import CroppedTemplateButton
from drift.ui.sidebar import adapt_sidebar
from drift.ui.resource_battle import adapt_resource_battle
from drift.ui.guild_selection import GuildSelectionGuard, GuildReturnGuard, SELECTORS as GUILD_SELECTORS
from drift.ui.popup_raid import PopupRaidReturn
from drift.ui.popup_cruise import PopupCruiseReturn
from drift.ui.mailbox_close import MailboxCloseGuard, CLOSES as MAILBOX_CLOSES
from drift.core.util import hsv_mask, number, png_bytes, rectangle
from drift.core.matcher import (observe_native_crown_return, observe_popup_equipment_return,
                                popup_reclick_decision, scan_rules)
from drift.core.exceptions import GameStuck, RetryableStage
from drift.tasks.optional_timeout import PotionAdRecovery
from drift.core.progress import ProgressGuard
from drift.core.observability import stop_request_context
from drift.core.pages import build
from drift.core.ui import Navigator, nav_requested
from drift.perception.daily_bonus import BONUS_ID, DailyBonusVisual, supported_bonus
from drift.tasks.pass_daily import PASS_ID, PassDaily
from drift.tasks.guild_daily import GUILD_ID, GuildDaily
from drift.tasks.restaurant_upgrade import RESTAURANT_ID, RestaurantUpgrade
from drift.tasks.return_parallel import ReturnParallel
from drift.perception.restaurant import CATEGORIES
from drift.core.winruntime import prefer_system_msvc_runtime

# 先载入系统 MSVC 运行库,再让 numpy / cv2 / onnxruntime 进来
prefer_system_msvc_runtime(quiet=True)
from drift.core.session import LoopState
def _default_device_factory(config, stop):
    """延迟取用设备层:迁移期回退到根模块,避免 app ↔ drift 循环导入。"""
    try:
        from drift.device.adb import AdbDevice
    except ImportError:  # pragma: no cover - 设备层就位后不可达
        from app import AdbDevice  # type: ignore[no-redef]
    return AdbDevice(config, stop)
from drift.core.logger import get_logger, setup_file_logging
ROOT = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parents[1]
BUNDLE = Path(getattr(sys, "_MEIPASS", None) or ROOT)
UPSTREAM = "77f4d01fcd2b0acab05a4d89260e8a0a9cd03d21"
# 一个步骤里最多按页面图发几次"退回主画面"的点击;用尽仍没回去就按原样停机。
NAV_CLICK_LIMIT = 3
# Repeated task faults may indicate a persistent unsafe condition. Keep the
# stop-on-failure protection after this many consecutive failed restaurant runs.
RESTAURANT_FAILURE_LIMIT = 3
# 搜索框至少比模板大这么多像素(ALAS 的 Button.match 默认 offset 是 30/3)。
# 以前把 search_area 写得和模板一样大,UI 位移几像素就整条规则失配。
SEARCH_SLACK_Y, SEARCH_SLACK_X = 30, 6


def search_with_slack(search, template, width, height, *, slack_y=SEARCH_SLACK_Y,
                      slack_x=SEARCH_SLACK_X):
    """给搜索框保底余量:与"模板外扩 slack"取并集,再夹到画面内。

    只在原有搜索框更小时才扩大,所以已经留足余量的规则完全不受影响。
    规则可以写 ``"search_slack": false`` 显式退出(例如同行重复控件需要严格范围)。
    """
    x1, y1, x2, y2 = template
    left = max(0, x1 - slack_x)
    top = max(0, y1 - slack_y)
    right = min(width, x2 + slack_x)
    bottom = min(height, y2 + slack_y)
    return (min(search[0], left), min(search[1], top),
            max(search[2], right), max(search[3], bottom))
_PATCHABLE = (
    "time", "subprocess", "png_bytes", "rectangle",
    "ParallelWorld", "EquipmentUpgrade", "ParallelSkills", "BackpackSale",
    "MeterCalibration", "StoneFarm", "RaidCastAssist", "CruiseSetup", "MuMuCapture", "FrameRuntime", "BossInputEvidence",
    "GameRecovery", "DailySchedule", "TaskSchedule",
)
def _lookup(name):
    """取可补丁名字的当前值:优先 app 侧(测试 patch('app.X') 会改它)。
    用 id() 比较而不是 `is not`,因为补丁值可能是 MagicMock,其比较运算被改写。
    """
    module = sys.modules.get("app")
    if module is not None and name in _PATCHABLE:
        bound = getattr(module, name, None)
        mine = globals().get(name)
        if bound is not None and id(bound) != id(mine):
            return bound
    return globals().get(name)
def _time():
    """时钟:patch('app.time') 也要影响本模块的时间判断。"""
    return _lookup("time")


def fresh_run_state(*, mode, active_rule_ids, revision, scoped):
    """每一轮运行开始时要重置的面板状态。

    ``last_stop_request`` 必须一起清空：它是"当前运行"的控制记录，如果上一次运行
    的停止请求留下来，运行层守护会把它误判成用户主动停止而不再自动重启硬停的运行。
    """
    return dict(mode=mode, frames=0, matches=0, recoveries=0, clicks=0, last_match=None,
                last_error=None, stage="main", numbers={}, parallel=None, equipment=None,
                backpack=None, skills=None, meter=None, stone=None, defense=None,
                pass_daily=None, guild_selection=None, active_rule_ids=active_rule_ids,
                stop_reason=None, last_stop_request=None, applied_revision=revision,
                scoped=scoped)


class Clicker:
    def __init__(self, root=ROOT, device_factory=None):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.device_factory = device_factory or _default_device_factory
        self.config_revision = 0
        self.restaurant_failures = 0
        self.frames = OrderedDict()
        self.events = deque(maxlen=80)
        self.state = {"mode": "idle", "frames": 0, "matches": 0, "clicks": 0,
                      "recoveries": 0,
                      "warnings": [],
                      "last_match": None, "last_error": None, "frame_id": None, "last_stop_request": None,
                      "stage": "main", "numbers": {}, "parallel": None, "equipment": None, "backpack": None, "skills": None, "meter": None, "stone": None, "defense": None, "pass_daily": None,
                      "applied_revision": 0, "scoped": False}
        self.digit_reader = None
        self.runtime = None
        self._storage_banner = None
        self._storage_banner_loaded = False
        self.boss_evidence = None
        self.fishing_evidence = None
        self.portal_evidence = None
        self.sailing_evidence = None
        self.daily_evidence = None
        self.guild_evidence = None
        self.popup_evidence = None
        self.mailbox_evidence = None
        self.pass_evidence = None
        self.guild_daily_evidence = None
        self.restaurant_evidence = None
        self.config = json.loads((self.root / "config.json").read_text(encoding="utf-8-sig"))
        self.daily_schedule = _lookup("DailySchedule")(self.root, self.config['serial'])
        self.daily_schedule.require_bonus = any(supported_bonus(r) and r['enabled'] for r in self.config['rules'])
        self.task_schedule = _lookup("TaskSchedule")(self.root, self.config['serial'])
        width, height = self.config["resolution"]
        if not (1 <= width <= 4096 and 1 <= height <= 4096):
            raise ValueError("分辨率配置无效")
        number(self.config["poll_interval"], 0.1, 60)
        number(self.config["max_consecutive_clicks"], 1, 10000)
        for directory in ("assets", "captures", "logs"):
            (self.root / directory).mkdir(exist_ok=True)
        self.device = self.device_factory(self.config, self.stop_event)
        self.log = get_logger(f"mumu.{id(self)}")
        setup_file_logging(self.log, self.root / "logs")
        self.event("助手已就绪。选择功能后，可开始运行或识别预览。")
        if self.daily_schedule.warning:
            self.event(self.daily_schedule.warning)
        if self.task_schedule.warning:
            self.event(self.task_schedule.warning)
    def event(self, message):
        with self.lock:
            self.events.append({"time": datetime.now().strftime("%H:%M:%S"), "message": message})
        self.log.info(message)
    def running(self):
        return self.thread is not None and self.thread.is_alive()
    def idle_only(self):
        if self.running():
            raise ValueError("请先停止运行，再修改目标")
    def retry_stone_farm(self, data):
        with self.lock:
            self.idle_only()
            required = {'failed_sha256','allow_ordinary_cancel'}
            optional = ('allow_boss_abandon','allow_boss_abandon_menu','allow_continuous_cancel','allow_post_start_restore','allow_first_result_restore','allow_manual_rank_progress','allow_observed_lower_restore','allow_observed_cycle_restore','allow_return_entry_restore','allow_map_target_repair')
            read_only = [name for name in ('allow_observed_lower_restore','allow_observed_cycle_restore')
                         if data.get(name) is True]
            if read_only:
                if len(read_only) != 1 or set(data) != {'failed_sha256',read_only[0]}:
                    raise ValueError('只读接续只接受原失败哈希，不能同时授权其他游戏输入')
            elif (not required.issubset(data) or set(data)-required-set(optional)
                    or data['allow_ordinary_cancel'] is not True
                    or any(name in data and data[name] is not True for name in optional)):
                raise ValueError('需要明确允许普通鱼安全停钓及原失败记录哈希')
            options = {name:True for name in optional if data.get(name) is True}
            result = _lookup('StoneFarm').request_retry(self.root, data['failed_sha256'], **options)
            self.event('已登记一次降阶重试授权；原失败记录、次数和截止时间未修改')
            return result
    def clear_stone_farm(self, data):
        with self.lock:
            self.idle_only()
            if set(data) != {'sha256'} or not isinstance(data['sha256'], str):
                raise ValueError('请刷新交接记录后再清除')
            result = handoff_journal.retire(self.root, data['sha256'], source='manual_clear')
            if result['retired']:
                self.state['stone'] = None
                self.event('已归档并清除当前交接的待处理状态；历史输入和截止保留，不点击游戏、不自动重试本次交接')
            return result
    def clear_meter_calibration(self, data):
        with self.lock:
            self.idle_only()
            if set(data) != {'sha256'} or not isinstance(data['sha256'], str):
                raise ValueError('请刷新标尺记录后再清除')
            result = meter_journal.retire(self.root/'meter_calibration_state.json',
                data['sha256'], source='manual_clear')
            if result['retired']:
                self.state['meter'] = None
                self.event('已归档本次标尺执行；保留原次数和截止，只读等待实际恢复，不点击游戏或重开旧窗口')
            return result

    def retry_meter_calibration(self, data):
        with self.lock:
            self.idle_only()
            if set(data) != {'failed_sha256'}:
                raise ValueError('需要原标尺失败记录哈希')
            result = _lookup('MeterCalibration').request_retry(self.root,data['failed_sha256'])
            self.event('已登记一次标尺修复重试；原失败记录、次数和截止时间未修改')
            return result
    def persist(self, config=None):
        temporary = self.root / "config.json.tmp"
        temporary.write_text(json.dumps(self.config if config is None else config, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.root / "config.json")
    def commit_config(self, config):
        # Save a whole workflow atomically; a failed write leaves the running config intact.
        self.persist(config)
        self.config = config
        self.device.config = config
        self.daily_schedule.require_bonus = any(supported_bonus(r) and r['enabled'] for r in config['rules'])
        self.config_revision += 1
    def remember(self, image):
        identifier = uuid.uuid4().hex
        with self.lock:
            self.frames[identifier] = image.copy()
            while len(self.frames) > 8:
                self.frames.popitem(last=False)
            self.state["frame_id"] = identifier
        self.observe_storage_warning(image)
        return identifier
    def observe_storage_warning(self, frame):
        """Read-only banner observation; no task, input or success transition."""
        if not self._storage_banner_loaded:
            path = self.root/'assets/storage/storage-full-banner-source.png'
            if path.is_file():
                try:
                    with Image.open(path) as source:
                        reference = np.asarray(source.convert('RGB'))
                    if reference.shape != (1280,720,3):
                        raise ValueError('仓库横幅参考图尺寸不符')
                    self._storage_banner = CroppedTemplateButton(reference[238:262,294:416],
                        (274,208,436,292), [1], 'storage_full', source_area=(294,238,416,262),
                        required_templates=[((205,264,517,294), reference[264:294,205:517])])
                except (OSError, ValueError):
                    # Optional diagnostics cannot stop or change existing work.
                    self._storage_banner = None
            self._storage_banner_loaded = True
        visible = bool(self._storage_banner is not None and frame.shape == (1280,720,3)
                       and self._storage_banner.match(frame, similarity=.96))
        with self.lock:
            warnings = self.state.get('warnings', [])
            appeared = visible and 'storage_full' not in warnings
            self.state['warnings'] = [warning for warning in warnings if warning != 'storage_full']
            if visible:
                self.state['warnings'].append('storage_full')
        if appeared:
            self.event('保管箱已满：请到港口或农场清理仓库。')
    def capture(self):
        with self.lock:
            if self.running() and self.state["frame_id"]:
                return self.frame_info(self.state["frame_id"])
        return self.frame_info(self.remember(self.device.screenshot()))
    def frame_info(self, identifier):
        with self.lock:
            image = self.frames[identifier]
            height, width = image.shape[:2]
            return {"id": identifier, "width": width, "height": height}
    def frame_png(self, identifier):
        with self.lock:
            image = self.frames.get(identifier)
            if image is None:
                raise ValueError("截图已过期，请刷新截图")
        return _lookup("png_bytes")(image)
    def status(self):
        # Maa shutdown may wait for a callback that updates this app. Do not
        # hold the app lock while taking the framework's independent snapshot.
        performance = self.runtime.snapshot() if self.runtime is not None else None
        with self.lock:
            return {**self.state, "running": self.running(), "rules": self.config["rules"],
                    "features": self.feature_status(),
                    "settings": {key: self.config[key] for key in ("adb_path", "poll_interval", "max_consecutive_clicks")},
                    "settings_pending": self.running() and self.state["applied_revision"] != self.config_revision,
                    "numbers": dict(self.state["numbers"]),
                    "warnings": list(self.state.get('warnings', [])),
                    "serial": self.config["serial"], "resolution": self.config["resolution"],
                    "events": list(self.events), "upstream": UPSTREAM,
                    "performance": performance, "stone_record": handoff_journal.status(self.root),
                    "meter_record": meter_journal.status(self.root)}
    def feature_status(self):
        daily = self.daily_schedule.status()
        return feature_states(self.config, daily,
                              self.task_schedule.status(self.config, daily, self.running()))
    def change_task_time(self, data):
        with self.lock:
            self.idle_only()
            if set(data) != {'id', 'next_run'}:
                raise ValueError('需要任务 ID 和下次开启时间')
            feature = next((f for f in self.feature_status() if f['id'] == data['id']), None)
            if feature is None or not feature['available']:
                raise ValueError('任务不存在或规则不完整')
            self.task_schedule.set_next(data['id'], data['next_run'],
                                        self.daily_schedule.status().get(data['id']))
            self.event(f"已保存{feature['name']}的下次开启时间")
            return next(f for f in self.feature_status() if f['id'] == data['id'])
    def change_portal_mode(self, data):
        """Select one existing portal controller without rewriting its saved progress."""
        with self.lock:
            self.idle_only()
            if set(data) != {'mode'} or data['mode'] not in ('off', 'stone', 'legacy'):
                raise ValueError('次元石模式必须为关闭、降阶刷最后BOSS或旧版任务练级')
            updated = deepcopy(self.config)
            by_id = {rule['id']: rule for rule in updated['rules']}
            for feature_id, mode in (('stone_farm', 'stone'), ('parallel_world', 'legacy')):
                identifiers = FEATURE_BY_ID[feature_id]['rules']
                if any(key not in by_id for key in identifiers):
                    raise ValueError('次元石模式规则不完整，请先恢复缺少的规则')
                for key in identifiers:
                    by_id[key]['enabled'] = data['mode'] == mode
            self.commit_config(updated)
            label = {'off':'关闭', 'stone':'降阶刷最后BOSS', 'legacy':'旧版任务练级'}[data['mode']]
            self.event('已保存次元石循环模式：' + label + '；各项进度和武器强化设置保留')
            return {'mode': data['mode']}

    def change_feature(self, data):
        with self.lock:
            feature = FEATURE_BY_ID.get(data.get("id"))
            if feature is None:
                raise ValueError("功能不存在")
            cruise_change = feature['id'] == 'parallel_skills'
            if self.running() and self.state["scoped"] and not cruise_change:
                raise ValueError("请先停止单项测试，再调整功能")
            if not set(data) <= {"id", "enabled", "interval", "mana_threshold", "training", "growth_points", "targets", "cruise_settings"}:
                raise ValueError("不支持的功能设置")
            if "enabled" in data and not isinstance(data["enabled"], bool):
                raise ValueError("功能开关必须为布尔值")
            if feature['id'] == 'return_parallel' and 'interval' in data:
                raise ValueError('返回平行世界只支持功能开关')
            if "interval" in data or "mana_threshold" in data or "training" in data or "growth_points" in data or "targets" in data:
                if not cruise_change:
                    self.idle_only()
            updated = deepcopy(self.config)
            members = {r["id"]: r for r in updated["rules"] if r["id"] in feature["rules"]}
            if len(members) != len(feature["rules"]):
                raise ValueError("功能规则不完整，请先恢复缺少的规则")
            if "enabled" in data:
                if feature['id'] == 'return_parallel':
                    updated['return_parallel_enabled'] = data['enabled']
                for rule in members.values():
                    rule["enabled"] = data["enabled"]
            if "interval" in data:
                interval = number(data["interval"], 0.2, 259200 if feature['id']=='restaurant_upgrade' else
                                  86400 if feature['id'] in ('meter_calibration','guild_daily') else 3600)
                for rule in members.values():
                    if rule.get("stage", "main") == "main" or rule['id'] in ('driftinggirl_parallel_world', 'driftinggirl_equipment_upgrade', 'driftinggirl_backpack_sell', 'driftinggirl_parallel_skills', 'driftinggirl_meter_calibration', 'driftinggirl_stone_farm', RESTAURANT_ID):
                        rule["interval"] = interval
            if "mana_threshold" in data:
                value = data["mana_threshold"]
                if feature["id"] != "mana" or isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("法力阈值必须为整数")
                number(value, 1, 100)
                members["driftinggirl_mana_open"]["number_condition"]["value"] = value
                members["driftinggirl_mana_open"]["name"] = f"法力：左下数值低于 {value} 时打开技能"
            if "training" in data:
                if feature['id'] != 'parallel_world':
                    raise ValueError('练级设置仅用于平行世界')
                members['driftinggirl_parallel_world']['training'] = training_options(data['training'])
            if 'growth_points' in data:
                if feature['id'] != 'parallel_skills' or not isinstance(data['growth_points'], bool):
                    raise ValueError('技能点加速开关仅用于平行世界技能，且必须为布尔值')
                members['driftinggirl_parallel_skills']['growth_points'] = data['growth_points']
                if 'cruise_settings' in members['driftinggirl_parallel_skills']:
                    members['driftinggirl_parallel_skills']['cruise_settings']['skills']['growth_points']=data['growth_points']
            if 'cruise_settings' in data:
                if not cruise_change:
                    raise ValueError('巡航设置仅用于巡航功能')
                rule = members['driftinggirl_parallel_skills']
                options = normalize_cruise(data['cruise_settings'], growth_points=rule.get('growth_points',True))
                rule['cruise_settings'] = options
                rule['growth_points'] = options['skills']['growth_points']
            if 'targets' in data:
                if feature['id'] != 'restaurant_upgrade':
                    raise ValueError('强化类别设置仅用于餐品强化')
                targets = data['targets']
                if (not isinstance(targets, list)
                        or any(not isinstance(name, str) or name not in CATEGORIES for name in targets)):
                    raise ValueError('强化类别必须为字符串数组，且每项必须是九个餐品类别之一')
                members[RESTAURANT_ID]['targets'] = list(dict.fromkeys(targets))
            if cruise_change and 'cruise_settings' not in members['driftinggirl_parallel_skills']:
                members['driftinggirl_parallel_skills']['cruise_settings'] = cruise_options(members['driftinggirl_parallel_skills'])
            self.commit_config(updated)
            action = "已开启" if data.get("enabled") else "已关闭" if "enabled" in data else "已保存"
            suffix = "，返回主画面后生效" if self.running() else ""
            self.event(f"{action}：{feature['name']}{suffix}")
            return next(f for f in self.feature_status() if f["id"] == feature["id"])
    def change_settings(self, data):
        with self.lock:
            self.idle_only()
            if not data or not set(data) <= {"poll_interval", "max_consecutive_clicks"}:
                raise ValueError("不支持的运行设置")
            updated = deepcopy(self.config)
            if "poll_interval" in data:
                updated["poll_interval"] = number(data["poll_interval"], 0.1, 60)
            if "max_consecutive_clicks" in data:
                value = data["max_consecutive_clicks"]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("重复点击上限必须为整数")
                number(value, 1, 10000)
                updated["max_consecutive_clicks"] = value
            self.commit_config(updated)
            self.event("运行设置已保存")
    def add_rule(self, data):
        with self.lock:
            self.idle_only()
            image = self.frames.get(data["frame_id"])
            if image is None:
                raise ValueError("用于框选的截图已过期，请刷新后重新框选")
            height, width = image.shape[:2]
            area = _lookup("rectangle")(data["area"], width, height)
            x1, y1, x2, y2 = area
            patch = image[y1:y2, x1:x2]
            if min(patch.shape[:2]) < 8 or np.max(patch.std(axis=(0, 1))) < 2:
                raise ValueError("请选择至少 8×8 像素且包含图案的区域，避免纯色背景")
            name = str(data.get("name", "")).strip()[:80]
            if not name:
                raise ValueError("请输入目标名称")
            identifier = uuid.uuid4().hex
            rule = {"id": identifier, "name": name, "enabled": True,
                    "asset": f"assets/{identifier}.png", "area": list(area),
                    "search_area": [0, 0, width, height],
                    "threshold": number(data.get("threshold", 0.92), 0.5, 1),
                    "interval": number(data.get("interval", 1), 0.2, 3600),
                    "stable_frames": int(number(data.get("stable_frames", 2), 1, 10))}
            (self.root / rule["asset"]).write_bytes(_lookup("png_bytes")(image))
            self.config["rules"].append(rule)
            self.persist()
            self.event(f"已保存目标：{name}")
            return rule
    def change_rule(self, data):
        with self.lock:
            feature_id = RULE_FEATURE.get(data.get("id"))
            if feature_id:
                if data.get("delete"):
                    raise ValueError("内置流程请使用功能开关，不能删除单个步骤")
                return self.change_feature({"id": feature_id, "enabled": data["enabled"]})
            self.idle_only()
            rule = next((r for r in self.config["rules"] if r["id"] == data["id"]), None)
            if rule is None:
                raise ValueError("目标不存在")
            if data.get("delete"):
                self.config["rules"].remove(rule)
            else:
                rule["enabled"] = bool(data["enabled"])
            self.persist()
    def load_rule(self, rule):
        mailbox_guarded = False
        guild_confirm_titles = {
            'driftinggirl_guild_potion_confirm': ('potion', [309,410,410,438]),
            'driftinggirl_guild_material_confirm': ('material', [284,410,438,438]),
        }
        guild_confirm = guild_confirm_titles.get(rule.get('id'))
        if (guild_confirm is not None
                and rule.get('asset') == f'assets/guild-shop/{guild_confirm[0]}-confirm.png'
                and rule.get('required_source_areas') ==
                    [[228,573,257,603],[338,576,528,605],guild_confirm[1]]):
            # The centered coin/question move with the number of price digits.
            # Keep the product title, green confirmation button and independent
            # confirmation-price equality with this run's quantity selection.
            rule = deepcopy(rule)
            rule['required_source_areas'] = [guild_confirm[1]]
        if (rule.get('id') in {'driftinggirl_guild_close',
                              'driftinggirl_guild_potion_select','driftinggirl_guild_potion_skip',
                              'driftinggirl_guild_material_select','driftinggirl_guild_material_skip'}
                and rule.get('asset') == 'assets/guild-shop/guild.png'
                and rule.get('source_crop') == [175,337,328,374]
                and rule.get('required_source_areas',[])[:2] ==
                    [[91,469,132,511],[526,350,553,376]]):
            # The balance icon moves when its number loses a digit. The fixed
            # guild title and selected tab still identify the currency page.
            # Keep every product/price/close guard, stock/balance OCR, purchase
            # quantity/confirmation condition and fresh close-context check.
            rule = deepcopy(rule)
            rule['required_source_areas'].pop(1)
        popup_layout = (rule.get('id') == 'driftinggirl_task_popup_close'
                        and rule.get('asset') == 'assets/task-popup/notice.png'
                        and rule.get('source_crop') == [618,406,674,453]
                        and rule.get('search_area') == [565,365,710,500]
                        and rule.get('click_source_area') == [627,412,665,449]
                        and rule.get('required_source_areas') == [[315,465,350,492],[271,548,302,597],[106,785,162,810]])
        if (rule.get('id')=='driftinggirl_task_popup_close'
                and rule.get('asset')=='assets/task-popup/notice.png'
                and rule.get('source_crop')==[618,406,674,453]
                and rule.get('required_source_areas')==[[315,465,350,492],[271,548,302,597],[106,785,162,810]]
                and (self.root/'assets/task-popup/kraken-stage3.png').is_file()):
            # A real Kraken stage-3 notice has different fixed title/body
            # strokes. Keep every existing crop, colour and .93 guard.
            rule={**rule,'asset_variants':[*rule.get('asset_variants',[]),
                                         'assets/task-popup/kraken-stage3.png']}
        if (rule.get('id') == 'driftinggirl_task_popup_return'
                and rule.get('asset') == 'assets/task-popup/main.png'
                and rule.get('source_crop') == [279,884,307,910]
                and rule.get('search_area') == [276,881,310,913]
                and rule.get('required_source_areas') ==
                    [[211,1140,266,1189],[542,1140,597,1183]]
                and rule.get('action') == 'observe'
                and (self.root/'assets/task-popup/main-idle.png').is_file()):
            # The idle checkbox has a different fixed check/box appearance.
            # Use the real post-notice frame, retaining .93, both footer
            # companions, brightness, absent notice and consecutive frames.
            rule={**rule,'asset_variants':[*rule.get('asset_variants',[]),
                                         'assets/task-popup/main-idle.png']}
        bonus_supported = any(supported_bonus(r) for r in self.config.get('rules', []))
        if (bonus_supported and rule.get('id') == 'driftinggirl_daily_unavailable'
                and rule.get('asset') == 'assets/daily-reward/premium-claimed.png'
                and rule.get('click_source_area') == [634,247,695,296]):
            # Free already claimed still needs the independent right card.
            rule = {**rule, 'action': 'observe', 'next_stage': 'daily_bonus'}
        if (rule.get('id')=='driftinggirl_daily_bonus'
                and rule.get('asset')=='assets/daily-reward/premium-ready.png'
                and rule.get('source_crop')==[177,336,310,366]
                and rule.get('click_source_area')==[410,430,605,500]
                and rule.get('required_source_areas')==[[177,336,310,366],[266,442,316,483]]):
            # This shipped guard required the free card's gems even after
            # claiming them. Use the existing PREMIUM tab and close cross;
            # leave saved configuration and custom guard definitions intact.
            rule=deepcopy(rule)
            rule['required_source_areas']=[[171,250,263,280],[637,251,688,294]]
        if (rule.get('id') == 'driftinggirl_mailbox_close'
                and rule.get('asset') == 'assets/mailbox-empty-native.png'
                and rule.get('source_crop') == [298,570,408,596]
                and rule.get('search_area') == [240,530,470,650]
                and rule.get('click_source_area') == [612,170,690,230]
                and not rule.get('required_source_areas')
                and 'color_condition' not in rule):
            rule = deepcopy(rule)
            rule['required_source_areas'] = [[144,228,197,253], [488,987,582,1019],
                                             [615,175,685,225]]
            rule['color_condition'] = dict(source_area=[400,550,520,590],
                hsv=[[0,0,180],[179,255,255]], minimum_fraction=.7)
            mailbox_guarded = True
        if (rule.get('id') == 'driftinggirl_mailbox_limit_close'
                and rule.get('asset') == 'assets/mailbox-limit-native.png'
                and rule.get('source_crop') == [216,343,453,369]
                and rule.get('search_area') == [190,325,475,390]
                and rule.get('click_source_area') == [612,170,690,230]
                and 'color_condition' not in rule
                and rule.get('required_source_areas') == [[144,228,197,253],[488,987,582,1019]]):
            # The shipped crop included the changing mail date below the
            # limit notice. Match its fixed full text, and require the close
            # cross too. Keep saved UI options and custom geometry untouched.
            rule = deepcopy(rule)
            rule['source_crop'] = [216,319,390,350]
            rule['search_area'] = [190,310,475,365]
            rule['required_source_areas'].append([615,175,685,225])
            rule['color_condition'] = dict(source_area=[400,550,520,590],
                hsv=[[0,0,180],[179,255,255]], minimum_fraction=.7)
            mailbox_guarded = True
        if 'training' in rule:
            if rule['id'] != 'driftinggirl_parallel_world':
                raise ValueError('练级设置仅用于平行世界')
            training_options(rule['training'])
        width, height = self.config["resolution"]
        kind = rule.get("kind", "template")
        if kind not in ("template", "green_reward_list"):
            raise ValueError("识别规则类型无效")
        search = _lookup("rectangle")(rule["search_area"], width, height)
        file = (self.root / rule["asset"]).resolve()
        if not file.is_relative_to((self.root / "assets").resolve()):
            raise ValueError("目标图片必须位于 assets 目录内")
        reference = np.asarray(Image.open(file).convert("RGB"))
        cropped = "source_crop" in rule
        if cropped:
            area = _lookup("rectangle")(rule["source_crop"], reference.shape[1], reference.shape[0])
            if (rule.get('id') in {'driftinggirl_guild_open','driftinggirl_guild_return'}
                    and rule.get('asset') == 'assets/guild-shop/main.png'
                    and rule.get('source_crop') == [649,371,698,415]
                    and rule.get('search_area') == [610,340,720,450]
                    and rule.get('scales',[1]) == [1]):
                # The diagonal cart handle encloses the animated sky. Match
                # the fixed merchandise/cart core instead; preserve the search,
                # bright-icon guard, sidebar layouts and original click area.
                area = (658,378,693,410)
        else:
            area = _lookup("rectangle")(rule["area"], width, height)
            if reference.shape[:2] != (height, width):
                raise ValueError("目标截图与模拟器分辨率不同")
            if search[2] - search[0] < area[2] - area[0] or search[3] - search[1] < area[3] - area[1]:
                raise ValueError("搜索区域小于目标图片")
        patch = reference[area[1]:area[3], area[0]:area[2]]
        if np.max(patch.std(axis=(0, 1))) < 2:
            raise ValueError("不支持纯色模板")
        if cropped and rule.get("search_slack", True):
            # 裁图模板:搜索框至少比模板大 SLACK,免得零余量规则一有位移就失配。
            scales = [float(value) for value in rule.get("scales", [1])] or [1.0]
            scale = max(scales)
            template = (area[0], area[1],
                        area[0] + max(8, round((area[2] - area[0]) * scale)),
                        area[1] + max(8, round((area[3] - area[1]) * scale)))
            search = search_with_slack(search, template, width, height)
        number(rule["threshold"], 0.5, 1)
        number(rule["interval"], 0.2, 259200 if rule['id']==RESTAURANT_ID else
               86400 if rule['id'] in ('driftinggirl_meter_calibration',GUILD_ID) else 3600)
        number(rule["stable_frames"], 1, 10)
        for field in ("stage", "next_stage"):
            if field in rule and (not isinstance(rule[field], str) or not rule[field].strip()):
                raise ValueError("流程步骤必须是非空字符串")
        number(rule.get("stage_timeout", 30), 1, 3600)
        number(rule.get("min_stage_seconds", 0), 0, 3600)
        number(rule.get("next_delay", 0), 0, 60)
        retry_limit = rule.get('retry_limit')
        if retry_limit is not None:
            if (isinstance(retry_limit, bool) or not isinstance(retry_limit, int)
                    or rule.get('action') != 'observe'
                    or rule.get('stage', 'main') in ('main', '*')
                    or not rule.get('next_stage')):
                raise ValueError('有限步骤重试必须是带后续步骤的观察规则')
            number(retry_limit, 1, 10)
        if rule.get("action", "click") not in ("click", "observe"):
            raise ValueError("动作必须为 click 或 observe")
        interrupt = rule.get('interrupt', False)
        if (not isinstance(interrupt, bool)
                or (rule.get('stage') == '*') != interrupt
                or (interrupt and (not cropped or kind != 'template'
                    or rule.get('action', 'click') != 'click' or not rule.get('next_stage')
                    or rule.get('next_stage') in ('main', '*')
                    or not isinstance(rule.get('required_source_areas'), list)
                    or len(rule['required_source_areas']) < 2))):
            raise ValueError('优先弹窗规则需要全局步骤、组合模板、关闭动作和返回步骤')
        if "stop_after" in rule and not isinstance(rule["stop_after"], bool):
            raise ValueError("stop_after 必须为布尔值")
        # 中途超时的处理策略:默认停机(要人看画面);置 true 则跳过本步骤回到主界面,
        # 本轮功能不算完成、按原间隔重试。用于"领不上就跳过"的日常收取流程。
        if "skip_on_timeout" in rule and not isinstance(rule["skip_on_timeout"], bool):
            raise ValueError("skip_on_timeout 必须为布尔值")
        if "absent_rule" in rule and (not isinstance(rule["absent_rule"], str) or not rule["absent_rule"]):
            raise ValueError("需要等待消失的模板必须指定规则 ID")
        recheck = rule.get('recheck_before_click', False)
        if (not isinstance(recheck, bool) or (recheck and
                (rule.get('stage', 'main') != 'main' or rule.get('action', 'click') != 'click'
                 or not cropped or kind != 'template' or rule.get('absent_rule')
                 or any(rule.get(key) for key in ('purchase_condition', 'quantity_condition', 'confirmation_condition'))))):
            raise ValueError('点击前复核仅支持主界面的模板入口，不能用于购买或后续确认步骤')
        timeout_grace = number(rule.get('timeout_grace', 0), 0, 10)
        if timeout_grace and (rule.get('action') != 'observe' or rule.get('next_stage') != 'main'
                or rule.get('stage', 'main') in ('main', '*') or rule.get('recovery_entry')
                or len(rule.get('required_source_areas', [])) < 2 or not rule.get('color_condition')
                or rule['stable_frames'] < 3):
            raise ValueError('超时确认宽限仅支持组合识别清晰主界面的连续观察，不允许点击或入口重试')
        recovery_entry = rule.get('recovery_entry')
        if recovery_entry is not None:
            entry = next((r for r in self.config['rules'] if r['id'] == recovery_entry), None)
            if (not isinstance(recovery_entry, str) or entry is None
                    or entry.get('stage', 'main') != 'main'
                    or entry.get('next_stage') != rule.get('stage')
                    or rule.get('action') != 'observe' or rule.get('next_stage') != 'main'
                    or interrupt or rule.get('stop_after', False)
                    or not cropped or len(rule.get('required_source_areas', [])) < 2
                    or not rule.get('color_condition') or rule['stable_frames'] < 3
                    or rule.get('min_stage_seconds', 0) < 5):
                raise ValueError('入口恢复必须只观察原入口的下一步，并组合检查清晰界面及连续帧')
        asset_variants = rule.get('asset_variants', [])
        if (not isinstance(asset_variants, list) or len(asset_variants) > 5
                or any(not isinstance(path, str) or not path for path in asset_variants)
                or (asset_variants and (not cropped or kind != 'template'))):
            raise ValueError('备用模板需要裁图规则和最多五个图片路径')
        click_source_area = rule.get("click_source_area")
        if click_source_area is not None:
            if not cropped or not isinstance(click_source_area, list) or len(click_source_area) != 4:
                raise ValueError("相对点击区域仅支持裁图模板，格式为 [左, 上, 右, 下]")
            cx1, cy1, cx2, cy2 = [number(v, -4096, 4096) for v in click_source_area]
            if cx1 >= cx2 or cy1 >= cy2:
                raise ValueError("相对点击区域无效")
        condition = rule.get("number_condition")
        guards = ('number_condition', 'counter_condition', 'purchase_condition', 'quantity_condition', 'confirmation_condition')
        if sum(rule.get(field) is not None for field in guards) > 1:
            raise ValueError('一条规则只能使用一种数字条件')
        if condition is not None:
            if not cropped or not isinstance(condition, dict):
                raise ValueError("数字条件需要裁图模板及条件配置")
            _lookup("rectangle")(condition["source_area"], reference.shape[1], reference.shape[0])
            if condition["operator"] not in ("lt", "gt"):
                raise ValueError("数字条件仅支持小于 lt 或大于 gt")
            number(condition["value"], 0, 9999)
            number(condition.get("confidence", 0.93), 0.5, 1)
        counter = rule.get("counter_condition")
        if counter is not None:
            if not cropped or not isinstance(counter, dict) or condition is not None:
                raise ValueError("每日计数需要裁图模板，且不能与单数字条件混用")
            _lookup("rectangle")(counter["source_area"], reference.shape[1], reference.shape[0])
            if counter.get("mode", "used") not in ("used", "remaining", "ordinal"):
                raise ValueError("每日计数模式必须为 used、remaining 或 ordinal")
            if counter.get("preprocess", "raw") not in ("raw", "white_text", "white_text_consensus"):
                raise ValueError("每日计数预处理必须为 raw、white_text 或 white_text_consensus")
            expected = counter.get("expected_total")
            if isinstance(expected, bool) or not isinstance(expected, int):
                raise ValueError("每日上限必须是整数")
            number(expected, 1, 999)
            number(counter.get("confidence", 0.93), 0.5, 1)
        required_areas = rule.get("required_source_areas", [])
        required_tolerance = number(rule.get('required_tolerance', 2), 1, 40)
        for field, areas in (('purchase_condition', ('counter_area', 'balance_area')),
                             ('quantity_condition', ('quantity_area', 'price_area'))):
            purchase = rule.get(field)
            if purchase is None:
                continue
            if not cropped or not isinstance(purchase, dict):
                raise ValueError('购买条件需要裁图模板和条件配置')
            for key in areas:
                _lookup("rectangle")(purchase[key], reference.shape[1], reference.shape[0])
            number(purchase.get('confidence', .93), .5, 1)
            if not isinstance(purchase.get('label'), str) or not purchase['label']:
                raise ValueError('购买条件需要标签')
            keys = ('unit_price', 'expected_total') if field == 'purchase_condition' else ('unit_price', 'maximum', 'bundle_size')
            for key in keys:
                value = purchase.get(key, 1) if key == 'bundle_size' else purchase[key]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError('购买限额和价格必须为整数')
                number(value, 1, 999)
            modes = ('available', 'unavailable') if field == 'purchase_condition' else ('increment', 'buy')
            if purchase.get('mode') not in modes:
                raise ValueError('购买条件模式无效')
            if field == 'quantity_condition':
                if purchase.get('preprocess', 'tight') not in ('tight', 'purple_text'):
                    raise ValueError('购买数量预处理方式无效')
                if purchase.get('template_prefix') not in (None,'potion','material'):
                    raise ValueError('购买数量模板类型无效')
                origin = purchase.get('stock_rule')
                stock_rule = next((r for r in self.config['rules'] if r['id'] == origin), None)
                stock_condition = stock_rule.get('purchase_condition', {}) if stock_rule else {}
                if (stock_condition.get('mode') != 'available'
                        or stock_condition.get('expected_total') != purchase['maximum']
                        or stock_condition.get('unit_price') != purchase['unit_price']):
                    raise ValueError('购买数量必须关联同一商品的限额检测')
        confirmation = rule.get('confirmation_condition')
        if confirmation is not None:
            if not cropped or not isinstance(confirmation, dict):
                raise ValueError('购买确认需要裁图模板和条件配置')
            _lookup("rectangle")(confirmation['price_area'], reference.shape[1], reference.shape[0])
            number(confirmation.get('confidence', .93), .5, 1)
            if not isinstance(confirmation.get('label'), str) or not confirmation['label']:
                raise ValueError('购买确认需要标签')
            origin = next((r for r in self.config['rules'] if r['id'] == confirmation.get('quantity_rule')), None)
            if (not origin or origin.get('quantity_condition', {}).get('mode') != 'buy'
                    or origin.get('next_stage') != rule.get('stage')):
                raise ValueError('购买确认必须关联前一步已核对的商品数量')
        if not isinstance(required_areas, list) or (required_areas and not cropped):
            raise ValueError("组合图案检测需要裁图模板和区域列表")
        required_templates = []
        for companion in required_areas:
            companion = _lookup("rectangle")(companion, reference.shape[1], reference.shape[0])
            companion_image = reference[companion[1]:companion[3], companion[0]:companion[2]]
            if min(companion_image.shape[:2]) < 8 or np.max(companion_image.std(axis=(0, 1))) < 2:
                raise ValueError("组合图案必须至少 8×8 像素且包含纹理")
            required_templates.append((companion, companion_image))
        color_condition = rule.get('color_condition')
        if color_condition is not None:
            if not cropped or not isinstance(color_condition, dict):
                raise ValueError('颜色条件需要裁图模板及条件配置')
            _lookup("rectangle")(color_condition['source_area'], reference.shape[1], reference.shape[0])
            number(color_condition['minimum_fraction'], 0.01, 1)
            bounds = color_condition.get('hsv')
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError('颜色条件必须指定两个 HSV 边界')
            for bound in bounds:
                if not isinstance(bound, list) or len(bound) != 3:
                    raise ValueError('HSV 边界必须包含三个整数')
                for value, maximum in zip(bound, (179, 255, 255)):
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise ValueError('HSV 边界必须包含三个整数')
                    number(value, 0, maximum)
            if any(low > high for low, high in zip(*bounds)):
                raise ValueError('HSV 下限不能超过上限')
        match_hsv = rule.get("match_hsv")
        if match_hsv is not None:
            if not cropped or not isinstance(match_hsv, list) or len(match_hsv) != 2:
                raise ValueError("颜色轮廓仅支持裁图模板，格式为 [[H下限,S下限,V下限],[H上限,S上限,V上限]]")
            for bound in match_hsv:
                if not isinstance(bound, list) or len(bound) != 3:
                    raise ValueError("HSV 边界必须包含三个整数")
                for value, maximum in zip(bound, (179, 255, 255)):
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise ValueError("HSV 边界必须包含三个整数")
                    number(value, 0, maximum)
            if any(low > high for low, high in zip(*match_hsv)):
                raise ValueError("HSV 下限不能超过上限")
        if cropped:
            button = CroppedTemplateButton(patch, search, rule.get("scales", [1]), rule["id"],
                                           source_area=area, click_source_area=click_source_area,
                                           required_templates=required_templates, match_hsv=match_hsv,
                                           color_condition=color_condition, required_tolerance=required_tolerance)
            button.mailbox_close_guarded = mailbox_guarded
            if (guild_confirm is not None
                    and rule.get('asset')==f'assets/guild-shop/{guild_confirm[0]}-confirm.png'):
                # Both products use blue guild points. The material reference
                # shows a two-digit amount, so its coin is horizontally shifted.
                # Locate the canonical coin afresh instead of cropping that
                # shifted reference at the old three-digit coordinates.
                if guild_confirm[0] == 'potion':
                    currency_reference = reference
                else:
                    with Image.open(self.root/'assets/guild-shop/potion-confirm.png') as image:
                        currency_reference = np.asarray(image.convert('RGB'))
                button.guild_confirmation_currency=currency_reference[573:603,228:257].copy()
                if guild_confirm[0] == 'material':
                    # The independently captured 80-point dialog has a slightly
                    # different coin rendering. Keep its verified inner core
                    # as a second template, with the same .94 match threshold.
                    button.guild_confirmation_currency=(button.guild_confirmation_currency,
                        reference[573:603,237:266].copy())
            for asset in asset_variants:
                alternate_rule = {**rule, 'asset': asset, 'asset_variants': []}
                _, alternate, _ = Clicker.load_rule(self, alternate_rule)
                button.variants.extend(alternate.variants)
            arrival = self.root/'assets/task-popup/kraken-arrival.png'
            if popup_layout and arrival.is_file():
                # An arrival row moves the top up 25 px and reward down 25 px.
                # Keep independent fixed guards, not wider search tolerances.
                with Image.open(arrival) as image:
                    source = np.asarray(image.convert('RGB'))
                guards = ([315,440,350,467], [271,523,302,572],
                          [106,810,162,835], [131,724,190,747])
                alternate = CroppedTemplateButton(source[381:428,618:674],
                    search, rule.get('scales',[1]), rule['id'],
                    source_area=[618,381,674,428],
                    required_templates=[(p,source[p[1]:p[3],p[0]:p[2]]) for p in guards],
                    match_hsv=match_hsv, required_tolerance=required_tolerance)
                # Anchor size and relative close/colour areas remain the same.
                button.variants.extend(alternate.variants)
            if kind == "green_reward_list":
                from reward_list import GreenRewardListButton
                button = GreenRewardListButton(button, rule["reward_list"],
                                               (reference.shape[1], reference.shape[0]), rule["id"])
            if supported_bonus(rule):
                button.daily_bonus_visual = DailyBonusVisual(self.root)
            return rule, adapt_sidebar(rule, adapt_resource_battle(rule, button, reference)), (1, 1)
        if kind == "green_reward_list":
            raise ValueError("绿色奖励列表需要裁图定位模板")
        button = Button(area=area, color=(0, 0, 0), button=area,
                        file=str(file), name=rule["id"])
        button.ensure_template()
        offset = tuple(s - a for s, a in zip(search, area))
        return rule, button, offset
    def start(self, preview=True, *, rule_ids=None, stop_on_main=False, max_clicks=None, timeout=None):
        with self.lock:
            self.idle_only()
            self.daily_schedule.require_bonus = any(supported_bonus(r) and r['enabled'] for r in self.config['rules'])
            if rule_ids is not None:
                if (not isinstance(rule_ids, list) or not rule_ids
                        or any(not isinstance(key, str) for key in rule_ids)
                        or len(set(rule_ids)) != len(rule_ids)):
                    raise ValueError("单项测试需要不重复的目标 ID 列表")
                available = {r["id"] for r in self.config["rules"] if r["enabled"]}
                if not set(rule_ids) <= available:
                    raise ValueError("单项测试目标不存在或未启用")
            if not isinstance(stop_on_main, bool):
                raise ValueError("完成一轮后停止必须为布尔值")
            if max_clicks is not None:
                if isinstance(max_clicks, bool) or not isinstance(max_clicks, int):
                    raise ValueError("点击上限必须是整数")
                number(max_clicks, 1, 1000)
            if timeout is not None:
                timeout = number(timeout, 1, 3600)
            selected_ids = None if rule_ids is None else set(rule_ids)
            recovery_ids = {r['id'] for r in self.config['rules']
                            if r['enabled'] and r['id'] in RECOVERY_RULE_IDS}
            if selected_ids is not None and recovery_ids == set(RECOVERY_RULE_IDS):
                selected_ids.update(recovery_ids)
            rules = [self.load_rule(deepcopy(r)) for r in self.config["rules"]
                     if r["enabled"] and (selected_ids is None or r["id"] in selected_ids)]
            if (not preview and not rules
                    and not (rule_ids is None and self.config.get('return_parallel_enabled') is True)):
                raise ValueError("请先开启至少一项功能或自定义目标")
            if not preview:
                stages = {rule.get("stage", "main") for rule, _, _ in rules} | {'main'}
                identifiers = {rule["id"] for rule, _, _ in rules}
                for rule, _, _ in rules:
                    following = rule.get("next_stage")
                    if following and not rule.get("stop_after", False) and following not in stages:
                        raise ValueError(f"请同时启用「{rule['name']}」的后续步骤")
                    if rule.get("absent_rule") and rule["absent_rule"] not in identifiers:
                        raise ValueError(f"请同时启用「{rule['name']}」所依赖的消失检测模板")
                    if rule.get('recovery_entry') and rule['recovery_entry'] not in identifiers:
                        raise ValueError(f"请同时启用「{rule['name']}」的原入口")
                    stock = rule.get('quantity_condition', {}).get('stock_rule')
                    if stock and stock not in identifiers:
                        raise ValueError(f"请同时启用「{rule['name']}」的商品限额检测")
                    quantity = rule.get('confirmation_condition', {}).get('quantity_rule')
                    if quantity and quantity not in identifiers:
                        raise ValueError(f"请同时启用「{rule['name']}」的购买数量检测")
            self.stop_event.clear()
            self.state.update(fresh_run_state(
                mode="preview" if preview else "clicking",
                active_rule_ids=[r["id"] for r, _, _ in rules],
                revision=self.config_revision, scoped=rule_ids is not None))
            self.state['return_parallel'] = None
            self.thread = threading.Thread(target=self.run,
                                           args=(rules, preview, stop_on_main, max_clicks, timeout,
                                                 rule_ids is None, self.config_revision), daemon=True)
            self.thread.start()
        self.event("识别预览已启动" if preview else "自动点击已启动")
    def stop(self, *, source='direct', peer=None):
        already_requested = self.stop_event.is_set()
        self.stop_event.set()
        if self.runtime is not None:self.runtime.request_stop()
        request = stop_request_context(source, peer)
        with self.lock:
            request.update(was_running=self.running(), already_requested=already_requested,
                           stage=self.state.get('stage'), frame_id=self.state.get('frame_id'),
                           clicks=self.state.get('clicks'))
            self.state['last_stop_request'] = request
            if request['was_running']:
                self.state["mode"] = "stopping"
        self.event('停止请求观测：'+json.dumps(request, ensure_ascii=False, separators=(',', ':')))
        self.event("已请求停止")
    def number_condition(self, rule, button, frame):
        condition = rule.get("number_condition")
        if condition is None:
            return True, None
        box = button.source_to_screen(condition["source_area"])
        height, width = frame.shape[:2]
        reading = None
        if 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height:
            if self.digit_reader is None:
                from digit_ocr import DigitReader
                self.digit_reader = DigitReader(self.root)
            reading = self.digit_reader.read(frame[box[1]:box[3], box[0]:box[2]])
        value = reading.value if reading is not None else None
        confidence = reading.confidence if reading is not None else 0.0
        passed = value is not None and confidence >= condition.get("confidence", 0.93)
        if passed:
            limit = number(condition["value"], 0, 9999)
            passed = value < limit if condition["operator"] == "lt" else value > limit
        with self.lock:
            self.state["numbers"][rule["id"]] = {
                "label": condition.get("label", rule["name"]), "value": value,
                "text": reading.text if reading is not None else "", "confidence": round(confidence, 4),
                "passed": bool(passed), "operator": condition["operator"], "limit": condition["value"],
                "box": list(box), "frame": self.state["frames"],
            }
        return bool(passed), value
    def counter_condition(self, rule, button, frame):
        condition = rule["counter_condition"]
        box = button.source_to_screen(condition["source_area"])
        height, width = frame.shape[:2]
        reading = None
        if 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height:
            if self.digit_reader is None:
                from digit_ocr import DigitReader
                self.digit_reader = DigitReader(self.root)
            from digit_ocr import read_daily_counter
            crop = frame[box[1]:box[3], box[0]:box[2]]
            reading = read_daily_counter(self.digit_reader, crop, condition.get("preprocess", "raw"),
                                         condition.get("confidence", 0.93))
        current = reading.current if reading else None
        total = reading.total if reading else None
        confidence = reading.confidence if reading else 0.0
        mode = condition.get("mode", "used")
        valid = (current is not None and total is not None and 0 <= current <= total
                 and total == condition["expected_total"]
                 and confidence >= condition.get("confidence", 0.93))
        remaining = None
        if valid:
            remaining = (total - current if mode == "used" else current if mode == "remaining"
                         else total - current + 1 if current >= 1 else 0)
        passed = bool(valid and remaining > 0)
        if current is None or total is None:
            skip_reason = getattr(reading, 'reason', None) or "未读清完整每日计数"
        elif total != condition["expected_total"]:
            skip_reason = f"上限读数与预期 {condition['expected_total']} 不符"
        elif confidence < condition.get("confidence", 0.93):
            skip_reason = f"数字置信度 {confidence:.1%}，低于 {condition.get('confidence', 0.93):.0%}"
        elif not passed:
            skip_reason = "今日可领取次数已用完"
        else:
            skip_reason = None
        with self.lock:
            self.state["numbers"][rule["id"]] = {
                "kind": "daily_counter", "label": condition.get("label", rule["name"]),
                "value": current, "limit": total, "expected_total": condition["expected_total"],
                "mode": mode, "remaining": remaining, "text": reading.text if reading else "",
                "confidence": round(confidence, 4), "passed": passed,
                "skip_reason": skip_reason,
                "box": list(box), "frame": self.state["frames"],
            }
        return passed, (current, total)
    def purchase_condition(self, rule, button, frame):
        from purchase_checks import (stock_check, quantity_check, quantity_crop, guild_price_crop,
                                     QuantityTemplates, read_stock_counter, read_guild_balance,
                                     confirmation_price_crop, confirmation_check, read_moving_confirmation_price)
        if self.digit_reader is None:
            from digit_ocr import DigitReader
            self.digit_reader = DigitReader(self.root)
        height, width = frame.shape[:2]
        def crop(area):
            x1, y1, x2, y2 = button.source_to_screen(area)
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError('购买读数区域超出画面')
            return frame[y1:y2, x1:x2]
        condition = rule.get('purchase_condition')
        if condition is not None:
            counter = read_stock_counter(self.digit_reader, crop(condition['counter_area']), condition.get('confidence', .93))
            balance_image = crop(condition['balance_area'])
            if (rule.get('id') in {'driftinggirl_guild_potion_select','driftinggirl_guild_potion_skip',
                                  'driftinggirl_guild_material_select','driftinggirl_guild_material_skip'}
                    and condition.get('balance_area') == [560,350,641,377]):
                balance = read_guild_balance(self.digit_reader, balance_image, condition.get('confidence', .93))
            else:
                balance = self.digit_reader.read(balance_image)
            passed, key, result = stock_check(counter, balance, condition)
        elif rule.get('confirmation_condition') is not None:
            condition = rule['confirmation_condition']
            coin=getattr(button,'guild_confirmation_currency',None)
            price=(read_moving_confirmation_price(self.digit_reader,crop([180,550,540,625]),coin)
                   if coin is not None else
                   self.digit_reader.read(confirmation_price_crop(crop(condition['price_area']))))
            with self.lock:
                selection = deepcopy(self.state['numbers'].get(condition['quantity_rule']))
            passed, key, result = confirmation_check(price, selection, condition)
        else:
            condition = rule['quantity_condition']
            quantity_image = crop(condition['quantity_area'])
            quantity = self.digit_reader.read(quantity_crop(quantity_image, condition.get('preprocess', 'tight')))
            prefix = condition.get('template_prefix')
            if prefix and (quantity.value is None or quantity.confidence < condition.get('confidence',.93)):
                if not hasattr(self,'quantity_templates'):
                    self.quantity_templates = {}
                if prefix not in self.quantity_templates:
                    self.quantity_templates[prefix] = QuantityTemplates(self.root,prefix)
                quantity = self.quantity_templates[prefix].read(quantity_image)
            price = self.digit_reader.read(guild_price_crop(crop(condition['price_area'])))
            with self.lock:
                stock = deepcopy(self.state['numbers'].get(condition['stock_rule']))
            passed, key, result = quantity_check(quantity, price, stock, condition)
        with self.lock:
            self.state['numbers'][rule['id']] = {**result, 'frame': self.state['frames']}
        return passed, key
    def daily_cleanup_context(self, frame, matchers, allow_ready=False):
        """Identify only the existing, fully guarded claimed PREMIUM close."""
        entry = matchers.get('driftinggirl_daily_close')
        free = matchers.get('driftinggirl_daily_claim')
        if entry is None or free is None or frame.shape[:2] != (1280,720):
            return None
        rule, button, offset = entry
        other, guard, guard_offset = free
        if (not rule['enabled'] or not other['enabled']
                or not button.match(frame,offset=offset,similarity=rule['threshold'])
                or guard.match(frame,offset=guard_offset,similarity=other['threshold'])):
            return None
        bonus = matchers.get(BONUS_ID)
        visual = getattr(bonus[1], 'daily_bonus_visual', None) if bonus else None
        if visual is not None:
            state = visual.state(frame)
            if state not in (('ready','gray') if allow_ready else ('gray',)):
                return None
        title=frame[336:366,177:310]
        x1,y1,x2,y2=button.button
        cross=frame[y1:y2,x1:x2].astype(np.int16)
        if (float(((title.max(axis=2)>=180)&(title.min(axis=2)>=100)).mean())<.5
                or int(((cross[:,:,0]>=200)&(cross[:,:,0]>cross[:,:,1]*1.4)
                        &(cross[:,:,0]>cross[:,:,2]*1.4)).sum())<500):
            return None
        # Fixed header digits only; native sailing's changing location and
        # countdown are intentionally excluded from this menu-close context.
        rank_pixels=frame[10:62,72:126].min(axis=2)>130
        if int(rank_pixels.sum())<50:
            return None
        rank = rank_pixels.tobytes()
        stage = frame[46:66,35:56].astype(np.int16)
        world = ((stage[:,:,1]>=100)&(stage[:,:,1]>stage[:,:,0]+15)
                 &(stage[:,:,2]>stage[:,:,0]+15)).tobytes()
        return rank,world,tuple(button.button)

    @staticmethod
    def daily_return_visible(frame,matchers):
        entry=matchers.get('driftinggirl_daily_return')
        premium=matchers.get('driftinggirl_daily_premium')
        if entry is None or premium is None:
            return False
        rule,button,offset=entry
        other,guard,guard_offset=premium
        return (button.match(frame,offset=offset,similarity=rule['threshold'])
                and not guard.match(frame,offset=guard_offset,similarity=other['threshold']))

    def daily_completion_visible(self, feature, rule, frame, matchers):
        if feature == 'daily_reward':
            # These rules require the claimed checkmark and absence of the free card.
            bonus = matchers.get(BONUS_ID)
            visual = getattr(bonus[1], 'daily_bonus_visual', None) if bonus else None
            return (rule['id'] == 'driftinggirl_daily_close' and visual.state(frame) == 'gray'
                    if visual is not None else
                    rule['id'] in ('driftinggirl_daily_close', 'driftinggirl_daily_unavailable'))
        if feature == 'mailbox':
            # 邮箱是"每天一次":只要进过邮箱并点过全部领取(或在超上限分支关掉过邮箱),
            # 本周期就算完成,之后即使再出现新邮件提示也等到下一个 08:00。
            return rule['id'] in ('driftinggirl_mailbox_claim_all',
                                  'driftinggirl_mailbox_close',
                                  'driftinggirl_mailbox_limit_close',
                                  'driftinggirl_mailbox_return')
        if feature != 'guild_shop' or rule['id'] != 'driftinggirl_guild_close':
            return False
        # Check the actual post-purchase stock, not just clicks or affordability.
        for item in ('potion', 'material'):
            entry = matchers.get(f'driftinggirl_guild_{item}_skip')
            if entry is None:
                return False
            stock_rule, button, offset = entry
            if not button.match(frame, offset=offset, similarity=stock_rule['threshold']):
                return False
            passed, _ = self.purchase_condition(stock_rule, button, frame)
            reading = self.state['numbers'].get(stock_rule['id'], {})
            if not passed or not reading.get('valid') or reading.get('remaining') != 0:
                return False
        return True
    def run(self, rules, preview, stop_on_main=False, max_clicks=None, timeout=None,
            follow_settings=False, revision=0):
        module = ModuleBase(SimpleNamespace(BUTTON_OFFSET=(0, 0)), self.device)
        runtime = _lookup("FrameRuntime")(self.device, self.stop_event, lambda: self.digit_reader)
        self.runtime = runtime
        st = LoopState()
        st.recovery_only = (not follow_settings and bool(rules)
                            and all(rule['id'] in RECOVERY_RULE_IDS for rule, _, _ in rules))
        # 页面图导航(ALAS page link + 反向 BFS):只在规则显式写了 ui_nav 时启用,
        # 用来把"没等到预期画面"从停机改成"朝主画面退一步"。
        navigator = None

        def navigation():
            nonlocal navigator
            if navigator is None:
                # 页面图只收已启用的规则:关掉的功能不该被导航点开。
                enabled = [rule for rule in self.config['rules'] if rule.get('enabled')]
                pages = build(enabled, allowed_actions=set().union(*OPTIONAL_CLOSES.values()))
                # st.matchers 平时只装当前阶段的规则;导航要跨页识别,先把图里
                # 用到的按钮全部装好,否则检测不到页、也点不到边。
                by_id = {rule['id']: rule for rule in enabled}
                for page in pages.values():
                    for rule_id in {*page.check, *page.links.values()}:
                        if rule_id and rule_id not in st.matchers and rule_id in by_id:
                            st.matchers[rule_id] = self.load_rule(deepcopy(by_id[rule_id]))
                navigator = Navigator(pages, lambda rid: nav_appear(rid))
            return navigator

        def nav_appear(rule_id):
            entry = st.matchers.get(rule_id)
            if entry is None:
                return False
            rule, button, offset = entry
            return module.appear(button, offset=offset, similarity=rule['threshold'])
        guild_selection = GuildSelectionGuard()
        guild_return = GuildReturnGuard()
        guild_return_probe = None
        popup_raid = PopupRaidReturn(self.root)
        popup_cruise = PopupCruiseReturn()
        mailbox_cleanup_used = False
        mailbox_cleanup_frames = 0
        mailbox_cleanup_since = None
        mailbox_cleanup_key = None
        mailbox_cleanup_deadline = None
        mailbox_close_guard = MailboxCloseGuard()
        mailbox_pending_evidence = None
        guild_cleanup_deadline = None
        guild_cleanup_origin = None
        guild_cleanup_rule = None
        optional_cleanup = None
        event_potion_skips = 0
        cleanup_evidence = None
        guild_orphan_used = False
        guild_orphan_key, guild_orphan_since, guild_orphan_frames = None, None, 0
        guild_pending_evidence = None
        popup_pending_evidence = None
        def readonly_guild_world():
            nonlocal guild_return_probe
            if stone is not None:
                return stone
            if guild_return_probe is None:
                if self.digit_reader is None:
                    from digit_ocr import DigitReader
                    self.digit_reader = DigitReader(self.root)
                def no_input(*_):
                    raise RuntimeError('公会返回只读观察器禁止游戏输入')
                guild_return_probe = StoneFarm(self.root,self.digit_reader,no_input,self.event)
            return guild_return_probe
        st.stable = {rule['id']: 0 for rule, _, _ in rules}
        st.stage_started_at = _time().monotonic()
        st.daily_period = self.daily_schedule.period()
        st.scheduled = not preview and follow_settings
        def finish_scheduled_daily(success=False):
            if st.scheduled_daily is not None:
                feature, interval, started_period = st.scheduled_daily
                daily = self.daily_schedule.status()[feature]
                if daily['period'] != started_period:
                    daily = {**daily, 'period': started_period, 'completed': False}
                self.task_schedule.finish(feature, interval, daily, success=success)
                st.scheduled_daily = None
        def begin_daily_cleanup(stage='daily_after_claim'):
            # Cleanup is a separate, bounded return, never a renewed award
            # attempt. Its close and return remain the configured rules.
            now = _time().monotonic()
            finish_scheduled_daily(False)
            st.daily_attempt,st.daily_verified = None,False
            st.daily_cleanup_deadline = now+40
            st.stage,st.stage_trigger = stage,'每日奖励：安全关闭未完成的窗口'
            st.stage_started_at,st.stage_ready_at,st.stage_deadline = now,now,now+20
            st.skip_on_timeout = False
            st.stable = {key:0 for key in st.stable}
            st.number_values.clear();st.target_keys.clear()
            with self.lock:
                self.state['stage'] = st.stage
        def step_scheduled_controller(feature, label, controller, image):
            invoke = lambda fn, picture: runtime.call(label, fn, picture)
            if feature == 'parallel_skills' and getattr(controller,'force_check',False):
                owned=invoke(controller.step,image)
                if controller.cycles:
                    controller.force_check=False
                    self.task_schedule.finish(feature,controller.interval,success=True)
                return owned
            if st.scheduled:
                return self.task_schedule.step_controller(feature, controller, image, invoke)
            return invoke(controller.step, image)
        st.number_values = {}
        st.target_keys = {}
        st.matchers = {rule["id"]: (rule, button, offset) for rule, button, offset in rules}
        scoped_rule_ids = set(st.matchers)
        potion_cleanup = None
        def recover_potion(frame):
            nonlocal potion_cleanup
            if preview or (st.stage != 'potion_ad_confirm'
                           and not (potion_cleanup is not None and potion_cleanup.active)):
                return False
            if potion_cleanup is None:
                potion_cleanup = PotionAdRecovery(self.root)
            cleanup = potion_cleanup
            now = _time().monotonic()
            if not cleanup.active:
                if not cleanup.visible(frame):
                    return False
                cleanup.begin(st.stage_deadline)
                entry = st.matchers[cleanup.ENTRY][0]
                self.task_schedule.finish('potion_ad', entry['interval'], success=False)
                st.daily_attempt, st.daily_verified = None, False
                st.entry_recoveries.setdefault(cleanup.ENTRY, deque()).append(now)
                st.stage = 'potion_ad_cancel_return'
                st.skip_on_timeout = False
                with self.lock:
                    self.state['stage'] = st.stage
                    self.state['potion_ad'] = cleanup.status()
                self.event(f'药水广告：误入返回现实世界确认框，取消本次广告，不记成功（{cleanup.attempts}/{cleanup.LIMIT}）')
            if cleanup.deadline is None or now >= cleanup.deadline:
                raise RuntimeError('药水广告取消后未在原期限内确认主画面，保留未完成状态并停止')
            if not cleanup.cancelled and cleanup.visible(frame):
                expected = tuple(cleanup.button.button)
                def cancel_wrong_world():
                    with getattr(self.device, 'io_lock', nullcontext()):
                        fresh = runtime.capture()
                        self.remember(fresh)
                        with self.lock:self.state['frames'] += 1
                        if (self.stop_event.is_set() or _time().monotonic() >= cleanup.deadline
                                or st.run_deadline is not None and _time().monotonic() >= st.run_deadline
                                or max_clicks is not None and self.state['clicks'] >= max_clicks
                                or not cleanup.visible(fresh) or tuple(cleanup.button.button) != expected):
                            return False
                        # Reserve before dispatch. An uncertain input must never
                        # buy a second cancellation of the same modal.
                        cleanup.cancelled = True
                        with self.lock:self.state['potion_ad'] = cleanup.status()
                        self.device.click(cleanup.button)
                        with self.lock:self.state['clicks'] += 1
                        return True
                runtime.call('potion_ad_cancel', cancel_wrong_world)
            elif cleanup.observe_return(frame, st.matchers, now):
                cleanup.active, cleanup.deadline = False, None
                st.reset_after_return()
                st.stage_started_at = now
                st.stable = {key: 0 for key in st.stable}
                st.number_values.clear(); st.target_keys.clear()
                st.previous, st.previous_progress, st.consecutive = None, None, 0
                with self.lock:
                    self.state['stage'] = 'main'
                    self.state['potion_ad'] = cleanup.status()
                    self.state['recoveries'] += 1
                self.event('药水广告：取消后连续三帧三秒确认主画面，本次不记成功，其他任务继续')
                if cleanup.attempts >= cleanup.LIMIT:
                    self.event('药水广告：误入回城已达本轮三次上限，本轮不再尝试药水入口')
                if stop_on_main:
                    with self.lock:self.state['stop_reason'] = 'observation_complete'
                    self.stop_event.set()
            return True
        parallel = None
        equipment = None
        backpack = None
        skills = None
        calibrator = None
        stone = None
        cruise = None
        raid = None
        setup = None
        pass_task = None
        guild_daily = None
        restaurant = None
        return_parallel = None
        def parallel_click(x, y, label):
            x, y = int(x), int(y)
            self.device.click(SimpleNamespace(button=(max(0, x-5), max(0, y-5),
                                                      min(720, x+5), min(1280, y+5))))
            with self.lock:
                self.state['clicks'] += 1
                self.state['matches'] += 1
                self.state['last_match'] = {'name': label, 'score': 1.,
                                            'box': [x-5, y-5, x+5, y+5]}
            self.event(f'点击：{label}，坐标 ({x}, {y})')
        def parallel_swipe(x1, y1, x2, y2, label):
            with self.device.io_lock:
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                self.device.command('shell', 'input', 'swipe', str(x1), str(y1),
                                    str(x2), str(y2), '480')
            with self.lock:
                self.state['clicks'] += 1
            self.event(label)
        def make_parallel():
            entry = next((rule for rule, _, _ in rules
                          if rule['id'] == 'driftinggirl_parallel_world'), None)
            drain = False
            if any(r['id'] == 'driftinggirl_stone_farm' for r, _, _ in rules):
                # The native-route portal task owns handoffs. Never revive an
                # old disabled training route or run two portal controllers.
                return None
            native_meter = any(r['id'] == 'driftinggirl_meter_calibration' for r, _, _ in rules)
            if entry is None and native_meter:
                # Native AUTO now belongs to the user. A stale disabled training
                # journal must not stop/reconfigure it for this independent task.
                return None
            if not preview and entry is None:
                # If the task was disabled while paused, finish its saved game
                # AUTO route before letting other enabled tasks take the UI.
                try:
                    saved = json.loads((self.root/'training_state.json').read_text(encoding='utf-8-sig'))
                    if saved.get('sailing_enabled') is True:
                        entry = next((r for r in self.config['rules']
                                      if r['id'] == 'driftinggirl_parallel_world'),None)
                        drain = entry is not None
                except (OSError,ValueError,TypeError):
                    pass
            if preview or entry is None:
                return None
            if self.digit_reader is None:
                from digit_ocr import DigitReader
                self.digit_reader = DigitReader(self.root)
            controller = _lookup("ParallelWorld")(self.root, parallel_click, parallel_swipe,
                                 self.event, self.digit_reader, entry['interval'], entry.get('training'),
                                 fishing_action=fishing_action,sailing_action=sailing_click,
                                 defense_action=defense_click,completion_action=completion_click,
                                 portal_action=portal_click)
            if drain:
                controller.training.options['battle_tap'] = False
            controller.sailing_shutdown = drain
            controller.only_sailing_shutdown = drain
            return controller
        def portal_click(action, owner=None):
            owner = parallel if owner is None else owner
            with self.device.io_lock:
                if (self.stop_event.is_set()
                        or max_clicks is not None and self.state['clicks'] >= max_clicks
                        or st.run_deadline is not None and _time().monotonic() >= st.run_deadline):
                    raise InterruptedError('已停止')
                returning = action['kind'].startswith('return-')
                abandoning_boss = owner is stone and action['kind'] in ('lower-boss-abandon','lower-boss-abandon-menu')
                fast_contact = returning or abandoning_boss
                cancelling = owner is stone and action['kind'] == 'handoff-cancel'
                def stopped():
                    return (self.stop_event.is_set()
                        or max_clicks is not None and self.state['clicks'] >= max_clicks
                        or st.run_deadline is not None and _time().monotonic() >= st.run_deadline)
                started = _time().monotonic()
                capture = runtime.capture
                fresh = capture()
                if cancelling:
                    scan_deadline = started+6
                    reserved = False
                    for index in range(40):
                        if stopped() or _time().monotonic() >= scan_deadline or not owner.same_cancel_context(action,fresh):
                            return False
                        if owner.validate_portal(action,fresh) and _time().monotonic()-started <= .12:
                            if reserved:
                                break
                            # Persist the conservative attempt before contact,
                            # then capture again. Disk writes may outlive the
                            # 120 ms input frame; never spend that stale frame
                            # or reserve a second attempt for the same scan.
                            if not owner.reserve_input(action):
                                return False
                            reserved = True
                        if index == 39:
                            return False
                        started = _time().monotonic()
                        fresh = capture()
                frame_id = self.remember(fresh)
                # The cancelling loop just checked this exact fresh frame.
                # Repeating its image work eats the fleeting cast window.
                if not cancelling and not owner.validate_portal(action,fresh):
                    return False
                if (self.stop_event.is_set()
                        or max_clicks is not None and self.state['clicks'] >= max_clicks
                        or st.run_deadline is not None and _time().monotonic() >= st.run_deadline):
                    raise InterruptedError('已停止')
                if (fast_contact or cancelling) and _time().monotonic()-started > .12:
                    return False
                if cancelling and _time().monotonic() >= scan_deadline:
                    return False
                if not cancelling and hasattr(owner, 'reserve_input') and not owner.reserve_input(action):
                    return False
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                if (fast_contact or cancelling) and _time().monotonic()-started > .12:
                    return False
                if cancelling and _time().monotonic() >= scan_deadline:
                    return False
                metadata = dict(input_id=uuid.uuid4().hex,frame_id=frame_id,
                    kind=action['kind'],x=action['x'],y=action['y'],
                    label=action['label'],rank=owner.rank,
                    completed_rank_40=owner.completed_rank_40,
                    controller='stone_farm' if owner is stone else 'parallel_world',
                    input_begin_time=_time().time(),capture_age_ms=round((_time().monotonic()-started)*1000,2))
                if cancelling:
                    metadata['continuous_cast_stop'] = action.get('continuous_stop') is True
                if fast_contact or cancelling:
                    if action['kind']=='return-confirm' or owner is stone and action['kind']=='lower-boss-abandon-menu':
                        # Unity's modal button needs a complete stationary
                        # contact. Keep the battle SDK's 40 ms release intact;
                        # this is the existing 80 ms ADB menu gesture instead.
                        if stopped():
                            raise InterruptedError('已停止')
                        self.device.command('shell','input','swipe',str(action['x']),str(action['y']),
                                            str(action['x']),str(action['y']),'80')
                        contact = dict(input_transport='adb-stationary',hold_ms=80)
                    else:
                        contact = self.device.touch_fast(action['x'],action['y'],started,stopped)
                    if contact is None:
                        return False
                    metadata.update(contact)
                    with self.lock:
                        self.state['clicks'] += 1
                        self.state['matches'] += 1
                        self.state['last_match'] = {'name':action['label'],'score':1.,
                            'box':[action['x']-5,action['y']-5,action['x']+5,action['y']+5]}
                    self.event(f"点击：{action['label']}，坐标 ({action['x']}, {action['y']})")
                elif owner is stone and action['kind'] == 'first-pan':
                    X,Y,milliseconds=action['swipe']
                    self.device.command('shell','input','swipe',str(action['x']),str(action['y']),
                                        str(X),str(Y),str(milliseconds))
                    metadata.update(input_transport='adb-swipe',swipe=action['swipe'])
                    with self.lock:
                        self.state['clicks'] += 1
                        self.state['matches'] += 1
                    self.event('滑动：'+action['label'])
                else:
                    parallel_click(action['x'],action['y'],action['label'])
                    metadata['input_transport'] = 'adb'
                if hasattr(owner, 'input_issued'):
                    owner.input_issued(action, metadata)
                after = None
                if not self.stop_event.is_set():
                    after = capture()
                    metadata['after_frame_id'] = self.remember(after)
                    metadata['after_time'] = _time().time()
                if self.portal_evidence is None:
                    self.portal_evidence = _lookup("BossInputEvidence")(self.root,self.event,
                        folder='portal-input',label='传送门')
                self.portal_evidence.record(metadata,fresh.copy(),after.copy() if after is not None else None)
            return True
        def completion_click():
            with self.device.io_lock:
                if (self.stop_event.is_set()
                        or max_clicks is not None and self.state['clicks'] >= max_clicks
                        or st.run_deadline is not None and _time().monotonic() >= st.run_deadline):
                    raise InterruptedError('已停止')
                fresh = runtime.capture()
                self.remember(fresh)
                if not parallel.completion_visible(fresh):
                    return False
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                parallel_click(360,840,'平行世界：确认本阶段探索结束')
                if equipment is not None:
                    equipment.yielded_for_completion()
            return True
        def defense_click(expected, owner=None, rank=None):
            owner = parallel if owner is None else owner
            attack_rank = owner.rank if rank is None else rank
            with self.device.io_lock:
                captured = {}
                pending = []
                if self.boss_evidence is None:
                    self.boss_evidence = _lookup("BossInputEvidence")(self.root,self.event)
                def capture_defense_frame():
                    started = _time().monotonic()
                    fresh = runtime.capture()
                    identifier = self.remember(fresh)
                    if pending:
                        metadata,before = pending.pop()
                        metadata.update(after_frame_id=identifier,after_time=_time().time())
                        self.boss_evidence.record(metadata,before,fresh.copy())
                    captured.update(frame=fresh,frame_id=identifier,started=started)
                    return fresh
                def defense_stopped():
                    return (self.stop_event.is_set()
                            or owner is stone and not owner.enabled()
                            or owner is stone and owner.active and not owner.input_allowed()
                            or max_clicks is not None and self.state['clicks'] >= max_clicks
                            or st.run_deadline is not None and _time().monotonic() >= st.run_deadline)
                def hit_defense_target(target):
                    if defense_stopped():
                        raise InterruptedError('已停止')
                    # This applies to both native and fresh ADB transports.
                    # A slow capture must never authorize a late contact.
                    if _time().monotonic()-captured['started'] > .12:
                        return False
                    x,y = target['x'],target['y']
                    metadata = dict(input_id=uuid.uuid4().hex,frame_id=captured['frame_id'],
                        rank=attack_rank,controller_rank=attack_rank,
                        controller='stone_farm' if owner is stone else 'parallel_world',
                        kind=target.get('kind','defense'),
                        target=dict(target),input_begin_time=_time().time(),
                        transport=getattr(self.device,'screenshot_transport','test'),
                        capture_to_command_ms=round((_time().monotonic()-captured['started'])*1000,1),
                        gesture='stationary_touch',hold_ms=80,result='issued_only')
                    # A DOWN/UP within one Unity frame can be lost by gameplay
                    # input polling. Hold for several frames, with no movement
                    # or unguarded repeat at the old coordinates.
                    if hasattr(self.device,'touch_fast'):
                        contact=self.device.touch_fast(x,y,captured['started'],defense_stopped)
                        if contact is None:
                            return False
                        metadata.update(contact)
                    else:
                        self.device.command('shell','input','swipe',str(x),str(y),str(x),str(y),'80')
                        metadata.update(input_transport='adb')
                    metadata['input_end_time'] = _time().time()
                    pending.append((metadata,captured['frame'].copy()))
                    with self.lock:
                        self.state['clicks'] += 1
                        self.state['matches'] += 1
                        self.state['last_match'] = {'name':('战斗：点击屏幕补充伤害' if target.get('kind') == 'damage'
                                                             else '海神：连续点击发黄光的触手武器'),
                            'score':1.,'box':[x-5,y-5,x+5,y+5],
                            'frame_id':captured['frame_id'],'input_id':metadata['input_id'],
                            'hold_ms':metadata['hold_ms'],'transport':metadata['transport'],
                            'input_transport':metadata['input_transport'],'result':'issued_only'}
                try:
                    # First-cast result votes need fresh outer frames before
                    # their original deadline; a four-second burst must not
                    # consume the remaining verification window. Other fights
                    # retain the existing burst and per-input guards.
                    limits = {'seconds':.5} if owner is stone and owner.phase=='first_verify' else {}
                    taps = owner.defense.burst(capture_defense_frame,hit_defense_target,
                                              attack_rank,defense_stopped,**limits)
                finally:
                    if pending:
                        metadata,before = pending.pop()
                        metadata.update(after_frame_id=None,after_reason='burst_ended')
                        self.boss_evidence.record(metadata,before)
            counts = owner.defense.last_burst
            if counts.get('defense_taps'):
                self.event(f"海神反击：连续核对新截图后点击充能武器 {counts['defense_taps']} 次")
            if counts.get('damage_taps'):
                self.event(f"战斗补伤害：新截图核对后点击屏幕 {counts['damage_taps']} 次；充能触手优先")
            return taps
        def close_defense_equipment(owner=None, rank=None):
            owner = parallel if owner is None else owner
            attack_rank = owner.rank if rank is None else rank
            if equipment is None or _time().monotonic() < equipment.next_action_at:
                return False
            with self.device.io_lock:
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                fresh = runtime.capture()
                self.remember(fresh)
                if (not owner.defense.owns(fresh,attack_rank)
                        or owner is stone and not owner.enabled()
                        or not equipment.validate_battle_close(fresh)):
                    return False
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                parallel_click(360,1166,'海神战斗：关闭遮挡战斗按钮的鱼饼强化面板')
                equipment.yielded_for_battle()
            return True
        def close_boss_sailing_equipment():
            if equipment is None or _time().monotonic() < equipment.next_action_at:
                return False
            with self.device.io_lock:
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                fresh = runtime.capture()
                self.remember(fresh)
                if (not parallel.boss_sailing_close_guard(fresh)
                        or not equipment.validate_battle_close(fresh)):
                    return False
                parallel_click(360,1166,'BOSS 航行：到达切换点，先收起鱼饼强化面板')
                equipment.yielded_for_sailing()
            return True
        def sailing_click(action, x, y, label):
            with self.device.io_lock:
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                fresh = runtime.capture()
                self.remember(fresh)
                controller = parallel.sailing
                if not controller.validate(action,fresh):
                    self.event('自动航行：点击前画面或目标变化，等待重新识别')
                    return False
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                controller.before_click(action)
                metadata = (dict(input_id=uuid.uuid4().hex,frame_id=self.state.get('frame_id'),
                    action=action,profile=controller.profile,rank=parallel.rank,
                    x=x,y=y,time=_time().time(),result='issued_only')
                    if controller.profile == 'boss' else None)
                parallel_click(x,y,label)
                if metadata is not None:
                    after = None
                    if not self.stop_event.is_set():
                        after = runtime.capture()
                        metadata['after_frame_id'] = self.remember(after)
                    if self.sailing_evidence is None:
                        self.sailing_evidence = _lookup("BossInputEvidence")(self.root,self.event,
                            folder='sailing-input',label='自动航行')
                    self.sailing_evidence.record(metadata,fresh.copy(),after.copy() if after is not None else None)
            return True
        def fishing_action(action, data, owner=None, guard=None, reserve=None, allowed=None,
                           route_expected=None):
            owner = parallel if owner is None else owner
            observed = []
            def fishing_stopped():
                return (self.stop_event.is_set()
                         or guard is not None and not (allowed or calibrator.input_allowed)()
                        or max_clicks is not None and self.state['clicks'] >= max_clicks
                        or st.run_deadline is not None and _time().monotonic() >= st.run_deadline)
            def raise_fishing_stop():
                if (guard is not None and allowed is None and not self.stop_event.is_set()
                        and not (max_clicks is not None and self.state['clicks'] >= max_clicks)
                        and not (st.run_deadline is not None and _time().monotonic() >= st.run_deadline)
                        and not calibrator.input_allowed()):
                    raise MeterWorkflowLimit('标尺输入前校准窗口已到期或功能已关闭')
                raise InterruptedError('已停止')
            with self.device.io_lock:
                if fishing_stopped():
                    raise_fishing_stop()
                meter = owner.fishing
                # Only the independent calibration opts into preserving native
                # sailing. Navigation/training/stone handoffs still require OFF.
                keep_route = bool(guard is not None and callable(route_expected)
                                  and route_expected() is True)
                cancelling = action in ('cancel','scan_cancel')
                captured = {}
                def capture_cancel_frame():
                    started = _time().monotonic()
                    image = runtime.capture()
                    identifier = self.remember(image)
                    captured.update(frame=image,frame_id=identifier,started=started)
                    return image
                if cancelling:
                    if self.fishing_evidence is None:
                        self.fishing_evidence = _lookup("BossInputEvidence")(self.root,self.event,
                            folder='fishing-input',label='钓鱼计量')
                    def usable_cancel_frame(image):
                        return _time().monotonic()-captured['started'] <= .12
                    def stopped_route(image):
                        if (not owner._parallel(image) or not keep_route
                                and owner.training.options['auto_sailing']
                                and owner.training.state.get('sailing_enabled')
                                or guard is not None and not guard(image)):
                            return None
                        return owner.sailing.running(image)
                    scan_options = {'expected_route': True} if keep_route else {}
                    fresh = meter.scan_cancel_frame(capture_cancel_frame,stopped_route,
                                                    data,fishing_stopped,usable_cancel_frame,**scan_options)
                    if fresh is None:
                        return False
                else:
                    fresh = runtime.capture()
                    self.remember(fresh)
                    route = owner.sailing.running(fresh) if owner.sailing else False
                    if ((route is not True if keep_route else route is True or owner.sailing
                            and owner.training.options['auto_sailing']
                            and (route is not False or owner.training.state.get('sailing_enabled')))
                            or guard is not None and not guard(fresh)):
                        self.event('钓鱼计量：航巡状态或校准上下文变化，等待重新确认')
                        return False
                    if not meter.validate(action, fresh, data):
                        self.event('钓鱼计量：操作前画面变化，等待重新识别')
                        return False
                    actual_start = getattr(meter,'validated_start',None) if action == 'start' else None
                    if actual_start is not None and actual_start[0] is fresh:
                        # Input receipt and log must describe this fresh frame,
                        # rather than the slightly different planned selection.
                        data = actual_start[1]
                if fishing_stopped():
                    raise_fishing_stop()
                if reserve is not None and not reserve(action):
                    return False
                if fishing_stopped():
                    raise_fishing_stop()
                calibration_metadata = None
                if guard is not None and not cancelling:
                    calibration_metadata = dict(input_id=uuid.uuid4().hex,
                        frame_id=self.state.get('frame_id'),kind='meter_'+action,
                        rank=owner.training.state.get('rank'),input_begin_time=_time().time(),
                        target=dict(value=data.get('value'),goal=data.get('goal')),
                        result='issued_only')
                    calibration_before = fresh.copy()
                if cancelling:
                    # The X is the left-hand label; target the fish-shaped
                    # cancellation button itself, rather than that label.
                    x,y = 690,1090
                    if _time().monotonic()-captured['started'] > .12:
                        return False
                    before = fresh.copy()
                    metadata = dict(input_id=uuid.uuid4().hex,frame_id=captured['frame_id'],
                        kind='meter_cancel',rank=owner.training.state.get('rank'),
                        target=dict(x=x,y=y),input_begin_time=_time().time(),
                        transport=getattr(self.device,'screenshot_transport','test'),
                        capture_to_command_ms=round((_time().monotonic()-captured['started'])*1000,1),
                        result='issued_only')
                    if hasattr(self.device,'touch_fast'):
                        contact = self.device.touch_fast(x,y,captured['started'],fishing_stopped)
                        if contact is None:
                            return False
                        metadata.update(contact)
                    else:
                        self.device.command('shell','input','tap',str(x),str(y))
                        metadata.update(input_transport='adb',hold_ms=0)
                    metadata['input_end_time'] = _time().time()
                    label = '钓鱼计量：尝试取消已确认的普通鱼准备状态，等待休眠证据'
                    if not self.stop_event.is_set():
                        observed.append(runtime.capture())
                        metadata.update(after_frame_id=self.remember(observed[-1]),after_time=_time().time())
                    self.fishing_evidence.record(metadata,before,observed[-1].copy() if observed else None)
                elif action == 'resume_cast':
                    x, y = 360, 1010
                    if calibration_metadata is not None:
                        calibration_metadata['gesture'] = dict(x=x,y=y,duration_ms=80)
                    self.device.command('shell', 'input', 'swipe', str(x), str(y),
                                        str(x), str(y), '80')
                    label = '标尺校准：仅短点灰色鱼钩恢复钓鱼，等待真实填充或战斗，不计校准成功'
                elif action in ('drag','reveal'):
                    x, y, to_y = data['x'], data['y'], data['to_y']
                    if calibration_metadata is not None:
                        calibration_metadata['gesture'] = dict(x=x,y=y,to_y=to_y,duration_ms=meter.DRAG_MS)
                    self.device.command('shell', 'input', 'swipe', str(x), str(y),
                                        str(x), str(to_y), str(meter.DRAG_MS))
                    label = ('钓鱼计量：移开遮挡范围标签的数字框，重新读取上下限' if action == 'reveal'
                             else f"钓鱼计量：拖动数字框 {data['value']} → 目标 {data['goal']}，角色 Lv{data['player']}")
                else:
                    x, y = 360, 1010
                    hold_feedback = getattr(self.device,'hold_feedback',None)
                    if callable(hold_feedback):
                        # The gesture owns its during-hold capture. No numeric
                        # cache may survive across those separately acquired pixels.
                        runtime.end_frame()
                        observed.append(hold_feedback(x,y,meter.HOLD_MS))
                    else:
                        self.device.command('shell', 'input', 'swipe', str(x), str(y),
                                            str(x), str(y), str(meter.HOLD_MS))
                    label = f"钓鱼计量：已调到 {data['value']}，长按中央圆盘 {meter.HOLD_MS/1000:g} 秒开始钓鱼"
                    # Fast fishing can finish before the next OCR-heavy loop.
                    # Capture immediately while the same device lock is held.
                    if not self.stop_event.is_set():
                        observed.append(runtime.capture())
                        self.remember(observed[-1])
                if action in ('resume_cast','start') and not fishing_stopped():
                    # 取消钓鱼的 X 只在抛竿后约 1.5~2 秒内出现(2026-10-23 实测):
                    # 抛完立刻补一帧,把这个短窗口写进证据,但不拖住主循环。
                    probe = runtime.capture()
                    self.remember(probe)
                    if meter.cancel_button(probe):
                        self.event('钓鱼计量：抛竿后捕捉到取消 X（约 2 秒窗口）')
                if calibration_metadata is not None:
                    calibration_metadata['input_end_time'] = _time().time()
                    if not observed and not fishing_stopped():
                        observed.append(runtime.capture())
                        self.remember(observed[-1])
                    if self.fishing_evidence is None:
                        self.fishing_evidence = _lookup("BossInputEvidence")(self.root,self.event,
                            folder='fishing-input',label='钓鱼计量')
                    after_id = self.state.get('frame_id') if observed else None
                    calibration_metadata['after_frame_id'] = after_id
                    self.fishing_evidence.record(calibration_metadata,calibration_before,
                                                observed[-1].copy() if observed else None)
                    if action == 'start' and len(observed) >= 2:
                        feedback = dict(input_id=uuid.uuid4().hex,
                            parent_input_id=calibration_metadata['input_id'],kind='meter_hold_feedback',
                            frame_id=uuid.uuid4().hex,after_frame_id=after_id,
                            hold_ms=meter.HOLD_MS,time=_time().time(),result='observed_only')
                        self.fishing_evidence.record(feedback,observed[0].copy(),observed[-1].copy())
            with self.lock:
                self.state['clicks'] += 1
                self.state['matches'] += 1
                self.state['last_match'] = {'name': label, 'score': 1.,
                                           'box': [x-5, y-5, x+5, y+5]}
            self.event(label)
            return observed if observed else True
        def equipment_hold(x, y, seconds, weapon, label):
            with self.device.io_lock:
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                fresh = runtime.capture()
                self.remember(fresh)
                if (not equipment.validate_hold(fresh, weapon, x, y)
                        or parallel and not parallel.boss_sailing_hold_allowed(fresh)
                        or parallel and parallel.defense and parallel.defense.owns(fresh,parallel.rank)
                        or stone and (stone.defense.encounter(fresh,39)
                                      or stone.defense._match(fresh,'timer',(188,125,610,165),.94)
                                      or stone.defense._match(fresh,'swords',(307,962,415,1057),.96))):
                    self.event('武器强化：长按前画面变化，等待重新识别')
                    return False
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                self.device.command('shell', 'input', 'swipe', str(x), str(y),
                                    str(x), str(y), str(int(seconds * 1000)))
            with self.lock:
                self.state['clicks'] += 1
                self.state['matches'] += 1
                self.state['last_match'] = {'name': label, 'score': 1.,
                                           'box': [x-5, y-5, x+5, y+5]}
            self.event(f'{label}，{seconds} 秒，坐标 ({x}, {y})')
            return True
        def make_equipment():
            entry = next((r for r, _, _ in rules if r['id'] == 'driftinggirl_equipment_upgrade'), None)
            if preview or entry is None:
                return None
            def capture_equipment():
                fresh = runtime.capture()
                self.remember(fresh)
                return fresh
            return _lookup("EquipmentUpgrade")(self.root, parallel_click, parallel_swipe,
                                    equipment_hold, self.event, entry['interval'],
                                    main_guard=lambda frame: bool(parallel is not None and parallel.sailing is not None
                                                                 and parallel.sailing.running(frame) is not None),
                                    capture=capture_equipment,
                                    scroll_guard=lambda frame: parallel is None
                                        or parallel.boss_sailing_hold_allowed(frame),
                                    opening_guard=lambda frame: not (
                                        stone is not None and (stone.defense.encounter(frame,39)
                                            or stone.defense._match(frame,'timer',(188,125,610,165),.94)
                                            or stone.defense._match(frame,'swords',(307,962,415,1057),.96))
                                        or parallel is not None and parallel.defense is not None
                                        and (parallel.defense.owns(frame,parallel.rank)
                                             or parallel.defense._match(frame,'timer',(188,125,610,165),.94)
                                             or parallel.defense._match(frame,'swords',(307,962,415,1057),.96))))
        def backpack_click(x, y, label, action):
            with self.device.io_lock:
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                if max_clicks is not None and self.state['clicks'] >= max_clicks:
                    raise InterruptedError('已达到点击次数限制')
                if st.run_deadline is not None and _time().monotonic() >= st.run_deadline:
                    raise InterruptedError('已达到运行时间限制')
                fresh = runtime.capture()
                self.remember(fresh)
                if (parallel is not None and parallel.sailing is not None
                        and parallel.sailing.running(fresh) is None):
                    # A recognized bag/map covers gameplay's lower AUTO. Its
                    # fresh top OFF icon and current world/RANK must positively
                    # establish stopping; unknown icons never authorize input.
                    panel_stopped = backpack.stopped_panel(fresh)
                    same_world_rank = parallel.sailing_context('bag',fresh)
                    if not panel_stopped or not same_world_rank:
                        self.event('背包出售：航行状态未知，拒绝输入并等待重新确认停止')
                        return False
                if (parallel is not None and parallel.sailing is not None
                        and (parallel.sailing.active or parallel.training.state.get('sailing_enabled')
                             or parallel.sailing.running(fresh) is True)):
                    self.event('背包出售：航行尚未停稳，等待重新确认')
                    return False
                if not backpack.validate(action, fresh, x, y):
                    self.event('背包出售：操作前画面变化，等待重新识别')
                    return False
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                if max_clicks is not None and self.state['clicks'] >= max_clicks:
                    raise InterruptedError('已达到点击次数限制')
                if st.run_deadline is not None and _time().monotonic() >= st.run_deadline:
                    raise InterruptedError('已达到运行时间限制')
                parallel_click(x, y, label)
            return True
        def backpack_swipe(x1, y1, x2, y2, label):
            with self.device.io_lock:
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                if max_clicks is not None and self.state['clicks'] >= max_clicks:
                    raise InterruptedError('已达到点击次数限制')
                if st.run_deadline is not None and _time().monotonic() >= st.run_deadline:
                    raise InterruptedError('已达到运行时间限制')
                fresh = runtime.capture()
                self.remember(fresh)
                if (parallel is not None and parallel.sailing is not None
                        and (parallel.sailing.active or parallel.training.state.get('sailing_enabled'))):
                    self.event('背包出售：航行尚未停稳，等待重新确认地图')
                    return False
                if not backpack.validate_scan(fresh):
                    self.event('背包出售：滑动前世界、RANK、地点或停航状态变化，等待新图')
                    return False
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                if max_clicks is not None and self.state['clicks'] >= max_clicks:
                    raise InterruptedError('已达到点击次数限制')
                if st.run_deadline is not None and _time().monotonic() >= st.run_deadline:
                    raise InterruptedError('已达到运行时间限制')
                parallel_swipe(x1, y1, x2, y2, label)
            return True
        def make_backpack():
            entry = next((r for r, _, _ in rules if r['id'] == 'driftinggirl_backpack_sell'), None)
            if preview or entry is None:
                return None
            if self.digit_reader is None:
                from digit_ocr import DigitReader
                self.digit_reader = DigitReader(self.root)
            controller = _lookup("BackpackSale")(self.root, backpack_click, self.event, self.digit_reader,
                                      entry['interval'], swipe=backpack_swipe)
            controller.enabled = lambda: any(r['id'] == 'driftinggirl_backpack_sell' and r['enabled'] for r in self.config['rules'])
            controller.main_guard = lambda frame: bool(parallel is not None and parallel.sailing is not None
                                                       and parallel.sailing.running(frame) is not None)
            controller.before_route = lambda frame: (parallel is None or parallel.sailing is None
                                                     or not parallel.sailing.step(frame, False))
            return controller
        def skill_click(x, y, label, action):
            with self.device.io_lock:
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                fresh = runtime.capture()
                self.remember(fresh)
                if action in ('open_growth','confirm') and getattr(skills,'options',None) is not None:
                    live=next((r for r in self.config['rules'] if r['id']=='driftinggirl_parallel_skills'),{})
                    desired=cruise_options(live)['skills']
                    if not live.get('enabled') or not desired['enabled'] or not desired['growth_points']:
                        skills.growth_points=False
                        return False  # Next owned step cancels an open purchase or closes the panel.
                if not skills.validate(action, fresh, x, y):
                    self.event('技能加速：点击前画面或余额变化，等待重新识别')
                    return False
                if self.stop_event.is_set():
                    raise InterruptedError('已停止')
                skills.before_click(action, fresh)
                parallel_click(x, y, label)
            return True
        def make_skills():
            entry = next((r for r, _, _ in rules if r['id'] == 'driftinggirl_parallel_skills'), None)
            if preview or entry is None:
                return None
            if self.digit_reader is None:
                from digit_ocr import DigitReader
                self.digit_reader = DigitReader(self.root)
            options=cruise_options(entry)['skills'] if 'cruise_settings' in entry else None
            if options is not None and not options['enabled']:
                return None
            controller = _lookup("ParallelSkills")(self.root, skill_click, self.event, self.digit_reader,
                                  entry['interval'], entry.get('growth_points', True),
                                  main_guard=lambda frame: bool((parallel is not None and parallel.sailing is not None
                                                                and parallel.sailing.running(frame) is not None)
                                                               or stone is not None and stone.sailing.running(frame) is not None),
                                  **({'options':options} if options is not None else {}))
            controller.force_check=options is not None
            return controller

        def configured_cruise():
            entry=next((r for r,_,_ in rules if r['id']=='driftinggirl_parallel_skills'),None)
            return cruise_options(entry)['cruise'] if entry and 'cruise_settings' in entry else None
        def calibration_sailing(action, x, y, label):
            def stopped():
                return (self.stop_event.is_set() or not calibrator.input_allowed()
                        or max_clicks is not None and self.state['clicks'] >= max_clicks
                        or st.run_deadline is not None and _time().monotonic() >= st.run_deadline)
            def raise_sailing_stop():
                if (not self.stop_event.is_set()
                        and not (max_clicks is not None and self.state['clicks'] >= max_clicks)
                        and not (st.run_deadline is not None and _time().monotonic() >= st.run_deadline)
                        and not calibrator.input_allowed()):
                    raise MeterWorkflowLimit('标尺航巡输入前恢复窗口已到期或功能已关闭')
                raise InterruptedError('已停止')
            with self.device.io_lock:
                if stopped():
                    raise_sailing_stop()
                fresh = runtime.capture()
                self.remember(fresh)
                if not calibrator.validate_sailing(action, fresh):
                    return False
                if stopped():
                    raise_sailing_stop()
                if not calibrator.reserve_sailing_input(action):
                    return False
                if stopped():
                    raise_sailing_stop()
                if action == 'stop' and not calibrator.stop_input_allowed():
                    # A slow reservation/save must not consume the remaining
                    # OFF-proof budget. Keep its reserved count; never replay.
                    return False
                metadata = dict(input_id=uuid.uuid4().hex,frame_id=self.state.get('frame_id'),
                    kind='meter_sailing_'+action,rank=calibrator.journal.get('rank'),
                    input_begin_time=_time().time(),result='issued_only')
                before = fresh.copy()
                parallel_click(x, y, label)
                metadata['input_end_time'] = _time().time()
                after = None
                if not self.stop_event.is_set():
                    after = runtime.capture()
                    metadata['after_frame_id'] = self.remember(after)
                if self.sailing_evidence is None:
                    self.sailing_evidence = _lookup("BossInputEvidence")(self.root,self.event,
                        folder='sailing-input',label='自动航行')
                self.sailing_evidence.record(metadata,before,after.copy() if after is not None else None)
            return True
        def make_calibrator(retire_expired=False):
            entry = next((r for r, _, _ in rules if r['id'] == 'driftinggirl_meter_calibration'), None)
            if preview or entry is None:
                return None
            if self.digit_reader is None:
                from digit_ocr import DigitReader
                self.digit_reader = DigitReader(self.root)
            controller = _lookup("MeterCalibration")(self.root, self.digit_reader,
                lambda action, data: fishing_action(action, data, owner=controller.world,
                    guard=(controller.resume_frame_allowed if action == 'resume_cast'
                           else controller.meter_frame_allowed),
                    reserve=controller.reserve_meter_input,
                    route_expected=controller.expected_route),
                calibration_sailing, self.event, entry['interval'])
            controller.enabled = lambda: any(r['id'] == 'driftinggirl_meter_calibration' and r['enabled']
                                              for r in self.config['rules'])
            controller.preserve_native_sale = configured_cruise() is not None
            if retire_expired:
                # A controller without the retirement hook keeps its own
                # startup behavior instead of breaking the whole run.
                retire = getattr(controller, 'retire_expired_on_start', None)
                if callable(retire):
                    retire()
            return controller
        def cruise_blockers(include_setup=True):
            # 规则本体在 drift/tasks/cruise_gate.py(可单测),这里只负责把运行时状态喂进去。
            # An archived (retired) meter issues no further input: it only reads
            # the screen until the native route is back. Letting it veto the
            # cruise watchdog deadlocks both sides, because that release proof
            # itself needs the route running. Only a live meter may block.
            return cruise_gate_blockers(include_setup=include_setup, stage=st.stage,
                handling_interrupt=st.handling_interrupt,
                setup_busy=getattr(setup, 'busy', False),
                daily_cleanup=st.daily_cleanup_deadline is not None, calibrator=calibrator,
                stone=stone,
                tasks=(equipment, backpack, skills, pass_task, guild_daily, restaurant, parallel))

        def observe_meter_candidate(reason, *, finish=False, detail=None):
            method = getattr(calibrator, 'finish_candidate_observation' if finish
                             else 'observe_candidate_blocked', None)
            if callable(method):
                observed = method(reason) if finish else method(reason, detail)
                if observed:
                    with self.lock:
                        self.state['meter'] = calibrator.status()

        def make_cruise_watchdog():
            # Recover only the user's native stone-farming route. A legacy
            # training route and a disabled task keep their own controllers.
            if configured_cruise() is not None:
                return None  # Saved cruise settings also own stopped-route recovery.
            if (preview or stone is None or not callable(getattr(stone, 'screen', None))
                    or not callable(getattr(stone, 'enabled', None)) or not stone.enabled()):
                return None
            view = CruiseView(stone)
            def click(kind, x, y, label, expected):
                def allowed():
                    return (not self.stop_event.is_set() and stone is view.owner and stone.enabled()
                            and not any(cruise_blockers().values()) and controller.input_allowed(kind)
                            and not (max_clicks is not None and self.state['clicks'] >= max_clicks)
                            and not (st.run_deadline is not None
                                     and _time().monotonic() >= st.run_deadline))
                with self.device.io_lock:
                    result = fresh_cruise_input(capture=runtime.capture,
                        validate=lambda fresh: view.validate(kind, fresh, expected), allowed=allowed,
                        touch=lambda fresh, began: self.device.touch_fast(x, y, began,
                            lambda: not allowed()),
                        clock=_time().monotonic,
                        rejected=lambda detail: controller.observe_input_rejection(kind, detail))
                    if result is None:
                        return False
                    before, receipt = result
                    metadata = dict(input_id=uuid.uuid4().hex, frame_id=self.remember(before),
                        kind=kind, stage=expected['scene'][0], rank=expected['scene'][1],
                        input_end_time=_time().time(), result='issued_only', **receipt)
                    with self.lock:
                        self.state['clicks'] += 1
                        self.state['matches'] += 1
                        self.state['last_match'] = dict(name='航巡看门狗：'+label, score=1.,
                                                       box=[x-5,y-5,x+5,y+5])
                    self.event(f'航巡看门狗：{label}，坐标 ({x}, {y})')
                    after = None
                    if not self.stop_event.is_set():
                        after = runtime.capture()
                        metadata['after_frame_id'] = self.remember(after)
                    if self.sailing_evidence is None:
                        self.sailing_evidence = _lookup('BossInputEvidence')(self.root,self.event,
                            folder='sailing-input',label='自动航行')
                    self.sailing_evidence.record(metadata,before.copy(),
                        after.copy() if after is not None else None)
                return True
            controller = _lookup('CruiseWatchdog')(running=stone.sailing.running,
                observe=view.observe, click=click, event=self.event)
            return controller
        def make_cruise_setup():
            """开局航巡配置:主画面按用户要求配好目标与自动项,再开始航巡。"""
            options=configured_cruise()
            if options is not None:
                return make_saved_cruise(options) if options['enabled'] and not preview else None
            if (preview or stone is None or not callable(getattr(stone,'screen',None))
                    or not callable(getattr(stone,'enabled',None)) or not stone.enabled()):
                return None
            # The four-target setup belongs to native stone farming. Legacy
            # training and other round trips retain their own saved routes.
            view= CruiseView(stone)
            def click(x, y, label):
                def allowed():
                    return (not self.stop_event.is_set() and stone is view.owner and stone.enabled()
                            and not any(cruise_blockers(include_setup=False).values())
                            and (controller.deadline is None or _time().monotonic()<controller.deadline)
                            and not (max_clicks is not None and self.state['clicks']>=max_clicks)
                            and not (st.run_deadline is not None and _time().monotonic()>=st.run_deadline))
                with self.device.io_lock:
                    result=fresh_cruise_input(capture=runtime.capture,allowed=allowed,
                        validate=lambda fresh:view.validate_setup(fresh,controller.input_context,(x,y),controller.desired),
                        touch=lambda fresh,began:self.device.touch_fast(x,y,began,lambda:not allowed()),
                        clock=_time().monotonic)
                    if result is None:return False
                    before,receipt=result
                    metadata=dict(input_id=uuid.uuid4().hex,frame_id=self.remember(before),
                        kind='cruise_setup',point=[x,y],input_end_time=_time().time(),result='issued_only',**receipt)
                    with self.lock:self.state['clicks']+=1
                    self.event(label+f'，坐标 ({x}, {y})')
                    after=None
                    if not self.stop_event.is_set():
                        after=runtime.capture();metadata['after_frame_id']=self.remember(after)
                    if self.sailing_evidence is None:
                        self.sailing_evidence=_lookup('BossInputEvidence')(self.root,self.event,
                            folder='sailing-input',label='自动航行')
                    self.sailing_evidence.record(metadata,before,after)
                    return True
            controller=_lookup('CruiseSetup')(sailing=stone.sailing,click=click,event=self.event,
                clock=_time().monotonic,observe=view.observe_setup)
            return controller

        def make_saved_cruise(options):
            if self.digit_reader is None:
                from digit_ocr import DigitReader
                self.digit_reader=DigitReader(self.root)
            owner=stone
            if owner is None:
                owner=StoneFarm(self.root,self.digit_reader,lambda action:False,self.event,
                               state_path=self.root/'cruise-context-unused.json')
                owner.enabled=lambda:True
            view=SettingsView(owner,self.root,self.digit_reader)
            entry=next(r for r,_,_ in rules if r['id']=='driftinggirl_parallel_skills')
            def allowed(action=None):
                current=next((r for r in self.config['rules'] if r['id']=='driftinggirl_parallel_skills'),None)
                if action=='fish-start' and (not current or not current.get('enabled')
                        or not cruise_options(current)['cruise']['enabled']):
                    return False
                return (not self.stop_event.is_set() and current is not None
                        and (current['enabled'] or controller.busy or not follow_settings)
                        and not any(cruise_blockers(include_setup=False).values())
                        and controller.deadline is not None and _time().time()<controller.deadline
                        and not (max_clicks is not None and self.state['clicks']>=max_clicks)
                        and not (st.run_deadline is not None and _time().monotonic()>=st.run_deadline))
            def click(action,x,y,hold_ms):
                if action=='tickets':
                    live=next((r for r in self.config['rules'] if r['id']=='driftinggirl_parallel_skills'),{})
                    desired=cruise_options(live)['cruise']
                    if (not live.get('enabled') or not desired['enabled'] or not desired['auto_sale']
                            or not 0<=controller.ticket_before<desired['ticket_min']):
                        controller._set('start')
                        return False
                def touch(fresh,began):
                    if not controller.reserve(action) or not allowed(action):return None
                    if hold_ms==40:
                        return self.device.touch_fast(x,y,began,lambda:not allowed(action))
                    if self.device.screenshot_transport=='mumu-native' and self.device.native_capture is not None:
                        return self.device.native_capture.touch(x,y,self.stop_event,began,
                            lambda:not allowed(action),hold=hold_ms/1000)
                    if _time().monotonic()-began>.120:return None
                    self.device.command('shell','input','swipe',str(x),str(y),str(x),str(y),str(hold_ms))
                    return dict(input_transport='adb-stationary',hold_ms=hold_ms)
                with self.device.io_lock:
                    rejection_observer=getattr(controller,'observe_input_rejection',None)
                    result=fresh_cruise_input(capture=runtime.capture,
                        allowed=lambda:allowed(action) and controller.allowed(action),
                        validate=lambda fresh:view.validate_action(fresh,controller.input_context,action,options),
                        touch=touch,clock=_time().monotonic,
                        rejected=(lambda detail:rejection_observer(action,detail))
                            if callable(rejection_observer) else None)
                    if result is None:return False
                    before,receipt=result
                    metadata=dict(input_id=uuid.uuid4().hex,frame_id=self.remember(before),
                        kind='cruise-settings:'+action,point=[x,y],policy=controller.policy,
                        input_end_time=_time().time(),result='issued_only',**receipt)
                    with self.lock:self.state['clicks']+=1
                    self.event(f'巡航设置输入：{action} ({x},{y}) {hold_ms}ms')
                    after=None
                    if not self.stop_event.is_set():
                        after=runtime.capture();metadata['after_frame_id']=self.remember(after)
                    if self.sailing_evidence is None:
                        self.sailing_evidence=BossInputEvidence(self.root,self.event,folder='sailing-input',label='巡航设置')
                    self.sailing_evidence.record(metadata,before,after)
                    return True
            controller=CruiseSettings(root=self.root,view=view,options=options,click=click,
                                      event=self.event,interval=entry['interval'])
            return controller
        def make_raid_cast():
            """RAID 甩杆辅助:战斗场地没出现 BOSS 时补一次点击(最多两次)。"""
            if preview:
                return None
            def owner():
                for candidate in (parallel, stone):
                    if candidate is not None and getattr(candidate, 'fishing', None) is not None:
                        return candidate
                return None
            def battle(frame):
                who = owner()
                return bool(who is not None and who.fishing.real_battle(frame))
            def cast(x, y, label):
                if self.stop_event.is_set():
                    return False
                parallel_click(x, y, label)
                return True
            if owner() is None:
                return None
            return _lookup('RaidCastAssist')(battle=battle, cast=cast, event=self.event)
        def make_stone(retire_expired=False):
            entry = next((r for r, _, _ in rules if r['id'] == 'driftinggirl_stone_farm'), None)
            if preview or entry is None:
                return None
            if self.digit_reader is None:
                from digit_ocr import DigitReader
                self.digit_reader = DigitReader(self.root)
            controller = _lookup('StoneFarm')(self.root, self.digit_reader,
                lambda action: portal_click(action, owner=controller), self.event, entry['interval'])
            controller.defense.action = lambda target: defense_click(
                target,owner=controller,rank=controller.defense.active_rank)
            controller.defense.scan = True
            controller.meter_action = lambda action, data: fishing_action(action, data,
                owner=controller.world, guard=controller.meter_frame_allowed,
                reserve=controller.reserve_meter_input, allowed=controller.input_allowed)
            controller.enabled = lambda: any(r['id'] == 'driftinggirl_stone_farm' and r['enabled']
                                              for r in self.config['rules'])
            policy=configured_cruise()
            controller.cruise_policy=deepcopy(policy) if policy and policy['enabled'] else None
            controller.observed_restore_allowed = lambda: not self.stop_event.is_set()
            if retire_expired:
                # Same tolerance as the meter controller: a stub or an older
                # controller without the hook must not stop the run.
                retire = getattr(controller, 'retire_expired_on_start', None)
                if callable(retire):
                    retire()
            return controller
        def call_stone(label, function, *args, **kwargs):
            nonlocal frame
            # Only this known controller boundary may consume its tagged
            # failure. Unknown tasks and unsafe UI states retain the outer stop.
            if not stone.active and not self.task_schedule.ready('stone_farm'):
                return False
            try:
                return runtime.call(label, function, *args, **kwargs)
            except RetryableStage as exc:
                retry = exc.retry_seconds
                if (exc.task != 'stone_farm' or type(retry) not in (int, float)
                        or not math.isfinite(retry) or retry <= 0
                        or st.stage != 'main' or st.handling_interrupt
                        or self.stop_event.is_set()):
                    raise
                fresh = runtime.capture()
                if not runtime.call('stone_retire_retryable', stone.retire_retryable, fresh):
                    raise
                frame = fresh
                self.remember(frame)
                with self.lock:self.state['frames'] += 1
                self.task_schedule.finish('stone_farm', stone.interval,
                                          success=False, retry_seconds=retry)
                st.stable = {key: 0 for key in st.stable}
                st.number_values.clear(); st.target_keys.clear()
                with self.lock:
                    self.state['stone'] = stone.status()
                    self.state['stage'] = 'main'
                self.event(f'次元石：本次交接已归档且未计成功，{retry:g} 秒后重新检查；其他任务继续（{exc}）')
                return False
        def pass_click(action,x,y,context):
            with self.device.io_lock:
                def stopped():
                    return (self.stop_event.is_set() or _time().monotonic()>=pass_task.deadline
                        or max_clicks is not None and self.state['clicks']>=max_clicks
                        or st.run_deadline is not None and _time().monotonic()>=st.run_deadline)
                if stopped():return False
                fresh=runtime.capture();before_id=self.remember(fresh)
                period=self.daily_schedule.period()
                if not pass_task.validate(action,fresh,context,period) or stopped():return False
                if not pass_task.reserve(action):return False
                metadata=dict(input_id=uuid.uuid4().hex,action=action,frame_id=before_id,
                    x=x,y=y,period=period,time=_time().time(),result='reserved')
                after=None
                try:
                    if stopped():return False
                    metadata['result']='uncertain'
                    parallel_click(x,y,'PASS每日奖励：'+{'open':'打开PASS','tab':'选择每日页','claim':'领取免费奖励','close':'关闭PASS'}[action])
                    metadata['result']='issued_only'
                    after=runtime.capture();metadata['after_frame_id']=self.remember(after)
                finally:
                    if self.pass_evidence is None:
                        self.pass_evidence=_lookup('BossInputEvidence')(self.root,self.event,limit=16,folder='pass-daily',label='PASS每日奖励')
                    self.pass_evidence.record(metadata,fresh,after)
                return True
        def make_pass():
            entry=next((r for r,_,_ in rules if r['id']==PASS_ID),None)
            if preview or entry is None:return None
            if self.digit_reader is None:
                from digit_ocr import DigitReader
                self.digit_reader=DigitReader(self.root)
            controller=PassDaily(self.root,self.digit_reader,pass_click,self.event)
            controller.interval=entry['interval']
            return controller
        def restaurant_click(action):
            with self.device.io_lock:
                def stopped():
                    return (self.stop_event.is_set() or not restaurant.allowed()
                        or max_clicks is not None and self.state['clicks']>=max_clicks
                        or st.run_deadline is not None and _time().monotonic()>=st.run_deadline)
                if stopped():return False
                fresh=runtime.capture();before_id=self.remember(fresh)
                if not restaurant.validate(action,fresh) or stopped():return False
                if not restaurant.reserve(action) or stopped():return False
                receipt=dict(input_id=uuid.uuid4().hex,frame_id=before_id,action=action,
                    time=_time().time(),result='uncertain',controller='restaurant_upgrade')
                after=None
                try:
                    x,y=action['x'],action['y']
                    X,Y,ms=action.get('swipe',(x,y,action.get('hold_ms',80)))
                    self.device.command('shell','input','swipe',str(x),str(y),str(X),str(Y),str(ms))
                    with self.lock:self.state['clicks']+=1
                    receipt['result']='issued_only'
                    restaurant.input_issued(action,receipt)
                    self.event('餐品强化输入：'+action['kind']+f' ({x},{y}) {ms}ms')
                    if not self.stop_event.is_set():
                        after=runtime.capture();receipt['after_frame_id']=self.remember(after)
                        if action['kind'].startswith('scan:'):restaurant.observe_map_scan(fresh,after)
                finally:
                    if self.restaurant_evidence is None:
                        self.restaurant_evidence=BossInputEvidence(self.root,self.event,limit=180,
                            folder='restaurant-input',label='餐品强化')
                    self.restaurant_evidence.record(receipt,fresh,after)
                return True

        def make_restaurant():
            entry=next((r for r,_,_ in rules if r['id']==RESTAURANT_ID),None)
            if preview or entry is None:return None
            if self.digit_reader is None:
                from digit_ocr import DigitReader
                self.digit_reader=DigitReader(self.root)
            task=RestaurantUpgrade(self.root,self.digit_reader,restaurant_click,self.event,
                                   self.config['serial'],entry['interval'],targets=entry.get('targets'))
            task.retire_expired_on_start()
            task.enabled=lambda:any(r['id']==RESTAURANT_ID and r['enabled'] for r in self.config['rules'])
            return task

        def step_restaurant(image,allow_start=True):
            was_active=restaurant.active
            due=allow_start and self.task_schedule.ready('restaurant_upgrade')
            try:
                owned=runtime.call('restaurant_upgrade',restaurant.step,image,due=due)
            except RuntimeError as exc:
                # Count failed runs on the runner, not frames or controllers:
                # waiting and config reloads must not permit infinite retries.
                self.restaurant_failures+=1
                restaurant.abandon(str(exc))
                self.task_schedule.finish('restaurant_upgrade',restaurant.interval,
                    success=False,retry_seconds=1800)
                st.stable={key:0 for key in st.stable}
                st.number_values.clear();st.target_keys.clear()
                with self.lock:
                    self.state['restaurant_upgrade']=restaurant.status()
                    self.state['stage']='main'
                if self.restaurant_failures>=RESTAURANT_FAILURE_LIMIT:
                    self.event(f'餐品强化：连续失败达到 {RESTAURANT_FAILURE_LIMIT} 次，停止运行（{exc}）')
                    raise
                self.event(f'餐品强化：本轮中止，30 分钟后重试，运行继续（{exc}）')
                return False
            if not was_active and restaurant.active:
                self.task_schedule.begin('restaurant_upgrade',restaurant.interval)
            if restaurant.done is not None:
                if restaurant.done:self.restaurant_failures=0
                # A retired run must come back in half an hour, not after the
                # three-day interval: the cookbook consumes the backlog.
                retry=(1800 if not restaurant.done
                       and getattr(restaurant,'journal',{}).get('retirement') else None)
                if retry:
                    self.task_schedule.finish('restaurant_upgrade',restaurant.interval,
                        success=bool(restaurant.done),retry_seconds=retry)
                else:
                    self.task_schedule.finish('restaurant_upgrade',restaurant.interval,success=bool(restaurant.done))
                if stop_on_main:
                    with self.lock:self.state['stop_reason']='cycle_complete' if restaurant.done else 'observation_complete'
                    self.stop_event.set()
            with self.lock:
                self.state['restaurant_upgrade']=restaurant.status()
                if owned:self.state['stage']='restaurant_'+restaurant.phase
            return owned

        def return_parallel_enabled():
            return (not preview and follow_settings
                    and self.config.get('return_parallel_enabled') is True)

        def return_parallel_action(action):
            with self.device.io_lock:
                def stopped():
                    return (self.stop_event.is_set() or not return_parallel_enabled()
                            or max_clicks is not None and self.state['clicks'] >= max_clicks
                            or st.run_deadline is not None and _time().monotonic() >= st.run_deadline)
                if stopped():return False
                fresh = runtime.capture()
                self.remember(fresh)
                if not return_parallel.validate(action, fresh) or stopped():return False
                x, y = action['x'], action['y']
                if action.get('swipe'):
                    X, Y, ms = action['swipe']
                    self.device.command('shell', 'input', 'swipe', str(x), str(y), str(X), str(Y), str(ms))
                else:
                    # 2026-10-27：与本文件 guild_daily_click 一致，用 80 毫秒定点点按。
                    # 零时长的 input tap 会被这里带过渡动画的地图/弹窗控件吞掉。
                    self.device.command('shell', 'input', 'swipe', str(x), str(y), str(x), str(y), '80')
                with self.lock:self.state['clicks'] += 1
                self.event('返回平行世界输入：' + action['kind'] + f' ({x},{y})')
                return True

        def make_return_parallel():
            task = (ReturnParallel(self.root, return_parallel_action, self.event, self.config['serial'])
                    if return_parallel_enabled() else None)
            if task is not None:task.enabled = return_parallel_enabled
            with self.lock:self.state['return_parallel'] = task.status() if task else None
            return task

        def guild_daily_click(action):
            with self.device.io_lock:
                def stopped():
                    return (self.stop_event.is_set() or not guild_daily.allowed()
                        or max_clicks is not None and self.state['clicks']>=max_clicks
                        or st.run_deadline is not None and _time().monotonic()>=st.run_deadline)
                if stopped():return False
                began=_time().monotonic();fresh=runtime.capture();before_id=self.remember(fresh)
                if not guild_daily.validate(action,fresh) or stopped():return False
                fast=action['kind']=='cancel'
                if fast and _time().monotonic()-began>.12:return False
                if not guild_daily.reserve(action) or stopped():return False
                receipt=dict(input_id=uuid.uuid4().hex,frame_id=before_id,action=action,
                    time=_time().time(),result='uncertain',controller='guild_daily')
                after=None
                try:
                    x,y=action['x'],action['y']
                    if fast:
                        contact=self.device.touch_fast(x,y,began,stopped)
                        if contact is None:return False
                        receipt.update(contact)
                        with self.lock:self.state['clicks']+=1
                    elif action.get('swipe'):
                        X,Y,ms=action['swipe']
                        self.device.command('shell','input','swipe',str(x),str(y),str(X),str(Y),str(ms))
                        with self.lock:self.state['clicks']+=1
                    else:
                        self.device.command('shell','input','swipe',str(x),str(y),str(x),str(y),'80')
                        with self.lock:self.state['clicks']+=1
                    receipt['result']='issued_only'
                    guild_daily.input_issued(action,receipt)
                    self.event('公会任务输入：'+action['kind']+f' ({x},{y})')
                    if not self.stop_event.is_set():
                        after=runtime.capture();receipt['after_frame_id']=self.remember(after)
                finally:
                    if self.guild_daily_evidence is None:
                        self.guild_daily_evidence=BossInputEvidence(self.root,self.event,limit=80,
                            folder='guild-daily-input',label='公会任务')
                    self.guild_daily_evidence.record(receipt,fresh,after)
                return True
        def make_guild_daily():
            entry=next((r for r,_,_ in rules if r['id']==GUILD_ID),None)
            if preview or entry is None:return None
            if self.digit_reader is None:
                from digit_ocr import DigitReader
                self.digit_reader=DigitReader(self.root)
            task=GuildDaily(self.root,self.digit_reader,guild_daily_click,self.event,self.config['serial'])
            task.retire_expired_on_start()
            task.enabled=lambda:any(r['id']==GUILD_ID and r['enabled'] for r in self.config['rules'])
            return task
        def step_guild_daily(image,allow_start=True):
            daily=self.daily_schedule.status()['guild_daily']
            was_active=guild_daily.active
            due=(allow_start and not daily['completed'] and self.task_schedule.ready('guild_daily',daily))
            try:
                owned=runtime.call('guild_daily',guild_daily.step,image,due=due,period=daily['period'])
            except RuntimeError as exc:
                # 单个任务超时不该停掉整个 runner:2026-10-26 18:12 公会往返超时直接把
                # 主循环带走,18:12:20→21:43:25 停摆 3 h 31 min,期间所有任务全丢。
                retry=1800
                self.event(f'公会任务：本次往返中止，{retry//60} 分钟后重试，运行继续（{exc}）')
                try:guild_daily.abandon()
                except Exception:self.event('公会任务：中止时清理记录失败，按原样保留')
                self.task_schedule.finish('guild_daily',guild_daily.interval,daily,
                    success=False,retry_seconds=retry)
                with self.lock:self.state['guild_daily']=guild_daily.status()
                return False
            if not was_active and guild_daily.active:
                self.task_schedule.begin('guild_daily',guild_daily.interval,daily)
            if guild_daily.done is not None:
                if guild_daily.done:self.daily_schedule.mark_complete('guild_daily',guild_daily.started_period)
                self.task_schedule.finish('guild_daily',guild_daily.interval,self.daily_schedule.status()['guild_daily'],
                    success=bool(guild_daily.done))
            with self.lock:
                self.state['guild_daily']=guild_daily.status()
                if owned:self.state['stage']='guild_daily_'+guild_daily.phase
            return owned
        def step_pass(image):
            daily=self.daily_schedule.status()['pass_daily']
            due=(not daily['completed'] and self.task_schedule.ready('pass_daily',daily))
            was_active=pass_task.active
            if was_active and not any(r['id']==PASS_ID and r['enabled'] for r in self.config['rules']):
                pass_task.claim_allowed=False
            ready=(pass_task.main_visible(image) or
                   (self.daily_return_visible(image,st.matchers) if 'driftinggirl_daily_return' in st.matchers else
                    pass_task.entry(image)))
            owned=runtime.call('pass_daily',pass_task.step,image,due=due,period=daily['period'],main_ready=ready)
            if not was_active and pass_task.active:
                self.task_schedule.begin('pass_daily',pass_task.interval,daily)
            if pass_task.done is not None:
                if pass_task.done:
                    self.daily_schedule.mark_complete('pass_daily',pass_task.started_period)
                self.task_schedule.finish('pass_daily',pass_task.interval,self.daily_schedule.status()['pass_daily'],
                    success=bool(pass_task.done),retry_seconds=None if pass_task.done else pass_task.interval)
            return owned
        st.run_deadline = _time().monotonic() + timeout if timeout is not None else None
        st.defense_pause_at = None
        st.completion_pause_at = None
        def capture_loop_frame():
            return runtime.capture_loop()
        def record_failure(exc):
            with self.lock:
                self.state["last_error"] = str(exc)
            self.event(f"运行停止：{exc}")
            self.log.exception("Runner failed")
            if self.device.image is not None:
                path = self.root / "logs" / f"error-{datetime.now():%Y%m%d-%H%M%S}.png"
                path.write_bytes(_lookup("png_bytes")(self.device.image))
                with self.lock:
                    failure = {key: deepcopy(self.state.get(key)) for key in
                               ('stage', 'frames', 'clicks', 'last_error', 'last_match', 'numbers', 'guild_selection')}
                failure.update(stage_trigger=st.stage_trigger,
                               stage_seconds=round(_time().monotonic() - st.stage_started_at, 3),
                               stable_frames=st.stable)
                if optional_cleanup is not None:
                    failure['optional_cleanup'] = optional_cleanup.status()
                path.with_suffix('.json').write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding='utf-8')
        try:
            runtime.start_engine()
            self.event('MaaFramework 执行器已启动；保留 NextRun 与实时触发调度')
            parallel = make_parallel()
            equipment = make_equipment()
            backpack = make_backpack()
            skills = make_skills()
            calibrator = make_calibrator(retire_expired=True)
            stone = make_stone(retire_expired=True)
            pass_task = make_pass()
            guild_daily = make_guild_daily()
            restaurant = make_restaurant()
            return_parallel = make_return_parallel()
            recovery = (_lookup("GameRecovery")(self.root, self.device, self.event, parallel_click,
                                     capture=runtime.capture)
                        if not preview and rules
                        and callable(getattr(self.device, 'game_state', None)) else None)
            # 打转检测只在正式运行时启用:单项识别预览没有 stage 迁移。
            progress = None if preview else ProgressGuard(_time().monotonic)
            for frame in module.loop(skip_first=False, capture=capture_loop_frame):
                if self.stop_event.is_set():
                    break
                if st.run_deadline is not None and _time().monotonic() >= st.run_deadline:
                    with self.lock:
                        self.state["stop_reason"] = "timeout"
                    self.event("本次测试达到等待时限，已停止")
                    self.stop_event.set()
                    break
                if st.stage != st.nav_stage:
                    # 换了步骤就重置退回点击预算,避免上一阶段的计数影响本阶段。
                    st.nav_stage, st.nav_clicks = st.stage, 0
                if st.stage != st.logged_stage:
                    # 只写日志,不发事件:排查"为什么又跑了这一步"时需要阶段迁移轨迹。
                    previous, st.logged_stage = st.logged_stage, st.stage
                    if previous is not None:
                        self.log.info('阶段切换：%s → %s（触发：%s）',
                                      previous, st.stage, st.stage_trigger or '无')
                    else:
                        self.log.info('阶段开始：%s（触发：%s）', st.stage, st.stage_trigger or '无')
                    # 面板的 stage 以前只在十几个分支里同步,于是经常显示旧值
                    # (日志已经进了 raid_battle,面板还停在 game_recovery)。统一在这里同步。
                    with self.lock:
                        self.state['stage'] = st.stage
                    if previous == 'raid_wait' and st.stage == 'raid_battle':
                        # 2026-10-22 更新后:点完开始进入战斗场地时 BOSS 不会自己出现,
                        # 要再点一次屏幕让角色甩杆才开打。
                        if raid is None:
                            raid = make_raid_cast()
                        if raid is not None:
                            raid.arm()
                if progress is not None and progress.note(st.stage):
                    # 阶段时间都够,却一直 A→B→A 往返:这是"画面没进展",
                    # 比等到某个 deadline 更早暴露问题,也避免整轮静默空转。
                    raise GameStuck('画面在 '
                                    + '、'.join(f'{edge} ×{count}' for edge, count in
                                                progress.status()['edges'].items())
                                    + ' 之间反复往返，未取得进展')
                self.remember(frame)
                if mailbox_pending_evidence is not None:
                    receipt, before = mailbox_pending_evidence
                    receipt['after_frame_id'] = self.state['frame_id']
                    receipt['after_at'] = time.time()
                    self.mailbox_evidence.record(receipt, before, frame.copy())
                    mailbox_pending_evidence = None
                if guild_pending_evidence is not None:
                    receipt, before = guild_pending_evidence
                    receipt['after_frame_id'] = self.state['frame_id']
                    receipt['after_at'] = time.time()
                    self.guild_evidence.record(receipt, before, frame.copy())
                    guild_pending_evidence = None
                if popup_pending_evidence is not None:
                    receipt, before = popup_pending_evidence
                    receipt['after_frame_id'] = self.state['frame_id']
                    receipt['after_at'] = time.time()
                    self.popup_evidence.record(receipt, before, frame.copy())
                    popup_pending_evidence = None
                current_period = self.daily_schedule.period()
                if not preview and current_period != st.daily_period:
                    for key in DAILY_ENTRIES:
                        module.interval_timer.pop(key, None)
                        if key in st.stable:
                            st.stable[key] = 0
                    st.daily_period = current_period
                    self.event('已进入北京时间 08:00 刷新后的新周期，每日奖励和公会商店可重新检查')
                with self.lock:
                    self.state["frames"] += 1
                    self.state["last_match"] = None
                    if st.stage != 'main' or st.handling_interrupt:
                        observe_meter_candidate('stage_owned', detail=st.stage)
                    # Finish the current workflow with its snapshot. Only replace tasks
                    # on the main screen, so disabling mail never strands an open mailbox.
                    sailing = getattr(parallel,'sailing',None)
                    if follow_settings and revision != self.config_revision and sailing is not None:
                        upcoming = next((r for r in self.config['rules']
                                         if r['id'] == 'driftinggirl_parallel_world'),None)
                        options = training_options(upcoming.get('training')) if upcoming else None
                        boss_route = sailing.profile == 'boss'
                        cutoff_changed = (options is not None and boss_route
                            and options['boss_auto_sailing_until_rank'] !=
                            parallel.training.options['boss_auto_sailing_until_rank'])
                        if ((sailing.active or parallel.training.state.get('sailing_enabled'))
                                and (not upcoming or not upcoming['enabled'] or not options['enabled']
                                     or cutoff_changed
                                     or (not options['boss_auto_sailing_until_rank'] if boss_route
                                         else not options['auto_sailing']))):
                            parallel.sailing_shutdown = True
                    if ((follow_settings or 'driftinggirl_parallel_skills' in scoped_rule_ids)
                            and st.stage == "main" and (equipment is None or not equipment.active)
                            and (backpack is None or not backpack.active)
                            and (skills is None or not skills.active)
                            and (calibrator is None or not calibrator.active)
                            and (stone is None or not stone.active)
                            and (pass_task is None or not pass_task.active)
                            and (guild_daily is None or not guild_daily.active)
                            and (restaurant is None or not restaurant.active)
                            and not getattr(setup,'busy',False)
                            and (return_parallel is None or not return_parallel.active or not return_parallel_enabled())
                            and (sailing is None or not sailing.active
                                 and (not parallel.sailing_shutdown or parallel.shutdown_complete))
                            and revision != self.config_revision):
                        rules = [self.load_rule(deepcopy(r)) for r in self.config["rules"] if r["enabled"]
                                 and (follow_settings or r['id'] in scoped_rule_ids)]
                        st.matchers = {r["id"]: (r, b, o) for r, b, o in rules}
                        st.stable = {r["id"]: 0 for r, _, _ in rules}
                        st.number_values.clear()
                        st.target_keys.clear()
                        st.previous, st.previous_progress, st.consecutive = None, None, 0
                        revision = self.config_revision
                        self.state.update(active_rule_ids=list(st.matchers), applied_revision=revision, numbers={})
                        self.event("功能开关已应用" if rules else "所有功能已关闭，等待开启任务")
                        parallel = make_parallel()
                        equipment = make_equipment()
                        backpack = make_backpack()
                        skills = make_skills()
                        observe_meter_candidate('settings_reload', finish=True)
                        calibrator = make_calibrator()
                        stone = make_stone()
                        cruise = make_cruise_watchdog()
                        setup = None
                        pass_task = make_pass()
                        guild_daily = make_guild_daily()
                        restaurant = make_restaurant()
                        return_parallel = make_return_parallel()
                        if recovery is None and not preview and rules and callable(getattr(self.device, 'game_state', None)):
                            recovery = _lookup("GameRecovery")(self.root, self.device, self.event, parallel_click,
                                                               capture=runtime.capture)
                        self.state['parallel'] = parallel.status() if parallel else None
                        self.state['equipment'] = equipment.status() if equipment else None
                        self.state['backpack'] = backpack.status() if backpack else None
                        self.state['skills'] = skills.status() if skills else None
                        self.state['meter'] = calibrator.status() if calibrator else None
                        self.state['stone'] = stone.status() if stone else None
                if recover_potion(frame):
                    if runtime.wait(self.config['poll_interval']):
                        break
                    continue
                recovery_owner = (parallel if parallel is not None else
                                  stone if stone is not None and stone.enabled() else None)
                if recovery is not None and rules:
                    if recovery_owner is not None:
                        ready = (recovery_owner.training.main(frame)
                                 or recovery_owner.sailing is not None and recovery_owner.sailing.running(frame) is not None)
                        known = ready or recovery_owner._parallel(frame) or recovery_owner._observatory(frame)
                    else:
                        known,ready=recovery.scene(frame)
                    # Recovery's fresh login guard holds device.io_lock.
                    # Keep its capture and input in the same Maa callback so
                    # capture cannot queue on another thread waiting for it.
                    recovery_started = _time().monotonic()
                    result = runtime.call('game_recovery', recovery.step, frame, known, ready)
                    recovery_wait = st.account_recovery_wait(result, recovery_started, _time().monotonic())
                    if optional_cleanup is not None and st.stage_deadline is not None:
                        st.stage_deadline = min(st.stage_deadline, optional_cleanup.deadline)
                    if result == 'ready':
                        # Only observation and fresh controllers follow a restart.
                        # Keep cooldowns, daily records and the persisted training file.
                        st.reset_after_return()
                        optional_cleanup = None
                        st.stage_started_at = _time().monotonic()
                        st.stage_ready_at = 0
                        st.handling_interrupt = False
                        st.interrupt_started_at = None
                        finish_scheduled_daily()
                        st.daily_attempt, st.daily_verified = None, False
                        st.stable = {r['id']: 0 for r, _, _ in rules}
                        st.number_values.clear()
                        st.target_keys.clear()
                        st.previous, st.previous_progress, st.consecutive = None, None, 0
                        parallel, equipment = make_parallel(), make_equipment()
                        backpack = make_backpack()
                        skills = make_skills()
                        observe_meter_candidate('game_recovered', finish=True)
                        calibrator = make_calibrator(retire_expired=True)
                        stone = make_stone(retire_expired=True)
                        if pass_task is not None and pass_task.active:
                            self.task_schedule.finish('pass_daily',pass_task.interval,
                                self.daily_schedule.status()['pass_daily'],success=False)
                        pass_task=make_pass()
                        guild_daily=make_guild_daily()
                        restaurant=make_restaurant()
                        return_parallel=make_return_parallel()
                        cruise=make_cruise_watchdog()
                        with self.lock:
                            self.state['recoveries'] += 1
                            self.state['stage'] = 'main'
                            self.state['parallel'] = parallel.status() if parallel else None
                            self.state['equipment'] = equipment.status() if equipment else None
                            self.state['backpack'] = backpack.status() if backpack else None
                            self.state['skills'] = skills.status() if skills else None
                            self.state['meter'] = calibrator.status() if calibrator else None
                            self.state['stone'] = stone.status() if stone else None
                    elif result == 'wait':
                        observe_meter_candidate('game_recovery')
                        with self.lock:
                            self.state['stage'] = 'game_recovery'
                        if max_clicks is not None and self.state['clicks'] >= max_clicks:
                            with self.lock:
                                self.state['stop_reason'] = 'click_limit'
                            self.stop_event.set()
                            break
                        if runtime.wait(self.config['poll_interval']):
                            break
                        continue
                if optional_cleanup is not None:
                    now = _time().monotonic()
                    if now >= optional_cleanup.deadline:
                        raise RuntimeError('可跳过任务的安全关闭超过40秒，仍未识别主画面；保留未完成记录，拒绝未知画面点击')
                    if optional_cleanup.observe_return(frame, st.matchers, now):
                        finish_scheduled_daily(False)
                        st.daily_attempt, st.daily_verified = None, False
                        st.reset_after_return()
                        st.handling_interrupt = False
                        st.interrupt_started_at = None
                        st.stage_started_at = now
                        st.stable = {key:0 for key in st.stable}
                        st.number_values.clear(); st.target_keys.clear()
                        with self.lock:
                            self.state['stage'] = 'main'
                            self.state['optional_cleanup'] = {**optional_cleanup.status(), 'returned': True}
                        optional_cleanup = None
                        self.event('可跳过任务：连续三帧三秒确认已返回主画面，其他任务继续；本次不记领奖完成')
                        if stop_on_main:
                            with self.lock:self.state['stop_reason'] = 'cycle_complete'
                            self.stop_event.set()
                            break
                        continue
                # Detect a known modal before an active RANK40 workflow can
                # consume the frame. Closing it does not renew any saved
                # portal/fishing deadline or allow an expired input again.
                interruption = None
                if not preview and not st.handling_interrupt:
                    for rule, button, offset in rules:
                        if rule.get('interrupt') and module.appear(button, offset=offset, similarity=rule['threshold']):
                            interruption = rule['id']
                            break
                if interruption is not None:
                    observe_meter_candidate('popup', detail=interruption)
                    if st.interrupt_started_at is None:
                        st.interrupt_started_at = _time().monotonic()
                    if _time().monotonic() - st.interrupt_started_at >= 15:
                        raise RuntimeError('任务弹窗持续遮挡且无法稳定关闭，已停止')
                else:
                    st.interrupt_started_at = None
                if cruise is not None and (interruption is not None or any(cruise_blockers().values())):
                    cruise.suspend()
                if (not preview and stone is not None and stone.phase == 'first_verify'
                        and (interruption is not None or st.handling_interrupt)):
                    stone._confirmed(None)
                    stone.first_result_rank = None
                completion_owner = (parallel if parallel is not None and not getattr(parallel,'only_sailing_shutdown',False)
                                    else stone if parallel is None else None)
                if (not preview and completion_owner is not None
                        and interruption is None and not st.handling_interrupt
                        and callable(getattr(completion_owner,'completion_step',None))):
                    completion_started = _time().monotonic()
                    ready = None
                    if completion_owner is parallel:
                        ready = (parallel.training.main(frame)
                                 or parallel.sailing is not None and parallel.sailing.running(frame) is not None
                                 or equipment is not None and equipment.panel(frame))
                        runtime.call('boss_route_rank',parallel.observe_boss_sailing_rank,frame,ready=ready)
                    completion_owned = (call_stone('completion',stone.completion_step,frame,ready=ready)
                                        if completion_owner is stone else
                                        runtime.call('completion',completion_owner.completion_step,frame,ready=ready))
                    if completion_owned or st.completion_pause_at is not None:
                        until = _time().monotonic()
                        duration = until-(st.completion_pause_at if st.completion_pause_at is not None else completion_started)
                        for controller in (skills,equipment,backpack,parallel,pass_task):
                            if controller is not None:
                                controller.suspend(duration)
                        st.stage_started_at += duration
                        st.stage_ready_at += duration
                        if st.stage_deadline is not None:
                            st.stage_deadline += duration
                        st.completion_pause_at = until if completion_owned else None
                    if completion_owned:
                        with self.lock:
                            self.state['stage'] = 'parallel_completion' if completion_owner is parallel else 'stone_' + stone.phase
                            if completion_owner is stone:
                                self.state['stone'] = stone.status()
                        if max_clicks is not None and self.state['clicks'] >= max_clicks:
                            with self.lock:
                                self.state['stop_reason'] = 'click_limit'
                            self.stop_event.set()
                            break
                        if runtime.wait(self.config['poll_interval']):
                            break
                        continue
                # Outcome flashes are brief. Observe before controller delays,
                # equipment ownership or a collecting workflow can skip them.
                # This never clicks and rejects obscured/non-game frames.
                if parallel is not None:
                    if backpack is not None and backpack.active:
                        parallel.observe_backpack_rank(frame,backpack)
                    runtime.call('observe',parallel.observe,frame)
                    with self.lock:
                        self.state['parallel'] = parallel.status()
                if (not preview and popup_raid.observe(st, frame,
                        now=_time().monotonic(), stopped=self.stop_event.is_set())
                        and popup_raid.restore(st, now=_time().monotonic(),
                                               stopped=self.stop_event.is_set())):
                    with self.lock:
                        self.state['stage'] = st.stage
                        self.state['matches'] += 1
                    self.event('任务弹窗已关闭：连续三帧三秒确认原RAID等待页，接续原等待流程；保留原截止时间，不追加点击')
                    continue
                if (not preview and interruption is None and parallel is not None
                        and parallel.boss_sailing_stop_requested()):
                    route_main = parallel.sailing.running(frame) is not None
                    route_panel = equipment is not None and equipment.validate_battle_close(frame)
                    if route_main or route_panel:
                        # Stop game AUTO before optional battle taps or another
                        # long upgrade can carry the early route past its cutoff.
                        # Inputs retain independent new-frame world/rank/panel
                        # or running-AUTO validation; no old coordinates repeat.
                        if route_panel:
                            close_boss_sailing_equipment()
                        else:
                            runtime.call('boss_route_stop',parallel._stop_sailing,frame)
                        with self.lock:
                            self.state['parallel'] = parallel.status()
                            self.state['stage'] = 'parallel_boss_sailing_handoff'
                        if runtime.wait(self.config['poll_interval']):
                            break
                        continue
                # Ordinary resource battles can close the crown list too.
                # Keep this read-only confirmation outside the battle-owned
                # stable counters, which are deliberately reset every frame.
                crown_rule = st.matchers.get('driftinggirl_achievements_crown')
                crown_recovery_rule = st.matchers.get('driftinggirl_achievements_parallel_battle_recover')
                crown_ready = st.matchers.get('driftinggirl_achievements_ready')
                crown_fold = st.matchers.get('driftinggirl_achievements_fold')
                resource_context = None
                if (not preview and not st.handling_interrupt and interruption is None
                        and parallel is not None and st.stage == 'achievements_claim'
                        and crown_rule is not None and crown_recovery_rule is not None
                        and crown_ready is not None and crown_fold is not None
                        and 'driftinggirl_achievements_open' in st.matchers
                        and st.stage_trigger == crown_rule[0]['name']
                        and st.stage_deadline is not None and _time().monotonic() < st.stage_deadline
                        and not module.appear(crown_ready[1],offset=crown_ready[2],similarity=crown_ready[0]['threshold'])
                        and not module.appear(crown_fold[1],offset=crown_fold[2],similarity=crown_fold[0]['threshold'])):
                    resource_context = parallel.resource_battle_recovery_context(frame)
                if resource_context is None:
                    st.resource_crown_context,st.resource_crown_frames = None,0
                    st.resource_crown_since = None
                else:
                    st.resource_crown_frames = (st.resource_crown_frames+1
                        if resource_context == st.resource_crown_context else 1)
                    if resource_context != st.resource_crown_context:
                        st.resource_crown_since = _time().monotonic()
                    st.resource_crown_context = resource_context
                if (st.resource_crown_frames >= max(3,crown_recovery_rule[0]['stable_frames'] if crown_recovery_rule else 3)
                        and st.resource_crown_since is not None
                        and _time().monotonic()-st.resource_crown_since >= max(3,crown_recovery_rule[0].get('min_stage_seconds',0))):
                    # Preserve the original entry cooldown and bounded retry
                    # history. This abandons an interrupted attempt, never
                    # records a claimed reward or extends its old deadline.
                    history = st.entry_recoveries.setdefault('driftinggirl_achievements_open',deque())
                    now = _time().monotonic()
                    while history and now-history[0] >= 300:
                        history.popleft()
                    if len(history) >= 3:
                        raise RuntimeError('皇冠奖励入口五分钟内已恢复三次仍未打开，已停止')
                    history.append(now)
                    st.stage,st.stage_deadline,st.stage_trigger = 'main',None,None
                    st.stage_started_at,st.stage_ready_at = now,0
                    finish_scheduled_daily()
                    st.daily_attempt,st.daily_verified = None,False
                    st.stable = {key:0 for key in st.stable}
                    st.number_values.clear();st.target_keys.clear()
                    st.previous,st.previous_progress,st.consecutive = None,None,0
                    st.resource_crown_context,st.resource_crown_frames = None,0
                    st.resource_crown_since = None
                    with self.lock:
                        self.state['recoveries'] += 1
                        self.state['matches'] += 1
                        self.state['stage'] = 'main'
                    self.event('皇冠奖励被普通资源战斗打断：已连续确认当前世界/RANK和已验证航行，保留入口间隔恢复；本次不记领奖完成')
                # Fishing can also interrupt the first shop-open attempt.
                # Native AUTO has no legacy training controller, so its proof
                # independently reads the current world/RANK and route. Only
                # abandon an uncompleted entry, preserving cooldown/deadline.
                entry_key,next_key,entry_label = {
                    'guild_special':('driftinggirl_guild_open','driftinggirl_guild_special','公会商店'),
                    'event_potion_shop':('driftinggirl_event_potion_open','driftinggirl_event_potion_select','活动药水'),
                }.get(st.stage,(None,None,None))
                resource_entry = st.matchers.get(entry_key)
                resource_next = st.matchers.get(next_key)
                entry_context = None
                if (not preview and not st.handling_interrupt and interruption is None
                        and not self.stop_event.is_set() and (parallel is not None or stone is not None)
                        and resource_entry is not None and resource_next is not None
                        and st.stage_trigger == resource_entry[0]['name']
                        and st.stage_deadline is not None and _time().monotonic() < st.stage_deadline
                        and not module.appear(resource_next[1],offset=resource_next[2],
                                              similarity=resource_next[0]['threshold'])):
                    if parallel is not None:
                        verified = parallel.resource_battle_recovery_context(frame)
                        if verified is not None:entry_context = (entry_key,'training',verified)
                    elif stone is not None:
                        verified = stone.native_entry_recovery_context(frame)
                        if verified is not None:entry_context = (entry_key,'native',verified)
                if entry_context is None:
                    st.resource_entry_context,st.resource_entry_frames = None,0
                    st.resource_entry_since = None
                else:
                    st.resource_entry_frames = (st.resource_entry_frames+1
                        if entry_context == st.resource_entry_context else 1)
                    if entry_context != st.resource_entry_context:
                        st.resource_entry_since = _time().monotonic()
                    st.resource_entry_context = entry_context
                if (st.resource_entry_frames >= 3 and st.resource_entry_since is not None
                        and _time().monotonic()-st.resource_entry_since >= 3
                        and not self.stop_event.is_set() and st.stage_deadline is not None
                        and _time().monotonic() < st.stage_deadline):
                    history = st.entry_recoveries.setdefault(entry_key,deque())
                    now = _time().monotonic()
                    while history and now-history[0] >= 300:
                        history.popleft()
                    if len(history) >= 3:
                        raise RuntimeError(f'{entry_label}入口五分钟内已恢复三次仍未打开，已停止')
                    history.append(now)
                    st.stage,st.stage_deadline,st.stage_trigger = 'main',None,None
                    st.stage_started_at,st.stage_ready_at = now,0
                    finish_scheduled_daily()
                    st.daily_attempt,st.daily_verified = None,False
                    st.stable = {key:0 for key in st.stable}
                    st.number_values.clear();st.target_keys.clear()
                    st.previous,st.previous_progress,st.consecutive = None,None,0
                    st.resource_entry_context,st.resource_entry_frames = None,0
                    st.resource_entry_since = None
                    with self.lock:
                        self.state['recoveries'] += 1
                        self.state['matches'] += 1
                        self.state['stage'] = 'main'
                    if entry_context[1] == 'native':
                        self.event(f'{entry_label}入口未展开：已连续三帧三秒确认当前世界/RANK、游戏航巡和实际钓鱼画面，保留入口间隔恢复；本次不记购买完成')
                    else:
                        self.event(f'{entry_label}入口被普通资源战斗打断：已连续三帧三秒确认当前世界/RANK和已验证航行，保留入口间隔恢复；本次不记购买完成')
                if (not preview and guild_cleanup_deadline is not None
                        and st.stage != 'main' and _time().monotonic() >= guild_cleanup_deadline):
                    raise RuntimeError('公会窗口清理已到原截止，未确认返回，已停止；不重复关闭或购买')
                if (not preview and guild_return.observe(frame,st,readonly_guild_world,
                        now=_time().monotonic(),stopped=self.stop_event.is_set(),
                        interrupted=interruption is not None)
                        and not self.stop_event.is_set() and _time().monotonic()<st.stage_deadline):
                    # A return can credit only stock verified before the close;
                    # the combat picture itself proves no purchase or portal cycle.
                    if st.daily_attempt is not None:
                        feature, started_period = st.daily_attempt
                        if st.daily_verified:
                            self.daily_schedule.mark_complete(feature,started_period)
                        finish_scheduled_daily(self.daily_schedule.is_complete(feature))
                    st.daily_attempt, st.daily_verified = None, False
                    st.reset_after_return()
                    st.stage_started_at = _time().monotonic()
                    st.stable = {key:0 for key in st.stable}
                    st.number_values.clear();st.target_keys.clear()
                    guild_cleanup_deadline = guild_cleanup_origin = guild_cleanup_rule = None
                    with self.lock:
                        self.state['stage'] = 'main'
                        self.state['matches'] += 1
                    self.event('公会商品：连续三帧三秒确认窗口已关闭及当前世界/RANK实际钓鱼；只读返回，没有追加点击或补记购买')
                    if stop_on_main:
                        with self.lock:self.state['stop_reason'] = 'cycle_complete'
                        self.stop_event.set()
                        break
                    continue
                if observe_native_crown_return(frame, st, module, stone, preview=preview,
                        interruption=interruption, stop=self.stop_event, clock=_time):
                    # Combat clears the general scanner's votes each frame.
                    # Commit only this separately verified read-only return;
                    # retain the entry timer, schedule and all game journals.
                    st.reset_after_return()
                    st.stage_started_at = _time().monotonic()
                    st.stable = {key:0 for key in st.stable}
                    st.number_values.clear();st.target_keys.clear()
                    st.previous,st.previous_progress,st.consecutive = None,None,0
                    st.crown_return_context,st.crown_return_frames,st.crown_return_since = None,0,None
                    with self.lock:
                        self.state['stage'] = 'main'
                        self.state['matches'] += 1
                    self.event('皇冠奖励：已在战斗接管前连续三帧三秒确认菜单关闭、当前世界/RANK和实际钓鱼，返回主画面；没有追加点击')
                    if stop_on_main:
                        with self.lock:self.state['stop_reason'] = 'cycle_complete'
                        self.stop_event.set()
                        break
                    continue
                if not preview and interruption is None:
                    if stone is not None and stone.active:
                        call_stone('stone_deadline', stone.check_deadline)
                    if (st.stage == 'main' and not st.handling_interrupt and calibrator is not None
                            and getattr(calibrator, 'awaiting_recovery', False)):
                        runtime.call('meter_retired_observation', calibrator.pre_battle_step, frame)
                        with self.lock:
                            self.state['meter'] = calibrator.status()
                    if (st.stage == 'main' and not st.handling_interrupt
                            and calibrator is not None and calibrator.active
                            and calibrator.phase in ('stop','route_context','prepare')
                            and runtime.call('meter_stop_before_battle',
                                             calibrator.pre_battle_step, frame)):
                        # A saved stop must not wait behind repeated battle
                        # bursts. The controller rejects BOSS/unknown frames;
                        # only its existing fresh guarded AUTO input can run.
                        with self.lock:
                            self.state['meter'] = calibrator.status()
                            self.state['stage'] = 'meter_' + calibrator.phase
                        if runtime.wait(self.config['poll_interval']):
                            break
                        continue
                    if stone is not None and getattr(stone,'phase',None) == 'first_verify':
                        # Observing a cast result issues no input. It must run
                        # before a real boss burst can skip the UI controllers.
                        call_stone('stone_first_observe',stone.observe_first_cast,frame)
                        # The ordinary controller must not vote on this same
                        # outer capture again when combat did not take it.
                        stone.first_preobserved_frame = frame
                        if stone.active:
                            call_stone('stone_deadline', stone.check_deadline)
                    defense_started = _time().monotonic()
                    defense_owner = parallel if parallel is not None else (
                        stone if stone is not None and stone.enabled() else None)
                    if (defense_owner is stone and stone is not None
                            and (getattr(stone,'boss_abandon_handoff',False)
                                 or getattr(stone,'handoff_stop_requested',False)
                                  or getattr(stone,'first_result_observation_pending',False)
                                 or getattr(stone,'first_route_resume_requested',False))):
                        # The one explicitly approved lower handoff must reach
                        # its OFF/battle/idle confirmation; combat bursts cannot
                        # starve it or renew its original return deadline.
                        defense_owner = None
                    attack_rank = (stone.defense.encounter_rank(frame) if defense_owner is stone and stone is not None
                                   else parallel.rank if parallel is not None else None)

                    def step_defense():
                        if (defense_owner is not None and equipment is not None
                                and defense_owner.defense and defense_owner.defense.owns(frame,attack_rank)
                                and equipment.validate_battle_close(frame)):
                            close_defense_equipment(defense_owner,attack_rank)
                        return bool(defense_owner and defense_owner.defense
                                    and defense_owner.defense.step(frame,attack_rank))
                    # Enable the per-frame resolver only for this attack call.
                    # Every burst capture rechecks the actual boss encounter;
                    # the configured switches remain unchanged for OCR/tasks.
                    guard = getattr(defense_owner, 'attack_options', None)
                    if guard is not None:
                        with guard():
                            defense_owned = runtime.call('battle',step_defense)
                    else:
                        defense_owned = runtime.call('battle',step_defense)
                    if stone is not None and stone.active:
                        call_stone('stone_deadline', stone.check_deadline)
                    with self.lock:
                        self.state['defense'] = defense_owner.defense.status() if defense_owner and defense_owner.defense else None
                    if defense_owned or st.defense_pause_at is not None:
                        # UI tasks retain their remaining time, not a deadline
                        # consumed by a higher-priority verified boss encounter.
                        until = _time().monotonic()
                        duration = until-(st.defense_pause_at if st.defense_pause_at is not None else defense_started)
                        for controller in (skills,equipment,backpack,parallel,pass_task):
                            if controller is not None:
                                controller.suspend(duration)
                        st.stage_started_at += duration
                        st.stage_ready_at += duration
                        if st.stage_deadline is not None:
                            st.stage_deadline += duration
                        st.defense_pause_at = until if defense_owned else None
                    if defense_owned:
                        observe_meter_candidate('battle_defense')
                        st.popup_map_frames = 0
                        st.stable = {key:0 for key in st.stable}
                        st.number_values.clear()
                        st.target_keys.clear()
                        with self.lock:
                            self.state['parallel'] = parallel.status() if parallel else None
                            self.state['stone'] = stone.status() if stone else None
                            self.state['defense'] = defense_owner.defense.status()
                            self.state['equipment'] = equipment.status() if equipment else None
                            self.state['backpack'] = backpack.status() if backpack else None
                            self.state['skills'] = skills.status() if skills else None
                            self.state['meter'] = calibrator.status() if calibrator else None
                            self.state['stage'] = ('parallel_boss_defense' if defense_owner.defense.owns(frame,attack_rank)
                                                   else 'parallel_battle_tap')
                        if max_clicks is not None and self.state['clicks'] >= max_clicks:
                            with self.lock:
                                self.state['stop_reason'] = 'click_limit'
                            self.stop_event.set()
                            break
                        if runtime.wait(min(self.config['poll_interval'],.02)):
                            break
                        continue
                if not preview and st.handling_interrupt and st.stage == 'task_popup_return':
                    close_rule=st.matchers.get('driftinggirl_task_popup_close')
                    return_rule=st.matchers.get('driftinggirl_task_popup_return')
                    notice_gone=(close_rule is not None and not module.appear(close_rule[1],
                        offset=close_rule[2],similarity=close_rule[0]['threshold']))
                    return_ready=return_rule is not None and _time().monotonic() >= st.stage_ready_at
                    if notice_gone:
                        st.popup_reclick=0
                        st.popup_reclick_frames=0
                    else:
                        st.popup_reclick_frames+=1
                    panel_return=observe_popup_equipment_return(frame,st,equipment,
                        notice_gone=notice_gone and return_ready,stop=self.stop_event,clock=_time)
                    native_task_return=(bool(stone.observe_task_popup_return(frame,notice_gone=notice_gone))
                                        if return_ready and stone is not None else False)
                    restaurant_return=(bool(restaurant.observe_popup_return(frame,notice_gone=notice_gone))
                                       if return_ready and restaurant is not None else False)
                    cruise_return=popup_cruise.observe(st,frame,setup,
                        notice_gone=notice_gone and return_ready,now=_time().monotonic(),
                        stopped=self.stop_event.is_set())
                    visible=(notice_gone and return_ready and parallel is not None
                             and parallel.popup_map_return(frame))
                    st.popup_map_frames=st.popup_map_frames+1 if visible else 0
                    pending_observer=getattr(stone,'popup_return_pending',None) if stone is not None else None
                    native_pending=bool(return_ready and callable(pending_observer)
                                        and pending_observer(frame,notice_gone=notice_gone))
                    if (native_pending and not self.stop_event.is_set()
                            and (st.stage_deadline is None
                                 or _time().monotonic() < st.stage_deadline)):
                        # 原生任务的身份证明优先于“界面回到主界面”这种通用证据:
                        # 弹窗关掉之后两者会同时成立,通用规则一确认就把 stage 放回
                        # main,stone_farm 再抛竿就再也补不上三秒灰色鱼钩证明。
                        # 只在它确实在等这份证明时挂住本阶段,截止时间到了照常往下走。
                        if runtime.wait(self.config['poll_interval']):
                            break
                        continue
                    if not (panel_return or native_task_return or restaurant_return or cruise_return or visible) and not self.stop_event.is_set():
                        # 红叉有时被弹窗自身的滑入动画吃掉:切到 task_popup_return
                        # 之后 stage 过滤只保留本阶段规则,那条 stage="*" 的中断
                        # 规则不再参与扫描,继续傻等只能撞上 15 秒期限停机。
                        # 只有所有返回路径都不成立时才重按,避免抢在正常恢复之前。
                        # 重按前必须重新凑够 stable_frames:红叉已经证明过是稳的,
                        # 但两次点击之间仍要重新取证,避免"没看清就再点一下"。
                        frames_required=(int(number(close_rule[0].get('stable_frames',3),1,99))
                                         if close_rule is not None else 3)
                        decision=('wait' if notice_gone or st.popup_reclick_frames<frames_required
                                  else popup_reclick_decision(True,return_ready,st.popup_reclick))
                        if decision=='give_up':
                            raise RuntimeError('任务弹窗：红叉连续三次未关掉，已停止；请在单项验证时确认画面')
                        if (decision=='reclick' and close_rule is not None
                                and module.appear_then_click(close_rule[1],offset=close_rule[2],
                                    interval=close_rule[0]['interval'],similarity=close_rule[0]['threshold'])):
                            st.popup_reclick+=1
                            st.popup_reclick_frames=0
                            now=_time().monotonic()
                            delay=max(2.0,float(close_rule[0]['interval']))
                            st.stage_started_at=now
                            st.stage_ready_at=now+delay
                            st.stage_deadline=now+max(delay+2.0,number(close_rule[0].get('stage_timeout',15),1,3600))
                            st.stable={key:0 for key in st.stable}
                            with self.lock:
                                self.state['clicks']+=1
                            self.event(f'任务弹窗：红叉没有关掉弹窗，{delay:.0f} 秒后重按（第 {st.popup_reclick} 次）')
                            if runtime.wait(self.config['poll_interval']):
                                break
                            continue
                    if panel_return or native_task_return or restaurant_return or cruise_return or visible and st.popup_map_frames >= return_rule[0]['stable_frames']:
                        # An NPC notice can be opened over a navigation map.
                        # Confirm the map without clicking or completing a rank.
                        st.stage,st.stage_deadline,st.stage_trigger='main',None,None
                        st.stage_started_at=st.stage_ready_at=_time().monotonic()
                        st.handling_interrupt=False;st.popup_map_frames=0
                        st.popup_reclick=0;st.popup_reclick_frames=0
                        st.stable={key:0 for key in st.stable}
                        st.number_values.clear();st.target_keys.clear()
                        with self.lock:
                            self.state['stage']='main'
                            self.state['matches']+=1
                        self.event('任务弹窗已关闭，连续三帧三秒确认原航巡设置；交还当前控制器，保留原截止与次数，不记配置完成'
                                   if cruise_return else '任务弹窗已关闭，已连续确认原餐品图鉴或食材页；按保存进度继续，不重复已用输入'
                                   if restaurant_return else '任务弹窗已关闭，已连续三帧三秒确认原鱼饼强化面板；接续未完成强化，不追加弹窗输入'
                                   if panel_return else '任务弹窗已关闭，已连续三帧三秒确认当前任务停航和灰色鱼钩；保留唤醒次数继续单次抛竿'
                                   if native_task_return else '任务弹窗已关闭，已连续确认当前平行世界地图；保留导航进度继续')
                        continue
                else:
                    st.popup_map_frames=0
                    # 离开中断阶段:重按计数必须归零,否则下次弹窗会带着旧次数直接 give_up。
                    st.popup_reclick=0;st.popup_reclick_frames=0
                if (return_parallel is not None and return_parallel_enabled()
                        and st.stage == 'main' and interruption is None
                        and not any(task is not None and task.active for task in
                                    (equipment, backpack, skills, calibrator, stone, pass_task, guild_daily, restaurant))
                        and not getattr(setup, 'busy', False)
                        and (parallel is None or parallel.phase in ('detect', 'play'))):
                    owned = runtime.call('return_parallel', return_parallel.step, frame)
                    with self.lock:
                        self.state['return_parallel'] = return_parallel.status()
                        if owned:self.state['stage'] = 'return_parallel_' + return_parallel.phase
                    if owned:
                        observe_meter_candidate('task_owned', detail='return_parallel')
                        if runtime.wait(self.config['poll_interval']):break
                        continue
                if (not preview and restaurant is not None and restaurant.active
                        and st.stage=='main' and interruption is None):
                    step_restaurant(frame,allow_start=False)
                    observe_meter_candidate('task_owned',detail='restaurant_upgrade')
                    if runtime.wait(self.config['poll_interval']):break
                    continue
                if (not preview and guild_daily is not None and guild_daily.active
                        and st.stage=='main' and interruption is None):
                    step_guild_daily(frame,allow_start=False)
                    observe_meter_candidate('task_owned',detail='guild_daily')
                    if runtime.wait(self.config['poll_interval']):break
                    continue
                if not preview:
                    if (st.stage=='main' and interruption is None and guild_daily is not None
                            and getattr(guild_daily,'read_only_return_pending',False) is True):
                        # Observe the native settings page even when cruise owns
                        # it. allow_start=False keeps expired guild inputs retired.
                        step_guild_daily(frame,allow_start=False)
                    if cruise is None and st.stage == 'main':
                        cruise = make_cruise_watchdog()
                    # 开局航巡配置：先把珍珠/琥珀/蓝色宝石/BOSS 与自动售卖、自动强化配好，
                    # 配好之后才轮到航巡守卫去"打开并开始"，两者不能抢菜单。
                    setup_owned = False
                    sailing_needed = (configured_cruise() is not None or stone is not None
                                      or (calibrator is not None and calibrator.enabled()))
                    if (st.stage == 'main' and interruption is None
                            and sailing_needed and not any(cruise_blockers(include_setup=False).values())):
                        if setup is None:
                            setup = make_cruise_setup()
                        if setup is not None and not setup.done:
                            if getattr(cruise, 'suspend', None) is not None:
                                cruise.suspend()
                            setup_owned = runtime.call('cruise_setup', setup.step, frame)
                            with self.lock:
                                self.state['cruise_setup'] = setup.status()
                    if setup_owned or bool(getattr(setup,'busy',False)) and interruption is None and st.stage=='main':
                        if setup_owned:
                            with self.lock:self.state['matches'] += 1
                        if runtime.wait(self.config['poll_interval']):
                            break
                        continue
                    gates = dict(interruption=interruption is not None, **cruise_blockers())
                    cruise_owned = False
                    if cruise is not None:
                        if any(gates.values()):
                            cruise.suspend()
                        else:
                            cruise_owned = runtime.call('cruise_watchdog', cruise.step, frame)
                        with self.lock:
                            self.state['cruise'] = cruise.status()
                            self.state['cruise_gate'] = gates
                    if cruise_owned:
                        observe_meter_candidate('cruise_watchdog')
                        # Waiting on this menu also owns the UI; no daily
                        # entry or upgrade may insert a second input here.
                        if runtime.wait(self.config['poll_interval']):
                            break
                        continue
                if (not preview and interruption is None and st.stage == 'raid_battle'
                        and raid is not None and raid.step(frame)):
                    # 甩杆输入独占这一帧,避免和 raid_confirm 的观察规则抢。
                    with self.lock:
                        self.state['matches'] += 1
                        self.state['raid_cast'] = raid.status()
                    if runtime.wait(self.config['poll_interval']):
                        break
                    continue
                if not preview and interruption is None and st.stage == 'main':
                    bonus_entry=st.matchers.get(BONUS_ID)
                    bonus_visual=getattr(bonus_entry[1],'daily_bonus_visual',None) if bonus_entry else None
                    bonus_ready=(bonus_visual is not None and bonus_entry[0]['enabled']
                                 and bonus_visual.state(frame)=='ready')
                    daily_status=self.daily_schedule.status()['daily_reward']
                    check_bonus=(bonus_ready or bonus_visual is not None
                                 and bonus_visual.state(frame)=='gray'
                                 and daily_status.get('free_completed') and daily_status.get('bonus_pending'))
                    context=self.daily_cleanup_context(frame,st.matchers,allow_ready=bonus_ready)
                    st.daily_orphan_frames=(st.daily_orphan_frames+1
                        if context is not None and context==st.daily_orphan_key
                        else 1 if context is not None else 0)
                    st.daily_orphan_key=context
                    if context is not None:
                        if st.daily_orphan_frames>=3:
                            if check_bonus:
                                now=_time().monotonic()
                                st.daily_attempt=('daily_reward',current_period)
                                st.daily_verified=False
                                st.stage,st.stage_trigger='daily_bonus','每日奖励：继续领取可用附赠奖励'
                                st.stage_started_at=st.stage_ready_at=now
                                st.stage_deadline=now+20;st.skip_on_timeout=True
                                st.stable={key:0 for key in st.stable}
                                if st.scheduled:
                                    interval=st.matchers['driftinggirl_daily_open'][0]['interval']
                                    self.task_schedule.begin('daily_reward',interval,self.daily_schedule.status()['daily_reward'])
                                    st.scheduled_daily=('daily_reward',interval,current_period)
                                with self.lock:self.state['stage']=st.stage
                                self.event('每日奖励：免费已领，连续新图确认附赠奖励可领，继续右侧奖励'
                                           if bonus_ready else '每日奖励：免费已领，附赠奖励为灰色，跳过并核对返回')
                            else:
                                begin_daily_cleanup()
                                self.event('每日奖励：连续新图确认已领取商店仍展开，先安全关闭；不补记领奖完成')
                        if runtime.wait(self.config['poll_interval']):
                            break
                        continue
                if (not preview and st.daily_cleanup_deadline is not None
                        and st.stage!='main' and _time().monotonic()>=st.daily_cleanup_deadline):
                    # 清算窗口用完仍认不出画面:本次作废,但不算代码故障(不写错误截图,
                    # 面板标"可重试"),由调度稍后再试。
                    raise RetryableStage(
                        '每日奖励清算在 40 秒总窗口内未确认主画面，本次不记完成；保留记录稍后重试')
                if not preview and interruption is None and st.stage == 'main' and not mailbox_cleanup_used:
                    mailbox_context = MailboxCloseGuard.context(frame, st.matchers)
                    if mailbox_context is not None:
                        if mailbox_context != mailbox_cleanup_key:
                            mailbox_cleanup_key = mailbox_context
                            mailbox_cleanup_frames = 0
                            mailbox_cleanup_since = _time().monotonic()
                        mailbox_cleanup_frames += 1
                        if mailbox_cleanup_frames >= 3 and _time().monotonic()-mailbox_cleanup_since >= 3:
                            # Close an already-open, fully identified empty/limit page.
                            # Never repeat claim_all or renew the expired claim.
                            now = _time().monotonic()
                            mailbox_cleanup_used = True
                            mailbox_cleanup_deadline = now+20
                            st.stage,st.stage_trigger = 'mailbox_done','邮箱：安全关闭遗留窗口'
                            st.stage_started_at,st.stage_ready_at,st.stage_deadline = now,now,now+20
                            st.daily_attempt,st.daily_verified = None,False
                            st.stable = {key:0 for key in st.stable}
                            with self.lock:self.state['stage'] = st.stage
                            self.event('邮箱：连续确认已有领空或超上限窗口，只关闭并核对返回，不重复领取')
                        if runtime.wait(self.config['poll_interval']):break
                        continue
                    mailbox_cleanup_frames,mailbox_cleanup_since = 0,None
                    mailbox_cleanup_key = None
                if not preview and interruption is None and st.stage == 'main' and not guild_orphan_used:
                    context = GuildSelectionGuard.list_context(frame, st.matchers)
                    now = _time().monotonic()
                    if context is not None:
                        if context != guild_orphan_key:
                            guild_orphan_key, guild_orphan_since, guild_orphan_frames = context, now, 0
                        guild_orphan_frames += 1
                        if guild_orphan_frames >= 3 and now-guild_orphan_since >= 3:
                            # A new close-only operation, not a renewed purchase.
                            # The old failed run and its quantities remain untouched.
                            guild_orphan_used = True
                            guild_cleanup_deadline = now+20
                            guild_cleanup_origin, guild_cleanup_rule = context, None
                            st.stage, st.stage_trigger = 'guild_close', '公会商品：安全关闭遗留列表'
                            st.stage_started_at = st.stage_ready_at = now
                            st.stage_deadline = guild_cleanup_deadline
                            st.skip_on_timeout = False
                            st.daily_attempt, st.daily_verified = None, False
                            st.stable = {key:0 for key in st.stable}
                            st.number_values.clear(); st.target_keys.clear()
                            with self.lock:self.state['stage'] = st.stage
                            self.event('公会商品：连续三帧三秒确认遗留列表，只关闭并核对返回，不重复选择或购买')
                        if runtime.wait(self.config['poll_interval']):break
                        continue
                    guild_orphan_key, guild_orphan_since, guild_orphan_frames = None, None, 0
                if not preview and interruption is None and st.stage == 'main':
                    owned = False
                    equipment_checked = False
                    previous_cycles = equipment.cycles if equipment else 0
                    previous_bag_cycles = backpack.cycles if backpack else 0
                    previous_skill_cycles = skills.cycles if skills else 0
                    previous_meter_cycles = calibrator.cycles if calibrator else 0
                    previous_meter_active = calibrator.active if calibrator else False
                    skills_owned = False
                    meter_owned = False
                    stone_owned = False
                    pass_owned = False
                    restaurant_owned = False
                    if pass_task is not None and (pass_task.active or pass_task.panel(frame)):
                        pass_owned=step_pass(frame)
                        owned=pass_owned
                    if not owned and backpack is not None and backpack.active:
                        owned = runtime.call('backpack',backpack.step,frame,parallel.rank if parallel else None)
                    if (not owned and equipment is not None
                            and (equipment.active or equipment.panel(frame))):
                        if (parallel is not None and parallel.boss_sailing_stop_requested()
                                and equipment.validate_battle_close(frame)):
                            close_boss_sailing_equipment()
                            owned = True
                        else:
                            owned = step_scheduled_controller('equipment_upgrade', 'equipment', equipment, frame)
                        equipment_checked = True
                    if (not owned and skills is not None
                            and (skills.active or skills.panel(frame) or skills.modal(frame)
                                 or skills.owns_purchase_prompt() and skills.purchase_prompt(frame))):
                        skills_owned = step_scheduled_controller('parallel_skills', 'skills', skills, frame)
                        owned = skills_owned
                    if not owned and calibrator is not None and calibrator.active:
                        meter_owned = runtime.call('meter_calibration', calibrator.step, frame)
                        owned = meter_owned
                    if not owned and stone is not None:
                        stone_owned = call_stone('stone_farm', stone.step, frame)
                        owned = stone_owned
                    if not owned and guild_daily is not None:
                        owned=step_guild_daily(frame,allow_start=(parallel is None or parallel.phase=='play'))
                    if not owned and restaurant is not None:
                        restaurant_owned=step_restaurant(frame,allow_start=(parallel is None or parallel.phase=='play'))
                        owned=restaurant_owned
                    if owned and not meter_owned:
                        observe_meter_candidate('task_owned', detail=(
                            'pass_daily' if pass_owned else 'stone_farm' if stone_owned else
                            'parallel_skills' if skills_owned else
                            'equipment_upgrade' if equipment_checked else 'backpack'))
                    elif parallel is not None and calibrator is not None and not calibrator.active:
                        observe_meter_candidate('legacy_parallel')
                    if (not owned and not meter_owned and calibrator is not None
                            and (parallel is None or calibrator.active)):
                        meter_owned = runtime.call('meter_calibration', calibrator.step, frame)
                        owned = meter_owned
                    if (not owned and backpack is not None and
                            not (parallel and parallel.training.state.get('sailing_sleep_recovery')) and
                            (parallel is None or parallel.phase == 'play' and parallel.rank is not None)):
                        owned = runtime.call('backpack',backpack.step,frame,parallel.rank if parallel else None)
                    parallel_owned = not owned and parallel is not None and runtime.call('parallel',parallel.step,frame)
                    if (parallel is not None and getattr(parallel,'only_sailing_shutdown',False)
                            and parallel.shutdown_complete):
                        parallel = None
                        with self.lock:
                            self.state['parallel'] = None
                            self.state['stage'] = 'main'
                        if runtime.wait(self.config['poll_interval']):
                            break
                        continue
                    owned = owned or parallel_owned
                    if not owned and pass_task is not None:
                        pass_owned=step_pass(frame)
                        owned=pass_owned
                    if not owned and skills is not None:
                        skills_owned = step_scheduled_controller('parallel_skills', 'skills', skills, frame)
                        owned = skills_owned
                    if not owned and equipment is not None and not equipment_checked:
                        owned = step_scheduled_controller('equipment_upgrade', 'equipment', equipment, frame)
                    with self.lock:
                        self.state['parallel'] = parallel.status() if parallel else None
                        self.state['equipment'] = equipment.status() if equipment else None
                        self.state['backpack'] = backpack.status() if backpack else None
                        self.state['skills'] = skills.status() if skills else None
                        self.state['meter'] = calibrator.status() if calibrator else None
                        self.state['stone'] = stone.status() if stone else None
                        self.state['pass_daily'] = pass_task.status() if pass_task else None
                        self.state['stage'] = ('parallel_boss_defense' if defense_owned else
                                               'stone_' + stone.phase if stone_owned else
                                               'meter_' + calibrator.phase if meter_owned else
                                               'pass_daily_' + pass_task.phase if pass_owned and pass_task.active else
                                               'restaurant_' + restaurant.phase if restaurant_owned else
                                               'backpack_' + backpack.phase if owned and backpack is not None and (backpack.active or backpack.cycles > previous_bag_cycles) else
                                               'parallel_' + parallel.phase if parallel_owned else
                                               'skills_' + skills.phase if skills_owned else
                                               'equipment_' + equipment.phase if owned and equipment and not pass_owned else 'main')
                    if stop_on_main and pass_task is not None and pass_task.done is not None:
                        with self.lock:self.state['stop_reason']='cycle_complete' if pass_task.done else 'observation_complete'
                        self.stop_event.set()
                        break
                    if (stop_on_main and calibrator is not None
                            and (calibrator.cycles > previous_meter_cycles
                                 or previous_meter_active and not calibrator.active)):
                        completed = calibrator.cycles > previous_meter_cycles
                        with self.lock:
                            self.state['stop_reason'] = 'cycle_complete' if completed else 'observation_complete'
                        self.event('标尺校准已完成并恢复原航巡状态，单项测试停止' if completed
                                   else '旧标尺校准记录已只读核对结束，单项测试停止')
                        self.stop_event.set()
                        break
                    if stop_on_main and skills is not None and skills.cycles > previous_skill_cycles:
                        with self.lock:
                            self.state['stop_reason'] = 'cycle_complete'
                        self.event('平行世界技能已返回主画面，单项测试停止')
                        self.stop_event.set()
                        break
                    if stop_on_main and backpack is not None and backpack.cycles > previous_bag_cycles:
                        with self.lock:
                            self.state['stop_reason'] = 'cycle_complete'
                        self.event('背包出售已返回原地点，单项测试停止')
                        self.stop_event.set()
                        break
                    if stop_on_main and equipment is not None and equipment.cycles > previous_cycles:
                        with self.lock:
                            self.state['stop_reason'] = 'cycle_complete'
                        self.event('武器强化已返回主画面，单项测试停止')
                        self.stop_event.set()
                        break
                    if owned:
                        if max_clicks is not None and self.state['clicks'] >= max_clicks:
                            with self.lock:
                                self.state['stop_reason'] = 'click_limit'
                            self.stop_event.set()
                            break
                        returning = ((parallel is not None
                            and parallel.phase in ('return_start','return_confirm','await_original'))
                            or stone is not None and stone.phase in ('return','return_confirm'))
                        if runtime.wait(min(self.config['poll_interval'],.02)
                                                if returning else self.config['poll_interval']):
                            break
                        continue
                guild_missed = False
                if (not preview and interruption is None and not st.handling_interrupt
                        and guild_cleanup_deadline is None):
                    guild_missed = guild_selection.observe(frame, st.matchers, self.purchase_condition,
                            stage=st.stage, trigger=st.stage_trigger, started=st.stage_started_at,
                            deadline=st.stage_deadline, now=_time().monotonic(),
                            stopped=self.stop_event.is_set())
                    with self.lock:self.state['guild_selection'] = guild_selection.status()
                if guild_missed:
                    # Close an unchanged list without retrying selection or purchase.
                    # Both close and return share the original, unextended deadline.
                    guild_cleanup_deadline = st.stage_deadline
                    guild_cleanup_origin = guild_selection.origin
                    guild_cleanup_rule = guild_selection.rule_id
                    guild_selection.clear()
                    finish_scheduled_daily(False)
                    st.daily_attempt, st.daily_verified = None, False
                    st.stage = 'guild_close'
                    st.stage_ready_at = _time().monotonic()
                    st.stable = {key: 0 for key in st.stable}
                    st.number_values.clear(); st.target_keys.clear()
                    with self.lock: self.state['stage'] = st.stage
                    self.event('公会商品：选择后仍是同一商品列表，连续三秒确认库存未变；不重复购买，按原截止关闭并延后重试')
                stage_expired = (interruption is None and st.stage_deadline is not None
                                 and _time().monotonic() >= st.stage_deadline)
                if not preview and interruption is None and _time().monotonic() < st.stage_ready_at:
                    if runtime.wait(max(0, min(self.config["poll_interval"], st.stage_ready_at - _time().monotonic()))):
                        break
                    continue
                scan_candidates = optional_cleanup.rules(rules) if optional_cleanup is not None else rules
                if not preview and potion_cleanup is not None and potion_cleanup.attempts >= potion_cleanup.LIMIT:
                    scan_candidates = [item for item in scan_candidates if item[0]['id'] != potion_cleanup.ENTRY]
                if not preview and event_potion_skips >= 3:
                    scan_candidates = [item for item in scan_candidates if item[0]['id'] != 'driftinggirl_event_potion_open']
                candidate, grace_visible, previous_visible = runtime.call('rule_scan',scan_rules,
                    frame, scan_candidates, module, st.matchers, load_rule=self.load_rule, preview=preview,
                    interruption=interruption, stage=st.stage, stage_expired=stage_expired,
                    stage_deadline=st.stage_deadline, stage_started_at=st.stage_started_at,
                    stage_trigger=st.stage_trigger, stable=st.stable, number_values=st.number_values,
                    target_keys=st.target_keys, previous=st.previous, scheduled=st.scheduled,
                    daily_schedule=self.daily_schedule, task_schedule=self.task_schedule,
                    config=self.config, state=self.state, lock=self.lock,
                    number_condition=self.number_condition, counter_condition=self.counter_condition,
                    purchase_condition=self.purchase_condition,
                    daily_entries=DAILY_ENTRIES, clock=_time)
                if optional_cleanup is not None and (candidate is None or interruption is not None):
                    optional_cleanup.reset_input_proof()
                if (not preview and st.stage == 'mailbox_return'
                        and MailboxCloseGuard.page_visible(frame, st.matchers)):
                    # Missing modal parts reject an input, but cannot prove
                    # the window closed. Preserve the original text-only guard.
                    if candidate is not None and candidate[0]['next_stage'] == 'main':
                        candidate = None
                    grace_visible = False
                    for key in ('driftinggirl_mailbox_return','driftinggirl_mailbox_parallel_return'):
                        if key in st.stable:st.stable[key] = 0
                mailbox_retry = False
                if (not preview and optional_cleanup is None and candidate is None and mailbox_close_guard.ready(
                        frame, st, now=_time().monotonic(), stopped=self.stop_event.is_set(),
                        interrupted=interruption is not None)):
                    candidate = st.matchers[mailbox_close_guard.origin[0]]
                    mailbox_retry = True
                if stage_expired and not grace_visible:
                    if candidate is None and st.skip_on_timeout:
                        if st.stage.startswith('daily_'):
                            self.event(f'「{st.stage_trigger}」超时，跳过本次收取；先安全关闭并确认返回，不记领奖完成')
                            begin_daily_cleanup('daily_return' if self.daily_return_visible(frame,st.matchers)
                                                else 'daily_after_claim')
                            continue
                        # The graph is diagnostic only. Every input still goes
                        # through scan_rules and the new-frame close guard.
                        if nav_requested(rules, st.stage):
                            step = navigation().step()
                            self.event(f'页面图退回建议：{step.kind} {step.button or ""}；仍由安全关闭流程逐帧核对')
                        optional_cleanup = OptionalCleanup(st.stage, _time().monotonic(),
                            original_deadline=st.stage_deadline, original_trigger=st.stage_trigger)
                        actions = optional_cleanup.prepare(rules, self.load_rule)
                        if optional_cleanup.family == 'event_potion':
                            event_potion_skips += 1
                            if event_potion_skips == 3:
                                self.event('活动药水：本轮已跳过 3 次，本轮不再扫描商店入口，其他任务继续')
                        st.matchers.update({r['id']: (r,b,o) for r,b,o in actions})
                        self.event(f'「{st.stage_trigger}」超时，跳过本次；40秒内仅安全关闭，不再领取或重开；原截止 {st.stage_deadline}')
                        finish_scheduled_daily(False)
                        st.daily_attempt, st.daily_verified = None, False
                        st.stage, st.stage_deadline = optional_cleanup.stage, optional_cleanup.deadline
                        st.stage_trigger = '可跳过任务：安全退出未完成的收奖'
                        st.skip_on_timeout = False
                        st.stage_started_at = _time().monotonic()
                        st.stage_ready_at = 0
                        st.stable = {key: 0 for key in st.stable}
                        st.stable.update({r['id']: 0 for r,_,_ in actions})
                        st.number_values.clear(); st.target_keys.clear()
                        with self.lock:
                            self.state['stage'] = st.stage
                            self.state['optional_cleanup'] = optional_cleanup.status()
                        continue
                    if (candidate is None and st.daily_cleanup_deadline is not None
                            and st.stage.startswith('daily_')):
                        # 每日清算原先过了 20 秒阶段线就停整轮;改成先吃满 40 秒总窗口,
                        # 仍然认不出才作废本次(可重试),不写成代码故障。
                        now = _time().monotonic()
                        if st.stage_deadline is not None and now + 1 < st.daily_cleanup_deadline:
                            st.stage_deadline = min(st.daily_cleanup_deadline, now + 20)
                            st.stable = {key: 0 for key in st.stable}
                            self.event('每日奖励：清算窗口内还未认出画面，延长到总截止再试，不追加输入')
                            if runtime.wait(self.config['poll_interval']):
                                break
                            continue
                        raise RetryableStage(
                            '每日奖励清算在总窗口内未确认主画面，本次不记完成；保留记录稍后重试')
                    if candidate is None:
                        raise RuntimeError(f"执行「{st.stage_trigger}」后未识别到后续界面，已停止；请在单项验证时确认画面")
                # A disappearing/reappearing housekeeping icon is a new episode,
                # not another failed attempt at the same unchanged screen.
                st.absent_frames = 0 if previous_visible else st.absent_frames + 1
                if st.absent_frames >= 2:
                    st.previous = None
                    st.previous_progress = None
                    st.consecutive = 0
                if candidate and not self.stop_event.is_set():
                    rule, button, offset = candidate
                    cleanup_click = optional_cleanup is not None and bool(rule.get('cleanup_origin'))
                    if (cleanup_click and not optional_cleanup.ready(
                            rule['id'], button, frame, _time().monotonic())):
                        if runtime.wait(self.config['poll_interval']):
                            break
                        continue
                    bonus_entry=st.matchers.get(BONUS_ID)
                    bonus_visual=getattr(bonus_entry[1],'daily_bonus_visual',None) if bonus_entry else None
                    if (not preview and bonus_visual is not None
                            and rule['id'] in (BONUS_ID,'driftinggirl_daily_close')):
                        availability=bonus_visual.state(frame)
                        # A live claim is issued once, then the next stage must
                        # see gray before closing/recording a completed check.
                        if (rule['id']=='driftinggirl_daily_close'
                                and st.daily_cleanup_deadline is None and availability!='gray'):
                            if runtime.wait(self.config['poll_interval']):break
                            continue
                        context=self.daily_cleanup_context(frame,st.matchers,allow_ready=True)
                        frame=runtime.capture();self.remember(frame)
                        with self.lock:self.state['frames']+=1
                        if (self.stop_event.is_set() or context is None
                                or _time().monotonic()>=st.stage_deadline
                                or self.daily_cleanup_context(frame,st.matchers,allow_ready=True)!=context
                                or bonus_visual.state(frame)!=availability
                                or not button.match(frame,offset=offset,similarity=rule['threshold'])):
                            st.stable={key:0 for key in st.stable}
                            continue
                        if rule['id']==BONUS_ID and availability=='gray':
                            rule={**rule,'action':'observe','name':'每日奖励：附赠奖励为灰色，跳过点击'}
                    recovery_entry = rule.get('recovery_entry') if not preview else None
                    retry_limit = rule.get('retry_limit') if not preview else None
                    if recovery_entry:
                        history = st.entry_recoveries.setdefault(recovery_entry, deque())
                        now = _time().monotonic()
                        while history and now - history[0] >= 300:
                            history.popleft()
                        if len(history) >= 3:
                            raise RuntimeError(f"「{st.matchers[recovery_entry][0]['name']}」五分钟内已恢复三次仍未打开，已停止")
                    if retry_limit:
                        # 原先这里借用扫描循环的 key(循环变量泄漏),现在显式取候选规则 id
                        history = st.entry_recoveries.setdefault(rule["id"], deque())
                        now = _time().monotonic()
                        while history and now - history[0] >= 300:
                            history.popleft()
                        if len(history) >= retry_limit:
                            raise RuntimeError(f"「{rule['name']}」五分钟内已重试 {retry_limit} 次，已停止")
                    if not preview and rule.get('interrupt'):
                        now = _time().monotonic()
                        while st.interrupt_times and now - st.interrupt_times[0] >= 300:
                            st.interrupt_times.popleft()
                        if len(st.interrupt_times) >= 3:
                            raise RuntimeError('任务弹窗在五分钟内反复出现，已关闭三次并停止；请检查游戏提示')
                    progress_key = getattr(button, "progress_key", None)
                    unchanged = rule["id"] == st.previous and progress_key == st.previous_progress
                    if not preview and unchanged and st.consecutive >= self.config["max_consecutive_clicks"]:
                        raise RuntimeError("同一目标连续点击次数达到限制，已停止；请检查界面或调整规则")
                    ready_to_act = True
                    potion_entry = rule['id'] == PotionAdRecovery.ENTRY
                    sidebar_click = (getattr(button, 'sidebar_layout', False)
                                     and rule.get('action', 'click') == 'click')
                    guild_select = rule['id'] in GUILD_SELECTORS
                    guild_close = rule['id'] == 'driftinggirl_guild_close'
                    popup_close = rule['id'] == PopupRaidReturn.CLOSE
                    mailbox_close = (rule['id'] in MAILBOX_CLOSES
                                     and getattr(button, 'mailbox_close_guarded', False))
                    mailbox_context = None
                    mailbox_return_window = (st.stage_started_at, st.stage_deadline)
                    guild_context = None
                    if not preview and (potion_entry or cleanup_click or rule.get('recheck_before_click') or sidebar_click or guild_select or guild_close or popup_close or mailbox_close):
                        previous_box = tuple(button.button)
                        previous_cleanup = optional_cleanup.input_context(frame) if cleanup_click else None
                        previous_popup = PopupRaidReturn.notice_identity(frame) if popup_close else None
                        previous_guild_page = (GuildSelectionGuard.list_context(frame,st.matchers)
                                               if guild_close else None)
                        previous_mailbox = (MailboxCloseGuard.context(frame, st.matchers, rule['id'])
                                            if mailbox_close else None)
                        frame = runtime.capture()
                        self.remember(frame)
                        with self.lock:
                            self.state['frames'] += 1
                        visible = module.appear(button, offset=offset, similarity=rule['threshold'])
                        if visible and (sidebar_click or cleanup_click) and rule.get('absent_rule'):
                            guard_rule,guard_button,guard_offset=st.matchers[rule['absent_rule']]
                            module.appear(guard_button,offset=guard_offset,similarity=guard_rule['threshold'])
                            visible=guard_button.last_similarity<=guard_rule['threshold']
                        check = (self.purchase_condition if guild_select else
                                 self.counter_condition if rule.get('counter_condition') else self.number_condition)
                        passed, value = check(rule, button, frame) if visible else (False, None)
                        ready_to_act = (not self.stop_event.is_set() and passed
                                        and value == st.number_values.get(rule['id'])
                                        and tuple(button.button) == previous_box
                                        and (not (sidebar_click or guild_select or guild_close) or st.stage_deadline is None
                                             or _time().monotonic() < st.stage_deadline))
                        if ready_to_act and guild_select:
                            guild_context = GuildSelectionGuard.context(
                                frame, st.matchers, rule['id'], self.purchase_condition)
                            ready_to_act = (guild_context is not None
                                and guild_context[0] == value and tuple(button.button) == previous_box
                                and not self.stop_event.is_set()
                                and (st.stage_deadline is None or _time().monotonic() < st.stage_deadline))
                        if ready_to_act and guild_close:
                            ready_to_act = (previous_guild_page is not None
                                and GuildSelectionGuard.list_context(frame,st.matchers)==previous_guild_page
                                and not self.stop_event.is_set()
                                and (st.stage_deadline is None or _time().monotonic()<st.stage_deadline))
                        if ready_to_act and popup_close:
                            ready_to_act = (previous_popup is not None
                                and PopupRaidReturn.notice_identity(frame) == previous_popup
                                and st.interrupt_started_at is not None
                                and _time().monotonic()-st.interrupt_started_at < 15
                                and (st.run_deadline is None or _time().monotonic()<st.run_deadline)
                                and (max_clicks is None or self.state['clicks']<max_clicks)
                                and not self.stop_event.is_set())
                        if ready_to_act and mailbox_close:
                            mailbox_context = MailboxCloseGuard.context(frame, st.matchers, rule['id'])
                            ready_to_act = (previous_mailbox is not None and mailbox_context == previous_mailbox
                                and (not mailbox_retry or mailbox_context == mailbox_close_guard.origin
                                     and mailbox_close_guard.attempts == 1)
                                and (st.stage_deadline is None or _time().monotonic() < st.stage_deadline)
                                and (st.run_deadline is None or _time().monotonic() < st.run_deadline)
                                and (max_clicks is None or self.state['clicks'] < max_clicks)
                                and not self.stop_event.is_set())
                        if ready_to_act and cleanup_click:
                            ready_to_act = (previous_cleanup is not None
                                and optional_cleanup.input_context(frame) == previous_cleanup
                                and optional_cleanup.allowed(rule['id'])
                                and _time().monotonic() < optional_cleanup.deadline
                                and (st.run_deadline is None or _time().monotonic() < st.run_deadline)
                                and (max_clicks is None or self.state['clicks'] < max_clicks)
                                and not self.stop_event.is_set())
                        if not ready_to_act:
                            if cleanup_click:
                                optional_cleanup.reset_input_proof()
                            if mailbox_retry:
                                mailbox_close_guard.frames, mailbox_close_guard.since = 0, None
                            st.stable = {key: 0 for key in st.stable}
                            st.number_values.clear()
                            st.target_keys.clear()
                            with self.lock:
                                self.state['last_match'] = None
                            self.event('入口位置或触发条件已变化，取消本次点击并重新识别')
                    if not ready_to_act:
                        acted = False
                    elif preview:
                        from alas_core.timer import Timer
                        module.interval_timer[rule["id"]] = Timer(rule["interval"]).start()
                        acted = True
                    elif rule.get("action", "click") == "observe":
                        acted = module.appear(button, offset=offset, interval=rule["interval"],
                                              similarity=rule["threshold"])
                    else:
                        if guild_cleanup_deadline is not None and rule['id']=='driftinggirl_guild_close':
                            frame=runtime.capture();self.remember(frame)
                            with self.lock:self.state['frames']+=1
                            fresh_context = (GuildSelectionGuard.context(frame,st.matchers,guild_cleanup_rule,
                                                                          self.purchase_condition)
                                             if guild_cleanup_rule is not None else
                                             GuildSelectionGuard.list_context(frame,st.matchers))
                            if (self.stop_event.is_set() or _time().monotonic()>=guild_cleanup_deadline
                                    or fresh_context!=guild_cleanup_origin):
                                st.stable={key:0 for key in st.stable}
                                continue
                        if st.daily_cleanup_deadline is not None and rule['id']=='driftinggirl_daily_close':
                            previous_context=self.daily_cleanup_context(frame,st.matchers,allow_ready=True)
                            frame=runtime.capture();self.remember(frame)
                            with self.lock:
                                self.state['frames']+=1
                            if (self.stop_event.is_set() or _time().monotonic()>=st.stage_deadline
                                    or _time().monotonic()>=st.daily_cleanup_deadline
                                    or previous_context is None
                                    or self.daily_cleanup_context(frame,st.matchers,allow_ready=True)!=previous_context):
                                st.stable={key:0 for key in st.stable}
                                continue
                        if st.scheduled and rule['id'] in DAILY_ENTRIES:
                            module.interval_timer.pop(rule['id'], None)
                        bonus_before = (frame.copy() if rule['id']==BONUS_ID
                                        and bonus_visual is not None else None)
                        bonus_frame_id = self.state['frame_id'] if bonus_before is not None else None
                        guild_before = frame.copy() if guild_select or guild_close else None
                        guild_frame_id = self.state['frame_id'] if guild_select or guild_close else None
                        popup_before = frame.copy() if popup_close else None
                        popup_frame_id = self.state['frame_id'] if popup_close else None
                        mailbox_before = frame.copy() if mailbox_close else None
                        mailbox_frame_id = self.state['frame_id'] if mailbox_close else None
                        cleanup_before = frame.copy() if cleanup_click else None
                        cleanup_frame_id = self.state['frame_id'] if cleanup_click else None
                        def issue_candidate():
                            nonlocal frame
                            if cleanup_click and (self.stop_event.is_set()
                                    or _time().monotonic() >= optional_cleanup.deadline
                                    or not optional_cleanup.allowed(rule['id'])
                                    or st.run_deadline is not None and _time().monotonic() >= st.run_deadline
                                    or max_clicks is not None and self.state['clicks'] >= max_clicks):
                                return False
                            if potion_entry:
                                # Reacquire identity at dispatch under the same
                                # input lock; a scan's cached location is not a tap.
                                with getattr(self.device, 'io_lock', nullcontext()):
                                    previous_box = tuple(button.button)
                                    frame = runtime.capture()
                                    self.remember(frame)
                                    with self.lock:self.state['frames'] += 1
                                    visible = button.match(frame, offset=offset, similarity=rule['threshold'])
                                    passed, value = self.counter_condition(rule, button, frame) if visible else (False, None)
                                    if (not passed or value != st.number_values.get(rule['id'])
                                            or tuple(button.button) != previous_box or self.stop_event.is_set()
                                            or st.run_deadline is not None and _time().monotonic() >= st.run_deadline
                                            or max_clicks is not None and self.state['clicks'] >= max_clicks):
                                        self.event('药水广告：触点前当前帧身份或计数复核未通过，取消本次点击')
                                        return False
                                    return module.appear_then_click(button, offset=offset,
                                        interval=rule['interval'], similarity=rule['threshold'])
                            return module.appear_then_click(button, offset=offset,
                                interval=rule['interval'], similarity=rule['threshold'])
                        try:
                            acted = runtime.call('rule_'+rule['id'],issue_candidate)
                        except Exception as exc:
                            if cleanup_click:
                                optional_cleanup.uncertain = True
                                with self.lock:self.state['optional_cleanup'] = optional_cleanup.status()
                                raise RuntimeError('安全关闭输入结果不确定，已停止并保留尝试状态，不追加点击') from exc
                            raise  # An uncertain issued input must stop, never be supplemented.
                        if acted and bonus_before is not None:
                            receipt={'input_id':uuid.uuid4().hex,'before_frame_id':bonus_frame_id,
                                     'rule_id':BONUS_ID,'xy':[(button.button[0]+button.button[2])//2,
                                                           (button.button[1]+button.button[3])//2],
                                     'before_state':'ready','issued_only':True}
                            after=None
                            try:
                                after=runtime.capture()
                                receipt['after_frame_id']=self.remember(after)
                                receipt['after_state']=bonus_visual.state(after)
                                with self.lock:self.state['frames']+=1
                            finally:
                                if self.daily_evidence is None:
                                    self.daily_evidence=_lookup('BossInputEvidence')(
                                        self.root,self.event,limit=16,folder='daily-bonus',label='附赠奖励')
                                self.daily_evidence.record(receipt,bonus_before,after)
                    if acted:
                        did_click = not preview and rule.get("action", "click") == "click"
                        if did_click and cleanup_click:
                            optional_cleanup.issued(rule['id'])
                            with self.lock:self.state['optional_cleanup'] = optional_cleanup.status()
                            receipt = dict(input_id=uuid.uuid4().hex, rule_id=rule['id'],
                                before_frame_id=cleanup_frame_id, issued_at=time.time(), issued_only=True,
                                cleanup=optional_cleanup.status(),
                                xy=[(button.button[0]+button.button[2])//2,(button.button[1]+button.button[3])//2])
                            after = None
                            try:
                                after = runtime.capture()
                                receipt['after_frame_id'] = self.remember(after)
                                with self.lock:self.state['frames'] += 1
                            finally:
                                if cleanup_evidence is None:
                                    cleanup_evidence = _lookup('BossInputEvidence')(
                                        self.root,self.event,limit=16,folder='optional-cleanup',label='可跳过任务退出')
                                cleanup_evidence.record(receipt,cleanup_before,after)
                        if did_click and mailbox_close:
                            mailbox_close_guard.issued(mailbox_context, retry=mailbox_retry)
                            if self.mailbox_evidence is None:
                                self.mailbox_evidence = _lookup('BossInputEvidence')(
                                    self.root,self.event,limit=16,folder='mailbox-input',label='邮箱关闭')
                            receipt = dict(input_id=uuid.uuid4().hex,rule_id=rule['id'],
                                before_frame_id=mailbox_frame_id,issued_at=time.time(),issued_only=True,
                                input_transport='adb-stationary',hold_ms=80,retry=mailbox_retry,
                                attempt=mailbox_close_guard.attempts,
                                original_return_deadline=mailbox_return_window[1] if mailbox_retry else None,
                                xy=[(button.button[0]+button.button[2])//2,(button.button[1]+button.button[3])//2])
                            mailbox_pending_evidence = receipt,mailbox_before
                            if mailbox_retry:
                                self.event('邮箱：连续三帧三秒确认关闭未生效，仅追加一次关闭，保留原返回截止时间')
                        if did_click and guild_select:
                            guild_selection.arm(rule['id'], guild_context)
                            with self.lock:self.state['guild_selection'] = guild_selection.status()
                        elif did_click:
                            guild_selection.clear()
                        if did_click and (guild_select or guild_close):
                            if self.guild_evidence is None:
                                self.guild_evidence = _lookup('BossInputEvidence')(
                                    self.root,self.event,limit=16,folder='guild-input',label='公会商品')
                            receipt = dict(input_id=uuid.uuid4().hex, rule_id=rule['id'],
                                before_frame_id=guild_frame_id, issued_at=time.time(), issued_only=True,
                                stock=list(guild_context[0]) if guild_context is not None else None,
                                input_transport='adb-stationary',hold_ms=80,
                                xy=[(button.button[0]+button.button[2])//2,(button.button[1]+button.button[3])//2])
                            guild_pending_evidence = receipt, guild_before
                        if did_click and popup_close:
                            popup_cruise.arm(st,frame,setup,stopped=self.stop_event.is_set())
                            if self.popup_evidence is None:
                                self.popup_evidence = _lookup('BossInputEvidence')(
                                    self.root,self.event,limit=16,folder='popup-input',label='任务弹窗')
                            receipt = dict(input_id=uuid.uuid4().hex,rule_id=rule['id'],
                                before_frame_id=popup_frame_id,issued_at=time.time(),issued_only=True,
                                input_transport='adb',hold_ms=0,
                                xy=[(button.button[0]+button.button[2])//2,(button.button[1]+button.button[3])//2])
                            popup_pending_evidence = receipt,popup_before
                        if retry_limit:
                            # 同上:显式用候选规则 id,不依赖扫描循环变量
                            st.entry_recoveries[rule["id"]].append(_time().monotonic())
                        if not preview and rule['id'] == 'driftinggirl_mailbox_limit_close':
                            from alas_core.timer import Timer
                            module.interval_timer['driftinggirl_mailbox_open'] = Timer(900).start()
                            self.event('邮箱物品超过获取上限，保留未领邮件，15 分钟后再检查')
                        if recovery_entry:
                            st.entry_recoveries[recovery_entry].append(_time().monotonic())
                            finish_scheduled_daily()
                            st.daily_attempt, st.daily_verified = None, False
                            with self.lock:
                                self.state['recoveries'] += 1
                            self.event('入口流程被打断，已确认当前画面；保留原间隔重新检测入口，本次不记为完成')
                        if not preview:
                            if rule.get('interrupt'):
                                finish_scheduled_daily()
                                st.daily_attempt, st.daily_verified = None, False
                            elif rule['id'] in DAILY_ENTRIES:
                                st.daily_attempt = (DAILY_ENTRIES[rule['id']], current_period)
                                st.daily_verified = False
                                if st.scheduled:
                                    feature = DAILY_ENTRIES[rule['id']]
                                    daily = {**self.daily_schedule.status()[feature], 'period': current_period}
                                    self.task_schedule.begin(feature, rule['interval'], daily)
                                    st.scheduled_daily = (feature, rule['interval'], current_period)
                            elif st.daily_attempt is not None:
                                feature, started_period = st.daily_attempt
                                if self.daily_completion_visible(feature, rule, frame, st.matchers):
                                    st.daily_verified = True
                                if rule['id'] == DAILY_RETURNS[feature]:
                                    if st.daily_verified:
                                        with self.lock:
                                            saved = self.daily_schedule.mark_complete(feature, started_period)
                                        if saved:
                                            self.event(f'{FEATURE_BY_ID[feature]["name"]}本周期已完成，等待北京时间下一次 08:00 刷新')
                                    elif feature == 'guild_shop':
                                        self.event('公会商品尚未确认全部买满，保留按原间隔重试')
                                    finish_scheduled_daily(self.daily_schedule.is_complete(feature))
                                    st.daily_attempt, st.daily_verified = None, False
                        if did_click and rule.get('interrupt'):
                            if rule['id'] == PopupRaidReturn.CLOSE:
                                popup_raid.arm(st, frame, now=_time().monotonic(),
                                               stopped=self.stop_event.is_set())
                            st.handling_interrupt = True
                            st.interrupt_started_at = None
                            st.interrupt_times.append(_time().monotonic())
                            with self.lock:
                                self.state['recoveries'] += 1
                        with self.lock:
                            self.state["matches"] += 1
                            self.state["clicks"] += int(did_click)
                        verb = "预览匹配" if preview else "点击" if did_click else "确认"
                        detail = (f"，第 {button.target_key + 1} 行，本帧可领 {button.green_count} 项"
                                  if hasattr(button, "green_count") else "")
                        if did_click:
                            x1, y1, x2, y2 = button.button
                            detail += f"，坐标 ({(x1+x2)//2}, {(y1+y2)//2})"
                        self.event(f"{verb}：{rule['name']}{detail}，相似度 {button.last_similarity:.3f}")
                        st.consecutive = st.consecutive + 1 if unchanged else 1
                        st.previous = rule["id"]
                        st.previous_progress = progress_key
                        st.absent_frames = 0
                        # A click can change the whole screen; require fresh confirmations.
                        st.stable = {key: 0 for key in st.stable}
                        st.number_values.clear()
                        st.target_keys.clear()
                        if not preview and rule.get("next_stage"):
                            recovered = st.handling_interrupt and rule['next_stage'] == 'main'
                            st.stage = rule["next_stage"]
                            st.stage_started_at = _time().monotonic()
                            st.stage_trigger = rule["name"]
                            st.stage_ready_at = _time().monotonic() + number(rule.get("next_delay", 0), 0, 60)
                            st.stage_deadline = (_time().monotonic() + number(rule.get("stage_timeout", 30), 1, 3600)
                                              if st.stage != "main" else None)
                            if mailbox_retry:
                                st.stage_started_at, st.stage_deadline = mailbox_return_window
                            if mailbox_cleanup_deadline is not None:
                                if st.stage == 'main':
                                    mailbox_cleanup_deadline = None
                                    self.event('邮箱：遗留窗口已关闭并确认返回，不补记旧领取完成')
                                else:
                                    st.stage_deadline = min(st.stage_deadline, mailbox_cleanup_deadline)
                            if guild_cleanup_deadline is not None:
                                if st.stage == 'main':
                                    guild_cleanup_deadline = guild_cleanup_origin = guild_cleanup_rule = None
                                    self.event('公会商品：未打开的购买步骤已安全退出，本次不记完成，保留已购商品和下次检查时间')
                                else:
                                    st.stage_deadline=min(st.stage_deadline,guild_cleanup_deadline)
                            if st.daily_cleanup_deadline is not None:
                                if st.stage=='main':
                                    st.daily_cleanup_deadline=None
                                    st.daily_orphan_key=None;st.daily_orphan_frames=0
                                    self.event('每日奖励：已确认窗口关闭并返回主画面，本次跳过不记领奖完成')
                                else:
                                    st.stage_deadline=min(st.stage_deadline,st.daily_cleanup_deadline)
                            if optional_cleanup is not None:
                                # Only observe_return may finish cleanup. A normal
                                # observe or interrupt transition cannot bypass it.
                                if not rule.get('interrupt'):
                                    st.stage = optional_cleanup.stage
                                st.stage_deadline = optional_cleanup.deadline
                            # 这个步骤的中途超时策略:true = 跳过回主界面(领不上就跳过)
                            st.skip_on_timeout = (bool(optional_timeout_policy(rule).get('skip_on_timeout')) and st.stage != 'main'
                                                  and st.daily_cleanup_deadline is None and optional_cleanup is None)
                            with self.lock:
                                self.state["stage"] = st.stage
                            if recovered:
                                st.handling_interrupt = False
                                self.event('任务弹窗已关闭，主界面已恢复，继续识别已开启任务')
                            if stop_on_main and st.stage == "main" and not recovery_entry and (not recovered or st.recovery_only):
                                with self.lock:
                                    self.state["stop_reason"] = "cycle_complete"
                                self.event("本项流程已返回主画面，测试停止")
                                self.stop_event.set()
                                break
                        if not preview and max_clicks is not None and self.state["clicks"] >= max_clicks:
                            with self.lock:
                                self.state["stop_reason"] = "click_limit"
                            self.event("本次测试达到点击上限，已停止")
                            self.stop_event.set()
                            break
                        if not preview and rule.get("stop_after", False):
                            self.event(str(rule.get("stop_message", "本次步骤已执行，运行已停止。")))
                            self.stop_event.set()
                            break
                if runtime.wait(self.config["poll_interval"]):
                    break
        except InterruptedError:
            # A task deadline can expire inside a fresh input guard. It is
            # not a user stop: preserve its failure and screenshot as when
            # step() observes the same expired journal on the next loop.
            if (not self.stop_event.is_set() and calibrator is not None
                    and calibrator.active and not calibrator.input_allowed()):
                record_failure(RuntimeError('标尺校准或航巡恢复窗口已到期，已停止并保留原记录'))
        except GameStuck as exc:
            # 同一对画面反复往返:保留现场并明确建议重启游戏,不要当成代码故障。
            record_failure(exc)
            with self.lock:
                self.state['stop_reason'] = 'game_stuck'
            self.event(f'画面没有进展（{exc}）；建议重启游戏后继续')
        except RetryableStage as exc:
            # Unowned/unknown/unsafe failures keep the original clean stop.
            # Safely released task failures are handled inside call_stone.
            with self.lock:
                self.state['stop_reason'] = 'retryable'
                self.state['last_error'] = str(exc)
            self.log.warning('Retryable stage stopped the run: %s', exc)
            self.event(f'运行已停止，本步骤未计完成；保留记录，需恢复后重试：{exc}')
        except Exception as exc:
            record_failure(exc)
        finally:
            observe_meter_candidate('worker_exit', finish=True)
            if st.scheduled:
                self.task_schedule.release()
            runtime.close()
            if cleanup_evidence is not None:
                cleanup_evidence.close()
            if self.boss_evidence is not None:
                self.boss_evidence.close()
                self.boss_evidence = None
            if self.fishing_evidence is not None:
                self.fishing_evidence.close()
                self.fishing_evidence = None
            if self.portal_evidence is not None:
                self.portal_evidence.close()
                self.portal_evidence = None
            if self.guild_daily_evidence is not None:
                self.guild_daily_evidence.close()
                self.guild_daily_evidence = None
            if self.restaurant_evidence is not None:
                self.restaurant_evidence.close()
                self.restaurant_evidence = None
            if self.sailing_evidence is not None:
                self.sailing_evidence.close()
                self.sailing_evidence = None
            if self.daily_evidence is not None:
                self.daily_evidence.close()
                self.daily_evidence = None
            if self.guild_evidence is not None:
                if guild_pending_evidence is not None:
                    receipt, before = guild_pending_evidence
                    self.guild_evidence.record(receipt, before)
                self.guild_evidence.close()
                self.guild_evidence = None
            if self.popup_evidence is not None:
                if popup_pending_evidence is not None:
                    receipt,before = popup_pending_evidence
                    self.popup_evidence.record(receipt,before)
                self.popup_evidence.close()
                self.popup_evidence = None
            if self.mailbox_evidence is not None:
                if mailbox_pending_evidence is not None:
                    receipt,before = mailbox_pending_evidence
                    self.mailbox_evidence.record(receipt,before)
                self.mailbox_evidence.close()
                self.mailbox_evidence = None
            if self.pass_evidence is not None:
                self.pass_evidence.close()
                self.pass_evidence=None
            with self.lock:
                self.state["mode"] = "idle"
            self.event("运行已停止")


