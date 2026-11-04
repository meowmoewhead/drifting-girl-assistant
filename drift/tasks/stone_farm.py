"""Native AUTO farming: lower one observed stage at RANK39 for the last boss.

Only the portal handoff owns the UI. The game's route, fishing and levelling
remain native. Closing a gate is a separate, persisted RANK40-only operation.
"""
import hashlib
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np

from drift.perception.digit_ocr import read_numeric_views
from drift.core.exceptions import RetryableStage
from drift.core.journal import read_record, archive_and_replace
from drift.tasks.boss_defense import BossDefense
from drift.tasks.fishing_meter import FishingMeter
from drift.tasks.meter_calibration import MeterCalibration, _ScreenContext
from drift.tasks.parallel_screen import ParallelScreen
from drift.tasks.parallel_world import ParallelWorld
from drift.tasks.training_sailing import TARGETS, TrainingSailing
from drift.tasks import native_auto_sale
from drift.tasks import handoff_journal


class StoneFarm(ParallelScreen):
    # These existing methods are read-only portal probes, not navigation steps.
    _portal_words = ParallelWorld._portal_words
    _portal_number = ParallelWorld._portal_number
    _gate_manager = ParallelWorld._gate_manager
    _lower_prompt = ParallelWorld._lower_prompt
    _npc_ready = ParallelWorld._npc_ready
    _gate_active = staticmethod(ParallelWorld._gate_active)
    completion_visible = ParallelWorld.completion_visible
    settings_key = MeterCalibration.settings_key
    _match = ParallelScreen.match

    START_STAGE, FINAL_STAGE, HANDOFF_RANK = 9, 8, 39
    TARGET_VALUES = (True, False, True, True, True)
    LIMIT = 600
    PHASES = {'idle', 'completion', 'stop', 'stop_fishing', 'return', 'return_confirm',
              'original', 'npc', 'manager', 'lower_confirm', 'lower_verify',
              'close_confirm', 'closed', 'search', 'new_gate', 'panel_close',
              'enter', 'enter_confirm', 'arrive', 'route_open', 'route_menu',
              'route_running', 'fishing_stop', 'fishing_prepare',
              'first_map', 'first_select', 'first_close', 'first_idle', 'first_verify'}
    INPUT_LIMITS = {'completion': 1, 'stop': 3, 'handoff-cancel': 2, 'lower-boss-abandon': 1,
                    'lower-boss-abandon-menu': 1, 'return-request': 8,
                    'return-confirm': 3, 'npc': 3, 'lower': 1,
                    'lower-confirm': 1, 'close-gate': 1, 'close-confirm': 1,
                    'activate': 1, 'search-confirm': 1, 'panel-close': 2,
                    'enter': 2, 'enter-confirm': 2, 'route-open': 3,
                    'route-start': 1, 'enable-auto-sale': 1, 'restore-stop': 3, 'restore-route-open': 3,
                    'restore-route-start': 1, 'first-map': 3, 'first-select': 1,
                    'first-close': 3, 'first-wake': 1, 'first-cast': 1, 'first-pan': 16,
                    **{'target:'+n: 1 for n in TARGETS}, 'cruise-auto-sale':1,'cruise-auto-upgrade':1}

    @staticmethod
    def stage_pair(record):
        """A journal binds one adjacent pair; never reinterpret an old trip."""
        source, target = record.get('from_stage'), record.get('to_stage')
        if type(source) is not int or type(target) is not int:
            return None
        if record.get('reason') == 'lower' and 2 <= source <= 99 and target == source-1:
            return source, target
        if record.get('reason') == 'cycle' and 1 <= source < 99 and target == source+1:
            return target, source
        return None

    def __init__(self, root, reader, action, event, interval=2, *, state_path=None,
                 defense_action=None):
        super().__init__(root, reader)
        self.root, self.action, self.event = Path(root), action, event
        self.interval = max(0.2, float(interval))
        self.path = Path(state_path) if state_path else self.root/'stone_farm_state.json'
        self.journal = dict(version=1, pending=False, cycles=0, handoffs=0, recoveries=0, phase='idle')
        if self.path.is_file():
            saved = json.loads(self.path.read_text(encoding='utf-8-sig'))
            if (saved.get('version') != 1 or type(saved.get('pending')) is not bool
                    or saved.get('phase') not in self.PHASES
                    or saved.get('pending') and (
                        saved.get('reason') not in ('lower', 'cycle')
                        or saved.get('rank') != (39 if saved['reason'] == 'lower' else 40)
                        or not isinstance(saved.get('deadline'), (int, float))
                        or not np.isfinite(saved['deadline'])
                        or self.stage_pair(saved) is None
                        or saved.get('entry_rank') != (39 if saved['reason'] == 'lower' else 0)
                        or saved.get('phase') == 'idle'
                        or 'continuous_cast_stop' in saved and type(saved['continuous_cast_stop']) is not bool
                        or 'allow_boss_abandon' in saved and type(saved['allow_boss_abandon']) is not bool
                        or saved.get('allow_boss_abandon') is True and (
                            saved.get('reason') != 'lower' or saved.get('rank') != 39
                            or saved.get('from_stage') != 9 or saved.get('to_stage') != 8)
                        or 'boss_abandon_menu_retry' in saved and type(saved['boss_abandon_menu_retry']) is not bool
                        or saved.get('boss_abandon_menu_retry') is True and (
                            saved.get('allow_boss_abandon') is not True
                            or saved.get('attempts',{}).get('lower-boss-abandon') != 1)
                        or not isinstance(saved.get('phase_deadline'), (int,float))
                        or not np.isfinite(saved['phase_deadline'])
                        or not isinstance(saved.get('return_deadline'), (int,float))
                        or not np.isfinite(saved['return_deadline'])
                        or not isinstance(saved.get('attempts'), dict)
                        or any(k not in self.INPUT_LIMITS or type(v) is not int or v < 0
                               or v > self.INPUT_LIMITS[k] for k,v in saved['attempts'].items()))):
                raise RuntimeError('次元石降阶记录无效，已停止并保留原记录')
            self.journal = saved
        # Old configurations need no migration. An in-flight journal retains
        # its pair; otherwise fresh native frames can adopt the current stage.
        self.START_STAGE, self.FINAL_STAGE = self.stage_pair(self.journal) or (9, 8)
        self.stage_probe = self.stage_probe_since = self.stage_probe_seen_at = None
        self.stage_probe_frames = 0
        self.stage_probe_frame = None
        self.world = _ScreenContext(root, reader, lambda: None)
        self.training = self.world.training  # read-only, never the training_state.json journal
        self.meter_action = lambda action, data: False
        self.fishing = FishingMeter(root, self.training,
                                   lambda action, data: self.meter_action(action, data), event)
        self.world.fishing = self.fishing
        self.fishing.range_cache = self.journal.get('restore_meter_limits')
        self.sailing = TrainingSailing(root, self.training, None, event, self.fishing)
        self.world.sailing = self.sailing
        # Native sailing has no legacy ParallelWorld worker. Keep its battle
        # controller independent of navigation/training, and enable attacks
        # only when each new frame proves the actual Guardian or Sea God.
        # Guardian uses damage taps only; Sea God retains both attack paths.
        self.training.options.update(boss_interrupt=False, battle_tap=False)
        self.defense = BossDefense(root, self.training, defense_action, event,
                                   scan=defense_action is not None, final_boss_attacks=True)
        self.portal_frames = {}
        for name in ('parallel-manager', 'parallel-manager-inactive',
                     'parallel-lower-question', 'observatory-npc', 'observatory-npc-day'):
            raw = cv2.imread(str(self.root/'assets/recovery'/f'{name}.png'))
            if raw is None or raw.shape[:2] != (1280, 720):
                raise RuntimeError('次元石降阶缺少传送门模板')
            self.portal_frames[name] = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        self.completion_frame = self._load('assets/recovery/parallel-complete.png')
        self.return_icon = self._load('assets/stone-farm/return-icon.png')
        self.boss_abandon_icon = self._load('assets/stone-farm/boss-abandon.png')
        self.cast_stop_fish = self._load('assets/stone-farm/cast-stop-fish.png')
        fish = self.cast_stop_fish.astype(np.int16)
        self.cast_stop_mask = cv2.dilate(((fish[:,:,2]-fish[:,:,0] > 25)
                                         & (fish[:,:,2] > 140)).astype('uint8'),
                                        np.ones((3,3),np.uint8))
        abandon = cv2.cvtColor(self.boss_abandon_icon,cv2.COLOR_RGB2HSV)
        self.boss_abandon_mask = cv2.dilate(((abandon[:,:,0] < 20) & (abandon[:,:,1] > 100)
            & (abandon[:,:,2] > 120)).astype('uint8'),np.ones((3,3),np.uint8))
        destination = self._load('assets/stone-farm/choose-destination.png')
        self.destination_ink = destination.max(axis=2) < 60
        self.menu_zero = self._load('assets/stone-farm/menu-rank-zero.png')
        self.menu_one = self._load('assets/stone-farm/menu-rank-one.png')
        self.map_legend = self._load('assets/task-popup/map-legend.png')
        self.moving_label = self._load('assets/backpack/moving.png')
        self.first_map_core = self._load('assets/stone-farm/first-map-skull-core.png')
        self.selected_first_map_core = self._load('assets/stone-farm/selected-first-map-skull-core.png')
        self.lower_map_core = self._load('assets/stone-farm/lower-map-skull-core.png')
        self.lower_map_bone = self._load('assets/stone-farm/lower-map-skull-bone.png')
        self.world.assets['success-check'] = self._load('assets/training/success-check.png')
        self.native_entry_auto_label = self._load('assets/stone-farm/native-entry-auto-label.png')
        # The swirl moves; only the metal frame and its four corners are stable.
        self.return_mask = np.ones(self.return_icon.shape[:2], np.uint8)
        self.return_mask[8:-8, 8:-8] = 0
        # RANK40 adds a yellow notification badge over the bottom-right metal
        # corner. Keep the other fixed frame edges and the .96 threshold;
        # neither the animated swirl nor the badge identifies this button.
        self.return_mask[35:, 30:] = 0
        self.phase = self.journal['phase'] if self.active else 'idle'
        self.enabled = lambda: True
        self.observed_restore_allowed = lambda: True
        self.rank = self.journal.get('rank') if self.active else None
        self.current_stage = None
        self.key = self.since = self.last_seen = None
        self.seen = 0
        self.next_at = 0
        self.off_verified = False  # A restart requires new OFF proof.
        self.boss_abandon_key = None
        self.cancel_anchor = None
        self.fishing_stopped = False
        self.hold = None
        self.return_probe = None
        self.return_wait_logged_at = None
        self.menu_key = None
        self.meter_context = None
        self.first_idle_key = None
        self.first_map_target = None
        self.first_map_verified = False
        self.first_result_rank = self.first_result_frame = None
        self.first_result_observed = False
        self.retry_path = self.path.with_name('stone_farm_retry.json')
        self.retry = None
        if self.retry_path.is_file():
            request = json.loads(self.retry_path.read_text(encoding='utf-8-sig'))
            if (self.active and request.get('failed_sha256') == hashlib.sha256(self.path.read_bytes()).hexdigest()
                    and (request.get('allow_ordinary_cancel') is True
                         or request.get('mode') == 'observed_lower_restore'
                         and request.get('allow_observed_lower_restore') is True
                         or request.get('mode') == 'observed_cycle_restore'
                         and request.get('allow_observed_cycle_restore') is True)):
                self.retry = request

    @staticmethod
    def request_retry(root, failed_sha256, *, allow_boss_abandon=False, allow_boss_abandon_menu=False,
                      allow_continuous_cancel=False, allow_post_start_restore=False,
                      allow_first_result_restore=False, allow_manual_rank_progress=False,
                      allow_observed_lower_restore=False, allow_observed_cycle_restore=False,
                       allow_return_entry_restore=False, allow_map_target_repair=False):
        """Explicit one-run approval, without resetting the failed journal."""
        root = Path(root)
        config = json.loads((root/'config.json').read_text(encoding='utf-8-sig'))
        if not any(r['id'] == 'driftinggirl_stone_farm' and r['enabled'] for r in config['rules']):
            raise ValueError('请先启用次元石降阶功能')
        raw = (root/'stone_farm_state.json').read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        journal = json.loads(raw.decode('utf-8-sig'))
        pair = StoneFarm.stage_pair(journal)
        upper_stage, lower_stage = pair or (None, None)
        returned_failure = (journal.get('phase') in ('stop_fishing','return','return_confirm')
                            and (time.time() > journal.get('return_deadline',float('inf'))
                                 or journal.get('attempts',{}).get('return-request',0) >= StoneFarm.INPUT_LIMITS['return-request']))
        route_failure = (journal.get('phase') == 'route_menu'
                         and journal.get('verified_gate_stage') == journal.get('to_stage')
                         and journal.get('entered_verified_at') and time.time() > journal.get('phase_deadline',float('inf'))
                         and any(a.get('kind') == 'enter-confirm' for a in journal.get('issued_inputs',[])))
        route_open_failure = (journal.get('phase') == 'route_open'
                              and journal.get('reason') == 'lower'
                              and journal.get('verified_gate_stage') == journal.get('to_stage')
                              and journal.get('entered_verified_at') and journal.get('first_cast_verified_at')
                              and journal.get('first_cast_verified_rank') == journal.get('entry_rank') == 39
                              and time.time() > journal.get('phase_deadline',float('inf'))
                              and any(a.get('kind') == 'first-cast' for a in journal.get('issued_inputs',[]))
                              and journal.get('attempts',{}).get('first-cast') == 1
                              and not journal.get('attempts',{}).get('restore-route-start')
                              and not journal.get('route_open_repair_runs'))
        fishing_failure = (journal.get('phase') in ('route_running','fishing_prepare')
                           and journal.get('verified_gate_stage') == lower_stage
                           and journal.get('entered_verified_at')
                           and time.time() > journal.get('phase_deadline',float('inf'))
                           and any(a.get('kind') == 'route-start' for a in journal.get('issued_inputs',[]))
                           and not journal.get('meter_attempts',{}).get('start'))
        post_start_failure = (allow_post_start_restore is True
            and journal.get('phase') == 'fishing_prepare'
            and journal.get('verified_gate_stage') == lower_stage and journal.get('entered_verified_at')
            and time.time() > journal.get('phase_deadline',float('inf'))
            and journal.get('meter_attempts',{}).get('start') == 1
            and not journal.get('post_start_restore_runs')
            and not journal.get('attempts',{}).get('restore-route-start')
            and journal.get('restore_signature'))
        gate_failure = (journal.get('phase') == 'manager'
                        and journal.get('resumed_after_unverified_return') is True
                        and journal.get('returned_at')
                        and time.time() > journal.get('phase_deadline',float('inf')))
        startup_failure = (journal.get('phase') == 'fishing_prepare'
                           and journal.get('verified_gate_stage') == upper_stage
                           and journal.get('entered_verified_at')
                           and time.time() > journal.get('phase_deadline',float('inf'))
                           and any(a.get('kind') == 'enter-confirm' for a in journal.get('issued_inputs',[]))
                           and not journal.get('meter_attempts'))
        first_failure = (journal.get('phase') == 'route_running'
                         and journal.get('verified_gate_stage') == upper_stage
                         and journal.get('entered_verified_at')
                         and time.time() > journal.get('phase_deadline',float('inf'))
                         and any(a.get('kind') == 'route-start' for a in journal.get('issued_inputs',[]))
                         and not journal.get('attempts',{}).get('first-cast')
                         and not journal.get('meter_attempts',{}).get('start'))
        map_target_repair = (allow_map_target_repair is True
                             and journal.get('first_map_repair_runs') == 1
                             and not journal.get('first_map_target_repair_runs')
                             and journal.get('reason') == 'lower'
                             and journal.get('first_unselected') is True)
        # A notice can consume the map-open click before the map appears.
        # The explicit, hash-bound repair can reuse the map restoration path
        # only while an original map allowance remains; never replenish it.
        interrupted_map_open = (journal.get('phase') == 'first_map'
            and journal.get('first_unselected') is True
            and journal.get('first_signature')
            and 0 < journal.get('attempts',{}).get('first-map',0) < StoneFarm.INPUT_LIMITS['first-map']
            and not journal.get('meter_attempts')
            and not any(journal.get('attempts',{}).get(k) for k in ('first-wake','first-close')))
        first_map_failure = ((journal.get('phase') == 'first_select' or interrupted_map_open)
                             and journal.get('verified_gate_stage') == journal.get('to_stage')
                             and journal.get('entered_verified_at')
                             and time.time() > journal.get('phase_deadline',float('inf'))
                             and any(a.get('kind') == 'first-map' for a in journal.get('issued_inputs',[]))
                             and not journal.get('attempts',{}).get('first-cast')
                             and not journal.get('attempts',{}).get('first-select')
                             and (not journal.get('first_map_repair_runs') or map_target_repair))
        if (type(allow_map_target_repair) is not bool or allow_map_target_repair
                and not (map_target_repair and first_map_failure)):
            raise ValueError('选点修复接续仅限一次未选中任务的降阶地图失败；保留全部已用次数')
        first_close_failure = (journal.get('phase') == 'first_close'
            and journal.get('first_unselected') is True
            and journal.get('verified_gate_stage') == upper_stage and journal.get('entered_verified_at')
            and time.time() > journal.get('phase_deadline',float('inf'))
            and journal.get('first_selected_rank') == 0
            and isinstance(journal.get('first_selected_point'),list)
            and len(journal['first_selected_point']) == 2
            and all(type(v) is int for v in journal['first_selected_point'])
            and all(any(a.get('kind') == k for a in journal.get('issued_inputs',[]))
                    for k in ('enter-confirm','first-map','first-select'))
            and journal.get('attempts',{}).get('first-select') == 1
            and not any(journal.get('attempts',{}).get(k)
                        for k in ('first-close','first-wake','first-cast','restore-route-open','restore-route-start'))
            and not any(a.get('kind') in ('first-close','first-wake','first-cast','restore-route-open','restore-route-start')
                        for a in journal.get('issued_inputs',[]))
            and not journal.get('meter_attempts') and not journal.get('first_close_repair_runs'))
        first_idle_failure = (journal.get('phase') == 'first_idle'
                              and journal.get('verified_gate_stage') == journal.get('to_stage')
                              and journal.get('entered_verified_at')
                              and time.time() > journal.get('phase_deadline',float('inf'))
                              and journal.get('first_selected_rank') == journal.get('entry_rank')
                              and all(any(a.get('kind') == kind for a in journal.get('issued_inputs',[]))
                                      for kind in ('first-select','first-wake'))
                              and journal.get('attempts',{}).get('first-wake') == 1
                              and not journal.get('attempts',{}).get('first-cast')
                              and not journal.get('first_idle_repair_runs'))
        first_result_failure = (journal.get('phase') == 'first_verify'
                                and journal.get('verified_gate_stage') == upper_stage
                                and journal.get('entered_verified_at')
                                and time.time() > journal.get('phase_deadline',float('inf'))
                                and journal.get('attempts',{}).get('first-cast') == 1
                                and journal.get('first_selected_rank') == 0
                                and all(any(a.get('kind') == kind for a in journal.get('issued_inputs',[]))
                                        for kind in ('first-select','first-cast'))
                                and not journal.get('first_result_repair_runs'))
        lower_first_result_failure = (journal.get('phase') == 'first_verify'
            and journal.get('verified_gate_stage') == lower_stage and journal.get('entered_verified_at')
            and time.time() > journal.get('phase_deadline',float('inf'))
            and journal.get('first_selected_rank') == journal.get('entry_rank') == 39
            and journal.get('attempts',{}).get('first-cast') == 1
            and all(any(a.get('kind') == kind for a in journal.get('issued_inputs',[]))
                    for kind in ('enter-confirm','first-select','first-cast'))
            and not journal.get('first_result_repair_runs')
            and not any(journal.get('attempts',{}).get(k)
                        for k in ('restore-route-open','restore-route-start')))
        first_entry_failure = (journal.get('phase') == 'fishing_stop'
                               and journal.get('first_recovery') is True
                               and journal.get('verified_gate_stage') == upper_stage
                               and journal.get('entered_verified_at')
                               and time.time() > journal.get('phase_deadline',float('inf'))
                               and all(any(a.get('kind') == kind for a in journal.get('issued_inputs',[]))
                                       for kind in ('route-start','restore-stop'))
                               and not any(journal.get('attempts',{}).get(kind)
                                           for kind in ('first-map','first-select','first-wake','first-cast'))
                               and not journal.get('meter_attempts')
                               and not journal.get('first_entry_repair_runs'))
        route_result_failure = (journal.get('phase') == 'route_running'
                                and journal.get('verified_gate_stage') == upper_stage
                                and journal.get('entered_verified_at')
                                and journal.get('first_cast_verified_rank') == 1
                                and journal.get('first_cast_verified_at')
                                and time.time() > journal.get('phase_deadline',float('inf'))
                                and all(any(a.get('kind') == kind for a in journal.get('issued_inputs',[]))
                                        for kind in ('first-cast','restore-route-start'))
                                and not journal.get('route_result_repair_runs'))
        lower = (journal.get('reason') == 'lower' and journal.get('rank') == 39
                 and pair is not None
                 and journal.get('entry_rank') == 39)
        cycle = (journal.get('reason') == 'cycle' and journal.get('rank') == 40
                 and pair is not None
                 and journal.get('entry_rank') == 0 and journal.get('rank40_verified') is True)
        return_entry_failure = ((lower or cycle) and returned_failure
            and journal.get('phase') == 'return'
            and journal.get('off_verified_at') and journal.get('fishing_stopped_at')
            and journal.get('continuous_cast_stop') is True
            and not journal.get('attempts',{}).get('return-request')
            and not journal.get('return_entry_repair_runs'))
        if (type(allow_return_entry_restore) is not bool or allow_return_entry_restore and not (
                return_entry_failure and not any((allow_boss_abandon,allow_boss_abandon_menu,
                    allow_continuous_cancel,allow_post_start_restore,allow_first_result_restore,
                    allow_manual_rank_progress,allow_observed_lower_restore,allow_observed_cycle_restore)))):
            raise ValueError('返回入口恢复仅限已停钓但未发出返回输入的超时；保留旧次数且仅允许一次')
        observed_cycle_failure = (cycle and first_result_failure
            and not journal.get('observed_cycle_restore_runs')
            and all(any(a.get('kind') == k for a in journal.get('issued_inputs',[]))
                    for k in ('return-request','return-confirm','enter-confirm','first-select','first-cast')))
        if (type(allow_observed_cycle_restore) is not bool or allow_observed_cycle_restore and not (
                observed_cycle_failure and not any((allow_boss_abandon,allow_boss_abandon_menu,
                    allow_continuous_cancel,allow_post_start_restore,allow_first_result_restore,
                    allow_manual_rank_progress,allow_observed_lower_restore)))):
            raise ValueError('新轮只读接续仅限已有单次首竿的验证超时；不允许新增输入或重置期限')
        observed_lower_failure = (lower and fishing_failure
            and journal.get('phase') == 'route_running'
            and not journal.get('observed_lower_restore_runs')
            and all(any(a.get('kind') == k for a in journal.get('issued_inputs',[]))
                    for k in ('lower','lower-confirm','enter-confirm','route-start')))
        if (type(allow_observed_lower_restore) is not bool or allow_observed_lower_restore and not (
                observed_lower_failure and not any((allow_boss_abandon,allow_boss_abandon_menu,
                    allow_continuous_cancel,allow_post_start_restore,allow_first_result_restore,
                    allow_manual_rank_progress,allow_observed_cycle_restore)))):
            raise ValueError('已恢复钓鱼的只读接续仅限本次降阶航巡超时；不允许新增输入或重置期限')
        manual_progress_failure = (cycle and journal.get('phase') == 'first_close'
            and journal.get('verified_gate_stage') == upper_stage and journal.get('entered_verified_at')
            and time.time() > journal.get('phase_deadline',float('inf'))
            and journal.get('first_selected_rank') == 0
            and all(any(a.get('kind') == k for a in journal.get('issued_inputs',[]))
                    for k in ('first-map','first-select','first-close'))
            and not journal.get('attempts',{}).get('first-cast')
            and not journal.get('meter_attempts') and not journal.get('manual_rank_progress_runs'))
        if (type(allow_manual_rank_progress) is not bool or allow_manual_rank_progress and not (
                manual_progress_failure and not any((allow_boss_abandon,allow_boss_abandon_menu,
                    allow_continuous_cancel,allow_post_start_restore,allow_first_result_restore)))):
            raise ValueError('手动推进接续仅限本次首任务移动超时；不重复抛竿或标尺输入')
        if (digest != failed_sha256 or not journal.get('pending')
                or not (lower and (returned_failure or route_failure or route_open_failure or fishing_failure or post_start_failure or first_map_failure or first_idle_failure
                                  or allow_first_result_restore is True and lower_first_result_failure)
                        or cycle and (returned_failure or gate_failure or startup_failure or route_failure or first_failure or first_map_failure or first_close_failure or first_idle_failure or first_result_failure or first_entry_failure or route_result_failure
                                      or allow_manual_rank_progress and manual_progress_failure))):
            raise ValueError('仅可为已核对的降阶返回失败另开观察，原记录必须一致')
        if (type(allow_first_result_restore) is not bool or allow_first_result_restore and not (
                lower and lower_first_result_failure and not any((allow_boss_abandon,
                    allow_boss_abandon_menu,allow_continuous_cancel,allow_post_start_restore)))):
            raise ValueError('降阶抛竿结果恢复仅限本次已降一阶的RANK39；保留已用抛竿且仅恢复未用航巡步骤')
        if (type(allow_post_start_restore) is not bool or allow_post_start_restore and not (
                lower and post_start_failure and not allow_boss_abandon
                and not allow_boss_abandon_menu and not allow_continuous_cancel)):
            raise ValueError('长按后恢复仅限本次已降一阶的RANK39的一次已有长按；不能重置输入或重复申请')
        if (type(allow_boss_abandon) is not bool or allow_boss_abandon and not (
                lower and journal.get('from_stage') == 9 and journal.get('to_stage') == 8
                and returned_failure and journal.get('phase') == 'stop_fishing'
                and time.time() > journal.get('return_deadline',float('inf'))
                and not journal.get('attempts',{}).get('lower-boss-abandon'))):
            raise ValueError('放弃BOSS仅限已核对的9阶RANK39停钓超时，且本场尚未尝试放弃')
        menu_receipt = StoneFarm._boss_abandon_receipt(root,journal) if allow_boss_abandon_menu else None
        if (type(allow_boss_abandon_menu) is not bool or allow_boss_abandon_menu and not (
                not allow_boss_abandon and lower and returned_failure
                and journal.get('from_stage') == 9 and journal.get('to_stage') == 8
                and journal.get('phase') == 'stop_fishing'
                and time.time() > journal.get('return_deadline',float('inf'))
                and journal.get('allow_boss_abandon') is True
                and journal.get('attempts',{}).get('lower-boss-abandon') == 1
                and not journal.get('attempts',{}).get('lower-boss-abandon-menu')
                and not journal.get('boss_abandon_menu_retry') and menu_receipt)):
            raise ValueError('追加80毫秒接触仅限本次9阶RANK39降阶，须保留一次已释放的原生输入回执且尚未追加')
        if (type(allow_continuous_cancel) is not bool or allow_continuous_cancel and not (
                not allow_boss_abandon and not allow_boss_abandon_menu
                and returned_failure and (lower or cycle)
                and journal.get('phase') == 'stop_fishing'
                and not journal.get('continuous_stop_repair_runs')
                and journal.get('attempts',{}).get('handoff-cancel',0) < StoneFarm.INPUT_LIMITS['handoff-cancel'])):
            raise ValueError('持续停钓修复仅限本次停钓超时，保留已有次数且不能再次申请同一修复')
        if (lower and returned_failure and not allow_boss_abandon_menu and not allow_continuous_cancel
                and any(journal.get('attempts',{}).get(k) for k in ('lower-boss-abandon','lower-boss-abandon-menu'))):
            raise ValueError('已有放弃BOSS尝试不能通过普通重试重置次数')
        request = dict(failed_sha256=digest, approved_at=time.time(), expires_at=time.time()+300,
                       allow_ordinary_cancel=True, mode='lower_first_result_restore' if allow_first_result_restore
                       else 'route_result_restore' if cycle and route_result_failure
                       else 'first_entry_restore' if cycle and first_entry_failure
                       else 'first_result_restore' if cycle and first_result_failure
                       else 'first_close_restore' if cycle and first_close_failure
                       else 'first_map_restore' if first_map_failure
                       else 'first_idle_restore' if first_idle_failure
                       else 'route_open_restore' if route_open_failure
                       else 'first_cast_restore' if cycle and first_failure
                       else 'startup_restore' if cycle and startup_failure
                       else 'gate_restore' if cycle and gate_failure
                       else 'fishing_restore' if fishing_failure
                       else 'route_restore' if route_failure else 'return_retry')
        if first_map_failure and interrupted_map_open:
            request['map_open_interrupted'] = True
        if allow_return_entry_restore:
            request.update(mode='return_entry_restore',allow_return_entry_restore=True)
        if allow_map_target_repair:
            request['allow_map_target_repair'] = True
        if allow_boss_abandon:
            request['allow_boss_abandon'] = True
        if allow_boss_abandon_menu:
            request.update(mode='boss_abandon_menu_retry',allow_boss_abandon_menu=True,
                           native_abandon_receipt_sha256=menu_receipt)
        if allow_continuous_cancel:
            request.update(mode='continuous_cast_stop_retry',allow_continuous_cancel=True)
        if allow_post_start_restore:
            request.update(mode='post_start_restore',allow_post_start_restore=True)
        if allow_manual_rank_progress:
            request.update(mode='manual_rank_progress_restore',allow_manual_rank_progress=True)
        if allow_observed_lower_restore:
            request.update(mode='observed_lower_restore',allow_observed_lower_restore=True,
                           allow_ordinary_cancel=False,expires_at=time.time()+60)
        if allow_observed_cycle_restore:
            request.update(mode='observed_cycle_restore',allow_observed_cycle_restore=True,
                           allow_ordinary_cancel=False,expires_at=time.time()+60)
        path = root/'stone_farm_retry.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(request,indent=2)+'\n',encoding='utf-8')
        temporary.replace(path)
        return request

    @staticmethod
    def _boss_abandon_receipt(root, journal):
        """A known completed native contact permits one separately approved menu attempt."""
        issued = [a for a in journal.get('issued_inputs',[]) if isinstance(a,dict) and a.get('kind') == 'lower-boss-abandon']
        planned = [a for a in journal.get('inputs',[]) if isinstance(a,dict) and a.get('kind') == 'lower-boss-abandon']
        if len(issued) != 1 or len(planned) != 1:
            return None
        prior = issued[0]
        identifier = prior.get('input_id')
        if not isinstance(identifier,str) or not re.fullmatch('[0-9a-f]{32}',identifier):
            return None
        path = Path(root)/'logs/portal-input'/('input-'+identifier+'.json')
        try:
            if path.stat().st_size > 131072:
                return None
            raw = path.read_bytes()
            receipt = json.loads(raw.decode('utf-8-sig'))
        except (OSError,ValueError):
            return None
        if not isinstance(receipt,dict):
            return None
        numbers = [receipt.get(k) for k in ('input_begin_time','touch_down_time','touch_up_time','capture_age_ms')]
        if any(type(n) not in (int,float) or not np.isfinite(n) for n in numbers):
            return None
        began,down,up,age = numbers
        if (receipt.get('input_id') != identifier or receipt.get('frame_id') != prior.get('frame_id')
                or receipt.get('kind') != 'lower-boss-abandon' or receipt.get('controller') != 'stone_farm'
                or receipt.get('rank') != 39 or receipt.get('completed_rank_40') is not False
                or receipt.get('input_transport') != 'mumu-native' or prior.get('transport') != 'mumu-native'
                or receipt.get('hold_ms') != 40 or prior.get('time') != began
                or not journal.get('started_at',float('inf')) <= began <= journal.get('return_deadline',0)
                or down < began or not .035 <= up-down <= .12 or not 0 <= age <= 120
                or type(receipt.get('x')) is not int or type(receipt.get('y')) is not int
                or not 0 <= receipt['x'] < 720 or not 0 <= receipt['y'] < 1280
                or (receipt['x'],receipt['y']) != (planned[0].get('x'),planned[0].get('y'))
                or receipt.get('native_xy') != [receipt.get('x'),receipt.get('y')]
                or any(receipt.get(k) for k in ('error','uncertain_down','down_failed'))):
            return None
        return hashlib.sha256(raw).hexdigest()

    def _load(self, relative):
        raw = cv2.imread(str(self.root/relative))
        if raw is None:
            raise RuntimeError('次元石降阶模板缺失：'+relative)
        return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    @property
    def active(self):
        return self.journal['pending']

    @property
    def completed_rank_40(self):
        return bool(self.active and self.journal.get('reason') == 'cycle'
                    and self.journal.get('rank') == 40 and self.journal.get('rank40_verified'))

    @property
    def ordinary_cancel_allowed(self):
        # The user authorised the same stopped ordinary-fish guard for both
        # the RANK39 handoff and the actual completed RANK40 return. No other
        # stage, rank, or legacy calibration gains cancellation permission.
        return bool(self.active and self.journal.get('allow_ordinary_cancel') is True
                    and self.stage_pair(self.journal) is not None
                    and ((self.journal.get('reason') == 'lower'
                          and self.journal.get('rank') == 39)
                         or self.completed_rank_40))

    @property
    def boss_abandon_handoff(self):
        # One explicit, hash-bound recovery approval for this lower handoff.
        # It is never copied to a new automatic handoff or to meter recovery.
        lower = (self.active and not self.continuous_cancel_allowed
                 and self.journal.get('reason') == 'lower'
                 and (self.journal.get('from_stage'),self.journal.get('rank'),
                      self.journal.get('to_stage'),self.journal.get('entry_rank')) == (9,39,8,39))
        approved = (self.journal.get('allow_boss_abandon') is True
                    or self.retry is not None and (
                        self.retry.get('mode') == 'return_retry' and self.retry.get('allow_boss_abandon') is True
                        or self.retry.get('mode') == 'boss_abandon_menu_retry'
                        and self.retry.get('allow_boss_abandon_menu') is True))
        return bool(lower and approved and self.phase in ('stop','stop_fishing'))

    @property
    def continuous_cancel_allowed(self):
        return bool(self.ordinary_cancel_allowed and self.journal.get('continuous_cast_stop') is True)

    @property
    def handoff_stop_requested(self):
        return bool(self.continuous_cancel_allowed and self.phase in ('stop','stop_fishing'))

    @property
    def first_route_resume_requested(self):
        # Finish only the route restoration after an actually verified cast.
        # Battle ownership must not starve its finite confirmation window.
        if (self.retry is not None and self.retry.get('mode') in ('lower_first_result_restore','observed_lower_restore','observed_cycle_restore')
                and self.enabled() and time.time() <= self.retry['expires_at']):
            return True  # Read-only fresh result votes precede every input.
        return bool(self._first_scope() and self.input_allowed()
                    and (self.journal.get('first_cast_verified_at')
                         or self.journal.get('selected_fishing_verified_at'))
                    and self.phase in ('route_open','route_menu','route_running'))

    @property
    def boss_abandon_kind(self):
        return 'lower-boss-abandon-menu' if self.journal.get('boss_abandon_menu_retry') is True else 'lower-boss-abandon'

    def status(self):
        return dict(phase=self.phase, pending=self.active, rank=self.rank,
                    world_stage=self.current_stage, start_stage=self.START_STAGE,
                    final_stage=self.FINAL_STAGE, adaptive_stage=True,
                    handoff_rank=39, cycles=self.journal.get('cycles', 0),
                    handoffs=self.journal.get('handoffs', 0), hold=self.hold,
                    deadline=self.journal.get('deadline'),
                    boss_abandon_approved=self.boss_abandon_handoff,
                    boss_abandon_attempts=self.journal.get('attempts',{}).get('lower-boss-abandon',0),
                    boss_abandon_menu_attempts=self.journal.get('attempts',{}).get('lower-boss-abandon-menu',0),
                    continuous_cast_stop=self.continuous_cancel_allowed,
                    continuous_stop_attempts=self.journal.get('attempts',{}).get('handoff-cancel',0),
                    targets=['珍珠', '琥珀', '蓝色宝石', 'BOSS'],
                    retirement=self.journal.get('retirement'),
                    preserves_native_extras=True)

    def handoff_due(self, screen):
        if not screen:
            return False
        stage, rank = screen['stage'], screen['rank']
        return (rank == 40 and stage == self.FINAL_STAGE
                or rank == self.HANDOFF_RANK and 2 <= stage <= 99
                and (stage == self.START_STAGE or stage > self.START_STAGE))

    def observe_stage_policy(self, frame, screen):
        """Read-only adoption, independent of RANK trigger confirmation."""
        if self.active or frame is self.stage_probe_frame:
            return False
        self.stage_probe_frame = frame
        candidate = None
        if (screen and not screen['menu'] and not screen['completion']
                and 2 <= screen['stage'] <= 99 and 0 <= screen['rank'] <= 39):
            stage = screen['stage']
            if stage > self.START_STAGE or stage != self.START_STAGE and screen['rank'] < 39:
                candidate = (stage, stage-1)
        now = time.monotonic()
        if candidate is None:
            self.stage_probe = self.stage_probe_since = self.stage_probe_seen_at = None
            self.stage_probe_frames = 0
            return False
        if (candidate != self.stage_probe or self.stage_probe_seen_at is None
                or now-self.stage_probe_seen_at > 8):
            self.stage_probe, self.stage_probe_since, self.stage_probe_frames = candidate, now, 1
        else:
            self.stage_probe_frames += 1
        self.stage_probe_seen_at = now
        if self.stage_probe_frames < 3 or now-self.stage_probe_since < 3:
            return False
        self.START_STAGE, self.FINAL_STAGE = candidate
        self.event(f'次元石：连续新图确认当前第{self.START_STAGE}阶段，'
                   f'本轮RANK39降到第{self.FINAL_STAGE}阶段打最后BOSS；保留旧交接记录')
        return True

    def retire_expired_on_start(self):
        # Called only at a new worker/recovered-game boundary; constructors
        # and ordinary controller recreation remain read-only.
        if not self.active or not self.path.is_file() or not handoff_journal.expiry(self.journal, time.time()):
            return False
        if self.retry is not None and time.time() < self.retry.get('expires_at', 0):
            return False  # Honour an existing, hash-bound recovery request.
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        result = handoff_journal.retire(self.path.parent, digest, source='startup_expired')
        self.journal = json.loads(self.path.read_text(encoding='utf-8-sig'))
        self.phase, self.retry = 'idle', None
        self.event(f'次元石：启动时已归档过期交接（{result["previous_phase"]}）；保留原次数和截止，不重试同一交接，其他任务继续')
        return True

    def _save(self):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.journal, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        temporary.replace(self.path)

    def retire_retryable(self, frame):
        """Release an expired handoff only on its fresh unobscured main.

        Keep the failed handoff byte-for-byte and block replay of its contexts,
        just as startup retirement does. Explicit recovery approvals keep ownership.
        """
        returning = self.phase in ('stop_fishing','return','return_confirm')
        first_verifying = self.phase == 'first_verify'
        if (not self.active or self.retry is not None
                or self.phase != 'route_running' and not returning and not first_verifying
                or (returning or first_verifying) and self.journal.get('phase') != self.phase):
            return False
        if first_verifying:
            issued = {a.get('kind') for a in self.journal.get('issued_inputs', [])}
            if (not self._first_scope()
                    or self.journal.get('first_selected_rank') != self.journal.get('entry_rank')
                    or self.journal.get('attempts', {}).get('first-cast') != 1
                    or not {'enter-confirm', 'first-select', 'first-cast'} <= issued):
                return False
        expired = handoff_journal.expiry(self.journal, time.time())
        screen = self.screen(frame)
        identity = (screen and screen['stage'] == self.journal['from_stage']
                    and screen['rank'] == self.journal['rank']
                    and screen['route'] is False and self._off_icon(frame)
                    if returning else screen and screen['stage'] == self.journal['to_stage']
                    and self.journal['entry_rank'] <= screen['rank'] <= 40)
        main = self.fishing._main_visible(frame)
        if first_verifying:
            # The cast allowance is already spent. An expired result wait may
            # release ownership on its actual stopped, selected destination;
            # it must not cast again or claim that fishing has started. The
            # normal cruise controller can then reconcile the saved policy.
            context = self._first_result_context(frame, screen)
            identity = screen and context is not None and screen['route'] is False
            main = context is not None  # Includes independent bright-main guards.
        if (not expired or not screen or screen['menu'] or screen['completion']
                or not identity
                or not main):
            return False
        raw, record = read_record(self.path, '交接')
        if record != self.journal:
            raise ValueError('次元石交接记录已变化，保留原记录并停止')
        digest = hashlib.sha256(raw).hexdigest()
        archive = self.path.parent/'logs/stone-farm-retired'/f'{digest}.json'
        updated = dict(record)
        updated.update(pending=False, phase='idle', retirement=dict(
            at=time.time(), source='retryable_stage', sha256=digest, archive=str(archive),
            previous_phase=self.phase, expired_fields=expired, released=False,
            blocked_contexts=handoff_journal.blocked_contexts(record),
            backoff_seconds=handoff_journal.BACKOFF_SECONDS))
        archive_and_replace(self.path, raw, updated, archive, digest, '交接', 'stone_farm_retry.json')
        self.journal, self.phase = updated, 'idle'
        self.key = self.since = self.last_seen = None
        self.seen, self.off_verified = 0, False
        self.hold = '到期交接已归档，等待退避和实际阶段或RANK推进；不计成功'
        return True

    def _set(self, phase, seconds=45):
        if phase == self.phase:
            return
        self.phase = phase
        deadline = min(self.journal['deadline'],time.time()+seconds)
        if phase.startswith('first_'):
            deadline = min(deadline, self._first_phase_limit(phase))
        self.journal.update(phase=phase, phase_deadline=deadline, phase_started_at=time.time())
        self.first_close_wait_reason = None
        self.key = self.since = self.last_seen = None
        self.seen = 0
        if phase == 'first_verify':
            self.first_result_rank = self.first_result_frame = None
            self.first_result_observed = False
        self._save()
        self.event('次元石降阶：'+phase)

    def _confirmed(self, key, seconds=3):
        now = time.monotonic()
        if key is None:
            self.key = self.since = None
            self.seen = 0
            return False
        if key != self.key or self.last_seen is None or now-self.last_seen > 8:
            self.key, self.since, self.seen = key, now, 1
        else:
            self.seen += 1
        self.last_seen = now
        return self.seen >= 3 and now-self.since >= seconds

    def signature(self, frame):
        # Deep fixed text only. No sea, moving countdown or meter animation.
        parts = []
        for x,y,X,Y in ((71,9,136,59), (80,9,130,59), (13,45,68,72)):
            crop = frame[y:Y,x:X].min(axis=2)
            parts.append((crop >= crop.max()*.84).tobytes().hex())
        # The stage number is cyan, so a white/min-channel mask can omit it.
        # Bind its actual strokes at the same bright and dim thresholds used
        # by stage OCR. This adds no OCR or cached numeral to a fresh guard.
        crop = frame[45:71,40:68].astype(np.int16)
        for threshold in (80,90,110,130):
            ink = ((crop[:,:,1]-crop[:,:,0]>=10)
                   & (crop[:,:,2]-crop[:,:,0]>=15)
                   & (crop.max(axis=2)>=threshold))
            parts.append(ink.tobytes().hex())
        # The complete AUTO settings panel dims the header: five antialiased
        # edge pixels flicker across 60 while its solid letter cores stay put.
        cutoff = 35 if self.phase == 'route_menu' else 60
        parts.append((frame[12:34,286:434].max(axis=2)<cutoff).tobytes().hex())
        return parts

    def _off_icon(self, frame):
        return (self.sailing._fixed(frame, 'auto-label', (186,34,253,65), .90)
                and self.sailing._fixed(frame, 'inactive-face', (194,4,246,42), .96)
                and not self.sailing._fixed(frame, 'active-arrow', (194,4,246,42), .96))

    def _battle(self, frame):
        return (self.defense._match(frame, 'timer', (188,125,610,165), .94)
                or self.defense._match(frame, 'swords', (307,962,415,1057), .96))

    def _idle_fishing(self, frame):
        return (self.fishing._main_visible(frame)
                and not self._battle(frame) and not self.fishing.blue_fill(frame)
                and (self.fishing.at_port(frame) or not self.fishing.busy(frame)
                     and self.fishing.idle_icon(frame) is not None))

    def boss_abandon_context(self, frame):
        # No OCR, old coordinates or scene background in this fast input guard.
        # Actual stage/RANK OCR is confirmed in step(); signature binds that
        # proof to the new frame, with independent real Sea God battle guards.
        if (frame.shape != (1280,720,3) or not self._parallel(frame)
                or not self._off_icon(frame) or self.sailing.running(frame) is not False
                or not self.defense.encounter(frame,39) or not self.defense._battle_footer(frame)):
            return None
        scores = cv2.matchTemplate(frame[158:210,584:642],self.boss_abandon_icon,
                                  cv2.TM_CCOEFF_NORMED,mask=self.boss_abandon_mask)
        scores = np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1)
        _,score,_,location = cv2.minMaxLoc(scores)
        if score < .97:
            return None
        h,w = self.boss_abandon_icon.shape[:2]
        point = (584+location[0]+w//2,158+location[1]+h//2)
        return (tuple(self.signature(frame)),point)

    def _no_destination(self, frame):
        # Only the actual fixed "choose destination" text, never its moving
        # arrow or sea, identifies a new gate with no fishing location yet.
        if frame.shape[:2] != (1280,720):
            return False
        ink = frame[12:34,286:434].max(axis=2) < 60
        expected = int(self.destination_ink.sum())
        return bool(expected >= 100 and .9*expected <= int(ink.sum()) <= 1.1*expected
                    and float(ink[self.destination_ink].mean()) >= .98)

    def cancel_context(self, frame, *, continuous=False):
        if (frame.shape != (1280,720,3) or not self._parallel(frame)
                or not self._off_icon(frame) or self.sailing.running(frame) is not False
                or not self.fishing._main_visible(frame)
                or not continuous and (self.training.boss_mode(frame) or self._battle(frame))):
            return None
        self.training.state['rank'] = self.journal['rank']
        data = self.fishing.cast_data(frame)
        if continuous and data is not None:
            # Real cast frames move a pale glow across the left number-box
            # margin. Keep exact saturated digit cores, not that moving glow.
            # This does not alter any meter calibration/cancellation probe.
            x,y,X,Y = data['box']
            hsv = cv2.cvtColor(frame[y+5:Y-3,x+4:X-7],cv2.COLOR_RGB2HSV)
            ink = ((hsv[:,:,0]>=15)&(hsv[:,:,0]<=35)&(hsv[:,:,1]>100)&(hsv[:,:,2]>170))
            if int(ink.sum()) < 8:
                # Some real numeral strokes are pale cream (S~60-100), not
                # saturated gold. cast_data already bound their exact >60
                # foreground, box, RANK/place and world strokes. Keep that
                # complete fingerprint; never replace it with the empty gold
                # mask, loosen equality, or use a cached numeric value.
                return data
            data['digit_strokes'] = hashlib.sha256(ink.tobytes()).hexdigest()
        return data

    def same_cancel_context(self, action, frame):
        current = self.cancel_context(frame,continuous=action.get('continuous_stop') is True
                                      and self.continuous_cancel_allowed)
        return (current is not None and self.signature(frame) == action.get('signature')
                and current == action.get('cancel_context'))

    def continuous_preparation(self, frame):
        # The blood-bar X only abandons the current fish. This separate fish
        # silhouette plus orange X stops continuous fishing before combat.
        if (self._battle(frame) or self.fishing.at_port(frame)
                or not self.fishing.cast_preparation(frame,allow_fill=True,allow_task=True)):
            return False
        scores = cv2.matchTemplate(frame[1052:1124,663:720],self.cast_stop_fish,
                                  cv2.TM_CCOEFF_NORMED,mask=self.cast_stop_mask)
        scores = np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1)
        return bool(cv2.minMaxLoc(scores)[1] >= .97 and int(self.cast_stop_mask.sum()) >= 500)

    def _stage(self, frame):
        dimmed = (self.sailing.menu(frame)
                  or self.completion_visible(frame,verify_rank=False)
                  or self._fixed(frame,'return-question',(223,556,506,629),.95))
        views = []
        for x,y,X,Y in ((40,45,68,71),(39,44,69,72)):
            crop=frame[y:Y,x:X].astype(np.int16)
            for threshold in ((80,90) if dimmed else (110,130)):
                ink=((crop[:,:,1]-crop[:,:,0]>=10) & (crop[:,:,2]-crop[:,:,0]>=15)
                     & (crop.max(axis=2)>=threshold))
                ys,xs=np.where(ink)
                if len(xs)<8:return None
                binary=np.where(ink,0,255).astype('uint8')[ys.min():ys.max()+1,xs.min():xs.max()+1]
                views.append(np.repeat(np.pad(binary,3,constant_values=255)[:,:,None],3,axis=2))
        readings=read_numeric_views(self.reader,views)
        if (any(r.value is None or not 1<=r.value<=99 or r.confidence<.95 for r in readings)
                or len({r.value for r in readings})!=1):
            return None
        return readings[0].value

    def _gate_stage(self, frame):
        original = ParallelWorld._gate_stage(self, frame)
        self.gate_probe = dict(legacy=original,views=[])
        if original is not None and original != 1:
            return original
        active = self._gate_manager(frame)
        if active is None or frame.shape != (1280,720,3):
            return None
        # The old single-digit crops clip "10". Preserve its complete outline
        # in two raw views; two inner white-ink views independently confirm it.
        # The animated portal rim is excluded from the inner binary crops.
        # No desired stage, old result or confidence reduction participates.
        dy = 0 if active else 66
        views = []
        for x,y,X,Y in ((292,377,434,475),(286,374,440,478)):
            views.append(frame[y+dy:Y+dy,x:X])
        crop=frame[375+dy:470+dy,298:426]
        for threshold in (200,220):
            ink = np.all(crop >= threshold, axis=2)
            views.append(np.repeat(np.where(ink,0,255).astype('uint8')[:,:,None],3,axis=2))
        readings = read_numeric_views(self.reader, views)
        self.gate_probe['views']=[dict(value=r.value,confidence=round(r.confidence,4)) for r in readings]
        if (len(readings) != 4 or any(r.value is None or r.confidence < .95 for r in readings)
                or len({r.value for r in readings}) != 1):
            return None
        value = readings[0].value
        return value if value == 1 and original == 1 or 10 <= value <= 99 else None

    def screen(self, frame):
        if frame.shape[:2] != (1280,720) or not self._parallel(frame):
            return None
        menu = self.sailing.menu(frame)
        completion = self.completion_visible(frame, verify_rank=False)
        if not menu and not completion and self.sailing.running(frame) is None:
            return None
        if menu and not self._off_icon(frame):
            return None
        rank, stage = self._rank_value(frame), self._stage(frame)
        if rank is None or stage is None:
            return None
        return dict(rank=rank, stage=stage, menu=menu, completion=completion,
                    route=self.sailing.running(frame) if not menu and not completion else None)

    def native_entry_recovery_context(self, frame):
        """Read-only proof that an unopened menu returned to native fishing.

        Native AUTO no longer owns a legacy training journal. Read the actual
        world, RANK and fixed location from this frame instead; neither the
        cached controller RANK nor an old verified route authorises recovery.
        This only abandons a first menu entry, never cancels fishing or credits
        a purchase, a new fishing start, or a completed portal workflow.
        """
        if (not self.enabled() or self.active or self.phase != 'idle'
                or frame.shape != (1280,720,3)
                or not self.fishing._main_visible(frame)
                or self.fishing.at_port(frame)):
            return None
        current = self.screen(frame)
        if (current is None and self._parallel(frame)
                and not self.sailing.menu(frame)
                and not self.completion_visible(frame,verify_rank=False)
                and self.sailing.running(frame) is None
                and self.sailing._fixed(frame,'active-arrow',(194,4,246,42),.96)
                and not self.sailing._fixed(frame,'inactive-face',(194,4,246,42),.96)):
            # A real AUTO letter variant falls just below the old RGB match.
            # Keep its .90 threshold and the independent .96 running arrow;
            # this extra sample only authorises read-only entry abandonment,
            # never a sailing, fishing, portal or purchase input.
            score = cv2.minMaxLoc(cv2.matchTemplate(frame[32:67,184:255],
                self.native_entry_auto_label,cv2.TM_CCOEFF_NORMED))[1]
            if np.isfinite(score) and score >= .90:
                rank,stage = self._rank_value(frame),self._stage(frame)
                if rank is not None and stage is not None:
                    current = dict(rank=rank,stage=stage,route=True,menu=False,completion=False)
        if (current is None or not 1 <= current['stage'] <= 99
                or not 0 <= current['rank'] < 40 or current['route'] is not True
                or current['menu'] or current['completion']):
            return None
        battle = (self.defense._battle_footer(frame)
                  and self.defense._match(frame,'timer',(188,125,610,165),.94)
                  and self.defense._match(frame,'damage-sword',(153,205,192,245),.96))
        success = self.training._match(frame,'success-check',(290,955,432,1060),.92)
        if not (battle or success or self.fishing.blue_fill(frame,main_known=True)):
            return None
        return (current['rank'],current['stage'],tuple(self.signature(frame)))

    def _sailing_menu_rank(self, frame):
        """The dim RANK0/1 font needs strict fixed digits, not saved defaults.

        Keep the original OCR votes and exact zero/one templates. An outlined
        digit may also use four agreeing raw foreground crops; every strong
        old vote must agree. No saved RANK participates in recognition.
        """
        if (frame.shape[:2] != (1280,720) or not self.sailing.menu(frame)
                or not self._parallel(frame)):
            return None
        views, masks = [], []
        for x,y,X,Y in ((71,9,136,59),(80,9,130,59)):
            for threshold in (110,130):
                ink = np.all(frame[y:Y,x:X] >= threshold,axis=2)
                views.append(np.repeat(np.where(ink,0,255).astype('uint8')[:,:,None],3,axis=2))
                masks.append((ink,x,X,threshold))
        readings = read_numeric_views(self.reader,views)
        self.menu_rank_probe=dict(method='white_ink',views=[dict(value=r.value,
            confidence=round(r.confidence,4)) for r in readings])
        if (all(r.value is not None and 0 <= r.value <= 40 and r.confidence >= .95 for r in readings)
                and len({r.value for r in readings}) == 1):
            return readings[0].value
        # A two-digit RANK such as 15 loses part of its leading 1 in the
        # narrow x=80 crop. Use two complete fixed crops at both thresholds;
        # retain four >=.95 agreeing votes and reject conflicting strong OCR.
        complete=[]
        for x,y,X,Y in ((71,9,136,59),(70,8,137,60)):
            for threshold in (110,130):
                ink=np.all(frame[y:Y,x:X]>=threshold,axis=2)
                complete.append(np.repeat(np.where(ink,0,255).astype('uint8')[:,:,None],3,axis=2))
        full_readings=read_numeric_views(self.reader,complete)
        self.menu_rank_probe=dict(method='complete_white_ink',views=[dict(value=r.value,
            confidence=round(r.confidence,4)) for r in full_readings])
        if (len(full_readings)==4
                and all(r.value is not None and 0 <= r.value <= 40 and r.confidence >= .95 for r in full_readings)
                and len({r.value for r in full_readings})==1
                and all(r.value in (None,full_readings[0].value) or r.confidence < .95 for r in readings)):
            return full_readings[0].value
        for digit,template,minimum in ((0,self.menu_zero,500),(1,self.menu_one,300)):
            if any(r.value not in (None,digit) for r in readings):
                continue
            valid = True
            for ink,x,X,threshold in masks:
                expected = np.all(template[:,x-71:X-71] >= threshold,axis=2)
                foreground=[]
                for mask in (ink,expected):
                    count,labels,stats,_ = cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
                    large = [i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA] >= 80]
                    if len(large) != 1:
                        valid = False
                        break
                    foreground.append(labels == large[0])
                if not valid:
                    break
                actual,reference = foreground
                union = int((actual | reference).sum())
                if union < minimum or float((actual & reference).sum())/union < .99:
                    valid = False
                    break
            if valid:
                return digit
        return self._outlined_menu_rank(frame,readings+full_readings)

    def _original(self, frame):
        return (not self._parallel(frame) and self._observatory(frame)
                and not self._map(frame) and self.sailing.running(frame) is False
                and not self._battle(frame))

    def _return_button(self, frame):
        # Optional event icons can disappear and shift this entry up one row.
        # Cover the two observed layouts with the same fixed metal-frame mask
        # and threshold. More than one qualifying entry is ambiguous.
        if frame.shape != (1280,720,3):
            self.return_probe = dict(reason='invalid_size')
            return None
        x,y,X,Y = 630,690,720,970
        scores = cv2.matchTemplate(frame[y:Y,x:X], self.return_icon,
                                   cv2.TM_CCOEFF_NORMED, mask=self.return_mask)
        scores = np.nan_to_num(scores, nan=-1, posinf=-1, neginf=-1)
        _,score,_,point = cv2.minMaxLoc(scores)
        count,_,_,_ = cv2.connectedComponentsWithStats((scores >= .96).astype('uint8'),8)
        self.return_probe = dict(score=round(float(score),6), candidates=count-1,
                                 region=(x,y,X,Y))
        if score < .96 or count != 2:
            return None
        h,w = self.return_icon.shape[:2]
        center = (x+point[0]+w//2, y+point[1]+h//2)
        self.return_probe['center'] = center
        return center

    def _report_return_wait(self, reason):
        self.hold = reason
        now = time.monotonic()
        if self.return_wait_logged_at is None or now-self.return_wait_logged_at >= 15:
            self.return_wait_logged_at = now
            self.event('次元石返回等待：'+reason)

    def _expected(self, frame, *, entry=False):
        screen = self.screen(frame)
        stage = self.journal['to_stage'] if entry else self.journal['from_stage']
        rank = self.journal['entry_rank'] if entry else self.journal['rank']
        if (entry and self.completed_rank_40 and rank == 0
                and (self.journal.get('first_cast_verified_at') or self.journal.get('selected_fishing_verified_at')
                     or self.journal.get('manual_rank_progress_observed_rank') == 1)
                and self.phase in ('route_open','route_menu','route_running')):
            minimum = 1 if self.journal.get('manual_rank_progress_observed_rank') == 1 else rank
            return screen if screen and screen['stage'] == stage and minimum <= screen['rank'] < 39 else None
        return screen if screen and (screen['stage'], screen['rank']) == (stage,rank) else None

    def _first_scope(self):
        # Both a new gate and a lowered gate can arrive at a dormant task.
        # Reuse the selected-task, one-tap workflow; neither scope permits
        # dragging, holding or cancelling a task BOSS.
        new_round = (self.completed_rank_40 and self.stage_pair(self.journal) is not None
                     and self.journal.get('entry_rank') == 0)
        lowered = (self.active and self.journal.get('reason') == 'lower'
                   and self.stage_pair(self.journal) is not None
                   and self.journal.get('verified_gate_stage') == self.journal.get('to_stage')
                   and self.journal.get('rank') == self.journal.get('entry_rank') == 39)
        return bool((new_round or lowered) and self.journal.get('entered_verified_at'))

    def _first_idle(self, frame, *, allow_running=False):
        # A single tap may start the newly selected first task; a task BOSS
        # never gains permission for meter dragging, a hold, or cancellation.
        route = self.sailing.running(frame)
        if (not self._first_scope() or frame.shape[:2] != (1280,720)
                or not self._parallel(frame) or self._map(frame)
                or not self.fishing._main_visible(frame) or not self.training.boss_mode(frame)
                or self._rank_value(frame) != self.journal['entry_rank']
                or self._stage(frame) != self.journal['to_stage']
                or route is None or not allow_running and route is not False
                or not allow_running and not self._off_icon(frame)
                or self.fishing.blue_fill(frame) or self.fishing.busy(frame)
                or self._moving(frame)
                or self._battle(frame) or self.fishing.at_port(frame)
                or self.fishing.idle_icon(frame) not in ('hook','sleep')
                or self._fixed(frame,'next-arrow',(307,64,398,127),.88)):
            return None
        box = self.fishing.box(frame)
        value = self.fishing.selected(frame,box) if box else None
        value_known = (value == 1 if self.journal['entry_rank'] == 0
                       else value is not None and 1 <= value <= 10000)
        return (self.signature(frame),tuple(box),value) if value_known else None

    def _moving(self, frame):
        if frame.shape[:2] != (1280,720):
            return False
        scores = cv2.matchTemplate(frame[38:79,283:480],self.moving_label,cv2.TM_CCOEFF_NORMED)
        score = cv2.minMaxLoc(scores)[1]
        return bool(np.isfinite(score) and score >= .90)

    def _first_unselected_idle(self, frame, *, ordinary=False, allow_running=False):
        # This state can open a real task map only. It never authorises a cast,
        # meter gesture or cancellation without a fresh actual task selection.
        route = self.sailing.running(frame) if frame.shape[:2] == (1280,720) else None
        # Lowering can re-enter RANK39 with "choose destination" and the
        # continuous cast still stopped. Select the actual current task before
        # starting a route; travel with a gray hook cannot restart fishing.
        # An ordinary, already named destination remains a new-round-only path.
        lowered = (self.journal.get('reason') == 'lower'
                   and self.stage_pair(self.journal) is not None
                   and self.journal.get('entry_rank') == 39)
        if (not self._first_scope()
                or self.journal.get('verified_gate_stage') != self.journal.get('to_stage')
                or ordinary and lowered
                or frame.shape[:2] != (1280,720)
                or not self._parallel(frame) or self._map(frame)
                or self._stage(frame) != self.journal['to_stage']
                or self._rank_value(frame) != self.journal['entry_rank']
                or route is None or not allow_running and route is not False
                or route is False and not self._off_icon(frame)
                or not self.fishing._main_visible(frame) or self.training.boss_mode(frame)
                or self._moving(frame) or self._battle(frame) or self.fishing.at_port(frame)
                or self.fishing.blue_fill(frame) or self.fishing.busy(frame)
                or self.fishing.idle_icon(frame) not in ('hook','sleep')):
            return None
        unselected = self._no_destination(frame)
        if ordinary:
            # Native rare-first sailing may choose an ordinary destination
            # before the first quest. Its RANK0 dial has a zero lower bound;
            # it must select the actual task rather than enter meter recovery.
            # This origin only authorises opening a map, never a cast/gesture.
            if unselected or int((frame[12:34,286:434].max(axis=2)<60).sum()) < 24:
                return None
        elif not unselected:
            return None
        box = self.fishing.box(frame)
        value = self.fishing.selected(frame,box) if box else None
        value_known = (value == 1 if self.journal['entry_rank'] == 0
                       else value is not None and 1 <= value <= 10000)
        return (self.signature(frame),tuple(box),value) if value_known else None

    def _first_map_origin(self, frame):
        return (self._first_idle(frame) or self._first_unselected_idle(frame)
                or self._first_unselected_idle(frame,ordinary=True))

    def _prepare_first_task(self, frame):
        now=time.time()
        self.journal.update(first_recovery=True,fishing_recovery=True,
            first_signature=self.signature(frame),
            first_deadline=min(self.journal['phase_deadline'],
                self.journal.get('first_deadline',float('inf')),now+60),
            # Allocate all travel/cast bounds once. A distant real task takes
            # minutes to reach; waiting for arrival issues no fishing input.
            # Existing records without these fields retain their old bounds.
            first_travel_deadline=min(self.journal['deadline'],now+420),
            first_cast_deadline=min(self.journal['deadline'],now+480),
            first_unselected=not self.training.boss_mode(frame))
        self._save()
        self._set('first_map',30)

    def _first_map_context(self, frame):
        if (not self._first_scope() or frame.shape[:2] != (1280,720)
                or not self._parallel(frame) or not self._map(frame) or not self._off_icon(frame)
                or self._rank_value(frame) != self.journal['entry_rank']
                or self._stage(frame) != self.journal['to_stage']):
            return False
        score = cv2.minMaxLoc(cv2.matchTemplate(frame[1136:1241,9:405],self.map_legend,cv2.TM_CCOEFF_NORMED))[1]
        return bool(np.isfinite(score) and score >= .93)

    def _first_map_ready(self, frame):
        return bool(self._first_map_context(frame)
                    and self.signature(frame) == self.journal.get('first_signature'))

    def _first_selected_map(self, frame):
        issued = {a.get('kind') for a in self.journal.get('issued_inputs',[])}
        self.first_selected_map_reason = '缺少对应的选点记录'
        if (not self.journal.get('first_unselected')
                or self.journal.get('first_selected_rank') != self.journal['entry_rank']
                or 'first-select' not in issued):
            return False
        self.first_selected_map_reason = '当前地图/世界/RANK/停航未确认'
        if not self._first_map_context(frame):
            return False
        self.first_selected_map_reason = '缺少已选任务标记'
        if not self.training.boss_mode(frame):
            return False
        target = self._map_task_target(frame,selected_point=self.journal.get('first_selected_point'))
        self.first_selected_map_reason = '当前露出骷髅未识别' if not target else '当前骷髅位置与选点不符'
        valid = bool(target and list(target[:2]) == self.journal.get('first_selected_point'))
        if valid:
            self.first_selected_map_reason = '已核对当前选中任务'
        return valid

    def _map_task_target(self, frame, *, selected_point=None):
        if selected_point is None:
            target = ParallelWorld._map_task_target(self,frame)
            if target is not None:
                return target
        unselected_first = (self.journal.get('first_unselected')
                            and self._first_scope()
                            and self._no_destination(frame) and self._first_map_ready(frame))
        if (not self._first_map_context(frame)
                or not self.training.boss_mode(frame) and not unselected_first):
            return None
        # The current avatar hides the left skull. This real first-map sample
        # stores only its exposed right eye/outline and red lower jaw; every
        # other pixel is zero and excluded. Neither an avatar nor old map
        # coordinates participate in locating the selected task.
        lowered = self.journal['entry_rank'] == 39 and self.journal.get('reason') == 'lower'
        template = self.lower_map_core if lowered else self.first_map_core
        templates = [template]
        if not lowered and selected_point is not None:
            # A second real RANK0 raster variant is used only to re-find an
            # already receipted selection. It cannot select another task.
            templates.append(self.selected_first_map_core)
        # The RANK39 portrait covers more of the left skull. Its exposed
        # right crescent starts at x19; neither sample includes the portrait.
        parts = ((slice(8,27),slice(19,31)) if lowered else (slice(10,27),slice(23,31)),
                 (slice(27,33),slice(4,29)))
        variants = [(item,parts) for item in templates]
        if lowered and self.journal.get('first_unselected') and (
                unselected_first or selected_point is not None):
            # The avatar can cover the right eye as well. The lower bone/teeth
            # and red jaw remain separate fixed shapes in the real RANK39 map.
            # Both must match independently; no portrait pixels are included.
            variants.append((self.lower_map_bone,(
                (slice(22,27),slice(4,29)),(slice(27,33),slice(4,29)))))
        area = (20,175,700,1115)
        if selected_point is not None:
            # Closing an already selected map only needs to re-find that
            # skull. Search new pixels around the receipted selection instead
            # of scanning the entire map twice on every confirmation frame.
            # This never selects/navigates using saved coordinates, and the
            # caller still requires the freshly located centre to be equal.
            if (not isinstance(selected_point,(tuple,list)) or len(selected_point)!=2
                    or any(type(v) is not int for v in selected_point)):
                return None
            x,y=selected_point
            if not 20<=x<700 or not 175<=y<1115:
                return None
            area=(max(20,x-40),max(175,y-40),min(700,x+40),min(1115,y+40))
        left,top,right,bottom=area
        for template,parts in variants:
            mask = np.zeros(template.shape[:2],np.uint8)
            for r,c in parts:
                mask[r,c] = 1
            scores = cv2.matchTemplate(frame[top:bottom,left:right],template,
                                      cv2.TM_CCOEFF_NORMED,mask=mask)
            scores = np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1)
            _,score,_,(a,b) = cv2.minMaxLoc(scores)
            if score < .97:
                continue
            x,y = a+left,b+top
            h,w = template.shape[:2]
            crop = frame[y:y+h,x:x+w]
            matched = [cv2.matchTemplate(crop[r,c],template[r,c],cv2.TM_CCOEFF_NORMED)[0,0]
                       for r,c in parts]
            if any(not np.isfinite(value) or value < .97 for value in matched):
                continue
            rgb = crop.astype(np.int16)
            red = (rgb[:,:,0]>=130)&(rgb[:,:,0]>rgb[:,:,1]*1.4)&(rgb[:,:,0]>rgb[:,:,2]*1.4)
            if int((red & (mask>0)).sum()) < 45 or int(((crop.min(axis=2)>=170) & (mask>0)).sum()) < 40:
                continue
            return x+w//2,y+h//2,float(score)
        return None

    def _selected_first_fishing(self, frame):
        """Observe native fishing after closing the actual selected first task.

        This proof never substitutes a cast receipt. It only avoids tapping an
        already fishing game and requires the issued selection and map close.
        """
        issued = {a.get('kind') for a in self.journal.get('issued_inputs',[])}
        saved = self.journal.get('first_signature')
        if (not self.input_allowed() or not self._first_scope() or not self.completed_rank_40
                or self.journal.get('first_selected_rank') != 0
                or not {'first-select','first-close'} <= issued or 'first-cast' in issued
                or not saved or frame.shape != (1280,720,3) or not self._parallel(frame)
                or not self._off_icon(frame) or self._map(frame) or self.sailing.menu(frame)
                or self.sailing.running(frame) is not False
                or not self.fishing._main_visible(frame) or self.fishing.at_port(frame)
                or self._moving(frame) or self.signature(frame)[-1] != saved[-1]):
            return None
        screen = self.screen(frame)
        if (not screen or screen['stage'] != self.journal['to_stage'] or screen['rank'] not in (0,1)
                or screen['menu'] or screen['route'] is not False):
            return None
        battle = (self.defense._match(frame,'timer',(188,125,610,165),.94)
                  and self.defense._match(frame,'swords',(307,962,415,1057),.96)
                  and self.defense._match(frame,'damage-sword',(153,205,192,245),.96)
                  and self.defense._battle_footer(frame))
        return screen if self.fishing.blue_fill(frame,main_known=True) or battle else None

    def _first_step(self, frame):
        if time.time() > self._first_phase_limit(self.phase):
            raise RuntimeError('任务抛竿恢复超过60秒，保留任务确认与输入次数；'+self._first_close_diagnostic())
        phase = self.phase
        if phase == 'first_map':
            if self._first_map_ready(frame):
                self._set('first_select')
            elif self._first_map_origin(frame) is not None:
                self._click('first-map',350,30,'任务抛竿恢复：打开真实地图核对当前任务',frame)
            return True
        if phase == 'first_select':
            ready = self._first_map_ready(frame)
            target = self._map_task_target(frame) if ready else None
            point = tuple(target[:2]) if target else None
            reason = ('等待同世界、阶段、RANK、地点和停航地图' if not ready else
                      '当前地图未识别完整骷髅或露出的骨面和红色下颌' if not point else
                      '已定位当前骷髅，等待连续新图确认')
            self.first_select_wait_reason = reason
            self.hold = reason
            now = time.monotonic()
            if now-getattr(self,'first_select_logged_at',float('-inf')) >= 5:
                self.first_select_logged_at = now
                remaining = min(self.journal['deadline'],self.journal['phase_deadline'],
                                self._first_phase_limit(phase))-time.time()
                self.event(f'任务选点等待：{reason}；剩余{max(0,remaining):.1f}秒')
            if ready and point is None:
                used=self.journal.get('attempts',{}).get('first-pan',0)
                if used < self.INPUT_LIMITS['first-pan'] and self._confirmed(('first-map-search',self.signature(frame))):
                    # The last resource point may be far from the new quest.
                    # RANK39's final map extension is west; the first quest is
                    # east. Only pan a newly verified stopped map. Never use
                    # an old BOSS coordinate or renew the first-task deadline.
                    west=self.journal['entry_rank']==39
                    x,X=(120,610) if west else (600,120)
                    self._click('first-pan',x,600,'任务不在当前视野，向'+('西' if west else '东')+'搜索地图',
                                frame,swipe=[X,600,400])
                return True
            if self._confirmed(('first-task',point,self.signature(frame)) if point else None):
                self.first_map_target, self.first_map_verified = point,True
                if self._click('first-select',*point,'任务抛竿恢复：重新选择已核对的骷髅任务',frame):
                    self.journal['first_selected_rank'] = self.journal['entry_rank']
                    self.journal['first_selected_point'] = list(point)
                    self._save()
                    self._set('first_close',max(0,self.journal.get('first_travel_deadline',
                              self.journal['first_deadline'])-time.time()))
            return True
        if phase == 'first_close':
            self.report_first_close_wait('核对已选任务地图、停止AUTO及当前世界/RANK')
            if self.journal.get('first_unselected'):
                # Selecting the first destination intentionally changes the
                # header. Rebind it only to the issued selection and the same
                # fresh skull, world/RANK, stopped map and fixed legend.
                if self._map(frame):
                    ready = self._first_selected_map(frame)
                    if self._confirmed(('selected-first-map',self.signature(frame)) if ready else None):
                        self.journal['first_signature'] = self.signature(frame)
                        self._save()
                        clicked = self._click('first-close',668,1187,'任务抛竿恢复：关闭已重新选择的任务地图',frame)
                        self.report_first_close_wait('已发送关图，等待真实主画面' if clicked else
                            '关图输入前的新截图守卫拒绝：世界/RANK/停航/已选任务/截止之一未满足')
                    if not ready and time.monotonic()-getattr(self,'first_map_wait_logged_at',float('-inf')) >= 5:
                        self.first_map_wait_logged_at=time.monotonic()
                        self.first_close_wait_reason = self.first_selected_map_reason
                        remaining=min(self.journal['deadline'],self.journal['phase_deadline'],
                                      self.journal['first_deadline'])-time.time()
                        self.event(f'首任务关图等待：{self.first_selected_map_reason}，'
                                   f'剩余{max(0,remaining):.1f}秒；不追加选点或抛竿')
                else:
                    self.report_first_close_wait('地图未识别；等待已选地点的真实钓鱼、停航空闲或到达证明')
                    issued = {a.get('kind') for a in self.journal.get('issued_inputs',[])}
                    fishing = self._selected_first_fishing(frame)
                    if fishing:
                        if getattr(self,'selected_fishing_frame',None) is frame:
                            return True
                        self.selected_fishing_frame=frame
                        if self._confirmed(('selected-first-fishing',self.journal['to_stage'],self.signature(frame)[-1])):
                            self.check_deadline()
                            self.journal.update(selected_fishing_verified_at=time.time(),
                                selected_fishing_verified_rank=fishing['rank'])
                            self.rank,self.current_stage=fishing['rank'],self.journal['to_stage']
                            self._save()
                            self.event('首任务地图已关闭，连续新图确认游戏已在钓鱼；不追加首竿输入，继续核对航巡')
                            self._set('route_open')
                        return True
                    if ({'first-select','first-close'} <= issued and self._moving(frame)
                            and self._parallel(frame) and self._off_icon(frame)
                            and self.signature(frame) == self.journal.get('first_signature')):
                        now=time.monotonic()
                        if now-getattr(self,'first_travel_logged_at',float('-inf')) >= 15:
                            self.first_travel_logged_at=now
                            self.event('首任务地图已关闭，正在前往已选任务；只读等待，不重置阶段截止或输入次数')
                    key = self._first_idle(frame) if {'first-select','first-close'} <= issued else None
                    if self._confirmed(('selected-first-main',key) if key is not None else None):
                        self.journal['first_signature'] = self.signature(frame)
                        self._save()
                        self.first_idle_key = None
                        self._set('first_idle',15)
                return True
            if self._first_map_ready(frame):
                clicked = self._click('first-close',668,1187,'任务抛竿恢复：关闭已重新选择的任务地图',frame)
                self.report_first_close_wait('已发送关图，等待真实主画面' if clicked else '关图前新图守卫拒绝输入')
            elif self._first_idle(frame) is not None:
                self.first_idle_key = None
                self._set('first_idle',15)
            return True
        if phase == 'first_idle':
            key = self._first_idle(frame)
            if not self._confirmed(('first-idle',key) if key is not None else None):
                return True
            self.first_idle_key = key
            icon = self.fishing.idle_icon(frame)
            kind = 'first-wake' if icon == 'sleep' else 'first-cast'
            if self._click(kind,360,1010,'任务抛竿恢复：单次唤醒' if icon == 'sleep' else '任务抛竿恢复：单次抛竿，等待真实钓鱼',frame):
                self.first_idle_key = None
                self.key = None
                if kind == 'first-cast':
                    # A cast actually issued inside its fixed input window
                    # gets its own read-only 15s
                    # observation, still capped by the original overall limit.
                    # Old journals never acquire this on restart or timeout.
                    self.journal['first_observation_deadline'] = min(
                        self.journal['deadline'], time.time()+15)
                    self._set('first_verify',15)
                else:
                    # Start the existing 15-second wake verification at the
                    # actual wake input. The total first-task deadline stays
                    # unchanged and a second wake remains forbidden.
                    self.journal['phase_deadline']=min(self.journal['deadline'],
                        self._first_phase_limit('first_idle'),time.time()+15)
                    self._save()
            return True
        if getattr(self,'first_preobserved_frame',None) is not frame:
            self.observe_first_cast(frame)
        return True

    def _first_result_screen(self, frame):
        # Only actual battle, success or the next RANK proves a cast worked.
        screen = self.screen(frame)
        success = self.training._match(frame,'success-check',(290,955,432,1060),.92)
        battle = (self.training.boss_mode(frame)
                  and self.defense._match(frame,'timer',(188,125,610,165),.94)
                  and self.defense._match(frame,'swords',(307,962,415,1057),.96)
                  and self.defense._match(frame,'damage-sword',(153,205,192,245),.96)
                  and self.defense._battle_footer(frame))
        issued = any(a['kind'] == 'first-cast' for a in self.journal.get('issued_inputs',[]))
        entry_rank = self.journal['entry_rank']
        proof = bool(issued and screen and screen['stage'] == self.journal['to_stage']
                     and not screen['menu'] and self._off_icon(frame)
                     and (screen['route'] is False or screen['rank'] == 40 and screen['completion'])
                     and (screen['rank'] == entry_rank+1
                          or screen['rank'] == entry_rank and (battle or success)
                          and self.signature(frame) == self.journal.get('first_signature')))
        self.first_result_checks = dict(
            scene=(screen['stage'], screen['rank']) if screen else None,
            stopped=self._off_icon(frame), battle=bool(battle), success=bool(success),
            cast_issued=issued, result_verified=proof)
        return screen if proof else None

    @property
    def first_result_observation_pending(self):
        # After a real result vote, preserve time for the remaining read-only
        # votes instead of spending it on a four-second optional combat burst.
        # This authorizes no input and changes none of the saved deadlines.
        return (self.phase == 'first_verify' and self.input_allowed()
                and self.seen > 0 and self.first_result_rank is not None)

    def observe_first_cast(self, frame):
        """Read-only result votes run before combat can own the worker loop."""
        if self.phase != 'first_verify' or not self.input_allowed():
            return False
        if getattr(self,'first_result_frame',None) is frame:
            return False
        self.first_result_frame = frame
        screen = self._first_result_screen(frame)
        context = self._first_result_context(frame, screen)
        if not self.input_allowed():
            return False  # OCR cannot commit after the original deadline.
        previous_rank = getattr(self,'first_result_rank',None)
        # A real battle/success/rank advance proves the issued cast worked.
        # Then confirm the same fresh stopped context for three frames/seconds;
        # the brief battle animation need not remain on every observation.
        # Without that real result, context alone supplies no votes. Unknown or
        # changed identity invalidates both proof and the consecutive context.
        if (context is None or previous_rank is not None and context[1] < previous_rank):
            self.first_result_rank = None
            self.first_result_observed = False
            confirmed = self._confirmed(None)
        else:
            self.first_result_rank = context[1]
            key = ('first-proof',context[0],self.journal['entry_rank'],context[2])
            if screen is not None:
                self.first_result_observed = True
            if self.first_result_observed:
                confirmed = self._confirmed(key)
            else:
                confirmed = False
                self._confirmed(None)
        # Keep bounded result evidence in the normal log. A timeout alone must
        # not be diagnosed as a missed cast or justify another input.
        observation = (screen['stage'],screen['rank']) if screen else None
        now = time.monotonic()
        if (not hasattr(self,'first_observation_logged_at')
                or observation != getattr(self,'first_observation_logged',None)
                or now-self.first_observation_logged_at >= 3 or confirmed):
            elapsed = max(0,now-self.since) if self.since is not None else 0
            remaining = min(self.journal['deadline'],self.journal['phase_deadline'],
                            self._first_phase_limit(self.phase))-time.time()
            self.event(f'首竿只读验证：结果={observation}，有效结果新帧={self.seen}，'
                       f'跨度={elapsed:.2f}秒，剩余={max(0,remaining):.2f}秒，'
                       f'当前上下文={context[:2] if context else None}，'
                       f'上下文检查={self.first_result_context_reason}，'
                       f'结果检查={getattr(self,"first_result_checks",None)}；不追加抛竿')
            self.first_observation_logged,self.first_observation_logged_at=observation,now
        if confirmed:
            self.journal.update(first_cast_verified_at=time.time(),first_cast_verified_rank=context[1])
            self._save()
            self._set('route_running' if context[1] == 40 else 'route_open')
            return True
        return False

    def _first_result_context(self, frame, screen):
        """Fresh identity for read-only votes; never authorises an input."""
        saved = self.journal.get('first_signature')
        self.first_result_context_reason = 'shape_or_saved_task'
        if frame.shape != (1280,720,3) or not saved:
            return None
        for reason,valid in (('world',self._parallel(frame)),
                             ('stopped_auto',self._off_icon(frame)),
                             ('map',not self._map(frame)),
                             ('settings',not self.sailing.menu(frame))):
            if not valid:
                self.first_result_context_reason = reason
                return None
        # This is read-only cast confirmation. The adaptive dial checkbox is
        # animated and is not evidence of whether the already-issued cast
        # worked. Keep the two fixed, bright currency controls and every world,
        # stopped-route, map/menu, numeric and location guard. Meter/input
        # guards continue using their stricter adaptive-AUTO check.
        completion = bool(screen and screen['rank'] == 40 and screen['completion'])
        main = (all(self.training._match(frame,name,box,.91) for name,box in
                    (('guard-star',(208,1137,269,1192)),
                     ('guard-diamond',(539,1137,600,1186))))
                and float((frame[1147:1168,558:585].max(axis=2)>=210).mean())>=.30)
        if not completion and not main:
            self.first_result_context_reason = 'bright_main'
            return None
        rank = screen['rank'] if screen else self._rank_value(frame)
        stage = screen['stage'] if screen else self._stage(frame)
        location = self.signature(frame)[-1]
        for reason,valid in (('stage',stage == self.journal['to_stage']),
                ('rank',rank in (self.journal['entry_rank'],self.journal['entry_rank']+1)),
                ('location',location == saved[-1])):
            if not valid:
                self.first_result_context_reason = reason
                return None
        self.first_result_context_reason = 'verified'
        return stage,rank,location

    def _popup_resume_key(self, frame, *, notice_gone):
        """The idle screen key only when this controller owns the resume.

        A fresh popup closes onto the ordinary main screen, so "notice gone"
        alone cannot tell a native resume apart from any other return.  Only
        the freshly selected, woken task that has issued no cast yet may claim
        it, and the screen must still show the same signature and hook.
        """
        issued = {a.get('kind') for a in self.journal.get('issued_inputs', [])}
        eligible = (notice_gone and self.phase == 'first_idle' and self.input_allowed()
                    and self.journal.get('first_selected_rank') == self.journal.get('entry_rank')
                    and {'first-select', 'first-wake'} <= issued
                    and self.journal.get('attempts', {}).get('first-wake') == 1
                    and not self.journal.get('attempts', {}).get('first-cast'))
        if not eligible:
            return None
        key = self._first_idle(frame)
        if (key is None or self.fishing.idle_icon(frame) != 'hook'
                or self.signature(frame) != self.journal.get('first_signature')):
            return None
        return key

    def popup_return_pending(self, frame, *, notice_gone):
        """True while a native resume is observable but not yet proven.

        The runner asks this before letting the generic return rule close the
        popup stage: that rule only proves the screen came back, while the
        first cast still needs the consecutive gray-hook proof collected in
        ``observe_task_popup_return``.  Leaving the stage early would throw
        that proof away and the resumed cast would be rejected.  Pure read --
        it never advances the confirmation counter.
        """
        key = self._popup_resume_key(frame, notice_gone=notice_gone)
        return key is not None and self.first_idle_key != key

    def observe_task_popup_return(self, frame, *, notice_gone):
        """Only the freshly selected, woken task can resume an NPC close.

        Share the consecutive gray-hook proof with first_idle, but issue no
        input, renew no timer and credit no fishing or portal completion.
        """
        key = self._popup_resume_key(frame, notice_gone=notice_gone)
        if key is None:
            return self._confirmed(None)
        confirmed = self._confirmed(('first-idle', key))
        if confirmed:
            self.first_idle_key = key
        return confirmed

    def _first_phase_limit(self, phase):
        if phase == 'first_verify':
            return self.journal.get('first_observation_deadline',
                                    self.journal.get('first_deadline',0))
        if phase == 'first_close':
            return self.journal.get('first_travel_deadline',self.journal.get('first_deadline',0))
        if phase == 'first_idle':
            return self.journal.get('first_cast_deadline',self.journal.get('first_deadline',0))
        return self.journal.get('first_deadline',0)

    def input_allowed(self):
        return (self.active and self.enabled() and time.time() <= self.journal['deadline']
                and time.time() <= self.journal['phase_deadline']
                and (not self.phase.startswith('first_')
                     or time.time() <= self._first_phase_limit(self.phase)))

    def check_deadline(self):
        """Combat cannot hide an expired active handoff from its controller."""
        if self.active and self.retry is None and not self.input_allowed():
            # Identity alone is not permission to continue: the runner must
            # independently verify safe release and archive the failed journal.
            raise RetryableStage('次元石降阶流程到期或功能关闭，保留交接记录'
                f'（阶段={self.phase}，阶段截止={self.journal["phase_deadline"]:.3f}，总截止={self.journal["deadline"]:.3f}）；'
                + ('返回步骤；最近等待：'+str(self.hold) if self.phase in ('return','return_confirm')
                   else self._first_close_diagnostic()), task='stone_farm', retry_seconds=1800)

    def _first_close_diagnostic(self):
        if self.phase=='manager':
            return ('传送门管理页；最近阶段复核：'+str(getattr(self,'gate_probe',None))
                    +'；保留已用次数，不猜阶段或续期')
        if self.phase == 'first_select':
            return ('选择当前任务；最近等待：'
                    + (getattr(self,'first_select_wait_reason',None) or '本次尚未取得有效新图')
                    + '；保留原输入次数，不续期')
        if self.phase != 'first_close':
            return '保留原输入次数，不续期'
        used = self.journal.get('attempts',{}).get('first-close',0)
        issued = sum(a.get('kind') == 'first-close' for a in self.journal.get('issued_inputs',[]))
        return (f'关闭首任务地图（不是关次元门），关图尝试{used}/3、已发回执{issued}；最近等待：'
                + (getattr(self,'first_close_wait_reason',None) or '本次尚未取得有效新图'))

    def report_first_close_wait(self, reason):
        self.first_close_wait_reason = reason
        self.hold = reason
        now = time.monotonic()
        if now-getattr(self,'first_close_report_at',float('-inf')) >= 5:
            self.first_close_report_at = now
            remaining = min(self.journal['deadline'], self.journal['phase_deadline'],
                            self.journal.get('first_deadline',0))-time.time()
            self.event(self._first_close_diagnostic()+f'；剩余{max(0,remaining):.1f}秒')

    def meter_frame_allowed(self, frame):
        # Actual entry RANK/stage plus fixed strokes bind every gesture to the
        # stopped ordinary location. No boss, port, battle or fill authorises it.
        return bool(self.input_allowed() and not self.journal.get('manual_rank_progress_runs')
            and self.phase == 'fishing_prepare'
            and self.off_verified and frame.shape[:2] == (1280,720)
            and self._parallel(frame) and self._off_icon(frame)
            and self.sailing.running(frame) is False
            and self.fishing._main_visible(frame) and not self.training.boss_mode(frame)
            and not self._no_destination(frame)
            and not self.fishing.at_port(frame) and not self._battle(frame)
            and not self.fishing.blue_fill(frame) and not self.fishing.busy(frame)
            and self.fishing.idle_icon(frame) in ('hook','sleep')
            and self.signature(frame) == self.journal.get('restore_signature')
            and self.fishing._context(frame) == self.meter_context)

    def reserve_meter_input(self, action):
        if self.journal.get('manual_rank_progress_runs'):
            return False
        if not self.input_allowed() or self.phase != 'fishing_prepare':
            return False
        if self.journal.get('post_start_restore_runs'):
            return False  # This recovery only observes the already running fishing.
        if action not in ('drag','reveal','start'):
            return False  # Restoration never cancels existing fishing.
        counts = self.journal.setdefault('meter_attempts',{})
        used = counts.get('start',0) if action == 'start' else counts.get('drag',0)+counts.get('reveal',0)
        if used >= (1 if action == 'start' else 4):
            raise RuntimeError('降阶后的钓鱼恢复已到输入上限，保留记录且不重复长按')
        counts[action] = counts.get(action,0)+1
        self._save()
        return True

    def _prepare_fishing(self, frame):
        self.meter_context = self.fishing._context(frame)
        self.journal['restore_signature'] = self.signature(frame)
        self.training.state = dict(mode='farming',rank=self.journal['entry_rank'],sailing_enabled=False,
                                   sailing_sleep_recovery=True)
        self.fishing.idle_since = None
        self._save()
        self._set('fishing_prepare',90)

    def _route_action(self, kind):
        return 'restore-'+kind if self.journal.get('fishing_recovery') else kind

    def reserve_input(self, action):
        if not self.input_allowed():
            return False
        name = action['kind']
        counts = self.journal.setdefault('attempts', {})
        if name not in self.INPUT_LIMITS or counts.get(name, 0) >= self.INPUT_LIMITS[name]:
            raise RuntimeError('次元石降阶操作达到上限，已停止并保留交接记录：'+name)
        counts[name] = counts.get(name, 0)+1
        self.journal.setdefault('inputs', []).append(dict(kind=name, time=time.time(),
            phase=self.phase, x=action['x'], y=action['y']))
        self._save()
        return True

    def cruise_priority(self, frame):
        # Keep the actual destination priority; no target or auto checkbox in this key.
        import hashlib
        parts=[]
        for x,X in ((260,415),(434,594)):
            crop=frame[520:574,x:X]
            parts.append((crop.max(axis=2)<70).tobytes())
            hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
            gold=(hsv[:,:,0]>=10)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>100)&(hsv[:,:,2]>180)
            parts.append(bytes([int(gold.mean()>.25)]))
        return hashlib.sha256(b''.join(parts)).hexdigest()

    def cruise_values_match(self, frame):
        policy=getattr(self,'cruise_policy',None)
        return bool(policy and all(self.sailing.selected(frame,n) is v for n,v in policy['targets'].items())
                    and self.sailing.sale_selected(frame) is policy['auto_sale']
                    and self.sailing.upgrade_selected(frame) is policy['auto_upgrade'])

    def input_issued(self, action, metadata):
        # Reservations are conservative attempts, never proof of an input.
        self.journal.setdefault('issued_inputs', []).append(dict(
            kind=action['kind'], input_id=metadata['input_id'], frame_id=metadata['frame_id'],
            time=metadata['input_begin_time'], transport=metadata['input_transport']))
        self._save()

    def _click(self, kind, x, y, label, frame, **values):
        action = dict(kind=kind, x=x, y=y, label='次元石：'+label,
                      signature=self.signature(frame), phase=self.phase, **values)
        result = self.action(action)
        self.next_at = time.monotonic()+1.5
        return result is not False

    def _begin(self, screen):
        lower = screen['rank'] == 39 and 2 <= screen['stage'] <= 99
        if not lower and not (screen['rank'] == 40 and 1 <= screen['stage'] < 99):
            raise ValueError('次元石交接须有实际RANK39降阶或RANK40通关阶段')
        now = time.time()
        self.journal = dict(version=1, pending=True, reason='lower' if lower else 'cycle',
            rank=screen['rank'], from_stage=screen['stage'],
            to_stage=screen['stage']-1 if lower else screen['stage']+1,
            entry_rank=39 if lower else 0, started_at=now, deadline=now+self.LIMIT,
            phase='completion' if screen['completion'] else 'stop', phase_deadline=now+180,
            return_deadline=now+180, attempts={}, inputs=[],
            allow_ordinary_cancel=True,continuous_cast_stop=True,
            rank40_verified=not lower, gate_closed=False,
            cycles=self.journal.get('cycles',0), handoffs=self.journal.get('handoffs',0),
            recoveries=self.journal.get('recoveries',0))
        self.START_STAGE, self.FINAL_STAGE = self.stage_pair(self.journal)
        self.rank = screen['rank']
        self.phase = self.journal['phase']
        self.off_verified = False
        self.fishing_stopped = False
        self.boss_abandon_key = None
        self.cancel_anchor = None
        self.key = None
        self._save()
        self.event(f'次元石：第{self.START_STAGE}阶段已到 RANK39，先停航返回并降至第{self.FINAL_STAGE}阶段，保留本轮进度' if lower
                   else f'次元石：第{self.FINAL_STAGE}阶段已连续确认 RANK40，先停航返回，关闭旧门后重开第{self.START_STAGE}阶段')

    def validate_portal(self, action, frame):
        """Every input is authorised by a fresh frame plus the saved budget."""
        if (not self.input_allowed() or frame.shape[:2] != (1280,720)
                or action.get('phase') != self.phase
                or action.get('signature') != self.signature(frame)):
            return False
        kind, point = action['kind'], (action['x'], action['y'])
        if kind == 'completion':
            return (self.phase == 'completion' and self.completed_rank_40
                    and self.completion_visible(frame, verify_rank=False) and point == (360,840))
        if kind in ('stop','restore-stop'):
            # The outer observation establishes actual stage/rank; fixed strokes
            # bind that evidence to this new fast input frame.
            return (self.phase == ('fishing_stop' if kind == 'restore-stop' else 'stop')
                    and self._parallel(frame) and self.sailing.running(frame) is True and point == (233,45))
        if kind == 'handoff-cancel':
            continuous = action.get('continuous_stop') is True and self.continuous_cancel_allowed
            return (self.phase == 'stop_fishing' and self.ordinary_cancel_allowed and self.off_verified
                    and time.time() <= self.journal['return_deadline'] and point == (690,1090)
                    and self.same_cancel_context(action,frame)
                    and (self.continuous_preparation(frame) if continuous
                         else self.fishing.ordinary_preparation(frame,allow_fill=True)))
        if kind in ('lower-boss-abandon','lower-boss-abandon-menu'):
            current = self.boss_abandon_context(frame)
            return (self.boss_abandon_handoff and self.phase == 'stop_fishing'
                    and self.journal.get('allow_boss_abandon') is True and self.off_verified
                    and time.time() <= self.journal['return_deadline']
                    and kind == self.boss_abandon_kind and not self.journal.get('attempts',{}).get(kind)
                    and current is not None and current == self.boss_abandon_key
                    and current == action.get('abandon_context') and point == current[1])
        if kind.startswith('first-'):
            if (not self._first_scope() or not self.off_verified
                    or time.time() > self._first_phase_limit(self.phase)
                    or self.signature(frame) != self.journal.get('first_signature')):
                return False
            if kind == 'first-map':
                return self.phase == 'first_map' and point == (350,30) and self._first_map_origin(frame) is not None
            if kind == 'first-pan':
                west=self.journal['entry_rank']==39
                return (self.phase == 'first_select' and self._first_map_ready(frame)
                        and self._map_task_target(frame) is None
                        and point == ((120,600) if west else (600,600))
                        and action.get('swipe') == ([610,600,400] if west else [120,600,400]))
            if kind == 'first-select':
                target = self._map_task_target(frame) if self._first_map_ready(frame) else None
                return (self.phase == 'first_select' and self.first_map_verified
                        and target is not None and tuple(target[:2]) == point == self.first_map_target)
            selected = (self.journal.get('first_selected_rank') == self.journal['entry_rank']
                        and any(a['kind'] == 'first-select' for a in self.journal.get('issued_inputs',[])))
            if kind == 'first-close':
                return (self.phase == 'first_close' and selected and point == (668,1187)
                        and self._first_map_ready(frame)
                        and (not self.journal.get('first_unselected') or self._first_selected_map(frame)))
            if kind in ('first-wake','first-cast'):
                return (self.phase == 'first_idle' and selected and point == (360,1010)
                        and self.first_idle_key is not None and self._first_idle(frame) == self.first_idle_key
                        and self._confirmed(('first-idle',self.first_idle_key))
                        and self.fishing.idle_icon(frame) == ('sleep' if kind == 'first-wake' else 'hook'))
            return False
        if kind.startswith('return-'):
            if (self.phase not in ('return','return_confirm') or not self.off_verified
                    or not self.fishing_stopped
                    or time.time() > self.journal['return_deadline']
                    or not self._parallel(frame) or self._map(frame) or not self._off_icon(frame)):
                return False
            if kind == 'return-confirm':
                return (self.phase == 'return_confirm' and point == (208,788)
                        and (self.fishing.idle_icon(frame,main_known=True) is not None
                             or self.fishing.at_port(frame,main_known=True))
                        and not self.fishing.blue_fill(frame,main_known=True) and not self._battle(frame)
                        and self._fixed(frame, 'return-question', (223,556,506,629), .95)
                        and self._fixed(frame, 'continue', (63,741,356,834), .96)
                        and float((frame[480:510,150:180].max(axis=2)>=210).mean()) >= .8)
            return (self.phase == 'return' and self.sailing.running(frame) is False
                    and self._idle_fishing(frame) and self._return_button(frame) == point)
        # All original-world input requires an explicit stopped header, even
        # when a modal dims the underlying lower AUTO checkbox.
        if kind in ('route-open','route-start','restore-route-open','restore-route-start',native_auto_sale.ACTION,
                    'cruise-auto-sale','cruise-auto-upgrade') or kind.startswith('target:'):
            if not self.off_verified or not self._parallel(frame) or not self._off_icon(frame):
                return False
            if kind in ('route-open','restore-route-open'):
                return self.phase == 'route_open' and self.sailing.running(frame) is False and point == (233,45)
            settings = self.settings_key(frame)
            policy=getattr(self,'cruise_policy',None)
            if policy is not None:
                if (self.phase!='route_menu' or not settings
                        or self.cruise_priority(frame)!=self.journal.get('cruise_priority')):
                    return False
                if kind in ('route-start','restore-route-start'):
                    return point==(360,873) and settings==self.menu_key and self.cruise_values_match(frame)
                if kind=='cruise-auto-sale':
                    actual=self.sailing.sale_selected(frame)
                    return point==(576,647) and actual is not None and actual is not policy['auto_sale']
                if kind=='cruise-auto-upgrade':
                    actual=self.sailing.upgrade_selected(frame)
                    return point==(576,761) and actual is not None and actual is not policy['auto_upgrade']
                if kind.startswith('target:'):
                    name=kind.split(':',1)[1]
                    actual=self.sailing.selected(frame,name) if name in TARGETS else None
                    return (name in TARGETS and point==(TARGETS[name][0],458) and actual is not None
                            and actual is not policy['targets'][name])
                return False
            if not settings or settings[1] != self.journal.get('native_extras'):
                return False
            if kind == native_auto_sale.ACTION:
                return (self.phase == 'route_menu' and point == (576,647)
                        and native_auto_sale.valid_off(self,frame,settings))
            if kind in ('route-start','restore-route-start'):
                return (self.phase == 'route_menu' and point == (360,873)
                        and settings == self.menu_key and settings[0] == self.TARGET_VALUES
                        and self.sailing.sale_selected(frame) is True)
            name = kind.split(':',1)[1]
            return (self.phase == 'route_menu' and name in TARGETS
                    and point == (TARGETS[name][0],458)
                    and self.sailing.selected(frame,name) is not self.TARGET_VALUES[list(TARGETS).index(name)])
        if self._parallel(frame) or not self._observatory(frame) or not self._off_icon(frame):
            return False
        if kind == 'npc':
            return self.phase == 'npc' and self._npc_ready(frame) and point == (482,548)
        if kind == 'lower':
            return (self.phase == 'manager' and self.journal['reason'] == 'lower'
                    and not self.completed_rank_40 and self._gate_manager(frame) is True
                    and self._gate_stage(frame) == self.journal['from_stage'] and point == (357,873))
        if kind == 'lower-confirm':
            return (self.phase == 'lower_confirm' and self.journal['reason'] == 'lower'
                    and self._lower_prompt(frame) == self.journal['to_stage'] and point == (208,788))
        if kind in ('close-gate','close-confirm'):
            if not self.completed_rank_40 or self.stage_pair(self.journal) is None:
                return False
            if kind == 'close-gate':
                return (self.phase == 'manager' and self._gate_manager(frame) is True
                        and self._gate_stage(frame) == self.journal['from_stage'] and point == (354,739))
            return (self.phase == 'close_confirm' and point == (208,788)
                    and bool(self.journal.get('attempts',{}).get('close-gate'))
                    and self._close_question(frame))
        if kind == 'activate':
            return (self.phase == 'closed' and self.completed_rank_40
                    and self.journal.get('gate_closed') and self._gate_manager(frame) is False
                    and point == (355,803))
        if kind == 'search-confirm':
            return (self.phase == 'search' and self.completed_rank_40
                    and self.journal.get('gate_closed') and point == (208,786)
                    and self._fixed(frame,'search-question',(194,555,525,620),.95)
                    and self._fixed(frame,'continue',(63,741,356,834),.96))
        if kind == 'panel-close':
            return (self.phase == 'panel_close' and self._gate_manager(frame) is True
                    and self._gate_stage(frame) == self.journal['to_stage'] and point == (590,395))
        if kind == 'enter':
            return (self.phase == 'enter' and self._original(frame) and self._gate_active(frame)
                    and self.journal.get('verified_gate_stage') == self.journal['to_stage']
                    and point == (354,482))
        if kind == 'enter-confirm':
            return (self.phase == 'enter_confirm' and point == (208,787)
                    and self._fixed(frame,'enter-question',(220,556,502,620),.95)
                    and self._fixed(frame,'continue',(63,741,356,834),.96))
        return False

    def _close_question(self, frame):
        # Unlike a generic Continue button, this identifies the actual discard
        # prompt. It is deliberately unavailable until a real sample exists.
        path = self.root/'assets/stone-farm/close-question.png'
        if not path.is_file():
            return False
        reference = self._load('assets/stone-farm/close-question.png')
        if reference.shape != frame.shape:
            return False
        for x,y,X,Y in ((190,535,530,650),(63,741,356,834)):
            score = cv2.matchTemplate(frame[y:Y,x:X], reference[y:Y,x:X], cv2.TM_CCOEFF_NORMED)[0,0]
            if not np.isfinite(score) or score < .96:
                return False
        return float((frame[480:510,150:180].max(axis=2)>=210).mean()) >= .8

    def _finish(self, frame):
        issued = {a['kind'] for a in self.journal.get('issued_inputs',[])}
        external_return = self.journal.get('resumed_after_unverified_return') is True
        external_close = self.journal.get('resumed_after_unverified_close') is True
        manual_progress = bool(self.journal.get('manual_rank_progress_runs'))
        restoration_only = self.journal.get('restoration_only') is True
        # First-cast recovery may reach a route for the first time. Its issued
        # input is restore-route-start; requiring an earlier route-start here
        # would reject a successfully restored route after all live checks.
        required = {'activate','enter','enter-confirm',self._route_action('route-start')}
        if not external_close:
            required.add('close-gate')
        if not external_return:
            required.update(('return-request','return-confirm'))
        if self.journal['reason'] == 'cycle' and not (
                self.completed_rank_40 and self.journal.get('gate_closed')
                and required.issubset(issued)):
            raise RuntimeError('次元石循环缺少实际 RANK40 和关门证明')
        archive = self.path.parent/'logs/stone-farm'
        archive.mkdir(parents=True, exist_ok=True)
        self.journal.update(pending=False, phase='idle', finished_at=time.time(),
            restored_stage=self.journal['to_stage'],
            restored_rank=self.rank if manual_progress or restoration_only else self.journal['entry_rank'],
            restored_fishing=True)
        field = 'handoffs' if self.journal['reason'] == 'lower' else 'recoveries' if external_return or external_close or manual_progress or restoration_only else 'cycles'
        if external_return or external_close:
            self.journal['resolution'] = 'reopened_after_unverified_return_or_close'
            self.journal['missing_return_input_proof'] = sorted({'return-request','return-confirm'}-issued)
            self.journal['missing_cycle_input_proof'] = sorted({'return-request','return-confirm','close-gate'}-issued)
        if manual_progress:
            self.journal['resolution'] = 'route_restored_after_manual_rank_progress'
        elif restoration_only and not (external_return or external_close):
            self.journal['resolution'] = 'route_restored_after_approved_recovery'
        self.journal[field] = self.journal.get(field,0)+1
        self._save()
        identifier = hashlib.sha256(self.path.read_bytes()).hexdigest()
        (archive/(identifier+'.json')).write_bytes(self.path.read_bytes())
        self.phase = 'idle'
        self.off_verified = False
        self.key = None
        self.event('次元石：已按手动推进后的当前RANK恢复航巡和实际钓鱼，旧首竿失败保留，不计完整自动循环' if manual_progress
                   else '次元石：返回或关门输入未核实，仅确认门状态、重开与钓鱼恢复，不计完整循环' if external_return or external_close
                   else '次元石：已确认本次获准的航巡和实际钓鱼恢复，旧失败及次数保留，不计完整自动循环' if field == 'recoveries'
                   else f'次元石：已保留 RANK39 降到第{self.journal["to_stage"]}阶段，四项航巡及实际钓鱼已恢复' if field == 'handoffs'
                   else f'次元石：RANK40 后关门、重开第{self.journal["to_stage"]}阶段 RANK0及实际钓鱼已确认')

    def completion_step(self, frame, *, ready=None):
        """An actual RANK40 completion owns UI before ordinary menu tasks.

        Detection only requests the existing step controller. It still needs
        independent stage/RANK consensus and every original fresh input guard.
        Active lower-stage work cannot be converted into a completed cycle.
        """
        if not self.enabled():
            return False
        if self.active:
            if not self.completed_rank_40:
                return False
        elif not self.completion_visible(frame, verify_rank=False):
            return False
        return self.step(frame)

    def step(self, frame):
        if self.retry is not None:
            if not self.enabled() or time.time() > self.retry['expires_at']:
                raise RuntimeError('新降阶观察授权已过期，原失败记录未修改')
            restore_route = self.retry.get('mode') == 'route_restore'
            restore_route_open = self.retry.get('mode') == 'route_open_restore'
            restore_fishing = self.retry.get('mode') == 'fishing_restore'
            restore_post_start = (self.retry.get('mode') == 'post_start_restore'
                                  and self.retry.get('allow_post_start_restore') is True)
            restore_gate = self.retry.get('mode') == 'gate_restore'
            restore_startup = self.retry.get('mode') == 'startup_restore'
            restore_manual_progress = (self.retry.get('mode') == 'manual_rank_progress_restore'
                and self.retry.get('allow_manual_rank_progress') is True)
            restore_first = self.retry.get('mode') == 'first_cast_restore'
            restore_first_map = self.retry.get('mode') == 'first_map_restore'
            restore_first_close = self.retry.get('mode') == 'first_close_restore'
            restore_first_idle = self.retry.get('mode') == 'first_idle_restore'
            restore_first_result = self.retry.get('mode') == 'first_result_restore'
            restore_lower_first_result = self.retry.get('mode') == 'lower_first_result_restore'
            restore_first_entry = self.retry.get('mode') == 'first_entry_restore'
            restore_route_result = self.retry.get('mode') == 'route_result_restore'
            restore_observed_lower = (self.retry.get('mode') == 'observed_lower_restore'
                                      and self.retry.get('allow_observed_lower_restore') is True)
            restore_observed_cycle = (self.retry.get('mode') == 'observed_cycle_restore'
                                      and self.retry.get('allow_observed_cycle_restore') is True)
            restore_observed = restore_observed_lower or restore_observed_cycle
            restore_boss_menu = self.retry.get('mode') == 'boss_abandon_menu_retry'
            restore_continuous_stop = (self.retry.get('mode') == 'continuous_cast_stop_retry'
                                       and self.retry.get('allow_continuous_cancel') is True)
            restore_return_entry = (self.retry.get('mode') == 'return_entry_restore'
                                    and self.retry.get('allow_return_entry_restore') is True)
            screen = self._expected(frame,entry=restore_route or restore_route_open or restore_fishing or restore_post_start or restore_startup or restore_first or restore_first_map or restore_first_idle or restore_first_entry)
            original_retry = bool(self.retry.get('mode') == 'return_retry'
                                  and self.completed_rank_40 and self._original(frame))
            settings = self.settings_key(frame) if screen and screen['menu'] else None
            if restore_observed:
                if not self.observed_restore_allowed():
                    raise RuntimeError('只读接续收到停止请求，原超时记录未修改')
                screen = self.screen(frame)
                battle = (self.defense._battle_footer(frame)
                          and self.defense._match(frame,'timer',(188,125,610,165),.94)
                          and self.defense._match(frame,'damage-sword',(153,205,192,245),.96)
                          and self.defense._match(frame,'swords',(307,962,415,1057),.96))
                current_scope = bool(screen and (
                    restore_observed_lower and (screen['stage'],screen['rank']) == (self.journal['to_stage'],39)
                    or restore_observed_cycle and screen['stage'] == self.journal['to_stage'] and 1 <= screen['rank'] < 39))
                stopped = bool(current_scope
                    and not screen['menu'] and not screen['completion'] and screen['route'] is True
                    and self.fishing._main_visible(frame) and not self.fishing.at_port(frame)
                    and (self.fishing.blue_fill(frame,main_known=True) or battle))
            elif restore_manual_progress:
                screen = self.screen(frame)
                stopped = bool(screen and (screen['stage'],screen['rank']) == (self.journal['to_stage'],1)
                    and not screen['menu'] and screen['route'] is False and self._off_icon(frame)
                    and self._idle_fishing(frame) and not self._moving(frame))
            elif restore_post_start:
                stopped = bool(screen and not screen['menu'] and screen['route'] is False
                    and self.signature(frame) == self.journal.get('restore_signature')
                    and self.fishing.post_start_evidence(frame))
            elif restore_route_result:
                # Only consume already restored, currently visible fishing.
                # This repair posts no input and renews no expired deadline.
                screen = self.screen(frame)
                stopped = bool(screen and screen['stage'] == self.journal['to_stage'] and 1 <= screen['rank'] < 39
                               and not screen['menu'] and screen['route'] is True
                               and self.fishing._main_visible(frame)
                               and (self.fishing.blue_fill(frame,main_known=True) or self._battle(frame)))
            elif restore_first_entry:
                stopped = self._first_unselected_idle(frame) is not None
            elif restore_lower_first_result:
                # Observe the existing cast only. No new cast, cancellation,
                # meter input, return or lowering is permitted by this repair.
                screen = self._first_result_screen(frame)
                stopped = bool(screen and screen['stage'] == self.journal['to_stage'] and screen['rank'] == 39
                               and not screen['menu'] and screen['route'] is False)
            elif restore_first_result:
                # The original 15-second cast observation remains expired.
                # A separate restoration may only consume the now-real RANK1
                # result; it never repeats a cast or renews its input window.
                screen = self.screen(frame)
                stopped = bool(screen and not screen['menu'] and screen['stage'] == self.journal['to_stage']
                               and screen['rank'] == 1 and screen['route'] is False
                               and self._off_icon(frame) and self.fishing._main_visible(frame))
            elif restore_route_open:
                stopped = bool(screen and not screen['menu'] and screen['route'] is False
                               and self._off_icon(frame) and self.fishing._main_visible(frame)
                               and self.signature(frame) == self.journal.get('first_signature')
                               and (self._battle(frame) or self.fishing.blue_fill(frame)))
            elif restore_first_idle:
                stopped = bool(self._first_idle(frame) is not None
                               and self.fishing.idle_icon(frame) == 'hook'
                               and self.signature(frame) == self.journal.get('first_signature'))
            elif restore_first_close:
                # The user may already have closed this selected RANK0 map.
                # A fresh dormant task can proceed straight to the unused
                # first cast; never reopen/reselect or invent a close receipt.
                stopped = bool(self._first_selected_map(frame) or self._first_idle(frame) is not None)
            elif restore_first_map:
                # The user may have closed the old task map while stopped.
                # Reopen only from the same freshly verified dormant task;
                # retain all map attempts and the one wake/cast limits.
                stopped = bool(self._first_map_ready(frame)
                               or self._first_map_origin(frame) is not None
                               and self.signature(frame) == self.journal.get('first_signature'))
                if self.retry.get('allow_map_target_repair'):
                    stopped = bool(self._first_map_ready(frame) and self._map_task_target(frame))
            elif restore_first:
                stopped = bool(screen and not screen['menu'] and self._first_idle(frame,allow_running=True) is not None)
            elif restore_startup:
                stopped = bool(screen and not screen['menu'] and screen['route'] is False
                    and (self._no_destination(frame) and self._idle_fishing(frame)
                         or self._first_unselected_idle(frame,ordinary=True) is not None))
            elif restore_gate:
                stopped = bool(self.completed_rank_40 and self._off_icon(frame)
                               and self._gate_manager(frame) is False and self._gate_stage(frame) == self.journal['to_stage'])
            elif restore_route:
                stopped = bool(settings and self._off_icon(frame))
            elif restore_return_entry:
                # The user may have resumed native sailing while the worker
                # was stopped. Observe it; the formal stop phase must prove
                # OFF and idle again before any return input.
                stopped = bool(screen and not screen['menu'] and not screen['completion']
                    and screen['route'] is not None and self.fishing._main_visible(frame))
            elif restore_fishing:
                stopped = bool(screen and not screen['menu'] and screen['route'] is not None
                    and (self._first_idle(frame,allow_running=True) is not None
                         or not self.training.boss_mode(frame) and not self._battle(frame)
                         and not self.fishing.blue_fill(frame) and self.fishing.idle_icon(frame)))
            else:
                stopped = bool(original_retry or screen and screen['route'] is False and not screen['menu'])
            key = ('retry-gate' if restore_gate else 'retry-original' if original_retry else 'retry',self.signature(frame),settings) if stopped else None
            if restore_return_entry and stopped:
                key = ('retry-return-entry',self.signature(frame),screen['route'])
            if not self._confirmed(key):
                return True
            if (restore_observed or restore_first_close) and (not self.observed_restore_allowed() or not self.enabled()
                    or time.time() > self.retry['expires_at']):
                raise RuntimeError('只读接续已停止或到期，原超时记录未修改')
            raw = self.path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != self.retry['failed_sha256']:
                raise RuntimeError('降阶失败记录已变化，不重置限额')
            folder = self.path.parent/'logs/stone-farm-failed'
            folder.mkdir(parents=True,exist_ok=True)
            archive = folder/(self.retry['failed_sha256']+'.json')
            if archive.exists() and archive.read_bytes() != raw:
                raise RuntimeError('原降阶失败归档冲突')
            archive.write_bytes(raw)
            approval = self.retry
            if restore_observed:
                # Resolve the now-real scene only. The original failed window,
                # inputs and automatic handoff/cycle counts stay unchanged.
                scope = 'lower' if restore_observed_lower else 'cycle'
                rank,stage = screen['rank'],screen['stage']
                self.journal.update(pending=False,phase='idle',
                    resolution=('observed_fishing_after_expired_lower_handoff' if restore_observed_lower
                                else 'observed_fishing_after_expired_first_cast'),
                    recovery_source='unverified_after_timeout',restored_fishing=True,
                    restored_stage=stage,restored_rank=rank,recovered_at=time.time(),
                    previous_failed_sha256=approval['failed_sha256'],retry_approved_at=approval['approved_at'],
                    recoveries=self.journal.get('recoveries',0)+1)
                self.journal['observed_'+scope+'_restore_runs']=1
                self.journal['observed_'+scope+'_proof']=dict(frames=self.seen,
                    seconds=time.monotonic()-self.since,stage=stage,rank=rank,route=True,game_inputs=0)
                self._save()
                self.phase,self.rank,self.current_stage='idle',rank,stage
                self.off_verified=False
                self.key=None
                self.retry=None
                self.event('次元石：只读确认超时后已有航巡与实际钓鱼，保留旧期限和输入记录；不计自动降阶或完整循环')
                return True
            elif restore_manual_progress:
                now=time.time()
                self.journal.update(manual_rank_progress_runs=1,manual_rank_progress_observed_rank=1,
                    manual_rank_progress_started_at=now,restoration_only=True,fishing_recovery=True,
                    deadline=now+90,phase='route_open',phase_deadline=now+45)
                self.journal.pop('native_extras',None)  # Adopt current native UI extras.
                self.phase,self.off_verified='route_open',True
                self.rank,self.current_stage=1,self.journal['to_stage']
                self.key=self.since=self.last_seen=None
                self.seen=0
            elif restore_first_close:
                # One explicitly requested, hash-bound recovery. Never select
                # again or reset a used input; archive the old window exactly.
                now=time.time()
                already_closed=not self._map(frame)
                phase='first_idle' if already_closed else 'first_close'
                self.journal.update(first_close_repair_runs=1,first_close_repair_started_at=now,
                    first_close_previous_deadlines={k:self.journal.get(k)
                        for k in ('deadline','phase_deadline','first_deadline','return_deadline')},
                    deadline=now+180,phase_deadline=now+(15 if already_closed else 45),first_deadline=now+60,
                    phase=phase,first_map_closed_before_recovery=already_closed,
                    first_signature=self.signature(frame),restoration_only=True,fishing_recovery=True)
                self.journal.pop('native_extras',None)
                self.phase,self.off_verified=phase,True
                self.rank,self.current_stage=0,self.journal['to_stage']
                self.key=self.since=self.last_seen=None
                self.seen=0
            elif restore_post_start:
                # Archive the expired window exactly; this one-shot repair
                # only restores the route after three seconds of real fishing.
                # All old meter/portal inputs, start time and deadlines survive.
                now = time.time()
                self.journal.update(post_start_restore_runs=1,post_start_restore_started_at=now,
                    post_start_previous_deadline=self.journal['deadline'],
                    post_start_previous_phase_deadline=self.journal['phase_deadline'],
                    deadline=now+90,phase_deadline=now+45,phase='route_open',
                    restored_fishing_proof='observed_fishing_after_prior_hold',fishing_recovery=True)
                self.phase,self.off_verified = 'route_open',True
                self.rank,self.current_stage = screen['rank'],screen['stage']
                self.key = self.since = self.last_seen = None
                self.seen = 0
            elif restore_continuous_stop:
                # A separately bounded repair archives the failed run and
                # retains all reservations, receipts and the original start.
                now = time.time()
                self.journal.update(continuous_cast_stop=True,continuous_stop_repair_runs=1,
                                    continuous_stop_repair_started_at=now,
                                    deadline=now+self.LIMIT,return_deadline=now+180,
                                    phase='stop',phase_deadline=now+180)
                self.rank,self.current_stage = screen['rank'],screen['stage']
                self.phase = 'stop'
                self.off_verified = self.fishing_stopped = False
                self.cancel_anchor = self.boss_abandon_key = None
                self.key = self.since = self.last_seen = None
                self.seen = 0
            elif restore_return_entry:
                now = time.time()
                self.journal.update(return_entry_repair_runs=1,return_entry_repair_started_at=now,
                    return_entry_previous_deadlines={k:self.journal.get(k)
                        for k in ('deadline','phase_deadline','return_deadline')},
                    deadline=now+self.LIMIT,return_deadline=now+180,
                    phase='stop',phase_deadline=now+180)
                self.rank,self.current_stage = screen['rank'],screen['stage']
                self.phase = 'stop'
                self.off_verified = self.fishing_stopped = False
                self.cancel_anchor = self.boss_abandon_key = None
                self.key = self.since = self.last_seen = None
                self.seen = 0
            elif restore_boss_menu:
                receipt = self._boss_abandon_receipt(self.path.parent,self.journal)
                if receipt is None or receipt != approval.get('native_abandon_receipt_sha256'):
                    raise RuntimeError('原生放弃输入回执已变化，不能追加接触')
                # This new explicit approval has its own bounded observation.
                # Preserve every old reservation/receipt and archive the old deadlines.
                now = time.time()
                self.journal.update(boss_abandon_menu_retry=True,boss_abandon_menu_started_at=now,
                                    native_abandon_receipt_sha256=receipt,
                                    deadline=now+self.LIMIT,return_deadline=now+180,
                                    phase='stop',phase_deadline=now+180)
                self.rank,self.current_stage = screen['rank'],screen['stage']
                self.phase = 'stop'
                self.off_verified = self.fishing_stopped = False
                self.boss_abandon_key = None
                self.key = self.since = self.last_seen = None
                self.seen = 0
            elif restore_route_result:
                self.journal.update(route_result_repair_runs=1,
                                    previous_failed_sha256=approval['failed_sha256'],
                                    retry_approved_at=approval['approved_at'])
                self.rank,self.current_stage = screen['rank'],screen['stage']
                self._finish(frame)
                self.retry = None
                return False
            elif restore_route or restore_route_open or restore_fishing or restore_startup or restore_first or restore_first_map or restore_first_idle or restore_first_result or restore_lower_first_result or restore_first_entry:
                # A separate bounded restoration run retains every original
                # input count and completed portal proof. It never returns or
                # lowers again, and the failed run remains byte-for-byte saved.
                now = time.time()
                # First-task selection/casting keeps its 60-second deadline.
                # An explicitly repaired, interrupted map-open also needs the
                # existing route-menu/running phases after that task succeeds.
                # Allocate this at repair creation, never renew a live deadline.
                map_open_restore = restore_first_map and approval.get('map_open_interrupted') is True
                limit = 180 if restore_fishing or restore_startup or restore_first or restore_first_entry or map_open_restore else 60 if restore_first_map or restore_first_idle else 90
                self.journal.update(started_at=now,deadline=now+limit,phase_deadline=now+limit,
                                    restoration_only=True)
                self.journal.pop('native_extras',None)
                self.off_verified = restore_route or restore_route_open or restore_startup or restore_first_map or restore_first_idle or restore_first_result or restore_lower_first_result or restore_first_entry
                self.current_stage = self._stage(frame) if restore_first_map else screen['stage']
                self.key = self.since = self.last_seen = None
                self.seen = 0
                if restore_first_entry:
                    self.journal.update(first_entry_repair_runs=1,first_unselected=True,
                                        first_signature=self.signature(frame),first_deadline=now+60)
                    self._set('first_map',30)
                elif restore_route_open:
                    self.journal.update(route_open_repair_runs=1,fishing_recovery=True)
                    self.phase=self.journal['phase']='route_open'
                elif restore_first_result or restore_lower_first_result:
                    if restore_lower_first_result:
                        self.journal['first_result_previous_started_at'] = json.loads(raw)['started_at']
                        self.journal['first_result_previous_deadline'] = json.loads(raw)['deadline']
                        self.journal['first_result_previous_phase_deadline'] = json.loads(raw)['phase_deadline']
                    self.journal.update(first_result_repair_runs=1,fishing_recovery=True,
                                        first_cast_verified_at=now,first_cast_verified_rank=screen['rank'])
                    self.rank = screen['rank']
                    self._set('route_open')
                elif restore_first_idle:
                    self.journal.update(first_deadline=now+60,first_idle_repair_runs=1,
                                        phase='first_idle',phase_deadline=now+15)
                    self.phase='first_idle'
                elif restore_first_map:
                    self.journal.update(first_deadline=now+60,first_map_repair_runs=1)
                    if approval.get('allow_map_target_repair'):
                        self.journal['first_map_target_repair_runs'] = 1
                    # Re-enter the explicitly new repair phase even when its
                    # name equals the archived failure's phase. _set otherwise
                    # returns early and leaves the total window as its limit.
                    self.phase = 'idle'
                    on_map = self._map(frame)
                    self._set('first_select' if on_map else 'first_map',45 if on_map else 30)
                elif restore_first:
                    self.journal['first_recovery'] = True
                    self.journal['fishing_recovery'] = True
                    self.phase = self.journal['phase'] = 'fishing_stop'
                elif restore_fishing:
                    self.journal['fishing_recovery'] = True
                    if self._first_idle(frame,allow_running=True) is not None:
                        self.journal['first_recovery'] = True
                    self.phase = self.journal['phase'] = 'fishing_stop'
                elif restore_startup:
                    if self._first_unselected_idle(frame,ordinary=True) is not None:
                        self._prepare_first_task(frame)
                    else:
                        self.phase = self.journal['phase'] = 'route_open'
            elif restore_gate:
                now = time.time()
                self.journal.update(started_at=now,deadline=now+180,
                                    phase='closed',phase_deadline=now+45,
                                    gate_closed=True,closed_verified_at=now,
                                    resumed_after_unverified_close=True,restoration_only=True)
                self.phase = 'closed'
                self.key = self.since = self.last_seen = None
                self.seen = 0
            elif original_retry:
                # A new actual observatory can safely resume closing a saved,
                # completed gate. The unexplained return is never credited as
                # an issued input or a complete unattended cycle.
                now = time.time()
                self.journal.update(started_at=now,deadline=now+self.LIMIT,
                                    phase='original',phase_deadline=now+45,
                                    resumed_after_unverified_return=True,
                                    returned_by_recovery_observation_at=now)
                self.phase = 'original'
                self.off_verified = self.fishing_stopped = False
                self.key = self.since = self.last_seen = None
                self.seen = 0
            else:
                self._begin(screen)
                if approval.get('allow_boss_abandon') is True:
                    self.journal['allow_boss_abandon'] = True
            self.journal.update(previous_failed_sha256=approval['failed_sha256'],
                                retry_approved_at=approval['approved_at'])
            self._save()
            self.retry = None
            if restore_return_entry:
                self.event('次元石：已原样归档返回入口超时，另开获准的180秒停钓返回观察；保留全部已用次数，重新确认停航停钓')
                return True
            if restore_first_close:
                self.event('首任务关图：已原样归档旧失败，另开获准的180秒恢复；首任务最多60秒，保留全部已用次数')
                return True  # The close must use another new frame.
            self.event('次元石：已原样归档失败记录，连续新图核对后另开获准的安全停钓降阶观察')
            if restore_post_start:
                return True  # Every subsequent route input needs a new frame.
        if not self.active:
            if not self.enabled():
                return False
            screen = self.screen(frame)
            if screen:
                self.rank, self.current_stage = screen['rank'], screen['stage']
            self.observe_stage_policy(frame, screen)
            retirement = self.journal.get('retirement')
            if retirement and not retirement.get('released'):
                context = [screen['stage'],screen['rank']] if screen and not screen['menu'] else None
                if context is None:
                    self._confirmed(None)
                    self.hold = '旧交接已归档，当前画面不可读，不重复输入；等待实际阶段或RANK推进'
                    return False
                # Identity alone is not permission to continue: replay of the
                # same archived screen stays blocked for the retry backoff and
                # is then released only on independently re-confirmed frames,
                # never blindly. The block is bounded so a context that is its
                # own next working position (e.g. the [8,40] lowering) cannot
                # deadlock the task forever.
                backoff = retirement.setdefault('backoff_seconds', handoff_journal.BACKOFF_SECONDS)
                archived_at = retirement.get('at')
                elapsed = (time.time()-archived_at if type(archived_at) in (int,float)
                           else backoff)
                if context in retirement['blocked_contexts'] and elapsed < backoff:
                    self._confirmed(None)
                    self.hold = ('旧交接已归档，当前交接不重复输入；阻塞画面退避中，'
                                 f'等待退避窗口结束（剩余约{backoff-elapsed:.0f}秒）或实际阶段或RANK推进')
                    return False
                if not self._confirmed(('retired-progress',*context)):
                    return False
                retirement.update(released=True, released_context=context, released_at=time.time())
                self._save()
                self._confirmed(None)
                self.hold = None
            trigger = bool(screen and (screen['stage'],screen['rank']) in
                           ((self.START_STAGE,39),(self.FINAL_STAGE,40)) and not screen['menu'])
            if not trigger:
                self._confirmed(None)
                return False
            if not self._confirmed((screen['stage'],screen['rank'])):
                self.hold = '连续核对当前阶段与RANK，暂缓新的菜单任务'
                return True
            self._begin(screen)
        self.check_deadline()
        if time.monotonic() < self.next_at:
            return True
        self.hold = self.phase
        phase = self.phase
        if phase.startswith('first_'):
            return self._first_step(frame)
        if phase == 'fishing_stop':
            screen = self._expected(frame,entry=True)
            if not screen or screen['menu']:
                self._confirmed(None)
                return True
            if screen['route'] is True:
                self.off_verified = False
                self._confirmed(None)
                self._click('restore-stop',233,45,'恢复钓鱼前先停止游戏航巡',frame)
            elif self._confirmed(('restore-off',self.signature(frame))):
                self.off_verified = True
                origin = self._first_map_origin(frame)
                if self.journal.get('first_recovery') or origin is not None:
                    if origin is not None:
                        self._prepare_first_task(frame)
                else:
                    self._prepare_fishing(frame)
            return True
        if phase == 'fishing_prepare':
            screen = self._expected(frame,entry=True)
            if not screen or screen['menu'] or screen['route'] is not False:
                self.fishing.idle_since = None
                return True
            if not self.off_verified:
                if not self._confirmed(('restart-restore-off',self.signature(frame))):
                    return True
                self.off_verified = True
                self._prepare_fishing(frame)
            # A resumed, untouched RANK0 ordinary stop uses the same bounded
            # real-map selection as a fresh first entry. No expired deadline
            # or previously used meter/cast input may be renewed here.
            first_origin = (self._first_unselected_idle(frame,ordinary=True)
                if not self.journal.get('meter_attempts')
                and not any(self.journal.get('attempts',{}).get(k)
                    for k in ('first-map','first-select','first-wake','first-cast')) else None)
            if first_origin is not None:
                if self._confirmed(('ordinary-first-origin',first_origin)):
                    self._prepare_first_task(frame)
                return True
            if self._no_destination(frame):
                # A new RANK0 gate must choose a destination through native
                # sailing before any meter gesture or long press is allowed.
                if not self.journal.get('meter_attempts'):
                    self._set('route_open')
                return True
            if (not self.journal.get('meter_attempts')
                    and self.signature(frame) == self.journal.get('restore_signature')
                    and self.fishing._context(frame) != self.meter_context):
                # Entry fading changes absolute white pixels, although the
                # actual RANK and fixed location remain identical. Reprove a
                # stable context before the first input, without renewing the
                # workflow deadline or accepting a changed location.
                context = self.fishing._context(frame)
                if self._confirmed(('restore-context',self.signature(frame),context)):
                    self.meter_context = context
                    self.fishing.idle_since = None
                return True
            if self.journal.get('meter_attempts',{}).get('start') and not self.fishing.start_fill_verified:
                if self.fishing.phase != 'starting':
                    self.fishing.arm_start_observation(frame)
                self.fishing.confirm_start(frame)
            if self.fishing.blue_fill(frame) or self.fishing.real_battle(frame):
                paired = self.fishing.start_fill_verified
                if self.journal.get('meter_attempts',{}).get('start') and not paired:
                    self.hold = '长按结束后连续新图确认真实填充或战斗；不重复输入'
                    return True
                if self._confirmed(('restored-fishing',self.signature(frame))):
                    self.journal['restored_fishing_proof'] = (self.fishing.start_proof or 'paired_hold') if paired else 'observed_existing_fishing'
                    self._save()
                    self._set('route_open')
                return True
            if self.meter_frame_allowed(frame):
                self.fishing.step(frame)
                self.journal['restore_meter_limits'] = self.fishing.range_cache
                self._save()
            else:
                self.fishing.idle_since = None
                self.hold = '等待同一普通地点、停航、无填充和空闲圆盘'
            return True
        if phase == 'completion':
            if self.completion_visible(frame, verify_rank=False):
                if self._click('completion',360,840,f'确认第{self.journal["from_stage"]}阶段探索结束',frame):
                    self._set('stop',180)
            elif self._expected(frame):
                self._set('stop',180)
            return True
        if phase == 'stop':
            screen = self._expected(frame)
            if not screen or screen['menu']:
                return True
            if screen['route'] is True:
                self.off_verified = False
                self._confirmed(None)
                self._click('stop',233,45,'先停止游戏自动航巡',frame)
            elif screen['route'] is False and self._confirmed(('off',screen['stage'],screen['rank'])):
                self.off_verified = True
                context = self.cancel_context(frame,continuous=True) if self.continuous_cancel_allowed else None
                self.cancel_anchor = (self.signature(frame),context) if context is not None else None
                self.journal['off_verified_at'] = time.time()
                self._save()
                self._set('stop_fishing',180)
            return True
        if phase == 'stop_fishing':
            if time.time() > self.journal['return_deadline']:
                raise RetryableStage('次元石停钓及返回超过180秒，保留进度，等待新图安全归档本轮',
                                     task='stone_farm', retry_seconds=1800)
            idle = self._idle_fishing(frame)
            context = (self.cancel_context(frame,continuous=True)
                       if self.continuous_cancel_allowed and not idle else None)
            anchored = (self.off_verified and context is not None
                        and self.cancel_anchor == (self.signature(frame),context))
            # Actual OCR established the stopped stage/RANK in stop(). New
            # exact strokes bind each capture to that proof without delaying
            # the fleeting cast window with another full numeric batch.
            screen = (dict(stage=self.journal['from_stage'],rank=self.journal['rank'],
                           route=False,menu=False) if anchored else self._expected(frame))
            if not screen or screen['menu']:
                self._confirmed(None)
                return True
            if screen['route'] is True:
                self.off_verified = self.fishing_stopped = False
                self.boss_abandon_key = None
                self.cancel_anchor = None
                self._set('stop',180)
                return True
            if not self.off_verified:
                self.off_verified = self._confirmed(('restart-off',self.signature(frame)) if screen['route'] is False else None)
                if self.off_verified and context is not None:
                    self.cancel_anchor = (self.signature(frame),context)
                return True
            if idle:
                self.boss_abandon_key = None
                if self._confirmed(('fishing-idle',self.signature(frame))):
                    self.fishing_stopped = True
                    self.journal['fishing_stopped_at'] = time.time()
                    self._save()
                    self._set('return',180)
                return True
            if self.continuous_cancel_allowed:
                self.boss_abandon_key = None
                self._confirmed(None)
                if context is not None and self.journal.get('attempts',{}).get('handoff-cancel',0) < self.INPUT_LIMITS['handoff-cancel']:
                    self.cancel_anchor = (self.signature(frame),context)
                    self._click('handoff-cancel',690,1090,'等待甩竿时右下鱼形红叉停止持续钓鱼',
                                frame,cancel_context=context,continuous_stop=True)
                return True
            if (self.boss_abandon_handoff
                    and not self.journal.get('attempts',{}).get(self.boss_abandon_kind)):
                context = self.boss_abandon_context(frame)
                if context is not None:
                    if self._confirmed((self.boss_abandon_kind,context)):
                        self.boss_abandon_key = context
                        self._click(self.boss_abandon_kind,*context[1],
                            '获准降阶时仅放弃本场海神，随后核对实际停钓',frame,abandon_context=context)
                        self.boss_abandon_key = None
                        self._confirmed(None)
                    return True
            self.boss_abandon_key = None
            self._confirmed(None)
            if self.ordinary_cancel_allowed:
                context = self.cancel_context(frame)
                if context is not None:
                    self._click('handoff-cancel',690,1090,'仅取消停航后的普通鱼准备阶段',frame,cancel_context=context)
            return True
        if phase in ('return','return_confirm'):
            if time.time() > self.journal['return_deadline']:
                raise RetryableStage('次元石返回超过180秒，保留进度，等待新图安全归档本轮；最近等待：'+str(self.hold),
                                     task='stone_farm', retry_seconds=1800)
            if self._original(frame):
                self._set('original')
                return True
            # Saved flags do not prove either stop after a restart. A modal
            # without this in-memory proof cannot authorise confirmation.
            if not self.off_verified or not self.fishing_stopped:
                self._report_return_wait('等待重新确认停航及停钓；保存标记不能代替当前画面')
                if self._expected(frame) and self.sailing.running(frame) is False:
                    self._set('stop_fishing',180)
                return True
            if not self._parallel(frame) or not self._off_icon(frame):
                self._report_return_wait('当前平行世界或顶部停止标记未确认')
                return True
            if self._fixed(frame,'return-question',(223,556,506,629),.95):
                if phase != 'return_confirm':
                    self._set('return_confirm',180)
                if self._click('return-confirm',208,788,'保存本轮位置并返回观测所',frame):
                    self.next_at = time.monotonic()+3
                    self._report_return_wait('已发出返回确认，等待实际原世界画面')
                else:
                    self._report_return_wait('返回确认的新截图守卫未通过，未发出输入')
                return True
            if self.sailing.running(frame) is False and self._idle_fishing(frame):
                point = self._return_button(frame)
                if point:
                    if phase != 'return':
                        self._set('return',180)
                    if self._click('return-request',*point,'点击当前新图的返回次元门',frame):
                        self.next_at = time.monotonic()+2
                        self._report_return_wait('已发出返回请求，等待确认窗口')
                    else:
                        self._report_return_wait('返回入口的新截图守卫未通过，未发出输入')
                else:
                    self._report_return_wait('未找到唯一返回入口（固定边框阈值0.96）：'+str(self.return_probe))
            else:
                self._report_return_wait('等待当前画面再次确认停航和未钓鱼')
            return True
        if phase == 'original':
            if self._confirmed(('original',) if self._original(frame) else None):
                self.journal['returned_at'] = time.time()
                self._save()
                self._set('npc')
            return True
        if phase == 'npc':
            if self._gate_manager(frame) is not None:
                self._set('manager')
            elif self._confirmed(('npc',) if self._npc_ready(frame) else None):
                if self._click('npc',482,548,'打开观测所管理菜单',frame):
                    self.next_at = time.monotonic()+2
            return True
        if phase == 'manager':
            active = self._gate_manager(frame)
            stage = self._gate_stage(frame) if active is not None else None
            if stage is None:
                self.hold='传送门管理页尚未读清阶段：'+str(getattr(self,'gate_probe',None))
                now=time.monotonic()
                if now-getattr(self,'gate_wait_logged_at',float('-inf'))>=5:
                    self.gate_wait_logged_at=now
                    self.event('次元石门管理等待：'+self.hold)
            if active is False and self.completed_rank_40:
                if not self.journal['attempts'].get('close-gate'):
                    self.journal['resumed_after_unverified_close'] = True
                    self._save()
                self._set('closed')
                return True
            key = ('manager',stage) if active is True and stage is not None else None
            if not self._confirmed(key):
                return True
            if self.journal['reason'] == 'lower':
                if stage == self.journal['to_stage']:
                    self.journal['verified_gate_stage'] = stage
                    self._save()
                    self._set('panel_close')
                elif stage == self.journal['from_stage'] and self._click('lower',357,873,'只降低当前世界一阶，不关闭传送门',frame):
                    self._set('lower_confirm')
            elif stage == self.journal['from_stage'] and self._click('close-gate',354,739,'RANK40通关后关闭旧传送门',frame):
                self._set('close_confirm',90)
            return True
        if phase == 'lower_confirm':
            target = self._lower_prompt(frame)
            if self._confirmed(('lower',target) if target == self.journal['to_stage'] else None):
                if self._click('lower-confirm',208,788,f'确认第{self.journal["from_stage"]}阶段降至第{self.journal["to_stage"]}阶段',frame):
                    self._set('lower_verify')
            elif self._gate_manager(frame) is True and self._gate_stage(frame) == self.journal['to_stage']:
                self._set('lower_verify')
            return True
        if phase == 'lower_verify':
            stage = self._gate_stage(frame) if self._gate_manager(frame) is True else None
            if self._confirmed(('lowered',stage) if stage == self.journal['to_stage'] else None):
                self.journal['verified_gate_stage'] = stage
                self.journal['lower_verified_at'] = time.time()
                self._save()
                self._set('panel_close')
            return True
        if phase in ('close_confirm','closed'):
            inactive = self._gate_manager(frame) is False
            if self._confirmed(('closed',) if inactive else None):
                self.journal['gate_closed'] = True
                self.journal['closed_verified_at'] = time.time()
                self._save()
                if phase != 'closed':
                    self._set('closed')
                elif self._click('activate',355,803,f'重开新的第{self.journal["to_stage"]}阶段世界',frame):
                    self._set('search')
            elif phase == 'close_confirm' and self._close_question(frame):
                if self._click('close-confirm',208,788,'确认关闭已通关的旧门',frame):
                    self._set('closed')
            return True
        if phase == 'search':
            if self._gate_manager(frame) is True:
                self._set('new_gate')
            elif self._fixed(frame,'search-question',(194,555,525,620),.95):
                if self._click('search-confirm',208,786,'确认搜索新世界',frame):
                    self._set('new_gate')
            return True
        if phase == 'new_gate':
            stage = self._gate_stage(frame) if self._gate_manager(frame) is True else None
            if self._confirmed(('new-gate',stage) if stage == self.journal['to_stage'] else None):
                self.journal['verified_gate_stage'] = stage
                self._save()
                self._set('panel_close')
            return True
        if phase == 'panel_close':
            if self._gate_manager(frame) is True:
                if self._click('panel-close',590,395,'关闭已核对阶段的说明',frame):
                    self._set('enter')
            elif self._original(frame):
                self._set('enter')
            return True
        if phase == 'enter':
            if self._fixed(frame,'enter-question',(220,556,502,620),.95):
                self._set('enter_confirm')
            elif self._click('enter',354,482,'进入已核对的世界，保留进度',frame):
                self._set('enter_confirm')
            return True
        if phase == 'enter_confirm':
            if self._expected(frame, entry=True):
                self._set('arrive',90)
            elif self._click('enter-confirm',208,787,'确认进入次元门',frame):
                self._set('arrive',90)
            return True
        if phase == 'arrive':
            screen = self._expected(frame, entry=True)
            if self._confirmed(('arrive',screen['stage'],screen['rank']) if screen and screen['route'] is False else None):
                self.off_verified = True
                self.journal['entered_verified_at'] = time.time()
                self._save()
                if self._first_map_origin(frame) is not None:
                    self._prepare_first_task(frame)
                elif (self._idle_fishing(frame) and not self.training.boss_mode(frame)
                        and not self.fishing.at_port(frame) and not self._no_destination(frame)):
                    self._prepare_fishing(frame)
                else:
                    self._set('route_open')
            return True
        if phase == 'route_open':
            screen = self._expected(frame, entry=True)
            if screen and screen['menu']:
                self._set('route_menu')
            elif screen and screen['route'] is False:
                if not self.off_verified:
                    self.off_verified = self._confirmed(('entry-off',screen['stage'],screen['rank']))
                    return True
                if self._click(self._route_action('route-open'),233,45,'打开航巡设置，核对自动贩卖',frame):
                    self._set('route_menu')
            return True
        if phase == 'route_menu':
            screen = self._expected(frame, entry=True)
            settings = self.settings_key(frame) if screen and screen['menu'] else None
            if not settings or not self._confirmed(settings):
                return True
            policy=getattr(self,'cruise_policy',None)
            if policy is not None:
                if 'cruise_priority' not in self.journal:
                    self.journal['cruise_priority']=self.cruise_priority(frame)
                    self._save()
                if self.cruise_priority(frame)!=self.journal['cruise_priority']:
                    self.hold='目的地优先顺序已变化，等待重新核对'
                    return True
                self.off_verified=True;self.menu_key=settings
                for name,(x,_,label) in TARGETS.items():
                    if self.sailing.selected(frame,name) is not policy['targets'][name]:
                        self._click('target:'+name,x,458,'按巡航设置切换'+label,frame)
                        self.key=None
                        return True
                for kind,actual,wanted,y in (
                    ('cruise-auto-sale',self.sailing.sale_selected(frame),policy['auto_sale'],647),
                    ('cruise-auto-upgrade',self.sailing.upgrade_selected(frame),policy['auto_upgrade'],761)):
                    if actual is None:return True
                    if actual is not wanted:
                        self._click(kind,576,y,'按巡航设置切换自动项',frame)
                        self.key=None
                        return True
                if self._click(self._route_action('route-start'),360,873,'按已保存的巡航设置启动',frame):
                    self._set('route_running',90)
                return True
            sale_ready = native_auto_sale.verified_on(self,frame,settings,self.journal['attempts'])
            if 'native_extras' not in self.journal:
                self.journal['native_extras'] = settings[1]
                self._save()
            if self.journal['native_extras'] != settings[1]:
                self.hold = '航巡优先顺序或自动出售/强化已在线修改，等待重新核对'
                return True
            self.off_verified = True
            self.menu_key = settings
            if not sale_ready:
                self.hold = '等待确认自动贩卖勾选'
                if (not self.journal['attempts'].get(native_auto_sale.ACTION)
                        and native_auto_sale.arm(self,frame,settings)):
                    self._click(native_auto_sale.ACTION,576,647,'勾选游戏自动贩卖',frame)
                    self.key = None
                return True
            for (name,(x,_,label)), actual, wanted in zip(TARGETS.items(),settings[0],self.TARGET_VALUES):
                if actual is not wanted:
                    self._click('target:'+name,x,458,('勾选' if wanted else '取消')+label,frame)
                    self.key = None
                    return True
            if self._click(self._route_action('route-start'),360,873,'启动珍珠、琥珀、蓝宝石与BOSS四项航巡',frame):
                self._set('route_running',90)
            return True
        if phase == 'route_running':
            screen = self.screen(frame)
            # A fast final catch can finish before the three running frames.
            # Its actual RANK40 notice proves fishing occurred; archive only
            # the lower handoff, then establish the separate RANK40 cycle.
            completed = bool(self.journal['reason'] == 'lower' and screen
                             and (screen['stage'],screen['rank']) == (self.journal['to_stage'],40)
                             and screen['completion'])
            if completed:
                if self._confirmed(('restored-and-completed',self.journal['to_stage'],40)):
                    self.journal['handoff_rank40_notice_at'] = time.time()
                    self._finish(frame)
                return True
            # The native game can advance the first boss before three frames
            # arrive. Accept forward ranks only on the newly verified stage.
            valid = bool(screen and screen['stage'] == self.journal['to_stage']
                         and self.journal['entry_rank'] <= screen['rank'] < 40
                         and screen['route'] is True)
            # The destination name and a gray hook can appear while sailing.
            # Movement alone cannot prove dormancy, but actual fishing during
            # travel remains valid evidence that the route has been restored.
            fishing = valid and (self.fishing.blue_fill(frame,main_known=True) or self._battle(frame))
            if valid and not fishing and self._moving(frame):
                self._confirmed(None)
                return True
            if self.journal.get('manual_rank_progress_runs') and not fishing:
                self._confirmed(None)
                self.hold='手动推进后仅恢复航巡并等待真实钓鱼；不重复首竿、标尺或返回'
                return True
            first_idle = ((self._first_idle(frame,allow_running=True)
                or self._first_unselected_idle(frame,ordinary=True,allow_running=True))
                if valid and not fishing else None)
            if first_idle is not None and not self.journal.get('attempts',{}).get('first-cast'):
                if self._confirmed(('dormant-first-task',first_idle)):
                    self.journal.update(first_recovery=True,fishing_recovery=True)
                    self._save()
                    self._set('fishing_stop',180)
                return True
            dormant = (valid and not fishing and not self.training.boss_mode(frame)
                       and not self.fishing.at_port(frame) and self.fishing.idle_icon(frame)
                       and not self.journal.get('meter_attempts',{}).get('start')
                       and not self.journal.get('fishing_recovery'))
            if dormant:
                if self._confirmed(('dormant-native-route',self.signature(frame))):
                    self.journal['fishing_recovery'] = True
                    self._save()
                    self._set('fishing_stop',180)
                return True
            if self._confirmed(('restored',screen['stage']) if fishing else None):
                self.rank, self.current_stage = screen['rank'],screen['stage']
                self._finish(frame)
            return True
        raise RuntimeError('未知次元石降阶步骤：'+phase)
