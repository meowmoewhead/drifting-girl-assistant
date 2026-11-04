"""Image-driven parallel-world rank loop for Drifting Girl.

The irreversible portal close is reachable only after three confident RANK 40
readings from the parallel world.  No coordinates are used unless their screen
has first been identified by an image template or the rank/world guard.
"""
from collections import deque
from contextlib import contextmanager
from pathlib import Path
import time

import cv2
import numpy as np

from drift.tasks.boss_training import BossTraining
from drift.tasks.fishing_meter import FishingMeter
from drift.tasks.training_sailing import TrainingSailing
from drift.tasks.boss_defense import BossDefense
from drift.tasks.final_boss_attack import forced_attacks, is_final_boss
from drift.tasks.final_boss_attack import options_for as final_boss_options
from drift.tasks.parallel_screen import ParallelScreen
from drift.perception.digit_ocr import read_numeric_views


class ParallelWorld(ParallelScreen):
    """平行世界导航与练级状态机。

    画面判定(badge / RANK / 阶段笔画 / 航行菜单 RANK)继承自
    :class:`~drift.tasks.parallel_screen.ParallelScreen` —— 同一份判定也被独立的
    标尺等级校准复用,避免两处各写一遍。本类只覆盖 :meth:`_match`(自己的模板匹配
    器,含地图 BOSS 图标的二次校验),基类由此拿到 ``_fixed`` / ``_parallel`` 等。
    """

    def __init__(self, root: Path, click, swipe, event, digit_reader, interval=2,
                 training=None, training_state_path=None, fishing_action=None, sailing_action=None,
                 defense_action=None, completion_action=None, portal_action=None):
        super().__init__(root, digit_reader)
        legend = Path(root)/'assets/task-popup/map-legend.png'
        self.map_legend = (cv2.cvtColor(cv2.imread(str(legend)),cv2.COLOR_BGR2RGB)
                           if legend.is_file() else None)
        self.click = click
        self.swipe = swipe
        self.event = event
        self.reader = digit_reader
        self.interval = max(1.2, float(interval))
        self.phase = 'detect'
        self.phase_since = time.monotonic()
        self.missing_parallel_since = None
        self.next_action_at = 0.0
        self.rank = None
        self.latest_rank_value = None
        self.latest_rank_frame = None
        self.rank_readings = deque(maxlen=3)
        self.backpack_rank_readings = deque(maxlen=3)
        self.backpack_rank_key = None
        self.target_rank = None
        self.completed_rank_40 = False
        self.map_swipes = 0
        self.map_refreshes = 0
        self.map_open_attempts = 0
        self.rounds = 0
        self.last_progress_at = time.monotonic()
        self.training = BossTraining(root, digit_reader, event, training, training_state_path)
        self.fishing = FishingMeter(root, self.training, fishing_action, event) if fishing_action else None
        self.sailing = TrainingSailing(root,self.training,sailing_action,event,self.fishing,
                                      self.sailing_context) if sailing_action else None
        self.defense = BossDefense(root,self.training,defense_action,event,scan=True) if defense_action else None
        self.sailing_shutdown = False
        self.shutdown_complete = False
        completion = Path(root)/'assets/recovery/parallel-complete.png'
        self.completion_frame = (cv2.cvtColor(cv2.imread(str(completion)),cv2.COLOR_BGR2RGB)
                                 if completion.is_file() else None)
        if completion_action is not None and (self.completion_frame is None
                                              or self.completion_frame.shape[:2] != (1280,720)):
            raise RuntimeError('平行世界通关弹窗模板不完整，已停止')
        self.completion_action = completion_action
        self.completion_since = None
        self.completion_frames = self.completion_clear_frames = 0
        self.completion_clicked = False
        self.portal_action = portal_action
        self.portal_frames = {}
        for name in ('parallel-manager', 'parallel-manager-inactive',
                     'parallel-lower-question', 'observatory-npc', 'observatory-npc-day'):
            path = Path(root)/'assets/recovery'/f'{name}.png'
            self.portal_frames[name] = (cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
                                        if path.is_file() else None)
        if portal_action is not None and any(value is None or value.shape[:2] != (1280,720)
                                             for value in self.portal_frames.values()):
            raise RuntimeError('传送门管理与降阶模板不完整，已停止')
        self.gate_stage_verified = False
        self.gate_stage_readings = deque(maxlen=3)
        self.lower_from = None
        self.cat_frames = self.cat_attempts = 0
        self.return_since = None
        self.quest_idle_key = self.quest_idle_since = None
        self.quest_idle_frames = 0
        self.quest_recovery_since = self.quest_recovery_key = None
        self.quest_recovery_rank = None
        self.quest_open_attempts = self.quest_map_frames = 0
        self.quest_map_target = None
        self.quest_map_selected = False
        self.quest_wake_ranks = set()
        self.quest_cast_ranks = set()
        self.quest_recoveries = 0
        self.boss_sailing_rank_frame = None
        self.boss_sailing_idle_key = self.boss_sailing_idle_since = None
        self.boss_sailing_idle_frames = 0
        self.boss_handoff_since = None
        self.boss_handoff_map_frames = 0
        self.boss_handoff_signature = None

    @staticmethod
    def _return_signature(frame):
        # Only fixed number/world/location strokes; never the animated sea,
        # fishing dial, travel countdown or a previous input coordinate.
        parts = []
        for x,y,X,Y in ((71,9,136,59),(80,9,130,59),(13,45,60,72)):
            crop = frame[y:Y,x:X].min(axis=2)
            peak = float(crop.max())
            parts.append((crop >= peak*.84).tobytes())
        parts.append((frame[12:34,286:434].max(axis=2)<80).tobytes())
        return tuple(parts)

    def _return_ready(self, frame, confirm=False, verify_rank=True):
        if (frame.shape[:2] != (1280,720) or not self.completed_rank_40
                or self.rank != 40 or not self._parallel(frame) or self._map(frame)
                or self.sailing is None
                or not self.sailing._fixed(frame,'auto-label',(186,34,253,65),.90)
                or not self.sailing._fixed(frame,'inactive-face',(194,4,246,42),.96)
                or self.sailing._fixed(frame,'active-arrow',(194,4,246,42),.96)
                or verify_rank and self._rank_value(frame) != 40):
            return False
        if confirm:
            return bool(self._fixed(frame,'return-question',(223,556,506,629),.95)
                and self._fixed(frame,'continue',(63,741,356,834),.96)
                and float((frame[480:510,150:180].max(axis=2)>=210).mean()) >= .8)
        return (self.sailing.running(frame) is False
                and not self._fixed(frame,'return-question',(223,556,506,629),.90)
                and not (self.defense and (
                    self.defense._match(frame,'timer',(188,125,610,165),.94)
                    or self.defense._match(frame,'swords',(307,962,415,1057),.96))))

    def _portal_words(self, frame, name, boxes, offset=0):
        reference = self.portal_frames.get(name)
        if reference is None or frame.shape[:2] != (1280,720):
            return False
        for x,y,X,Y in boxes:
            score = cv2.matchTemplate(frame[y+offset:Y+offset,x:X],reference[y:Y,x:X],
                                      cv2.TM_CCOEFF_NORMED)[0,0]
            if not np.isfinite(score) or score < .96:
                return False
        return True

    def _gate_manager(self, frame):
        """Return active/inactive only for the actual bright observatory menu."""
        if self._parallel(frame) or not self._observatory(frame):
            return None
        if self._fixed(frame,'gate-close',(154,685,565,785),.92):
            active,offset = True,0
        elif self._fixed(frame,'gate-activate',(154,751,565,852),.92):
            active,offset = False,66
        else:
            return None
        words = (self._portal_words(frame,'parallel-manager',
                    ((230,563,371,599),(420,563,492,599),(248,628,375,652))) if active
                 else self._portal_words(frame,'parallel-manager-inactive',
                    ((230,631,371,669),(420,631,492,669),(248,696,375,722))))
        if (not words
                or float((frame[535+offset:550+offset,132:152].max(axis=2)>=210).mean()) < .8):
            return None
        return active

    def _portal_number(self, frame, boxes):
        values = []
        for x,y,X,Y in boxes:
            reading = self.reader.read(frame[y:Y,x:X])
            if (reading.value is None or not 1 <= reading.value <= 100
                    or reading.confidence < .94):
                return None
            values.append(reading.value)
        return values[0] if len(values) >= 2 and len(set(values)) == 1 else None

    def _gate_stage(self, frame):
        active = self._gate_manager(frame)
        if active is None:
            return None
        dy = 0 if active else 66
        return self._portal_number(frame,((323,377+dy,408,475+dy),(328,382+dy,403,469+dy)))

    def _lower_prompt(self, frame):
        if (self._parallel(frame) or not self._observatory(frame)
                or not self._portal_words(frame,'parallel-lower-question',
                    ((211,539,508,568),(313,580,416,615),(202,623,514,648)))
                or not self._fixed(frame,'continue',(63,741,356,834),.92)
                or float((frame[480:510,150:180].max(axis=2)>=210).mean()) < .8):
            return None
        return self._portal_number(frame,((282,584,308,615),(284,586,307,614)))

    def _npc_ready(self, frame):
        return (not self._parallel(frame) and self._observatory(frame) and not self._map(frame)
                and self.sailing is not None and self.sailing.running(frame) is False
                and not (self.defense and (
                    self.defense._match(frame,'timer',(188,125,610,165),.94)
                    or self.defense._match(frame,'swords',(307,962,415,1057),.96)))
                and any(self._portal_words(frame,name,((460,515,507,624),))
                        for name in ('observatory-npc','observatory-npc-day')))

    def validate_portal(self, action, frame):
        """Recheck the exact menu/number on a new frame before every input."""
        kind = action['kind']
        if kind.startswith('handoff-'):
            if (self.boss_handoff_since is None
                    or time.monotonic()-self.boss_handoff_since > 60
                    or action.get('rank') != self.rank
                    or self.training.state.get('boss_sailing_handoff_rank') != self.rank
                    or action.get('signature') != self._return_signature(frame)):
                return False
            if kind == 'handoff-map':
                return (self.phase == 'boss_handoff_map' and self.map_open_attempts < 3
                        and time.monotonic()-self.phase_since <= 30
                        and (action['x'],action['y']) == (350,30)
                        and self._boss_handoff_ready(frame))
            if not self._boss_handoff_ready(frame,map_ready=True):
                return False
            if kind == 'handoff-select':
                target = self._map_task_target(frame)
                return (self.phase == 'find_boss' and self.boss_handoff_map_frames >= 3
                        and target is not None and tuple(target[:2]) == (action['x'],action['y']))
            if kind == 'handoff-close':
                return (self.phase == 'close_boss_map' and self.target_rank == self.rank
                        and (action['x'],action['y']) == (668,1187))
            return False
        if kind.startswith('quest-'):
            return self._validate_quest_input(action,frame)
        if kind == 'next-map':
            return (action.get('rank') == self.rank and self.map_open_attempts < 3
                    and (self.phase == 'play' or
                         self.phase in ('find_boss','await_boss_refresh')
                         and time.monotonic()-self.phase_since <= 30)
                    and self._next_map_ready(frame)
                    and action.get('signature') == self._return_signature(frame))
        if kind in ('return-request','return-confirm'):
            return ((self.return_since is None or time.monotonic()-self.return_since <= 180)
                    and self._return_ready(frame,kind=='return-confirm',verify_rank=False)
                    and action.get('signature') == self._return_signature(frame))
        if kind == 'npc':
            return self._npc_ready(frame)
        if kind == 'lower':
            return (self._gate_manager(frame) is True
                    and self._gate_stage(frame) == action['stage']
                    and 0 < self.training.options['parallel_stage_limit'] < action['stage'])
        if kind == 'lower-confirm':
            return (self.lower_from == action['stage']+1
                    and self._lower_prompt(frame) == action['stage']
                    and 0 < self.training.options['parallel_stage_limit'] <= action['stage'])
        if kind == 'panel-close':
            limit = self.training.options['parallel_stage_limit']
            stage = self._gate_stage(frame) if limit else None
            return self._gate_manager(frame) is True and (not limit or stage is not None and stage <= limit)
        if kind == 'activate':
            return self._gate_manager(frame) is False
        if kind == 'close-gate':
            return (self.completed_rank_40 and self.rank == 40
                    and self._gate_manager(frame) is True)
        return False

    def _next_map_ready(self, frame):
        """A missed arrow tap may retry only on the same stopped main screen."""
        if (frame.shape[:2] != (1280,720) or type(self.rank) is not int
                or not 0 <= self.rank < 40 or self.completed_rank_40
                or not self._parallel(frame) or self._map(frame)
                or self._rank_value(frame) != self.rank
                or self.sailing is None or self.sailing.running(frame) is not False
                or self.defense is None
                or self.defense._match(frame,'timer',(188,125,610,165),.94)
                or self.defense._match(frame,'swords',(307,962,415,1057),.96)):
            return False
        return bool(self._fixed(frame,'next-arrow',(307,64,398,127),.88)
                    and float((frame[75:117,318:389].max(axis=2)>=210).mean()) > .5)

    def _open_next_map(self, frame, label):
        if self.portal_action is None:
            # Retain the lightweight controller interface used by callers
            # without the worker's fresh-input callback.
            self._set('find_boss')
            return self._act(351,96,label,2)
        if not self._next_map_ready(frame):
            return True
        if self.phase != 'find_boss':
            self.map_open_attempts = 0
        if self.map_open_attempts >= 3:
            return True
        if self._portal_act('next-map',351,96,label,frame=frame):
            # A retry stays in this phase, preserving its original deadline.
            self._set('find_boss')
            self.map_open_attempts += 1
        return True

    def _portal_act(self, kind, x, y, label, *, stage=None, delay=2, frame=None):
        if self.portal_action is None:
            return self._act(x,y,label,delay)
        action = dict(kind=kind,x=x,y=y,label=label,stage=stage)
        if kind.startswith(('return-','quest-','handoff-')) or kind == 'next-map':
            action['signature'] = self._return_signature(frame)
        if kind == 'next-map' or kind.startswith(('quest-','handoff-')):
            action['rank'] = self.rank
        accepted = self.portal_action(action)
        wait = .05 if kind.startswith('return-') else max(delay,self.interval)
        self.next_action_at = time.monotonic() + (wait if accepted else .2)
        return bool(accepted)

    def _quest_idle_context(self, frame, *, verify_rank=True):
        """Read-only proof of an idle current task; never adjust a boss meter."""
        if (self.portal_action is None or self.fishing is None or self.sailing is None
                or self.defense is None or self.training.state['mode'] != 'idle'
                or self.target_rank not in (None,self.rank)
                or type(self.rank) is not int or not 0 <= self.rank < 39
                or self.completed_rank_40 or self.training.state.get('sailing_enabled')
                or frame.shape[:2] != (1280,720) or not self._parallel(frame) or self._map(frame)
                or not self.fishing._main_visible(frame) or not self.training.boss_mode(frame)
                or self.sailing.running(frame) is not False
                or self.fishing.idle_icon(frame) not in ('sleep','hook')
                or self.fishing.blue_fill(frame) or self.fishing.busy(frame)
                or self.fishing.at_port(frame)
                or self.defense._match(frame,'timer',(188,125,610,165),.94)
                or self.defense._match(frame,'swords',(307,962,415,1057),.96)
                or self._fixed(frame,'next-arrow',(307,64,398,127),.88)
                or (self._rank_value(frame) if verify_rank else self.latest_rank_value) != self.rank):
            return None
        box = self.fishing.box(frame)
        value = self.fishing.selected(frame,box) if box else None
        if value is None:
            return None
        return self._return_signature(frame),tuple(box),value

    def _observe_quest_idle(self, frame, *, verify_rank=True):
        if self.phase not in ('play','quest_start','quest_awake'):
            return
        key = self._quest_idle_context(frame,verify_rank=verify_rank)
        if self.phase == 'quest_awake' and self.fishing.idle_icon(frame) != 'hook':
            key = None
        if key is None or key != self.quest_idle_key:
            self.quest_idle_key = key
            self.quest_idle_since = time.monotonic() if key is not None else None
            self.quest_idle_frames = 1 if key is not None else 0
        else:
            self.quest_idle_frames += 1

    def _quest_idle_verified(self):
        return (self.quest_idle_key is not None and self.quest_idle_frames >= 3
                and self.quest_idle_since is not None
                and time.monotonic()-self.quest_idle_since >= 3)

    def _quest_map_ready(self, frame, *, verify_rank=True):
        return (self.quest_recovery_key is not None and self.popup_map_return(frame,verify_rank=verify_rank)
                and self._return_signature(frame) == self.quest_recovery_key[0]
                and self.sailing._fixed(frame,'auto-label',(186,34,253,65),.90)
                and self.sailing._fixed(frame,'inactive-face',(194,4,246,42),.96)
                and not self.sailing._fixed(frame,'active-arrow',(194,4,246,42),.96))

    def _quest_task_target(self, frame, *, verify_rank=True):
        if not self._quest_map_ready(frame,verify_rank=verify_rank):
            return None
        return self._map_task_target(frame)

    def _map_task_target(self, frame):
        target = self._match(frame,'map-boss',(20,175,700,1115),.88)
        if target is not None:
            return target
        # At the already selected task, the player's portrait covers the
        # skull's upper left. Its entire right eye/outline and red lower jaw
        # remain visible. Require both independent shapes at one location;
        # do not substitute a portrait or a remembered map coordinate.
        if not self.training.boss_mode(frame):
            return None
        template = self.assets['map-boss']
        mask = np.zeros(template.shape[:2],np.uint8)
        mask[10:27,23:31] = 1
        mask[27:33,4:29] = 1
        scores = cv2.matchTemplate(frame[175:1115,20:700],template,
                                  cv2.TM_CCOEFF_NORMED,mask=mask)
        scores = np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1)
        _,score,_,(a,b) = cv2.minMaxLoc(scores)
        if score < .97:
            return None
        x,y = a+20,b+175
        h,w = template.shape[:2]
        crop = frame[y:y+h,x:x+w]
        for r,c in ((slice(10,27),slice(23,31)),(slice(27,33),slice(4,29))):
            matched = cv2.matchTemplate(crop[r,c],template[r,c],cv2.TM_CCOEFF_NORMED)[0,0]
            if not np.isfinite(matched) or matched < .97:
                return None
        rgb = crop.astype(np.int16)
        red = (rgb[:,:,0]>=130)&(rgb[:,:,0]>rgb[:,:,1]*1.4)&(rgb[:,:,0]>rgb[:,:,2]*1.4)
        if int(red.sum()) < 20 or int((crop.min(axis=2)>=170).sum()) < 100:
            return None
        return x+w//2,y+h//2,float(score)

    def _validate_quest_input(self, action, frame):
        if (self.quest_recovery_since is None
                or time.monotonic()-self.quest_recovery_since > 60
                or self.rank != self.quest_recovery_rank or action.get('rank') != self.rank
                or self.training.state['mode'] != 'idle' or self.completed_rank_40
                or self.training.state.get('sailing_enabled')
                or action.get('signature') != self._return_signature(frame)):
            return False
        kind = action['kind']
        if kind == 'quest-map':
            return ((action['x'],action['y']) == (350,30)
                    and self.phase == 'quest_map' and self.quest_open_attempts < 3
                    and time.monotonic()-self.phase_since <= 30
                    and self._quest_idle_context(frame) == self.quest_recovery_key)
        if kind == 'quest-select':
            target = self._quest_task_target(frame)
            return (self.phase == 'quest_select' and self.quest_map_frames >= 3
                    and target is not None
                    and tuple(target[:2]) == self.quest_map_target == (action['x'],action['y']))
        if kind == 'quest-close':
            return ((action['x'],action['y']) == (668,1187)
                    and self.phase == 'quest_close' and self.quest_map_selected
                    and self.target_rank == self.rank and self._quest_map_ready(frame))
        if kind in ('quest-wake','quest-cast'):
            waking = kind == 'quest-wake'
            return ((action['x'],action['y']) == (360,1010)
                    and self.phase in (('quest_start',) if waking else ('quest_start','quest_awake'))
                    and self.quest_map_selected
                    and self.target_rank == self.rank and self.rank not in self.quest_cast_ranks
                    and (not waking or self.rank not in self.quest_wake_ranks)
                    and self.fishing.idle_icon(frame) == ('sleep' if waking else 'hook')
                    and self._quest_idle_verified()
                    and self.quest_idle_key == self.quest_recovery_key
                    and self._quest_idle_context(frame) == self.quest_recovery_key)
        return False

    def _quest_progress(self, frame):
        if (self.phase != 'quest_verify' or not self._parallel(frame)
                or self._map(frame) or not self.fishing._main_visible(frame)):
            return False
        # A waiting cast is not recovery proof and needs no second header OCR.
        # Read the full new rank only when this frame contains a real result,
        # a complete battle, or a rank already established by step().
        battle_candidate = (self.training.boss_mode(frame)
                  and self.defense._match(frame,'timer',(188,125,610,165),.94)
                  and self.defense._match(frame,'damage-sword',(153,205,192,245),.96)
                  and self.defense._battle_footer(frame))
        success_candidate = self.training._match(frame,'success-check',(290,955,432,1060),.92)
        advanced_candidate = type(self.rank) is int and self.quest_recovery_rank < self.rank < 40
        if not (battle_candidate or success_candidate or advanced_candidate):
            return False
        actual = self._rank_value(frame)
        same = (actual == self.quest_recovery_rank
                and self._return_signature(frame) == self.quest_recovery_key[0])
        battle = same and battle_candidate
        success = same and success_candidate
        advanced = advanced_candidate and actual == self.rank
        if battle or success or advanced:
            self.quest_recoveries += 1
            self.quest_recovery_since = None
            self.quest_idle_key = self.quest_idle_since = None
            self.quest_idle_frames = 0
            self._set('play','任务休眠恢复：新截图已确认实际战斗或钓鱼结果，继续原任务')
            return True
        return False

    def _quest_step(self, frame):
        self._timeout(60,'重新确认当前任务并恢复抛竿',since=self.quest_recovery_since)
        if self.rank != self.quest_recovery_rank:
            # Rank/world changes never authorize an old task or an old cast.
            self._set('play')
            self.quest_recovery_since = None
            return True
        if self.phase == 'quest_map':
            if self._quest_map_ready(frame,verify_rank=False):
                self._set('quest_select')
                return True
            self._timeout(30,'休眠恢复打开当前任务地图')
            if self.quest_open_attempts < 3 and self._portal_act('quest-map',350,30,
                    '任务休眠恢复：打开地图重新确认当前任务',frame=frame):
                self.quest_open_attempts += 1
            return True
        if self.phase == 'quest_select':
            target = self._quest_task_target(frame,verify_rank=False)
            point = tuple(target[:2]) if target else None
            self.quest_map_frames = self.quest_map_frames+1 if point and point == self.quest_map_target else int(point is not None)
            self.quest_map_target = point
            if self.quest_map_frames >= 3 and self._portal_act('quest-select',*point,
                    '任务休眠恢复：重新选择已连续确认的当前骷髅任务',frame=frame):
                self.target_rank = self.rank
                self.quest_map_selected = True
                self._set('quest_close')
            return True
        if self.phase == 'quest_close':
            if not self._map(frame):
                self.quest_idle_key = self.quest_idle_since = None
                self.quest_idle_frames = 0
                self._set('quest_start')
            else:
                self._portal_act('quest-close',668,1187,'任务休眠恢复：关闭已重新选定的任务地图',frame=frame)
            return True
        if self.phase in ('quest_start','quest_awake'):
            if self.phase == 'quest_awake':
                self._timeout(15,'唤醒后的灰色鱼钩确认')
            if self._quest_idle_verified():
                icon = self.fishing.idle_icon(frame)
                if icon == 'sleep' and self.phase == 'quest_start':
                    if self._portal_act('quest-wake',360,1010,
                            '任务休眠恢复：单次唤醒Zzz，等待灰色鱼钩重新确认',frame=frame):
                        self.quest_wake_ranks.add(self.rank)
                        self.quest_idle_key = self.quest_idle_since = None
                        self.quest_idle_frames = 0
                        self._set('quest_awake')
                elif icon == 'hook' and self._portal_act('quest-cast',360,1010,
                        '任务休眠恢复：灰色鱼钩已连续确认，单次抛竿并等待实际钓鱼证据',frame=frame):
                    self.quest_cast_ranks.add(self.rank)
                    self._set('quest_verify')
            return True
        self._timeout(15,'单次抛竿后的实际战斗或钓鱼结果')
        self._quest_progress(frame)
        return True

    def completion_visible(self, frame, *, verify_rank=True):
        if (self.completion_frame is None or frame.shape[:2] != (1280,720)
                or not self._parallel(frame)):
            return False
        if float((frame[580:620,200:250].max(axis=2)>=220).mean()) < .8:
            return False
        # Fixed words and the bright confirmation button exclude the changing
        # tier number, portal animation and the dimmed screen underneath.
        for x,y,X,Y in ((306,591,412,619),(305,650,502,687),
                        (230,730,497,760),(306,807,419,868)):
            score = cv2.matchTemplate(frame[y:Y,x:X],self.completion_frame[y:Y,x:X],
                                      cv2.TM_CCOEFF_NORMED)[0,0]
            if not np.isfinite(score) or score < .97:
                return False
        green = cv2.cvtColor(frame[795:888,158:564],cv2.COLOR_RGB2HSV)
        if float(((green[:,:,0]>=25)&(green[:,:,0]<=45)&(green[:,:,2]>=120)).mean()) < .75:
            return False
        return not verify_rank or self._rank_value(frame) == 40

    def completion_step(self, frame, *, ready=False):
        """Dismiss only the verified end notice; never establish portal clearance."""
        if self.completion_action is None:
            return False
        now = time.monotonic()
        if self.completion_visible(frame,verify_rank=False):
            if self.completion_since is None:
                self.completion_since = now
            self.completion_clear_frames = 0
            valid = self._rank_value(frame) == 40
            self.completion_frames = self.completion_frames+1 if valid else 0
            if self.completion_frames >= 3 and not self.completion_clicked:
                self.completion_clicked = bool(self.completion_action())
        elif self.completion_since is not None:
            self.completion_frames = 0
            self.completion_clear_frames = self.completion_clear_frames+1 if ready else 0
            if self.completion_clear_frames >= 2:
                self.completion_since = None
                self.completion_clear_frames = 0
                self.completion_clicked = False
                return False
        if self.completion_since is not None:
            if now-self.completion_since > 15:
                raise RuntimeError('平行世界通关确认窗口未安全消失，已停止并保留画面')
            return True
        return False

    def _sailing_wanted(self):
        state = self.training.state
        return (not self.sailing_shutdown and self.training.options['enabled']
                and self.training.options['auto_sailing'] and state['mode'] == 'farming'
                and state.get('rank') == self.rank and self.rank is not None and self.rank < 40
                and not self.training.ready_to_retry())

    def _boss_sailing_wanted(self):
        state,options = self.training.state,self.training.options
        limit = options['boss_auto_sailing_until_rank']
        return (not self.sailing_shutdown and options['enabled'] and limit > 0
                and state['mode'] == 'idle' and type(self.rank) is int
                and 0 <= self.rank < limit and not self.completed_rank_40
                and state.get('boss_sailing_sleep_rank') != self.rank)

    def _boss_handoff_ready(self, frame, *, map_ready=False, allow_battle=False):
        if (frame.shape[:2] != (1280,720) or type(self.rank) is not int
                or not 0 <= self.rank < 40 or self.completed_rank_40
                or self.training.state['mode'] != 'idle'
                or self.training.state.get('sailing_enabled')
                or self.sailing is None or self.sailing.active or self.defense is None
                or not self._parallel(frame) or self._rank_value(frame) != self.rank
                or self._stage_strokes(frame) is None or self.latest_rank_frame is None
                or self._stage_strokes(frame) != self._stage_strokes(self.latest_rank_frame)):
            return False
        if (self.boss_handoff_signature is not None
                and self._return_signature(frame)[:3] != self.boss_handoff_signature[:3]):
            return False
        if map_ready:
            return (self.popup_map_return(frame)
                    and self.sailing._fixed(frame,'inactive-face',(194,4,246,42),.96)
                    and not self.sailing._fixed(frame,'active-arrow',(194,4,246,42),.96))
        return (not self._map(frame) and self.sailing.running(frame) is False
                and int((frame[12:34,286:434].max(axis=2)<70).sum()) >= 10
                and (allow_battle or (not self.defense._match(frame,'timer',(188,125,610,165),.94)
                and not self.defense._match(frame,'swords',(307,962,415,1057),.96))))

    def _boss_handoff_begin(self, frame):
        if not self._boss_handoff_ready(frame,map_ready=self._map(frame)):
            return False
        self.training.state['boss_sailing_handoff_rank'] = self.rank
        self.training._save()
        self.boss_handoff_since = time.monotonic()
        self.boss_handoff_signature = self._return_signature(frame)
        self.boss_handoff_map_frames = self.map_open_attempts = self.map_swipes = 0
        self._set('boss_handoff_map','BOSS 航行交接：停航后重新定位当前骷髅任务，避免停在港口等待')
        return True

    def _route_step(self, frame):
        if self._boss_sailing_wanted():
            return self.sailing.step(frame,True,profile='boss',rank=self.rank)
        return self.sailing.step(frame,self._sailing_wanted())

    def boss_sailing_stop_requested(self):
        state = self.training.state
        if not state.get('sailing_enabled') or state.get('sailing_profile') != 'boss':
            return False
        limit = self.training.options['boss_auto_sailing_until_rank']
        return (self.sailing_shutdown or not limit
                or not self.training.options['enabled']
                or state.get('boss_sailing_sleep_rank') == self.rank
                or any(type(value) is int and value >= limit
                       for value in (self.rank,self.latest_rank_value)))

    def observe_boss_sailing_rank(self, frame, *, ready=False):
        """Keep the cutoff visible while a recognized upgrade panel owns UI.

        Only the existing multi-crop rank decoder and three distinct frames
        establish a new rank. The shared frame is not voted twice by step().
        """
        if (not self.training.state.get('sailing_enabled')
                or self.training.state.get('sailing_profile') != 'boss'):
            return
        if (not ready or frame.shape[:2] != (1280,720) or not self._parallel(frame)
                or self.sailing.menu(frame) or self._stage_strokes(frame) is None
                or self.latest_rank_frame is not None
                and self._stage_strokes(frame) != self._stage_strokes(self.latest_rank_frame)):
            self.rank_readings.clear()
            return
        if self.boss_sailing_rank_frame is not frame:
            self._rank(frame)
            self.boss_sailing_rank_frame = frame

    def boss_sailing_close_guard(self, frame):
        return (self.boss_sailing_stop_requested() and self._parallel(frame)
                and self.rank is not None and self._rank_value(frame) == self.rank
                and self._stage_strokes(frame) is not None
                and self.latest_rank_frame is not None
                and self._stage_strokes(frame) == self._stage_strokes(self.latest_rank_frame))

    def boss_sailing_hold_allowed(self, frame):
        if not (self.training.state.get('sailing_enabled')
                and self.training.state.get('sailing_profile') == 'boss'):
            return True
        return (self._boss_sailing_wanted() and not self.boss_sailing_stop_requested()
                and self._parallel(frame) and self._rank_value(frame) == self.rank
                and self._stage_strokes(frame) is not None
                and self.latest_rank_frame is not None
                and self._stage_strokes(frame) == self._stage_strokes(self.latest_rank_frame))

    def _observe_boss_route_idle(self, frame):
        # A green route is not fishing proof. Hand persistent Zzz/gray hook
        # back to the existing bounded task-selection/wake/cast controller.
        key = None
        if (self.fishing and self.defense and self._boss_sailing_wanted()
                and self.training.state.get('sailing_verified')
                and self.sailing.running(frame) is True
                and self.latest_rank_value == self.rank
                and self.training.boss_mode(frame) and not self.fishing.at_port(frame)
                and self.fishing.idle_icon(frame,main_known=True) in ('sleep','hook')
                and not self.fishing.blue_fill(frame,main_known=True)
                and not self.fishing.busy(frame,main_known=True)
                and not self.defense._match(frame,'timer',(188,125,610,165),.94)
                and not self.defense._match(frame,'swords',(307,962,415,1057),.96)
                and not self._fixed(frame,'next-arrow',(307,64,398,127),.88)):
            key = self._return_signature(frame)
        if key is None or key != self.boss_sailing_idle_key:
            self.boss_sailing_idle_key = key
            self.boss_sailing_idle_since = time.monotonic() if key is not None else None
            self.boss_sailing_idle_frames = 1 if key is not None else 0
        else:
            self.boss_sailing_idle_frames += 1
        if (key is not None and self.boss_sailing_idle_frames >= 3
                and time.monotonic()-self.boss_sailing_idle_since >= 3):
            self.training.state['boss_sailing_sleep_rank'] = self.rank
            self.training._save()
            self.event('BOSS 航行持续休眠且无填充：先停航，再交回当前任务恢复')

    def resource_battle_recovery_context(self, frame):
        """Read-only proof of ordinary fishing on an already verified route.

        A resource battle can close a housekeeping page just as a boss can.
        It is not a reward completion, a new route or a fishing start.
        """
        state = self.training.state
        if (frame.shape[:2] != (1280,720) or self.phase != 'play'
                or type(self.rank) is not int or not 0 <= self.rank < 40
                or self.sailing is None or self.defense is None or self.fishing is None
                or not self._sailing_wanted() or self.training.retry_level() is None
                or not state.get('sailing_enabled')
                or not state.get('sailing_verified') or self.sailing.running(frame) is not True
                or not self._parallel(frame) or self.training.boss_mode(frame)
                or self.fishing.at_port(frame) or self._rank_value(frame) != self.rank
                or not self.training.main(frame) or not self.defense._battle_footer(frame)
                or not self.defense._match(frame,'timer',(188,125,610,165),.94)
                or not self.defense._match(frame,'damage-sword',(153,205,192,245),.96)):
            return None
        # Fixed world-stage, RANK and location strokes; exclude the moving
        # travel countdown, fish animation, health amount and battle timer.
        return (self.rank, self.defense._rank_strokes(frame),
                np.all(frame[45:68,37:59] >= 160,axis=2).tobytes(),
                (np.max(frame[12:34,286:434],axis=2) < 80).tobytes())

    def _stop_sailing(self, frame):
        return bool(self.sailing and self.sailing.step(frame,False))


    def sailing_context(self, action, frame):
        if not self._parallel(frame) or self.rank is None:
            return False
        if action == 'bag' and self.latest_rank_frame is not None:
            stage = self._stage_strokes(frame)
            if stage is None or stage != self._stage_strokes(self.latest_rank_frame):
                return False
        menu = self.sailing.menu(frame)
        if menu and (not self.sailing._fixed(frame,'auto-label',(186,34,253,65),.90)
                     or not self.sailing._fixed(frame,'inactive-face',(194,4,246,42),.96)
                     or self.sailing._fixed(frame,'active-arrow',(194,4,246,42),.96)):
            return False
        actual = self._sailing_menu_rank(frame) if menu else self._rank_value(frame)
        if actual != self.rank:
            return False
        if action == 'start' or action.startswith('target:'):
            return (self._boss_sailing_wanted() if self.sailing.profile == 'boss'
                    else self._sailing_wanted())
        return True

    def _set(self, phase, message=None):
        if self.phase != phase:
            self.phase = phase
            self.phase_since = time.monotonic()
            if phase == 'find_boss':
                self.map_open_attempts = 0
            self.missing_parallel_since = None
            if message:
                self.event(message)

    def _match(self, frame, name, area, threshold=.88):
        """本类自己的模板匹配器(含地图 BOSS 图标的二次校验)。

        基类通过 ``match`` 转发到这里,所以基类的 ``_fixed`` / ``_parallel`` 等
        判定与本类原有行为完全一致;测试与内部调用仍可继续用 ``_match``。
        """
        x1, y1, x2, y2 = area
        patch = self.assets[name]
        if x2 - x1 < patch.shape[1] or y2 - y1 < patch.shape[0]:
            return None
        scores = cv2.matchTemplate(frame[y1:y2, x1:x2], patch, cv2.TM_CCOEFF_NORMED)
        _, score, _, position = cv2.minMaxLoc(scores)
        if np.isfinite(score) and score >= threshold:
            return (x1 + position[0] + patch.shape[1] // 2,
                    y1 + position[1] + patch.shape[0] // 2, score)
        if name != 'map-boss':
            return None
        # Map terrain changes behind this small icon. Its white skull/black
        # outline remain fixed; also require the surrounding complete icon at
        # the same location, so an isolated white face cannot become a target.
        body = patch[4:25,3:27]
        scores = cv2.matchTemplate(frame[y1:y2,x1:x2],body,cv2.TM_CCOEFF_NORMED)
        _, score, _, position = cv2.minMaxLoc(scores)
        if not np.isfinite(score) or score < max(.91,threshold):
            return None
        x,y = x1+position[0]-3,y1+position[1]-4
        h,w = patch.shape[:2]
        if not (x1 <= x <= x2-w and y1 <= y <= y2-h):
            return None
        guard = float(cv2.matchTemplate(frame[y:y+h,x:x+w],patch,cv2.TM_CCOEFF_NORMED)[0,0])
        if not np.isfinite(guard) or guard < .80:
            return None
        return x+w//2,y+h//2,score

    def match(self, frame, name, area, threshold=.88):
        """基类判定入口:转发到本类的 :meth:`_match`,保持与原实现一致。"""
        return self._match(frame, name, area, threshold)

    # _fixed / _parallel / _observatory / _map / _rank_value / _stage_strokes
    # 由 ParallelScreen 基类提供(与标尺校准共用同一份画面判定);本类只保留
    # popup_map_return 用到的地图图例校验。

    def resolved_attack_options(self, frame=None):
        """当前帧确认守护者或海神时，额外开启各自的攻击。"""
        encounter = bool(frame is not None and self.defense
                         and self.defense.encounter(frame,self.rank))
        guardian = bool(frame is not None and self.defense
                        and self.defense.guardian_encounter(frame,self.rank))
        return final_boss_options(self.training,self.rank,encounter=encounter,guardian=guardian)

    def final_boss_attack_enabled(self, frame=None):
        """返回这一帧的有效开关；未提供新帧时只报告配置值。"""
        resolved = self.resolved_attack_options(frame)
        return bool(resolved['boss_interrupt'] and resolved['battle_tap'])

    def final_boss_attack_forced(self, frame=None):
        """40 BOSS 规则额外强制打开了哪些攻击(配置里原本关着的那些)。"""
        encounter = bool(frame is not None and self.defense
                         and self.defense.encounter(frame,self.rank))
        guardian = bool(frame is not None and self.defense
                        and self.defense.guardian_encounter(frame,self.rank))
        return forced_attacks(self.training,self.rank,encounter=encounter,guardian=guardian)

    @contextmanager
    def attack_options(self, frame=None):
        """仅在攻击调用期间启用逐帧解析，不改写训练和 OCR 的配置。"""
        defense = self.defense
        previous = getattr(defense, 'final_boss_attacks', False)
        if defense is not None:
            defense.final_boss_attacks = True
        try:
            yield self.resolved_attack_options(frame)
        finally:
            if defense is not None:
                defense.final_boss_attacks = previous

    def popup_map_return(self, frame, *, verify_rank=True):
        """Only a complete bright map in the current world can resume a popup."""
        if (self.map_legend is None or frame.shape[:2] != (1280,720)
                or self.rank is None or not self._parallel(frame) or not self._map(frame)
                or (self.latest_rank_value if not verify_rank and frame is self.latest_rank_frame
                    else self._rank_value(frame)) != self.rank):
            return False
        scores=cv2.matchTemplate(frame[1136:1241,9:405],self.map_legend,cv2.TM_CCOEFF_NORMED)
        score=cv2.minMaxLoc(scores)[1]
        return bool(np.isfinite(score) and score >= .93)

    @staticmethod
    def _gate_active(frame):
        patch = cv2.cvtColor(frame[426:535, 303:409], cv2.COLOR_RGB2HSV)
        return float((patch[:, :, 2] > 140).mean()) > .40


    def _rank(self, frame):
        value = self._rank_value(frame)
        self.latest_rank_value = value
        self.latest_rank_frame = frame
        self.rank_readings.append(value)
        if len(self.rank_readings) == 3 and value is not None and all(item == value for item in self.rank_readings):
            if self.rank is not None and value < self.rank:
                # Rank never decreases within one portal. A lower OCR result
                # must not restart a quest or satisfy a reset condition.
                return self.rank
            if value != self.rank:
                self.rank = value
                self.map_refreshes = 0
                self.last_progress_at = time.monotonic()
                self.event(f'平行世界 RANK {value}')
            if self.defense:
                self.defense.confirm_rank(frame,value)
            if value == 40:
                self.completed_rank_40 = True
        return self.rank

    def _act(self, x, y, label, delay=1.2):
        self.click(x, y, label)
        self.next_action_at = time.monotonic() + max(delay, self.interval)
        return True

    def _slide(self, direction):
        if direction == 'right':
            self.swipe(120, 600, 610, 600, '地图向左查找观测所')
        else:
            self.swipe(600, 600, 120, 600, '地图查找下一任务')
        self.map_swipes += 1
        self.next_action_at = time.monotonic() + self.interval
        return True

    def _timeout(self, seconds, label, *, since=None):
        if time.monotonic() - (self.phase_since if since is None else since) > seconds:
            raise RuntimeError(f'平行世界「{label}」超过 {seconds} 秒仍未确认界面，已停止以避免误点')

    def status(self):
        return {'phase': self.phase, 'rank': self.rank, 'target_rank': self.target_rank,
                'completed_rank_40': self.completed_rank_40, 'rounds': self.rounds,
                # 打 40 BOSS 时这两个攻击会被强制打开;forced 说明哪一个是本功能额外开的
                'final_boss_attack': self.final_boss_attack_enabled(),
                'final_boss_attack_forced': self.final_boss_attack_forced(),
                'final_boss': is_final_boss(self.rank),
                'map_swipes': self.map_swipes, 'map_refreshes': self.map_refreshes,
                'map_open_attempts': self.map_open_attempts,
                'quest_recoveries': self.quest_recoveries,
                'quest_wake_attempts': len(self.quest_wake_ranks),
                'quest_cast_attempts': len(self.quest_cast_ranks),
                'stage_limit': self.training.options['parallel_stage_limit'],
                'gate_stage_verified': self.gate_stage_verified,
                'training': self.training.status(),
                'sailing': self.sailing.status() if self.sailing else None,
                'fishing': self.fishing.status() if self.fishing else None,
                'defense': self.defense.status() if self.defense else None}

    def _entered_parallel(self):
        self.return_since = None
        self.quest_idle_key = self.quest_idle_since = None
        self.quest_idle_frames = 0
        self.quest_recovery_since = self.quest_recovery_key = None
        self.quest_cast_ranks.clear()
        self.quest_wake_ranks.clear()
        self.training.clear()
        self.target_rank = None
        self.completed_rank_40 = False
        self.rank = None
        self.rank_readings.clear()
        self._set('play', '已进入平行世界，等待 RANK 读取')

    def observe(self, frame):
        """Record visible outcomes even when another workflow owns the UI."""
        # Obscuring workflows reset idle proof immediately. The free main
        # controller collects its numeric proof after this SAME frame's rank
        # OCR below; do not run the expensive header OCR twice per new frame.
        if (self.phase in ('play','quest_start','quest_awake') and self.fishing
                and not self.fishing._main_visible(frame)):
            self.quest_idle_key = self.quest_idle_since = None
            self.quest_idle_frames = 0
        if self.quest_recovery_since is not None:
            self._quest_progress(frame)
        encounter = bool(self.rank == 39 and self.defense and not self._map(frame)
                         and self.defense.encounter(frame,self.rank))
        if (encounter and self.phase in ('play','find_boss','close_boss_map','refresh_boss_map','await_boss_refresh')):
            # The native boss name and timer prove that navigation has already
            # reached its target; a next-task arrow need not appear mid-fight.
            self.target_rank = self.rank
            if self.phase != 'play':
                self.map_swipes = self.map_refreshes = 0
                self._set('play','已确认当前海神战斗，完成任务地点返回')
        if self.phase == 'play' and self._parallel(frame):
            self.missing_parallel_since = None
            if self.rank is not None and self.rank < 40:
                self.training.observe_outcome(frame, self.rank)
            # Header OCR takes much longer than the short native charging
            # cue. Do not run it between two defense windows while this new
            # frame still shows a real charging core. Outcome observation
            # above remains immediate; each touch retains its own full guard.
            # charging 决定"这段要不要让出读数窗口",绑的是配置里的开关原值 ——
            # 打 40 BOSS 时攻击是否执行由下面 damage_priority 的解析器单独决定,
            # 两者不要混(混了就会在关掉组织攻击时改变 BOSS 等级的读取时机)。
            charging = bool(encounter and self.training.options['boss_interrupt']
                            and self.defense._locate_core(frame))
            with self.attack_options(frame) as attack:
                damage_priority = bool(encounter and attack['battle_tap']
                                       and self.training._cached_boss(self.rank)
                                       and self.defense.damage_battle(frame,self.rank))
            if charging or damage_priority:
                # Resume the original two consecutive-frame level consensus
                # after charging; never join readings across an interruption.
                self.training.boss_header_readings.clear()
            else:
                self.training.observe_boss_header(frame,self.rank,selected=encounter)
            if self.fishing:
                known = bool(self.sailing and self.sailing.running(frame) is not None)
                self.fishing.observe(frame,main_known=known)

    def observe_backpack_rank(self, frame, backpack):
        """Establish the visible rank while a saved sale owns the UI.

        A pending bag workflow can precede the normal navigation detector on
        restart. It may read a complete main screen or recognized OFF panel,
        but cannot navigate,
        change the saved training target or infer a rank from that target.
        """
        if (frame.shape[:2] != (1280,720) or not backpack.active
                or self.phase not in ('detect','play')):
            self.backpack_rank_readings.clear()
            self.backpack_rank_key = None
            return False
        running = self.sailing.running(frame) if self.sailing else None
        known_main = backpack.main(frame) and (running is True or running is False)
        # Reading a running main screen does not authorize a sale: the
        # original fresh-input OFF guard must still stop and confirm AUTO.
        if not (known_main or backpack.stopped_panel(frame)) or not self._parallel(frame):
            self.backpack_rank_readings.clear()
            self.backpack_rank_key = None
            return False
        stage_key = self._stage_strokes(frame)
        if stage_key is None:
            self.backpack_rank_readings.clear()
            self.backpack_rank_key = None
            return False
        if self.rank is not None:
            # A boss can finish while equipment owns the UI. Reconcile only a
            # single forward rank on a positively stopped, bright main/panel,
            # in the same stage as the last actual rank frame. Never use this
            # read to advance a farming journal, a running route or RANK40.
            if (self.training.state.get('mode') != 'idle'
                    or self.training.state.get('sailing_enabled')
                    or not 0 <= self.rank < 39 or self.latest_rank_frame is None
                    or not (running is False or backpack.stopped_panel(frame))
                    or stage_key != self._stage_strokes(self.latest_rank_frame)):
                self.backpack_rank_readings.clear()
                self.backpack_rank_key = None
                return False
            if np.array_equal(np.all(frame[9:59,71:136]>=240,axis=2),
                              np.all(self.latest_rank_frame[9:59,71:136]>=240,axis=2)):
                self.backpack_rank_readings.clear()
                self.backpack_rank_key = None
                return False
        value = self._rank_value(frame)
        saved = self.training.state
        if (value is None or not 0 <= value < 40
                or self.rank is not None and value != self.rank+1
                or saved.get('mode') == 'farming' and saved.get('rank') != value):
            self.backpack_rank_readings.clear()
            self.backpack_rank_key = None
            return False
        key = stage_key,self._return_signature(frame)
        if key != self.backpack_rank_key:
            self.backpack_rank_readings.clear()
            self.backpack_rank_key = key
        self.backpack_rank_readings.append(value)
        if len(self.backpack_rank_readings) < 3 or any(
                item != value for item in self.backpack_rank_readings):
            return False
        previous = self.rank
        self.rank = self.latest_rank_value = value
        self.latest_rank_frame = frame
        self.event(f'背包恢复：已连续确认当前平行世界 RANK {value}'
                   + (f'，更新菜单期间旧缓存 RANK {previous}' if previous is not None else ''))
        if self.phase == 'detect':
            self._set('play','背包恢复：已识别当前世界，保留出售和练级记录')
        return True

    def suspend(self, seconds):
        if self.phase not in ('play','detect') and seconds > 0:
            self.phase_since += seconds

    def step(self, frame):
        """Return True while this feature owns the UI; False lets other tasks run."""
        if frame.shape[:2] != (1280, 720):
            raise RuntimeError('平行世界规则只支持 720×1280 模拟器画面')
        if self.sailing and self.sailing.active:
            # The verified hold may have completed at the deadline boundary.
            # Commit that read-only result before the outer timeout or wait,
            # with the same fresh world/rank/OFF/fill checks as its controller.
            if self.sailing._adopt_new_start(frame,self._sailing_wanted()):
                return True
            # Context rejection must not starve the sailing workflow's
            # original timeout while its panel hides the main-screen header.
            self.sailing._timeout()
        now = time.monotonic()
        if self.boss_handoff_since is not None:
            self._timeout(60,'BOSS 航行交接定位当前任务',since=self.boss_handoff_since)
        if self.phase.startswith('quest_'):
            self._timeout(60,'重新确认当前任务并恢复抛竿',since=self.quest_recovery_since)
        if self.phase in ('return_start','return_confirm','await_original'):
            if self.return_since is None:
                self.return_since = now
            self._timeout(180,'返回现实世界',since=self.return_since)
        if now < self.next_action_at:
            return self.phase not in ('play', 'detect')
        parallel = self._parallel(frame)
        # The accelerated three-dot waiting cast can end before full rank and
        # level OCR finishes. Service a saved preparation first when its fixed
        # rank strokes still match a previously verified current world. RANK
        # changes fall through to the normal three-reading completion guard.
        if (parallel and self.phase == 'play' and self.sailing and self.fishing
                and self.training.state.get('sailing_sleep_recovery')
                and self.rank == self.training.state.get('rank') and self.rank is not None
                and self.fishing.world_matches(frame)):
            if self.sailing.step(frame,self._sailing_wanted()):
                return True
        previous_rank = self.rank
        if parallel:
            if self.boss_sailing_rank_frame is not frame:
                self._rank(frame)
            self._observe_quest_idle(frame,verify_rank=False)
            pending_handoff = self.training.state.get('boss_sailing_handoff_rank')
            if (self.boss_handoff_since is not None and type(pending_handoff) is int
                    and self.rank is not None and self.rank > pending_handoff
                    and self.boss_handoff_signature is not None
                    and self._return_signature(frame)[2] == self.boss_handoff_signature[2]):
                self.training.state.pop('boss_sailing_handoff_rank',None)
                self.training._save()
                self.boss_handoff_since = self.boss_handoff_signature = None
                self._set('play','BOSS 航行交接期间已连续确认任务前进，继续当前 RANK 流程')
        else:
            self.latest_rank_value = None
            self.latest_rank_frame = None
            self.quest_idle_key = self.quest_idle_since = None
            self.quest_idle_frames = 0
            self.rank_readings.clear()
        if (self.sailing and self.sailing.menu(frame)
                and (self.sailing.active or self.training.state.get('mode') == 'farming'
                     or self.sailing_shutdown)):
            self.training.readings.clear()
            if self._route_step(frame):
                return True
        if (parallel and self.fishing and self.training.state.get('mode') == 'farming'
                and self.rank == self.training.state.get('rank')
                and not self.fishing.world_matches(frame)):
            # A stale/corrupt main frame must not promote the saved character
            # level and trigger a premature boss retry. Continue collecting
            # the normal rank votes above so a genuine new rank can still win.
            self.training.readings.clear()
            return True
        if self.sailing_shutdown and self.sailing:
            if not self.sailing.step(frame,False):
                self.shutdown_complete = True
            # Keep this frame owned until pending settings can replace us.
            return True
        if (not parallel and self.sailing and self.training.state.get('sailing_enabled')
                and self.training.state.get('sailing_profile') == 'boss'):
            # An unexpected world is never proof that a saved route stopped.
            # Keep ownership and the existing bounded stop guard; do not start
            # original-world maps or portal navigation underneath it.
            return self.sailing.step(frame,False)
        if self.sailing and self.sailing.active:
            if self._route_step(frame):
                return True
        if (parallel and self.rank == 40 and self.completed_rank_40
                and self.phase in ('find_boss', 'close_boss_map', 'find_training', 'close_training_map',
                                   'refresh_boss_map', 'await_boss_refresh')):
            self._set('play')
        elif (parallel and previous_rank is not None and self.rank > previous_rank
                and self.phase in ('find_boss','close_boss_map','refresh_boss_map','await_boss_refresh')):
            self.training.clear()
            self.target_rank = None
            self.map_swipes = 0
            self._set('refresh_boss_map',f'RANK {self.rank} 已确认，关闭旧地图并重新打开下一任务')

        if self.phase.startswith('quest_'):
            return self._quest_step(frame)

        if self.phase == 'detect':
            if parallel:
                self._set('play', '已识别平行世界，开始按 RANK 寻找下一任务')
            elif self.portal_action is not None and self._gate_manager(frame) is not None:
                self._set(('verify_gate_stage' if self.training.options['parallel_stage_limit']
                           else 'close_gate_panel') if self._gate_manager(frame) else 'activate_click')
            elif self._observatory(frame) and not self._map(frame):
                self._set('enter_gate' if self._gate_active(frame) else 'open_gate')
            else:
                self._set('open_original_map')

        if self.phase == 'open_original_map':
            if self._map(frame):
                self._set('find_observatory')
                return True
            self._timeout(30, '打开地图')
            return self._act(350, 30, '平行世界：打开原世界地图', 2)

        if self.phase == 'find_observatory':
            if not self._map(frame):
                self._timeout(20, '查找观测所地图')
                return True
            location = self._match(frame, 'map-observatory', (20, 175, 700, 1115), .88)
            if location:
                self._set('close_travel_map')
                return self._act(location[0], location[1], '平行世界：选择传送观测所', 1.5)
            if self.map_swipes >= 16:
                raise RuntimeError('地图中未找到传送观测所，已停止以避免误选地点')
            return self._slide('right')

        if self.phase == 'close_travel_map':
            if not self._map(frame):
                self._set('travel_observatory')
                return True
            self._set('travel_observatory')
            return self._act(668, 1187, '平行世界：关闭地图等待到达观测所', 2)

        if self.phase == 'travel_observatory':
            if self._observatory(frame) and not self._map(frame):
                # The header names the destination during travel, so also require
                # the characteristic stone gate in the central screen region.
                gate = frame[386:577, 252:461]
                cyan = cv2.cvtColor(gate, cv2.COLOR_RGB2HSV)
                if float(((cyan[:, :, 0] >= 80) & (cyan[:, :, 0] <= 105)
                          & (cyan[:, :, 1] > 65)).mean()) > .23:
                    self._set('enter_gate' if self._gate_active(frame) else 'open_gate')
                    return True
            self._timeout(600, '前往传送观测所')
            return True

        if self.phase == 'open_gate':
            if not self._observatory(frame):
                self._timeout(30, '未激活传送门')
                return True
            if self._gate_active(frame):
                self._set('enter_gate')
                return True
            self._set('search_confirm')
            return self._act(355, 478, '平行世界：开启未激活传送门', 2)

        if self.phase == 'search_confirm':
            if self._fixed(frame, 'search-question', (194, 555, 525, 620), .90):
                self._set('await_gate_active')
                return self._act(208, 786, '平行世界：确认搜索新世界', 3)
            if self._observatory(frame) and self._gate_active(frame):
                self._set('await_gate_active')
                return True
            self._timeout(30, '搜索新世界确认')
            return True

        if self.phase == 'await_gate_active':
            if self._fixed(frame, 'panel-close', (548, 352, 638, 433), .85):
                self._set('verify_gate_stage' if self.training.options['parallel_stage_limit'] else 'close_gate_panel')
                return True
            if self._observatory(frame) and self._gate_active(frame):
                self._set('enter_gate')
                return True
            self._timeout(45, '等待传送门激活')
            return True

        if self.phase == 'close_gate_panel':
            if not self._fixed(frame, 'panel-close', (548, 352, 638, 433), .85):
                self._set('enter_gate')
                return True
            if self._portal_act('panel-close',590,395,'平行世界：关闭传送门说明'):
                self._set('enter_gate')
            return True

        if self.phase == 'verify_gate_stage':
            self._timeout(45,'读取并核对传送门阶段')
            stage = self._gate_stage(frame) if self._gate_manager(frame) is True else None
            self.gate_stage_readings.append(stage)
            if (stage is None or len(self.gate_stage_readings) < 3
                    or not all(value == stage for value in self.gate_stage_readings)):
                return True
            limit = self.training.options['parallel_stage_limit']
            if limit and stage > limit:
                if self._portal_act('lower',357,873,'平行世界：降低当前传送门难度',stage=stage):
                    self.lower_from = stage
                    self.gate_stage_readings.clear()
                    self._set('lower_stage_confirm')
                return True
            self.gate_stage_verified = True
            self._set('close_gate_panel',f'已连续确认传送门第 {stage} 阶，保留当前难度')
            return True

        if self.phase == 'lower_stage_confirm':
            self._timeout(30,'确认降低平行世界难度')
            target = self._lower_prompt(frame)
            self.gate_stage_readings.append(target)
            if (self.lower_from is None or target != self.lower_from-1
                    or len(self.gate_stage_readings) < 3
                    or not all(value == target for value in self.gate_stage_readings)):
                return True
            if self._portal_act('lower-confirm',208,788,'平行世界：确认降低一阶',stage=target):
                self.gate_stage_readings.clear()
                self._set('await_lower_stage')
            return True

        if self.phase == 'await_lower_stage':
            self._timeout(30,'确认降阶实际结果')
            stage = self._gate_stage(frame) if self._gate_manager(frame) is True else None
            self.gate_stage_readings.append(stage)
            if (stage is not None and stage == self.lower_from-1
                    and len(self.gate_stage_readings) == 3
                    and all(value == stage for value in self.gate_stage_readings)):
                self.lower_from = None
                self._set('verify_gate_stage',f'新截图连续确认已降到第 {stage} 阶')
            return True

        if self.phase == 'stage_menu':
            self._timeout(30,'打开阶段管理菜单')
            if self._gate_manager(frame) is False:
                self._set('activate_click')
                return True
            if self._gate_manager(frame) is True:
                self.cat_frames = 0
                self.gate_stage_readings.clear()
                self._set('verify_gate_stage')
                return True
            self.cat_frames = self.cat_frames+1 if self._npc_ready(frame) else 0
            if self.cat_frames >= 3 and self.cat_attempts < 3:
                if self._portal_act('npc',482,548,'平行世界：查看当前传送门阶段',delay=3):
                    self.cat_frames = 0
                    self.cat_attempts += 1
            return True

        if self.phase == 'enter_gate':
            if not self._observatory(frame):
                self._timeout(40, '进入传送门')
                return True
            if not self._gate_active(frame):
                self._set('open_gate')
                return True
            if self.training.options['parallel_stage_limit'] and not self.gate_stage_verified:
                self.cat_frames = self.cat_attempts = 0
                self._set('stage_menu')
                return True
            self._set('enter_confirm')
            return self._act(354, 482, '平行世界：进入已激活传送门', 2)

        if self.phase == 'enter_confirm':
            if self._fixed(frame, 'enter-question', (220, 556, 502, 620), .90):
                self._set('await_parallel')
                return self._act(208, 787, '平行世界：确认进入', 4)
            if parallel:
                self._entered_parallel()
                return True
            self._timeout(30, '进入平行世界确认')
            return True

        if self.phase == 'await_parallel':
            if parallel:
                self._entered_parallel()
                return True
            self._timeout(90, '进入平行世界加载')
            return True

        if self.phase == 'boss_handoff_map':
            if self._boss_handoff_ready(frame,map_ready=True):
                self.boss_handoff_map_frames += 1
                if self.boss_handoff_map_frames >= 3:
                    self._set('find_boss')
                return True
            self.boss_handoff_map_frames = 0
            self._timeout(30,'BOSS 航行交接打开当前任务地图')
            if (self.map_open_attempts < 3 and self._boss_handoff_ready(frame)
                    and self._return_signature(frame) == self.boss_handoff_signature
                    and self._portal_act('handoff-map',350,30,
                        'BOSS 航行交接：打开当前地点地图重新寻找任务',frame=frame)):
                self.map_open_attempts += 1
            return True

        if self.phase == 'play':
            if not parallel:
                if self.missing_parallel_since is None:
                    self.missing_parallel_since = now
                self._timeout(40, '平行世界主画面', since=self.missing_parallel_since)
                return True
            self.missing_parallel_since = None
            if self.rank is None:
                self._timeout(90, '读取 RANK')
                return True
            if self.rank == 40 and self.completed_rank_40:
                if (self._map(frame) and self.sailing
                        and (self.sailing.active or self.training.state.get('sailing_enabled'))):
                    return self._act(668, 1187, '平行世界：关闭已打开的地图', 2)
                if self._stop_sailing(frame):
                    return True
                self.training.clear()
                if self._map(frame):
                    return self._act(668,1187,'平行世界：关闭已打开的地图',2)
                self._set('return_start', 'RANK 40 已连续确认，准备返回并重开次元门')
                return True
            if self.training.state.get('rank', self.rank) != self.rank:
                if self._map(frame):
                    return self._act(668,1187,'平行世界：关闭地图，先停止上一轮练级航行',2)
                if self._stop_sailing(frame):
                    return True
                self.training.clear()
            if self.sailing and self.training.state.get('sailing_profile') == 'boss':
                self._observe_boss_route_idle(frame)
                if not self._boss_sailing_wanted():
                    if self._stop_sailing(frame):
                        return True
                    self.target_rank = None
                    if self.rank < 40:
                        self.training.state['boss_sailing_handoff_rank'] = self.rank
                    self.event('前期 BOSS 航行已停航，恢复当前 RANK 的任务与练级流程')
                    self.training.state.pop('sailing_profile',None)
                    self.training.state.pop('sailing_started_rank',None)
                    self.training._save()
            if self.sailing and self._boss_sailing_wanted():
                if self._map(frame):
                    # A manual/pending map is closed before opening AUTO. The
                    # original navigation controller resumes after the cutoff.
                    return self._act(668,1187,'前期 BOSS 航行：关闭已打开的地图',2)
                if not self._quest_idle_verified():
                    return self._route_step(frame)
            handoff = self.training.state.get('boss_sailing_handoff_rank')
            # A cold restart can find the old cutoff already committed OFF at
            # a real port. Re-establish the current task from its NEW map; the
            # visible anchor plus fixed port marker never supplies a boss level.
            cold_port = (0 < self.training.options['boss_auto_sailing_until_rank'] <= self.rank < 40
                         and self.target_rank is None and self.fishing is not None
                         and self.fishing.at_port(frame))
            if handoff == self.rank or cold_port:
                if self._boss_handoff_begin(frame):
                    return True
            elif handoff is not None:
                self.training.state.pop('boss_sailing_handoff_rank',None)
                self.training._save()
            if self._map(frame) and self.training.state['mode'] == 'farming':
                self._set('close_training_map')
                return True
            if self._map(frame):
                self.map_swipes = 0
                self._set('find_boss', '已打开任务地图，继续查找骷髅任务点')
                return True
            if self.training.state.get('sailing_enabled') and not self._sailing_wanted():
                if self._stop_sailing(frame):
                    return True
            training_action = self.training.evaluate(frame,self.rank,
                main_known=bool(self.sailing and self.sailing.running(frame) is True),
                quest_selected=self.target_rank == self.rank)
            if (self.sailing and self.training.state['mode'] == 'farming'
                    and self.training.options['auto_sailing']):
                if self.training.player_level is None:
                    self._timeout(90,'读取自动航行练级的角色等级')
                    return True
                return self.sailing.step(frame,self._sailing_wanted())
            if (self.sailing and (self.training.state.get('sailing_enabled')
                                  or training_action == 'retry')):
                if self._stop_sailing(frame):
                    return True
            if (training_action == 'retry' and self.target_rank == self.rank
                    and self.training.state.get('rank') == self.rank
                    and self.training.ready_to_retry()
                    and self.training.main(frame) and self.training.boss_mode(frame)):
                # A failure can start farming after the level gate is already
                # met. The fresh main/quest marker proves we never left the
                # selected task point; opening its map would hide the fight.
                # Sailing still has to relinquish control above this check.
                self.event('已在当前任务 BOSS 点并达到重试等级，继续本地点挑战，无需再次打开地图')
                return False
            if training_action in ('train', 'retry'):
                self.map_swipes = 0
                self._set('find_training' if training_action == 'train' else 'find_boss')
                return self._act(350, 30, '平行世界：打开练级地点地图' if training_action == 'train'
                                 else '平行世界：练级后打开任务 BOSS 地图', 2)
            if training_action in ('farming', 'trial') or self.training.state['mode'] == 'farming':
                # Navigation/retry decisions take priority; only an idle, non-boss
                # dial is adjusted. Active casts continue alongside enabled upgrades.
                return bool(self.fishing and self.fishing.step(frame))
            if self._quest_idle_verified():
                if self.rank in self.quest_cast_ranks:
                    raise RuntimeError('当前任务单次恢复后再次持续休眠，已停止，禁止盲目重复抛竿')
                self.quest_recovery_rank = self.rank
                self.quest_recovery_key = self.quest_idle_key
                self.quest_recovery_since = now
                self.quest_open_attempts = self.quest_map_frames = 0
                self.quest_map_selected = False
                self.quest_map_target = None
                self._set('quest_map','当前任务连续休眠且无填充，先重新确认地图任务，再尝试单次恢复')
                return self._quest_step(frame)
            if self.target_rank is not None:
                if self.rank > self.target_rank:
                    self.target_rank = None
                    self.event('升级完成，寻找下一地图任务')
                else:
                    # The game itself auto-fishes and may require lengthy level gains.
                    # Other enabled collection tasks can run while waiting here.
                    return False
            if self._fixed(frame, 'next-arrow', (307, 64, 398, 127), .88):
                if self._stop_sailing(frame):
                    return True
                self.map_swipes = 0
                return self._open_next_map(frame,'平行世界：打开下一任务地图')
            return False

        if self.phase == 'find_training':
            if not self._map(frame):
                self._timeout(30, '打开练级地图')
                return True
            if (self.rank is None or self.rank >= 40
                    or self.training.state.get('rank') != self.rank):
                self._set('play')
                return True
            boss = self._match(frame, 'map-boss', (20, 175, 700, 1115), .88)
            target = self.training.training_target(frame, boss)
            if target:
                self._set('close_training_map')
                return self._act(*target, '平行世界：选择非任务 BOSS 地点钓鱼练级', 2)
            if self.map_swipes >= 4:
                # A failed map search is never permission to retry the boss
                # below the saved level target. Keep the farming checkpoint.
                raise RuntimeError('练级地图未找到安全的非任务地点，已保留练级进度，停止以避免未达目标等级就返回试打')
            return self._slide('left' if self.map_swipes < 2 else 'right')

        if self.phase == 'close_training_map':
            if not self._map(frame):
                self._set('play')
                self.last_progress_at = now
                return True
            self._set('play')
            self.last_progress_at = now
            return self._act(668, 1187, '平行世界：关闭地图，钓鱼练级并继续武器强化', 8)

        if self.phase == 'find_boss':
            if self.boss_handoff_since is not None and not self._boss_handoff_ready(frame,map_ready=True):
                self._timeout(30,'BOSS 航行交接确认任务地图')
                return True
            if not self._map(frame):
                self._timeout(30, '打开下一任务地图')
                if self.portal_action is not None:
                    return self._open_next_map(frame,'平行世界：重新确认下一任务地图入口')
                return True
            target = (self._map_task_target(frame) if self.boss_handoff_since is not None
                      else self._match(frame, 'map-boss', (20,175,700,1115), .88))
            if target:
                if self.rank is None or self.rank >= 40:
                    raise RuntimeError('地图任务点击前 RANK 校验失败')
                label = f'平行世界：选择 RANK {self.rank + 1} 任务地点'
                if self.boss_handoff_since is not None:
                    if not self._portal_act('handoff-select',*target[:2],label,frame=frame):
                        return True
                    self.target_rank = self.rank
                    self._set('close_boss_map')
                    return True
                self.target_rank = self.rank
                self._set('close_boss_map')
                return self._act(target[0],target[1],label,2)
            if self.map_swipes >= 24:
                if self.map_refreshes >= 1:
                    raise RuntimeError('重新打开任务地图后仍未找到下一个 BOSS 标记，已停止以避免误选地点')
                self.map_refreshes += 1
                self.target_rank = None
                self._set('refresh_boss_map','地图未见任务标记，先关闭并重新打开下一任务地图')
                return True
            return self._slide('left' if self.map_swipes < 12 else 'right')

        if self.phase == 'refresh_boss_map':
            if not parallel:
                self._timeout(30,'刷新任务地图')
                return True
            self._set('await_boss_refresh')
            if self._map(frame):
                return self._act(668,1187,'平行世界：关闭旧任务地图',2)
            return True

        if self.phase == 'await_boss_refresh':
            # Use the same visible world/arrow guards as the normal next-task
            # path. Fast catch effects can cover the lower fishing controls.
            if (parallel and not self._map(frame)
                    and self._fixed(frame,'next-arrow',(307,64,398,127),.88)
                    and float((frame[75:117,318:389].max(axis=2) >= 210).mean()) > .5):
                self.map_swipes = 0
                return self._open_next_map(frame,'平行世界：重新打开下一任务地图')
            self._timeout(30,'关闭旧地图并确认下一任务入口')
            return True

        if self.phase == 'close_boss_map':
            if not self._map(frame):
                if self.boss_handoff_since is not None:
                    if not self._boss_handoff_ready(frame,allow_battle=True):
                        return True
                    self.training.state.pop('boss_sailing_handoff_rank',None)
                    self.training._save()
                    self.boss_handoff_since = None
                    self.boss_handoff_signature = None
                self._set('play')
                return True
            if self.boss_handoff_since is not None:
                self._portal_act('handoff-close',668,1187,
                    'BOSS 航行交接：关闭已选择当前任务的地图',frame=frame)
                return True
            self._set('play')
            return self._act(668, 1187, '平行世界：关闭地图等待自动战斗', 2)

        if self.phase == 'return_start':
            if not self.completed_rank_40 or self.rank != 40 or not parallel:
                raise RuntimeError('关闭次元门前 RANK 40 守卫失效')
            if self._map(frame):
                return self._act(668, 1187, '平行世界：关闭地图准备返回', 2)
            if self.portal_action is not None:
                if self._return_ready(frame,True):
                    self._set('return_confirm')
                    return True
                if not self._return_ready(frame):
                    return True
            if self._portal_act('return-request',673,936,'平行世界：请求返回现实世界',frame=frame):
                self._set('return_confirm')
            return True

        if self.phase == 'return_confirm':
            if self._fixed(frame, 'return-question', (223, 556, 506, 629), .90):
                if self.portal_action is not None and not self._return_ready(frame,True):
                    return True
                if self._portal_act('return-confirm',208,788,'平行世界：确认保存位置并返回',frame=frame):
                    self._set('await_original')
                return True
            if not parallel:
                self._set('await_original')
                return True
            if now - self.phase_since > 8:
                self._set('return_start', '战斗中无法返回，等待后重试')
                self.next_action_at = now + 4
            return True

        if self.phase == 'await_original':
            if not parallel and self._observatory(frame) and not self._map(frame):
                self.return_since = None
                self._set('cat_open')
                return True
            if parallel and now - self.phase_since > 9:
                self._set('return_start', '返回未生效，等待非战斗状态重试')
                self.next_action_at = now + 4
            self._timeout(180, '返回现实世界')
            return True

        if self.phase == 'cat_open':
            if not self.completed_rank_40:
                raise RuntimeError('次元门重开前缺少 RANK 40 通关记录')
            if self.portal_action is not None:
                self._timeout(30,'等待停航和传送门 NPC')
                self.cat_frames = self.cat_frames+1 if self._npc_ready(frame) else 0
                if self.cat_frames < 3:
                    return True
            if self._portal_act('npc',482,548,'平行世界：查看传送门管理菜单',delay=3):
                self.cat_frames = 0
                self.cat_attempts = 1
                self._set('cat_menu')
            return True

        if self.phase == 'cat_menu':
            if self._fixed(frame, 'gate-close', (154, 685, 565, 785), .88):
                self._set('close_click')
                return True
            if self._fixed(frame, 'gate-activate', (154, 751, 565, 852), .88):
                self._set('activate_click')
                return True
            self._timeout(30, '传送门管理菜单')
            if self.portal_action is not None:
                self.cat_frames = self.cat_frames+1 if self._npc_ready(frame) else 0
                if self.cat_frames >= 3 and self.cat_attempts < 3:
                    if self._portal_act('npc',482,548,'平行世界：新截图确认菜单未打开后重试',delay=3):
                        self.cat_frames = 0
                        self.cat_attempts += 1
            return True

        if self.phase == 'close_click':
            if not self.completed_rank_40 or self.rank != 40:
                raise RuntimeError('拒绝关闭传送门：本轮未确认 RANK 40')
            if not self._fixed(frame, 'gate-close', (154, 685, 565, 785), .88):
                self._set('cat_menu')
                return True
            if self._portal_act('close-gate',354,739,'平行世界：RANK 40 后关闭传送门'):
                self._set('close_confirm')
            return True

        if self.phase == 'close_confirm':
            if self._fixed(frame, 'gate-activate', (154, 751, 565, 852), .88):
                self._set('activate_click')
                return True
            if self._fixed(frame, 'continue', (63, 741, 356, 834), .92):
                if not self.completed_rank_40 or self.rank != 40:
                    raise RuntimeError('拒绝确认关闭传送门：RANK 40 守卫失效')
                self._set('gate_closed_check')
                return self._act(208, 788, '平行世界：确认 RANK 40 通关后关闭', 3)
            self._timeout(30, '关闭传送门确认')
            return True

        if self.phase == 'gate_closed_check':
            if self._fixed(frame, 'gate-activate', (154, 751, 565, 852), .88):
                self._set('activate_click')
                return True
            if self._observatory(frame) and not self._gate_active(frame):
                self._set('cat_open')
                return True
            self._timeout(45, '传送门关闭结果')
            return True

        if self.phase == 'activate_click':
            if not self._fixed(frame, 'gate-activate', (154, 751, 565, 852), .88):
                self._set('cat_menu')
                return True
            if self._portal_act('activate',355,803,'平行世界：激活新一轮传送门'):
                self.gate_stage_verified = False
                self.gate_stage_readings.clear()
                self._set('search_confirm')
                self.rounds += 1
            return True

        raise RuntimeError(f'未知平行世界步骤：{self.phase}')
