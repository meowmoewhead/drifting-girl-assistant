"""Reactive dial calibration without taking over the game's native AUTO route."""
import hashlib
import json
from collections import deque
from pathlib import Path
import time

import cv2
import numpy as np

from drift.perception.digit_ocr import read_numeric_views
from drift.tasks.fishing_meter import FishingMeter
from drift.tasks.parallel_screen import ParallelScreen
from drift.tasks.training_sailing import TARGETS, TrainingSailing
from drift.tasks import native_auto_sale
from drift.tasks import meter_journal
from drift.tasks.meter_limits import MeterWorkflowLimit
from drift.core.observability import CandidateObservation


class _ScreenContext(ParallelScreen):
    """标尺校准自己的只读画面上下文。

    只要"是不是平行世界 / RANK / 阶段笔画"这三样 —— 复用
    :class:`ParallelScreen` 的判定,因此与平行世界功能的识别结果一致,但**不引入
    导航、练级、任务状态机**。``training`` 属性是一份兼容垫片,让
    :class:`FishingMeter` / :class:`TrainingSailing` 照常读写 ``sailing_enabled`` /
    ``fishing_meter_limits``,并把改动落到校准自己的记录文件里。
    """

    def __init__(self, root, reader, save):
        super().__init__(root, reader)
        self.rank = None
        # FishingMeter / TrainingSailing 的"主界面"判定用主界面守卫模板
        # (entry / guard-star / guard-diamond / main-auto,位于 assets/equipment)。
        # 平行世界功能从 BossTraining 拿这些,校准这里直接把同名模板并入自己的
        # 匹配表,规则与 BossTraining.main 完全一致。
        guards = ('entry', 'guard-star', 'guard-diamond', 'main-auto')
        base = Path(root) / 'assets' / 'equipment'
        loaded = {key: cv2.cvtColor(cv2.imread(str(base / f'{key}.png')), cv2.COLOR_BGR2RGB)
                  for key in guards}
        if any(value is None for value in loaded.values()):
            raise RuntimeError('标尺校准缺少主界面守卫模板')
        self.assets.update(loaded)
        boss = cv2.imread(str(Path(root) / 'assets/navigation/parallel-boss-battle.png'))
        if boss is None:
            raise RuntimeError('标尺校准缺少任务 BOSS 守卫模板')
        self.assets['boss-mode'] = cv2.cvtColor(boss, cv2.COLOR_BGR2RGB)[38:75, 254:289]
        success = cv2.imread(str(Path(root) / 'assets/training/success-check.png'))
        if success is None:
            raise RuntimeError('标尺校准缺少钓鱼结果模板')
        self.assets['success-check'] = cv2.cvtColor(success, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(self.assets['success-check'], cv2.COLOR_RGB2HSV)
        strokes = ((hsv[:,:,0] >= 25) & (hsv[:,:,0] <= 90)
                   & (hsv[:,:,1] >= 80) & (hsv[:,:,2] >= 90)).astype('uint8')
        self.success_mask = cv2.dilate(strokes, np.ones((3,3), np.uint8))
        # 垫片要能读数字(角色等级 / 计量上下限)、要能做模板匹配(main / _match),
        # 所以把 reader 和本上下文都交给它
        self.training = _TrainingStub(save, reader, self)


class _TrainingStub:
    """FishingMeter / TrainingSailing 需要的最小训练上下文。

    两个模块用到的成员(实测统计):``state`` / ``boss_mode`` / ``set_sailing`` /
    ``_number`` / ``_match`` / ``options`` / ``_save`` / ``ready_to_retry`` /
    ``levels`` / ``reader`` / ``main``。这里逐项提供,并保持与
    :class:`~drift.tasks.boss_training.BossTraining` 相同的读数投票与主界面判定规则,
    但**不带任务 BOSS、练级选项、失败计数等状态机**。
    """

    def __init__(self, save, reader=None, screen=None):
        self.reader = reader
        self.screen = screen
        self._save = save
        self.state = {}
        # auto_sailing 必须存在:运行期的 fishing_action 包装器会读它判断"航行是否已停"
        # (原来那份 ParallelWorld 就是带这两个键构造的)。meter_auto_adjust 打开才允许
        # 用取消旧填充的方式校正。
        self.options = {'auto_sailing': True, 'meter_auto_adjust': True}
        self.readings = deque(maxlen=3)

    # --- 读数 ---
    def _number(self, frame, boxes):
        views, regions = [], []
        for index, (x1, y1, x2, y2) in enumerate(boxes):
            crop = frame[y1:y2, x1:x2]
            ink = np.all(crop >= 200, axis=2)
            binary = np.repeat(np.where(ink, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
            for view in (crop, binary):
                views.append(view)
                regions.append(index)
        votes = {}
        for index, reading in zip(regions, read_numeric_views(self.reader, views)):
            if reading.value is not None and 1 <= reading.value <= 10000 and reading.confidence >= .93:
                votes.setdefault(reading.value, set()).add(index)
        # Even a strong conflicting view invalidates this frame.
        if len(votes) == 1:
            value, regions = next(iter(votes.items()))
            if len(regions) == 2:
                return value
        return None

    def levels(self, frame, *, main_known=False):
        """只读一次角色等级;不计算 BOSS 等级差(校准用不到)。"""
        if not main_known and not self.main(frame):
            self.readings.clear()
            return None
        player = self._number(frame, ((76, 1248, 137, 1278), (79, 1249, 136, 1277)))
        self.readings.append((player, None))
        return player

    # --- 画面判定 ---
    def _match(self, frame, name, area, threshold=.88):
        return self.screen.match(frame, name, area, threshold)

    def main(self, frame):
        boxes = {'entry': (327, 1140, 378, 1192), 'guard-star': (208, 1137, 269, 1192),
                 'guard-diamond': (539, 1137, 600, 1186), 'main-auto': (276, 881, 310, 913)}
        return (all(self._match(frame, name, box, .8 if name == 'entry' else .91)
                    for name, box in boxes.items())
                and float((np.max(frame[1147:1168, 558:585], axis=2) >= 210).mean()) >= .30)

    def boss_mode(self, frame):
        """独立读取任务骷髅，不引入导航或练级状态机。"""
        return (frame.shape[:2] == (1280, 720)
                and self._match(frame, 'boss-mode', (248, 32, 295, 81), .92) is not None)

    # --- 状态 ---
    def set_sailing(self, enabled, verified=False, *, profile=None, rank=None):
        if self.state.get('sailing_enabled') != enabled:
            self.state['sailing_enabled'] = bool(enabled)
        self.state['sailing_verified'] = bool(enabled and verified)
        if self._save:
            self._save()

    def ready_to_retry(self):
        return False


class MeterCalibration:
    START_PROOFS = ('paired_hold', 'observed_max_fill', 'observed_target_fill', 'post_release_fishing')
    # 触发阈值:角色等级比标尺高出多少级才校准。
    # 2026-10-11 由 100 改为 50;2026-10-23 由 50 改为 80 —— 实测游戏会自己补标尺
    # (0 次手动拖动时数值仍 983→1612 反复变化),差 50 上下的小差距不值得抢窗口。
    # 正好等于 80 不触发(判据是严格大于)。
    GAP = 80
    LIMIT = 90
    CANDIDATE_LIMIT = 300
    RESTORE_RESERVE = 30
    CONFIRM_SECONDS = 3
    # Candidate votes only establish whether to open a fresh execution window.
    # OCR-heavy main loops can be 10-14s apart. Keep their context/number votes
    # separate from the stricter six-second stop/start observation guards.
    CANDIDATE_FRAME_GAP = 15
    # A released failed window is eligible for a new, independently guarded
    # check after a short backoff. It did not earn a successful cooldown.
    FAILURE_BACKOFF = 300
    # Reserve three OFF observations: the existing consensus permits up to
    # six seconds between frames. Do not start a stop at the window's edge.
    STOP_CONFIRM_BUDGET = 15

    def __init__(self, root, reader, meter_action, sailing_action, event, interval=60):
        self.path = Path(root)/'meter_calibration_state.json'
        self.event, self.sailing_action = event, sailing_action
        self.interval = max(0.2, float(interval))
        self.journal = dict(version=1, pending=False, cycles=0, next_check=0)
        if self.path.is_file():
            saved = json.loads(self.path.read_text(encoding='utf-8-sig'))
            if (saved.get('version') != 1 or type(saved.get('pending')) is not bool
                    or saved.get('pending') and
                    (type(saved.get('rank')) is not int or not 0 <= saved['rank'] < 40
                     or type(saved.get('stage')) is not int
                     or type(saved.get('restore_running')) is not bool
                     or not isinstance(saved.get('deadline'), (int, float))
                     or not np.isfinite(saved['deadline']))):
                raise RuntimeError('独立标尺校准记录无效，已停止')
            self.journal = saved
        # Explicit starts may retire execution ownership, never the restoration
        # debt or spent input budget. The constructor itself remains read-only.
        self.resumed_pending = self.active
        # 复用识别器与既有的受守护圆盘手势;**不再构造平行世界导航/练级状态机**
        # (标尺校准只读画面,不需要导航)。屏幕判定来自共享的 ParallelScreen。
        self.world = _ScreenContext(root, reader, self._save)
        training = self.journal.get('training', {})
        if (not isinstance(training, dict)
                or self.active and 'rank' in training and training['rank'] != self.journal['rank']):
            raise RuntimeError('标尺校准上下文记录无效，已停止并保留恢复记录')
        # FishingMeter loads its range cache during construction, so restore
        # the saved context and recovery flags before constructing it.
        self.world.training.state = (training if self.active or self.awaiting_recovery else
            dict(mode='idle', fishing_meter_limits=training.get('fishing_meter_limits', {})))
        self.world.rank = self.journal.get('rank') if self.active else None
        self.fishing = FishingMeter(root, self.world.training, meter_action, event)
        self.sailing = TrainingSailing(root, self.world.training, self._no_navigation,
                                       event, self.fishing,
                                       context_check=self.fishing._context)
        # 运行期的 fishing_action 包装器把 owner(即本 world)当平行世界对象用:
        # 要取 .fishing / .sailing / training.options / _parallel。挂上去即可照常工作
        # (2026-10-10 真机触发时正是缺这层接口而报 no attribute 'fishing')。
        self.world.fishing, self.world.sailing = self.fishing, self.sailing
        self.sailing_action = sailing_action
        # This instance is activated only by the independent player-gap trigger.
        # Once saved, any positive correction can use the old-fill cancel path.
        self.fishing.RECALIBRATE_GAP = 1
        counts = self.journal.get('attempts', {})
        self.fishing.cancels = counts.get('cancel', 0)
        self.fishing.drags = counts.get('drag', 0) + counts.get('reveal', 0)
        self.fishing.starts = counts.get('start', 0)
        self.fishing.current = self.journal.get('value')
        self.fishing.goal = self.journal.get('target', self.journal.get('trigger_player', self.journal.get('high')))
        trigger_player = self.journal.get('trigger_player')
        self.fishing.saved_player_gap = bool(self.active
            and type(trigger_player) is int and type(self.fishing.current) is int
            and trigger_player-self.fishing.current > self.GAP)
        self.phase = ('observe_retired' if self.awaiting_recovery
                      else 'restore_fishing' if self.active and self.journal.get('fishing_restore')
                      else 'route_context' if self.active and self.preserves_route
                      else 'running' if self.active and self.journal.get('resume_requested')
                      else 'open' if self.active and isinstance(self.journal.get('restore_retry'),dict)
                           and self.journal['restore_retry'].get('from_stopped_main') is True
                           and not self.journal.get('sailing_attempts',{}).get('open',0)
                      else 'menu' if self.active and isinstance(self.journal.get('restore_retry'),dict)
                           and self.journal.get('proof') in MeterCalibration.START_PROOFS
                      else 'stop' if self.active else 'idle')
        self.cycles = self.journal.get('cycles', 0)
        self.enabled = lambda: True
        self.key = self.seen_at = None
        self.seen = 0
        self.next_at = 0
        self.last_seen = 0
        self.last_report = 0
        self.reported_reasons = {}
        # 最近一次"暂不触发"的原因,给面板显示(status().hold)
        self.hold = None
        # 只在"已取消但等不到恢复"时记:具体是哪一步没满足,便于用户和审阅定位
        self.observe_unmet = None
        # 触发前的门控:当前这一帧具体是哪条没通过(画面条件或读数)
        self.wait_unmet = None
        self.screen_reason = None
        self.off_verified = False
        self.route_verified = False
        self.fishing.start_route_expected = self.expected_route()
        self.context = self.journal.get('stopped_context')
        self.menu_key = None
        self.player = self.value = None
        self.expired_observe_until = time.monotonic()+15 if self.active else None
        self.untouched_expiry_observed = None
        self.retry_path = self.path.with_name('meter_calibration_retry.json')
        self.retry = (json.loads(self.retry_path.read_text(encoding='utf-8-sig'))
                      if self.retry_path.is_file() else None)
        self.retired_frame = None
        self.candidate_at = None
        self._candidate_trace = None
        self.restore_frame = None
        self.restore_cast_context = None

    @property
    def preserves_route(self):
        # Old records retain their original stop/restore obligations. A new
        # calibration records this policy explicitly; restart does not migrate it.
        return self.journal.get('route_policy') == 'preserve'

    def expected_route(self):
        return self.journal.get('restore_running') is True if self.preserves_route else False

    @property
    def awaiting_recovery(self):
        retired = self.journal.get('retirement')
        return bool(isinstance(retired, dict) and not retired.get('released'))

    def retire_expired_on_start(self):
        if not self.active or not meter_journal.expiry(self.journal, time.time()):
            return False
        # An existing, explicitly authorized and still valid repair request
        # keeps its own bounds; startup is not a new repair authorization.
        if isinstance(self.retry, dict) and self.retry.get('expires_at', 0) > time.time():
            return False
        return self._retire_window('启动时发现旧标尺执行已到期', source='startup_expired')

    def _retire_window(self, reason, *, source='workflow_limit'):
        raw = self.path.read_bytes()
        result = meter_journal.retire(self.path, hashlib.sha256(raw).hexdigest(),
            source=source, reason=reason, interval=self.interval)
        if not result['retired']:
            return False
        self.journal = json.loads(self.path.read_text(encoding='utf-8-sig'))
        self.world.training.state = self.journal.get('training', {})
        self.world.rank = None
        self.retry = None
        self._set('observe_retired')
        self.hold = '本轮已归档；只读等待实际钓鱼及原航巡恢复，不再追加输入'
        self.event(f'标尺校准：{reason}；已归档并跳过本轮，保留截止时间、已用次数和恢复记录，只读核对恢复状态')
        return True

    def _observe_retired(self, frame):
        # The runtime can ask both before combat and in the normal task pass.
        # The same captured array must never count as two confirmation frames.
        if frame is self.retired_frame:
            return False
        self.retired_frame = frame
        screen = self.screen(frame, observe_retired=True)
        evidence = ('fill' if screen and self.fishing.blue_fill(frame, main_known=True)
                    else 'battle' if screen and self.fishing.real_battle(frame) else None)
        value = player = None
        if screen and evidence:
            box = self.fishing.box(frame)
            value = self.fishing.selected(frame, box) if box else None
            player = self.world.training._number(frame, ((76,1248,137,1278),(79,1249,136,1277)))
        if value is None or player is None:
            self.key = self.seen_at = None
            self.seen = 0
            # 说清楚卡在哪一步:审阅问过"为什么已取消后一直不能恢复",只写
            # "等待原航巡"无法区分是画面不认、还是没有真实钓鱼、还是数字读不清。
            if screen is None:
                unmet = '画面不是已确认的平行世界主画面'
            elif evidence is None:
                unmet = '没有圆内浅蓝填充也没有实际战斗（例如鱼钩灰着）'
            elif value is None:
                unmet = '标尺数字读不清'
            else:
                unmet = '角色等级读不清'
            self.observe_unmet = unmet
            self._report_wait(f'旧标尺执行已归档，只读等待原航巡及真实钓鱼/战斗；当前未满足：{unmet}')
            return False
        self.observe_unmet = None
        self.player, self.value = player, value
        key = ('retired-restored', screen['stage'], screen['rank'], screen['route'],
               self.fishing._context(frame), value)
        if not self._confirmed(key):
            return False
        self.journal['retirement'].update(released=True, verified_at=time.time(),
            evidence=dict(stage=screen['stage'], rank=screen['rank'], route=screen['route'],
                          player=player, value=value, fishing=evidence),
            calibration_completed=False)
        self.journal['next_check'] = time.time()+self.failure_backoff()
        self._save()
        self._set('idle')
        self.hold = '已连续确认原航巡及实际钓鱼恢复；本轮不计校准成功，间隔后重新检测'
        self.event('标尺校准：连续三张新图、至少三秒确认原航巡与实际钓鱼恢复，已解除只读等待；旧失败和次数保留，未计校准成功')
        return False

    @staticmethod
    def _pre_input_failure(journal):
        player, value = journal.get('trigger_player'), journal.get('value')
        return (type(player) is int and type(value) is int
                and player-value > MeterCalibration.GAP
                and not any(journal.get('attempts',{}).values())
                and journal.get('sailing_attempts',{}).get('stop',0) >= 1
                and isinstance(journal.get('selected_signature'),dict)
                and journal.get('proof') is None)

    @staticmethod
    def _restore_only_failure(journal):
        return (journal.get('restore_running') is True
                and journal.get('proof') in MeterCalibration.START_PROOFS
                and (journal.get('sailing_attempts',{}).get('open') == 1
                     or not journal.get('sailing_attempts',{}).get('open',0)
                     and journal.get('proof') == 'post_release_fishing'
                     and journal.get('attempts',{}).get('start') == 1)
                and not journal.get('sailing_attempts',{}).get('start',0)
                and journal.get('resume_requested') is False
                and journal.get('restore_retry') is None)

    @staticmethod
    def request_retry(root, failed_sha256):
        """Register one explicit repair attempt; do not renew the failed file."""
        root = Path(root)
        config = json.loads((root/'config.json').read_text(encoding='utf-8-sig'))
        if not any(r['id'] == 'driftinggirl_meter_calibration' and r['enabled']
                   for r in config['rules']):
            raise ValueError('请先启用标尺校准')
        raw = (root/'meter_calibration_state.json').read_bytes()
        journal = json.loads(raw.decode('utf-8-sig'))
        digest = hashlib.sha256(raw).hexdigest()
        attempts = journal.get('attempts', {})
        if (failed_sha256 != digest or journal.get('version') != 1
                or journal.get('pending') is not True
                or type(journal.get('deadline')) not in (int,float)
                or not np.isfinite(journal['deadline']) or time.time() < journal['deadline']
                or not MeterCalibration._restore_only_failure(journal)
                   and (attempts.get('drag',0)+attempts.get('reveal',0) < 4
                        and not MeterCalibration._pre_input_failure(journal)
                        or attempts.get('start',0) or journal.get('resume_requested'))
                or not journal.get('stopped_context')):
            raise ValueError('只允许对已到期的未输入/拖动耗尽记录，或已确认填充且尚未恢复航巡的记录，申请一次修复')
        path = root/'meter_calibration_retry.json'
        if path.exists():
            raise ValueError('已有标尺修复重试请求，请先核查')
        now = time.time()
        request = dict(version=1,failed_sha256=digest,requested_at=now,expires_at=now+300)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(request,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        temporary.replace(path)
        return request

    def _apply_retry(self, frame):
        """Archive after fresh OFF and matching context proof; never send input."""
        request = self.retry
        if (not isinstance(request,dict)
                or set(request) != {'version','failed_sha256','requested_at','expires_at'}
                or request.get('version') != 1
                or type(request.get('requested_at')) not in (int,float)
                or not np.isfinite(request['requested_at'])
                or type(request.get('expires_at')) not in (int,float)
                or not np.isfinite(request['expires_at']) or time.time() >= request['expires_at']):
            raise RuntimeError('标尺修复重试请求无效或已到期，保留原恢复记录')
        raw = self.path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != request.get('failed_sha256'):
            raise RuntimeError('标尺失败记录已变化，拒绝重试')
        if self._restore_only_failure(self.journal):
            return self._apply_restore_retry(frame,raw,digest,request)
        screen = self.screen(frame)
        pre_input = self._pre_input_failure(self.journal)
        cast = (self.fishing.cast_data(frame)
                if pre_input and screen and not screen['menu'] and screen['route'] is False else None)
        data = (dict(context=cast['context'],value=self.journal['value'],
                     player=self.journal['trigger_player'],low=None,high=None)
                if cast and self._selection_signature(cast) == self.journal['selected_signature']
                else self.fishing.measure(frame,remember=False)
                if screen and not screen['menu'] and screen['route'] is False
                and not pre_input and self.fishing.idle_icon(frame)
                and not self.fishing.busy(frame) else None)
        strokes = self.world._stage_strokes(frame) if data else None
        valid = bool(data and strokes is not None
                     and data['context'] == self.journal.get('stopped_context')
                     and strokes.hex() == self.journal.get('stage_strokes'))
        if not valid:
            self.key = self.seen_at = None
            self.seen = 0
            return False
        if not self._confirmed(('repair-retry',screen['rank'],screen['stage'],
                                data['context'],data['value'],data['low'],data['high'])):
            return False
        if (self.path.read_bytes() != raw or time.time() >= request['expires_at']
                or not self.enabled()):
            raise RuntimeError('重试前记录或授权已变化，保留原恢复记录')
        archive = self.path.parent/'logs'/'meter-calibration'/('failed-'+digest+'.json')
        archive.parent.mkdir(parents=True,exist_ok=True)
        if archive.exists() and archive.read_bytes() != raw:
            raise RuntimeError('标尺失败记录归档冲突，已停止')
        if not archive.exists():
            archive.write_bytes(raw)
        now = time.time()
        self.journal.update(value=data['value'],high=data['high'],trigger_player=data['player'],
            deadline=now+self.LIMIT,attempts={},sailing_attempts={},proof=None,resolution=None,
            resume_requested=False,window_restarted_at=now,
            repair_retry=dict(failed_sha256=digest,archive=str(archive),
                              requested_at=request['requested_at'],verified_at=now))
        self.context = data['context']
        self.off_verified = True
        self.world.training.state.update(mode='farming',rank=screen['rank'],sailing_enabled=False,
            sailing_sleep_recovery=True,sailing_meter_adjustment=True)
        self.fishing.current,self.fishing.goal = data['value'],data.get('goal')
        self.journal['target'] = data.get('goal')
        self.fishing.drags = self.fishing.starts = self.fishing.cancels = 0
        self.fishing.start_fill_verified = False
        self.fishing.saved_player_gap = pre_input
        self.fishing.idle_since = None
        self.fishing.stable,self.fishing.key = 0,None
        self.fishing.phase,self.fishing.next_action_at = 'detect',0
        self.journal['selected_signature'] = self._selection_signature(self.fishing.cast_data(frame))
        self.resumed_pending = False
        self._save()
        self.retry_path.unlink(missing_ok=True)
        self.retry = None
        self._set('prepare')
        proof = '停航和原数字/地点笔画' if pre_input else '停航和空圆盘'
        self.event(f'标尺修复：连续三帧三秒确认原世界/RANK、{proof}；原失败次数及截止时间已原样归档，开始一次独立90秒重试')
        return True

    def _restore_repair_context(self, frame):
        screen = self.screen(frame)
        if screen is None:
            return None
        if screen['menu']:
            settings = self.settings_key(frame)
            if settings is not None:
                return ('menu', ('restore-menu',screen['rank'],screen['stage'],settings), settings)
        elif (screen['route'] is False
                and not self.journal.get('sailing_attempts',{}).get('open',0)
                and self.journal.get('proof') == 'post_release_fishing'
                and self.journal.get('attempts',{}).get('start') == 1
                and self.frame_blocker(frame) is None
                and (self.fishing.blue_fill(frame) or self.fishing.real_battle(frame))):
            return ('open', ('restore-main',screen['rank'],screen['stage'],self.context), None)
        return None

    def _apply_restore_retry(self,frame,raw,digest,request):
        # A completed hold must never be repeated to repair a menu-only fault.
        # Retain the original meter deadline and every meter attempt verbatim.
        context = self._restore_repair_context(frame)
        if context is None:
            self.key=self.seen_at=None
            self.seen=0
            return False
        phase, key, settings = context
        if not self._confirmed(key):
            return False
        if (self.path.read_bytes()!=raw or time.time()>=request['expires_at'] or not self.enabled()):
            raise RuntimeError('航巡恢复前记录或授权已变化，保留原记录')
        archive=self.path.parent/'logs'/'meter-calibration'/('failed-'+digest+'.json')
        archive.parent.mkdir(parents=True,exist_ok=True)
        if archive.exists() and archive.read_bytes()!=raw:
            raise RuntimeError('航巡恢复记录归档冲突，已停止')
        if not archive.exists():
            archive.write_bytes(raw)
        now=time.time()
        self.journal['restore_retry']=dict(failed_sha256=digest,archive=str(archive),
            requested_at=request['requested_at'],verified_at=now,deadline=now+40,
            from_stopped_main=phase=='open')
        self.off_verified=True
        self.menu_key=settings
        self._save()
        self.retry_path.unlink(missing_ok=True)
        self.retry=None
        self._set(phase)
        self.event('标尺修复：已确认原世界/RANK、停止AUTO及当前航巡选项；只接续一次40秒航巡恢复，原标尺截止时间和全部输入次数保留，不再取消/拖动/长按')
        return True

    def _restoring_only(self):
        retry=self.journal.get('restore_retry')
        return (isinstance(retry,dict) and self.active
                and self.journal.get('restore_running') is True
                and self.journal.get('proof') in MeterCalibration.START_PROOFS
                and (self.phase in ('menu','running')
                     or self.phase == 'open' and retry.get('from_stopped_main') is True)
                and type(retry.get('deadline')) in (int,float)
                and np.isfinite(retry['deadline']))

    @staticmethod
    def _no_navigation(*args, **kwargs):
        raise AssertionError('Independent meter task must never navigate or configure targets')

    @property
    def active(self):
        return bool(self.journal['pending'])

    def status(self):
        return dict(phase=self.phase, pending=self.active, player_level=self.player,
                    value=self.value, gap=None if self.player is None or self.value is None
                    else self.player-self.value, trigger_gap=self.GAP,
                    rank=self.journal.get('rank'), cycles=self.cycles,
                    deadline=self.journal.get('deadline'), fishing=self.fishing.status(),
                    proof=self.journal.get('proof'), resolution=self.journal.get('resolution'),
                    # 面板据此说明"为什么现在没在校准",不必翻日志
                    hold=self.hold, retirement=self.journal.get('retirement'),
                    observe_unmet=self.observe_unmet, wait_unmet=self.wait_unmet,
                    fishing_restore=self.journal.get('fishing_restore'),
                    candidate_started_at=self.candidate_at,
                    candidate_observation=(self._candidate_trace.snapshot()
                                           if getattr(self, '_candidate_trace', None) else None),
                    recovery_required=self.awaiting_recovery, preserves_native_settings=True,
                    route_policy=self.journal.get('route_policy', 'legacy_stop'),
                    next_check=self._next_check(), cooldown_seconds=self.interval,
                    wait_kind='failure_backoff' if self._failed_release_at() is not None else 'success_cooldown',
                    failure_backoff_seconds=self.failure_backoff())

    def failure_backoff(self):
        return min(self.interval, self.FAILURE_BACKOFF)

    def _failed_release_at(self):
        if self.active or self.awaiting_recovery:
            return None
        finished = self.journal.get('finished_at', 0)
        finished = finished if type(finished) in (int, float) and np.isfinite(finished) else 0
        candidates = []
        for key in ('retirement', 'resolution'):
            record = self.journal.get(key)
            if (not isinstance(record, dict) or record.get('calibration_completed') is not False
                    or key == 'retirement' and record.get('released') is not True):
                continue
            at = record.get('verified_at')
            if type(at) in (int, float) and np.isfinite(at) and at > finished:
                candidates.append(at)
        return max(candidates) if candidates else None

    def _next_check(self):
        # A changed UI interval also applies to the last successful calibration.
        # Reading this never rewrites the old deadline or a failure record.
        finished = self.journal.get('finished_at')
        failed_release = self._failed_release_at()
        if failed_release is not None:
            # Older builds wrote the successful interval into next_check even
            # for a failed calibration. Interpret the recorded release proof;
            # leave its archived bytes, deadline and spent inputs untouched.
            return max(failed_release+self.failure_backoff(),
                       finished+self.interval if type(finished) in (int, float) else 0)
        if self.awaiting_recovery:
            # 未解除的旧失败仍按记录里的绝对时间等，不能被改小的 UI 间隔提前放行。
            return self.journal.get('next_check', 0)
        if type(finished) in (int, float):
            # 用"当前"间隔重算成功冷却：把 UI 间隔改小时应立即生效，
            # 不能让上一次写进 next_check 的旧绝对值（例如 5400 秒）压住新间隔。
            return finished + self.interval
        return self.journal.get('next_check', 0)

    def _save(self):
        self.journal['training'] = self.world.training.state
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.journal, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        temporary.replace(self.path)

    def _set(self, phase):
        # Only a new in-process calibration may begin its adjustment budget
        # after the first OFF proof. Phase changes and process restarts cannot
        # renew a window that has already started or had a meter input.
        if (phase == 'prepare' and self.phase != 'prepare' and self.active
                and not self.preserves_route and not self.resumed_pending and not self.journal.get('window_restarted_at')
                and not any(self.journal.get('attempts', {}).values())):
            self.journal['deadline'] = time.time() + self.LIMIT
            self.journal['window_restarted_at'] = time.time()
            self._save()
        self.phase = phase
        self.key = self.seen_at = None
        self.seen = 0

    def _confirmed(self, key, *, seconds=CONFIRM_SECONDS, max_gap=6):
        now = time.monotonic()
        if key != self.key or now-self.last_seen > max_gap:
            self.key, self.seen_at, self.seen = key, now, 1
        else:
            self.seen += 1
        self.last_seen = now
        return self.seen >= 3 and now-self.seen_at >= seconds

    def _stage(self, frame):
        views = []
        thresholds = (80,90) if self.sailing.menu(frame) else (110,130)
        for x,y,X,Y in ((40,45,68,71), (39,44,69,72)):
            crop = frame[y:Y,x:X].astype(np.int16)
            for threshold in thresholds:
                ink = ((crop[:,:,1]-crop[:,:,0] >= 10)
                       & (crop[:,:,2]-crop[:,:,0] >= 15) & (crop.max(axis=2) >= threshold))
                ys,xs = np.where(ink)
                if len(xs) < 8:
                    return None
                binary = np.where(ink,0,255).astype('uint8')[ys.min():ys.max()+1,xs.min():xs.max()+1]
                binary = np.pad(binary,3,constant_values=255)
                views.append(np.repeat(binary[:,:,None],3,axis=2))
        readings = read_numeric_views(self.world.reader, views)
        if (any(r.value is None or not 1 <= r.value <= 99 or r.confidence < .95 for r in readings)
                or len({r.value for r in readings}) != 1):
            return None
        return readings[0].value

    def _expects_resumed_route(self):
        return (self.active and self.phase == 'running'
                and self.journal.get('restore_running') is True
                and self.journal.get('resume_requested') is True
                and self.journal.get('proof') in MeterCalibration.START_PROOFS
                and self.journal.get('sailing_attempts', {}).get('start') == 1)

    def _no_screen(self, reason):
        """记录"为什么这一帧不算可校准画面",供 _report_wait 写出具体条件。"""
        self.screen_reason = reason
        return None

    def screen(self, frame, *, resumed_route=False, observe_untouched=False, observe_retired=False,
               restore_fishing=False):
        # A restored native route may immediately visit its checked BOSS target.
        # This exception is read-only, after verified fill and our one Start.
        # Every input guard still calls the strict default screen probe.
        self.screen_reason = None
        if resumed_route and not self._expects_resumed_route():
            return self._no_screen('当前不是"刚恢复的航巡"上下文')
        if observe_untouched and not self._untouched_route_wait():
            return self._no_screen('当前不是"未动过航巡"的等待上下文')
        if observe_retired and not self.awaiting_recovery:
            return self._no_screen('当前没有待恢复的旧执行')
        if restore_fishing and not (self.active and self.preserves_route
                                    and self.phase == 'restore_fishing'
                                    and self.journal.get('attempts', {}).get('cancel', 0)):
            return self._no_screen('当前不是"恢复钓鱼"阶段或已没有取消次数')
        if frame.shape[:2] != (1280,720):
            return self._no_screen('分辨率不是 720x1280（实际 %sx%s）' % (frame.shape[1], frame.shape[0]))
        if not self.world._parallel(frame):
            return self._no_screen('不在平行世界画面（可能在世界切换或过场里）')
        if (not (resumed_route or observe_untouched or observe_retired or restore_fishing)
                and self.world.training.boss_mode(frame)):
            return self._no_screen('顶部有任务 BOSS 标记，不是可校准画面')
        menu = self.sailing.menu(frame)
        route = self.sailing.running(frame) if not menu else None
        if observe_retired and (menu or route is not self.journal.get('restore_running')
                               or not self.fishing._main_visible(frame)):
            return self._no_screen('旧执行恢复中：航巡或主画面元素还没回到记录状态')
        if restore_fishing and (menu or route is not self.expected_route()
                                or not self.fishing._main_visible(frame)):
            return self._no_screen('恢复钓鱼中：航巡状态或主画面元素不符')
        if (resumed_route or observe_untouched) and (menu or route is not True or not self.fishing._main_visible(frame)):
            return self._no_screen('航巡未在运行或主画面元素不全')
        if not menu and (route is None or self.fishing.at_port(frame)):
            return self._no_screen('航巡状态未知或船在港口')
        if menu and (not self.sailing._fixed(frame,'auto-label',(186,34,253,65),.90)
                or not self.sailing._fixed(frame,'inactive-face',(194,4,246,42),.96)
                or self.sailing._fixed(frame,'active-arrow',(194,4,246,42),.96)):
            return self._no_screen('航行设置菜单的标签或按钮不完整')
        rank, stage = self.world._rank_value(frame), self._stage(frame)
        if rank is None or not 0 <= rank < 40 or stage is None:
            return self._no_screen('RANK 或阶段数字没读出来')
        if self.active and (rank != self.journal['rank'] or stage != self.journal['stage']):
            return self._no_screen('与世界上下文不符（当前 RANK%s/%s，记录 RANK%s/%s）'
                                   % (rank, stage, self.journal['rank'], self.journal['stage']))
        return dict(rank=rank, stage=stage, menu=menu, route=route)

    def _measure_unmet(self, frame):
        """measure() 返回 None 时说清是哪一步没过;只读,不产生输入。"""
        if not self.fishing._main_visible(frame):
            return '主画面元素不完整'
        if not self.fishing.world_matches(frame):
            return '世界不匹配'
        if self.fishing.at_port(frame):
            return '船在港口'
        if not self.fishing.adaptive_auto_on(frame):
            return '自适应 AUTO 未开启'
        box = self.fishing.box(frame)
        if box is None:
            return '计量框没找到'
        if self.fishing.selected(frame, box) is None:
            return '计量数字读不清'
        if self.world.training._number(frame, ((76,1248,137,1278),(79,1249,136,1277))) is None:
            return '角色等级读不清'
        return '计量不可调整（自适应关或数据不足）'

    def _confirm_resumed_route(self, frame):
        screen = self.screen(frame, resumed_route=True)
        if screen is None or not (self.fishing.blue_fill(frame) or self.fishing.real_battle(frame)):
            self.key = self.seen_at = None
            self.seen = 0
            return False
        return self._confirmed(('own-resumed-route', screen['rank'], screen['stage'], True))

    @staticmethod
    def needs_adjustment(data):
        return bool(data and data.get('adaptive') is True
                    and data['player']-data['value'] > MeterCalibration.GAP
                    and (data['high'] is not None and data['value'] < data['high']
                         or data['high'] is None and data.get('y',0) >
                            FishingMeter.TOP+FishingMeter.MAX_ENDPOINT_SLOP))

    def settings_key(self, frame):
        if not self.sailing.menu(frame):
            return None
        values = tuple(self.sailing.selected(frame,name) for name in TARGETS)
        if any(value is None for value in values) or not any(values):
            return None
        # Keep both native extra checkboxes and destination priority unchanged;
        # exclude the sale-coupon counter, which is allowed to decrease.
        # Only the two button interiors belong to the priority fingerprint.
        # The old x=240..627 rectangle extended past the parchment into the
        # moving sea/character background, preventing three-frame agreement
        # even though all native options remained unchanged.
        bits = b''.join((np.max(frame[y:Y,x:X],axis=2) < threshold).tobytes()
                        for (x,y,X,Y),threshold in (((557,627,595,667),110),
                            ((557,741,595,781),110),((260,520,415,574),70),
                            ((434,520,594,574),70)))
        return values, hashlib.sha256(bits).hexdigest()

    def input_allowed(self):
        if not self.active or not self.enabled():
            return False
        deadline=(self.journal['restore_retry']['deadline'] if self._restoring_only()
                  else self.journal['deadline'])
        return time.time() < deadline

    def stop_input_allowed(self):
        return (self.input_allowed() and not self._restoring_only()
                and self.journal['deadline'] - time.time() >= self.STOP_CONFIRM_BUDGET)

    def window_blocker(self):
        """记录/时间类的守门:与画面无关,可单独判定。"""
        if not self.input_allowed():
            return '校准已过 90 秒窗口或功能已关闭'
        if not (self.route_verified if self.preserves_route else self.off_verified):
            return '尚未确认当前航巡及地点' if self.preserves_route else '尚未确认停航'
        return None

    def frame_blocker(self, frame):
        """画面类的守门:逐项检查,返回第一项不通过的原因。"""
        if frame.shape[:2] != (1280, 720):
            return f'画面尺寸不是 1280x720（当前 {frame.shape[:2]}）'
        if not self.world._parallel(frame):
            return '不在平行世界'
        if self.sailing.running(frame) is not self.expected_route():
            return '航巡状态变化或未知' if self.preserves_route else '自动航行仍在运行或状态未知'
        if self.world.training.boss_mode(frame):
            return ('任务 BOSS 标记且已确认实际战斗' if self.fishing.real_battle(frame)
                    else '当前有任务 BOSS 标记，禁止拖动或长按')
        if self.fishing.at_port(frame):
            return '停在港口'
        strokes = self.world._stage_strokes(frame)
        if strokes is None:
            return '阶段徽章读不到笔画'
        if strokes.hex() != self.journal.get('stage_strokes'):
            return '阶段已经变化（换到别的阶段了）'
        if self.fishing._context(frame) != self.context:
            return '地点或角色状态与校准确认时不一致'
        return None

    def meter_frame_blocker(self, frame):
        """返回当前帧为何不能开始调整计量(全部通过时返回 None)。

        这七项原先写成一个布尔表达式,一旦不成立就只是"没反应";2026-10-10 那次
        90 秒超时的现场记录里,``sailing_attempts`` 只有一次 ``stop``、``attempts`` 全空,
        说明整段时间都卡在这里 —— 具体卡哪一项必须能查。
        """
        return self.window_blocker() or self.frame_blocker(frame)

    def meter_frame_allowed(self, frame):
        # The short cancellation window uses freshly captured exact strokes,
        # rather than a second OCR pass that could consume its 120 ms budget.
        blocker = self.meter_frame_blocker(frame)
        if blocker is not None:
            if self.active and self.phase == 'prepare':
                self._report_wait(f'暂不能调整计量：{blocker}')
        elif self.hold:
            self.hold = None
        return blocker is None

    def reserve_meter_input(self, action):
        if self._restoring_only():
            return False
        if not self.input_allowed():
            return False
        if action == 'resume_cast':
            if (self.phase != 'restore_fishing' or not self.preserves_route
                    or not self.journal.get('attempts', {}).get('cancel', 0)
                    or self.journal.get('attempts', {}).get('start', 0)
                    or self.journal['deadline']-time.time() < 8):
                return False
            # A short restoration cast consumes the same unused start allowance.
            # Persist before input, including uncertain transport failures.
            self.journal.setdefault('attempts', {})['start'] = 1
            self.journal['fishing_restore'].update(input_reserved_at=time.time(),
                                                  action='resume_cast')
            self._save()
            return True
        if self.phase == 'restore_fishing':
            return False
        if (self.preserves_route and action in ('cancel', 'scan_cancel', 'drag', 'reveal')
                and self.journal['deadline']-time.time() <= self.RESTORE_RESERVE):
            return False
        action = 'cancel' if action == 'scan_cancel' else action
        counts = self.journal.setdefault('attempts', {})
        used = (counts.get('drag',0)+counts.get('reveal',0) if action in ('drag','reveal')
                else counts.get(action,0))
        limit = 4 if action in ('drag','reveal') else 2 if action == 'cancel' else 3
        if used >= limit:
            raise MeterWorkflowLimit('标尺校准已达到本轮输入上限')
        counts[action] = counts.get(action, 0)+1
        self._save()
        return True

    def reserve_sailing_input(self, action):
        if self.preserves_route:
            return False
        repair_open = (self._restoring_only() and action == 'open'
                       and self.journal['restore_retry'].get('from_stopped_main') is True
                       and self.phase == 'open')
        if self._restoring_only() and action not in ('start',native_auto_sale.ACTION) and not repair_open:
            return False
        if not self.input_allowed():
            return False
        if action == 'stop' and not self.stop_input_allowed():
            return False
        counts = self.journal.setdefault('sailing_attempts', {})
        if counts.get(action,0) >= (1 if action in ('start',native_auto_sale.ACTION) or repair_open else 3):
            raise MeterWorkflowLimit('标尺校准航巡操作达到本轮上限')
        counts[action] = counts.get(action,0)+1
        if action == 'start':
            # A process exit between Start and arrow confirmation must not
            # blindly toggle an already running native route on restart.
            self.journal['resume_requested'] = True
        self._save()
        return True

    def validate_sailing(self, action, frame):
        if self.preserves_route:
            return False
        repair_open = (self._restoring_only() and action == 'open'
                       and self.journal['restore_retry'].get('from_stopped_main') is True
                       and self.phase == 'open'
                       and not self.journal.get('sailing_attempts',{}).get('open',0))
        if self._restoring_only() and action not in ('start',native_auto_sale.ACTION) and not repair_open:
            return False
        if not self.input_allowed():
            return False
        screen = self.screen(frame)
        if screen is None:
            return False
        if action == 'stop':
            # screen() performs OCR; recheck the budget after that work.
            return (self.phase == 'stop' and screen['route'] is True
                    and self.stop_input_allowed())
        if action == 'open':
            return (self.phase == 'open' and screen['route'] is False and self.off_verified
                    and self.fishing._context(frame) == self.context)
        if action == 'start':
            return (self.phase == 'menu' and screen['menu'] and self.off_verified
                    and self.settings_key(frame) == self.menu_key
                    and (self.sailing.sale_selected(frame) is True
                         or getattr(self,'preserve_native_sale',False) and self.sailing.sale_selected(frame) is not None))
        if action == native_auto_sale.ACTION:
            return (self.phase == 'menu' and screen['menu'] and self.off_verified
                    and native_auto_sale.valid_off(self,frame,self.settings_key(frame)))
        return False

    def _click(self, action, x, y, label):
        result = self.sailing_action(action, x, y, label)
        self.next_at = time.monotonic()+1.5
        return result

    def _finish(self):
        self.cycles += 1
        self.journal.update(pending=False, cycles=self.cycles,
                            next_check=time.time()+self.interval, finished_at=time.time())
        self.world.training.state = {'mode': 'idle',
            'fishing_meter_limits': self.fishing.range_cache}
        self._save()
        self._set('idle')
        self.event('标尺校准：已确认实际钓鱼，航巡及原设置全程保留' if self.preserves_route
                   else '标尺校准：已确认实际填充并恢复原航巡设置' if self.journal['restore_running']
                   else '标尺校准：已确认实际填充，保留原停航状态')

    @staticmethod
    def _selection_signature(data):
        return dict(box=list(data['box']),digit_strokes=data['digit_strokes'],
                    world_strokes=data['world_strokes']) if data else None

    def _catch_result(self, frame):
        # Read-only result evidence. Mask the fixed green check and its narrow
        # outline, excluding the moving fill/background of the original crop.
        template = self.world.assets['success-check']
        scores = cv2.matchTemplate(frame[955:1060,290:432], template,
            cv2.TM_CCOEFF_NORMED, mask=self.world.success_mask)
        scores = np.where(np.isfinite(scores), scores, -1)
        _, score, _, (x,y) = cv2.minMaxLoc(scores)
        h,w = template.shape[:2]
        hsv = cv2.cvtColor(frame[955+y:955+y+h,290+x:290+x+w], cv2.COLOR_RGB2HSV)
        green = ((hsv[:,:,0]>=25)&(hsv[:,:,0]<=90)&(hsv[:,:,1]>=80)&(hsv[:,:,2]>=90))
        return score >= .97 and int((green & (self.world.success_mask>0)).sum()) >= 700

    def _progressed_route(self, frame):
        """Read-only evidence that an expired, older RANK no longer owns the UI.

        This never authorizes a meter or sailing input. In particular, the
        lower AUTO checkbox can glow during fishing; use the independent
        footer controls and actual green route arrow here. A genuine catch
        also proves a newer-RANK fishing cycle, without crediting this expired
        calibration as completed. Identified preparation can bridge that
        read-only consensus, but cannot complete it by itself.
        Native navigation may already target a BOSS at the newer RANK. That
        does not authorize any input; screen() retains its strict BOSS guard.
        """
        if (not self.active or self.journal.get('restore_running') is not True
                or frame.shape != (1280,720,3) or not self.world._parallel(frame)
                or self.sailing.menu(frame)
                or not self.sailing._fixed(frame,'auto-label',(186,34,253,65),.90)
                or not self.sailing._fixed(frame,'active-arrow',(194,4,246,42),.96)
                or self.sailing._fixed(frame,'inactive-face',(194,4,246,42),.96)
                or not all(self.world.training._match(frame,name,box,.91)
                    for name,box in (('guard-star',(208,1137,269,1192)),
                                     ('guard-diamond',(539,1137,600,1186))))
                or float((frame[1147:1168,558:585].max(axis=2)>=210).mean()) < .30):
            return None
        evidence = ('fill' if self.fishing.blue_fill(frame,main_known=True) else
                    'battle' if self.fishing.real_battle(frame) else
                    'catch' if self._catch_result(frame) else None)
        if evidence is None and (self.fishing.idle_icon(frame,main_known=True)
                                 or self.fishing.at_port(frame,main_known=True)):
            return None
        rank,stage=self.world._rank_value(frame),self._stage(frame)
        if (rank is None or not self.journal['rank'] < rank < 40
                or stage != self.journal['stage']):
            return None
        box=self.fishing.box(frame)
        value=self.fishing.selected(frame,box) if box else None
        player=self.world.training._number(frame,((76,1248,137,1278),(79,1249,136,1277)))
        if value is None or player is None:
            return None
        return dict(rank=rank,stage=stage,route=True,player=player,value=value,
                    context=self.fishing._context(frame),fishing_evidence=evidence)

    def _retire_external(self, data, *, kind):
        original=self.path.read_bytes()
        identifier=hashlib.sha256(original).hexdigest()
        archive=self.path.parent/'logs'/'meter-calibration'/('expired-'+identifier+'.json')
        archive.parent.mkdir(parents=True,exist_ok=True)
        if archive.is_file() and archive.read_bytes()!=original:
            raise RuntimeError('旧标尺校准记录归档冲突，已停止')
        if not archive.is_file():
            archive.write_bytes(original)
        self.journal.update(pending=False,next_check=time.time()+self.failure_backoff(),
            resolution=dict(kind=kind,verified_at=time.time(),calibration_completed=False,
                player=data['player'],value=data['value'],rank=data['rank'],stage=data['stage'],
                route=data['route'],fishing_evidence=data.get('fishing_evidence'),
                archive=str(archive),sha256=identifier))
        self.world.training.state=dict(mode='idle',fishing_meter_limits=self.fishing.range_cache)
        self.world.rank=None
        self._save()
        self._set('idle')
        if kind == 'deferred_before_any_input':
            self.hold = '本轮尚未发出任何输入，原航巡仍在实际钓鱼；保留到期记录，等待下次安全校准'
            self.event('标尺校准：停航前未取得操作窗口，连续三帧三秒确认原航巡实际钓鱼；已归档原截止时间和零输入记录，延期检测，不计校准成功')
        else:
            self.event(f"旧标尺校准已到期；连续确认当前 RANK{data['rank']}、标尺 {data['value']}、角色 Lv{data['player']}及原航巡实际钓鱼，已只读结束旧恢复等待；未计校准成功，差距仍大时待下次重新检测")

    def _untouched_route_wait(self):
        """Only a route we never touched can be deferred without restoring it."""
        return (self.active and self.phase == 'stop' and not self.off_verified
                and self.journal.get('restore_running') is True
                and self.journal.get('resume_requested') is False
                and self.journal.get('proof') is None
                and self.journal.get('stopped_context') is None
                and not any(self.journal.get('attempts', {}).values())
                and not any(self.journal.get('sailing_attempts', {}).values())
                and self.journal.get('restore_retry') is None)

    def pre_battle_step(self, frame):
        """Service a pending safe route stop before optional combat owns the loop.

        This cannot start calibration, cancel fishing or act on a task BOSS.
        The input callback still captures again and applies validate_sailing.
        """
        if self.awaiting_recovery:
            return self._observe_retired(frame)
        if not self.active or (self.phase != 'stop'
                               and not (self.preserves_route and self.phase in
                                        ('route_context','prepare','restore_fishing'))):
            return False
        if not self.input_allowed():
            # A never-issued wait can be observed even while a BOSS owns the
            # next combat burst. This issues no input and never takes combat
            # ownership; its original deadline remains expired throughout.
            if self._untouched_route_wait():
                self.step(frame)
            return False
        screen = self.screen(frame, restore_fishing=self.phase == 'restore_fishing')
        if screen is None or screen['menu']:
            return False
        return self.step(frame)

    def _defer_untouched(self, frame, screen):
        if (not self._untouched_route_wait() or not screen or screen['menu']
                or screen['route'] is not True
                or not (self.fishing.blue_fill(frame) or self.fishing.real_battle(frame))):
            self.key = self.seen_at = None
            self.seen = 0
            return False
        # No range cache or saved trigger value is used to infer a new meter.
        # R50/R51 encounters, menus, ports and changed stages were already
        # rejected by screen(). A task BOSS is allowed only in this read-only
        # zero-input retirement, never in validate_sailing or meter gestures.
        box = self.fishing.box(frame)
        value = self.fishing.selected(frame, box) if box else None
        player = self.world.training._number(frame, ((76,1248,137,1278), (79,1249,136,1277)))
        if value is None or player is None:
            self.key = self.seen_at = None
            self.seen = 0
            return False
        context = self.fishing._context(frame)
        if not self._confirmed(('untouched-route', screen['rank'], screen['stage'], context)):
            return False
        self.player, self.value = player, value
        self._retire_external(dict(player=player, value=value, rank=screen['rank'],
            stage=screen['stage'], route=True, fishing_evidence='fill_or_battle'),
            kind='deferred_before_any_input')
        return True

    def _resolve_expired(self, frame):
        # An already started native route can advance RANK before this old
        # journal is retired. Check that read-only case before the same-RANK
        # completion probe; its rejected frame resets that probe's consensus.
        screen = self.screen(frame)
        if self._untouched_route_wait():
            observed = screen or self.screen(frame, observe_untouched=True)
            if observed is not None:
                return self._defer_untouched(frame, observed)
        progressed = self._progressed_route(frame) if screen is None else None
        if progressed is not None:
            self.player,self.value=progressed['player'],progressed['value']
            key=('expired-rank-progress',progressed['rank'],progressed['stage'],
                 progressed['context'],progressed['value'])
            # Require three matching current-context frames over three seconds
            # AND a fresh actual catch/fill/battle in the completing frame.
            # The route alone never completes this read-only retirement; this
            # does not change the separate three-frame post-hold start proof.
            if not self._confirmed(key) or progressed['fishing_evidence'] is None:
                return False
            kind = ('verified_route_after_rank_progress' if self._expects_resumed_route()
                    else 'externally_resumed_after_rank_progress')
            self._retire_external(progressed,kind=kind)
            return True
        # The fill and sole route Start were verified inside the saved input
        # deadline. Reconfirm their result without touching a running route or
        # reopening the input window, including when it has reached a BOSS.
        if self._expects_resumed_route():
            if not self._confirm_resumed_route(frame):
                return False
            original = self.path.read_bytes()
            identifier = hashlib.sha256(original).hexdigest()
            archive = self.path.parent/'logs'/'meter-calibration'/('expired-'+identifier+'.json')
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.is_file() and archive.read_bytes() != original:
                raise RuntimeError('旧标尺校准记录归档冲突，已停止')
            if not archive.is_file():
                archive.write_bytes(original)
            self.journal['resolution'] = dict(kind='verified_resume_after_expiry',
                verified_at=time.time(), calibration_completed=True,
                archive=str(archive), sha256=identifier)
            self._finish()
            return True
        # Expired deadlines never authorize more input. A user/native change
        # may already have restored the route, with or without correcting the
        # meter. Observe it for three seconds, preserving failed attempt bytes;
        # an uncorrected meter remains eligible only after the next-check time.
        data = (self.fishing.measure(frame, remember=False)
                if screen and not screen['menu'] else None)
        corrected = bool(data and data['player']-data['value'] <= self.GAP)
        resumed_route = bool(screen and screen['route'] is True
                             and self.journal['restore_running'] is True)
        restored = bool(screen and data and screen['route'] is self.journal['restore_running']
                        and (corrected or resumed_route) and self.fishing.blue_fill(frame))
        if not restored:
            self.key = self.seen_at = None
            self.seen = 0
            return False
        self.player, self.value = data['player'], data['value']
        key = ('expired-restored',screen['rank'],screen['stage'],screen['route'],data['context'],data['value'])
        if not self._confirmed(key):
            return False
        kind = 'externally_restored' if corrected else 'externally_resumed_uncalibrated'
        self._retire_external(dict(data,rank=screen['rank'],stage=screen['stage'],route=screen['route']),kind=kind)
        return True

    def _observe_cancel_window(self, frame):
        # Do not spend a player/range OCR pass before the short ordinary cast.
        # Fixed selected-digit strokes must still match the measured trigger.
        # This observes combat/ambiguous frames; only scan_cancel_frame's fresh
        # ordinary-preparation + X guard can actually authorize a cancel.
        if (self.journal.get('attempts',{}).get('start',0)
                or self.fishing.start_fill_verified or self.fishing.idle_icon(frame)):
            return False
        reference = self.journal.get('selected_signature')
        data = self.fishing.cast_data(frame)
        if not reference or self._selection_signature(data) != reference:
            return False
        if self.fishing.cancels >= 2:
            raise MeterWorkflowLimit('标尺校准取消已达到本轮上限')
        if self.fishing.action('scan_cancel',data):
            self.fishing.cancels += 1
            self.fishing.idle_since = None
            self.fishing.phase = 'detect'
            self.fishing.next_action_at = time.monotonic()+1.5
        return True

    def _begin_fishing_restore(self, reason):
        self.journal['fishing_restore'] = dict(reason=reason, entered_at=time.time())
        self.restore_cast_context = None
        self.restore_frame = None
        self._set('restore_fishing')
        self._save()
        self.event(f'标尺校准：已尝试停钓后{reason}；停止调整，使用原90秒剩余时间仅恢复钓鱼，不计校准成功')

    def resume_frame_allowed(self, frame):
        """Only the separate short cast may operate at a newly reached task point."""
        return bool(self.input_allowed() and self.phase == 'restore_fishing'
            and self.journal['deadline']-time.time() >= 8
            and not self.journal.get('attempts', {}).get('start', 0)
            and self.restore_cast_context is not None
            and self.screen(frame, restore_fishing=True) is not None
            and self.fishing._context(frame) == self.restore_cast_context
            and self.fishing.resume_cast_data(frame) is not None)

    def _restore_fishing(self, frame):
        # The original deadline/used actions still apply; no new recovery window.
        if frame is self.restore_frame:
            return True
        self.restore_frame = frame
        screen = self.screen(frame, restore_fishing=True)
        if screen is None:
            self._set('restore_fishing')
            self.restore_cast_context = None
            self._report_wait('只恢复钓鱼：等待同世界/阶段/RANK、原航巡与明确主画面')
            return True
        context = self.fishing._context(frame)
        evidence = ('fill' if self.fishing.blue_fill(frame, main_known=True)
                    else 'battle' if self.fishing.real_battle(frame) else None)
        if evidence:
            # This includes fishing that continued after a missed cancel. Never
            # cancel it again, or claim the old meter has been calibrated.
            if self._confirmed(('restore-fishing-proof', screen['rank'], screen['stage'],
                                screen['route'], context)):
                self._retire_window('调整条件变化，仅恢复钓鱼')
                self.journal['retirement'].update(released=True, verified_at=time.time(),
                    evidence=dict(**screen, context=context, fishing=evidence),
                    calibration_completed=False)
                self.journal['next_check'] = time.time()+self.failure_backoff()
                self._save()
                self._set('idle')
                self.hold = '原航巡及真实钓鱼已恢复；本轮未计校准成功'
                self.event('标尺校准：连续三张新图至少三秒确认实际钓鱼和原航巡；仅恢复完成，原截止及次数已归档')
                return False
            return True
        data = self.fishing.resume_cast_data(frame)
        if data is None or self.journal.get('attempts', {}).get('start', 0):
            self._set('restore_fishing')
            self.restore_cast_context = None
            self._report_wait('只恢复钓鱼：等待灰色鱼钩连续休眠；已有启动输入时只核对结果')
            return True
        key = ('restore-idle', screen['rank'], screen['stage'], screen['route'],
               context, data['digit_strokes'], data['world_strokes'], tuple(data['box']))
        if self._confirmed(key):
            self.restore_cast_context = context
            self.fishing.action('resume_cast', data)
            self._set('restore_fishing')
            self.restore_cast_context = None
        return True

    def _candidate_ready(self, frame):
        # Three context/number observations are collected separately. A short
        # ordinary preparation need not last three frames; the input callback
        # still re-captures and validates the fish/X immediately before contact.
        if self.fishing.real_battle(frame):
            self._candidate_reason = 'battle'
            return False
        if self._catch_result(frame):
            self._candidate_reason = 'catch_result'
            return False
        if self.fishing.idle_icon(frame) and not self.fishing.busy(frame):
            self._candidate_reason = 'idle_ready'
            return True
        ready = self.fishing.ordinary_preparation(frame, allow_fill=True)
        self._candidate_reason = 'ordinary_preparation' if ready else 'no_preparation_window'
        return ready

    def _candidate_note(self, reason, *, detail=None, key=None):
        trace = getattr(self, '_candidate_trace', None)
        if trace is not None:
            trace.check(reason, detail=detail, key=key)

    def observe_candidate_blocked(self, reason, detail=None):
        if self.candidate_at is None or self.active:
            return False
        self._candidate_note('runtime_'+reason, detail=detail)
        return True

    def finish_candidate_observation(self, outcome):
        trace = getattr(self, '_candidate_trace', None)
        if trace is None:
            return False
        trace.finish(outcome)
        return True

    def _report_wait(self, reason, data=None):
        """限频地把"为什么现在不校准"记进日志,并留在状态里给面板显示。

        这条链上原来只有"触发"时才说话,于是用户只能看到功能没反应、面板上
        `player_level` 又是空的,无法判断是读数问题还是条件不满足。这里不超过
        每 60 秒一条,措辞只说事实(读到什么、差多少)。
        """
        self.hold = reason
        if data is not None:
            self.player, self.value = data['player'], data['value']
        now = time.monotonic()
        # Keep live hold/readings current, but identical reasons are not a
        # minute-by-minute log. Limit transitions too, to avoid menu flicker.
        previous = self.reported_reasons.get(reason)
        if now - self.last_report < 60 or previous is not None and now-previous < 1800:
            return
        self.last_report = now
        self.reported_reasons[reason] = now
        if len(self.reported_reasons) > 32:
            self.reported_reasons.pop(next(iter(self.reported_reasons)))
        if data is None:
            self.event(f'标尺校准：暂不触发 —— {reason}')
            return
        gap = data['player'] - data['value']
        self.event(f"标尺校准：暂不触发 —— {reason}（角色 Lv{data['player']}、标尺 {data['value']}、"
                   f"差距 {gap}，上限 {data['high']}）")

    def step(self, frame):
        if self.awaiting_recovery:
            return self._observe_retired(frame)
        try:
            return self._step(frame)
        except MeterWorkflowLimit as exc:
            if (self.active and self.preserves_route and self.phase != 'restore_fishing'
                    and self.journal.get('attempts', {}).get('cancel', 0)
                    and not self.journal.get('attempts', {}).get('start', 0)
                    and self.input_allowed() and self.journal['deadline']-time.time() >= 8):
                self._begin_fishing_restore(str(exc))
                return True
            self._retire_window(str(exc))
            return False

    def _step(self, frame):
        if self.active:
            if not self.enabled():
                raise RuntimeError('标尺校准途中功能已关闭，已停止并保留恢复记录')
            if self.retry is not None:
                if self._apply_retry(frame):
                    return True
                if self.expired_observe_until is not None and time.monotonic() < self.expired_observe_until:
                    return True
                raise RuntimeError('标尺修复重试未确认原地点停航和休眠，保留原失败记录')
            if time.time() >= self.journal['deadline']:
                if (self._untouched_route_wait()
                        and self.untouched_expiry_observed != self.journal['deadline']):
                    # A bounded read-only consensus does not renew the expired
                    # input budget. Used stop/cancel/drag/start limits retain
                    # their original failure behavior and cannot enter here.
                    self.untouched_expiry_observed = self.journal['deadline']
                    self.expired_observe_until = time.monotonic()+15
                if self._restoring_only():
                    if time.time() >= self.journal['restore_retry']['deadline']:
                        if self._resolve_expired(frame):
                            return False
                        # A restart needs the same bounded read-only consensus
                        # as an expired normal calibration. Neither deadline
                        # nor any input allowance changes during observation.
                        if self.expired_observe_until is not None and time.monotonic() < self.expired_observe_until:
                            return True
                        raise MeterWorkflowLimit('标尺填充已验证，但40秒航巡恢复未完成')
                elif self._resolve_expired(frame):
                    return False
                elif not self._restoring_only():
                    self._report_wait('上次校准已到期；只读核对原航巡是否恢复，保留原截止时间和尝试次数')
                    # Read-only verification cannot extend the saved input deadline.
                    if self.expired_observe_until is not None and time.monotonic() < self.expired_observe_until:
                        return True
                    raise MeterWorkflowLimit('标尺校准90秒内未完成')
        elif not self.enabled():
            self._report_wait('标尺校准功能已关闭')
            return False
        elif time.time() < self._next_check():
            self._report_wait('上次校准未完成；已确认钓鱼恢复，等待失败退避间隔结束'
                              if self._failed_release_at() is not None else
                              '上次成功校准后的冷却尚未结束')
            return False
        if self.active and self.preserves_route:
            cancelled = self.journal.get('attempts', {}).get('cancel', 0)
            if cancelled and self.phase != 'restore_fishing':
                reason = ('到达任务标记地点' if self.world.training.boss_mode(frame)
                          else '调整时间不足，预留恢复时间'
                          if not self.journal.get('attempts', {}).get('start', 0)
                          and self.journal['deadline']-time.time() <= self.RESTORE_RESERVE else None)
                if reason:
                    self._begin_fishing_restore(reason)
            if self.phase == 'restore_fishing':
                return self._restore_fishing(frame)
        if not self.active and self.candidate_at is not None and time.time()-self.candidate_at >= self.CANDIDATE_LIMIT:
            self.finish_candidate_observation('candidate_timeout')
            self.journal.update(candidate_result=dict(reason='no_safe_window', at=time.time(),
                                                     started_at=self.candidate_at, inputs=0),
                                next_check=time.time()+self.failure_backoff())
            self._save()
            self.candidate_at = None
            self._set('idle')
            self._report_wait('300秒内未出现合适的停钓/休眠窗口，未发出输入，延期检测')
            return False
        if self.phase == 'running':
            if self._confirm_resumed_route(frame):
                self._finish()
            return True
        if self._restoring_only() and not self.off_verified:
            # A process restart never reuses the original OFF proof or renews
            # this explicitly requested repair window.
            context = self._restore_repair_context(frame)
            if context is not None and self._confirmed(context[1]):
                self.off_verified = True
                self._set(context[0])
            return True
        # The already verified stopped context includes exact current RANK,
        # world/stage and location strokes. Avoid another numeric OCR pass
        # before the short cancellation scan; all inputs still take new frames.
        screen = (dict(rank=self.journal['rank'],stage=self.journal['stage'],menu=False,route=self.expected_route())
                  if self.active and self.phase == 'prepare' and self.meter_frame_allowed(frame)
                  else self.screen(frame))
        if screen is None:
            self.key = self.seen_at = None
            self.seen = 0
            # 写清是哪一条画面条件没过:泛泛的"不是主画面"无法区分世界/航巡/港口/RANK。
            self.wait_unmet = self.screen_reason or '画面条件未知'
            self._candidate_note('screen_guard', detail=self.wait_unmet)
            self._report_wait(f'当前画面不是已确认的平行世界主画面：{self.wait_unmet}')
            return self.active
        if not self.active:
            if screen['menu']:
                self._candidate_note('sailing_menu')
                self._report_wait('正在航行设置菜单里，等退出后再判断')
                return False
            data = self.fishing.measure(frame)
            if data:
                self.player, self.value = data['player'], data['value']
            if not self.needs_adjustment(data):
                if data is not None:
                    self.finish_candidate_observation('gap_closed' if data['player']-data['value'] <= self.GAP
                                                      else 'available_limit')
                    self.candidate_at = None
                self.key = self.seen_at = None
                self.seen = 0
                if data is None:
                    self.wait_unmet = self._measure_unmet(frame)
                    self._candidate_note('reading_unavailable', detail=self.wait_unmet)
                    self._report_wait(f'计量与角色等级没读清：{self.wait_unmet}')
                elif data['player'] - data['value'] <= self.GAP:
                    self._report_wait(f'差距没超过 {self.GAP} 级，无需校准', data)
                else:
                    self._report_wait('标尺已经在可用上限，无需校准', data)
                return False
            if self.candidate_at is None:
                self.candidate_at = time.time()
                if getattr(self, '_candidate_trace', None) is None:
                    self._candidate_trace = CandidateObservation(self.event, clock=time.monotonic)
                self._candidate_trace.begin(self.candidate_at, rank=screen['rank'], stage=screen['stage'],
                    player=data['player'], value=data['value'], high=data['high'])
            key = (screen['rank'], screen['stage'], screen['route'], data['context'], data['value'], data['high'])
            confirmed = self._confirmed(key,max_gap=self.CANDIDATE_FRAME_GAP)
            ready = confirmed and self._candidate_ready(frame)
            self._candidate_note(getattr(self, '_candidate_reason', 'window_unknown') if confirmed
                                 else 'context_confirmation', key=key)
            if not ready:
                self._report_wait('只读候选：等待普通鱼准备或明确休眠，再开始90秒执行窗口', data)
                return False
            self.journal.update(pending=True, rank=screen['rank'], stage=screen['stage'],
                restore_running=screen['route'], deadline=time.time()+self.LIMIT,
                value=data['value'], high=data['high'], attempts={}, sailing_attempts={}, auto_sale_guard=None,
                trigger_player=data['player'],
                route_policy='preserve', calibration_context=None,
                proof=None, resolution=None, resume_requested=False, stopped_context=None,
                window_restarted_at=None, retirement=None, retired_at=None, restore_retry=None,
                fishing_restore=None,
                stage_strokes=self.world._stage_strokes(frame).hex())
            self.candidate_at = None
            self.resumed_pending = False
            self.world.rank = screen['rank']
            self.world.training.state = dict(mode='farming', rank=screen['rank'],
                sailing_enabled=screen['route'], sailing_sleep_recovery=True, sailing_meter_adjustment=True)
            self.fishing.current, self.fishing.goal = data['value'], data.get('goal')
            self.journal['target'] = data.get('goal')
            self.fishing.saved_player_gap = True
            # cast_data labels the fixed strokes with the saved numeric value.
            self.journal['selected_signature'] = self._selection_signature(self.fishing.cast_data(frame))
            self.fishing.cancels = self.fishing.drags = self.fishing.starts = 0
            self.fishing.phase, self.fishing.next_action_at = 'detect', 0
            self.fishing.start_fill_verified = False
            self.off_verified = False
            self.route_verified, self.context = True, data['context']
            self.journal['calibration_context'] = self.context
            self.fishing.start_route_expected = self.expected_route()
            self.hold = None
            self._save()
            self._set('prepare')
            self.finish_candidate_observation('execution_started')
            self.event(f"标尺校准：角色 Lv{data['player']}、标尺 {data['value']}，差距超过{self.GAP}；保留航巡，仅停止持续钓鱼后校准")
            return True
        if time.monotonic() < self.next_at:
            return True
        if self.preserves_route:
            if screen['menu'] or screen['route'] is not self.expected_route():
                self.route_verified = False
                self.fishing.idle_since = None
                self._set('route_context')
                return True
            context = self.fishing._context(frame)
            if self.phase == 'prepare' and context != self.context:
                self.route_verified = False
                self.fishing.idle_since = None
                self.fishing.stable, self.fishing.key = 0, None
                self._set('route_context')
            if self.phase == 'route_context':
                if not self._confirmed(('route-context', screen['rank'], screen['stage'], screen['route'], context)):
                    return True
                # Re-read at a newly reached location; never move the deadline
                # or clear spent cancel/drag/start counts during this re-anchor.
                data = self.fishing.measure(frame)
                if data is None:
                    return True
                self.context, self.route_verified = context, True
                self.fishing.current = data['value']
                self.journal['calibration_context'] = context
                self.journal['selected_signature'] = self._selection_signature(self.fishing.cast_data(frame))
                self._save()
                self.fishing.idle_since = None
                self._set('prepare')
                return True
        elif screen['route'] is True and self.phase != 'running':
            self.off_verified = False
            self.world.training.state['sailing_enabled'] = True
            if self.phase != 'stop':
                self._set('stop')
        if self.phase == 'stop':
            if screen['route'] is True:
                if not self.stop_input_allowed():
                    self._report_wait('停航确认所需时间不足；保留当前航巡，不追加停航输入')
                    if self._defer_untouched(frame, screen):
                        return False
                    return True
                self._click('stop',233,45,'标尺校准：停止当前游戏航巡')
            elif screen['route'] is False:
                context = self.fishing._context(frame)
                if self._confirmed(context):
                    self.off_verified, self.context = True, context
                    self.journal['stopped_context'] = context
                    self.world.training.state['sailing_enabled'] = False
                    self._save()
                    self.fishing.idle_since = None
                    self._set('prepare')
            elif (screen['menu'] and self.journal.get('proof')
                    and self.journal.get('stopped_context')):
                settings = self.settings_key(frame)
                if settings is not None and self._confirmed(('stopped-menu',settings)):
                    self.off_verified = True
                    self._set('menu')
            return True
        if self.phase == 'prepare':
            if not self.meter_frame_allowed(frame):
                return True
            if self._observe_cancel_window(frame):
                return True
            started = bool(self.journal.get('attempts',{}).get('start',0))
            if started and not self.fishing.start_fill_verified:
                if self.fishing.phase != 'starting':
                    self.fishing.arm_start_observation(frame)
                self.fishing.confirm_start(frame)
            verified_start = bool(started and self.fishing.start_fill_verified
                                  and self.fishing.post_start_evidence(frame))
            # Idle preparation performs its own complete numeric read below.
            # Only an existing fill needs this additional read-only adoption
            # probe; duplicating it on every idle frame spends the 90s budget.
            data = (self.fishing.measure(frame)
                    if not started and self.fishing.blue_fill(frame) else None)
            at_target_fill = bool(data and self.fishing.close_enough(data))
            if verified_start or at_target_fill:
                proof = self.fishing.start_proof or 'paired_hold' if verified_start else 'observed_target_fill'
                # confirm_start already required three fresh post-release
                # frames spanning three seconds in this stopped context.
                # Do not repeat that observation and spend the route budget.
                post_release_verified = verified_start and proof == 'post_release_fishing'
                if post_release_verified or self._confirmed(('fill', self.context, proof)):
                    self.journal['proof'] = proof
                    self._save()
                    if self.journal['restore_running'] and not self.preserves_route:
                        self._set('open')
                    else:
                        self._finish()
                return True
            self.key = self.seen_at = None
            self.seen = 0
            self.fishing.prepare(frame)
            return True
        if self.phase == 'open':
            if screen['menu']:
                self._set('menu')
            elif screen['route'] is False:
                if self._click('open',233,45,'标尺校准：打开原航巡设置，核对自动贩卖') is not False:
                    self._set('menu')
            return True
        if self.phase == 'menu':
            if not screen['menu']:
                return True
            settings = self.settings_key(frame)
            if settings is not None and self._confirmed(settings):
                self.menu_key = settings
                if not native_auto_sale.verified_on(self,frame,settings,self.journal.get('sailing_attempts',{})):
                    if (not self.journal.get('sailing_attempts',{}).get(native_auto_sale.ACTION)
                            and native_auto_sale.arm(self,frame,settings)):
                        self._click(native_auto_sale.ACTION,576,647,'标尺校准：勾选游戏自动贩卖')
                        self.key = None
                    return True
                if self._click('start',360,873,'标尺校准：按游戏当前勾选项恢复航巡') is not False:
                    self._set('running')
            return True
        return True
