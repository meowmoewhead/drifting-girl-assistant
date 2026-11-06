"""MaaFramework task execution, with existing guarded Python game actions.

Each task step is a real Maa pipeline job. ALAS NextRun remains the scheduling
policy. A callback is invoked once, and its Python return value is independent
of the Maa job's execution status (False means the controller did not own UI).
The Maa device rejects generic input: only existing fresh-frame guarded actions
may touch the game. No retry, stale-frame fallback or default Maa delays.
"""
from collections import deque
from dataclasses import dataclass
import json
import re
import threading
import time


class MaaExecutionError(RuntimeError):
    pass


@dataclass
class Invocation:
    token: int
    node: str
    function: object
    args: tuple
    kwargs: dict
    value: object = None
    error: BaseException | None = None
    started: bool = False
    completed: bool = False


class MaaEngine:
    def __init__(self, device, stop):
        # Lazy import makes a missing/incompatible official runtime an explicit
        # startup failure; it never silently starts the former execution path.
        from maa.controller import CustomController
        from maa.custom_action import CustomAction
        from maa.define import LoggingLevelEnum
        from maa.library import Library
        from maa.resource import Resource
        from maa.tasker import Tasker
        import numpy as np

        self.stop = stop
        self.gate = threading.Lock()
        self.lock = threading.RLock()
        self.local = threading.local()
        self.pending = None
        self.nodes = set()
        self.sequence = 0
        self.completed = 0
        self.failed = 0
        self.recent = deque(maxlen=32)
        self.version = Library.version()
        self.closed = False
        self._tasker = None
        # Input screenshots and receipts are already saved by our bounded
        # evidence writers. Do not create a second unbounded framework trace.
        Tasker.set_log_dir('')
        Tasker.set_stdout_level(LoggingLevelEnum.Off)
        Tasker.set_save_draw(False)
        Tasker.set_debug_mode(False)
        Tasker.set_save_on_error(False)
        Tasker.set_reco_image_cache_limit(8)

        class GuardedDevice(CustomController):
            def connect(self): return not stop.is_set()
            def request_uuid(self): return 'drift-guarded-device'
            def get_features(self): return 0
            def screencap(self):
                if stop.is_set(): return np.empty((0,0,3),np.uint8)
                fresh = getattr(device,'screenshot_fast',device.screenshot)()
                # Maa expects BGR. Actual input guards use the original new
                # native RGB frame, never this optional recognition image.
                return np.ascontiguousarray(fresh[:,:,::-1])
            def reject(self, *args): return False
            start_app=stop_app=click=swipe=touch_down=touch_move=touch_up=reject
            click_key=input_text=key_down=key_up=reject

        owner = self
        class GuardedAction(CustomAction):
            def run(self, context, argv):
                try:
                    token = json.loads(argv.custom_action_param)['request']
                    with owner.lock:
                        request = owner.pending
                        if (request is None or request.token != token
                                or request.node != argv.node_name or request.started):
                            return False
                        request.started = True
                    if stop.is_set():
                        raise InterruptedError('已停止')
                    owner.local.inside = True
                    request.value = request.function(*request.args, **request.kwargs)
                    request.completed = True
                    return True
                except BaseException as exc:
                    # ctypes callbacks must not swallow Python exceptions or
                    # turn an uncertain input into an automatic second attempt.
                    with owner.lock:
                        if owner.pending is not None:
                            owner.pending.error = exc
                    return False
                finally:
                    owner.local.inside = False

        self._controller = GuardedDevice()
        self._controller.set_screenshot_use_raw_size(True)
        if not self._controller.post_connection().wait().succeeded:
            raise MaaExecutionError('MaaFramework 设备适配器初始化失败')
        self._resource = Resource()
        self._action = GuardedAction()
        if not self._resource.register_custom_action('DriftGuardedAction',self._action):
            raise MaaExecutionError('MaaFramework 动作注册失败')
        self._tasker = Tasker()
        if not self._tasker.bind(self._resource,self._controller) or not self._tasker.inited:
            raise MaaExecutionError('MaaFramework 任务执行器初始化失败')

    def call(self, name, function, *args, **kwargs):
        if self.stop.is_set():
            raise InterruptedError('已停止')
        # Fast combat scans, input guards and hold-time screenshots belong to
        # their current Maa job. Posting another job from that callback would
        # deadlock and would add queue latency between new frame and DOWN.
        if getattr(self.local,'inside',False):
            return function(*args, **kwargs)
        node = 'Drift_'+re.sub(r'[^a-zA-Z0-9_]', '_', name)
        with self.gate:
            if self.closed or self.stop.is_set():
                raise InterruptedError('已停止')
            if node not in self.nodes:
                if len(self.nodes) >= 256:
                    raise MaaExecutionError('MaaFramework 节点数量超过限制')
                pipeline = {node:dict(recognition='DirectHit',action='Custom',
                    custom_action='DriftGuardedAction',pre_delay=0,post_delay=0,
                    rate_limit=0,timeout=100,next=[],max_hit=1)}
                if not self._resource.override_pipeline(pipeline):
                    raise MaaExecutionError('MaaFramework 节点注册失败：'+name)
                self.nodes.add(node)
            self.sequence += 1
            request = Invocation(self.sequence,node,function,args,kwargs)
            with self.lock:
                self.pending = request
            began = time.monotonic()
            job = None
            try:
                job = self._tasker.post_task(node,{node:{'custom_action_param':{'request':request.token}}}).wait()
                if request.error is not None:
                    raise request.error
                if self.stop.is_set():
                    raise InterruptedError('已停止')
                if not job.succeeded or not request.completed:
                    raise MaaExecutionError('MaaFramework 任务执行失败：'+name)
                with self.lock:
                    self.completed += 1
                return request.value
            except BaseException:
                with self.lock:
                    self.failed += 1
                raise
            finally:
                with self.lock:
                    self.recent.append(dict(task=name,job_id=job.job_id if job else None,
                        invoked=request.started,completed=request.completed,
                        elapsed_ms=round((time.monotonic()-began)*1000,2)))
                    self.pending = None
                # Maa otherwise keeps task/reco details over an infinite run.
                # Never clear while a callback is still running.
                if not self._tasker.running:
                    self._tasker.clear_cache()

    def in_action(self):
        return bool(getattr(self.local,'inside',False))

    def request_stop(self):
        with self.lock:
            if self._tasker is not None and not self.closed:
                self._tasker.post_stop()

    def close(self):
        if self.in_action():
            raise MaaExecutionError('不能在 MaaFramework 动作内关闭当前执行器')
        with self.gate:
            with self.lock:
                if self.closed:
                    return
                self.closed = True
            # Native shutdown can wait for a callback. Status and that callback
            # must remain able to acquire the Python state lock while it waits.
            if self._tasker is not None:
                self._tasker.post_stop().wait()
                self._tasker.clear_cache()
            self._resource.clear_custom_action()
            self._tasker = None
            self._resource = None
            self._action = None
            self._controller = None

    def snapshot(self):
        with self.lock:
            return dict(name='MaaFramework',version=self.version,
                scheduler='ALAS NextRun / realtime',completed_jobs=self.completed,
                failed_jobs=self.failed,active_task=self.pending.node if self.pending else None,
                registered_nodes=len(self.nodes),closed=self.closed,recent_jobs=list(self.recent))


def framework_check():
    """Exercise the packaged native library without connecting to a device."""
    from types import SimpleNamespace
    def forbidden(*args):
        raise AssertionError('Framework self-check must not access the game')
    engine = MaaEngine(SimpleNamespace(screenshot=forbidden),threading.Event())
    try:
        marker = object()
        if engine.call('framework_check',lambda: marker) is not marker:
            raise MaaExecutionError('MaaFramework 自检返回值不符')
        return engine.snapshot()
    finally:
        engine.close()
