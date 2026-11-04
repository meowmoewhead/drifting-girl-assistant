"""ALAS-style fresh frame loop and bounded recognition timing.

The pinned ALAS screenshot loop measures its interval from capture start,
and its OCR submits multiple numeric regions together. Game controllers keep
their own deadlines, confirmations and fresh-input guards above this layer.
"""
from collections import deque
from contextlib import contextmanager
import threading
import time


class FrameRuntime:
    def __init__(self, device, stop, reader=lambda:None, clock=time.monotonic):
        self.device,self.stop,self.reader,self.clock=device,stop,reader,clock
        self.cycle_started = None
        self.capture_finished = None
        self.lock = threading.Lock()
        self.samples = {}
        self.counts = {}
        self.engine = None

    def start_engine(self):
        from drift.core.maa_engine import MaaEngine
        self.engine = MaaEngine(self.device,self.stop)

    def request_stop(self):
        if self.engine is not None:self.engine.request_stop()

    def record(self, name, seconds):
        with self.lock:
            self.samples.setdefault(name,deque(maxlen=128)).append(max(0.,seconds)*1000)
            self.counts[name] = self.counts.get(name,0)+1

    @contextmanager
    def measure(self, name):
        began=self.clock()
        try:yield
        finally:self.record(name,self.clock()-began)

    def call(self, name, function, *args, **kwargs):
        with self.measure(name):
            if self.engine is None or self.engine.in_action():
                return function(*args, **kwargs)
            def invoke():
                reader=self.reader()
                begin=getattr(reader,'begin_frame',None)
                end=getattr(reader,'end_frame',None)
                # ctypes may create a fresh Python thread state for each Maa
                # callback even on the same OS thread. Do not assume its TLS
                # survives jobs or transport a previous job's numeric cache.
                if callable(begin):begin()
                try:return function(*args, **kwargs)
                finally:
                    if callable(end):end()
            return self.engine.call(name,invoke)

    def close(self):
        if self.engine is not None:self.engine.close()
        self.end_frame()

    def end_frame(self):
        end=getattr(self.reader(),'end_frame',None)
        if callable(end):end()

    def capture(self):
        reader=self.reader()
        end=getattr(reader,'end_frame',None)
        if callable(end):end()
        if self.stop.is_set():raise InterruptedError('已停止')
        began=self.clock()
        # Every call acquires new pixels. The device alone owns the safe
        # native -> raw ADB -> fresh PNG fallback and dimension checks.
        capture=getattr(self.device,'screenshot_fast',self.device.screenshot)
        def capture_owned():
            # DigitReader scopes are thread-local. Maa callbacks use their own
            # execution thread, and must discard its previous frame as well.
            if callable(end):end()
            fresh=capture()
            if self.stop.is_set():raise InterruptedError('已停止')
            begin=getattr(reader,'begin_frame',None)
            if callable(begin):begin()
            return fresh
        frame=self.engine.call('capture',capture_owned) if self.engine is not None else capture()
        if self.stop.is_set():raise InterruptedError('已停止')
        self.record('capture',self.clock()-began)
        self.record('capture:'+getattr(self.device,'screenshot_transport','test'),self.clock()-began)
        begin=getattr(reader,'begin_frame',None)
        if callable(begin):begin()
        return frame

    def capture_loop(self):
        self.cycle_started=self.clock()
        frame=self.capture()
        self.capture_finished=self.clock()
        return frame

    def wait(self, interval):
        now=self.clock()
        if self.capture_finished is not None:
            self.record('process',now-self.capture_finished)
        remaining=max(0.,interval-(now-self.cycle_started)) if self.cycle_started is not None else interval
        stopped=self.stop.wait(remaining)
        if self.cycle_started is not None:self.record('cycle',self.clock()-self.cycle_started)
        return stopped

    def snapshot(self):
        with self.lock:
            rows={}
            for name,samples in self.samples.items():
                ordered=sorted(samples)
                rows[name]=dict(count=self.counts[name],window=len(ordered),
                    median_ms=round(ordered[len(ordered)//2],2),
                    p95_ms=round(ordered[min(len(ordered)-1,int(len(ordered)*.95))],2),
                    last_ms=round(samples[-1],2))
        stats=getattr(self.reader(),'statistics',None)
        native_status=getattr(self.device,'native_status',None)
        return dict(timings=rows,ocr=stats() if callable(stats) else None,
                    framework=self.engine.snapshot() if self.engine is not None else None,
                    transport=getattr(self.device,'screenshot_transport',None),
                    native=native_status() if callable(native_status) else None)
