"""Bounded, asynchronous native evidence for issued Sea God inputs.

An issued touch is not proof that the target was hit or a laser cancelled.

原名 ``boss_evidence.py``,现为设备层证据落盘;根模块仍可导入(兼容代理)。
"""
from collections import deque
import json
from pathlib import Path
import queue
import re
import threading
import time

import cv2


class BossInputEvidence:
    def __init__(self, root, warning, limit=64, folder='boss-input', label='海神'):
        self.root = Path(root)/'logs'/folder
        self.root.mkdir(parents=True,exist_ok=True)
        self.warning,self.limit = warning,limit
        self.label = label
        self.pending = queue.Queue(maxsize=32)
        self.records = deque(p.stem[6:] for p in sorted(self.root.glob('input-*.json'),
            key=lambda p:p.stat().st_mtime)
            if re.fullmatch(r'input-[0-9a-f]{32}\.json',p.name))
        self.thread = threading.Thread(target=self._write,daemon=True)
        self.thread.start()

    def record(self, metadata, before, after=None):
        # The caller transfers dedicated copies, so later captures cannot
        # replace pixels referenced by an already-issued input.
        try:
            self.pending.put_nowait((dict(metadata),before,after))
        except queue.Full:
            self.warning(self.label+'输入证据队列已满，本次仅保留尝试计数')

    def _save(self, path, image):
        ok,data = cv2.imencode('.png',cv2.cvtColor(image,cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_PNG_COMPRESSION,1])
        if not ok:
            raise RuntimeError(self.label+'原生证据编码失败')
        path.write_bytes(data.tobytes())

    def _write(self):
        while True:
            item = self.pending.get()
            try:
                if item is None:
                    return
                metadata,before,after = item
                identifier = metadata['input_id']
                if not re.fullmatch('[0-9a-f]{32}',identifier):
                    raise ValueError(self.label+'证据编号无效')
                metadata['before_file'] = identifier+'-before.png'
                self._save(self.root/metadata['before_file'],before)
                metadata['after_file'] = identifier+'-after.png' if after is not None else None
                if after is not None:
                    self._save(self.root/metadata['after_file'],after)
                (self.root/('input-'+identifier+'.json')).write_text(
                    json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
                self.records.append(identifier)
                while len(self.records)>self.limit:
                    old = self.records.popleft()
                    for name in (old+'-before.png',old+'-after.png','input-'+old+'.json'):
                        (self.root/name).unlink(missing_ok=True)
            except Exception as error:
                self.warning(self.label+'输入证据保存失败：'+str(error))
            finally:
                self.pending.task_done()

    def flush(self, timeout=5.0):
        """Wait until every queued record has been written.

        Shutdown and tests need a deterministic point where an issued input is
        known to have been persisted; the writer thread is asynchronous, so
        reading the folder right after the task ends races the encoder.
        """
        deadline = time.monotonic()+timeout
        while self.pending.unfinished_tasks:
            if time.monotonic() >= deadline:
                return False
            time.sleep(.002)
        return True

    def close(self):
        self.flush()
        try:
            self.pending.put(None,timeout=1)
        except queue.Full:
            return
        self.thread.join(2)
