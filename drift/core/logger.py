"""日志:一个进程一个 per-instance logger,写 logs/clicker.log 并滚动。

原先是 app.Clicker.__init__ 里内联配置的,搬到框架层后 UI 与任务共用同一套。
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)s %(message)s"


def get_logger(name):
    """按名字取 logger(测试会按 id 区分实例)。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def setup_file_logging(logger, directory, filename="clicker.log", max_bytes=2_000_000, backups=3):
    """给 logger 挂一个滚动文件 handler;重复调用不会叠加 handler。"""
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    target = str((directory / filename).resolve())
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler) and getattr(handler, "baseFilename", None) == target:
            return logger
    handler = RotatingFileHandler(directory / filename, maxBytes=max_bytes, backupCount=backups,
                                  encoding="utf-8")
    handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(handler)
    return logger
