"""控制面板 HTTP 层。

- :mod:`drift.web.server` —— ``make_handler(app)`` 生成处理器类,提供 ``/``、
  ``/static/style.css``、``/static/app.js``、``/api/status``、``/api/capture``、
  ``/api/frame/<id>.png``,以及需要 ``X-Session-Token`` 的 ``POST`` 端点
  (``/api/rules``、``/api/rules/change``、``/api/features/change``、
  ``/api/tasks/schedule``、``/api/settings``、``/api/start``、``/api/stop``)。
- :mod:`drift.web.cli` —— ``main()`` 实现 ``serve`` / ``diagnose`` / ``click-once``。

与 ``app`` 的关系
-----------------
``app.make_handler`` / ``app.main`` 由 ``app.py`` 转发到本包::

    from drift.web.server import make_handler
    from drift.web.cli import main

所以 ``from app import make_handler`` 这类原路径导入、``dev_tools/preview_ui.py``、
以及 ``python app.py serve|diagnose|click-once`` 与 ``mumu-panel.exe`` 打包入口
都保持不变。静态资源仍从 :data:`drift.core.paths.BUNDLE` 下的 ``web/`` 读取,
PyInstaller ``--add-data web;web`` 打包后依然找得到。

本层不认识 ``Clicker``:``app`` 参数按鸭子类型使用,``Clicker`` 仅在
:func:`drift.web.cli.main` 内部延迟导入,因此 ``drift.web`` 不反向依赖游戏任务层。
"""

from __future__ import annotations

from .cli import main  # noqa: F401
from .server import make_handler  # noqa: F401

__all__ = ["cli", "server", "make_handler", "main"]


