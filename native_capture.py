"""兼容外壳:实现已迁到 :mod:`drift.device.native`。

本模块通过 :func:`drift.compat.module_proxy` 把属性读写/删除转发到包内实现,
因此 ``from native_capture import MuMuCapture`` 与 ``patch('native_capture.X')``
都仍然命中同一份真实现。
"""

from drift.compat import module_proxy

module_proxy(__name__, "drift.device.native")
