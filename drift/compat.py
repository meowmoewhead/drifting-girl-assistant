"""兼容层:让实现搬进 drift 包后,原有根模块路径与 unittest.mock 补丁语义保持不变。

背景
----
45 个测试文件里有 170+ 处 `patch('equipment_upgrade.time')` / `patch('app.ParallelWorld')`
这类**按模块路径替换模块级属性**的补丁。测试各自 `import time` 并读自己的模块全局,
所以仅仅在根模块里 `from drift.x import Y` 是不够的:补丁会落在一个没人用的空壳上。

做法
----
根模块用本代理取代自己:把属性读写与删除**转发到包内真实现模块**,
于是 `patch('根模块.time')` 与 `patch('drift.x.time')` 命中的是同一个模块字典;
同时把写入同步进壳自己的 `__dict__`,这样:

- `from 根模块 import 类名` 照旧可用(绑定发生在导入语句求值时);
- `patch('根模块.属性')` 之后,**从根模块读回**该属性也能看到补丁值。

用法(根模块整体内容)::

    from drift.compat import module_proxy

    module_proxy(__name__, "drift.tasks.equipment_upgrade")
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


class _ModuleProxy(ModuleType):
    """把属性操作转发到目标模块的模块类型。"""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(_module_proxy_target(self), name)

    def __setattr__(self, name, value):
        setattr(_module_proxy_target(self), name, value)
        # 同步进壳自身,保证从根模块读回时能看到补丁后的值。
        ModuleType.__setattr__(self, name, value)

    def __delattr__(self, name):
        target = _module_proxy_target(self)
        if hasattr(target, name):
            delattr(target, name)
        if name in self.__dict__:
            ModuleType.__delattr__(self, name)


_PROXY_TARGET = "_module_proxy_target_"


def _module_proxy_target(module):
    target = module.__dict__.get(_PROXY_TARGET)
    if target is None:
        raise AttributeError("兼容代理尚未绑定目标模块")
    return target


def module_proxy(name, target_path):
    """把 sys.modules[name] 变成 target_path 的转发代理。

    必须在根模块被导入时调用一次;重复调用是幂等的。
    """
    current = sys.modules.get(name)
    if current is None:
        raise RuntimeError(f"模块 {name} 尚未导入,无法建立代理")
    if isinstance(current, _ModuleProxy):
        return current
    target = importlib.import_module(target_path)
    namespace = current.__dict__
    namespace[_PROXY_TARGET] = target
    # 公开名先绑定一次,满足 `from 根模块 import X`;之后的读写都走代理。
    namespace.update({key: value for key, value in vars(target).items() if not key.startswith("__")})
    current.__class__ = _ModuleProxy
    return current
