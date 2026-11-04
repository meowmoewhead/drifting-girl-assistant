"""boss_defense 的兼容外壳:实现在 :mod:`drift.tasks.boss_defense`。

保留本模块是为了让原有导入路径与 ``patch('boss_defense.<名字>')`` 语义继续有效 ——
代理会把属性读写与删除都转发到包内真实现。
"""

from drift.compat import module_proxy

module_proxy(__name__, "drift.tasks.boss_defense")
