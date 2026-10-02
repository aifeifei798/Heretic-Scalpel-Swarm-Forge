"""导出层：把训练产物打成可直接 ``from_pretrained`` 的发布包。

模块布局刻意做成"每个文件都能独立拷贝"：

* ``swarm_forge.py`` —— 训练侧 :mod:`forge_core.modeling.swarm` 的**逐字节副本**；
* ``telemetry.py`` / ``configuration_scalpel.py`` / ``modeling_scalpel.py``
  —— 零相对导入的运行时模块。

打包靠 ``shutil.copy``，不靠切源码；等价性由 ``selftest`` 断言。
"""

from .bundle import build_bundle
from .publish import publish

__all__ = ["build_bundle", "publish"]
