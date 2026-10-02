#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heretic-Scalpel Swarm Forge —— 兼容层（已废弃）。

真正的实现在 :mod:`forge_core`（``backend/src/forge_core``）。
本文件仅把旧命令行转发过去，让原有脚本与文档继续可用。

为什么留一个 shim 而不是直接删掉
--------------------------------
旧版把 887 行的训练/建模/导出/发布逻辑全都塞在这一个文件里，
很多 README、脚本和 ``forge_config.json`` 都按 ``python scalpel_forge.py
<cmd>`` 的方式调用它。直接删掉会让这些入口全部失效。

旧入口 → 新入口
---------------
=================  ===========================================
旧                 新
=================  ===========================================
``train``          ``python -m forge_core.cli train``
``chat``           移除了；改用发布包 + ``model.generate``
``export``         ``python -m forge_core.cli export``
``publish``        ``python -m forge_core.cli publish``
（无参数向导）      ``python -m forge_core.cli plan``
=================  ===========================================

原实现已备份至 ``/tmp`` 之外的版本库；这里**不再保留任何训练逻辑**，
以免出现"两套实现悄悄分叉"——那正是旧导出用
``src.split('# ---')[2]`` 切自己源码时埋下的隐患。

.. warning::
   ``build_pristine_dataset.py`` 同样已退役：它生成的 6400 条数据里
   只有 **41 条唯一内容**（156x 重复）。请改用
   ``python -m forge_core.cli datagen``，其内容唯一率经审计为 98.2%。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 让 shim 在没有安装包的情况下也能找到 backend/src
_SRC = Path(__file__).resolve().parent / "backend" / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# 本机环境的 socks 代理会让 huggingface_hub 抛
# "ValueError: Unknown scheme for proxy URL"。模型必须离线加载时，
# 直接在这里清掉，免得每个入口各写一遍。
for _v in ("ALL_PROXY", "all_proxy"):
    if os.environ.get(_v, "").startswith("socks://"):
        os.environ.pop(_v, None)


_ALIASES = {
    "train": "train",
    "export": "export",
    "publish": "publish",
    "datagen": "datagen",
    "audit": "audit",
    "inspect": "inspect",
    "plan": "plan",
    "smoke": "smoke",
}

_GONE = {
    "chat": ("`chat` 的交互控制台已移除。\n"
             "       新流程：训练 -> export -> 在发布包上调用 "
             "`model.generate()`。\n"
             "       热调推理超参见 `SwarmWrapper.set_scales()`。"),
    "menu": "无参数向导已移除，请直接用子命令（见本文件 docstring）。",
}


def main() -> int:
    argv = sys.argv[1:]

    if not argv or argv[0] in ("menu", "-h", "--help"):
        print(__doc__)
        print("可用子命令:", ", ".join(sorted(_ALIASES)))
        return 0

    cmd = argv[0].lower()
    if cmd in _GONE:
        print(_GONE[cmd], file=sys.stderr)
        return 2
    if cmd not in _ALIASES:
        print(f"未知命令 {cmd!r}。可用: {', '.join(sorted(_ALIASES))}",
              file=sys.stderr)
        return 2

    from forge_core.cli import main as cli_main

    sys.argv = ["forge", _ALIASES[cmd], *argv[1:]]
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
