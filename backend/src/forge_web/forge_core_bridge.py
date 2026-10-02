"""``forge_web`` → ``forge_core`` 的唯一接缝。

刻意收窄：Web 层只通过这里列出的符号触达 ML 库，而不是到处
``from forge_core.datagen import ...``。好处有二：

1. 换掉/重命名 ``forge_core`` 内部结构时，改动集中在这一个文件；
2. 一眼能看出"Web 层到底能碰到 ML 库的哪些能力"——
   也就是需要做权限控制的那部分。

反向依赖是被禁止的：``forge_core`` 里出现 ``forge_web`` 即为架构回退。
:func:`assert_boundary` 会在启动时检查这一点。
"""

from __future__ import annotations

import os
from pathlib import Path


def sanitize_env() -> None:
    """清掉会让 ``huggingface_hub`` 直接崩溃的 socks 代理变量。

    本机设置了 ``ALL_PROXY=socks://127.0.0.1:10808``，
    ``huggingface_hub`` 解析不了 ``socks://`` scheme，
    任何一次模型加载都会抛
    ``ValueError: Unknown scheme for proxy URL``。

    与其让每个调用点各自 try/except，不如在子进程启动前统一清掉——
    否则这个坑会在"从 CLI 能跑、从 Web 一跑就炸"的地方反复出现。
    """
    for name in ("ALL_PROXY", "all_proxy"):
        val = os.environ.get(name, "")
        if val.startswith("socks://"):
            os.environ.pop(name, None)


def python_exe() -> str:
    """用与当前服务同一个解释器去起子进程。"""
    import sys
    return sys.executable


def backend_src() -> Path:
    """``forge_core`` 源码目录，用于拼 ``PYTHONPATH``。"""
    # forge_web/settings.py -> forge_web -> src
    return Path(__file__).resolve().parent.parent


def assert_boundary() -> None:
    """确认 ``forge_core`` 没有反向依赖 Web 层。"""
    core = backend_src() / "forge_core"
    offenders: list[str] = []
    for py in core.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import forge_web", "from forge_web")):
                offenders.append(f"{py.name}: {stripped}")
    if offenders:
        raise RuntimeError(
            "forge_core 反向依赖了 Web 层，破坏分层：\n  "
            + "\n  ".join(offenders))


from forge_core.schema import (  # noqa: E402
    ALL_DOMAINS as DOMAINS,
    ArchProfile,
    DataGenConfig,
    ForgeConfig,
    PerturbConfig,
    TrainConfig,
    default_domain_to_macro,
)
from forge_core.datagen import BuildReport, build_dataset  # noqa: E402
from forge_core.datagen.audit import norm_content  # noqa: E402
from forge_core.train import train  # noqa: E402

__all__ = [
    "DOMAINS", "default_domain_to_macro",
    "ArchProfile", "DataGenConfig", "ForgeConfig", "PerturbConfig",
    "TrainConfig",
    "build_dataset", "BuildReport", "norm_content", "train",
    "sanitize_env", "python_exe", "backend_src", "assert_boundary",
]
