"""Web 层设置。

.. warning::
   **默认只绑定回环地址，且不带鉴权。**
   这是有意为之：本服务能读取任意本地路径、启动训练子进程、
   写入文件系统，等同于本机的远程代码执行能力。
   把它暴露到 ``0.0.0.0`` 就等于把这些能力开放给整个网络。

   若确实需要对外提供服务，至少要做到：
   1. 反向代理终止 TLS + 强制鉴权；
   2. 把 ``FORGE_WEB_ALLOW_PATHS`` 收紧到具体数据目录，
      杜绝 ``..`` 之类的路径穿越。

   之所以不做"简单鉴权"，是因为一个只绑 127.0.0.1 的服务在同机场景下
   鉴权只是给人虚假的安全感；真正需要防护时请走上面的正规路径。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str = field(default_factory=lambda: os.environ.get(
        "FORGE_WEB_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(
        os.environ.get("FORGE_WEB_PORT", "8848")))

    project_root: Path = field(default_factory=lambda: Path(
        os.environ.get("FORGE_WEB_ROOT", Path.cwd())).resolve())

    db_path: Path = field(default_factory=lambda: Path(
        os.environ.get("FORGE_WEB_DB", "data/forge_web.db")))

    #: 允许前端读写的目录白名单。空 = 只允许 project_root。
    allow_paths: tuple[Path, ...] = field(default_factory=tuple)

    #: 子进程超时（秒）。超时后整个进程组被杀掉。
    run_timeout_sec: int = field(default_factory=lambda: int(
        os.environ.get("FORGE_WEB_RUN_TIMEOUT", str(6 * 3600))))

    #: 同时允许多少个训练子进程。GPU 只有一块，超了就是 OOM 打架。
    max_concurrent_runs: int = field(default_factory=lambda: int(
        os.environ.get("FORGE_WEB_MAX_RUNS", "1")))

    @property
    def is_loopback(self) -> bool:
        return self.host in {"127.0.0.1", "localhost", "::1"}

    def resolved_allow(self) -> tuple[Path, ...]:
        if self.allow_paths:
            return tuple(p.resolve() for p in self.allow_paths)
        return (self.project_root,)


def get_settings() -> Settings:
    s = Settings()
    return s


__all__ = ["Settings", "get_settings"]
