"""SQLite 持久化。

用 SQLModel（SQLAlchemy + pydantic），好处是表结构直接由 pydantic 模型
推导，前后端共用同一份字段定义，不会漂移。

刻意**不引入异步 ORM**：这个后端的负载完全由训练子进程决定，
SQLite 的写入量微不足道，而 ``aiosqlite`` 会让事务与子进程生命周期的
交互更难推理。同步 SessionFactory + FastAPI 的线程池足够。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field
from sqlmodel import JSON, Column, Field as SQLField, Session, SQLModel, create_engine


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunState(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


# ---------------------------------------------------------------------------
# 表
# ---------------------------------------------------------------------------
class Dataset(SQLModel, table=True):
    """一份生成好的数据集。"""

    __tablename__ = "datasets"

    id: int | None = SQLField(default=None, primary_key=True)
    name: str = SQLField(index=True)
    path: str
    rows: int = 0
    val_ratio: float = 0.05
    per_domain: int = 200
    domains: list[str] = SQLField(default_factory=list, sa_column=Column(JSON))
    #: datagen 审计结果：{"checks": [{"name","ok","detail"}], "failed": n}
    audit: dict[str, Any] = SQLField(default_factory=dict, sa_column=Column(JSON))
    content_unique_ratio: float = 0.0
    created_at: str = SQLField(default_factory=_utcnow)


class Architecture(SQLModel, table=True):
    """大核 / 小核 / 领域映射的设计稿。"""

    __tablename__ = "architectures"

    id: int | None = SQLField(default=None, primary_key=True)
    name: str = SQLField(index=True)
    num_macro_cores: int
    macro_names: list[str] = SQLField(default_factory=list, sa_column=Column(JSON))
    macro_rank: int = 64
    num_micro_experts: int = 32
    micro_rank: int = 16
    #: domain -> macro index。前端拖拽产出的就是这个。
    domain_to_macro: dict[str, int] = SQLField(
        default_factory=dict, sa_column=Column(JSON))
    created_at: str = SQLField(default_factory=_utcnow)

    def validate_map(self) -> None:
        """领域映射必须覆盖已知领域，且取值落在 [0, M)。"""
        from .forge_core_bridge import DOMAINS

        if self.num_macro_cores < 1:
            raise ValueError("大核数必须 >= 1（第 0 号恒为只读底座）")
        bad = {d: m for d, m in self.domain_to_macro.items()
               if not 0 <= m < self.num_macro_cores}
        if bad:
            raise ValueError(
                f"领域映射越界（必须在 [0, {self.num_macro_cores})）：{bad}")
        missing = [d for d in DOMAINS if d not in self.domain_to_macro]
        if missing:
            raise ValueError(f"有 {len(missing)} 个领域未指派大核：{missing[:5]}…")


class Run(SQLModel, table=True):
    """一次训练子进程。"""

    __tablename__ = "runs"

    id: int | None = SQLField(default=None, primary_key=True)
    architecture_id: int | None = SQLField(default=None, foreign_key="architectures.id")
    dataset_id: int | None = SQLField(default=None, foreign_key="datasets.id")

    state: RunState = SQLField(default=RunState.queued, index=True)
    #: 训练超参快照，保证事后能复现这次 run 到底跑了什么
    config: dict[str, Any] = SQLField(default_factory=dict, sa_column=Column(JSON))

    steps_done: int = 0
    max_steps: int = 0
    best_val_loss: float | None = None
    last_lm_loss: float | None = None
    macro_dead: int | None = None
    micro_dead: int | None = None
    macro_dist: dict[str, Any] = SQLField(default_factory=dict, sa_column=Column(JSON))

    log_path: str = ""
    error: str | None = None
    created_at: str = SQLField(default_factory=_utcnow)
    finished_at: str | None = None


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------
_engines: dict[str, Any] = {}


def engine_for(db_path: str | Path):
    """按**路径**缓存引擎。

    不能用单个模块级单例：那样一旦 ``FORGE_WEB_DB`` 指向新的位置
    （换了数据目录、或像测试那样换临时目录），旧的 engine 仍会被复用，
    于是所有写入都落到已经不存在的文件上，报
    ``attempt to write a readonly database``。
    """
    key = str(Path(db_path).resolve())
    if key in _engines:
        return _engines[key]
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False：FastAPI 的同步端点跑在线程池里，
    # Session 会在不同线程间创建/关闭。
    eng = create_engine(
        f"sqlite:///{p}", connect_args={"check_same_thread": False},
        echo=False)
    SQLModel.metadata.create_all(eng)
    _engines[key] = eng
    return eng


def session(db_path: str | Path) -> Session:
    return Session(engine_for(db_path))


__all__ = ["Dataset", "Architecture", "Run", "RunState",
           "engine_for", "session", "_utcnow"]
