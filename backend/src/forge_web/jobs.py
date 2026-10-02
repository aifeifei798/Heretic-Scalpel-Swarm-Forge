"""把 Web 请求翻译成 ``forge_core`` 的调用。

这一层刻意很薄——它只负责**组装参数**，不含任何算法。
算法全在 ``forge_core`` 里，所以 CLI 与 Web 走的是同一条路径，
不会出现"网页能训、命令行训不出来"的差异。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .forge_core_bridge import (
    ArchProfile,
    DataGenConfig,
    ForgeConfig,
    PerturbConfig,
    TrainConfig,
    default_domain_to_macro,
)


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------
class DatasetSpec(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    per_domain: int = Field(default=200, ge=1, le=5000)
    val_ratio: float = Field(default=0.05, ge=0.0, lt=0.5)
    seed: int = Field(default=20260929, ge=0)
    domains: list[str] = Field(default_factory=list)
    out_name: str = Field(default="", max_length=200)

    @field_validator("domains")
    @classmethod
    def _known(cls, v: list[str]) -> list[str]:
        from .forge_core_bridge import DOMAINS

        bad = [d for d in v if d not in DOMAINS]
        if bad:
            raise ValueError(f"未知领域：{bad}；可选 {len(DOMAINS)} 个")
        return v

    @property
    def filename(self) -> str:
        if self.out_name:
            return self.out_name if self.out_name.endswith(".jsonl") \
                else f"{self.out_name}.jsonl"
        return f"ds-{self.name}.jsonl"


class ArchSpec(BaseModel):
    """大核 / 小核 / 领域映射。"""

    name: str = Field(default="default", max_length=120)
    num_macro_cores: int = Field(default=4, ge=1, le=16)
    macro_names: list[str] = Field(default_factory=list)
    macro_rank: int = Field(default=64, ge=1, le=512)
    num_micro_experts: int = Field(default=32, ge=1, le=512)
    micro_rank: int = Field(default=16, ge=1, le=512)
    #: 领域 -> 大核下标。留空则用内置均衡映射。
    domain_to_macro: dict[str, int] = Field(default_factory=dict)

    @field_validator("domain_to_macro")
    @classmethod
    def _known_domains(cls, v: dict[str, int]) -> dict[str, int]:
        from .forge_core_bridge import DOMAINS

        bad = [d for d in v if d not in DOMAINS]
        if bad:
            raise ValueError(f"未知领域：{bad}")
        return v

    @model_validator(mode="after")
    def _map_in_range(self) -> "ArchSpec":
        """领域映射必须落在 ``[0, num_macro_cores)``。

        必须用 ``model_validator`` 而不是 ``field_validator``：
        后者看不到同一模型里的其它字段，只能瞎猜一个上界
        （之前硬编码 16，于是 ``num_macro_cores=2`` 配 ``->7``
        也能通过校验，直到更下游才炸）。
        """
        for d, m in self.domain_to_macro.items():
            if not 0 <= m < self.num_macro_cores:
                raise ValueError(
                    f"领域 {d} 映射到大核 {m}，超出 "
                    f"[0, {self.num_macro_cores})")
        if self.macro_names and len(self.macro_names) != self.num_macro_cores:
            raise ValueError(
                f"macro_names 有 {len(self.macro_names)} 个，"
                f"但 num_macro_cores={self.num_macro_cores}")
        return self

    def resolved_names(self) -> list[str]:
        if self.macro_names:
            if len(self.macro_names) != self.num_macro_cores:
                raise ValueError(
                    f"macro_names 有 {len(self.macro_names)} 个，"
                    f"但 num_macro_cores={self.num_macro_cores}")
            return list(self.macro_names)
        return [f"M{i}" for i in range(self.num_macro_cores)]

    def resolved_map(self) -> dict[str, int]:
        """补全领域映射，缺省走内置的均衡分配。

        内置映射保证 32 个领域均分到各大核（每核 8 个）。若用户只
        拖动了其中几个领域，剩下的补齐时保持均衡——**不能**默认全填 0，
        否则大核 1..M-1 会一个领域都没有、直接死掉。

        越界值在这里已经被 :meth:`_map_in_range` 拒绝，因此不需要
        （也不应该）再偷偷夹取——悄悄修正会让用户以为自己配对了。
        """
        from .forge_core_bridge import DOMAINS

        full = dict(self.domain_to_macro)
        default = default_domain_to_macro()
        for d in DOMAINS:
            full.setdefault(d, default.get(d, 0) % self.num_macro_cores)
        return full

    def to_profile(self) -> ArchProfile:
        return ArchProfile(
            num_macro_cores=self.num_macro_cores,
            macro_names=self.resolved_names(),
            macro_rank=self.macro_rank,
            num_micro_experts=self.num_micro_experts,
            micro_rank=self.micro_rank,
            domain_to_macro=self.resolved_map(),
        )


class TrainSpec(BaseModel):
    architecture: ArchSpec
    dataset_path: str
    project_name: str = Field(default="Heretic-Scalpel-SwarmForge",
                              max_length=120)
    base_model_id: str = "aifeifei798/Heretic-Scalpel-E2B"

    max_steps: int = Field(default=200, ge=1, le=1_000_000)
    batch_size: int = Field(default=4, ge=1, le=512)
    grad_accum: int = Field(default=4, ge=1, le=512)
    max_length: int = Field(default=256, ge=32, le=8192)
    loss_chunk_size: int = Field(default=2048, ge=128, le=65536)

    lr: float = Field(default=5e-5, gt=0, le=1.0)
    lr_router: float = Field(default=1e-3, gt=0, le=1.0)
    eval_every: int = Field(default=25, ge=0)
    eval_batches: int = Field(default=8, ge=1, le=1024)

    micro_top_k: int = Field(default=2, ge=1, le=64)
    router_aux_weight: float = Field(default=0.05, ge=0.0)
    load_balance_weight: float = Field(default=0.05, ge=0.0)

    device: str = Field(default="auto", max_length=40)
    text_only: bool = True
    seed: int = Field(default=20260929, ge=0)
    export_dir: str = Field(default="", max_length=400)

    def to_forge_config(self) -> ForgeConfig:
        arch = self.architecture.to_profile()
        t = TrainConfig(
            batch_size=self.batch_size,
            grad_accum=self.grad_accum,
            max_length=self.max_length,
            loss_chunk_size=self.loss_chunk_size,
            lr_macro=self.lr,
            lr_micro=self.lr,
            lr_router=self.lr_router,
            router_aux_weight=self.router_aux_weight,
            load_balance_weight=self.load_balance_weight,
            micro_top_k=self.micro_top_k,
            max_steps=self.max_steps,
            eval_every=self.eval_every,
            eval_batches=self.eval_batches,
            device=self.device,
            seed=self.seed,
            param_dtype="bf16",
        )
        return ForgeConfig(
            project_name=self.project_name,
            base_model_id=self.base_model_id,
            data_path=self.dataset_path,
            arch=arch,
            train=t,
            text_only=self.text_only,
        )

    def to_cli_argv(self, config_path: Path, checkpoint: Path) -> list[str]:
        """转成 ``forge_core.cli train`` 的参数。

        走 CLI 而不是直接调 ``train()`` 的理由：监管器需要在**独立进程**里
        跑（见 :mod:`forge_web.runner`），而命令行是跨进程最稳定的接口，
        顺便也让"网页点的训练"和"手敲的训练"字面上完全一致。
        """
        argv = ["-m", "forge_core.cli", "train",
                "--config", str(config_path),
                "--data", self.dataset_path,
                "--max-steps", str(self.max_steps),
                "--batch-size", str(self.batch_size),
                "--grad-accum", str(self.grad_accum),
                "--max-length", str(self.max_length),
                "--loss-chunk", str(self.loss_chunk_size),
                "--device", self.device,
                "--eval-every", str(self.eval_every),
                "--checkpoint", str(checkpoint)]
        if self.export_dir:
            argv += ["--export", self.export_dir]
        return argv


class SwarmChatSpec(BaseModel):
    """一次试跑请求。"""

    checkpoint: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=8000)
    system: str | None = Field(default=None, max_length=8000)
    history: list[dict[str, str]] = Field(default_factory=list,
                                          max_length=40)
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    seed: int = Field(default=0, ge=0)
    top_k: int | None = Field(default=None, ge=1, le=512)
    compare: bool = False
    device: str = Field(default="cuda:0", max_length=40)
    timeout_sec: int = Field(default=600, ge=30, le=3600,
                             description="含底座加载时间。Gemma4 加载约 20~40s")


def datagen_config(spec: DatasetSpec, out_path: str) -> DataGenConfig:
    return DataGenConfig(
        samples_per_domain=spec.per_domain,
        val_ratio=spec.val_ratio,
        seed=spec.seed,
        perturb=PerturbConfig(),
        output_path=out_path,
        min_unique_ratio=0.95,
    )


def write_forge_config(cfg: ForgeConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cfg.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8")
    return path


__all__ = ["DatasetSpec", "ArchSpec", "TrainSpec", "SwarmChatSpec",
           "datagen_config", "write_forge_config"]
