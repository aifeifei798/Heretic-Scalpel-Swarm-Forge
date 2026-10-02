"""配置真相源。

整个系统只有这一处定义配置。CLI 的 ``--config x.json``、未来 API 的 job payload、
前端的表单字段，全部由这里的 pydantic 模型派生并校验。

修复记录：
  * B13 ``len(macro_names) != num_macro_cores`` 时导出/雷达会静默错位 —— 现在直接报错。
  * A2  ``big_target`` 只有 0/1 却配 4 个大核 —— 大核目标改为由
    :class:`ArchProfile.domain_to_macro` 显式映射算出，取值范围 ``[0, M)``。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# 领域常量：32 个专家语义域，前 16 个为文科(Arts)，后 16 个为理科(Sci/Code/Math)
# ---------------------------------------------------------------------------
ARTS_DOMAINS: tuple[str, ...] = (
    "Arts_Poetry", "Arts_Fiction", "Arts_Prose", "Arts_Drama", "Arts_Philosophy",
    "Arts_History", "Arts_Culture", "Arts_Rhetoric", "Arts_Humor", "Arts_MusicArt",
    "Arts_Psychology", "Arts_Translation", "Arts_Mythology", "Arts_Critique",
    "Arts_Summary", "Arts_DailyChat",
)

SCI_DOMAINS: tuple[str, ...] = (
    "Code_Algo", "Code_DS", "Code_Python", "Code_Debug", "Code_Design",
    "Code_SQL", "Math_Arith", "Math_Algebra", "Math_Geo", "Math_Calculus",
    "Math_Prob", "Math_Logic", "Sci_Physics", "Sci_Chem", "Sci_Biology",
    "Sci_CS_Core",
)

ALL_DOMAINS: tuple[str, ...] = ARTS_DOMAINS + SCI_DOMAINS
DOMAIN_INDEX: dict[str, int] = {d: i for i, d in enumerate(ALL_DOMAINS)}


def default_domain_to_macro() -> dict[str, int]:
    """默认把 32 个领域均衡切成 4 个大核组（每组 8 个领域）。

    这是 UI「架构设计器」的初始值；用户可在前端拖拽改写。
    """
    n_macro = 4
    out: dict[str, int] = {}
    for i, d in enumerate(ALL_DOMAINS):
        out[d] = i % n_macro
    return out


# ---------------------------------------------------------------------------
# 1. 数据集生成配置
# ---------------------------------------------------------------------------
class PerturbConfig(BaseModel):
    """扰动引擎开关。全部关掉即退化为"原始种子直出"（用于回归对比）。"""

    model_config = ConfigDict(extra="forbid")

    polite_prefix: bool = Field(
        True, description="礼貌/口语化前缀变体：''/请/帮我/能否/麻烦")
    request_suffix: bool = Field(
        True, description="请求后缀变体：''/。/，谢谢。/，请给出完整答案。")
    code_fence: bool = Field(
        True, description="代码围栏风格变体：``` / ~~~ / 四空格缩进块")
    latex_delim: bool = Field(
        True, description="行内/行间数学分隔符变体：$x$ / $$x$$")
    punctuation_style: bool = Field(
        True, description="中英文标点风格变体（仅作用于代码块之外）")

    # 说明：数字/题面参数的多样性由模板 slot 取值提供（见 datagen/seeds.py），
    # 而非对成文做全局正则改写——后者会破坏 O(n^2)、年份、代码常量等内容。


class DataGenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_path: str = Field(
        "dual_contrast_data.jsonl", description="输出 jsonl 路径")
    stats_path: str | None = Field(
        None, description="统计报告路径；None 时由 output_path 推导")
    samples_per_domain: int = Field(
        200, ge=1, le=20000, description="每个领域的目标条数")
    val_ratio: float = Field(
        0.05, ge=0.0, lt=0.5, description="验证集比例（按领域分层切分）")
    seed: int = Field(20260929, description="全局随机种子，保证可复现")
    min_unique_ratio: float = Field(
        0.95, ge=0.0, le=1.0,
        description="唯一率下限；低于此值直接报错退出，防止脏数据流入训练")
    perturb: PerturbConfig = Field(default_factory=PerturbConfig)

    def resolved_stats_path(self) -> Path:
        if self.stats_path:
            return Path(self.stats_path)
        p = Path(self.output_path)
        return p.with_name(p.stem + "_stats.json")


# ---------------------------------------------------------------------------
# 2. 架构配置（大核 / 小核定义）
# ---------------------------------------------------------------------------
class ArchProfile(BaseModel):
    """大核 / 小核结构定义。

    语义：
      * 大核 #0 恒为只读底座，``[1, M)`` 为 Dense LoRA。
      * 小核为 N 个 rank 更低、注入强度更小的微专家，1:1 绑定一个领域。
      * ``domain_to_macro`` 是 UI 架构设计器产出的映射表，训练时据此算
        ``macro_target ∈ [0, M)``，**数据本身不再存 big_target**。
    """

    model_config = ConfigDict(extra="forbid")

    num_macro_cores: int = Field(4, ge=1, le=32, description="大核总数 M")
    macro_names: list[str] = Field(
        default_factory=lambda: ["Arts_Anchor", "Code_Math_Core",
                                 "Sci_Reason_Core", "Humanities_Core"])
    macro_rank: int = Field(64, ge=1, le=512)
    macro_dense_scale: float = Field(
        0.1, ge=0.0, description="大核 LoRA 残差注入强度")
    num_micro_experts: int = Field(32, ge=1, le=256, description="小核微专家总数 N")
    micro_rank: int = Field(16, ge=1, le=256)
    micro_scale: float = Field(
        0.02, ge=0.0, description="小核残差强度（防复读）")
    domain_to_macro: dict[str, int] = Field(
        default_factory=default_domain_to_macro,
        description="domain -> 大核 id 映射表（UI 架构设计器产出）")

    @model_validator(mode="after")
    def _check_consistency(self) -> "ArchProfile":
        # B13: 名字数量与核数不一致会让导出 config 与雷达显示静默错位
        if len(self.macro_names) != self.num_macro_cores:
            raise ValueError(
                f"macro_names 有 {len(self.macro_names)} 个名字，"
                f"但 num_macro_cores={self.num_macro_cores}；两者必须一致"
            )
        if len(set(self.macro_names)) != len(self.macro_names):
            raise ValueError("macro_names 存在重复项")

        unknown = set(self.domain_to_macro) - set(ALL_DOMAINS)
        if unknown:
            raise ValueError(f"domain_to_macro 含未知领域: {sorted(unknown)}")
        missing = set(ALL_DOMAINS) - set(self.domain_to_macro)
        if missing:
            raise ValueError(f"domain_to_macro 缺少领域: {sorted(missing)}")
        bad = {d: m for d, m in self.domain_to_macro.items()
               if not 0 <= m < self.num_macro_cores}
        if bad:
            raise ValueError(
                f"以下领域的 macro id 越界 [0,{self.num_macro_cores}): {bad}")
        return self

    def macro_name(self, macro_id: int) -> str:
        if 0 <= macro_id < len(self.macro_names):
            return self.macro_names[macro_id]
        return f"MacroCore_{macro_id}"


# ---------------------------------------------------------------------------
# 3. 训练配置
# ---------------------------------------------------------------------------
class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(4, ge=1, le=512)
    grad_accum: int = Field(4, ge=1, le=512)
    max_length: int = Field(
        256, ge=32, le=8192,
        description="截断长度。实测样本真实长度约 66-150 token，"
                    "默认值从 512 降到 256 以砍掉大量 padding 算力")
    dynamic_padding: bool = Field(
        True, description="动态 padding 到 batch 内最长，而非 pad 到 max_length")
    length_grouped_batches: bool = Field(
        True, description="长度分桶采样，减少 padding 浪费")

    lr_macro: float = Field(5e-5, gt=0)
    lr_router: float = Field(
        1e-3, gt=0,
        description="router 学习率。比 LoRA 大一个量级：router 参数量极小"
                    "（H×M + H×N），且要学的映射比 LoRA 简单得多")
    lr_micro: float = Field(1e-4, gt=0)
    weight_decay: float = Field(0.01, ge=0.0)
    max_grad_norm: float = Field(1.0, gt=0, description="梯度裁剪")
    warmup_ratio: float = Field(0.05, ge=0.0, lt=1.0)
    lr_schedule: Literal["cosine", "linear", "constant"] = "cosine"
    fused_optimizer: bool = Field(
        True, description="fused AdamW；CPU 上会自动回退")

    # --- 损失权重 -----------------------------------------------------
    loss_chunk_size: int = Field(
        2048, ge=128, le=65536,
        description="分块交叉熵的每块监督 token 数。峰值显存约 "
                    "chunk*vocab*4 字节；vocab=262144 时 2048 -> 2.1GiB。"
                    "调小可省显存，调大略快。")
    lm_loss_weight: float = Field(1.0, gt=0)
    router_aux_weight: float = Field(
        0.05, ge=0.0,
        description="路由监督 CE 权重（辅助信号，不做硬 GT 门控）。"
                    "注意这是**对所有层求和**后的系数，35 层模型的"
                    "实际强度是 0.05*35 = 1.75 倍每层。"
                    "太小则 router 学不动（实测 0.1 求平均时 macro_ce "
                    "60 步纹丝不动停在 ln(4)=1.386）")
    load_balance_weight: float = Field(
        0.05, ge=0.0,
        description="Switch 风格负载均衡辅助损失，同时作用于大核与小核，"
                    "同样是**对所有层求和**（35 层 -> 1.75 倍每层）。"
                    "均衡时每层恰为 2（大核 1 + 小核 1），完全坍缩时升到 "
                    "M+N=36，梯度比均衡时大 18 倍。"
                    "实测：0.01 -> 大核 Arts 掉到 0.19；0.05 -> 0.21，"
                    "而验证 loss 两者相同（2.455 / 2.445），所以取 0.05")

    # --- 评测 ---------------------------------------------------------
    eval_every: int = Field(
        25, ge=0, le=100000,
        description="每多少个 optimizer step 在验证集上跑一次评测。0 = 关闭。"
                    "训练 loss 噪声极大（B=4 时步间波动能到 2 倍），"
                    "只有 held-out loss 才能证明真的在学")
    eval_batches: int = Field(
        8, ge=1, le=1024,
        description="每次评测用多少个验证 batch")

    # --- 路由 ---------------------------------------------------------
    micro_top_k: int = Field(
        2, ge=1, le=32,
        description="小核 top-k 稀疏激活。训练与推理使用同一套路由逻辑，"
                    "消除原来'训练走 GT 门控 / 推理走单专家'的错配")
    micro_route_mode: Literal["learned_topk", "gt_gate"] = Field(
        "learned_topk",
        description="learned_topk=训练也走 router（推荐）；"
                    "gt_gate=退化为用标签硬选专家（仅用于消融对比）")

    # --- 步数 / 评估 --------------------------------------------------
    max_steps: int = Field(
        50, ge=1, description="优化器步数上限（原 golden_stop_step 的正确语义）")
    log_every: int = Field(1, ge=1)
    ckpt_every: int = Field(0, ge=0, description="0 表示不按步存 ckpt")
    eval_every: int = Field(0, ge=0, description="0 表示不跑中途评估")
    seed: int = Field(20260929)
    device: str = Field("auto", description="auto/cuda:0/cpu")
    param_dtype: Literal["fp32", "bf16", "fp16"] = Field(
        "fp32",
        description="★可训练参数(LoRA/router)的精度。必须是 fp32——"
                    "bf16 上直接做 AdamW 会大量舍入丢失更新")

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"


class RuntimeConfig(BaseModel):
    """运行时（推理）配置。"""

    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.18, ge=0.5, le=3.0)
    micro_scale: float | None = Field(
        None, ge=0.0, description="热调小核强度；None 表示用 ArchProfile 的值")
    macro_dense_scale: float | None = Field(
        None, ge=0.0, description="热调大核注入强度")
    micro_top_k: int | None = Field(
        None, ge=1, le=32, description="热调小核 top-k")


# ---------------------------------------------------------------------------
# 4. 顶层配置
# ---------------------------------------------------------------------------
class ForgeConfig(BaseModel):
    """一次「数据生成 → 训练 → 导出」全流程的完整快照。"""

    model_config = ConfigDict(extra="forbid")

    project_name: str = "Heretic-Scalpel-SwarmForge"
    base_model_id: str = "aifeifei798/Heretic-Scalpel-E2B"
    data_path: str = "data/dual_contrast_data.jsonl"
    export_dir: str = "outputs/export"
    checkpoint_path: str = "outputs/swarm_weights.pt"
    text_only: bool = Field(
        False,
        description="丢弃视觉/音频塔只留语言塔。纯文本任务可省约 0.9GB 显存")
    trust_remote_code: bool = True
    output_proj_dims: bool = Field(
        True, description="LoRA 打在 mlp 的投影层上（gate/up/down_proj）")

    arch: ArchProfile = Field(default_factory=ArchProfile)
    datagen: DataGenConfig = Field(default_factory=DataGenConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "ForgeConfig":
        p = Path(path)
        if not p.exists():
            cfg = cls()
            p.parent.mkdir(parents=True, exist_ok=True)
            cfg.save(p)
            return cfg
        with open(p, "r", encoding="utf-8") as f:
            return cls.model_validate(json.load(f))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2,
                      ensure_ascii=False)
        return p

    def summary(self) -> dict[str, Any]:
        a = self.arch
        return {
            "project_name": self.project_name,
            "base_model_id": self.base_model_id,
            "data_path": self.data_path,
            "macro": {
                "count": a.num_macro_cores,
                "rank": a.macro_rank,
                "names": list(a.macro_names),
            },
            "micro": {
                "count": a.num_micro_experts,
                "rank": a.micro_rank,
                "scale": a.micro_scale,
            },
            "train": {
                "max_steps": self.train.max_steps,
                "effective_batch": self.train.batch_size * self.train.grad_accum,
                "max_length": self.train.max_length,
            },
        }


DEFAULT_CONFIG_PATH = "forge_config.json"
