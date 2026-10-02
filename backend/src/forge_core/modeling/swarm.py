"""模型装配：LoRA 模块、蜂群包装器、层定位。

修复记录
--------
* **B6 大核聚合的 ``[B,T,M,H]`` 物化**：原实现 ``torch.stack(macro_outs, dim=-2)``
  在 B=4/T=512/M=4/H=1536 的 bf16 下就是 24MB/层、x35 层 = 0.82GB 激活
  （还要被 gradient checkpointing 重算一遍），M=16 时涨到 3.28GB。
  现在改为 ``einsum``，中间结果只有 ``[B,T,H]``。
* **B2 小核路由训推错配**：原实现训练时用 ``micro_pool[GT标签]`` 硬门控
  （router 只从 aux CE 拿梯度，LM loss 完全不经过路由），
  推理时却用 ``chosen[0,-1]``（batch 首样本的最后一个 token）
  选**唯一一个**专家作用于全 batch。改用统一的 top-k learned 路由。
* **B2 专家坍缩**：新增 Switch 风格负载均衡辅助损失。
* **B7 bf16 训练**：LoRA/router 参数建在 fp32，前向用 autocast。
  bf16 上直接做 AdamW 会有大量更新被舍入丢弃。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class ScalpelLoRA(nn.Module):
    """低秩适配。``lora_B`` 零初始化，因此初始时严格等于恒等映射。"""

    def __init__(self, hidden_dim: int, rank: int, *,
                 alpha: float | None = None, dtype: torch.dtype = torch.float32,
                 device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.rank = rank
        self.scaling = (alpha if alpha is not None else rank) / rank
        self.lora_A = nn.Linear(hidden_dim, rank, bias=False,
                                dtype=dtype, device=device)
        self.lora_B = nn.Linear(rank, hidden_dim, bias=False,
                                dtype=dtype, device=device)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling


# ---------------------------------------------------------------------------
# 顶层无操作模块（用于汇总 router 统计）
# ---------------------------------------------------------------------------
class _Null(nn.Module):
    def forward(self, x):  # pragma: no cover
        return x


# ---------------------------------------------------------------------------
# 蜂群包装器
# ---------------------------------------------------------------------------
class SwarmWrapper(nn.Module):
    """把任意 MLP 包装成"大核汇聚 + 小核稀疏专家"的 MoE 层。

    语义
    ----
    * **大核 #0** 是只读底座（``base_core``），``[1, M)`` 是 Dense LoRA。
      所有大核输出写成 ``base_out + scale * delta_i``，因此汇聚结果为
      ``base_out + sum_i w_i * delta_i``——底座永远有一个单位权重，
      这保证了路由即使未训练也不会破坏底座行为。
    * **小核**是 N 个 rank 更低、注入强度更小的微专家，top-k 稀疏激活。

    与原实现的关键差异：训练与推理走**同一套** top-k 路由逻辑，
    不再出现"训练用 GT 标签硬门控、推理用单专家"的错配。
    """

    def __init__(self, base_core: nn.Module, *, hidden_dim: int,
                 num_macro: int = 4, macro_rank: int = 64,
                 num_micro: int = 32, micro_rank: int = 16,
                 micro_scale: float = 0.02, macro_scale: float = 0.1,
                 micro_top_k: int = 2,
                 param_dtype: torch.dtype = torch.float32,
                 device: str | torch.device = "cpu") -> None:
        super().__init__()
        if num_macro < 1:
            raise ValueError("num_macro 至少为 1（第 0 号恒为只读底座）")
        if not 1 <= micro_top_k <= num_micro:
            raise ValueError(
                f"micro_top_k={micro_top_k} 超出范围 [1, {num_micro}]")

        self.base_core = base_core
        self.num_macro = num_macro
        self.num_micro = num_micro
        self.micro_top_k = micro_top_k
        self.micro_scale = micro_scale
        self.macro_scale = macro_scale
        self.param_dtype = param_dtype

        # 大核 LoRA（可训练，fp32）
        self.macro_cores = nn.ModuleList([
            ScalpelLoRA(hidden_dim, macro_rank, alpha=macro_rank * 2,
                        dtype=param_dtype, device=device)
            for _ in range(num_macro - 1)
        ])

        # 路由器（可训练，fp32）
        self.router_macro = nn.Linear(hidden_dim, num_macro, bias=False,
                                      dtype=param_dtype, device=device)
        self.router_micro = nn.Linear(hidden_dim, num_micro, bias=False,
                                      dtype=param_dtype, device=device)
        nn.init.normal_(self.router_macro.weight, std=0.02)
        nn.init.normal_(self.router_micro.weight, std=0.02)

        # 小核池（可训练，fp32）
        self.micro_pool = nn.ModuleList([
            ScalpelLoRA(hidden_dim, micro_rank, alpha=micro_rank,
                        dtype=param_dtype, device=device)
            for _ in range(num_micro)
        ])

        # 统计：供训练循环读取
        self.last_macro_logits: torch.Tensor | None = None
        self.last_micro_logits: torch.Tensor | None = None
        self.last_macro_probs: torch.Tensor | None = None
        self.last_micro_topk: torch.Tensor | None = None
        self.last_micro_probs: torch.Tensor | None = None

    # -- 参数分组 ---------------------------------------------------
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        macro = [p for m in self.macro_cores for p in m.parameters()]
        micro = [p for m in self.micro_pool for p in m.parameters()]
        router = list(self.router_macro.parameters()) + \
            list(self.router_micro.parameters())
        return {"macro": macro, "micro": micro, "router": router}

    def set_scales(self, *, micro_scale: float | None = None,
                   macro_scale: float | None = None,
                   micro_top_k: int | None = None) -> None:
        """热调推理超参（Playground 面板会用到，无需重启模型）。"""
        if micro_scale is not None:
            self.micro_scale = float(micro_scale)
        if macro_scale is not None:
            self.macro_scale = float(macro_scale)
        if micro_top_k is not None:
            if not 1 <= micro_top_k <= self.num_micro:
                raise ValueError(
                    f"micro_top_k={micro_top_k} 超出范围 [1, {self.num_micro}]")
            self.micro_top_k = int(micro_top_k)

    # -- 路由 -------------------------------------------------------
    def _set_target(self, macro: torch.Tensor, micro: torch.Tensor) -> None:
        """由训练循环注入监督目标（仅用于 aux loss，不再参与前向门控）。"""
        self._macro_target = macro
        self._micro_target = micro

    def _topk_micro(self, logits: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """稀疏 top-k 路由。返回 ``(topk_idx, topk_probs)``，形状均为 ``[B, T, k]``。

        **不要缓存**：缓存只能按 batch size 建索引，而 logits 每次前向都在变，
        缓存会返回上一次的路由结果——这正是原实现"推理时用首样本末位 token
        选专家"那类 bug 的同源问题。topk 的开销远小于它带来的错误。

        概率用**在 top-k 子集上做 softmax**得到，而不是除以 top-k 之和：
        后者在 logits 全为负时会除以负数，产出巨大的负概率
        （实测可到 -5e7，直接把训练打飞）。
        """
        k = self.micro_top_k
        logits32 = logits.to(torch.float32)
        vals, idx = torch.topk(logits32, k, dim=-1)
        probs = torch.softmax(vals, dim=-1)
        return idx, probs

    def micro_output(self, x: torch.Tensor, idx: torch.Tensor,
                     probs: torch.Tensor) -> torch.Tensor:
        """按 top-k 索引加权求和小核输出。

        **不做 Python 循环、不调用 ``.item()``**——原实现对 batch 逐样本取
        ``.item()``，35 层 x batch 4 = 每步 140 次 GPU 同步，
        是训练速度的主要瓶颈。

        参数
        ----
        x: ``[B, T, H]``；idx / probs: ``[B, T, k]``（**每个 token 各自的 top-k**）。

        实现：把所有小核的低秩因子堆成 ``[N, r, H]`` / ``[N, H, r]``，
        把 ``B*T`` 视为展平后的 batch，用 ``index_select`` 一次取出
        ``[B*T*k, r, H]``，再两次 einsum 汇总。
        额外显存与 ``k`` 成正比（k=2 时约为展平 batch 的 2 倍），与 N 无关。
        """
        B, T, H = x.shape
        k = idx.shape[-1]

        As = torch.stack([m.lora_A.weight for m in self.micro_pool], dim=0)
        Bs = torch.stack([m.lora_B.weight for m in self.micro_pool], dim=0)
        scaling = torch.stack([
            torch.as_tensor(m.scaling, device=As.device, dtype=As.dtype)
            for m in self.micro_pool])
        dtype = As.dtype
        As, Bs, scaling = As.to(dtype), Bs.to(dtype), scaling.to(dtype)
        r = As.shape[1]

        flat = idx.reshape(-1)                       # [B*T*k]
        Ah = As.index_select(0, flat)                # [B*T*k, r, H]
        Bh = (Bs.index_select(0, flat)
              * scaling.index_select(0, flat).view(-1, 1, 1))

        # 展平成 N = B*T，按 token 位置广播 k 个专家
        xf = x.reshape(B * T, H)                     # [B*T, H]
        # [B*T, k, r] = [B*T, 1, H] x [B*T, k, r, H]
        h = torch.einsum("nH,nkrH->nkr", xf, Ah.view(B * T, k, r, H))
        # [B*T, k, H]
        o = torch.einsum("nkr,nkHr->nkH", h, Bh.view(B * T, k, H, r))
        # 按 top-k 概率加权求和 -> [B*T, H]
        w = probs.reshape(B * T, k).to(o.dtype).unsqueeze(-1)   # [B*T, k, 1]
        return (o * w).sum(dim=1).reshape(B, T, H)

    # -- 前向 -------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 底座是 bf16（省显存），LoRA/router 是 fp32（B7）。
        # autocast 只在 CUDA 上开启，CPU 训练没有 autocast，
        # 因此必须**显式对齐 dtype**：进包装器前把 x 提到 fp32，
        # 出去前再还原成调用方的 dtype（底座后续的 per_layer_input_gate
        # 等模块是 bf16，直接返回 fp32 会让它们报 mat1/mat2 不匹配）。
        in_dtype = x.dtype
        xd = x.to(self.param_dtype)
        base_out = self.base_core(x).to(self.param_dtype)

        # --- 大核：底座 + Dense LoRA 残差 ---
        macro_logits = self.router_macro(xd)
        macro_w = torch.softmax(macro_logits.to(torch.float32), dim=-1)
        self.last_macro_logits = macro_logits
        self.last_macro_probs = macro_w

        if self.num_macro == 1:
            out = base_out
        else:
            # 逐个大核 LoRA 累加，**不 stack 出 [B,T,M,H]**（B6）：
            # M=4/T=512/H=1536 的 bf16 张量是 24MB/层、x35 层 = 0.82GB
            # 激活（还要被 gradient checkpointing 重算一遍），M=16 时 3.28GB。
            # 这里只有一份 [B,T,H] 的临时量，峰值与 M 无关。
            #
            # 权重取 router 的 [1, M)：0 号恒为只读底座，其权重被丢弃
            # （等价于把底座概率重归一化），保证底座恒有一个有效通道。
            w = macro_w[:, :, 1:].to(self.param_dtype)
            acc = torch.zeros_like(base_out)
            for i, core in enumerate(self.macro_cores):
                wi = w[..., i].unsqueeze(-1)
                acc = acc + wi * (core(xd) * self.macro_scale)
            out = base_out + acc

        # --- 小核：top-k 稀疏 ---
        micro_logits = self.router_micro(xd)
        self.last_micro_logits = micro_logits
        topk_idx, topk_probs = self._topk_micro(micro_logits)
        self.last_micro_topk = topk_idx
        self.last_micro_probs = topk_probs

        micro_out = self.micro_output(xd, topk_idx, topk_probs)
        return (out + self.micro_scale * micro_out).to(in_dtype)

    # -- 辅助损失 ---------------------------------------------------
    def router_losses(self, mask: torch.Tensor,
                      macro_target: torch.Tensor,
                      micro_target: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (macro_ce, micro_ce, load_balance)。

        * CE：把每个 token 的 router logits 监督到该样本的领域标签。
          仅作为**辅助**信号（不再做硬门控），因此 LM loss 也会经由
          路由影响小核输出。
        * load_balance：Switch 风格，惩罚 top-k 概率的 batch 内不均衡，
          防止 32 个专家坍缩到少数几个。
        """
        if self.last_macro_logits is None:
            raise RuntimeError("router_losses 必须在 forward 之后调用")

        # 每个 token 都参与路由监督（**不只回答段**）：
        # 路由本就该同时依赖题面与回答，这样推理时整条序列的路由都有一致依据。
        # ``mask`` 是 [B, T] 的 bool，按行优先展开后与 repeat_interleave 对齐。
        B, T = mask.shape
        counts = mask.sum(-1)
        per_tok_macro = macro_target.repeat_interleave(counts)   # [n_true]
        per_tok_micro = micro_target.repeat_interleave(counts)
        macro_ce = F.cross_entropy(
            self.last_macro_logits[mask].float(), per_tok_macro)
        micro_ce = F.cross_entropy(
            self.last_micro_logits[mask].float(), per_tok_micro)

        # --- Switch 风格负载均衡 ---
        # f_i = 专家 i 的 dispatch 占比（Σf = 1）
        # P_i = 专家 i 的平均路由概率（ΣP = 1）
        # L = N · Σ_i f_i·P_i，均匀时恰为 1，完全坍缩时为 N。
        #
        # 必须在 [0, N) 上统计**重要性**；若只在 top-k 上算，
        # top-k 概率本来就归一化，指标恒等于 1，完全测不出坍缩。
        #
        # ★**绝不能 detach 概率**。dispatch 占比 f 来自 argmax/top-k，
        # 本来就不可微；Switch 损失的梯度**完全**来自 P 项。
        # 一旦 detach，整个 lb 就退化成常量，梯度为 0——
        # 均衡项看起来"算出来了"却完全不起作用，
        # 专家照样会一路坍缩到单一专家。
        lb = _switch_balance(self.last_micro_topk,
                             self.last_micro_probs.float(),
                             self.num_micro)

        # --- 大核同样要均衡 ---
        # 之前只给小核加了均衡项，实测把路由权重调大后，
        # 大核分布直接从均匀塌到 Arts=0.02 / Sci=0.55：
        # CE 只要求"预测对"，一旦某类更好预测，router 就会把概率
        # 全压过去，而**没有任何惩罚**。32 个领域映射到 4 个大核，
        # 塌掉一个就等于 1/4 的领域失去专属容量。
        if self.num_macro > 1:
            lb = lb + _dense_balance(
                self.last_macro_probs.float(), self.num_macro)

        return macro_ce, micro_ce, lb


def _switch_balance(topk: torch.Tensor, probs: torch.Tensor,
                    n_experts: int) -> torch.Tensor:
    """top-k 稀疏路由的 Switch 辅助损失：``N · Σ_i f_i·P_i``。

    Parameters
    ----------
    topk:  ``[B, T, k]`` 每个 token 选中的专家 id（**不可微**，只用来算 f）
    probs: ``[B, T, k]`` 对应的概率（**沿 k 归一化**，故 Σ_i P_i = 1）。
        **必须保留计算图**——梯度全部经由这一项回传。

    均匀路由时结果恰为 1；完全坍缩到一个专家时为 ``N``。
    """
    B, T, k = probs.shape
    n_tok = max(1, B * T)

    flat_idx = topk.reshape(-1)
    flat_p = probs.reshape(-1)
    dev, dt = flat_p.device, flat_p.dtype

    counts_i = torch.zeros(n_experts, device=dev, dtype=dt)
    counts_i.scatter_add_(0, flat_idx, torch.ones_like(flat_p))
    f = counts_i / (n_tok * k)                                  # Σ = 1

    prob_i = torch.zeros(n_experts, device=dev, dtype=dt)
    prob_i.scatter_add_(0, flat_idx, flat_p)
    P = prob_i / n_tok                                          # Σ = 1

    return n_experts * (f * P).sum()


def _dense_balance(probs: torch.Tensor, n_experts: int) -> torch.Tensor:
    """dense softmax 路由（top-1 dispatch）的 Switch 辅助损失。

    与 :func:`_switch_balance` 的关键区别：这里的 ``probs`` 是
    ``[B, T, N]`` 的**完整** softmax（Σ_i P_i = 1），而不是 top-k 之后
    重新归一化的概率。

    只把 argmax 位置上的概率喂进去会得到 ``Σ_i P_i ≈ 0.28``（即最大
    概率的均值）而非 1，整个量纲就错了——实测均衡时算出 0.335 而非 1，
    坍缩时也只有 1.2 而非 ``N``，几乎测不出坍缩。

    与 :func:`_switch_balance` 一样，``probs`` **必须保留计算图**：
    argmax 给出的 f 不可微，梯度完全经由 P 项回传。
    """
    B, T, N = probs.shape
    n_tok = max(1, B * T)
    dev, dt = probs.device, probs.dtype

    P = probs.mean(dim=(0, 1))                                  # [N], Σ = 1
    idx = probs.argmax(-1).reshape(-1)                          # [B*T]
    counts = torch.zeros(N, device=dev, dtype=dt)
    counts.scatter_add_(0, idx, torch.ones(n_tok, device=dev, dtype=dt))
    f = counts / n_tok                                          # Σ = 1

    return N * (f * P).sum()


# ---------------------------------------------------------------------------
# 层定位与装配
# ---------------------------------------------------------------------------
def locate_layers(model: nn.Module) -> Any:
    """定位 decoder 层列表。兼容多种模型族（Gemma/Llama/Mistral/Qwen）。"""
    candidates = [
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.layers,
    ]
    for get in candidates:
        try:
            obj = get(model)
            if obj is not None:
                return obj
        except AttributeError:
            continue
    raise RuntimeError(
        "无法定位模型主干 Layers！已尝试: "
        + ", ".join(get.__code__.co_firstlineno for get in candidates))


def wrap_layers(model: nn.Module, layers: Any, **kwargs) -> list[SwarmWrapper]:
    """把每层的 MLP 换成 :class:`SwarmWrapper`，返回新建的包装器列表。"""
    wrappers: list[SwarmWrapper] = []
    for layer in layers:
        wrappers.append(SwarmWrapper(layer.mlp, **kwargs))
        layer.mlp = wrappers[-1]
    return wrappers


def text_config(model: nn.Module):
    """取 text config（Gemma4 等多模态模型的 hidden_size 在 text_config 下）。"""
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


def hidden_dim_of(model: nn.Module, layers: Any) -> int:
    tc = text_config(model)
    h = getattr(tc, "hidden_size", None)
    if h:
        return int(h)
    # 退路：从 MLP 的输入维度反推
    return int(layers[0].mlp.gate_proj.in_features)


def strip_multimodal(model: nn.Module) -> nn.Module:
    """丢弃视觉/音频塔，只留语言塔（``text_only=True``）。

    本项目底座 E2B 含 vision+audio 塔共约 0.88GB，纯文本任务用不到。
    """
    for attr in ("vision_tower", "audio_tower", "visual",
                 "vision_model", "audio_model"):
        if hasattr(model, attr):
            try:
                setattr(model, attr, _Null())
            except Exception:
                pass
    inner = getattr(model, "model", None)
    if inner is not None:
        for attr in ("vision_tower", "audio_tower", "vision_model",
                     "audio_model", "embed_vision", "embed_audio"):
            if hasattr(inner, attr):
                try:
                    setattr(inner, attr, _Null())
                except Exception:
                    pass
    return model


__all__ = [
    "ScalpelLoRA", "SwarmWrapper", "locate_layers", "wrap_layers",
    "hidden_dim_of", "text_config", "strip_multimodal",
]