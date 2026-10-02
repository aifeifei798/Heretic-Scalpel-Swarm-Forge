"""训练循环。

与原 ``scalpel_forge.py`` 的关键差异（每条都对应一个已修的 bug）：

======================================  ====================================
修复                                      具体做法
======================================  ====================================
B4 prompt 被监督                          assistant-mask，只监督回答段
B5 padding 浪费 87%                      动态 padding + 长度分桶
B7 bf16 上做 AdamW                      可训练参数 fp32，前向 autocast
B8 工程缺失                              seed / warmup+cosine / grad clip /
                                         ckpt / resume / eval
B2 路由坍缩                              load-balance 辅助损失
A2 大核目标只有 0/1                      由 domain_to_macro 映射算出 [0, M)
======================================  ====================================
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from .dataset import (
    SwarmDataset,
    char_lengths,
    load_rows,
    load_tokenizer,
    make_collate,
    length_grouped_batches,
)
from .losses import chunked_causal_ce, encode_hidden
from .modeling import (
    SwarmWrapper,
    hidden_dim_of,
    locate_layers,
    strip_multimodal,
    wrap_layers,
)
from .schema import ForgeConfig

Emit = Callable[[dict[str, Any]], None]


def _noop(_: dict[str, Any]) -> None:  # pragma: no cover
    pass


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------
def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """只在回答段上算因果语言模型交叉熵。

    ★**必须错位一格**。因果 LM 里 ``logits[t]`` 预测的是 ``token t+1``，
    而 HF 的 ``model(labels=...)`` 内部会替你做这个 shift；自己算 CE 就必须
    自己 shift。漏掉这一步等于让模型"预测自己已经看到的 token"，
    loss 会比随机初始化还高（实测 17~19，而 ln(262144) = 12.5），
    而且**不会报任何错**，只会让训练看起来"在跑但学不到东西"。

    Parameters
    ----------
    logits: ``[B, T, V]``
    labels: ``[B, T]``，prompt 段与 padding 已置 -100
    mask:   ``[B, T]`` bool，padding 位为 False

    Returns
    -------
    标量 loss。若没有任何可监督位置，返回 ``nan``（调用方应报错而不是静默继续）。
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    sup = (shift_labels != -100) & mask[:, 1:]
    if not bool(sup.any()):
        return shift_logits.sum() * float("nan")
    return nn.functional.cross_entropy(
        shift_logits[sup], shift_labels[sup], ignore_index=-100)


def count_supervised(labels: torch.Tensor, mask: torch.Tensor) -> int:
    """回答段里实际参与监督的 token 数（B4 自检：必须 > 0）。"""
    return int(((labels[:, 1:] != -100) & mask[:, 1:]).sum())


# ---------------------------------------------------------------------------
# 复现性
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定所有随机源（B8 缺失项之一）。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 学习率调度
# ---------------------------------------------------------------------------
def lr_lambda(step: int, total: int, warmup: int, kind: str) -> float:
    """warmup + cosine/linear 调度（B8）。"""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    p = min(1.0, max(0.0, p))
    if kind == "linear":
        return 1.0 - p
    return 0.5 * (1.0 + math.cos(math.pi * p))


# ---------------------------------------------------------------------------
# 装配
# ---------------------------------------------------------------------------
@dataclass
class TrainContext:
    model: Any
    wrappers: list[SwarmWrapper]
    layers: Any
    device: str
    param_dtype: torch.dtype
    hidden_dim: int


def build_model(cfg: ForgeConfig, *, train: bool = True) -> TrainContext:
    """加载底座、装配 swarm 包装器、冻结除 LoRA 外的一切。"""
    from transformers import AutoModelForCausalLM

    device = cfg.train.resolved_device()
    compute_dtype = DTYPES[cfg.train.param_dtype]
    # 底座权重用 bf16（省显存），可训练参数用 fp32（B7）
    base_dtype = torch.bfloat16 if device != "cpu" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        dtype=base_dtype,
        trust_remote_code=cfg.trust_remote_code,
    )
    if cfg.text_only:
        strip_multimodal(model)
    model.to(device)

    model.config.use_cache = not train
    if train:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    for p in model.parameters():
        p.requires_grad = False

    layers = locate_layers(model)
    h = hidden_dim_of(model, layers)

    arch = cfg.arch
    wrappers = wrap_layers(
        model, layers,
        hidden_dim=h,
        num_macro=arch.num_macro_cores,
        macro_rank=arch.macro_rank,
        num_micro=arch.num_micro_experts,
        micro_rank=arch.micro_rank,
        micro_scale=arch.micro_scale,
        macro_scale=arch.macro_dense_scale,
        micro_top_k=cfg.train.micro_top_k,
        param_dtype=compute_dtype,
        device=device,
    )
    if train:
        model.gradient_checkpointing_enable()
        model.train()
    else:
        model.eval()
    return TrainContext(model=model, wrappers=wrappers, layers=layers,
                        device=device, param_dtype=compute_dtype,
                        hidden_dim=h)


def collect_params(wrappers: list[SwarmWrapper]
                   ) -> dict[str, list[nn.Parameter]]:
    macro, router, micro = [], [], []
    for w in wrappers:
        g = w.param_groups()
        macro += g["macro"]
        router += g["router"]
        micro += g["micro"]
    return {"macro": macro, "router": router, "micro": micro}


def build_optimizer(params: dict[str, list[nn.Parameter]],
                    cfg: ForgeConfig, total_steps: int):
    t = cfg.train
    groups = []
    for key, lr in (("macro", t.lr_macro), ("router", t.lr_router),
                    ("micro", t.lr_micro)):
        ps = [p for p in params[key] if p.requires_grad]
        if not ps:
            continue
        if t.fused_optimizer and ps[0].is_cuda:
            groups.append({"params": ps, "lr": lr, "weight_decay": t.weight_decay,
                           "fused": True})
        else:
            groups.append({"params": ps, "lr": lr, "weight_decay": t.weight_decay})
    return torch.optim.AdamW(groups, lr=t.lr_macro), groups


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
@dataclass
class RouteStats:
    """累计路由分布，用于训练后判断专家是否坍缩。"""

    macro_hits: torch.Tensor
    micro_hits: torch.Tensor
    tokens: int = 0
    macro_names: list[str] = field(default_factory=list)
    micro_top_k: int = 2

    @classmethod
    def zeros(cls, arch, device: str, micro_top_k: int = 2) -> "RouteStats":
        return cls(
            macro_hits=torch.zeros(arch.num_macro_cores, device=device),
            micro_hits=torch.zeros(arch.num_micro_experts, device=device),
            macro_names=list(arch.macro_names),
            micro_top_k=micro_top_k,
        )

    def update(self, w: SwarmWrapper, mask: torch.Tensor) -> None:
        """累计一次前向的路由分布。

        注意 ``last_micro_topk`` 形状是 ``[B, T, k]``，按 bool mask 索引后
        是 ``[n_tok, k]``——二维。``scatter_add_`` 要求 index 与 self 同维，
        因此必须先展平，否则会报 "Index tensor must have the same number
        of dimensions"。
        """
        with torch.no_grad():
            n_tok = int(mask.sum())
            if n_tok == 0:
                return
            dev = self.macro_hits.device

            m_idx = w.last_macro_logits[mask].argmax(-1)        # [n_tok]
            self.macro_hits.scatter_add_(
                0, m_idx.to(dev), torch.ones(n_tok, device=dev))

            u_idx = w.last_micro_topk[mask].reshape(-1)         # [n_tok*k]
            self.micro_hits.scatter_add_(
                0, u_idx.to(dev),
                torch.ones(u_idx.numel(), device=dev))

            self.tokens += n_tok

    def summary(self) -> dict[str, Any]:
        macro = self.macro_hits.cpu().tolist()
        micro = self.micro_hits.cpu().tolist()
        ms, ss = sum(macro), sum(micro)
        return {
            "tokens": self.tokens,
            "macro_dist": {name: round(v / max(1e-9, ms), 4)
                           for name, v in zip(self.macro_names, macro)},
            "micro_dist": [round(v / max(1e-9, ss), 5) for v in micro],
            "macro_dead": int((self.macro_hits == 0).sum()),
            "micro_dead": int((self.micro_hits == 0).sum()),
            "micro_kills": self.micro_top_k,
        }


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(cfg: ForgeConfig, model, ctx: TrainContext,
             val_rows: list[dict[str, Any]], *, emit: Emit = _noop,
             max_batches: int | None = None) -> dict[str, float]:
    """在 held-out 上算 loss。

    为什么必须有它
    --------------
    训练 loss 是在**刚更新过的权重**上、对**刚见过的 batch** 报的，
    而每个 batch 只有 ``batch_size`` 条样本、样本长度差异还很大，
    步与步之间的波动轻松到 2 倍（本项目实测 2.7 ↔ 6.6）。
    这种数字**无法区分"在学"和"在抖"**，更无法区分"在学"和"在过拟合"。

    所以固定一批验证样本，每隔 ``eval_every`` 步在 ``model.eval()`` 下
    重跑一次：**只有这条曲线下降才算真的学会了东西**。
    """
    t_cfg = cfg.train
    if not val_rows:
        return {}
    device = t_cfg.resolved_device()

    tok = load_tokenizer(cfg.base_model_id, trust_remote_code=cfg.trust_remote_code)
    ds = SwarmDataset(val_rows, tok, cfg.arch, max_length=t_cfg.max_length)
    collate = make_collate(ds, tok.pad_token_id)

    # 顺序固定（不打乱），保证不同时刻的 val loss 可比
    idx = list(range(len(ds)))
    batches = [idx[i:i + t_cfg.batch_size]
               for i in range(0, len(idx), t_cfg.batch_size)]
    if max_batches is not None:
        batches = batches[:max_batches]

    was_training = model.training
    model.eval()
    tot_lm = tot_mce = tot_nce = 0.0
    n_seen = n_batches = 0
    try:
        for bidx in batches:
            b = collate([ds[i] for i in bidx])
            input_ids = b["input_ids"].to(device)
            attn = b["attention_mask"].to(device)
            labels = b["labels"].to(device)
            macro_t = b["macro_target"].to(device)
            micro_t = b["micro_target"].to(device)
            mask = attn == 1
            if count_supervised(labels, mask) == 0:
                continue

            hidden = encode_hidden(model, input_ids, attn)
            lm, _ = chunked_causal_ce(model, hidden, labels, mask,
                                      chunk_size=t_cfg.loss_chunk_size)
            if not torch.isfinite(lm):
                continue

            mce = nce = 0.0
            if ctx.wrappers:
                acc_m = acc_n = 0.0
                for w in ctx.wrappers:
                    a, c_, _ = w.router_losses(mask, macro_t, micro_t)
                    acc_m += float(a)
                    acc_n += float(c_)
                mce = acc_m / len(ctx.wrappers)
                nce = acc_n / len(ctx.wrappers)

            k = int(mask.sum())
            tot_lm += float(lm) * k
            tot_mce += mce * k
            tot_nce += nce * k
            n_seen += k
            n_batches += 1
    finally:
        if was_training:
            model.train()

    if n_seen == 0:
        return {}
    out = {
        "val_lm_loss": tot_lm / n_seen,
        "val_macro_ce": tot_mce / n_seen,
        "val_micro_ce": tot_nce / n_seen,
        "val_tokens": float(n_seen),
        "val_batches": float(n_batches),
    }
    emit({"t": "eval", **out})
    return out


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train(cfg: ForgeConfig, *, emit: Emit = _noop,
          rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """跑一次训练，返回产物元信息。每步通过 *emit* 发出 NDJSON 事件。"""
    t_cfg = cfg.train
    set_seed(t_cfg.seed)
    device = t_cfg.resolved_device()

    rows = rows if rows is not None else load_rows(cfg.data_path, split="train")
    if not rows:
        raise ValueError(f"数据集为空或不存在：{cfg.data_path}")

    tok = load_tokenizer(cfg.base_model_id, trust_remote_code=cfg.trust_remote_code)
    ds = SwarmDataset(rows, tok, cfg.arch, max_length=t_cfg.max_length)
    collate = make_collate(ds, tok.pad_token_id)

    ctx = build_model(cfg, train=True)
    params = collect_params(ctx.wrappers)
    n_trainable = sum(p.numel() for ps in params.values() for p in ps)

    steps_per_epoch = math.ceil(len(ds) / t_cfg.batch_size)
    max_steps = t_cfg.max_steps
    optimizer, groups = build_optimizer(params, cfg, max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: lr_lambda(s, max_steps,
                            int(max_steps * t_cfg.warmup_ratio),
                            t_cfg.lr_schedule),
    )

    emit({
        "t": "train_start",
        "project": cfg.project_name,
        "base_model": cfg.base_model_id,
        "rows": len(ds),
        "max_steps": max_steps,
        "steps_per_epoch": steps_per_epoch,
        "effective_batch": t_cfg.batch_size * t_cfg.grad_accum,
        "max_length": t_cfg.max_length,
        "trainable_params": n_trainable,
        "macro": cfg.arch.num_macro_cores,
        "micro": cfg.arch.num_micro_experts,
        "micro_top_k": t_cfg.micro_top_k,
        "device": device,
        "param_dtype": t_cfg.param_dtype,
    })

    # --- batch 顺序：长度分桶（B5） ---
    rng = random.Random(t_cfg.seed)
    lengths = char_lengths(rows)
    if t_cfg.length_grouped_batches:
        batches = length_grouped_batches(lengths, t_cfg.batch_size, rng=rng)
    else:
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        batches = [idx[i:i + t_cfg.batch_size]
                   for i in range(0, len(idx), t_cfg.batch_size)]

    stats = RouteStats.zeros(cfg.arch, device, t_cfg.micro_top_k)
    amp = (device != "cpu" and t_cfg.param_dtype in ("bf16", "fp16"))
    amp_dtype = DTYPES[t_cfg.param_dtype]

    # held-out：只有这条曲线下降才算真的学会了东西
    val_rows = load_rows(cfg.data_path, split="val")

    model = ctx.model
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    step = micro = 0
    losses: list[float] = []
    hist: list[dict[str, Any]] = []
    stop = False

    bi = 0
    while not stop and step < max_steps:
        for batch_idx in batches:
            if step >= max_steps:
                break
            batch = [ds[i] for i in batch_idx]
            b = collate(batch)
            input_ids = b["input_ids"].to(device, non_blocking=True)
            attn = b["attention_mask"].to(device, non_blocking=True)
            labels = b["labels"].to(device, non_blocking=True)
            macro_t = b["macro_target"].to(device)
            micro_t = b["micro_target"].to(device)
            mask = attn == 1

            # B4 自检：若 assistant-mask 失效，一个 batch 里会一个可监督
            # token 都没有——那说明监督信号全打在题面上，必须立刻暴露。
            if count_supervised(labels, mask) == 0:
                raise RuntimeError(
                    "本 batch 没有任何可监督 token：assistant-mask 定位失败"
                    "（dataset.find_response_start 未匹配到 chat template），"
                    "此时 LM loss 无意义，拒绝继续训练。")

            with torch.autocast(device_type="cuda" if device != "cpu" else "cpu",
                                dtype=amp_dtype, enabled=amp):
                # B12：绝不取 out.logits。vocab=262144 时 [B,T,V] 一份
                # bf16 就有 0.5GiB@B=4,T=256，再 .float() 翻倍。
                # 这里只跑主干拿 hidden（[B,T,1536]，MB 级），
                # 再把需要监督的位置切块过 lm_head。
                hidden = encode_hidden(model, input_ids, attn)
                lm_loss, n_sup = chunked_causal_ce(
                    model, hidden, labels, mask,
                    chunk_size=t_cfg.loss_chunk_size)

                # ★路由辅助损失对层**求和**，权重按"每层强度"配置，
                # 因此实际梯度是 weight * n_layers * 每层梯度。
                # 对外报告时再除回层数（见下方 step 事件）。
                mce = nce = lb = torch.zeros((), device=device)
                n_layers = max(1, len(ctx.wrappers))
                for w in ctx.wrappers:
                    a, b_, c = w.router_losses(mask, macro_t, micro_t)
                    mce = mce + a
                    nce = nce + b_
                    lb = lb + c

            loss = (t_cfg.lm_loss_weight * lm_loss
                    + t_cfg.router_aux_weight * (mce + nce)
                    + t_cfg.load_balance_weight * lb)
            (loss / t_cfg.grad_accum).backward()

            micro += 1
            if micro % t_cfg.grad_accum == 0:
                if t_cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for ps in params.values() for p in ps],
                        t_cfg.max_grad_norm)
                optimizer.step()
                sched.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % t_cfg.log_every == 0 or step == max_steps:
                    # ★对外报告的是**每层均值**，不是求和。
                    # 辅助损失按层求和参与反向传播（权重才与深度无关），
                    # 但 35 层求和后 micro_ce 会显示成 97.06，
                    # 看着像炸了，其实 97.06/35 ≈ ln(32)。两个口径混用
                    # 会让"训练在跑"和"指标正常"看起来互相矛盾。
                    ev = {
                        "t": "step",
                        "step": step,
                        "max_steps": max_steps,
                        "lm_loss": float(lm_loss),
                        "macro_loss": float(mce) / n_layers,
                        "micro_loss": float(nce) / n_layers,
                        "load_balance": float(lb) / n_layers,
                        "macro_ce_sum": float(mce),
                        "micro_ce_sum": float(nce),
                        "load_balance_sum": float(lb),
                        "n_layers": n_layers,
                        "lr": optimizer.param_groups[0]["lr"],
                        "speed": micro * t_cfg.batch_size / (time.time() - t0),
                        "vram_gb": torch.cuda.max_memory_allocated() / 1024**3
                        if device != "cpu" else 0.0,
                        "elapsed": round(time.time() - t0, 2),
                    }
                    emit(ev)

                if (t_cfg.eval_every and val_rows
                        and (step % t_cfg.eval_every == 0 or step == max_steps)):
                    ev.update(evaluate(cfg, model, ctx, val_rows,
                                       emit=emit,
                                       max_batches=t_cfg.eval_batches))
                    hist.append({k: v for k, v in ev.items()
                                 if k not in ("t", "max_steps")})
                    losses.append(float(lm_loss))

                if t_cfg.ckpt_every and step % t_cfg.ckpt_every == 0:
                    save_checkpoint(cfg, model, ctx, step, Path(cfg.checkpoint_path))

                if step >= max_steps:
                    stop = True
                    break
            bi += 1

    # --- 收尾 ---
    if len(ctx.wrappers):
        with torch.no_grad():
            sample = [ds[i] for i in batches[0][:min(4, len(batches[0]))]]
            b = collate(sample)
            # 同样只跑主干：这里只是为了刷新 wrapper 的路由缓存，
            # 没必要为它物化一份 [B, T, 262144] 的 logits。
            _ = encode_hidden(model, b["input_ids"].to(device),
                              b["attention_mask"].to(device))
            for w in ctx.wrappers:
                stats.update(w, b["attention_mask"].to(device) == 1)

    ckpt = save_checkpoint(cfg, model, ctx, step, Path(cfg.checkpoint_path))
    summary = stats.summary()
    emit({
        "t": "train_done",
        "steps": step,
        "wall_sec": round(time.time() - t0, 2),
        "final_lm_loss": losses[-1] if losses else None,
        "checkpoint": str(ckpt),
        **summary,
    })
    if ds.mask_warnings:
        emit({"t": "warn", "msg": f"{ds.mask_warnings} 条样本未能定位 "
                                   f"assistant 段，已退化为全量监督"})

    # 历史曲线跟着 checkpoint 走，不往仓库根目录丢文件
    hist_path = Path(cfg.checkpoint_path).with_name("train_history.json")
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(
        json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "steps": step,
        "wall_sec": round(time.time() - t0, 2),
        "checkpoint": str(ckpt),
        "trainable_params": n_trainable,
        "mask_warnings": ds.mask_warnings,
        **summary,
    }


# ---------------------------------------------------------------------------
# 检查点
# ---------------------------------------------------------------------------
def save_checkpoint(cfg: ForgeConfig, model, ctx: TrainContext,
                    step: int, path: Path) -> Path:
    """只存可训练参数（LoRA + router）。

    全量存 9.5GB 底座没有意义——发布包会重新加载底座，只叠加这几十 MB 的
    swarm 权重（见 :mod:`forge_core.export`）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sd: dict[str, Any] = {"__meta__": {
        "step": step,
        "project": cfg.project_name,
        "base_model_id": cfg.base_model_id,
        "hidden_dim": ctx.hidden_dim,
        "arch": cfg.arch.model_dump(mode="json"),
        "format": "forge-swarm-v2",
    }}
    for i, w in enumerate(ctx.wrappers):
        for j, core in enumerate(w.macro_cores):
            sd[f"layer{i}.macro{j}"] = core.state_dict()
        sd[f"layer{i}.router_macro"] = w.router_macro.state_dict()
        sd[f"layer{i}.router_micro"] = w.router_micro.state_dict()
        for j, expert in enumerate(w.micro_pool):
            sd[f"layer{i}.micro{j}"] = expert.state_dict()
    torch.save(sd, path)
    return path


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


__all__ = ["train", "evaluate", "build_model", "collect_params", "set_seed",
           "lr_lambda", "RouteStats", "save_checkpoint", "load_checkpoint",
           "TrainContext", "causal_lm_loss", "count_supervised"]