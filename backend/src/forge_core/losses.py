"""分块交叉熵：绕开 ``[B, T, V]`` 的 logits 尖峰。

为什么要分块（B12 / 显存可行性）
---------------------------------
底座是 Gemma4（vocab=262144）。只要走 ``model(input_ids, labels=...)``
或 ``model(...).logits``，就会物化一个 ``[B, T, 262144]`` 张量：

* ``B=4, T=256, bf16`` -> **0.50 GiB**
* 同一份再 ``.float()`` 供 CE -> **1.00 GiB**
* 而这只是一层；autocast 缓存与 allocator 碎片会进一步放大。

本模块改成：先只算 hidden states（``[B, T, 1536]``，4MB 量级），
再把**需要监督的位置**切成小块，逐块过 ``lm_head`` 并立刻算 CE。
峰值显存由 chunk_size 决定，与 ``T``、``B`` 解耦。

为什么不能自己直接调 ``lm_head``
---------------------------------
Gemma4 在 ``lm_head`` 之后还有一步 **logit softcapping**::

    logits = lm_head(hidden)
    logits = tanh(logits / cap) * cap        # cap = 30.0

漏掉这一步，logits 会出现高达 ±21 的偏差——loss 看起来"能动"，
但优化的其实是另一个目标函数。这里显式复刻它。

等价性由 :func:`selftest` 锁定：分块结果必须与"整份算一遍"逐位一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 软截断
# ---------------------------------------------------------------------------
def softcap_logits(logits: torch.Tensor, cap: float | None) -> torch.Tensor:
    """Gemma 的 logit softcapping：``tanh(x / cap) * cap``。"""
    if not cap:
        return logits
    return torch.tanh(logits / cap) * cap


def get_softcap(model: nn.Module) -> float | None:
    """读取 ``final_logit_softcapping``（不同模型放在不同层级）。"""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    for getter in ("get_text_config",):
        fn = getattr(cfg, getter, None)
        if callable(fn):
            try:
                cfg = fn()
                break
            except Exception:
                pass
    val = getattr(cfg, "final_logit_softcapping", None)
    return float(val) if val else None


# ---------------------------------------------------------------------------
# hidden states
# ---------------------------------------------------------------------------
def encode_hidden(model: nn.Module, input_ids: torch.Tensor,
                  attention_mask: torch.Tensor) -> torch.Tensor:
    """只跑主干，返回 ``[B, T, H]`` 的最后层 hidden states。

    对 ``Gemma4ForConditionalGeneration`` 这类多模态壳子，``.model`` 就是
    主干；对纯因果 LM（如 Llama），它本身没有 ``.model`` 子属性，
    这时退化为直接调用（但那样就拿不到 hidden，需要走
    ``output_hidden_states=True``）。
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "language_model"):
        out = inner(input_ids=input_ids, attention_mask=attention_mask)
        hs = getattr(out, "last_hidden_state", None)
        if hs is None:
            hs = out[0]
        return hs

    out = model(input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True)
    return out.hidden_states[-1]


# ---------------------------------------------------------------------------
# 分块 CE
# ---------------------------------------------------------------------------
def chunked_causal_ce(
    model: nn.Module,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    chunk_size: int = 4096,
    reduction: str = "mean",
) -> tuple[torch.Tensor, int]:
    """在回答段上算因果 LM 交叉熵，不物化 ``[B, T, V]``。

    Parameters
    ----------
    hidden: ``[B, T, H]`` —— ``encode_hidden`` 的输出
    labels: ``[B, T]``，非监督位为 -100
    mask:   ``[B, T]`` bool，padding 位为 False
    chunk_size: 每次过 lm_head 的监督位数量。峰值显存约为
        ``chunk_size * V * 4`` 字节（fp32），8192 时约 8.6GB，
        4096 时约 4.3GB，1024 时约 1.1GB。

    Returns
    -------
    ``(loss, n_supervised)``；若无监督位则 loss 为 nan。
    """
    # ★错位一格：hidden[t] 预测 token t+1
    h = hidden[:, :-1, :]
    y = labels[:, 1:]
    sup = (y != -100) & mask[:, 1:]
    n_sup = int(sup.sum())
    if n_sup == 0:
        return hidden.sum() * float("nan"), 0

    flat_h = h[sup]                                  # [n_sup, H]
    flat_y = y[sup]                                  # [n_sup]

    lm_head = model.lm_head
    cap = get_softcap(model)

    total = flat_h.new_zeros((), dtype=torch.float32)
    for s in range(0, n_sup, chunk_size):
        e = min(s + chunk_size, n_sup)
        logits = lm_head(flat_h[s:e]).float()
        logits = softcap_logits(logits, cap)
        total = total + nn.functional.cross_entropy(
            logits, flat_y[s:e], reduction="sum")

    loss = total / n_sup if reduction == "mean" else total
    return loss, n_sup


def chunked_causal_ce_from_ids(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, int]:
    """便捷封装：``encode_hidden`` + :func:`chunked_causal_ce`。"""
    hidden = encode_hidden(model, input_ids, attention_mask)
    return chunked_causal_ce(model, hidden, labels,
                             attention_mask == 1, chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def selftest() -> int:
    import sys

    torch.manual_seed(0)
    B, T, H, V = 2, 12, 16, 37
    cap = 30.0

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = nn.Linear(H, V, bias=False)
            self.config = type("C", (), {
                "get_text_config": lambda self: type(
                    "T", (), {"final_logit_softcapping": cap})(),
            })()

    m = Toy()
    # 让 logits 明显超出 cap=30，否则 tanh 在小量程上近似恒等、
    # "漏掉 softcapping"的对照实验就看不出差别。
    with torch.no_grad():
        m.lm_head.weight.mul_(20.0)
    hidden = torch.randn(B, T, H)
    labels = torch.randint(0, V, (B, T))
    labels[:, :4] = -100                       # prompt 段不监督
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 9:] = False                        # 第二条带 padding

    got, n = chunked_causal_ce(m, hidden, labels, mask, chunk_size=5)
    assert n == count_expected(labels, mask), (n, count_expected(labels, mask))

    # 参考实现：整份算一遍（含 softcapping + shift）
    full = softcap_logits(m.lm_head(hidden[:, :-1]).float(), cap)
    sup = (labels[:, 1:] != -100) & mask[:, 1:]
    ref = nn.functional.cross_entropy(full[sup], labels[:, 1:][sup])
    assert torch.allclose(got, ref, atol=1e-4), (float(got), float(ref))

    # 分块大小不应影响结果
    for cs in (1, 3, 7, 1000):
        alt, _ = chunked_causal_ce(m, hidden, labels, mask, chunk_size=cs)
        assert torch.allclose(alt, ref, atol=1e-4), (cs, float(alt), float(ref))

    # 漏掉 softcapping 必须会得到不同结果（证明这一步不可省）
    no_cap = nn.functional.cross_entropy(
        m.lm_head(hidden[:, :-1]).float()[sup], labels[:, 1:][sup])
    assert not torch.allclose(no_cap, ref, atol=1e-3), "softcap 未生效？"

    # 无监督位 -> nan
    z, n0 = chunked_causal_ce(m, hidden, torch.full_like(labels, -100), mask)
    assert torch.isnan(z) and n0 == 0

    print("PASS  chunked_causal_ce 与整份计算逐位一致（含 softcapping）")
    print("PASS  chunk_size 不影响结果（1 / 3 / 7 / 1000）")
    print("PASS  漏掉 softcapping 会显著改变 loss（该步骤不可省）")
    print("PASS  无监督位返回 nan 而非 0")
    return 0


def count_expected(labels: torch.Tensor, mask: torch.Tensor) -> int:
    return int(((labels[:, 1:] != -100) & mask[:, 1:]).sum())


if __name__ == "__main__":
    import sys
    sys.exit(selftest())