"""训练关键不变量的自检。

这些断言针对**不会报错、只会静默毁掉训练**的那类缺陷。
它们捕捉过的真实 bug：

* ``test_causal_loss_shifts_logits`` —— 因果 LM 的 shift 漏掉。
  没有 shift 时 loss 是 17~19，而 ``ln(262144) = 12.5``，
  即比随机初始化还差；**训练照跑、loss 照打、什么都不报错**，
  但模型学不到任何东西。这是本次修复中最危险的一个。
* ``test_supervised_count_matches_response`` —— assistant-mask 失效
  （监督信号全打在题面上）。
* ``test_loss_below_random_on_oracle`` —— 端到端校验：
  一个"知道下一个 token"的模型必须拿到极低 loss。
* ``test_lr_schedule_warmup_cosine`` —— warmup+cosine 的形状。
* ``test_route_stats_scatter`` —— ``[B,T,k]`` 的 top-k 索引展平。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .train import (
    RouteStats,
    causal_lm_loss,
    count_supervised,
    lr_lambda,
)


V = 64          # 小 vocab，让 oracle 测试可精确构造
B, T = 2, 8


def _rand_labels(B_: int = B, T_: int = T) -> torch.Tensor:
    return torch.randint(0, V, (B_, T_))


# ---------------------------------------------------------------------------
# 1. shift
# ---------------------------------------------------------------------------
def test_causal_loss_shifts_logits() -> None:
    """loss 必须用 ``logits[t]`` 对 ``labels[t+1]``，而不是 ``labels[t]``。

    构造一个"完美预测器"：``logits[b,t]`` 在真实答案 token 上给极大 logit。
    若正确 shift，loss 应趋近 0；若漏掉 shift，loss 会很大。
    """
    labels = _rand_labels()
    mask = torch.ones(B, T, dtype=torch.bool)

    # oracle: 位置 t 的 logits 指向 labels[t+1]
    logits = torch.full((B, T, V), -20.0)
    for b in range(B):
        for t in range(T - 1):
            logits[b, t, int(labels[b, t + 1])] = 20.0

    loss = causal_lm_loss(logits, labels, mask)
    assert torch.isfinite(loss), loss
    assert float(loss) < 1e-3, f"正确 shift 下 oracle loss 应≈0，实得 {float(loss)}"

    # 反证：**真正不 shift**（logits[t] 对 labels[t]）会非常差。
    # 这正是线上跑出来的 bug——loss 17~19 > ln(262144)=12.5。
    unshifted = nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, V), labels[:, :-1].reshape(-1))
    assert float(unshifted) > 10.0, (
        f"对照组构造有误：未 shift 的 loss 应 > 10，实得 {float(unshifted)}")


def test_loss_below_random_on_oracle() -> None:
    """端到端：一个能预测下一 token 的模型，loss 必须远低于 ln(V)。"""
    labels = _rand_labels()
    mask = torch.ones(B, T, dtype=torch.bool)
    logits = torch.full((B, T, V), -20.0)
    for b in range(B):
        for t in range(T - 1):
            logits[b, t, int(labels[b, t + 1])] = 20.0

    loss = float(causal_lm_loss(logits, labels, mask))
    assert loss < math.log(V) * 0.05, (loss, math.log(V))


def test_uniform_logits_loss_equals_ln_vocab() -> None:
    """全零 logits + 随机标签时，loss 必须**精确**等于 ln(V)。

    这是 shift/展平/mask 三者最干净的"中性"校验：均匀分布下的交叉熵
    与标签无关，因此任何错位或展平错误都会立刻破坏这个等式。

    （注意：不能改用 iid N(0,1) logits 做这个校验——
    随机 logits 的期望交叉熵是 ln(V) + E[||z||²]/(2V) ≈ ln(V) + 0.5，
    那个 0.5 是数学性质而非 bug。）
    """
    torch.manual_seed(0)
    Bb, Tt = 64, 32
    labels = torch.randint(0, V, (Bb, Tt))
    mask = torch.ones(Bb, Tt, dtype=torch.bool)
    logits = torch.zeros(Bb, Tt, V)
    loss = float(causal_lm_loss(logits, labels, mask))
    assert abs(loss - math.log(V)) < 1e-4, (loss, math.log(V))


def test_padding_fraction_matches_shift() -> None:
    """padding 会让可监督位置少于 T-1；这里锁死正确的计数口径。"""
    labels = torch.randint(0, V, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[1, 6:] = False               # 第二条只有前 6 个有效
    # 有效监督位 = (T-1) + (6-1) = 9 + 5 = 14
    assert count_supervised(labels, mask) == 14


# ---------------------------------------------------------------------------
# 2. assistant mask
# ---------------------------------------------------------------------------
def test_supervised_count_matches_response() -> None:
    """只有回答段的 token 参与监督；题面必须被排除。"""
    labels = torch.full((1, 6), -100)
    labels[0, 4:] = torch.tensor([10, 11])      # 位置 4、5 是回答
    mask = torch.ones(1, 6, dtype=torch.bool)

    # shift 后可监督的是 t=3（预测 4）与 t=4（预测 5）
    assert count_supervised(labels, mask) == 2

    logits = torch.randn(1, 6, V)
    loss = causal_lm_loss(logits, labels, mask)
    assert torch.isfinite(loss), loss


def test_all_masked_raises_nan() -> None:
    """全 -100 时返回 nan（调用方据此报错），而不是静默返回 0。"""
    labels = torch.full((1, 5), -100)
    mask = torch.ones(1, 5, dtype=torch.bool)
    loss = causal_lm_loss(torch.randn(1, 5, V), labels, mask)
    assert torch.isnan(loss), loss


def test_padding_excluded_from_loss() -> None:
    """padding 位置的 label 即使不是 -100，也不得参与 loss。"""
    labels = torch.full((1, 6), -100)
    labels[0, 3] = 7
    labels[0, 5] = 9          # 这是 padding 位（mask=False）
    mask = torch.tensor([[True, True, True, True, False, False]])
    # t=2 预测 3（有效）；t=4 预测 5 但 mask[4]=False -> 排除
    assert count_supervised(labels, mask) == 1


# ---------------------------------------------------------------------------
# 3. 学习率调度
# ---------------------------------------------------------------------------
def test_routing_targets_in_range() -> None:
    """路由目标必须落在 router 的输出维度内。

    ★这是个**只在特定配置下才炸**的坑：小核目标原本直接取领域的
    全局下标 ``DOMAIN_INDEX[d]``（0~31），与 ``num_micro_experts``
    毫无关系。只要 N < 32（UI 上这是个自由输入框，绝大多数选择都 < 32），
    ``cross_entropy`` 就会抛 device-side assert，
    而且 C++ 断言会把栈指向错误的位置（实测报在 ``repeat_interleave``
    上，排查时极具误导性）。

    本项目所有早期测试都用 N=32，恰好躲过了它；直到 Web 层允许
    用户自由填 N 才暴露出来。
    """
    from .dataset import resolve_targets
    from .schema import ALL_DOMAINS, ArchProfile

    rows = [{"domain": d} for d in ALL_DOMAINS]
    default_map = {d: i % 4 for i, d in enumerate(ALL_DOMAINS)}

    for n_micro in (1, 2, 3, 5, 8, 16, 17, 31, 32, 64, 128):
        for n_macro in (1, 2, 4, 7):
            arch = ArchProfile(num_macro_cores=n_macro,
                               macro_names=[f"M{i}" for i in range(n_macro)],
                               num_micro_experts=n_micro,
                               domain_to_macro={
                                   d: i % n_macro
                                   for i, d in enumerate(ALL_DOMAINS)})
            tgts = resolve_targets(rows, arch)
            assert len(tgts) == len(rows)
            for t in tgts:
                assert 0 <= t.micro < n_micro, \
                    f"N={n_micro}: micro={t.micro} 越界（{t.domain}）"
                assert 0 <= t.macro < n_macro, \
                    f"M={n_macro}: macro={t.macro} 越界（{t.domain}）"


def test_routing_targets_reject_bad_macro_map() -> None:
    """越界的 macro 映射必须**明确报错**，不能等到 CUDA 断言。"""
    from .dataset import resolve_targets
    from .schema import ALL_DOMAINS, ArchProfile

    arch = ArchProfile(
        num_macro_cores=2, macro_names=["M0", "M1"], num_micro_experts=8,
        domain_to_macro={d: (i % 2) for i, d in enumerate(ALL_DOMAINS)})
    arch.domain_to_macro["Arts_Poetry"] = 9      # 故意越界
    try:
        resolve_targets([{"domain": "Arts_Poetry"}], arch)
    except ValueError as e:
        assert "超出" in str(e), e
    else:
        raise AssertionError("越界的 macro 映射竟然被接受了")


def test_lr_schedule_warmup_cosine() -> None:
    total, warmup = 100, 10
    lrs = [lr_lambda(s, total, warmup, "cosine") for s in range(total + 1)]
    assert abs(lrs[0] - 0.1) < 1e-9, lrs[0]                 # 线性升温
    assert abs(lrs[warmup - 1] - 1.0) < 1e-9, lrs[warmup - 1]
    assert all(lrs[i] <= lrs[i + 1] for i in range(warmup - 1))
    assert all(lrs[i] >= lrs[i + 1] for i in range(warmup, total - 1))
    assert abs(lrs[-1]) < 1e-6, lrs[-1]                       # cosine 收到 0
    assert all(v >= 0 for v in lrs)

    lin = [lr_lambda(s, total, warmup, "linear") for s in range(total + 1)]
    assert abs(lin[-1]) < 1e-6, lin[-1]


def test_lr_schedule_no_warmup() -> None:
    """warmup=0 时第一步不该是 0（否则第一步完全不更新）。"""
    lrs = [lr_lambda(s, 10, 0, "cosine") for s in range(11)]
    assert lrs[0] > 0, lrs[0]


# ---------------------------------------------------------------------------
# 4. 路由统计
# ---------------------------------------------------------------------------
def test_route_stats_scatter() -> None:
    """``[B,T,k]`` 的 top-k 索引必须展平后再 scatter。"""

    class FakeW:
        last_macro_logits = torch.randn(2, 5, 4)
        last_micro_topk = torch.randint(0, 6, (2, 5, 2))

    from .schema import ArchProfile
    arch = ArchProfile(num_macro_cores=4, macro_names=list("ABCD"),
                       num_micro_experts=6)
    st = RouteStats.zeros(arch, "cpu", micro_top_k=2)
    mask = torch.ones(2, 5, dtype=torch.bool)
    st.update(FakeW(), mask)   # 过去会抛维度不匹配

    assert st.tokens == 10
    assert abs(sum(st.macro_hits.tolist()) - 10) < 1e-6
    assert abs(sum(st.micro_hits.tolist()) - 20) < 1e-6   # n_tok * k

    s = st.summary()
    assert abs(sum(s["macro_dist"].values()) - 1.0) < 1e-3, s["macro_dist"]
    assert abs(sum(s["micro_dist"]) - 1.0) < 1e-3, s["micro_dist"]
    assert s["micro_kills"] == 2


def test_route_stats_empty_mask() -> None:
    """全 padding 的 batch 不应崩溃，也不应计入 tokens。"""
    class FakeW:
        last_macro_logits = torch.randn(2, 5, 4)
        last_micro_topk = torch.randint(0, 6, (2, 5, 2))

    from .schema import ArchProfile
    arch = ArchProfile(num_macro_cores=4, macro_names=list("ABCD"),
                       num_micro_experts=6)
    st = RouteStats.zeros(arch, "cpu")
    st.update(FakeW(), torch.zeros(2, 5, dtype=torch.bool))
    assert st.tokens == 0


if __name__ == "__main__":
    import sys
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
