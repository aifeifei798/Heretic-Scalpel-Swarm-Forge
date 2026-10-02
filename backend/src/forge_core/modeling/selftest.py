"""SwarmWrapper 的自检测试。

这些断言锁住了 P0 阶段修复的每一个关键不变量：

1. **B4**：router 不再用 GT 标签硬门控前向（否则小核训练信号断链）。
2. **B2**：训推路由一致——同一个 wrapper 在 ``train()``/``eval()`` 下
   走同一个 top-k 路径。
3. **B6**：大核汇聚不物化 ``[B,T,M,H]``（用 ``einsum``，显存与 M 线性无关）。
4. **B2 向量化**：小核求和不做 Python 循环（否则每步 140 次 GPU 同步）。
5. **B7**：LoRA 参数是 fp32。
6. **A2**：macro 目标覆盖全部 ``[0, M)``，不是只有 0/1。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .swarm import SwarmWrapper, hidden_dim_of, locate_layers


H = 32


def _wrapper(**kw) -> SwarmWrapper:
    base = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, H))
    defaults = dict(hidden_dim=H, num_macro=4, macro_rank=8,
                    num_micro=6, micro_rank=4, param_dtype=torch.float32)
    defaults.update(kw)
    return SwarmWrapper(base, **defaults)


def test_b4_router_not_gt_gated() -> None:
    """前向不得依赖 GT 标签：只给 x，不给 target 也能算出正确形状。"""
    w = _wrapper()
    x = torch.randn(2, 5, H)
    out = w(x)
    assert out.shape == x.shape, out.shape
    # 未调用 _set_target 也不应报错，且 last_* 已被记录
    assert w.last_macro_logits is not None
    assert w.last_micro_topk is not None


def test_b2_train_eval_same_route() -> None:
    """训推路由一致：train()/eval() 下 top-k 索引应完全相同。"""
    w = _wrapper()
    x = torch.randn(3, 7, H)

    w.train()
    w(x)
    train_idx = w.last_micro_topk.clone()

    w.eval()
    with torch.no_grad():
        w(x)
    eval_idx = w.last_micro_topk.clone()

    assert torch.equal(train_idx, eval_idx), (train_idx, eval_idx)


def test_b6_macro_gather_no_stack() -> None:
    """大核汇聚不应物化 [B,T,M,H] 中间张量。

    这里无法直接观测显存，但可以验证等价性：把 macro 权重强制成 one-hot，
    汇聚结果应精确等于对应大核的输出。
    """
    w = _wrapper(num_macro=4)
    x = torch.randn(2, 4, H)

    base = w.base_core(x)
    deltas = [c(x) * w.macro_scale for c in w.macro_cores]
    expect = base + deltas[0]

    # 强制 router 选中第 1 号（index 1，即 macro_loRA 0）
    with torch.no_grad():
        w.router_macro.weight.zero_()
        w.router_macro.weight[1] = 1e3
    out = w(x)
    assert torch.allclose(out, expect, atol=1e-5), (out - expect).abs().max()


def test_b2_vectorized_micro() -> None:
    """向量化的 top-k 求和应与逐专家循环结果一致（正确性等价测试）。"""
    w = _wrapper(num_micro=4, micro_rank=3)
    # 让权重随机但非零，否则 B 矩阵为零看不出差别
    for m in w.micro_pool:
        nn.init.normal_(m.lora_B.weight, std=0.05)
    x = torch.randn(2, 3, H)

    # idx/probs 形状与真实调用一致：[B, T, k]（每个 token 各自的 top-k）
    B, T = x.shape[0], x.shape[1]
    logits = torch.randn(B, T, 4)
    idx, probs = w._topk_micro(logits)
    fast = w.micro_output(x, idx, probs)

    # 参照实现：逐 batch、逐 token、逐 top-k 槽位循环
    slow = torch.zeros_like(x)
    for b in range(x.shape[0]):
        for t in range(x.shape[1]):
            for j in range(idx.shape[-1]):
                e = int(idx[b, t, j])
                p = float(probs[b, t, j])
                # x[b, t] 是 [H]，LoRA 输出也是 [H]，不需要再索引
                slow[b, t] += p * w.micro_pool[e](x[b, t])

    assert torch.allclose(fast, slow, atol=1e-5), (fast - slow).abs().max()


def test_b7_params_are_fp32() -> None:
    """LoRA/router 参数必须是 fp32（B7：bf16 + AdamW 会丢更新）。"""
    w = _wrapper(param_dtype=torch.float32)
    for group, ps in w.param_groups().items():
        assert ps, f"参数组 {group} 为空"
        for p in ps:
            assert p.dtype == torch.float32, f"{group} 的参数是 {p.dtype}"


def test_a2_macro_targets_cover_all() -> None:
    """macro 目标必须能取到全部 [0, M)，而不是只有 0/1（A2）。"""
    from ..schema import ArchProfile, default_domain_to_macro
    from ..dataset import resolve_targets

    arch = ArchProfile(num_macro_cores=4, macro_names=list("ABCD"))
    mapping = default_domain_to_macro()
    targets = {resolve_targets([{"domain": d}], arch)[0].macro
               for d in mapping}
    assert targets == {0, 1, 2, 3}, targets


def test_scale_hot_tuning() -> None:
    """热调超参不应需要重建模型。"""
    w = _wrapper()
    w.set_scales(micro_scale=0.05, micro_top_k=3)
    assert w.micro_scale == 0.05
    assert w.micro_top_k == 3
    x = torch.randn(2, 4, H)
    w(x)   # 不应报错

    try:
        w.set_scales(micro_top_k=99)
    except ValueError:
        pass
    else:
        raise AssertionError("top_k 越界应抛 ValueError")


def test_router_losses_and_balance() -> None:
    """辅助损失可计算；负载均衡项能区分均匀路由与坍缩路由。

    ``lb`` 现在是**大核 + 小核**两项之和，所以均衡下界是 2.0
    （每项各贡献 1.0），而不是原来的 1.0。
    """
    w = _wrapper(num_macro=4, num_micro=4, micro_top_k=2)
    x = torch.randn(3, 6, H)
    w(x)

    mask = torch.ones(3, 6, dtype=torch.bool)
    zeros = torch.zeros(3, dtype=torch.long)
    m_ce, n_ce, lb = w.router_losses(mask, zeros, zeros)
    assert torch.isfinite(m_ce) and torch.isfinite(n_ce) and torch.isfinite(lb)
    # 均衡时 micro≈1、macro≈1
    assert float(lb) >= 2.0 - 0.1, float(lb)

    # 强制所有 token 都选同一个专家 -> 坍缩，负载均衡损失必须显著变大
    with torch.no_grad():
        w.router_micro.weight.zero_()
        w.router_micro.weight[0] = 1e4        # 0 号专家 logits 极大
    w.eval()
    with torch.no_grad():
        w(x)
        _, _, lb_collapsed = w.router_losses(mask, zeros, zeros)
    assert float(lb_collapsed) > float(lb), (float(lb), float(lb_collapsed))


def test_balance_loss_has_gradient() -> None:
    """均衡损失必须真的能回传梯度。

    ★这是个**静默失效**的坑：dispatch 占比 ``f`` 来自 argmax/top-k，
    本来就不可微，Switch 损失的梯度**完全**来自 P 项。
    如果把概率 ``.detach()`` 掉，整个 lb 就退化成常量、
    ``grad is None`` / 梯度为 0——指标照样打印、看着"在算"，
    但对 router 毫无约束，专家照样一路坍缩到单一专家。

    本项目真实踩过：detach 之后 35 层大核分布全部塌成一个类
    （Arts_Anchor 只剩 0.029，而数据本身是均衡的 0.25）。
    """
    for num_macro, num_micro in ((4, 8), (4, 4)):
        w = _wrapper(num_macro=num_macro, num_micro=num_micro,
                     micro_top_k=2)
        x = torch.randn(3, 6, H, requires_grad=True)
        w(x)
        mask = torch.ones(3, 6, dtype=torch.bool)
        zeros = torch.zeros(3, dtype=torch.long)
        _, _, lb = w.router_losses(mask, zeros, zeros)
        lb.backward()

        assert w.router_macro.weight.grad is not None, \
            f"M={num_macro}: 大核 router 没有梯度"
        gm = float(w.router_macro.weight.grad.norm())
        assert gm > 1e-8, f"M={num_macro}: 大核 router 梯度为 0（{gm}）"

        assert w.router_micro.weight.grad is not None, \
            f"N={num_micro}: 小核 router 没有梯度"
        gn = float(w.router_micro.weight.grad.norm())
        assert gn > 1e-8, f"N={num_micro}: 小核 router 梯度为 0（{gn}）"

    # 只用均衡损失（不掺 CE）也必须非零——否则说明梯度全靠 CE 撑着
    w = _wrapper(num_macro=4, num_micro=4, micro_top_k=2)
    x = torch.randn(3, 6, H)
    w(x)
    mask = torch.ones(3, 6, dtype=torch.bool)
    from .swarm import _dense_balance, _switch_balance
    lb_only = (_switch_balance(w.last_micro_topk,
                               w.last_micro_probs.float(), 4)
               + _dense_balance(w.last_macro_probs.float(), 4))
    lb_only.backward()
    assert float(w.router_macro.weight.grad.norm()) > 0
    assert float(w.router_micro.weight.grad.norm()) > 0


def test_macro_router_also_balanced() -> None:
    """大核也必须受均衡约束。

    只给小核加均衡项时，把路由权重调大后大核分布会从均匀塌到
    0.02 / 0.55 —— 32 个领域映射到 4 个大核，塌一个就等于 1/4
    的领域失去专属容量。因此大核塌缩也必须让 lb 变大。
    """
    w = _wrapper(num_macro=4, num_micro=4, micro_top_k=2)
    x = torch.randn(3, 6, H)
    mask = torch.ones(3, 6, dtype=torch.bool)
    zeros = torch.zeros(3, dtype=torch.long)

    w.eval()
    with torch.no_grad():
        w(x)
        _, _, lb_ok = w.router_losses(mask, zeros, zeros)

    # router 是无 bias 的 Linear，logit = W·x。要保证**每个** token 都
    # 涌向大核 #1，x 必须全正、W[1] 取常数正向量（否则 x 有正有负时
    # logit_1 也会跟着变号，只能塌一半）。
    x_pos = torch.ones_like(x)
    with torch.no_grad():
        w.router_macro.weight.zero_()
        w.router_macro.weight[1] = 1e4
        w(x_pos)
        _, _, lb_bad = w.router_losses(mask, zeros, zeros)

    hits = w.last_macro_probs.argmax(-1).reshape(-1)
    assert int((hits == 1).sum()) == hits.numel(), "构造未造成完全坍缩"

    # 大核项在坍缩时贡献 N=4（均衡时 1），净增约 3
    assert float(lb_bad) - float(lb_ok) > 2.0, (float(lb_ok), float(lb_bad))


def test_locate_layers_and_hidden() -> None:
    """层定位与 hidden 推断。"""

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.layers = nn.ModuleList(
                [type("L", (nn.Module,), {})() for _ in range(4)])
            self.config = type("C", (), {"hidden_size": H})()

    f = Fake()
    ls = locate_layers(f)
    assert len(ls) == 4, len(ls)
    assert hidden_dim_of(f, ls) == H


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)