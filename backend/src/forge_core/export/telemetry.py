"""路由遥测：发布包运行时用来观察专家是否坍缩。

**本模块必须保持零相对导入**（只依赖 torch + 标准库）。
发布包会把 :mod:`forge_core.export` 下的模块原样拷贝到模型仓库根目录，
由 HF 的 ``trust_remote_code`` 机制通过 ``sys.path`` 解析，
因此不能 ``from .xxx import``，也不能依赖 ``forge_core`` 包本身。

一旦需要跨模块引用，请把符号复制进来而不是改成相对导入——
否则发布包会在用户 ``from_pretrained`` 时直接 ImportError。
"""

from __future__ import annotations

from typing import Any

import torch


class TelemetryTracker:
    """累计大核权重与小核命中次数。

    用途：训练/推理结束后判断路由是否坍缩。
    判据——``macro_dist`` 任一项显著低于 1/M、或 ``dead > 0``，
    就说明有专家既没被选中也没学到东西。

    刻意不用 ``nn.Module``：它是纯观测器，不该出现在 ``state_dict()`` 里，
    否则保存出来的权重会带上运行时垃圾，破坏可复现性。
    """

    def __init__(self, num_macro: int, num_micro: int = 0) -> None:
        self.num_macro = int(num_macro)
        self.num_micro = int(num_micro)
        self.reset()

    def reset(self) -> None:
        self.macro_hits = torch.zeros(self.num_macro, dtype=torch.float64)
        self.micro_hits = torch.zeros(self.num_micro, dtype=torch.float64)
        self.macro_weight_sum = torch.zeros(self.num_macro, dtype=torch.float64)
        self.tokens = 0

    # -- 记录 ---------------------------------------------------------
    @torch.no_grad()
    def record(self, macro_probs: torch.Tensor,
               micro_topk: torch.Tensor | None = None,
               mask: torch.Tensor | None = None) -> None:
        """累计一次前向。

        Parameters
        ----------
        macro_probs: ``[B, T, M]`` 大核 dense 概率
        micro_topk:  ``[B, T, k]`` 小核 top-k 索引
        mask:        ``[B, T]`` bool，padding 位不计入
        """
        p = macro_probs.detach().float()
        if mask is not None:
            m = mask.unsqueeze(-1)
            self.macro_weight_sum += (p * m).sum(dim=(0, 1)).double().cpu()
            n_tok = int(mask.sum())
            sel = p.argmax(-1)[mask]
        else:
            self.macro_weight_sum += p.sum(dim=(0, 1)).double().cpu()
            n_tok = p.shape[0] * p.shape[1]
            sel = p.argmax(-1).reshape(-1)
        self.tokens += n_tok

        self.macro_hits += torch.bincount(
            sel.reshape(-1).cpu(), minlength=self.num_macro).double()

        if micro_topk is not None and self.num_micro:
            k = micro_topk.detach()
            if mask is not None:
                k = k[mask]
            self.micro_hits += torch.bincount(
                k.reshape(-1).cpu(), minlength=self.num_micro).double()

    @torch.no_grad()
    def record_dense(self, macro_probs: torch.Tensor,
                     micro_topk: torch.Tensor) -> None:
        """全位置计入的快捷入口（生成时每个位置都是有效路由决策）。

        与 :meth:`record` 的区别只是不需要构造 padding mask ——
        hook 里逐调用现造一个 ``[B,T]`` 的全 True 张量纯属浪费。
        """
        self.record(macro_probs, micro_topk, mask=None)

    # -- 汇总 ---------------------------------------------------------
    def summary(self, macro_names: list[str] | None = None) -> dict[str, Any]:
        n = max(1, self.tokens)
        names = macro_names or [f"M{i}" for i in range(self.num_macro)]

        m = (self.macro_hits / n).tolist()
        ms = sum(m) or 1.0
        micro = (self.micro_hits / n).tolist()
        ss = sum(micro) or 1.0

        return {
            "tokens": self.tokens,
            "macro_dist": {name: round(v / ms, 5)
                           for name, v in zip(names, m)},
            "micro_dist": [round(v / ss, 5) for v in micro],
            "macro_dead": int((self.macro_hits == 0).sum()),
            "micro_dead": int((self.micro_hits == 0).sum()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {"tokens": self.tokens,
                "macro_hits": self.macro_hits.tolist(),
                "micro_hits": self.micro_hits.tolist()}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.tokens = int(sd.get("tokens", 0))
        self.macro_hits = torch.tensor(sd.get("macro_hits", []),
                                       dtype=torch.float64)
        self.micro_hits = torch.tensor(sd.get("micro_hits", []),
                                       dtype=torch.float64)
