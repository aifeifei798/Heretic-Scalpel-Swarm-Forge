"""推理 / 对话。

为什么不直接 ``from_pretrained`` 发布包
--------------------------------------
发布包（:mod:`forge_core.export`）是给**外部使用者**的：它只暴露
``generate``，路由超参烧在 ``config.json`` 里。

而调优时我们需要的东西它给不了：

* 逐 token 看路由分布，判断某个回答到底走了哪些专家；
* 不重启就改 ``micro_scale`` / ``macro_scale`` / ``micro_top_k``
  （:meth:`SwarmWrapper.set_scales`）;
* 对比"加载 swarm 权重"与"纯底座"的输出差异，确认适配器真的在起作用。

所以这里直接复用训练期的装配路径（:func:`forge_core.train.build_model`），
把 checkpoint 灌进去，再自己管 KV cache 与采样。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------
def _arch_from_conf(conf: dict) -> "ArchProfile":
    """从 bundle 的 config.json 还原 ArchProfile。

    发布包里的字段可能没有 ``domain_to_macro``——它不是推理期必须的，
    推理时路由已经在 wrapper 里硬编码。缺了就用默认均衡映射。
    """
    from .schema import ArchProfile

    kw: dict = dict(
        num_macro_cores=conf["num_macro_cores"],
        macro_names=conf.get("macro_names")
        or [f"M{i}" for i in range(conf["num_macro_cores"])],
        num_micro_experts=conf["num_micro_experts"],
        macro_rank=conf.get("macro_rank", 64),
        micro_rank=conf.get("micro_rank", 16),
    )
    if conf.get("domain_to_macro"):
        kw["domain_to_macro"] = conf["domain_to_macro"]
    return ArchProfile(**kw)


@dataclass
class GenStats:
    prompt_tokens: int = 0
    new_tokens: int = 0
    seconds: float = 0.0

    @property
    def tok_per_sec(self) -> float:
        return self.new_tokens / self.seconds if self.seconds > 0 else 0.0


class SwarmChat:
    """底座 + 已训练的 swarm 专家层。"""

    def __init__(self, model, wrappers, tok, cfg) -> None:
        self.model = model
        self.wrappers = wrappers
        self.tok = tok
        self.cfg = cfg
        self.stats = GenStats()

    # -- 构造 ---------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, cfg, checkpoint: str | Path, *,
                        device: str | None = None) -> "SwarmChat":
        from .dataset import load_tokenizer
        from .train import build_model, load_swarm_weights

        if device:
            cfg.train.device = device
        ctx = build_model(cfg, train=False)
        n, meta = load_swarm_weights(checkpoint, ctx.wrappers)

        tok = load_tokenizer(cfg.base_model_id,
                             trust_remote_code=cfg.trust_remote_code)
        self = cls(ctx.model, ctx.wrappers, tok, cfg)
        self.loaded = n
        self.meta = meta
        return self

    @classmethod
    def from_bundle(cls, bundle_dir: str | Path, *,
                    device: str | None = None) -> "SwarmChat":
        """从**发布包目录**装载 —— 走的正是用户 ship 的同一条加载路径。

        与 :meth:`from_checkpoint` 的关键区别：
        后者直接拿训练期的原始 checkpoint，绕过发布包的装配代码；
        前者走 ``AutoModelForCausalLM.from_pretrained(trust_remote_code=True)``
        ——发布包里的 ``ScalpelUniversalForCausalLM``。

        所以试跑打包模型时，用户测的 就是 ship 的东西。
        如果发布包的导出路径出 bug（比如代码切片、软截断漏掉），
        ``from_checkpoint`` 永远发现不了，只有 ``from_bundle`` 能暴露。
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        bundle_dir = Path(bundle_dir)
        if not bundle_dir.is_dir():
            raise ValueError(f"不是发布包目录：{bundle_dir}")

        # 用 ScalpelUniversalConfig 读回元信息，拼一个最小可用的 ForgeConfig
        import json as _json

        conf = _json.loads((bundle_dir / "config.json").read_text("utf-8"))

        from .schema import ArchProfile, ForgeConfig, TrainConfig

        cfg = ForgeConfig(
            project_name=conf.get("name", "from-bundle"),
            base_model_id=conf["base_model_name_or_path"],
            data_path="",
            arch=_arch_from_conf(conf),
            train=TrainConfig(micro_top_k=conf.get("micro_top_k", 2)),
        )

        tok = AutoTokenizer.from_pretrained(
            conf["base_model_name_or_path"],
            trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(bundle_dir), trust_remote_code=True,
            device_map=device or "auto",
            dtype=torch.bfloat16)

        # 这些与训练期的 wrapper 是同一个类型，方法完全一致：
        # set_scales / last_macro_logits / last_micro_topk 都可用
        wrappers = list(getattr(model, "wrappers", []) or [])
        if not wrappers:
            # 发布包在某些加载顺序下可能没挂 wrappers，兜底定位
            from .modeling import locate_layers
            inner = getattr(model, "model", None)
            if inner is not None:
                layers = locate_layers(inner)
                wrappers = [layer.mlp for layer in layers
                            if hasattr(layer.mlp, "last_macro_logits")]

        self = cls(model, wrappers, tok, cfg)
        self.loaded = sum(
            len(w.macro_cores) + len(w.micro_pool) + 2 for w in wrappers)
        self.meta = {
            "hidden_dim": conf.get("hidden_dim"),
            "step": None,
            "source": "bundle:" + bundle_dir.name,
        }
        return self

    # -- 热调 ---------------------------------------------------------
    def set_scales(self, *, micro_scale: float | None = None,
                   macro_scale: float | None = None,
                   micro_top_k: int | None = None) -> None:
        """不重启就改推理超参。"""
        for w in self.wrappers:
            w.set_scales(micro_scale=micro_scale, macro_scale=macro_scale,
                         micro_top_k=micro_top_k)

    # -- 路由统计 -----------------------------------------------------
    def route_of_last(self) -> dict[str, Any]:
        """上一次前向里，第一层 router 的选择分布。

        只看第一层：它是唯一在**每一层都在跑**的那一层，
        而 :class:`RouteStats` 那种全层累加会把 35 层的选择混在一起，
        反而看不出"这个回答走了哪个专家"。
        """
        from .train import RouteStats

        if not self.wrappers:
            return {}
        w = self.wrappers[0]
        st = RouteStats.zeros(self.cfg.arch, "cpu", self.cfg.train.micro_top_k)
        st.tokens = 1
        macro = w.last_macro_logits.argmax(-1).reshape(-1).cpu()
        micro = w.last_micro_topk.reshape(-1).cpu()
        st.macro_hits.scatter_add_(0, macro, torch.ones_like(macro, dtype=torch.float))
        st.micro_hits.scatter_add_(0, micro, torch.ones_like(micro, dtype=torch.float))
        st.tokens = int(macro.numel())
        return st.summary()

    # -- 生成 ---------------------------------------------------------
    def _ids(self, messages: Sequence[dict[str, str]], device: str):
        text = self.tok.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}

    @torch.inference_mode()
    def generate(self, messages: Sequence[dict[str, str]], *,
                 max_new_tokens: int = 256,
                 temperature: float = 0.0,
                 top_p: float = 0.95,
                 seed: int | None = None) -> tuple[str, GenStats]:
        """非流式生成。

        ``temperature=0`` 即贪心解码。
        """
        device = next(self.model.parameters()).device
        enc = self._ids(messages, str(device))
        self.stats = GenStats(prompt_tokens=int(enc["input_ids"].shape[1]))

        do_sample = temperature > 0
        if seed is not None:
            torch.manual_seed(seed)

        t0 = time.time()
        out = self.model.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        self.stats.seconds = time.time() - t0
        new = out[0][self.stats.prompt_tokens:]
        self.stats.new_tokens = int(new.numel())
        return self.tok.decode(new, skip_special_tokens=True), self.stats

    @torch.inference_mode()
    def stream(self, messages: Sequence[dict[str, str]], *,
               max_new_tokens: int = 256,
               temperature: float = 0.0,
               top_p: float = 0.95,
               seed: int | None = None) -> Iterator[str]:
        """流式生成，逐段 yield 文本。

        用 ``TextIteratorStreamer`` 把 token 流翻译成文本流，
        这样调用方可以边收边打印，而不必等整段生成完。
        """
        from threading import Thread

        from transformers import TextIteratorStreamer

        device = next(self.model.parameters()).device
        enc = self._ids(messages, str(device))
        self.stats = GenStats(prompt_tokens=int(enc["input_ids"].shape[1]))

        do_sample = temperature > 0
        if seed is not None:
            torch.manual_seed(seed)

        streamer = TextIteratorStreamer(
            self.tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
            streamer=streamer,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p)

        t0 = time.time()
        err: list[BaseException] = []

        def run() -> None:
            try:
                self.model.generate(**kwargs)
            except BaseException as e:          # noqa: BLE001
                err.append(e)
                streamer.end()

        th = Thread(target=run, daemon=True)
        th.start()
        n = 0
        for piece in streamer:
            if piece:
                n += len(self.tok.encode(piece, add_special_tokens=False))
                yield piece
        th.join()
        self.stats.seconds = time.time() - t0
        self.stats.new_tokens = n
        if err:
            raise err[0]

    # -- 便捷 ---------------------------------------------------------
    def ask(self, question: str, *, system: str | None = None,
            history: list[dict[str, str]] | None = None,
            **kw) -> tuple[str, GenStats]:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs += history or []
        msgs.append({"role": "user", "content": question})
        return self.generate(msgs, **kw)

    def compare_with_base(self, messages: Sequence[dict[str, str]],
                          **kw) -> dict[str, Any]:
        """关掉所有适配器，跑一遍纯底座，用于对照。

        这能回答一个关键问题：**权重到底起作用了没有？**
        如果关掉 scale 后输出几乎一样，说明 LoRA 学到的东西太弱，
        或者 scale 被热调成 0 了 —— 光看 loss 曲线是发现不了的。
        """
        prev = [(w.micro_scale, w.macro_scale, w.micro_top_k)
                for w in self.wrappers]
        try:
            for w in self.wrappers:
                w.set_scales(micro_scale=0.0, macro_scale=0.0)
            base_text, base_st = self.generate(messages, **kw)
        finally:
            for w, (ms, ms2, k) in zip(self.wrappers, prev):
                w.set_scales(micro_scale=ms, macro_scale=ms2, micro_top_k=k)
        swarm_text, swarm_st = self.generate(messages, **kw)
        return {
            "base": base_text,
            "swarm": swarm_text,
            "identical": base_text.strip() == swarm_text.strip(),
            "base_tok_per_sec": base_st.tok_per_sec,
            "swarm_tok_per_sec": swarm_st.tok_per_sec,
        }


__all__ = ["SwarmChat", "GenStats"]
