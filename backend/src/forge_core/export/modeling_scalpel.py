"""发布包入口：``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``。

**本模块必须保持零相对导入。** 见 :mod:`forge_core.export.telemetry` 的说明。

本文件只做三件事：加载底座、把 swarm 权重装回、转发 ``forward``。
路由实现**不复制到这里**——它来自随包发布的 ``swarm_forge.py``，
那是 :mod:`forge_core.modeling.swarm` 的**逐字节副本**。

为什么必须是副本而不是切片
--------------------------
旧实现在导出时读自己的源码，用
``open(__file__).read().split('# ------')[2]`` 之类按注释横线计数切段。
这意味着一旦**任何一处**注释横线增删，切出来的就是错的代码块，
而且不报错——它会切出一个语法合法但语义不同的片段。
:func:`forge_core.export.bundle.build_bundle` 现在直接 ``shutil.copy``
真实模块，并用 :mod:`forge_core.export.selftest` 验证副本与训练期
前向输出**逐位一致**。
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from configuration_scalpel import ScalpelUniversalConfig
from swarm_forge import (  # noqa: F401  (随包发布的逐字节副本)
    ScalpelLoRA,
    SwarmWrapper,
    hidden_dim_of,
    locate_layers,
    strip_multimodal,
    wrap_layers,
)
from telemetry import TelemetryTracker


class ScalpelUniversalForCausalLM(PreTrainedModel):
    """底座 + 已训练的 swarm 专家层。"""

    config_class = ScalpelUniversalConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = []

    def __init__(self, config: ScalpelUniversalConfig) -> None:
        super().__init__(config)
        self.model = None
        self.telemetry = TelemetryTracker(config.num_macro_cores,
                                          config.num_micro_experts)

    # -- 构造 ---------------------------------------------------------
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args,
                        **kwargs) -> "ScalpelUniversalForCausalLM":
        config = kwargs.pop("config", None)
        if config is None:
            config = ScalpelUniversalConfig.from_pretrained(
                pretrained_model_name_or_path, **{
                    k: v for k, v in kwargs.items()
                    if k in ("cache_dir", "revision", "local_files_only",
                             "proxies", "token", "trust_remote_code")})
        config.validate()

        dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", torch.bfloat16))
        device_map = kwargs.pop("device_map", None)
        sw_path = kwargs.pop("swarm_weights", None)

        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, dtype=dtype,
            device_map=device_map, trust_remote_code=True)
        if config.text_only:
            strip_multimodal(base)
        base.config.use_cache = False

        layers = locate_layers(base)
        # hidden_dim 以 checkpoint / config 为准，不从层结构反推
        wrappers = wrap_layers(
            base, layers,
            num_macro=config.num_macro_cores,
            num_micro=config.num_micro_experts,
            macro_rank=config.macro_rank,
            micro_rank=config.micro_rank,
            micro_scale=config.micro_scale,
            macro_scale=config.macro_dense_scale,
            micro_top_k=config.micro_top_k,
            hidden_dim=config.hidden_dim,
            param_dtype=torch.float32,
            device=next(base.parameters()).device,
        )

        inst = cls(config)
        inst.model = base
        inst.wrappers = wrappers

        if sw_path is None:
            sw_path = (f"{pretrained_model_name_or_path.rstrip('/')}"
                       f"/swarm_weights.pt")
        inst.load_swarm_weights(sw_path)
        inst.eval()
        return inst

    # -- 权重 ---------------------------------------------------------
    def load_swarm_weights(self, path: str) -> None:
        """从 ``save_checkpoint`` 的产物恢复 router / LoRA。

        只加载 wrapper 自己的参数，底座保持原样。
        """
        sd = torch.load(path, map_location="cpu", weights_only=False)
        meta = sd.get("__meta__", {})
        fmt = meta.get("format", "")
        if not fmt.startswith("forge-swarm-"):
            raise ValueError(
                f"权重格式不认得：{fmt!r}。期望 'forge-swarm-v2'。"
                "旧版 scalpel_forge.py 的检查点无法直接复用。")

        arch = meta.get("arch", {})
        m = int(arch.get("num_macro_cores", self.config.num_macro_cores))
        n = int(arch.get("num_micro_experts", self.config.num_micro_experts))
        if (m, n) != (self.config.num_macro_cores, self.config.num_micro_experts):
            raise ValueError(
                f"权重形状 (M={m}, N={n}) 与配置 "
                f"(M={self.config.num_macro_cores}, "
                f"N={self.config.num_micro_experts}) 不符")

        loaded = 0
        for i, w in enumerate(self.wrappers):
            for j, core in enumerate(w.macro_cores):
                key = f"layer{i}.macro{j}"
                if key in sd:
                    core.load_state_dict(sd[key]); loaded += 1
            for j, expert in enumerate(w.micro_pool):
                key = f"layer{i}.micro{j}"
                if key in sd:
                    expert.load_state_dict(sd[key]); loaded += 1
            w.router_macro.load_state_dict(sd[f"layer{i}.router_macro"])
            w.router_micro.load_state_dict(sd[f"layer{i}.router_micro"])
            loaded += 2
        self._n_loaded = loaded

    # -- 转发 ---------------------------------------------------------
    def forward(self, *args, **kwargs) -> Any:
        out = self.model(*args, **kwargs)
        attn = kwargs.get("attention_mask")
        ids = kwargs.get("input_ids")
        if attn is None and ids is not None:
            attn = torch.ones_like(ids)
        if attn is not None and self.wrappers:
            mask = attn.bool() if attn.dtype == torch.bool else attn == 1
            self.telemetry.record(self.wrappers[0].last_macro_probs,
                                  self.wrappers[0].last_micro_topk, mask)
        return out

    def generate(self, *args, **kwargs) -> Any:
        self.model.config.use_cache = True
        try:
            return self.model.generate(*args, **kwargs)
        finally:
            self.model.config.use_cache = False

    # ``nn.Module`` 要求同类型模块暴露一致的 device / dtype 属性
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()
