"""发布包入口：``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``。

★同包模块**必须用相对导入**（``from .swarm_forge import ...``）
--------------------------------------------------------------
这是被真实加载路径逼出来的，不是风格问题。

``trust_remote_code`` 的加载流程是：
1. 把 ``modeling_scalpel.py`` 拷进
   ``~/.cache/huggingface/modules/transformers_modules/<repo>/``；
2. **静态扫描**该文件的 import，逐个 ``importlib.import_module()``
   确认依赖存在（``dynamic_module_utils.check_imports``）；
3. 扫描到的**相对导入**目标会被一并复制进同一目录
   （``dynamic_module_utils`` 里对 ``module_needed`` 的 ``shutil.copyfile``）。

所以：
* 用绝对导入（``from swarm_forge import ...``）→ 第 2 步就炸，
  报 ``ImportError: This modeling file requires the following packages
  that were not found in your environment: swarm_forge``，
  并荒谬地建议你去 ``pip install swarm_forge``。
* 用相对导入 → 第 3 步自动把同级文件带过去，import 成立。

我最初写的是绝对导入，并且还在自检里断言"不得含相对导入"——
那条断言把错误认知固化了下来，等于主动给这个 bug 上了一道锁。

本文件只做三件事：加载底座、把 swarm 权重装回、转发 ``forward``。
路由实现**不复制到这里**——它来自随包发布的 ``swarm_forge.py``，
那是 :mod:`forge_core.modeling.swarm` 的**逐字节副本**。
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from .configuration_scalpel import ScalpelUniversalConfig
from .swarm_forge import (  # noqa: F401  (随包发布的逐字节副本)
    ScalpelLoRA,
    SwarmWrapper,
    hidden_dim_of,
    locate_layers,
    strip_multimodal,
    wrap_layers,
)
from .telemetry import TelemetryTracker


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
        inst._attach_telemetry()

        if sw_path is None:
            sw_path = (f"{pretrained_model_name_or_path.rstrip('/')}"
                       f"/swarm_weights.pt")
        inst.load_swarm_weights(sw_path)
        inst.eval()
        return inst

    def _attach_telemetry(self) -> None:
        """用 forward hook 采集路由分布。

        ★不能靠在 :meth:`forward` 里读 ``kwargs``。
        ``generate()`` 内部是以**位置参数**调用子模块的
        （``self.model.generate`` -> decoder -> layer -> mlp），
        根本不会经过本类重写的 ``forward``；就算经过，
        ``kwargs`` 里也未必有 ``attention_mask``。

        结果就是遥测静默地全是 0 —— 指标看起来"在工作"，
        实则一个数都没采到。所以这里在**每个 SwarmWrapper** 上挂
        forward hook：wrapper 一定会被调用，且它自己就缓存了
        上一轮的 ``last_macro_probs`` / ``last_micro_topk``。

        口径说明：``tokens`` 累加的是每次前向 ``x.shape[1]``，
        即**路由决策的位置数**。带 KV cache 生成时每步只喂 1 个
        新 token，所以这个数不等于 prompt+生成的总 token 数，
        但它正是"router 做了多少次决策"，也正是判断专家是否被
        使用的正确口径。
        """
        for w in self.wrappers:
            def hook(mod, args, _out, _self=self):
                probs = getattr(mod, "last_macro_probs", None)
                topk = getattr(mod, "last_micro_topk", None)
                if probs is None or topk is None:
                    return
                _self.telemetry.record_dense(probs.detach().float().cpu(),
                                             topk.detach().cpu())

            w.register_forward_hook(hook)

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
        # 遥测由 :meth:`_attach_telemetry` 挂的 forward hook 负责，
        # 这里只做转发 —— generate() 是以位置参数一路调到
        # layer.mlp 的，根本不会回到本类。
        return self.model(*args, **kwargs)

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
