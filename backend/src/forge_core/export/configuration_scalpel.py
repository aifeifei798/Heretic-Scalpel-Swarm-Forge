"""发布包的配置类。

**本模块必须保持零相对导入。** 见 :mod:`forge_core.export.telemetry` 的说明：
发布包会把这些模块原样拷贝到模型仓库根目录，由 ``trust_remote_code``
经 ``sys.path`` 解析，因此不能用 ``from .xxx import``。

与训练期结构的对应关系
---------------------
``hidden_dim`` **必须**从训练 checkpoint 的 ``__meta__`` 读取，
绝不能在这里猜一个默认值——底座换模型后 hidden_dim 变了，
猜出来的值会在 ``from_pretrained`` 时静默错配，
直到第一次前向才炸出 shape 不匹配。
"""

from __future__ import annotations

from transformers import PretrainedConfig


class ScalpelUniversalConfig(PretrainedConfig):
    """描述一个"底座 + swarm 权重"的组合模型。"""

    model_type = "scalpel_universal"
    attribute_map = {
        "num_macro_cores": "num_macro_cores",
        "num_micro_experts": "num_micro_experts",
    }

    def __init__(
        self,
        base_model_name_or_path: str = "",
        num_macro_cores: int = 4,
        num_micro_experts: int = 32,
        macro_rank: int = 64,
        micro_rank: int = 16,
        micro_top_k: int = 2,
        micro_scale: float = 0.02,
        macro_dense_scale: float = 0.1,
        hidden_dim: int = 0,
        macro_names: list[str] | None = None,
        text_only: bool = True,
        **kwargs,
    ) -> None:
        self.base_model_name_or_path = base_model_name_or_path
        self.num_macro_cores = int(num_macro_cores)
        self.num_micro_experts = int(num_micro_experts)
        self.macro_rank = int(macro_rank)
        self.micro_rank = int(micro_rank)
        self.micro_top_k = int(micro_top_k)
        self.micro_scale = float(micro_scale)
        self.macro_dense_scale = float(macro_dense_scale)
        self.hidden_dim = int(hidden_dim)
        self.macro_names = list(macro_names or
                                [f"M{i}" for i in range(num_macro_cores)])
        self.text_only = bool(text_only)
        super().__init__(**kwargs)

    def validate(self) -> None:
        super().validate()
        if self.num_macro_cores < 1:
            raise ValueError(
                f"num_macro_cores 必须 >= 1（第 0 号恒为只读底座），"
                f"当前 {self.num_macro_cores}")
        if self.hidden_dim <= 0:
            raise ValueError(
                "hidden_dim 未设置。导出时必须从 checkpoint 的 "
                "__meta__.hidden_dim 读取，不能依赖默认值。")
        if not 1 <= self.micro_top_k <= self.num_micro_experts:
            raise ValueError(
                f"micro_top_k={self.micro_top_k} 超出范围 "
                f"[1, {self.num_micro_experts}]")
        if len(self.macro_names) != self.num_macro_cores:
            raise ValueError(
                f"macro_names 有 {len(self.macro_names)} 个，"
                f"但 num_macro_cores={self.num_macro_cores}")
