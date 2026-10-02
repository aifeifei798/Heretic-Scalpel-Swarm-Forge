"""模型装配层。

只导出真正被外部使用的符号；``selftest`` 供 ``python -m`` 直接跑。
"""

from .swarm import (
    ScalpelLoRA,
    SwarmWrapper,
    hidden_dim_of,
    locate_layers,
    strip_multimodal,
    text_config,
    wrap_layers,
)

__all__ = [
    "ScalpelLoRA", "SwarmWrapper", "locate_layers", "wrap_layers",
    "hidden_dim_of", "text_config", "strip_multimodal",
]