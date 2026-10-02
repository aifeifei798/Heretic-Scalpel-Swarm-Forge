"""数据集生成子系统。

    from forge_core.datagen import build_dataset, DataGenConfig
    report = build_dataset(DataGenConfig(samples_per_domain=200))
"""

from .builder import BuildReport, build_dataset, length_stats, preview
from .perturb import Slot, SlotValue, Template, sample_domain
from .sanitize import is_clean, sanitize_text
from .seeds import DOMAIN_TEMPLATES

__all__ = [
    "BuildReport", "build_dataset", "preview", "length_stats",
    "Slot", "SlotValue", "Template", "sample_domain",
    "sanitize_text", "is_clean", "DOMAIN_TEMPLATES",
]
