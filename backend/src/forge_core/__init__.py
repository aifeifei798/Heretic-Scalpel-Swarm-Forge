"""Heretic-Scalpel Swarm Forge — 核心库。

设计边界（务必保持）：
  * ``forge_core`` 绝不 import fastapi / uvicorn —— 纯 ML，可独立单测与被 CLI 调用。
  * ``forge_api``（P2 阶段建立）绝不 import torch —— 只做编排，显存占用恒为 0。

子模块：
    schema      pydantic 配置模型（唯一配置真相源）
    datagen     模板 + 槽位扰动的数据集生成引擎
    dataset     tokenize / assistant-mask / 长度分桶
    routes      domain -> macro / micro 目标映射
    modeling    LoRA / swarm 包装 / 定位 / 遥测 / HF 适配
    train / infer / export / publish   四大动作
    cli         NDJSON 结构化日志的命令行入口
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
