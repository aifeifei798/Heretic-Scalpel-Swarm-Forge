"""把训练产物打成一个可直接 ``from_pretrained`` 的发布包。

关键设计：**拷贝真实模块，不切源码**
----------------------------------------
旧实现（``scalpel_forge.py`` 的 ``action_export``）读自己的源码，
用 ``src.split('# ------')[2]`` 按注释横线计数切段再拼进
``modeling_scalpel.py``。这有三处硬伤：

1. 改任何一处注释横线，切出来的就是错的代码块；
2. 切错不报错——语法往往仍然合法，只是语义变了；
3. 切出来的代码带着原文件的 imports 与上下文，依赖隐式约定。

这里改为把下列模块**逐字节**拷进发布目录：

======================  ==========================================
发布文件名              来源
======================  ==========================================
``swarm_forge.py``      ``forge_core.modeling.swarm``（路由实现本体）
``telemetry.py``        ``forge_core.export.telemetry``
``configuration_scalpel.py``  ``forge_core.export.configuration_scalpel``
``modeling_scalpel.py`` ``forge_core.export.modeling_scalpel``
======================  ==========================================

因此"发布包的路由"与"训练时的路由"是**同一份代码**，
等价性由 :mod:`forge_core.export.selftest` 断言（前向逐位一致），
而不是靠人肉维护两套实现。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..schema import ForgeConfig

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"待拷贝的模块不存在：{src}")
    shutil.copy(src, dst)


def build_bundle(cfg: ForgeConfig, checkpoint: str | Path,
                 out_dir: str | Path) -> dict[str, Any]:
    """生成发布包，返回其元信息。

    Parameters
    ----------
    cfg:        训练用的配置（提供 base_model / arch / train 超参）
    checkpoint: :func:`forge_core.train.save_checkpoint` 的产物
    out_dir:    发布目录（会被覆盖）

    Notes
    -----
    ``hidden_dim`` 与 arch 一律从 checkpoint 的 ``__meta__`` 读取。
    绝不从 config 默认值猜：底座一换模型 hidden_dim 就变，
    猜出来的值只会在用户第一次 ``from_pretrained`` 时炸 shape 不匹配。
    """
    import torch

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"检查点不存在：{ckpt_path}")

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # --- 代码：逐字节拷贝 -------------------------------------------
    _copy(_PKG / "modeling" / "swarm.py", out / "swarm_forge.py")
    _copy(_HERE / "telemetry.py", out / "telemetry.py")
    _copy(_HERE / "configuration_scalpel.py", out / "configuration_scalpel.py")
    _copy(_HERE / "modeling_scalpel.py", out / "modeling_scalpel.py")

    # --- 权重：只搬 swarm 部分，底座靠 base_model_name_or_path 拉 ---
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = sd.get("__meta__", {})
    if not str(meta.get("format", "")).startswith("forge-swarm-"):
        raise ValueError(
            f"权重格式不认得：{meta.get('format')!r}，"
            "期望 'forge-swarm-v2'")
    if "hidden_dim" not in meta:
        raise ValueError("检查点缺少 __meta__.hidden_dim，拒绝导出")

    torch.save(sd, out / "swarm_weights.pt")

    arch = meta["arch"]
    hf_config = {
        "architectures": ["ScalpelUniversalForCausalLM"],
        "model_type": "scalpel_universal",
        "base_model_name_or_path": meta.get("base_model_id")
                                    or cfg.base_model_id,
        "num_macro_cores": int(arch["num_macro_cores"]),
        "num_micro_experts": int(arch["num_micro_experts"]),
        "macro_rank": int(arch.get("macro_rank", cfg.arch.macro_rank)),
        "micro_rank": int(arch.get("micro_rank", cfg.arch.micro_rank)),
        "micro_top_k": int(cfg.train.micro_top_k),
        "micro_scale": float(cfg.arch.micro_scale),
        "macro_dense_scale": float(cfg.arch.macro_dense_scale),
        "hidden_dim": int(meta["hidden_dim"]),
        "macro_names": list(arch.get("macro_names", [])),
        "domain_to_macro": dict(arch.get("domain_to_macro", {})),
        "text_only": bool(getattr(cfg, "text_only", True)),
        "auto_map": {
            "AutoConfig": "configuration_scalpel.ScalpelUniversalConfig",
            "AutoModelForCausalLM": "modeling_scalpel.ScalpelUniversalForCausalLM",
        },
    }
    (out / "config.json").write_text(
        json.dumps(hf_config, indent=2, ensure_ascii=False), encoding="utf-8")

    (out / "README.md").write_text(_readme(cfg, hf_config, meta),
                                   encoding="utf-8")

    return {
        "dir": str(out),
        "files": sorted(p.name for p in out.iterdir()),
        "num_macro_cores": hf_config["num_macro_cores"],
        "num_micro_experts": hf_config["num_micro_experts"],
        "hidden_dim": hf_config["hidden_dim"],
        "step": meta.get("step"),
    }


def _readme(cfg: ForgeConfig, hf: dict[str, Any],
            meta: dict[str, Any]) -> str:
    names = "、".join(hf["macro_names"]) or "（未命名）"
    return f"""# {cfg.project_name}

基于 `{hf['base_model_name_or_path']}` 的 MoE"蜂群"微调产物。

## 架构

- 大核（macro）：{hf['num_macro_cores']} 个 —— {names}
  其中 0 号是只读底座，`[1, M)` 是 rank-{hf['macro_rank']} 的 Dense LoRA。
- 小核（micro）：{hf['num_micro_experts']} 个 rank-{hf['micro_rank']} 专家，
  每 token 稀疏激活 top-{hf['micro_top_k']}。
- hidden_dim：{hf['hidden_dim']}；训练步数：{meta.get('step')}

## 加载

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "<你的仓库名>",                 # 或本地目录
    trust_remote_code=True,
    device_map="auto",
    dtype="bfloat16",
)

print(model.telemetry.summary(model.config.macro_names))
```

`model.telemetry.summary()` 会给出大核/小核的命中分布与
`macro_dead` / `micro_dead`，用来确认专家没有坍缩。

## 生成

```python
print(model.generate("**用一句话**解释一下傅里叶变换的物理含义。",
                      max_new_tokens=256, do_sample=False))
```

## 注意

- 权重只包含 swarm 部分（router + LoRA），底座由
  `base_model_name_or_path` 在加载时拉取，因此本仓库很小。
- 大核恒有一个单位权重指向只读底座，所以路由没训练好时模型
  仍会退化为原底座行为，不会把底座改坏。
"""


__all__ = ["build_bundle"]
