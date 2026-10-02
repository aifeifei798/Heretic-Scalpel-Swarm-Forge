"""把一次对话推理放进独立子进程。

为什么又是子进程
----------------
加载底座会占十几 GB 显存。如果在 API 进程里直接
``from_pretrained``：

* 显存常驻，训练就 OOM；
* 更麻烦的是 **CUDA context 一旦建立就摘不掉**——推理结束、
  引用计数归零之后，allocator 仍持有那块 reserve。
  于是"试跑一次模型"会让后续所有训练都在低位显存里跑，
  而且看不出原因。

所以每次试跑都起一个短命子进程，跑完连同显存一起消失。
这也顺带避免了长时间生成时占住 API 的事件循环。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .forge_core_bridge import backend_src, python_exe, sanitize_env
from .settings import Settings

#: 单次生成的硬上限。底座 10GB 显存 + 生成时的激活，
#: 再长就 OOM 了，与其让用户等 5 分钟看一个 CUDA 报错，不如早失败。
MAX_NEW_TOKENS_CEILING = 1024


def _child_argv(spec: Any, ckpt: Path, root: Path) -> list[str]:
    return [python_exe(), "-u", "-m", "forge_web._chat_child",
            "--checkpoint", str(ckpt),
            "--config", str(root / "forge_config.json"),
            "--device", spec.device,
            "--prompt", spec.prompt,
            "--system", spec.system or "",
            "--history", json.dumps(spec.history, ensure_ascii=False),
            "--max-new-tokens", str(spec.max_new_tokens),
            "--temperature", str(spec.temperature),
            "--top-p", str(spec.top_p),
            "--seed", str(spec.seed),
            "--top-k", str(spec.top_k or 0),
            "--compare", "1" if spec.compare else "0"]


class ChatWorker:
    def __init__(self, settings: Settings) -> None:
        self.st = settings

    def run(self, spec: Any, ckpt: Path) -> dict[str, Any]:
        sanitize_env()
        env = dict(os.environ)
        env["PYTHONPATH"] = str(backend_src())
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("HF_HUB_OFFLINE", "1")
        env["TOKENIZERS_PARALLELISM"] = "false"

        argv = _child_argv(spec, ckpt, self.st.project_root)
        try:
            proc = subprocess.run(
                argv, cwd=str(self.st.project_root), env=env,
                capture_output=True, text=True,
                timeout=spec.timeout_sec,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"生成超时（{spec.timeout_sec}s）。"
                "底座加载本身就要 20~40 秒，可以调大 timeout_sec。") from e

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise RuntimeError(
                "推理失败：\n" + "\n".join(tail[-12:]))

        # 子进程把结果作为最后一行的 JSON 打出
        for line in reversed((proc.stdout or "").splitlines()):
            line = line.strip()
            if line.startswith("{") and '"text"' in line:
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        raise RuntimeError(
            "子进程没有返回可解析的结果。原始输出：\n"
            + (proc.stdout or "")[-800:])


__all__ = ["ChatWorker", "MAX_NEW_TOKENS_CEILING"]
