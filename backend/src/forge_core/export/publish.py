"""发布到 Hugging Face Hub。

替代旧实现的 ``os.system(f"huggingface-cli upload ...")``。

为什么不能留 ``os.system``
--------------------------
1. **返回值被丢弃**——shell 命令失败（仓库不存在、token 无效、
   网络不通）时 ``os.system`` 只返回一个非零整数，脚本照样
   打印"🎉 发布完成！"，用户以为推上去了其实没有。
2. **命令注入**——仓库名直接插进 shell 字符串，
   ``a/b; rm -rf ~`` 会被当成两条命令执行。
3. **不可靠凭据**——拿不到 ``HfApi`` 的 token 校验反馈，
   也没法区分"没登录"和"没权限"。

现在用 ``HfApi().create_repo`` / ``upload_folder``，
异常直接抛出；``--dry-run`` 只打印计划不联网。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def publish(bundle_dir: str | Path, repo_id: str, *,
            token: str | None = None,
            private: bool = False,
            commit_message: str | None = None,
            dry_run: bool = False) -> dict[str, Any]:
    """把发布包推送到 Hub。

    Parameters
    ----------
    bundle_dir: :func:`build_bundle` 产出的目录
    repo_id:    ``namespace/name``
    token:      不给则读环境变量 ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``
    dry_run:    只做本地校验并返回将要执行的计划，不联网

    Raises
    ------
    FileNotFoundError / RuntimeError / huggingface_hub 的原生异常
    """
    src = Path(bundle_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"发布目录不存在：{src}")
    required = ("config.json", "swarm_weights.pt", "modeling_scalpel.py",
                "configuration_scalpel.py", "swarm_forge.py", "telemetry.py")
    missing = [f for f in required if not (src / f).exists()]
    if missing:
        raise RuntimeError(
            f"发布包不完整，缺少：{missing}。请先跑 export。")

    plan = {"repo_id": repo_id, "source": str(src),
            "private": private, "files": sorted(p.name for p in src.iterdir())}

    if dry_run:
        plan["dry_run"] = True
        return plan

    # 延迟导入：没装 huggingface_hub 时也不该让 export 挂掉
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True,
                    repo_type="model")
    info = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(src),
        commit_message=commit_message or f"forge swarm weights: {repo_id}",
    )

    return {**plan, "dry_run": False,
            "url": f"https://huggingface.co/{repo_id}",
            "commit": getattr(info, "oid", None)}


__all__ = ["publish"]
