"""命令行入口：NDJSON 结构化日志 + 子进程友好。

设计要点
--------
* **stdout 只输出 NDJSON**（每行一个 JSON 事件），日志文本走 stderr。
  未来的 API 层可以逐行解析 stdout 并通过 SSE 推给前端，
  同时人眼在终端里也能正常阅读。
* 绝不 import ``fastapi``——本模块与 Web 层零耦合。

用法::

    forge datagen  --out dual_contrast_data.jsonl --per-domain 200
    forge audit    dual_contrast_data.jsonl
    forge inspect  dual_contrast_data.jsonl --domain Code_Algo --limit 3
    forge plan     --out forge_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .schema import DEFAULT_CONFIG_PATH, ForgeConfig


# ---------------------------------------------------------------------------
# NDJSON 事件输出
# ---------------------------------------------------------------------------
class Emitter:
    """向 stdout 发 NDJSON 事件；同时把人类可读的摘要打到 stderr。"""

    def __init__(self, *, verbose: bool = True, stream=None) -> None:
        self.verbose = verbose
        self.stream = stream if stream is not None else sys.stdout
        self.err = sys.stderr
        self._t0 = time.time()

    def __call__(self, event: dict[str, Any]) -> None:
        event.setdefault("elapsed", round(time.time() - self._t0, 3))
        self.stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.stream.flush()
        if self.verbose and event.get("t") not in ("log",):
            msg = event.get("msg") or self._summarize(event)
            if msg:
                print(msg, file=self.err)

    @staticmethod
    def _summarize(e: dict[str, Any]) -> str:
        t = e.get("t", "")
        if t == "domain_done":
            return f"  [{e['domain']:<18}] {e['rows']:>4} 条  (容量 {e['capacity']})"
        if t == "datagen_done":
            return (f"[datagen] {e['total_rows']} 条，唯一率 {e['unique_ratio']:.4f}，"
                    f"train/val = {e['train_rows']}/{e['val_rows']}，"
                    f"耗时 {e['duration_sec']}s")
        if t == "step":
            return (f"  step {e['step']:>4}/{e.get('max_steps', '?')} "
                    f"lm={e.get('lm_loss'):.4f} macro={e.get('macro_loss', 0):.4f} "
                    f"micro={e.get('micro_loss', 0):.4f} "
                    f"vram={e.get('vram_gb', 0):.1f}G")
        return ""


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------
def cmd_datagen(args: argparse.Namespace) -> int:
    from .datagen import build_dataset
    from .schema import DataGenConfig

    cfg = DataGenConfig(
        output_path=args.out,
        samples_per_domain=args.per_domain,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if args.no_perturb:
        from .schema import PerturbConfig
        cfg = cfg.model_copy(update={"perturb": PerturbConfig()})

    emit = Emitter(verbose=not args.quiet)
    report = build_dataset(cfg, emit=emit)
    if args.audit:
        from .datagen.audit import main as audit_main
        return audit_main(args.out)
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from .datagen.audit import main as audit_main
    return audit_main(args.path)


def cmd_inspect(args: argparse.Namespace) -> int:
    from .datagen import length_stats, preview

    rows = preview(args.path, limit=args.limit,
                   domains=[args.domain] if args.domain else None)
    for i, r in enumerate(rows, 1):
        print("=" * 70)
        print(f"[{i}] {r['domain']}  ({r.get('split', '-')})  id={r['id']}")
        print(f"  meta: {json.dumps(r.get('meta', {}), ensure_ascii=False)}")
        print("-" * 70)
        print(r["prompt"])
        print("-" * 70)
        print(r["response"])
    print("=" * 70)
    print("长度统计:", json.dumps(length_stats(args.path), ensure_ascii=False))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """跑一次训练。所有进度以 NDJSON 打到 stdout。"""
    import torch

    from .train import train

    cfg = ForgeConfig.load(args.config)
    if args.data:
        cfg.data_path = args.data
    if args.device:
        cfg.train.device = args.device
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.grad_accum:
        cfg.train.grad_accum = args.grad_accum
    if args.max_length:
        cfg.train.max_length = args.max_length
    if args.eval_every is not None:
        cfg.train.eval_every = args.eval_every
    if args.loss_chunk:
        cfg.train.loss_chunk_size = args.loss_chunk
    if args.lr:
        cfg.train.lr_macro = args.lr
        cfg.train.lr_router = args.lr
        cfg.train.lr_micro = args.lr
    if args.data_only:
        cfg.checkpoint_path = args.checkpoint
    elif args.checkpoint:
        cfg.checkpoint_path = args.checkpoint
    if args.limit:
        from .dataset import load_rows
        rows = load_rows(cfg.data_path, split="train", limit=args.limit)
        cfg.data_path = cfg.data_path  # rows 直接传给 train()
    else:
        rows = None

    emit = Emitter(verbose=not args.quiet)
    torch.manual_seed(cfg.train.seed)
    result = train(cfg, emit=emit, rows=rows)

    if args.checkpoint:
        result = {**result, "checkpoint": args.checkpoint}
    emit({"t": "train_summary", **result})
    if args.export:
        from .export import build_bundle

        ckpt = result.get("checkpoint") or cfg.checkpoint_path
        emit({"t": "export_start", "out": args.export, "checkpoint": ckpt})
        info = build_bundle(cfg, ckpt, args.export)
        emit({"t": "export_done", **info})
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """把已有 checkpoint 打成发布包（不训练）。"""
    from .export import build_bundle

    cfg = ForgeConfig.load(args.config)
    if args.data:
        cfg.data_path = args.data
    emit = Emitter(verbose=not args.quiet)
    info = build_bundle(cfg, args.checkpoint, args.out)
    emit({"t": "export_done", **info})
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    """把发布包推到 Hugging Face Hub。"""
    from .export import publish

    emit = Emitter(verbose=not args.quiet)
    try:
        info = publish(args.bundle, args.repo, token=args.token,
                       private=args.private, dry_run=args.dry_run)
    except Exception as e:                                  # noqa: BLE001
        emit({"t": "publish_failed", "error": f"{type(e).__name__}: {e}"})
        return 1
    emit({"t": "publish_done", **info})
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """CPU 上的最小规模冒烟测试：验证前向/反向/路由/保存全链路可跑。"""
    import torch
    import torch.nn as nn

    from .modeling import SwarmWrapper
    from .schema import ArchProfile, TrainConfig

    cfg = TrainConfig(device="cpu", param_dtype="fp32", max_length=64,
                      batch_size=2, grad_accum=1, max_steps=2)
    arch = ArchProfile(num_macro_cores=4, macro_names=list("ABCD"),
                       macro_rank=4, num_micro_experts=4, micro_rank=2)
    w = SwarmWrapper(nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 16)),
                     hidden_dim=16, num_macro=4, macro_rank=4,
                     num_micro=4, micro_rank=2, micro_top_k=2,
                     param_dtype=torch.float32, device="cpu")
    x = torch.randn(2, 5, 16)
    out = w(x)
    assert out.shape == x.shape

    loss = out.pow(2).mean()
    mce, nce, lb = w.router_losses(torch.ones(2, 5, dtype=torch.bool),
                                   torch.zeros(2, dtype=torch.long),
                                   torch.zeros(2, dtype=torch.long))
    (loss + 0.1 * (mce + nce) + 0.01 * lb).backward()
    grads = [p.grad is not None for ps in w.param_groups().values() for p in ps]
    assert all(grads), "部分参数没有梯度"
    print(f"smoke ok: out={tuple(out.shape)} loss={float(loss.detach()):.4f} "
          f"macro_ce={float(mce.detach()):.4f} micro_ce={float(nce.detach()):.4f} "
          f"lb={float(lb.detach()):.4f}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """打印/写出默认配置，或报告当前配置的摘要与容量预估。"""
    cfg = ForgeConfig.load(args.config)
    if args.data:
        from .datagen import preview
        rows = preview(args.data, limit=1)
        if rows:
            print(f"数据集样例 domain={rows[0]['domain']}")
    print(json.dumps(cfg.summary(), indent=2, ensure_ascii=False))
    cfg.save(args.config)
    print(f"\n配置已写入 {args.config}", file=sys.stderr)
    return 0


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "datagen": cmd_datagen,
    "audit": cmd_audit,
    "inspect": cmd_inspect,
    "train": cmd_train,
    "smoke": cmd_smoke,
    "plan": cmd_plan,
}


# ---------------------------------------------------------------------------
# 解析器
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forge",
        description="Heretic-Scalpel Swarm Forge CLI（NDJSON 事件流）")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="不向 stderr 打人类可读摘要")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("datagen", help="生成数据集")
    d.add_argument("--out", default="dual_contrast_data.jsonl")
    d.add_argument("--per-domain", type=int, default=200,
                   help="每个领域的条数（默认 200，32 领域共 6400）")
    d.add_argument("--val-ratio", type=float, default=0.05)
    d.add_argument("--seed", type=int, default=20260929)
    d.add_argument("--no-perturb", action="store_true",
                   help="关闭所有格式变体（仅行内容，用于回归对比）")
    d.add_argument("--audit", action="store_true",
                   help="生成后立即跑质量体检")
    d.set_defaults(func=cmd_datagen)

    a = sub.add_parser("audit", help="对已有数据集跑质量体检")
    a.add_argument("path", nargs="?", default="dual_contrast_data.jsonl")
    a.set_defaults(func=cmd_audit)

    i = sub.add_parser("inspect", help="预览数据集样本")
    i.add_argument("path", nargs="?", default="dual_contrast_data.jsonl")
    i.add_argument("--domain", default=None)
    i.add_argument("--limit", type=int, default=3)
    i.set_defaults(func=cmd_inspect)

    tr = sub.add_parser("train", help="训练（NDJSON 进度流）")
    tr.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    tr.add_argument("--data", default=None)
    tr.add_argument("--device", default=None, help="auto / cuda:0 / cpu")
    tr.add_argument("--max-steps", type=int, default=None)
    tr.add_argument("--batch-size", type=int, default=None)
    tr.add_argument("--grad-accum", type=int, default=None)
    tr.add_argument("--max-length", type=int, default=None)
    tr.add_argument("--limit", type=int, default=None, help="只用前 N 条（冒烟）")
    tr.add_argument("--checkpoint", default=None, help="检查点输出路径")
    tr.add_argument("--data-only", action="store_true",
                    help="仅保存 swarm 权重（默认即如此，保留以显式表达意图）")
    tr.add_argument("--eval-every", type=int, default=None,
                    help="每 N 步在验证集上评测一次（0 = 关闭）")
    tr.add_argument("--loss-chunk", type=int, default=None,
                    help="分块交叉熵每块的监督 token 数（调小省显存）")
    tr.add_argument("--lr", type=float, default=None, help="覆盖三个学习率")
    tr.add_argument("--export", default=None, help="训练后导出到该目录")
    tr.set_defaults(func=cmd_train)

    sm = sub.add_parser("smoke", help="CPU 最小规模自检（不需要 GPU）")
    sm.set_defaults(func=cmd_smoke)

    ex = sub.add_parser("export", help="把 checkpoint 打成发布包")
    ex.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ex.add_argument("--data", default=None)
    ex.add_argument("--checkpoint", required=True)
    ex.add_argument("--out", required=True, help="发布目录")
    ex.set_defaults(func=cmd_export)

    pb = sub.add_parser("publish", help="推送发布包到 Hugging Face Hub")
    pb.add_argument("bundle", help="发布目录")
    pb.add_argument("repo", help="目标仓库，形如 namespace/name")
    pb.add_argument("--token", default=None)
    pb.add_argument("--private", action="store_true")
    pb.add_argument("--dry-run", action="store_true",
                    help="只做本地校验并打印计划，不联网")
    pb.set_defaults(func=cmd_publish)

    pl = sub.add_parser("plan", help="初始化/查看配置")
    pl.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    pl.add_argument("--data", default=None)
    pl.set_defaults(func=cmd_plan)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())