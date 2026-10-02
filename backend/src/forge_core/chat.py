"""``forge chat`` —— 对训练好的模型做推理测试。

两种模式::

    # 一次性
    forge chat --prompt "写一个快排"

    # 交互式（多轮，带历史）
    forge chat

刻意做成"能看出路由在干什么"，而不是只打印一段文本：
每次回答后都会报告第一层 router 选了哪些专家、大核权重多少，
以及本次的速度。调参时这比 loss 曲线有用得多。
"""

from __future__ import annotations

import json
from typing import Any

from .schema import DEFAULT_CONFIG_PATH, ForgeConfig


def _bar(frac: float, width: int = 12) -> str:
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "█" * n + "·" * (width - n)


def _print_route(chat, macro_names: list[str]) -> None:
    st = chat.route_of_last()
    if not st:
        return
    names = list(macro_names) or list(st["macro_dist"])
    print("\n  路由（第一层）")
    for name in names:
        v = st["macro_dist"].get(name, 0.0)
        print(f"    {name:<18} {_bar(v)} {v * 100:5.1f}%")
    md = st.get("micro_dist") or []
    live = [i for i, v in enumerate(md) if v > 0]
    if live:
        top = sorted(range(len(md)), key=lambda i: -md[i])[:6]
        print("    小核 top6: " + "  ".join(f"#{i}:{md[i] * 100:.1f}%"
                                           for i in top))
        print(f"    活跃小核 {len(live)}/{len(md)}"
              + (f"  死掉 {st['micro_dead']}" if st.get("micro_dead") else ""))


def cmd_chat(args: Any) -> int:
    import torch  # noqa: F401  (确保 device 解析逻辑一致)

    from .infer import SwarmChat

    cfg = ForgeConfig.load(args.config)
    if args.data:
        cfg.data_path = args.data
    if args.device:
        cfg.train.device = args.device
    if args.top_k:
        cfg.train.micro_top_k = args.top_k

    chat = SwarmChat.from_checkpoint(cfg, args.checkpoint,
                                     device=args.device or None)
    print(f"底座   {cfg.base_model_id}")
    print(f"架构   M={cfg.arch.num_macro_cores}  N={cfg.arch.num_micro_experts}"
          f"  top-k={cfg.train.micro_top_k}")
    print(f"权重   {args.checkpoint}  已加载 {chat.loaded} 个张量"
          f"（step {chat.meta.get('step')}）")
    if args.micro_scale is not None or args.macro_scale is not None:
        chat.set_scales(micro_scale=args.micro_scale,
                        macro_scale=args.macro_scale)
        print(f"热调   micro_scale={args.micro_scale} "
              f"macro_scale={args.macro_scale}")

    gen_kw: dict[str, Any] = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 0:
        gen_kw.update(temperature=args.temperature, top_p=args.top_p,
                      seed=args.seed)
    else:
        gen_kw["seed"] = args.seed

    names = list(cfg.arch.macro_names)

    # -- 对照模式 -----------------------------------------------------
    if args.compare:
        msgs = [{"role": "user", "content": args.prompt}]
        r = chat.compare_with_base(msgs, **gen_kw)
        print("\n" + "=" * 68)
        print("【纯底座】（micro_scale=macro_scale=0）")
        print("-" * 68)
        print(r["base"])
        print("\n" + "=" * 68)
        print("【底座 + swarm】")
        print("-" * 68)
        print(r["swarm"])
        print("=" * 68)
        if r["identical"]:
            print("⚠ 两者输出**完全相同** —— 说明适配器几乎没起作用。")
            print("  可能原因：训练步数太少 / lr 太小 / scale 被设成 0 / "
                  "权重没真正加载。")
        else:
            print("✓ 输出不同，适配器确实在起作用。")
        print(f"  速度：底座 {r['base_tok_per_sec']:.1f} tok/s vs "
              f"swarm {r['swarm_tok_per_sec']:.1f} tok/s")
        _print_route(chat, names)
        return 0

    # -- 一次性 -------------------------------------------------------
    if args.prompt:
        system = open(args.system, encoding="utf-8").read() \
            if args.system else None
        text, st = chat.ask(args.prompt, system=system, **gen_kw)
        print("\n" + text)
        print(f"\n[{st.new_tokens} tok, {st.tok_per_sec:.1f} tok/s, "
              f"prompt {st.prompt_tokens} tok]")
        _print_route(chat, names)
        return 0

    # -- 交互式 -------------------------------------------------------
    if not sys_stdin_tty():
        print("非交互环境，请用 --prompt 提供问题", file=sys_stderr)
        return 2

    system = open(args.system, encoding="utf-8").read() if args.system else None
    history: list[dict[str, str]] = []
    print("\n输入问题开始对话。/reset 清空历史，/route 看累计分布，"
          "/scale <micro> <macro> 热调，/exit 退出。\n")
    while True:
        try:
            q = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q in {"/exit", "/quit", ":q"}:
            return 0
        if q == "/reset":
            history.clear()
            print("（历史已清空）")
            continue
        if q == "/route":
            _print_route(chat, names)
            continue
        if q.startswith("/scale"):
            parts = q.split()
            try:
                ms = float(parts[1]) if len(parts) > 1 else None
                mcs = float(parts[2]) if len(parts) > 2 else None
            except ValueError:
                print("用法：/scale <micro_scale> [macro_scale]")
                continue
            chat.set_scales(micro_scale=ms, macro_scale=mcs)
            print(f"（micro_scale={ms} macro_scale={mcs}）")
            continue

        history.append({"role": "user", "content": q})
        print("模型 > ", end="", flush=True)
        text, st = chat.generate(
            ([{"role": "system", "content": system}] if system else [])
            + history, **gen_kw)
        print(text)
        print(f"   [{st.new_tokens} tok, {st.tok_per_sec:.1f} tok/s]")
        history.append({"role": "assistant", "content": text})
        _print_route(chat, names)
        print()


def sys_stdin_tty() -> bool:
    import sys
    return sys.stdin.isatty()


def _default_checkpoint() -> str:
    from .schema import ForgeConfig as _C
    return _C().checkpoint_path


def register(sub) -> None:
    from .schema import DEFAULT_CONFIG_PATH

    p = sub.add_parser("chat", help="对训练好的模型做推理测试")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    p.add_argument("--data", default=None)
    p.add_argument("--checkpoint", default=_default_checkpoint(),
                   help="swarm 权重（save_checkpoint 的产物）")
    p.add_argument("--device", default=None, help="auto / cuda:0 / cpu")
    p.add_argument("--prompt", default=None, help="一次性提问；不给则进交互模式")
    p.add_argument("--system", default=None, help="system prompt 文件路径")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = 贪心解码（可复现）")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=None,
                   help="热调小核 top-k（1..N）")
    p.add_argument("--micro-scale", type=float, default=None)
    p.add_argument("--macro-scale", type=float, default=None)
    p.add_argument("--compare", action="store_true",
                   help="额外跑一遍纯底座做对照")
    p.set_defaults(func=cmd_chat)
