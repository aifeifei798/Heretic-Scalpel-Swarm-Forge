"""``forge_web`` 的推理子进程入口。

只由 :mod:`forge_web.chatproc` 起，不对外暴露。
最后一行向 stdout 打一个 JSON，供父进程解析——中间那些模型日志
混在前面没关系，父进程是**倒着扫**找 JSON 的。
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="forge_web._chat_child")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default="")
    ap.add_argument("--history", default="[]")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--compare", type=int, default=0)
    args = ap.parse_args(argv)

    # 必须先设好，否则 from transformers 之前 HF 会去抓 config
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from forge_core.infer import SwarmChat
    from forge_core.schema import ForgeConfig

    cfg = ForgeConfig.load(args.config)
    cfg.train.device = args.device
    if args.top_k:
        cfg.train.micro_top_k = args.top_k

    chat = SwarmChat.from_checkpoint(cfg, args.checkpoint, device=args.device)

    msgs = []
    if args.system:
        msgs.append({"role": "system", "content": args.system})
    try:
        msgs += json.loads(args.history)
    except json.JSONDecodeError:
        pass
    msgs.append({"role": "user", "content": args.prompt})

    gen_kw: dict = {"max_new_tokens": args.max_new_tokens,
                    "seed": args.seed}
    if args.temperature > 0:
        gen_kw.update(temperature=args.temperature, top_p=args.top_p)

    out: dict = {"meta": {"loaded": chat.loaded,
                          "step": chat.meta.get("step"),
                          "macro_names": list(cfg.arch.macro_names),
                          "num_macro": cfg.arch.num_macro_cores,
                          "num_micro": cfg.arch.num_micro_experts,
                          "top_k": cfg.train.micro_top_k}}

    if args.compare:
        r = chat.compare_with_base(msgs, **gen_kw)
        out.update({"text": r["swarm"], "base_text": r["base"],
                    "identical": r["identical"],
                    "tok_per_sec": r["swarm_tok_per_sec"]})
    else:
        text, st = chat.generate(msgs, **gen_kw)
        out.update({"text": text, "tok_per_sec": st.tok_per_sec,
                    "new_tokens": st.new_tokens,
                    "prompt_tokens": st.prompt_tokens})

    out["route"] = chat.route_of_last()
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
