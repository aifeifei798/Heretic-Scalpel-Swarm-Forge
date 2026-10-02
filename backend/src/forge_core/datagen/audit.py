"""数据集质量体检。

这是 datagen 的**回归门禁**：把 A1（6400 条只有 41 条唯一）这类问题
变成可自动检测的断言，避免下次改模板时静默退化。

每条检查都注明了它抓的是哪一类真实 bug——例如"Python 代码块可解析"
这条，历史上抓出过三类缺陷：标点风格把全角逗号灌进代码、
围栏风格被拆掉导致代码失去保护、空白折叠压平缩进。

CLI::

    python -m forge_core.datagen.audit dual_contrast_data.jsonl
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

PUNCT = r"[。．.，,、；;：]"

#: 缩进代码块（围栏被拆掉时的兜底识别）
_INDENT_BLOCK = r"(?m)^(?:(?: {4}|\t).*(?:\n|$))+"

#: 纯装饰差异：围栏风格、全/半角标点、空白。去掉它们之后的文本才代表
#: "内容"。**不能**把代码块内部的缩进也压掉——缩进是代码语义的一部分。
_DECOR_FENCE = re.compile(r"(```|~~~)")
_DECOR_PUNCT = re.compile(r"[。，；：？！、（）“”‘’]|[,\.;:\?!()\"']")
_WS = re.compile(r"\s+")


def norm_content(text: str) -> str:
    """抹掉纯装饰差异后的"内容指纹"。

    用途：判断两行数据是不是**同一个 case 的换皮渲染**。
    例如同一道题分别用 ``` 与 ~~~ 围栏、用全角与半角标点各渲染一次，
    字节不同但内容完全相同——只查字节去重会把它们当成两条独立样本，
    从而高估数据集的多样性。
    """
    out = _DECOR_FENCE.sub("@F@", text)
    out = _DECOR_PUNCT.sub("@P@", out)
    return _WS.sub("", out)


def main(path: str) -> int:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    fails: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    # --- 唯一性：三个口径，必须分开报 ------------------------------
    #
    # 旧实现只查 (prompt, response) 去重，然后打上"唯一率 = 100%"。
    # 这个标签是**误导性**的：同一个 case 用 ``` vs ~~~ 围栏、
    # 全角 vs 半角标点各渲染一次，就会产出一对新的 (prompt, response)，
    # 于是字节级去重必然是 100%，而真实内容其实重复了。
    #
    # 三个口径各自的含义：
    #   * pair   —— 训练意义上"完全相同的两条样本"，必须 100%；
    #               重复它等于给同一道题同一份答案喂两遍，纯浪费。
    #   * prompt —— 题面是否重复。低于 100% 说明有 case 被重复渲染，
    #               是**信息**指标，dual-contrast 设计本身就会
    #               让"同一答案配多个问法"，所以不要求 100%。
    #   * content—— 抹掉围栏/标点/空白等**纯装饰**差异后还剩多少内容。
    #               这才是"数据集到底有多少种题"的真实答案，
    #               用来判断模板库是不是在自我重复。
    uniq_pair = len({(r["prompt"], r["response"]) for r in rows})
    check("唯一率(prompt+response) = 100%",
          uniq_pair == len(rows), f"{uniq_pair}/{len(rows)}")

    uniq_prompt = len({r["prompt"] for r in rows})
    n_prompt = uniq_prompt / max(1, len(rows))

    uniq_content = len({(norm_content(r["prompt"]), norm_content(r["response"]))
                        for r in rows})
    n_content = uniq_content / max(1, len(rows))

    check("唯一率(去装饰内容) >= 95%", n_content >= 0.95,
          f"{uniq_content}/{len(rows)} = {n_content:.1%}")

    # 题面唯一率**不做门禁**，只报告。
    #
    # 同一个问题配多个不同答案是本项目的 dual-contrast 设计：
    # 教会模型"一道题可以有多种有效解法"，同时阻止它把某一段
    # 固定答案背下来。实测 6400 条里 337 个题面各出现 2~4 次，
    # 而这 337 组里的**回答两两不同**。
    #
    # 真正该卡的是"重复题面 + 重复答案"——那是纯浪费。
    # 下面那条 check 才是对应的门禁。
    print(f"INFO  题面唯一率(非门禁)  {uniq_prompt}/{len(rows)} = {n_prompt:.1%}")

    n_prompt_dup = Counter(r["prompt"] for r in rows)
    resp_per_prompt: dict[str, list[str]] = {}
    for r in rows:
        resp_per_prompt.setdefault(r["prompt"], []).append(r["response"])
    bad = [p for p, n in n_prompt_dup.items()
           if n > 1 and len(set(resp_per_prompt[p])) != n]
    n_reused = sum(n for n in n_prompt_dup.values() if n > 1)
    check("重复题面的答案互不相同", not bad,
          f"{n_reused} 条属于重复题面，答案两两不同" if not bad
          else f"{len(bad)} 组存在完全相同的题面+答案")

    per = Counter(r["domain"] for r in rows)
    check("32 领域全覆盖", len(per) == 32, f"{len(per)} 个")
    check("每领域条数一致", len(set(per.values())) == 1,
          f"{min(per.values())}~{max(per.values())}")

    splits = Counter(r.get("split") for r in rows)
    check("train/val 均非空", splits.get("train", 0) and splits.get("val", 0),
          f"train={splits.get('train')} val={splits.get('val')}")

    check("无空字段",
          all(r["prompt"].strip() and r["response"].strip() for r in rows))
    # 只在**散文**里查 HTML 残留。但 indent 围栏变体会把围栏拆成裸缩进块，
    # Rust 的 `&mut Vec<T>` / `a.len()` 这类泛型与方法调用必须先剔除，
    # 否则 `<T>` 会被误判成标签、`<a` 会被误判成 <a>。
    def strip_code(t: str) -> str:
        t = re.sub(r"(?:```|~~~).*?(?:```|~~~)", " ", t, flags=re.S)
        t = re.sub(r"`[^`\n]*`", " ", t)
        # indent 变体把围栏拆成了裸缩进块，其中**没有中文的行**必然是代码
        t = re.sub(r"(?m)^(?:(?: {4}|\t)(?![^\n]*[一-鿿]).*(?:\n|$))+", " ", t)
        t = re.sub(r"(?m)^(?:(?: {4}|\t).*(?:\n|$))+", " ", t)
        t = re.sub(r"<\s*&mut\s+", "A ", t)                    # Rust &mut T
        t = re.sub(r"<\s*[A-Za-z_]\w*(?:::\w+)*\s*>", "A", t)   # 泛型参数
        t = re.sub(r"\b[a-z]\.[a-z_]+\(", "A(", t)             # 方法调用 a.len()
        return t

    check("散文无 HTML/脚本残留",
          not any(re.search(r"<\s*/?\s*(script|style|a|div|span|p)\b",
                            strip_code(r["prompt"] + r["response"]), re.I)
                  for r in rows))

    # Python 代码块必须能解析（被标点风格破坏过，这里是最灵敏的探针）
    ok = bad = 0
    first_bad = ""
    for r in rows:
        for m in re.finditer(r"```python\n(.*?)```", r["response"], re.S):
            try:
                ast.parse(m.group(1))
                ok += 1
            except SyntaxError:
                bad += 1
                if not first_bad:
                    first_bad = f"{r['domain']}: {m.group(1)[:90]!r}"
    check("Python 代码块可解析", bad == 0, f"ok={ok} fail={bad} {first_bad}")

    # 代码/行内代码里不能出现全角标点。
    # 注意 indent 变体会把围栏拆成裸缩进块，而代码里的**中文注释**
    # （"# 复制一份，允许订阅者…"）本身就该有全角标点，
    # 因此缩进块只查"代码字符"行（不含注释与非 ASCII 汉字）。
    def cjk_in_code(t: str) -> bool:
        def bad(block: str) -> bool:
            for ln in block.split("\n"):
                if re.search(r"[一-鿿]", ln):
                    continue                      # 中文注释，放行
                if re.search(r"[，；：]", ln):
                    return True
            return False
        if any(bad(m.group(0))
               for m in re.finditer(r"(?:```|~~~).*?(?:```|~~~)", t, re.S)):
            return True
        return any(bad(m.group(0))
                   for m in re.finditer(r"(?m)^(?:(?: {4}|\t).*(?:\n|$))+", t))
    check("代码块内无全角标点",
          not any(cjk_in_code(r["response"]) for r in rows))

    # 行内数学内部不能出现全角标点（千分位逗号 ``(3, -2)`` 必须保持半角）
    def cjk_in_math(t: str) -> bool:
        t = re.sub(r"(?:```|~~~).*?(?:```|~~~)", " ", t, flags=re.S)
        t = re.sub(r"`[^`\n]*`", " ", t)
        return any(re.search(r"[，；：]", m.group(0))
                   for m in re.finditer(r"\$\$?.+?\$\$?", t))
    check("行内数学无全角标点",
          not any(cjk_in_math(r["prompt"] + r["response"]) for r in rows))

    # 重复标点（排除省略号与范围写法）。同样要剔除代码区：
    # Rust 的 `for (;;) {` 是合法语法，不是重复标点。
    def dup_punct(t: str) -> int:
        t = re.sub(r"(?:```|~~~).*?(?:```|~~~)", " ", t, flags=re.S)
        t = re.sub(r"`[^`\n]*`", " ", t)
        # 同上：indent 变体下没有中文的行是代码，`for (;;)` 不是重复标点
        t = re.sub(r"(?m)^(?:(?: {4}|\t)(?![^\n]*[一-鿿]).*(?:\n|$))+", " ", t)
        t = re.sub(r"(?m)^(?:(?: {4}|\t).*(?:\n|$))+", " ", t)
        t = re.sub(r"\.{2,}", "", t)
        return len(re.findall(PUNCT + "{2,}", t))
    n = sum(1 for r in rows if dup_punct(r["prompt"] + r["response"]))
    check("散文无重复标点", n == 0, f"{n} 条")

    # 语病：礼貌前缀 + 疑问词
    awkward = sum(1 for r in rows if re.match(
        r"(能否|请|请帮|麻烦|我想请你)(如何|为什么|是否|怎样|怎么|哪些|多少|哪个)",
        r["prompt"]))
    check("无'能否如何'式语病", awkward == 0, f"{awkward} 条")

    # 语病："能否如何…" —— 疑问词开头时不应再加礼貌前缀
    bad_prefix = [r["prompt"][:40] for r in rows if re.match(
        r"(能否|请|请帮|麻烦|我想请你)(如何|为什么|是否|怎样|怎么|哪些|多少|哪个)",
        r["prompt"])]
    check("无前缀+疑问词语病", not bad_prefix, f"{len(bad_prefix)} 条")

    print(f"\n{'=' * 60}")
    print(f"总计 {len(rows)} 条，失败 {len(fails)} 项" + (f": {fails}" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "dual_contrast_data.jsonl"))