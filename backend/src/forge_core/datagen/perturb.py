"""模板 + 槽位扰动引擎。

根治 A1：原 ``build_pristine_dataset.py`` 里 6400 条样本只有 41 条唯一内容
（156x 重复），因为所谓"微扰模板"在代码里根本没实现。

本模块的做法
------------
1. **行对齐（row-aligned）绑定**：一个模板内所有槽位共享同一个行号索引，
   即第 *i* 行同时决定题面与解答。这样"问 Rust 答 Python"这类错配在结构上
   不可能发生——代价是每个槽位必须与主槽位等长，由 :meth:`Template.validate`
   强制检查。领域无关的额外多样性（礼貌前缀、请求后缀、代码围栏、
   数学分隔符、标点风格）作为**独立维度**与行索引做笛卡尔积，
   因此不会破坏问答语义一致性。

2. **组合空间**用混合进制索引采样（:func:`decompose`），不物化笛卡尔积，
   因此即使组合数上百万也只占常数内存，并且可复现。

3. **格式变体**（代码围栏 / 数学分隔符 / 标点风格）是最后叠加的层，
   且只作用于代码块之外，不会破坏代码和 LaTeX。

槽位语法
--------
使用 ``{{slot_name}}`` 双花括号作为占位符（而非 ``str.format`` 的单花括号），
因为模板正文里有大量 LaTeX 花括号（``\\frac{10}{2}``）会与 format 冲突。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from ..schema import PerturbConfig

# 纯装饰指纹：围栏风格（``` vs ~~~）、全角/半角标点、空白折叠。
# 刻意定义在 datagen 内部而不从 audit 导入——审计是门禁层，
# 不该被数据通路反向依赖。
_DECOR_FENCE = re.compile(r"(```|~~~)")
_DECOR_PUNCT = re.compile(r"[。，；：？！、（）“”‘’]|[,\.;:\?!()\"']")
_WS = re.compile(r"\s+")


def norm_content(text: str) -> str:
    """抹掉纯装饰差异后的"内容指纹"。

    围栏风格、全角/半角标点、空白都不改变内容；
    **不压代码缩进**——缩进是代码语义的一部分。
    """
    out = _DECOR_FENCE.sub("@F@", text)
    out = _DECOR_PUNCT.sub("@P@", out)
    return _WS.sub("", out)

# 槽位占位符：双花括号 + 标识符
_SLOT_RE = re.compile(r"\{\{(\w+)\}\}")

#: 三种围栏风格。必须统一在这里声明，否则 :func:`_apply_punct` /
#: :func:`_tidy_punct` / :func:`_iter_protected` 会漏掉 tilde 变体。
FENCE_MARKS: tuple[str, ...] = ("```", "~~~")

# 围栏代码块（含语言标注）
_FENCE_RE = re.compile(
    r"^(?:```|~~~)([A-Za-z0-9_+-]*)[ \t]*\n(.*?)^(?:```|~~~)[ \t]*$", re.S | re.M)
# 行内数学 $...$（排除 $$...$$）
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)([^\$\n]+?)\$(?!\$)")
# 任意形态的数学：行内 $...$ 与行间 $$...$$
_ANY_MATH_RE = re.compile(r"\$\$.+?\$\$|(?<!\$)\$(?!\$)[^\$\n]+?\$\$?(?!\$)", re.S)
# 四空格/制表符缩进的代码块
_INDENT_BLOCK_RE = re.compile(r"(?m)^(?:(?: {4}|\t).*(?:\n|$))+")
# 行内代码 `x=1,2`
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
#: 行内代码里出现范围/省略写法时，western 风格不能吃掉它后面的空格：
#: ``BETWEEN .. AND ...`` -> ``BETWEEN .AND ...`` 会改变语义。
_SPACE_SENSITIVE_INLINE_RE = re.compile(r"`[^`\n]*\.\.[^`\n]*`")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SlotValue:
    """一个槽位取值捆绑体。

    ``prompt`` / ``response`` 可以是普通字符串（填入本槽位自身的名字），
    也可以是 ``dict``（一次填入多个占位符名）——后者用于表达
    "同一个语义案例同时供给 prompt 侧与 response 侧的多个占位符"，
    例如 Code_Debug 里 ``{{code}}`` 与 ``{{fix}}`` 必须同源。
    """

    prompt: str | dict[str, str]
    response: str | dict[str, str]


@dataclass(frozen=True)
class Slot:
    """一个独立于行号的风味槽位。

    ``decorative=True`` 表示该槽位只提供修饰性文本（如称呼"小明"、
    语言标注"Python"），允许只出现在模板一侧。
    非装饰槽位携带**语义内容**，必须在 prompt 与 response 两侧同名出现，
    否则会出现"问快速排序却答二分查找"这类错配。
    """

    name: str
    values: tuple[SlotValue, ...]
    weight: float = 1.0
    decorative: bool = False

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(f"槽位 {{{{{self.name}}}}} 没有任何取值")

    def provided_names(self) -> set[str]:
        out: set[str] = set()
        for v in self.values:
            for side in (v.prompt, v.response):
                if isinstance(side, dict):
                    out |= set(side)
        return out


#: 行槽位的保留名。一个模板最多有一个行槽位，它同时供给题面与解答。
ROW_SLOT = "case"


@dataclass(frozen=True)
class Template:
    """一个领域内的样本模板。

    槽位分两类，这个区分是本引擎最重要的不变量：

    * **行槽位**（名为 ``case``，最多一个）——每一行是一个完整的
      ``(题面, 解答)`` 语义配对。行 *i* 同时决定 prompt 与 response，
      因此"问 $x^2-5x+6$ 却答 $2x^2+7x-15$"这类错配在结构上不可能发生。
    * **风味槽位**（其余全部）——彼此独立、与行号无关的修饰维度，
      例如 ``lang``、``count``、``focus``。
      它们在 prompt 与 response 两侧必须使用**同一个占位符名**，
      以保证"问 Python 答 Rust"不会发生；:meth:`validate` 会检查这一点。

    组合空间 = ``(行数) × Π(风味槽位基数) × (格式变体基数)``。
    """

    domain: str
    prompt: str
    response: str
    slots: tuple[Slot, ...] = ()
    weight: float = 1.0
    tags: tuple[str, ...] = field(default_factory=tuple)

    # -- 槽位分类 ------------------------------------------------------
    @property
    def row_slot(self) -> Slot | None:
        for s in self.slots:
            if s.name == ROW_SLOT:
                return s
        return None

    @property
    def flavor_slots(self) -> tuple[Slot, ...]:
        return tuple(s for s in self.slots if s.name != ROW_SLOT)

    def used_slot_names(self) -> set[str]:
        return set(_SLOT_RE.findall(self.prompt)) | set(_SLOT_RE.findall(self.response))

    def provided_names(self) -> set[str]:
        out: set[str] = set()
        for s in self.slots:
            out.add(s.name)
            out |= s.provided_names()
        return out

    @property
    def n_rows(self) -> int:
        rs = self.row_slot
        return len(rs.values) if rs else 1

    def validate(self) -> None:
        used, provided = self.used_slot_names(), self.provided_names()
        missing = used - provided
        if missing:
            raise ValueError(
                f"模板 {self.domain} 引用了未提供的占位符: {sorted(missing)}")
        unused = provided - used
        if unused:
            raise ValueError(
                f"模板 {self.domain} 提供了未被引用的占位符: {sorted(unused)}")

        n_row = sum(1 for s in self.slots if s.name == ROW_SLOT)
        if n_row > 1:
            raise ValueError(
                f"模板 {self.domain} 有 {n_row} 个行槽位；最多只能有一个 '{ROW_SLOT}'")

        # 非装饰风味槽位必须在 prompt 与 response 两侧同名出现，
        # 否则会出现"问 Python 答 Rust"。装饰槽位与行槽位不受此限。
        p_only = set(_SLOT_RE.findall(self.prompt)) - set(_SLOT_RE.findall(self.response))
        r_only = set(_SLOT_RE.findall(self.response)) - set(_SLOT_RE.findall(self.prompt))
        for slot in self.flavor_slots:
            if slot.decorative:
                continue
            names = {slot.name} | slot.provided_names()
            for name in sorted(names & (p_only | r_only)):
                raise ValueError(
                    f"模板 {self.domain} 的语义槽位 {{{{{name}}}}} 只出现在模板的一侧"
                    f"（prompt={name in p_only}, response={name in r_only}）；"
                    f"非装饰槽位必须两侧同名，否则语义会错配。"
                    f"若它确实只是修饰语，请声明 sl(..., decorative=True)")

    # -- 渲染映射 ------------------------------------------------------
    def flavor_mapping(self, flavor_picks: dict[str, int]) -> tuple[dict[str, str], dict[str, str]]:
        """风味槽位的替换映射（与行号无关）。"""
        pmap: dict[str, str] = {}
        rmap: dict[str, str] = {}
        for slot in self.flavor_slots:
            sv = slot.values[flavor_picks[slot.name]]
            for target, src in ((pmap, sv.prompt), (rmap, sv.response)):
                if isinstance(src, dict):
                    target.update(src)
                else:
                    target[slot.name] = src
        return pmap, rmap

    def row_mapping(self, row: int) -> tuple[dict[str, str], dict[str, str]]:
        """行槽位的替换映射。"""
        rs = self.row_slot
        if rs is None:
            return {}, {}
        sv = rs.values[row]
        pmap: dict[str, str] = {}
        rmap: dict[str, str] = {}
        for target, src in ((pmap, sv.prompt), (rmap, sv.response)):
            if isinstance(src, dict):
                target.update(src)
            else:
                target[ROW_SLOT] = src
        return pmap, rmap


@dataclass
class RenderedSample:
    prompt: str
    response: str
    meta: dict


# ---------------------------------------------------------------------------
# 全局（领域无关）变体词表
# ---------------------------------------------------------------------------
#: 礼貌前缀（用于祈使句题面，如"实现单链表反转。"）
POLITE_PREFIXES: tuple[str, ...] = ("", "请", "请帮", "能否", "麻烦", "我想请你")

#: 题面已经是完整问句时，疑问词已承担询问语气，**不能再加前缀**——
#: 否则会拼出"能否如何…"这种病句。改用另一组与之兼容的引导语。
#: 两组前缀共用同一个"礼貌度"维度（基数取二者长度的最大值），
#: 因此长度不必相等；:func:`_fit_prefix` 按题面形态选用其中一组。
ASKING_PREFIXES: tuple[str, ...] = ("", "", "", "", "", "")

#: 题面以问号收尾 -> 完全不加任何前缀
_ASKING_RE = re.compile(r"[?？]\s*$")
#: 题面结尾标点。后缀自带标点（"，谢谢。"），因此追加前必须先剥掉题面
#: 末尾的标点，否则会拼出"…的象征隐喻。，请给出完整答案。"这种病句。
#: 只在**句末**生效，不能动中间的 ``$O(n^2)$`` 或 ``a.b.c``。
_TRAILING_PUNCT_RE = re.compile(r"[。．.，,、；;：:!！?？~～\s]+$")
#: 题面以疑问词开头 -> 用 :data:`ASKING_PREFIXES` 而不是 :data:`POLITE_PREFIXES`
_STARTS_WITH_QWORD = ("如何", "为什么", "怎样", "什么时候", "是否", "哪个", "哪些",
                     "怎么", "多少")

#: 请求后缀。前缀为空表示"不追加"，其余都自带标点，
#: 因此追加前需要先剥掉题面末尾的标点，否则会出现"。。，"。
REQUEST_SUFFIXES: tuple[str, ...] = (
    "", "", "，谢谢。", "，请给出完整答案。", "，并简要说明理由。", "。")

#: 围栏风格。
#: 刻意**不含** ``indent``：缩进块没有行首锚点，围栏一旦被拆掉，
#: 后续的标点保护与空白折叠就再也认不出代码边界，
#: 结果是中文标点被灌进代码、代码缩进被压平（实测 python 块解析失败率 ~8%）。
#: 保留 backtick / tilde 两种，两者都由 :data:`_FENCE_RE` 锚定行首。
FENCE_STYLES: tuple[str, ...] = ("backtick", "tilde")

LATEX_STYLES: tuple[str, ...] = ("inline", "block")

PUNCT_STYLES: tuple[str, ...] = ("mixed", "cjk", "western")


# ---------------------------------------------------------------------------
# 格式变体
# ---------------------------------------------------------------------------
def _apply_code_fence(text: str, style: str) -> str:
    if style == "backtick":
        return text

    def repl(m: re.Match[str]) -> str:
        lang, body = m.group(1), m.group(2)
        if style == "tilde":
            return f"~~~{lang}\n{body.rstrip()}\n~~~"
        # indent: 四空格缩进块（去掉围栏，语言标注转成注释行）
        comment = f"# {lang}\n" if lang else ""
        indented = "".join(
            ("    " + ln[4:] if ln.startswith("    ") else ln) + "\n"
            for ln in body.rstrip("\n").split("\n"))
        return comment + indented.rstrip("\n")

    out = _FENCE_RE.sub(repl, text)
    if style == "indent" and not _looks_like_code(out):
        # 内容不是纯代码，套缩进块会破坏可读性
        return text
    return out


def _looks_like_code(text: str) -> bool:
    """判断文本是否整体就是一段代码（决定能否套用 indent 围栏变体）。

    散文里出现 ``{`` ``}`` ``;`` ``#`` 并不代表是代码；但如果**绝大多数非空行**
    都以常见代码特征开头（缩进、关键字、括号结尾），那基本可以确定。
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return False
    codey = sum(
        1 for ln in lines
        if ln.startswith((" ", "\t", "}", ")", "#", "from ", "import ", "def ",
                          "class ", "func ", "package ", "public ", "private "))
        or ln.rstrip().endswith((":", "{", "}", ");", "()", "]", ","))
    )
    return codey >= max(2, int(len(lines) * 0.6))


def _has_inline_code(text: str) -> bool:
    """散文里出现行内代码时，标点风格切换会污染代码内容，放弃切换。"""
    return bool(_INLINE_CODE_RE.search(text))


def _apply_latex(text: str, style: str) -> str:
    if style == "inline":
        return text
    return _INLINE_MATH_RE.sub(lambda m: f"$${m.group(1)}$$", text)


def _apply_punct(text: str, style: str) -> str:
    if style == "mixed":
        return text

    def outside_protected(s: str) -> str:
        """把代码块、行内代码与数学挖空，只对散文部分做替换。

        这些区域必须一起挖空：``$(3, -2)$`` 里的逗号是**千分位分隔符**，
        换成中文逗号会破坏 LaTeX。
        """
        # 一次遍历完成挖空：从后往前替换，偏移量才不会失效。
        # 占位符按**出现顺序**编号，这样最后的还原也是顺序的。
        holes: list[str] = []
        matches = list(_iter_protected(s))
        for i in range(len(matches) - 1, -1, -1):
            m = matches[i]
            holes.insert(0, m.group(0))
            s = s[:m.start()] + f"\x00{i}\x00" + s[m.end():]

        if style == "cjk":
            # 全角标点自带间距，因此吃掉其后的空格（中文排版惯例）
            s = re.sub(r",[ \t]*", "，", s)
            s = re.sub(r";[ \t]*", "；", s)
            s = re.sub(r":[ \t]*(?=\S)", "：", s)
        else:  # western
            # 半角标点不带间距，因此补回一个空格。
            # 但 `BETWEEN .. AND` 这类范围写法已被挖空保护，不会被波及。
            s = re.sub(r"，(?![ \t])", ", ", s)
            s = re.sub(r"，[ \t]+", " ", s)
            s = re.sub(r"；(?![ \t])", "; ", s)
            s = re.sub(r"；[ \t]+", " ", s)
            s = re.sub(r"：(?![ \t])", ": ", s)
            s = re.sub(r"：[ \t]+", " ", s)

        for i, h in enumerate(holes):
            s = s.replace(f"\x00{i}\x00", h)
        return s

    return outside_protected(text)


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def _fill(text: str, mapping: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        return mapping[name] if name in mapping else m.group(0)

    return _SLOT_RE.sub(repl, text)


#: 参与合并的句末标点
_PUNCT_CLASS = "。．.，,、；;：:!！?？~～"
#: 折叠连续重复的句末标点：``…解。。请给出`` -> ``…解。请给出``。
#: 只吃标点，**绝不跨空白/换行**——否则 ``def f(x):\n    y`` 的换行会被吞掉。
#: 纯 ASCII 点串（``...`` 省略号、``BETWEEN .. AND`` 的范围写法）不折叠。
_DUP_PUNCT_RE = re.compile(
    rf"(?<!\.)([{_PUNCT_CLASS}])(?:[{_PUNCT_CLASS}]*)(?!\.)")


def _fit_prefix(prefix_idx: int, body: str) -> str:
    """按题面形态选出合适的引导语。

    * 题面以 ? ？ 收尾 -> 空引导语（已是完整问句）；
    * 题面以疑问词开头（如何/为什么/是否…）-> 用 :data:`ASKING_PREFIXES`，
      否则"能否" + "如何…" 会拼成病句；
    * 其余（祈使句）-> 用 :data:`POLITE_PREFIXES`。
    """
    if _ASKING_RE.search(body) or body.startswith(_STARTS_WITH_QWORD):
        return ASKING_PREFIXES[prefix_idx % len(ASKING_PREFIXES)]
    return POLITE_PREFIXES[prefix_idx % len(POLITE_PREFIXES)]


#: 需要整体保护的区域：围栏代码块、行内代码、数学。
#: 与 :func:`_apply_punct` 共用同一套正则。
#: 省略号 / 范围写法：``...``、``..``。这些点**不是**重复标点，折叠会改变语义
#: （``BETWEEN .. AND`` 会变成 ``BETWEEN .AND``）。
_ELLIPSIS_RE = re.compile(r"\.{2,}")

_PROTECTED_RES = (_FENCE_RE, _SPACE_SENSITIVE_INLINE_RE, _INLINE_CODE_RE,
                  _ANY_MATH_RE, _INDENT_BLOCK_RE, _ELLIPSIS_RE)


def _tidy_punct(text: str) -> str:
    """合并重复标点：``解。。请给`` -> ``解。请给``，``字。。`` -> ``字。``。

    围栏代码块与行内代码**必须整体跳过**：
    ``BETWEEN .. AND`` 里的两个点是省略写法而非重复标点，
    折叠它会得到 ``BETWEEN .AND`` 从而改变语义。
    """
    out: list[str] = []
    pos = 0
    for m in _iter_protected(text):
        out.append(_tidy_prose(text[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(_tidy_prose(text[pos:]))
    return "".join(out)


def _iter_protected(text: str):
    """按出现顺序产出所有受保护片段（互不重叠）。

    重叠的片段里，**最长的优先**（围栏代码块 > 行内代码 > 数学），
    这样 ```` ```x, y``` ```` 不会被内部的 ``$x$`` 规则切碎。
    """
    cands = sorted(
        (m for rx in _PROTECTED_RES for m in rx.finditer(text)),
        key=lambda m: (m.start(), -(m.end() - m.start())),
    )
    taken: list[tuple[int, int]] = []
    for m in cands:
        if any(not (m.end() <= s or m.start() >= e) for s, e in taken):
            continue
        taken.append(m.span())
        yield m
    return


def _tidy_prose(block: str) -> str:
    """折叠散文片段里的重复标点，**但保留首尾换行结构**。

    换行必须原样保留：受保护片段（围栏代码块）用 ``^`` 锚定行首，
    如果这里把段尾的 ``\\n\\n`` 削成 ``""``，拼接后 ``` 会贴到正文后面，
    围栏不再位于行首，整段代码随即失去保护（标点被灌进代码里）。
    """
    nl = len(block) - len(block.rstrip("\n"))
    cut = len(block) - nl
    # core = 需要折叠的部分；tail = 段尾换行（必须原样保留）
    core, tail = block[:cut], block[cut:]
    # 循环到不动点：``。，，。`` 一次替换后仍可能残留相邻标点
    prev = None
    while prev != core:
        prev = core
        core = _DUP_PUNCT_RE.sub(r"\1", core)
    return core + tail


#: 句末**可选**的空白：折叠重复标点时允许吃掉这些空白。
#: 注意不含 ``\\n`` —— 吃掉换行会让围栏代码块失去行首锚点。
_TRAIL_PAD = " \t"


def render_row(
    template: Template,
    row: int,
    flavor_picks: dict[str, int],
    *,
    polite_idx: int,
    suffix: str,
    fence: str,
    latex: str,
    punct: str,
) -> RenderedSample:
    fpmap, frmap = template.flavor_mapping(flavor_picks)
    rpmap, rrmap = template.row_mapping(row)
    pmap = {**fpmap, **rpmap}
    rmap = {**frmap, **rrmap}

    body = _fill(template.prompt, pmap).strip()
    if suffix:
        # 后缀自带标点，先剥掉题面末尾标点，避免"隐喻。，请给出完整答案。"
        body = _TRAILING_PUNCT_RE.sub("", body)
    prompt = _tidy_punct(_fit_prefix(polite_idx, body) + body + suffix)
    response = _tidy_punct(_fill(template.response, rmap))

    # 顺序很重要：先标点（挖掉代码），再数学，再代码围栏
    response = _apply_punct(response, punct)
    response = _apply_latex(response, latex)
    response = _apply_code_fence(response, fence)
    prompt = _apply_punct(prompt, punct)

    return RenderedSample(
        prompt=prompt.strip(),
        response=response.strip(),
        meta={
            "row": row,
            "flavors": {k: v for k, v in sorted(fpmap.items())},
            "styles": {"fence": fence, "latex": latex, "punct": punct},
        },
    )


# ---------------------------------------------------------------------------
# 组合空间：行索引 × 风格维度
# ---------------------------------------------------------------------------
def decompose(idx: int, radices: Sequence[int]) -> list[int]:
    """把一个混合进制整数拆成各维度的选择下标。"""
    out: list[int] = []
    for r in radices:
        out.append(idx % r)
        idx //= r
    return out


@dataclass(frozen=True)
class VariantSpace:
    """某领域下所有模板的组合空间。

    每个模板有**自己的**索引空间（模板自身不共享风味槽位的基数），
    领域总容量为各模板容量之和。这样两个模板可以各自声明
    ``figure`` 并带不同数量的取值，而不会像共享索引空间那样错位。

    维度顺序：``row`` -> 各风味槽位 -> 格式变体（polite/suffix/fence/latex/punct）。
    全部维度之间是笛卡尔积。
    """

    spaces: tuple[tuple[Template, tuple[int, ...], tuple[str, ...]], ...]

    @property
    def capacity(self) -> int:
        return sum(_prod(r) for _, r, _ in self.spaces)

    @property
    def n_rows(self) -> int:
        return sum(r[0] for _, r, _ in self.spaces)

    @property
    def n_styles(self) -> int:
        """风格组合数（各模板相同，取第一个作为代表）。"""
        if not self.spaces:
            return 1
        return _prod(self.spaces[0][1][1:])

    def describe(self) -> list[str]:
        out = []
        for t, radices, axes in self.spaces:
            out.append(f"{t.domain}#{t.prompt[:24]!r} {list(axes)}")
        return out


def _prod(xs) -> int:
    n = 1
    for x in xs:
        n *= x
    return n


def build_space(
    templates: Sequence[Template],
    cfg: PerturbConfig,
) -> VariantSpace:
    """按模板权重重复展开，为每个模板建立独立的索引空间。"""
    spaces = []
    for t in templates:
        t.validate()
        radices: list[int] = [t.n_rows]
        axes: list[str] = ["row"]
        for s in t.flavor_slots:
            radices.append(len(s.values))
            axes.append("flavor:" + s.name)
        if cfg.polite_prefix:
            # 两组前缀共用一个维度，故基数取二者长度的最大值
            radices.append(max(len(POLITE_PREFIXES), len(ASKING_PREFIXES)))
            axes.append("polite")
        if cfg.request_suffix:
            radices.append(len(REQUEST_SUFFIXES)); axes.append("suffix")
        if cfg.code_fence:
            radices.append(len(FENCE_STYLES)); axes.append("fence")
        if cfg.latex_delim:
            radices.append(len(LATEX_STYLES)); axes.append("latex")
        if cfg.punctuation_style:
            radices.append(len(PUNCT_STYLES)); axes.append("punct")
        reps = max(1, round(t.weight))
        spaces.extend([(t, tuple(radices), tuple(axes))] * reps)
    return VariantSpace(tuple(spaces))


def _spans(space: VariantSpace) -> list[tuple[int, int]]:
    """各模板空间在 0..capacity 区间上的 [start, end) 偏移。"""
    out, acc = [], 0
    for _, radices, _ in space.spaces:
        n = _prod(radices)
        out.append((acc, acc + n))
        acc += n
    return out


def render_index(space: VariantSpace, idx: int, cfg: PerturbConfig,
                 spans: list[tuple[int, int]]) -> RenderedSample:
    # 定位 idx 落在哪个模板空间
    k = 0
    for j, (start, end) in enumerate(spans):
        if start <= idx < end:
            k = j
            idx -= start
            break
    t, radices, _ = space.spaces[k]
    combo = decompose(idx, radices)

    row = combo.pop(0)
    flavor_picks = {s.name: combo.pop(0) for s in t.flavor_slots}

    polite_idx = 0
    suffix = ""
    fence, latex, punct = "backtick", "inline", "mixed"
    it = iter(combo)
    if cfg.polite_prefix:
        polite_idx = next(it)
    if cfg.request_suffix:
        suffix = REQUEST_SUFFIXES[next(it)]
    if cfg.code_fence:
        fence = FENCE_STYLES[next(it)]
    if cfg.latex_delim:
        latex = LATEX_STYLES[next(it)]
    if cfg.punctuation_style:
        punct = PUNCT_STYLES[next(it)]

    return render_row(t, row, flavor_picks, polite_idx=polite_idx,
                      suffix=suffix, fence=fence, latex=latex, punct=punct)


def sample_domain(
    templates: Sequence[Template],
    cfg: PerturbConfig,
    count: int,
    rng: random.Random,
) -> Iterator[RenderedSample]:
    """从一个领域的组合空间里无放回地抽 ``count`` 个唯一样本。

    组合空间不足时抛错——宁可让作者去加模板，也不要静默产出重复数据。
    """
    space = build_space(templates, cfg)
    spans = _spans(space)
    if space.capacity < count:
        raise ValueError(
            f"组合空间不足：capacity={space.capacity} < 需要 {count}。"
            f"各模板维度={space.describe()}；"
            f"请为该领域补充模板行数或减少格式变体维度")

    # 无放回：把索引空间整体打乱后依次消费。
    # 必须允许消费超过 count 个索引——不同组合可能渲染出相同文本
    # （例如某领域没有 $...$，latex 维度就是空操作），只取前 count 个会漏。
    order = list(range(space.capacity))
    rng.shuffle(order)

    # ★两轮采样：先只要"新内容"，再退让到"新字节"。
    #
    # 只按 (prompt, response) 去重会把**纯装饰变体**算成新样本：
    # 同一个 case 换 ``` / ~~~ 围栏、换全角/半角标点，
    # 字节不同但内容完全一样。于是 200 条的配额会被几十个真 case
    # 的换皮版本吃光——实测题面唯一率只有 91.4%（远低于 95% 的要求），
    # 而字节级去重却报 100%，把问题盖住了。
    #
    # 第一轮用 :func:`norm_content` 抹掉装饰差异后再判重，
    # 保证优先消耗掉真正的新内容；只有当内容容量不够时才用第二轮补齐。
    seen_bytes: set[tuple[str, str]] = set()
    seen_content: set[tuple[str, str]] = set()
    produced = 0

    for content_only in (True, False):
        for idx in order:
            if produced >= count:
                return
            s = render_index(space, idx, cfg, spans)
            key = (s.prompt, s.response)
            if key in seen_bytes:
                continue
            if content_only:
                ckey = (norm_content(s.prompt), norm_content(s.response))
                if ckey in seen_content:
                    continue
                seen_content.add(ckey)
            else:
                # 第二轮：内容已穷尽，允许装饰变体补齐配额
                seen_content.add((norm_content(s.prompt),
                                  norm_content(s.response)))
            seen_bytes.add(key)
            produced += 1
            yield s

    raise ValueError(
        f"整个组合空间（{space.capacity} 个索引）去重后只有 {produced} 条唯一样本，"
        f"少于目标 {count}。行数={space.n_rows}，风格组合={space.n_styles}。"
        f"请检查：1) 槽位取值是否有重复；"
        f"2) 风格变体对该领域是否为空操作"
        f"（例如纯代码领域没有 $...$ 数学，latex 维度不产生差异）")
