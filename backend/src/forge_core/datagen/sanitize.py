"""文本清洗。

修复 A4：原实现是"先剥标签 -> 再反转义实体"，于是 ``&lt;script&gt;`` 被还原成
真正的 ``<script>`` 反而存活下来；而且 ``re.sub(r"<[^>]+>")`` 会误吃代码/LaTeX
里合法的尖括号（泛型 ``List<T>``、HTML 教学语料本身）。

正确顺序：
  1. 先反转义实体（把 ``&lt;`` 变回 ``<``）
  2. 删除成对的 ``<script>`` / ``<style>`` **内容块**（含标签本身）
  3. 再剥剩余标签
  4. 折叠空白
"""

from __future__ import annotations

import html
import re

_SCRIPT_BLOCK_RE = re.compile(
    r"<\s*(script|style|noscript|iframe)\b[^>]*>.*?<\s*/\s*\1\s*>",
    re.S | re.I)
_UNCLOSED_SCRIPT_RE = re.compile(r"<\s*(script|style)\b[^>]*>.*\Z", re.S | re.I)
#: HTML 标签。必须能区分**标签**与**泛型**：
#: ``<div class="x">`` 是标签，而 ``<i32>`` / ``<T>`` / ``Vec<i32>`` 是类型。
#: 判据：HTML 标签名只会出现在一小撮白名单里（真实的网页标签集合），
#: 而泛型可以是任意标识符。用白名单而不是"看起来像标签"来判定，
#: 才不会把 ``Vec<i32>`` 削成 ``Vec``。
_HTML_TAG_NAMES = (
    "a|abbr|address|area|article|aside|audio|b|base|bdi|bdo|big|blockquote|body|br|"
    "button|canvas|caption|center|cite|code|col|colgroup|data|datalist|dd|del|details|"
    "dfn|dialog|div|dl|dt|em|embed|fieldset|figcaption|figure|footer|form|h1|h2|h3|"
    "h4|h5|h6|head|header|hgroup|hr|html|i|iframe|img|input|ins|kbd|label|legend|li|"
    "link|main|map|mark|menu|meta|meter|nav|noscript|object|ol|optgroup|option|output|"
    "p|picture|pre|progress|q|rp|rt|ruby|s|samp|script|section|select|slot|small|source|"
    "span|strong|style|sub|summary|sup|table|tbody|td|template|textarea|tfoot|th|thead|"
    "time|title|tr|track|u|ul|var|video|wbr"
)
_TAG_RE = re.compile(
    rf"</?(?:{_HTML_TAG_NAMES})(?:\s[^<>]*?)?/?>", re.I)
#: 行内连续空白（不含换行）
_BLANK_RE = re.compile(r"[ \t\x0b\f\r]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
#: 围栏代码块（缩进必须原样保留的区域）
#: 围栏代码块（缩进必须原样保留的区域）。`` 与 ``` 两种围栏都要覆盖，
#: 因为 perturb 的 ``tilde`` 风格会把 ``` 改写成 ~~~。
_FENCE_RE = re.compile(r"^(?:```|~~~)[\w+-]*[ \t]*\n.*?^(?:```|~~~)[ \t]*$",
                       re.S | re.M)
_MD_LINK_RE = re.compile(r"!?\[([^\]\n]{0,200})\]\((?:[^)\n]{0,400})\)")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")


def _fold_prose_line(line: str) -> str:
    """折叠一行**散文**：行尾去空白、行内连续空白压成一个。

    缩进一律清掉——散文不需要缩进，保留它只会让文本看起来像被截断的代码。
    """
    return _BLANK_RE.sub(" ", line).strip()


def _fold_whitespace(text: str) -> str:
    """折叠空白，但**围栏代码块内的缩进必须逐字保留**。

    这是最容易踩的坑：全局 ``re.sub(r"[ \\t]+", " ", text)`` 会把
    ``    return y`` 变成 `` return y``，让所有 Python 片段失去缩进语义、
    直接无法解析。因此这里按"围栏内 / 围栏外"分别处理。
    """
    out: list[str] = []
    pos = 0
    for m in _FENCE_RE.finditer(text):
        out.append(_fold_prose_block(text[pos:m.start()]))
        out.append(m.group(0))          # 代码块原样保留
        pos = m.end()
    out.append(_fold_prose_block(text[pos:]))
    return "".join(out)


def _fold_prose_block(block: str) -> str:
    return "".join(_fold_prose_line(ln) + "\n" for ln in block.split("\n"))


def sanitize_text(text: str) -> str:
    """把一条 prompt/response 清洗成干净纯文本。"""
    if not text:
        return ""

    # 0) 去掉零宽字符（爬取脏数据的典型残留）
    text = _ZERO_WIDTH_RE.sub("", text)

    # 1) 先反转义实体（顺序关键：否则 &lt;script&gt; 会被还原成真标签）
    text = html.unescape(text)

    # 2) 删除脚本/样式整块
    text = _SCRIPT_BLOCK_RE.sub(" ", text)
    text = _UNCLOSED_SCRIPT_RE.sub(" ", text)

    # 3) 剥剩余标签（保留泛型 List<T> 这类非 HTML 尖括号）
    text = _TAG_RE.sub(" ", text)
    text = text.replace("[/rawhtml]", " ")

    # 4) markdown 图片降级为纯文本，链接保留可见文字
    text = _MD_LINK_RE.sub(r"\1", text)

    # 5) 折叠空白 —— 围栏代码块内的缩进逐字保留（见 _fold_whitespace）
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _fold_whitespace(text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def is_clean(text: str) -> bool:
    """校验一条文本确实没有 HTML/脚本残留（用于生成后的自检）。"""
    if _TAG_RE.search(text):
        return False
    lowered = text.lower()
    return "<script" not in lowered and "<style" not in lowered
