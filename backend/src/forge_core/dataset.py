"""数据集：tokenize + assistant-mask + 长度分桶。

修复记录
--------
* **B4 prompt 被当标签监督**：原实现 ``labels = input_ids.clone()`` 只屏蔽
  padding，导致 ``<|turn>user\\n...`` 整段题面都被当作预测目标。
  现在用 :func:`find_response_start` 定位 assistant 段起点，之前的 token 置 -100。
* **B5 87% 算力浪费**：实测样本真实长度约 66~150 token，却 ``padding="max_length"``
  ��� 512，6400x512 = 328 万 token 里只有约 42 万是真实内容。
  现在默认动态 padding + 长度分桶。
* **B9 tokenizer 不一致**：``fix_mistral_regex`` 之前只在 chat 分支传，
  训练与推理 tokenization 不一致。统一到 :func:`load_tokenizer`。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .schema import ALL_DOMAINS, DOMAIN_INDEX, ArchProfile


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------
def load_tokenizer(base_model_id: str, *, trust_remote_code: bool = True):
    """唯一的 tokenizer 加载入口。

    ``fix_mistral_regex=True`` 必须在这里固定：transformers 会检查
    Mistral 的 tokenization regex，若不修则**训练与推理的切词结果不同**，
    这是 B9 的根因。
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=trust_remote_code,
        fix_mistral_regex=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# assistant 起点定位
# ---------------------------------------------------------------------------
#: 常见 chat template 的 assistant 起始标记。
#: Gemma 系列用 ``<|turn>model\\n``，Llama 用 ``\\n\\nassistant\\n`` 等。
_RESPONSE_MARKERS: tuple[str, ...] = (
    "<|turn>model\n",
    "<|start_header_id|>assistant<|end_header_id|>",
    "<|im_start|>assistant\n",
    "### Assistant:",
    "\n\nassistant\n",
    "\nAssistant:",
    "assistant\n",
)


def find_response_start(text: str, markers: Sequence[str] = _RESPONSE_MARKERS
                        ) -> tuple[int, bool]:
    """返回 assistant 回答在 *text* 中的字符起点，以及是否定位成功。

    定位失败时返回 ``(0, False)``，调用方应退化为"监督全部 token"
    并告警——宁可多监督，也不要静默地把监督信号打在题面上。
    """
    best = -1
    best_marker = ""
    for m in markers:
        i = text.find(m)
        if i >= 0 and (best < 0 or i < best):
            best, best_marker = i, m
    if best < 0:
        return 0, False
    return best + len(best_marker), True


# ---------------------------------------------------------------------------
# 路由目标
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RoutingTargets:
    """一条样本的路由监督目标。"""

    domain: str
    macro: int      # ∈ [0, M)
    micro: int      # ∈ [0, N)，默认与 domain_id 相同

    @property
    def macro_name(self) -> str:
        return self.domain


def resolve_targets(rows: Sequence[dict[str, Any]],
                   arch: ArchProfile) -> list[RoutingTargets]:
    """由 ``domain`` + ``domain_to_macro`` 映射表算出路由目标。

    这是 A2 的修复点：原实现把 ``big_target`` 写死成 0/1（文科/理科），
    配上 ``num_macro_cores=4`` 时第 3、4 号大核**永远拿不到正样本**，
    被 router 压到 -inf 后永不激活。现在目标取值范围是 ``[0, M)``，
    改大核数量无需改数据。

    ★``micro`` 必须对 ``num_micro_experts`` 取模。
    小核目标原本直接用领域的全局下标 ``DOMAIN_INDEX[d]``（0~31），
    与 ``num_micro_experts`` 毫无关系。只要用户把 N 配成 < 32
    （网页上这是个自由输入框，绝大多数选择都 < 32），
    目标就会超出 router 输出维度，
    ``cross_entropy`` 抛 ``device-side assert``——
    而且只在**第一个 batch**炸，C++ 层的断言还会把整条栈指向
    错误的位置（实测报在 ``repeat_interleave`` 上，与真正原因无关）。
    """
    n_micro = arch.num_micro_experts
    if n_micro < 1:
        raise ValueError(f"num_micro_experts 必须 >= 1，当前 {n_micro}")
    n_macro = arch.num_macro_cores

    out = []
    for r in rows:
        d = r["domain"]
        if d not in DOMAIN_INDEX:
            raise KeyError(f"未知领域 {d!r}；合法领域见 schema.ALL_DOMAINS")
        macro = arch.domain_to_macro[d]
        if not 0 <= macro < n_macro:
            raise ValueError(
                f"领域 {d} 映射到大核 {macro}，超出 [0, {n_macro})。"
                "请检查 domain_to_macro（UI 上拖拽后应重新保存）。")
        out.append(RoutingTargets(
            domain=d,
            macro=macro,
            micro=DOMAIN_INDEX[d] % n_micro,
        ))
    return out


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
class SwarmDataset:
    """带 assistant-mask 的因果语言模型数据集。"""

    def __init__(self, rows: list[dict[str, Any]], tokenizer,
                 arch: ArchProfile, *, max_length: int = 256,
                 with_labels: bool = True) -> None:
        self.tok = tokenizer
        self.max_length = max_length
        self.with_labels = with_labels
        self.rows = rows
        self.targets = resolve_targets(rows, arch)
        self.mask_warnings = 0
        self._cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        if i in self._cache:
            return self._cache[i]
        item = self._encode(i)
        self._cache[i] = item
        return item

    # -- 编码 ---------------------------------------------------------
    def _encode(self, i: int) -> dict[str, Any]:
        import torch

        row = self.rows[i]

        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
        text = self.tok.apply_chat_template(messages, tokenize=False)
        resp_start, ok = find_response_start(text)
        if not ok:
            self.mask_warnings += 1
            resp_start = 0

        enc = self.tok(text, max_length=self.max_length, truncation=True,
                       padding=False, return_tensors="pt")
        input_ids = enc["input_ids"][0]

        labels = input_ids.clone()
        if self.with_labels:
            # 只监督 assistant 段（B4 的修复）
            n_prompt = self._count_tokens_upto(text, resp_start, input_ids)
            if n_prompt:
                labels[:n_prompt] = -100

        return {"input_ids": input_ids, "labels": labels, "index": i}

    def _count_tokens_upto(self, text: str, char_pos: int, input_ids) -> int:
        """求"题面部分"对应的 token 数。

        直接对题面前缀单独 tokenize 比切字符串更稳（避免 BPE 边界效应）。
        """
        n = len(self.tok(text[:char_pos], add_special_tokens=True)["input_ids"])
        return min(n, len(input_ids) - 1)


def collate(batch: list[dict[str, Any]], targets: list[RoutingTargets],
            pad_token_id: int) -> dict[str, Any]:
    """动态 padding 的 collate（B5 的修复）。

    只 pad 到 batch 内最长，而非 ``max_length``。配合长度分桶后，
    padding 浪费从 ~87% 降到通常 <5%。

    ``targets`` 必须是**按 batch 内顺序**排列的目标列表；
    若传 ``None``，则按样本自带的 ``index`` 从全局 ``targets`` 里取。
    """
    import torch

    if targets is None:
        raise ValueError("collate 需要 targets；用 collate_from_dataset 构造")

    n = len(batch)
    maxlen = max(int(b["input_ids"].shape[0]) for b in batch)
    input_ids = torch.full((n, maxlen), pad_token_id, dtype=torch.long)
    labels = torch.full((n, maxlen), -100, dtype=torch.long)
    attn = torch.zeros((n, maxlen), dtype=torch.long)

    for i, b in enumerate(batch):
        L = int(b["input_ids"].shape[0])
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
        attn[i, :L] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn,
        "macro_target": torch.tensor([t.macro for t in targets], dtype=torch.long),
        "micro_target": torch.tensor([t.micro for t in targets], dtype=torch.long),
    }


def make_collate(dataset: "SwarmDataset", pad_token_id: int):
    """返回一个按 batch 内顺序自动取 targets 的 collate 函数。"""
    tgts = dataset.targets

    def _fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return collate(batch, [tgts[b["index"]] for b in batch], pad_token_id)

    return _fn


# ---------------------------------------------------------------------------
# 长度分桶采样器
# ---------------------------------------------------------------------------
def length_grouped_batches(lengths: Sequence[int], batch_size: int, *,
                            rng: random.Random, mega: int = 64) -> list[list[int]]:
    """把长度相近的样本放进同一 batch，显著减少 padding 浪费。

    做法：全局随机打乱 -> 按 ``mega * batch_size`` 分大块 ->
    块内按长度排序 -> 切成 batch。

    注意这是 **shuffle 后再排序**，因此 batch 之间仍是随机的；
    但同一 batch 内样本高度相关，可能引入梯度相关性。
    因此 :class:`SwarmTrainer` 里仍需配合正常的 batch 顺序打乱。
    """
    n = len(lengths)
    idx = list(range(n))
    rng.shuffle(idx)

    mega_size = mega * batch_size
    batches: list[list[int]] = []
    for s in range(0, n, mega_size):
        chunk = sorted(idx[s:s + mega_size], key=lambda i: lengths[i])
        for b in range(0, len(chunk), batch_size):
            batches.append(chunk[b:b + batch_size])
    rng.shuffle(batches)
    return batches


# ---------------------------------------------------------------------------
# 载入
# ---------------------------------------------------------------------------
def load_rows(path: str | Path, *, split: str | None = None,
              limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if split and r.get("split") != split:
                continue
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def char_lengths(rows: Sequence[dict[str, Any]]) -> list[int]:
    """字符长度代理指标（快速估算 batch 分组用，不需 tokenize）。"""
    return [len(r["prompt"]) + len(r["response"]) for r in rows]


__all__ = [
    "SwarmDataset", "RoutingTargets", "collate", "load_rows",
    "length_grouped_batches", "char_lengths", "load_tokenizer",
    "find_response_start", "resolve_targets", "ALL_DOMAINS", "DOMAIN_INDEX",
]