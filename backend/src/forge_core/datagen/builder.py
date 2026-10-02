"""数据集构建器：遍历 32 个领域，采样、去重、分层切分、落盘 + 统计。"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from ..schema import (
    ALL_DOMAINS,
    ARTS_DOMAINS,
    DOMAIN_INDEX,
    SCI_DOMAINS,
    DataGenConfig,
    default_domain_to_macro,
)
from .perturb import RenderedSample, sample_domain
from .sanitize import is_clean, sanitize_text
from .seeds import DOMAIN_TEMPLATES

Emit = Callable[[dict[str, Any]], None]


def _noop(_: dict[str, Any]) -> None:  # pragma: no cover
    pass


def _row_id(prompt: str, response: str) -> str:
    h = hashlib.sha256(f"{prompt}\x00{response}".encode("utf-8")).hexdigest()
    return f"sha1:{h[:20]}"


@dataclass
class BuildReport:
    output_path: str
    total_rows: int
    unique_rows: int
    unique_ratio: float
    per_domain: dict[str, int]
    train_rows: int
    val_rows: int
    capacity_per_domain: dict[str, int]
    duration_sec: float
    dirty_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "total_rows": self.total_rows,
            "unique_rows": self.unique_rows,
            "unique_ratio": round(self.unique_ratio, 6),
            "train_rows": self.train_rows,
            "val_rows": self.val_rows,
            "dirty_rows": self.dirty_rows,
            "duration_sec": round(self.duration_sec, 2),
            "per_domain": dict(sorted(self.per_domain.items(),
                                      key=lambda kv: DOMAIN_INDEX[kv[0]])),
            "capacity_per_domain": self.capacity_per_domain,
            "macro_coverage": self._macro_coverage(),
        }

    def _macro_coverage(self) -> dict[str, Any]:
        """按默认 4 大核分组统计每组领域数与样本数 —— A2/B2 的预检。"""
        mapping = default_domain_to_macro()
        per_macro_domains: dict[int, list[str]] = {}
        for d, m in mapping.items():
            per_macro_domains.setdefault(m, []).append(d)
        return {
            str(m): {
                "domains": len(ds),
                "rows": sum(self.per_domain.get(d, 0) for d in ds),
            }
            for m, ds in sorted(per_macro_domains.items())
        }


def build_dataset(cfg: DataGenConfig,
                  *,
                  emit: Emit = _noop,
                  domains: Iterable[str] | None = None) -> BuildReport:
    """按配置生成数据集并写入 jsonl。

    Raises:
        ValueError: 唯一率低于 ``cfg.min_unique_ratio``，或某领域组合空间不足。
    """
    t0 = time.time()
    rng = random.Random(cfg.seed)
    target_domains = list(domains) if domains is not None else list(ALL_DOMAINS)

    emit({"t": "datagen_start", "domains": len(target_domains),
          "samples_per_domain": cfg.samples_per_domain, "seed": cfg.seed})

    rows: list[dict[str, Any]] = []
    per_domain: Counter[str] = Counter()
    capacity: dict[str, int] = {}
    dirty = 0
    global_seen: set[str] = set()

    for domain in target_domains:
        templates = DOMAIN_TEMPLATES.get(domain)
        if not templates:
            raise KeyError(f"领域 {domain} 没有对应模板；"
                           f"可用领域: {sorted(DOMAIN_TEMPLATES)}")

        got = 0
        for sample in sample_domain(templates, cfg.perturb,
                                    cfg.samples_per_domain, rng):
            prompt = sanitize_text(sample.prompt)
            response = sanitize_text(sample.response)
            if not prompt or not response:
                continue
            if not (is_clean(prompt) and is_clean(response)):
                dirty += 1

            rid = _row_id(prompt, response)
            if rid in global_seen:          # 跨领域兜底去重
                continue
            global_seen.add(rid)

            rows.append({
                "id": rid,
                "domain": domain,
                "domain_id": DOMAIN_INDEX[domain],
                "group": "arts" if domain in ARTS_DOMAINS else "sci",
                "prompt": prompt,
                "response": response,
                "meta": sample.meta,
            })
            per_domain[domain] += 1
            got += 1

        from .perturb import build_space
        space = build_space(templates, cfg.perturb)
        capacity[domain] = space.capacity
        emit({"t": "domain_done", "domain": domain, "rows": got,
              "capacity": capacity[domain], "total": len(rows)})

    # --- 按领域分层切分 train/val -------------------------------------
    n_val_target = int(len(rows) * cfg.val_ratio)
    val_ids: set[str] = set()
    if n_val_target > 0:
        by_domain: dict[str, list[dict]] = {}
        for r in rows:
            by_domain.setdefault(r["domain"], []).append(r)
        # 每领域均匀取 ceil(ratio * n) 条，且至少 1 条（领域数 >1 时）
        for domain, items in by_domain.items():
            k = min(len(items), max(1, round(len(items) * cfg.val_ratio)))
            val_ids.update(r["id"] for r in items[:k])
    for r in rows:
        r["split"] = "val" if r["id"] in val_ids else "train"

    # --- 顺序打散 -------------------------------------------------------
    shuffle_rng = random.Random(cfg.seed + 1)
    shuffle_rng.shuffle(rows)

    # --- 落盘 -----------------------------------------------------------
    out = Path(cfg.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    unique_ratio = len(global_seen) / len(rows) if rows else 0.0
    report = BuildReport(
        output_path=str(out),
        total_rows=len(rows),
        unique_rows=len(global_seen),
        unique_ratio=unique_ratio,
        per_domain=dict(per_domain),
        train_rows=sum(1 for r in rows if r["split"] == "train"),
        val_rows=sum(1 for r in rows if r["split"] == "val"),
        capacity_per_domain=capacity,
        duration_sec=time.time() - t0,
        dirty_rows=dirty,
    )

    stats_path = cfg.resolved_stats_path()
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    emit({"t": "datagen_done", **report.to_dict()})

    if unique_ratio < cfg.min_unique_ratio:
        raise ValueError(
            f"唯一率 {unique_ratio:.4f} 低于下限 {cfg.min_unique_ratio:.4f}，"
            f"已写出 {out} 供排查，但请勿用于训练。"
            f"常见原因：某领域的槽位取值存在重复或渲染为相同文本。")

    return report


def preview(path: str | Path, limit: int = 20,
            domains: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """读取数据集前若干条（可选按领域过滤），供 CLI / API 预览。"""
    want = set(domains) if domains else None
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if want and row.get("domain") not in want:
                continue
            out.append(row)
            if len(out) >= limit:
                break
    return out


def length_stats(path: str | Path) -> dict[str, Any]:
    """字符长度分位数（粗粒度，不需 tokenizer，API 侧快速预览用）。"""
    lens: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            lens.append(len(r["prompt"]) + len(r["response"]))
    if not lens:
        return {"count": 0}
    lens.sort()

    def pct(p: float) -> int:
        return lens[min(len(lens) - 1, int(p * len(lens)))]

    return {
        "count": len(lens),
        "min": lens[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": lens[-1],
    }


__all__ = ["BuildReport", "build_dataset", "preview", "length_stats",
           "DOMAIN_TEMPLATES", "ARTS_DOMAINS", "SCI_DOMAINS"]
