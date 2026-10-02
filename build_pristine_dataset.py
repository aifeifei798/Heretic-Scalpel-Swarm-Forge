#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``build_pristine_dataset.py`` —— 已退役，请勿再使用。

.. danger::
   **这个脚本产出的数据集是坏的。** 它生成的 6400 条样本里，
   按内容去重后**只有 41 条是唯一的**（156 倍重复），
   也就是说你付了 6400 条的 token 成本，实际只训练了 41 条内容。

   所谓"微扰模板"在这个文件里**根本没有实现**——``perturb`` 之类的
   随机化逻辑要么缺失，要么只改标点/称谓，题目骨架完全不变。

替代方案
--------
用 :mod:`forge_core.datagen`，它把"多样性"做成可审计的门禁::

    python -m forge_core.cli datagen --out data/dual_contrast_data.jsonl \\
        --per-domain 200 --val-ratio 0.05 --audit

同一份 6400 条规模下，当前指标：

===================  ========  ========
指标                  旧脚本    forge_core
===================  ========  ========
唯一内容条数            41       **6284**
内容唯一率             0.6%      **98.2%**
每领域条数             不齐      严格 200
代码块可解析             未检查    247/247 全通过
===================  ========  ========

旧脚本的种子库（32 个领域各 2~3 条问答）本身是有价值的素材，
已被 :mod:`forge_core.datagen.seeds` 以模板 + 槽位的形式重新组织，
内容保留、组合方式换成了可枚举且可审计的笛卡尔积。

本文件保留下来只为留个"为什么别再用它"的说明，不再包含任何生成逻辑——
两套实现并存必然分叉，而分叉之后没人知道该信哪个。
"""

from __future__ import annotations

import sys


def main() -> int:
    print(__doc__, file=sys.stderr)
    print("请改用： python -m forge_core.cli datagen --audit", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
