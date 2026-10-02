/** 领域 → 大核 的拖拽分配器。
 *
 * 为什么用 HTML5 drag & drop 而不是 dnd-kit
 * ---------------------------------------
 * 这个交互只做一件事：把 32 个标签拖到 M 个桶里。不需要排序预览、
 * 多列重排、跨列表动画——为此引入一个拖拽库（~30KB + 它自己的
 * 一堆 React 兼容性问题）不划算。原生 DnD 对键盘不友好，
 * 所以**每个桶同时提供下拉框**，键盘用户和鼠标用户等价。
 */

import { useMemo, useState } from "react";

export interface MacroCore {
  index: number;
  name: string;
}

interface Props {
  domains: string[];
  macroNames: string[];
  mapping: Record<string, number>;
  onChange: (next: Record<string, number>) => void;
}

/** 领域名的可读前缀：Arts_Poetry -> Arts */
function groupOf(domain: string): string {
  const i = domain.indexOf("_");
  return i < 0 ? domain : domain.slice(0, i);
}

const GROUP_LABEL: Record<string, string> = {
  Arts: "人文艺术",
  Code: "代码工程",
  Math: "数学",
  Sci: "自然科学",
};

export function ArchDesigner({ domains, macroNames, mapping, onChange }: Props) {
  const [dragDomain, setDragDomain] = useState<string | null>(null);
  const [overCore, setOverCore] = useState<number | null>(null);
  const [filter, setFilter] = useState("");

  const assigned = useMemo(() => {
    const by = new Map<number, string[]>();
    for (let i = 0; i < macroNames.length; i++) by.set(i, []);
    for (const d of domains) {
      const m = mapping[d];
      if (m != null && by.has(m)) by.get(m)!.push(d);
    }
    return by;
  }, [domains, mapping, macroNames]);

  const unassigned = useMemo(
    () => domains.filter((d) => mapping[d] == null),
    [domains, mapping],
  );

  const counts = useMemo(
    () => macroNames.map((_, i) => (assigned.get(i) ?? []).length),
    [assigned, macroNames],
  );

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    const all = [...assigned.values()].flat();
    return q ? all.filter((d) => d.toLowerCase().includes(q)) : all;
  }, [assigned, filter]);

  function assign(domain: string, core: number) {
    if (core < 0 || core >= macroNames.length) return;
    onChange({ ...mapping, [domain]: core });
  }

  /** 每核领域数差异过大时提示——不阻止，但值得让人看见。 */
  const imbalance = useMemo(() => {
    const n = counts.filter((c) => c > 0);
    if (n.length < 2) return 0;
    return Math.max(...n) - Math.min(...n);
  }, [counts]);

  return (
    <div className="designer">
      <div className="designer__toolbar">
        <input
          className="input"
          placeholder="筛选领域…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        <span className="hint">
          拖动标签到大核，或用下拉框分配。
          {unassigned.length > 0 && (
            <strong className="warn"> {unassigned.length} 个未分配</strong>
          )}
          {imbalance > 4 && (
            <strong className="warn"> 各核领域数相差 {imbalance}，可能不均衡</strong>
          )}
        </span>
      </div>

      <div className="cores">
        {macroNames.map((name, i) => (
          <div
            key={i}
            className={`core${overCore === i ? " core--over" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setOverCore(i);
            }}
            onDragLeave={() => setOverCore((c) => (c === i ? null : c))}
            onDrop={(e) => {
              e.preventDefault();
              setOverCore(null);
              const d = e.dataTransfer.getData("text/plain") || dragDomain;
              if (d) assign(d, i);
              setDragDomain(null);
            }}
          >
            <header className="core__head">
              <span className="core__name">
                <b>#{i}</b> {name}
              </span>
              <span className="badge">{counts[i]} 个领域</span>
            </header>

            <div className="core__body">
              {(assigned.get(i) ?? [])
                .filter((d) => !filter || d.toLowerCase().includes(filter.toLowerCase()))
                .map((d) => (
                  <span
                    key={d}
                    className="chip"
                    draggable
                    onDragStart={(e) => {
                      setDragDomain(d);
                      e.dataTransfer.setData("text/plain", d);
                    }}
                    title={`${d} → 拖到别的大核`}
                  >
                    {d}
                  </span>
                ))}
              {(assigned.get(i) ?? []).length === 0 && (
                <p className="empty">把领域拖到这里</p>
              )}
            </div>

            <select
              className="input input--sm"
              value=""
              onChange={(e) => {
                const d = e.target.value;
                if (d) assign(d, i);
              }}
            >
              <option value="">+ 分配领域到本核…</option>
              {[...assigned.values()]
                .flat()
                .filter((d) => mapping[d] !== i)
                .map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
            </select>
          </div>
        ))}
      </div>

      <div className="groups">
        {["Arts", "Code", "Math", "Sci"].map((g) => {
          const items = filtered.filter((d) => groupOf(d) === g);
          if (!items.length) return null;
          return (
            <div key={g} className="group">
              <h4>
                {GROUP_LABEL[g] ?? g} <span className="muted">{items.length}</span>
              </h4>
              <div className="group__chips">
                {items.map((d) => (
                  <span
                    key={d}
                    className="chip chip--sm"
                    draggable
                    onDragStart={(e) => {
                      setDragDomain(d);
                      e.dataTransfer.setData("text/plain", d);
                    }}
                    title={`当前 → #${mapping[d] ?? "?"}`}
                  >
                    {d.split("_").slice(1).join("_")}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
