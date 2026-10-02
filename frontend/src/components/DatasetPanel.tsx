/** 数据集面板：定义 + 生成 + 预览 + 审计。 */

import { useEffect, useState } from "react";
import { api, type Dataset, type PreviewRow } from "../api";

export function DatasetPanel({
  datasets,
  onCreated,
  selected,
  onSelect,
}: {
  datasets: Dataset[];
  onCreated: (d: Dataset) => void;
  selected: Dataset | null;
  onSelect: (d: Dataset) => void;
}) {
  const [name, setName] = useState("main");
  const [perDomain, setPerDomain] = useState(200);
  const [valRatio, setValRatio] = useState(0.05);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [preview, setPreview] = useState<PreviewRow[]>([]);

  useEffect(() => {
    if (!selected) {
      setPreview([]);
      return;
    }
    api
      .preview(selected.id, 3)
      .then((r) => setPreview(r.rows))
      .catch((e) => setErr((e as Error).message));
  }, [selected]);

  async function create() {
    setBusy(true);
    setErr(null);
    try {
      const res = await api.createDataset({
        name,
        per_domain: perDomain,
        val_ratio: valRatio,
      });
      onCreated({
        id: res.id, name, path: res.path, rows: res.rows,
        per_domain: perDomain, val_ratio: valRatio,
        domains: [], content_unique_ratio: res.content_unique_ratio,
        created_at: new Date().toISOString(),
      });
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const total = perDomain * 32;

  return (
    <div className="dataset">
      <div className="card">
        <h3>生成数据集</h3>
        <div className="grid grid--3">
          <label>
            名称
            <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
          </label>
          <label>
            每领域条数
            <input className="input" type="number" min={1} max={5000}
              value={perDomain} onChange={(e) => setPerDomain(+e.target.value)} />
          </label>
          <label>
            验证集比例
            <input className="input" type="number" min={0} max={0.49} step={0.01}
              value={valRatio} onChange={(e) => setValRatio(+e.target.value)} />
          </label>
        </div>
        <p className="hint">
          32 个领域 × {perDomain} = <b>{total}</b> 条。
          生成后会自动审计；内容唯一率低于 95% 会明确报出来
          （不是字节级 100%——那是把"同一道题换个围栏风格"也算成新样本）。
        </p>
        <button className="btn btn--primary" onClick={create} disabled={busy}>
          {busy ? "生成中…" : "生成并审计"}
        </button>
        {err && <p className="error">{err}</p>}
      </div>

      <div className="card">
        <h3>已有数据集</h3>
        {datasets.length === 0 ? (
          <p className="muted">还没有数据集。</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>名称</th><th>行数</th><th>内容唯一率</th><th>路径</th><th />
              </tr>
            </thead>
            <tbody>
              {datasets.map((d) => (
                <tr key={d.id} className={selected?.id === d.id ? "row--on" : ""}>
                  <td>{d.name}</td>
                  <td>{d.rows}</td>
                  <td>
                    <span className={d.content_unique_ratio >= 0.95 ? "ok" : "warn"}>
                      {(d.content_unique_ratio * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="mono">{d.path.replace(/^.*\/(?=[^/]+$)/, "")}</td>
                  <td>
                    <button className="btn btn--sm" onClick={() => onSelect(d)}>选择</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {selected && preview.length > 0 && (
        <div className="card">
          <h3>预览 · {selected.name}</h3>
          {preview.map((r, i) => (
            <details key={i} className="sample">
              <summary>
                <span className="tag">{r.domain}</span> {r.prompt.slice(0, 90)}
                {r.prompt.length > 90 && "…"}
              </summary>
              <div className="sample__body">
                <h5>题面</h5>
                <pre>{r.prompt}</pre>
                <h5>回答</h5>
                <pre>{r.response}</pre>
              </div>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}
