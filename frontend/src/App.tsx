import { useEffect, useMemo, useState } from "react";
import { api, type Dataset, type DomainsPayload, type Health, type RunRow } from "./api";
import { ArchDesigner } from "./components/ArchDesigner";
import { DatasetPanel } from "./components/DatasetPanel";
import { DEFAULT_FORM, TrainPanel, type TrainForm } from "./components/TrainPanel";
import { Playground } from "./components/Playground";

type Step = "data" | "arch" | "train" | "play";

const STEPS: { id: Step; label: string }[] = [
  { id: "data", label: "① 数据集" },
  { id: "arch", label: "② 蜂群架构" },
  { id: "train", label: "③ 训练" },
  { id: "play", label: "④ 试跑" },
];

const PRESETS: { name: string; M: number; N: number; r: number; s: number }[] = [
  { name: "轻量（显存紧张）", M: 2, N: 8, r: 16, s: 8 },
  { name: "默认", M: 4, N: 32, r: 64, s: 16 },
  { name: "宽（大核多）", M: 8, N: 64, r: 64, s: 16 },
];

export default function App() {
  const [step, setStep] = useState<Step>("data");
  const [health, setHealth] = useState<Health | null>(null);
  const [domains, setDomains] = useState<DomainsPayload | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selected, setSelected] = useState<Dataset | null>(null);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [busy, setBusy] = useState(false);
  const [fatal, setFatal] = useState<string | null>(null);

  // 架构设计
  const [numMacro, setNumMacro] = useState(4);
  const [macroNames, setMacroNames] = useState<string[]>(
    ["Arts_Anchor", "Code_Math_Core", "Sci_Reason_Core", "Humanities_Core"],
  );
  const [numMicro, setNumMicro] = useState(32);
  const [macroRank, setMacroRank] = useState(64);
  const [microRank, setMicroRank] = useState(16);
  const [mapping, setMapping] = useState<Record<string, number>>({});

  const [form, setForm] = useState<TrainForm>(DEFAULT_FORM);

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setFatal((e as Error).message));
    api.domains().then((d) => {
      setDomains(d);
      setMapping(d.default_map);
    }).catch((e) => setFatal((e as Error).message));
    refreshDatasets();
    refreshRuns();
  }, []);

  function refreshDatasets() {
    api.listDatasets().then((ds) => {
      setDatasets(ds);
      setSelected((cur) => cur ?? ds[ds.length - 1] ?? null);
    }).catch(() => {});
  }
  function refreshRuns() {
    api.listRuns().then(setRuns).catch(() => {});
  }

  /** 大核数变化时重命名 + 夹紧映射，避免越界值流到后端。 */
  function setMacroCount(n: number) {
    const clamped = Math.max(1, Math.min(16, n));
    setNumMacro(clamped);
    setMacroNames((old) => {
      const next = [...old];
      while (next.length < clamped) next.push(`Macro_${next.length}`);
      return next.slice(0, clamped);
    });
    setMapping((m) => {
      const out: Record<string, number> = {};
      for (const [d, v] of Object.entries(m)) out[d] = v % clamped;
      return out;
    });
  }

  const architecture = useMemo(
    () => ({
      name: "web",
      num_macro_cores: numMacro,
      macro_names: macroNames,
      macro_rank: macroRank,
      num_micro_experts: numMicro,
      micro_rank: microRank,
      domain_to_macro: mapping,
    }),
    [numMacro, macroNames, macroRank, numMicro, microRank, mapping],
  );

  /** 训练参数量估算——改动 rank/专家数时让人立刻看到代价。 */
  const paramEstimate = useMemo(() => {
    const H = 1536;               // 底座 hidden（Gemma4-E2B）
    const M = Math.max(0, numMacro - 1);
    const perLayer =
      M * (2 * H * macroRank) + numMicro * (2 * H * microRank) +
      H * numMacro + H * numMicro;
    const total = perLayer * 35;             // 35 层
    return { total, mb: Math.round((total * 4) / 1024 ** 2) };
  }, [numMacro, numMicro, macroRank, microRank]);

  return (
    <div className="app">
      <header className="topbar">
        <h1>
          <span className="logo">⚒</span> Heretic-Scalpel Swarm Forge
        </h1>
        <div className="topbar__meta">
          {health ? (
            <>
              <span className="pill pill--ok">
                GPU 可用 {health.gpu_free_gb ?? "?"} GB
              </span>
              <span className="pill">{health.domains} 个领域</span>
              {health.active_runs.length > 0 && (
                <span className="pill pill--running">
                  {health.active_runs.length} 个训练在跑
                </span>
              )}
            </>
          ) : (
            <span className="pill">连接中…</span>
          )}
        </div>
      </header>

      {fatal && (
        <div className="banner banner--err">
          无法连接后端：{fatal}
          <br />
          请先启动 <code>PYTHONPATH=backend/src python -m uvicorn forge_web.app:app --port 8848</code>
        </div>
      )}

      <nav className="steps">
        {STEPS.map((s) => (
          <button
            key={s.id}
            className={`step${step === s.id ? " step--on" : ""}`}
            onClick={() => setStep(s.id)}
          >
            {s.label}
          </button>
        ))}
        <div className="steps__spacer" />
        <span className="muted">
          大核 {numMacro} · 小核 {numMicro} · 可训参数 ~{paramEstimate.mb} MB
        </span>
      </nav>

      <main>
        {step === "data" && (
          <DatasetPanel
            datasets={datasets}
            selected={selected}
            onSelect={setSelected}
            onCreated={(d) => {
              refreshDatasets();
              setSelected(d);
              setForm((f) => ({ ...f, dataset_path: d.path }));
            }}
          />
        )}

        {step === "arch" && (
          <div className="arch">
            <div className="card">
              <h3>规格</h3>
              <div className="presets">
                {PRESETS.map((p) => (
                  <button
                    key={p.name}
                    className={`btn btn--sm${
                      numMacro === p.M && numMicro === p.N ? " btn--primary" : ""
                    }`}
                    onClick={() => {
                      setMacroCount(p.M);
                      setNumMicro(p.N);
                      setMacroRank(p.r);
                      setMicroRank(p.s);
                    }}
                  >
                    {p.name}（{p.M}×{p.N}）
                  </button>
                ))}
              </div>

              <div className="grid grid--4">
                <label>
                  大核数 M
                  <input className="input" type="number" min={1} max={16}
                    value={numMacro}
                    onChange={(e) => setMacroCount(+e.target.value)} />
                  <span className="hint">0 号恒为只读底座</span>
                </label>
                <label>
                  小核数 N
                  <input className="input" type="number" min={1} max={512}
                    value={numMicro}
                    onChange={(e) => setNumMicro(Math.max(1, Math.min(512, +e.target.value)))} />
                </label>
                <label>
                  大核 rank
                  <input className="input" type="number" min={1} max={512}
                    value={macroRank}
                    onChange={(e) => setMacroRank(+e.target.value)} />
                </label>
                <label>
                  小核 rank
                  <input className="input" type="number" min={1} max={512}
                    value={microRank}
                    onChange={(e) => setMicroRank(+e.target.value)} />
                </label>
              </div>

              <div className="grid">
                {macroNames.map((n, i) => (
                  <label key={i}>
                    大核 #{i} 名称
                    <input className="input" value={n}
                      onChange={(e) => {
                        const next = [...macroNames];
                        next[i] = e.target.value;
                        setMacroNames(next);
                      }} />
                  </label>
                ))}
              </div>
            </div>

            <div className="card">
              <h3>领域 → 大核 分配</h3>
              {domains ? (
                <ArchDesigner
                  domains={domains.domains}
                  macroNames={macroNames}
                  mapping={mapping}
                  onChange={setMapping}
                />
              ) : (
                <p className="muted">加载领域列表…</p>
              )}
            </div>
          </div>
        )}

        {step === "train" && (
          <TrainPanel
            architecture={architecture}
            form={form}
            onForm={setForm}
            busy={busy}
            onRunStarted={() => {
              setBusy(false);
              refreshRuns();
            }}
          />
        )}

        {step === "play" && (
          <Playground busy={busy} />
        )}
      </main>

      {runs.length > 0 && step === "train" && (
        <section className="runs">
          <h3>历史训练</h3>
          <table className="table">
            <thead>
              <tr>
                <th>#</th><th>状态</th><th>步数</th>
                <th>最佳 val</th><th>死大核</th><th>死小核</th><th>时间</th>
              </tr>
            </thead>
            <tbody>
              {runs.slice(0, 10).map((r) => (
                <tr key={r.id}>
                  <td>{r.id}</td>
                  <td><span className={`pill pill--${r.state}`}>{r.state}</span></td>
                  <td>{r.steps_done}/{r.max_steps}</td>
                  <td>{r.best_val_loss?.toFixed(4) ?? "—"}</td>
                  <td>{r.macro_dead ?? "—"}</td>
                  <td>{r.micro_dead ?? "—"}</td>
                  <td className="mono">{r.created_at.slice(5, 16)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
