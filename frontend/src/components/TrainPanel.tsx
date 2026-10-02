/** 训练面板：参数表单 + SSE 实时进度 + 路由分布。 */

import { useEffect, useMemo, useRef, useState } from "react";
import { api, type ForgeEvent, type RunRow } from "../api";

export interface TrainForm {
  dataset_path: string;
  project_name: string;
  max_steps: number;
  batch_size: number;
  grad_accum: number;
  max_length: number;
  loss_chunk_size: number;
  lr: number;
  lr_router: number;
  eval_every: number;
  eval_batches: number;
  device: string;
  text_only: boolean;
}

export const DEFAULT_FORM: TrainForm = {
  dataset_path: "data/dual_contrast_data.jsonl",
  project_name: "Heretic-Scalpel-SwarmForge",
  max_steps: 200,
  batch_size: 4,
  grad_accum: 4,
  max_length: 256,
  // vocab=262144，分块峰值 ≈ chunk * 262144 * 4 字节。
  // 2048 -> ~2.1GB；显存紧张时调小。
  loss_chunk_size: 2048,
  lr: 5e-5,
  lr_router: 1e-3,
  eval_every: 25,
  eval_batches: 8,
  device: "auto",
  text_only: true,
};

interface Props {
  architecture: Record<string, unknown>;
  form: TrainForm;
  onForm: (f: TrainForm) => void;
  onRunStarted: (id: number) => void;
  busy: boolean;
}

interface Curve {
  step: number;
  train: number;
  val: number | null;
}

export function TrainPanel({ architecture, form, onForm, onRunStarted, busy }: Props) {
  const [curve, setCurve] = useState<Curve[]>([]);
  const [runId, setRunId] = useState<number | null>(null);
  const [status, setStatus] = useState<string>("idle");
  const [log, setLog] = useState<string[]>([]);
  const [summary, setSummary] = useState<RunRow | null>(null);
  const esRef = useRef<EventSource | null>(null);

  // 组件卸载时必须关掉 EventSource，否则浏览器会一直重连
  useEffect(() => () => esRef.current?.close(), []);

  function subscribe(id: number) {
    esRef.current?.close();
    setRunId(id);
    setCurve([]);
    setLog([]);
    setSummary(null);
    setStatus("running");

    const es = api.events(id);
    esRef.current = es;

    es.onmessage = () => {}; // 具名事件走下面的 addEventListener
    const handle = (raw: MessageEvent) => {
      let ev: ForgeEvent;
      try {
        ev = JSON.parse(raw.data);
      } catch {
        return;
      }
      onEvent(ev);
    };

    const interesting = ["train_start", "step", "eval", "train_done", "state", "warn"];
    for (const name of interesting) es.addEventListener(name, handle as EventListener);
    es.addEventListener("log", (e: Event) => {
      const ev = JSON.parse((e as MessageEvent).data) as ForgeEvent;
      const line = String(ev.line ?? "").trim();
      // 进度条每秒刷几十行，只留有信息量的
      if (line && !/Loading weights|\r/.test(line)) {
        setLog((l) => [...l.slice(-300), line]);
      }
    });
    es.onerror = () => {
      // EventSource 会自己重连；只有在服务端已关闭时才置终态
      if (es.readyState === EventSource.CLOSED) setStatus("closed");
    };

    function onEvent(ev: ForgeEvent) {
      switch (ev.t) {
        case "train_start":
          setStatus("running");
          break;
        case "step": {
          const step = Number(ev.step);
          setCurve((c) => {
            const next = [...c, { step, train: Number(ev.lm_loss), val: null }];
            return next.slice(-500);
          });
          break;
        }
        case "eval": {
          const v = Number(ev.val_lm_loss);
          setCurve((c) => {
            if (!c.length) return c;
            const next = [...c];
            next[next.length - 1].val = v;
            return next;
          });
          break;
        }
        case "train_done": {
          setSummary({
            id: 0, state: "running", steps_done: Number(ev.steps),
            max_steps: Number(ev.steps), best_val_loss: null,
            last_lm_loss: Number(ev.final_lm_loss), macro_dead: Number(ev.macro_dead),
            micro_dead: Number(ev.micro_dead),
            macro_dist: (ev.macro_dist ?? {}) as Record<string, number>,
            error: null, created_at: "", finished_at: "",
          } as RunRow);
          break;
        }
        case "state":
          setStatus(String(ev.state));
          if (ev.error) setLog((l) => [...l, `错误：${ev.error}`]);
          break;
        case "warn":
          setLog((l) => [...l, `⚠ ${ev.msg}`]);
          break;
      }
    }
  }

  async function start() {
    setStatus("starting");
    try {
      const res = await api.startRun({ architecture, ...form });
      onRunStarted(res.id);
      subscribe(res.id);
    } catch (e) {
      setStatus("error");
      setLog((l) => [...l, `启动失败：${(e as Error).message}`]);
    }
  }

  async function stop() {
    if (runId == null) return;
    await api.stopRun(runId);
  }

  const best = useMemo(() => {
    const vals = curve.map((c) => c.val).filter((v): v is number => v != null);
    return vals.length ? Math.min(...vals) : null;
  }, [curve]);

  const macroDist = summary?.macro_dist ?? {};
  const macroNames = Object.keys(macroDist);

  return (
    <div className="train">
      <div className="train__form">
        <fieldset disabled={busy || status === "running"}>
          <h3>训练参数</h3>
          <div className="grid">
            <label>
              数据集路径
              <input className="input" value={form.dataset_path}
                onChange={(e) => onForm({ ...form, dataset_path: e.target.value })} />
            </label>
            <label>
              项目名
              <input className="input" value={form.project_name}
                onChange={(e) => onForm({ ...form, project_name: e.target.value })} />
            </label>
            <label>
              总步数
              <input className="input" type="number" min={1} value={form.max_steps}
                onChange={(e) => onForm({ ...form, max_steps: +e.target.value })} />
            </label>
            <label>
              批大小
              <input className="input" type="number" min={1} value={form.batch_size}
                onChange={(e) => onForm({ ...form, batch_size: +e.target.value })} />
            </label>
            <label>
              梯度累积
              <input className="input" type="number" min={1} value={form.grad_accum}
                onChange={(e) => onForm({ ...form, grad_accum: +e.target.value })} />
            </label>
            <label>
              截断长度
              <input className="input" type="number" min={32} value={form.max_length}
                onChange={(e) => onForm({ ...form, max_length: +e.target.value })} />
            </label>
            <label>
              CE 分块
              <input className="input" type="number" min={128} step={512}
                value={form.loss_chunk_size}
                onChange={(e) => onForm({ ...form, loss_chunk_size: +e.target.value })} />
              <span className="hint">
                峰值显存 ≈ {((form.loss_chunk_size * 262144 * 4) / 1024 ** 3).toFixed(2)} GB
              </span>
            </label>
            <label>
              LoRA 学习率
              <input className="input" type="number" step="0.00001" value={form.lr}
                onChange={(e) => onForm({ ...form, lr: +e.target.value })} />
            </label>
            <label>
              Router 学习率
              <input className="input" type="number" step="0.0001" value={form.lr_router}
                onChange={(e) => onForm({ ...form, lr_router: +e.target.value })} />
            </label>
            <label>
              每 N 步评测
              <input className="input" type="number" min={0} value={form.eval_every}
                onChange={(e) => onForm({ ...form, eval_every: +e.target.value })} />
            </label>
            <label>
              设备
              <select className="input" value={form.device}
                onChange={(e) => onForm({ ...form, device: e.target.value })}>
                <option value="auto">auto</option>
                <option value="cuda:0">cuda:0</option>
                <option value="cpu">cpu</option>
              </select>
            </label>
            <label className="checkbox">
              <input type="checkbox" checked={form.text_only}
                onChange={(e) => onForm({ ...form, text_only: e.target.checked })} />
              仅文本（丢弃视觉/音频塔，省约 0.9GB 显存）
            </label>
          </div>
        </fieldset>

        <div className="train__actions">
          {status !== "running" ? (
            <button className="btn btn--primary" onClick={start} disabled={busy}>
              {busy ? "提交中…" : "▶ 一键训练"}
            </button>
          ) : (
            <button className="btn btn--danger" onClick={stop}>■ 停止</button>
          )}
          <span className={`pill pill--${status}`}>{status}</span>
          {best != null && <span className="pill pill--ok">最佳 val {best.toFixed(4)}</span>}
        </div>
      </div>

      <div className="train__out">
        <LossChart curve={curve} />
        {macroNames.length > 0 && (
          <div className="dist">
            <h4>大核路由分布</h4>
            <p className="hint">
              均衡时每项应接近 {100 / macroNames.length}%。
              某项塌到 0 说明该大核已失去专属领域。
            </p>
            <div className="dist__bars">
              {macroNames.map((n) => {
                const v = macroDist[n] ?? 0;
                return (
                  <div key={n} className="dist__row">
                    <span className="dist__name">{n}</span>
                    <div className="dist__bar">
                      <div className="dist__fill" style={{ width: `${v * 100}%` }} />
                    </div>
                    <span className="dist__val">{(v * 100).toFixed(1)}%</span>
                  </div>
                );
              })}
            </div>
            <p className="hint">
              死掉的大核 {summary?.macro_dead ?? 0} 个 · 小核 {summary?.micro_dead ?? 0} 个
            </p>
          </div>
        )}
        <details className="logbox" open={status !== "succeeded"}>
          <summary>日志（{log.length} 行）</summary>
          <pre>{log.join("\n") || "（暂无）"}</pre>
        </details>
      </div>
    </div>
  );
}

/** 极简折线图：不引图表库，避免为一个 sparkline 塞进 ~100KB 依赖。 */
function LossChart({ curve }: { curve: Curve[] }) {
  if (curve.length < 2) {
    return <div className="chart chart--empty">{curve.length ? "采集中…" : "尚未开始训练"}</div>;
  }
  const W = 640;
  const H = 160;
  const vals = curve.flatMap((c) => (c.val == null ? [c.train] : [c.train, c.val]));
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const span = Math.max(1e-6, hi - lo);
  const x = (i: number) => (i / (curve.length - 1)) * W;
  const y = (v: number) => H - ((v - lo) / span) * (H - 16) - 8;

  const trainPath = curve.map((c, i) => `${i ? "L" : "M"}${x(i)},${y(c.train)}`).join(" ");
  const valPts = curve.map((c, i) => (c.val == null ? null : [i, c.val] as const)).filter(Boolean) as readonly (readonly [number, number])[];
  const valPath = valPts.map(([i, v], k) => `${k ? "L" : "M"}${x(i)},${y(v)}`).join(" ");

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img"
        aria-label="训练损失与验证损失曲线">
        <path d={trainPath} fill="none" stroke="#f59e0b" strokeWidth="1.5" opacity="0.75" />
        {valPath && <path d={valPath} fill="none" stroke="#22d3ee" strokeWidth="2" />}
      </svg>
      <div className="chart__legend">
        <span className="k k--train">训练 loss</span>
        <span className="k k--val">验证 loss</span>
        <span className="muted">范围 {lo.toFixed(3)} ~ {hi.toFixed(3)}</span>
      </div>
    </div>
  );
}
