/** Playground：拿训练好的权重直接试跑，并显示路由分布。 */

import { useEffect, useRef, useState } from "react";
import { api } from "../api";

interface Msg {
  role: "user" | "assistant";
  content: string;
}

interface Checkpoint {
  path: string;
  mb: number;
}

interface RouteInfo {
  macro_dist: Record<string, number>;
  micro_dist: number[];
  macro_dead: number;
  micro_dead: number;
}

const SUGGESTIONS = [
  "用两句话解释什么是梯度下降。",
  "写一个 Python 函数，判断字符串是否为回文。",
  "为什么粗绳比细绳能承受更大的拉力？",
  "把「事必躬亲」翻译成英文并解释语气。",
];

export function Playground({ busy }: { busy: boolean }) {
  const [cks, setCks] = useState<Checkpoint[]>([]);
  const [ck, setCk] = useState("");
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<Msg[]>([]);
  const [pending, setPending] = useState(false);
  const [compare, setCompare] = useState(false);
  const [maxTok, setMaxTok] = useState(256);
  const [topK, setTopK] = useState(0);
  const [temp, setTemp] = useState(0);
  const [out, setOut] = useState<{
    text: string;
    base_text?: string;
    identical?: boolean;
    tok_per_sec: number;
    meta?: Record<string, unknown>;
  } | null>(null);
  const [route, setRoute] = useState<RouteInfo | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api
      .playground()
      .then((r) => {
        setCks(r.checkpoints);
        if (r.checkpoints.length) setCk(r.checkpoints[0].path);
      })
      .catch((e) => setErr((e as Error).message));
  }, []);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [history, pending]);

  async function send(text?: string) {
    const q = (text ?? input).trim();
    if (!q || !ck || pending) return;
    setInput("");
    setErr(null);
    const nextHist = [...history, { role: "user" as const, content: q }];
    setHistory(nextHist);
    setPending(true);
    setOut(null);
    try {
      const r = await api.chat({
        checkpoint: ck,
        prompt: q,
        history: history.slice(-8),
        max_new_tokens: maxTok,
        temperature: temp,
        compare,
        top_k: topK || undefined,
      });
      setOut(r);
      setRoute(r.route ?? null);
      setHistory([...nextHist, { role: "assistant", content: r.text }]);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setPending(false);
    }
  }

  function reset() {
    setHistory([]);
    setOut(null);
    setRoute(null);
    setErr(null);
  }

  const deadMacro = route?.macro_dead ?? 0;

  return (
    <div className="pg">
      <div className="pg__side">
        <div className="card">
          <h3>试跑设置</h3>
          <label>
            检查点
            <select className="input" value={ck} onChange={(e) => setCk(e.target.value)}>
              {cks.length === 0 && <option value="">（没有可用检查点）</option>}
              {cks.map((c) => (
                <option key={c.path} value={c.path}>
                  {c.path} · {c.mb} MB
                </option>
              ))}
            </select>
          </label>
          <label>
            最大生成长度：{maxTok}
            <input type="range" min={32} max={1024} step={32}
              value={maxTok} onChange={(e) => setMaxTok(+e.target.value)} />
          </label>
          <label>
            温度：{temp.toFixed(2)}
            <input type="range" min={0} max={1.5} step={0.05}
              value={temp} onChange={(e) => setTemp(+e.target.value)} />
            <span className="hint">0 = 贪心解码，可复现</span>
          </label>
          <label>
            小核 top-k
            <input className="input" type="number" min={0} placeholder="用训练时的值"
              value={topK || ""} onChange={(e) => setTopK(+e.target.value)} />
          </label>
          <label className="checkbox">
            <input type="checkbox" checked={compare}
              onChange={(e) => setCompare(e.target.checked)} />
            与纯底座对照
          </label>
          <div className="pg__btns">
            <button className="btn btn--sm" onClick={reset}>清空对话</button>
          </div>
        </div>

        {route && (
          <div className="card">
            <h3>路由（第一层）</h3>
            {Object.entries(route.macro_dist).map(([n, v]) => (
              <div key={n} className="dist__row">
                <span className="dist__name">{n}</span>
                <div className="dist__bar">
                  <div className="dist__fill" style={{ width: `${v * 100}%` }} />
                </div>
                <span className="dist__val">{(v * 100).toFixed(0)}%</span>
              </div>
            ))}
            <p className="hint">
              单次回答只有几十个路由决策，分布集中是正常的；
              要看整体是否坍缩请看训练结束时的 <code>macro_dead</code>。
              {deadMacro > 0 && ` 本次有 ${deadMacro} 个大核未被选中。`}
            </p>
            <p className="hint">
              活跃小核 {route.micro_dist.filter((v) => v > 0).length}/{route.micro_dist.length}
            </p>
          </div>
        )}
      </div>

      <div className="pg__main card">
        <h3>对话</h3>
        <div className="chatlog" ref={logRef}>
          {history.length === 0 && (
            <div className="chatlog__empty">
              <p>试试这些问题：</p>
              {SUGGESTIONS.map((s) => (
                <button key={s} className="btn btn--sm" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          )}
          {history.map((m, i) => (
            <div key={i} className={`msg msg--${m.role}`}>
              <span className="msg__who">{m.role === "user" ? "你" : "模型"}</span>
              <pre className="msg__body">{m.content}</pre>
            </div>
          ))}
          {pending && <div className="msg msg--assistant"><span className="msg__who">模型</span><p className="msg__body muted">加载底座并生成中…（首次约 30~60 秒）</p></div>}
        </div>

        {out?.base_text != null && (
          <details className="cmp">
            <summary>纯底座输出（对照）</summary>
            <pre>{out.base_text}</pre>
            {out.identical ? (
              <p className="error">⚠ 两者完全相同 —— 适配器几乎没起作用。检查训练步数 / lr / scale。</p>
            ) : (
              <p className="hint">✓ 输出不同，适配器确实生效。</p>
            )}
          </details>
        )}

        {err && <p className="error">{err}</p>}
        {out && (
          <p className="hint">
            {out.tok_per_sec.toFixed(1)} tok/s
            {out.meta?.step != null && ` · 权重 step ${out.meta.step}`}
            {out.meta?.loaded != null && ` · 加载 ${out.meta.loaded} 个张量`}
          </p>
        )}

        <div className="pg__input">
          <textarea
            className="input"
            rows={3}
            placeholder="输入问题，Enter 发送，Shift+Enter 换行"
            value={input}
            disabled={pending || busy || !ck}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
          />
          <button className="btn btn--primary" onClick={() => void send()}
            disabled={pending || busy || !ck || !input.trim()}>
            {pending ? "生成中…" : "发送"}
          </button>
        </div>
      </div>
    </div>
  );
}
