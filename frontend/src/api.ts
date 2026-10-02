/** 与后端的唯一通信层。
 *
 * 刻意把 fetch 集中在这里，而不是散在组件里：
 * - 错误信息在后端是 pydantic 的 `detail`（有时是数组），直接
 *   `String(err.detail)` 会得到 "[object Object]"；这里统一拍平；
 * - SSE 的重连参数（Last-Event-ID）只有这一处需要知道。
 */

const BASE = "";

export class ApiError extends Error {
  constructor(public status: number, message: string, public detail?: unknown) {
    super(message);
  }
}

function flatten(detail: unknown): string {
  if (detail == null) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d: any) => (typeof d === "string" ? d : d?.msg ?? JSON.stringify(d)))
      .join("；");
  }
  return JSON.stringify(detail);
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail: unknown;
    try {
      detail = (await res.json())?.detail;
    } catch {
      detail = await res.text().catch(() => "");
    }
    throw new ApiError(res.status, flatten(detail) || res.statusText, detail);
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

// --- 类型 ---------------------------------------------------------------
export interface Health {
  ok: boolean;
  gpu_free_gb: number | null;
  active_runs: number[];
  domains: number;
  project_root: string;
}

export interface DomainsPayload {
  domains: string[];
  default_map: Record<string, number>;
}

export interface Dataset {
  id: number;
  name: string;
  path: string;
  rows: number;
  per_domain: number;
  val_ratio: number;
  domains: string[];
  content_unique_ratio: number;
  created_at: string;
}

export interface DatasetCreated {
  id: number;
  path: string;
  rows: number;
  content_unique_ratio: number;
  report: unknown;
}

export interface PreviewRow {
  domain: string;
  id?: string;
  prompt: string;
  response: string;
}

export interface RunRow {
  id: number;
  state: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  steps_done: number;
  max_steps: number;
  best_val_loss: number | null;
  last_lm_loss: number | null;
  macro_dead: number | null;
  micro_dead: number | null;
  macro_dist: Record<string, number>;
  error: string | null;
  created_at: string;
  finished_at: string | null;
}

/** SSE 事件（后端把 CLI 的 NDJSON 原样透传）。 */
export interface ForgeEvent {
  t: string;
  seq?: number;
  [k: string]: unknown;
}

// --- 端点 ---------------------------------------------------------------
export const api = {
  health: () => req<Health>("/api/health"),
  domains: () => req<DomainsPayload>("/api/domains"),

  createDataset: (body: {
    name: string;
    per_domain: number;
    val_ratio: number;
    domains?: string[];
    out_name?: string;
  }) => req<DatasetCreated>("/api/datasets", { method: "POST", body: JSON.stringify(body) }),

  listDatasets: () => req<Dataset[]>("/api/datasets"),
  preview: (id: number, limit = 3) =>
    req<{ path: string; rows: PreviewRow[] }>(`/api/datasets/${id}/preview?limit=${limit}`),

  startRun: (body: unknown) =>
    req<{ id: number; state: string; log_path: string }>("/api/runs", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  listRuns: () => req<RunRow[]>("/api/runs"),
  stopRun: (id: number) => req<{ id: number; result: string }>(`/api/runs/${id}/stop`, { method: "POST" }),

  /**
   * 订阅训练事件。
   *
   * 用 `EventSource` 而不是手写 fetch 流：它自带断线重连，并且会自动
   * 带上 `Last-Event-ID` 请求头——后端据此从 backlog 补发，
   * 所以中途刷新页面不会丢掉已经发生的事件。
   */
  events: (runId: number) => new EventSource(`${BASE}/api/runs/${runId}/events`),

  logUrl: (runId: number) => `${BASE}/api/runs/${runId}/log`,

  playground: () =>
    req<{ artifacts: { kind: "checkpoint" | "bundle"; label: string; path: string; mb: number }[] }>("/api/playground"),
  chat: (body: {
    checkpoint: string; prompt: string; history?: { role: string; content: string }[];
    max_new_tokens: number; temperature: number; compare: boolean; top_k?: number;
  }) =>
    req<{
      text: string; base_text?: string; identical?: boolean;
      tok_per_sec: number; meta?: Record<string, unknown>;
      route?: { macro_dist: Record<string, number>; micro_dist: number[];
                macro_dead: number; micro_dead: number };
    }>("/api/chat", { method: "POST", body: JSON.stringify(body) }),
};
