"""FastAPI 应用。

设计要点
--------
* **不 import torch**。本模块只 import ``forge_web.db`` / ``jobs`` /
  ``runner``；所有涉及 torch 的调用都发生在**子进程**里。
  连 ``/api/health`` 的 GPU 信息也是用 ``nvidia-smi`` 子进程读的，
  不走 ``torch.cuda``——一旦在这里 import torch，API 进程就会分走
  一份 CUDA allocator，与训练进程抢显存。
  这样 API 进程只占几百 MB，且训练 OOM 不会波及网站。
* **SSE 而不是 WebSocket**。训练进度是单向的（服务端 -> 浏览器），
  SSE 走普通 HTTP，能穿代理、自带断线重连语义，还不用额外依赖。
* **事件带 ``seq``**。断线重连时前端带 ``Last-Event-ID`` 回来，
  服务端从 backlog 里补发，避免丢事件。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import select

from . import db
from .forge_core_bridge import DOMAINS, assert_boundary
from .jobs import ArchSpec, DatasetSpec, TrainSpec, datagen_config, write_forge_config
from .runner import Supervisor
from .settings import Settings, get_settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _gpu_free_gb() -> float | None:
    """用 nvidia-smi 读显存——**故意不 import torch**。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return round(int(out.stdout.strip().splitlines()[0]) / 1024, 2)
    except Exception:                            # noqa: BLE001
        return None


def create_app(settings: Settings | None = None) -> FastAPI:
    st = settings or get_settings()
    st.project_root.mkdir(parents=True, exist_ok=True)
    (st.project_root / st.db_path).parent.mkdir(parents=True, exist_ok=True)
    db_path = st.project_root / st.db_path

    assert_boundary()

    sup = Supervisor(log_dir=st.project_root / "data" / "logs",
                     max_concurrent=st.max_concurrent_runs)

    app = FastAPI(title="Heretic-Scalpel Swarm Forge", version="0.2.0",
                  description="定义数据集与蜂群架构，一键启动训练。")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173",
                       "http://127.0.0.1:8848", "http://localhost:8848"],
        allow_methods=["*"], allow_headers=["*"])

    if not st.is_loopback:
        import logging
        logging.getLogger("forge_web").warning(
            "服务绑定在 %s（非回环）且无鉴权。本服务可读取本地文件并启动"
            "子进程，等同于远程代码执行，请放在带鉴权的反向代理之后。", st.host)

    # -- 工具 ---------------------------------------------------------
    def ssn():
        return db.session(db_path)

    def resolve(rel: str) -> Path:
        """限制在允许目录内。

        必须 **resolve 之后再比前缀**：直接对原始字符串 startswith
        会被 ``data/../../etc/passwd`` 这类相对路径绕过。
        """
        p = Path(rel)
        if not p.is_absolute():
            p = st.project_root / p
        p = p.resolve()
        for base in st.resolved_allow():
            try:
                p.relative_to(base)
                return p
            except ValueError:
                continue
        raise HTTPException(403, f"路径超出允许范围：{rel}")

    # -----------------------------------------------------------------
    # 元信息
    # -----------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "gpu_free_gb": _gpu_free_gb(),
                "active_runs": sup.active(), "domains": len(DOMAINS),
                "project_root": str(st.project_root)}

    @app.get("/api/domains")
    def domains() -> dict[str, Any]:
        from .forge_core_bridge import default_domain_to_macro
        return {"domains": list(DOMAINS), "default_map": default_domain_to_macro()}

    # -----------------------------------------------------------------
    # 数据集
    # -----------------------------------------------------------------
    @app.post("/api/datasets")
    def create_dataset(spec: DatasetSpec = Body(...)) -> dict[str, Any]:
        """同步生成 + 审计。

        6400 条约 2s，同步做完再返回，前端能立刻展示唯一率，
        比异步任务 + 轮询简单得多，也不会出现"建完看不到结果"。
        """
        from .forge_core_bridge import build_dataset, norm_content

        out = resolve(f"data/{spec.filename}")
        out.parent.mkdir(parents=True, exist_ok=True)
        report = build_dataset(datagen_config(spec, str(out)),
                               domains=spec.domains or None)

        rows = [json.loads(l) for l in
                out.read_text(encoding="utf-8").splitlines() if l.strip()]
        ratio = len({(norm_content(r["prompt"]), norm_content(r["response"]))
                     for r in rows}) / max(1, len(rows))

        with ssn() as s:
            rec = db.Dataset(name=spec.name, path=str(out), rows=len(rows),
                             val_ratio=spec.val_ratio,
                             per_domain=spec.per_domain,
                             domains=spec.domains or list(DOMAINS),
                             content_unique_ratio=round(ratio, 4))
            s.add(rec); s.commit(); s.refresh(rec)
            return {"id": rec.id, "path": str(out), "rows": rec.rows,
                    "content_unique_ratio": rec.content_unique_ratio,
                    "report": report.to_dict()
                    if hasattr(report, "to_dict") else str(report)}

    @app.get("/api/datasets")
    def list_datasets() -> list[dict[str, Any]]:
        with ssn() as s:
            return [d.model_dump(mode="json")
                    for d in s.exec(select(db.Dataset)).all()]

    @app.get("/api/datasets/{dataset_id}/preview")
    def preview(dataset_id: int, limit: int = Query(3, ge=1, le=50)):
        with ssn() as s:
            rec = s.get(db.Dataset, dataset_id)
            if rec is None:
                raise HTTPException(404, "数据集不存在")
        # forge_core 是**兄弟包**（同一个 src 下），不是 forge_web 的子模块
        from forge_core.datagen import preview as _preview
        return {"path": rec.path, "rows": _preview(rec.path, limit=limit)}

    # -----------------------------------------------------------------
    # 架构
    # -----------------------------------------------------------------
    @app.post("/api/architectures")
    def save_architecture(spec: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            a = ArchSpec(**spec)
            profile = a.to_profile()          # 校验在 to_profile 里
        except Exception as e:                # noqa: BLE001
            raise HTTPException(422, str(e)) from e
        with ssn() as s:
            rec = db.Architecture(
                name=a.name, num_macro_cores=a.num_macro_cores,
                macro_names=profile.macro_names, macro_rank=a.macro_rank,
                num_micro_experts=a.num_micro_experts,
                micro_rank=a.micro_rank,
                domain_to_macro=profile.domain_to_macro)
            s.add(rec); s.commit(); s.refresh(rec)
            return rec.model_dump(mode="json")

    @app.get("/api/architectures")
    def list_architectures() -> list[dict[str, Any]]:
        with ssn() as s:
            return [a.model_dump(mode="json")
                    for a in s.exec(select(db.Architecture)).all()]

    # -----------------------------------------------------------------
    # 训练
    # -----------------------------------------------------------------
    @app.post("/api/runs")
    def start_run(spec: TrainSpec = Body(...)) -> dict[str, Any]:
        cfg = spec.to_forge_config()          # 越界/不一致在这里就报错
        data_path = resolve(spec.dataset_path)
        if not data_path.exists():
            raise HTTPException(400, f"数据集不存在：{data_path}")

        runs_dir = st.project_root / "data" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        with ssn() as s:
            arch = db.Architecture(
                name=spec.architecture.name,
                num_macro_cores=cfg.arch.num_macro_cores,
                macro_names=cfg.arch.macro_names,
                macro_rank=cfg.arch.macro_rank,
                num_micro_experts=cfg.arch.num_micro_experts,
                micro_rank=cfg.arch.micro_rank,
                domain_to_macro=cfg.arch.domain_to_macro)
            ds = db.Dataset(name=data_path.stem, path=str(data_path))
            s.add(arch); s.add(ds); s.commit()
            s.refresh(arch); s.refresh(ds)

            rec = db.Run(architecture_id=arch.id, dataset_id=ds.id,
                         state=db.RunState.queued, max_steps=spec.max_steps,
                         config=spec.model_dump(mode="json"))
            s.add(rec); s.commit(); s.refresh(rec)
            run_id = rec.id

        cfg.data_path = str(data_path)
        cfg.checkpoint_path = str(runs_dir / f"ckpt-{run_id}.pt")
        if spec.export_dir:
            cfg.export_dir = str(resolve(spec.export_dir))
        cfg_path = write_forge_config(cfg, runs_dir / f"config-{run_id}.json")
        argv = spec.to_cli_argv(cfg_path, Path(cfg.checkpoint_path))

        try:
            handle = sup.spawn(run_id, argv, cwd=st.project_root)
        except RuntimeError as e:
            with ssn() as s:
                r = s.get(db.Run, run_id)
                r.state = db.RunState.failed; r.error = str(e)
                r.finished_at = _now(); s.add(r); s.commit()
            raise HTTPException(409, str(e)) from e

        with ssn() as s:
            r = s.get(db.Run, run_id)
            r.state = db.RunState.running; r.log_path = str(handle.log_path)
            s.add(r); s.commit()

        threading.Thread(target=_persist_events, args=(sup, db_path, run_id),
                         daemon=True).start()
        return {"id": run_id, "state": "running", "argv": argv,
                "log_path": str(handle.log_path)}

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        with ssn() as s:
            return [r.model_dump(mode="json")
                    for r in s.exec(select(db.Run)
                                    .order_by(db.Run.id.desc())).all()]

    @app.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: int) -> dict[str, Any]:
        h = sup.get(run_id)
        if h is None:
            raise HTTPException(404, "run 不存在或已结束")
        how = h.stop()
        with ssn() as s:
            r = s.get(db.Run, run_id)
            r.state = db.RunState.cancelled; r.finished_at = _now()
            s.add(r); s.commit()
        return {"id": run_id, "result": how}

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: int, request: Request,
                         after: int = Query(0, ge=0)) -> StreamingResponse:
        """SSE 事件流。``after`` / ``Last-Event-ID`` 用于断线续传。

        这里刻意用 :meth:`RunHandle.poll` 的**非阻塞**轮询而不是
        :meth:`RunHandle.stream`：后者内部会阻塞在 Condition 上，
        放进事件循环会把整个服务卡住。
        """
        h = sup.get(run_id)
        if h is None:
            raise HTTPException(404, "run 不存在或已结束")
        start = after or int(request.headers.get("last-event-id") or 0)

        async def gen() -> AsyncIterator[str]:
            yield "retry: 2000\n\n"
            cursor = start
            while True:
                if await request.is_disconnected():
                    return
                batch, closed, state, error = h.poll(cursor)
                if batch:
                    for ev in batch:
                        cursor = ev["seq"]
                        yield (f"id: {ev['seq']}\n"
                               f"event: {ev.get('t', 'log')}\n"
                               f"data: {json.dumps(ev, ensure_ascii=False)}\n\n")
                    continue
                if closed:
                    yield (f"event: state\n"
                           f"data: {json.dumps({'t': 'state', 'state': state, 'error': error, 'seq': cursor + 1}, ensure_ascii=False)}\n\n")
                    return
                await asyncio.sleep(0.2)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/api/runs/{run_id}/log")
    def run_log(run_id: int):
        h = sup.get(run_id)
        p = h.log_path if h else None
        if p is None or not Path(p).exists():
            with ssn() as s:
                r = s.get(db.Run, run_id)
                if r is None or not r.log_path or not Path(r.log_path).exists():
                    raise HTTPException(404, "日志不存在")
                p = Path(r.log_path)
        return FileResponse(str(p), media_type="text/plain")

    app.state.settings = st
    app.state.supervisor = sup
    app.state.db_path = db_path
    return app


def _persist_events(sup: Supervisor, db_path: Path, run_id: int) -> None:
    """把子进程事件回写进数据库（刷新页面后仍能看到历史）。"""
    h = sup.get(run_id)
    if h is None:
        return

    def put(**fields: Any) -> None:
        with db.session(db_path) as s:
            r = s.get(db.Run, run_id)
            if r is None:
                return
            for k, v in fields.items():
                setattr(r, k, v)
            s.add(r)
            s.commit()

    def record_val(val: float | None) -> None:
        """只保留**历史最好**的验证 loss。

        注意不能直接覆盖：loss 曲线会抖动（实测步间波动能到 2 倍），
        存最后一次等于随机取一个点，显示出来毫无意义。
        """
        if val is None:
            return
        with db.session(db_path) as s:
            r = s.get(db.Run, run_id)
            if r is None:
                return
            prev = r.best_val_loss
            r.best_val_loss = val if prev is None else min(prev, val)
            s.add(r)
            s.commit()

    for ev in h.stream(after=0):
        t = ev.get("t")
        if t == "step":
            put(steps_done=int(ev.get("step", 0)),
                last_lm_loss=ev.get("lm_loss"))
            record_val(ev.get("val_lm_loss"))
        elif t == "train_done":
            put(steps_done=int(ev.get("steps", 0)),
                last_lm_loss=ev.get("final_lm_loss"),
                macro_dead=ev.get("macro_dead"),
                micro_dead=ev.get("micro_dead"),
                macro_dist=ev.get("macro_dist", {}))
        elif t == "state":
            mapping = {"succeeded": db.RunState.succeeded,
                       "failed": db.RunState.failed,
                       "cancelled": db.RunState.cancelled}
            put(state=mapping.get(ev.get("state", "failed"),
                                  db.RunState.failed),
                error=ev.get("error"), finished_at=_now())


app = create_app()


def main() -> None:                             # pragma: no cover
    import uvicorn

    st = get_settings()
    uvicorn.run("forge_web.app:app", host=st.host, port=st.port,
                reload=False, log_level="info")


__all__ = ["create_app", "app", "main"]
